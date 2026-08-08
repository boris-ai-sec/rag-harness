"""Bounded governed V0.3 export adapter for verified run-native evidence."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field

from rag_harness.contracts import (
    ExperimentDefinition,
    RunConfiguration,
    RunPackage,
    StrictModel,
)
from rag_harness.export_identity import (
    ADAPTER_VERSION,
    TARGET_CONTRACT_VERSION,
    canonical_json_bytes,
    compute_export_key,
    export_key_components,
    governed_object_id,
    safe_identifier_component,
    sha256_bytes,
    sha256_file,
)
from rag_harness.governed_validation import (
    ADAPTER_SIDECAR_PATHS,
    GOVERNED_PATHS,
    ExportValidationResult,
    validate_governed_export,
)
from rag_harness.v03_contracts import (
    EvidenceRequestReference,
    FrozenV03Profile,
    load_evidence_request,
)
from rag_harness.verification import verify_run_package


class GovernedExportResult(StrictModel):
    adapter_version: Literal["0.1"] = "0.1"
    export_status: Literal["created", "existing"]
    export_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_path: str
    case_id: str
    evidence_request_ref: str
    run_id: str
    experiment_id: str
    source_package_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    governed_object_ids: dict[str, str]
    validation: ExportValidationResult
    source_run_unchanged: Literal[True] = True
    governed_v0_3_export_claimed: Literal[True] = True
    client_system_verified: Literal[False] = False
    production_isolation_verified: Literal[False] = False


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _source_snapshot(run_directory: Path) -> dict[str, str]:
    return {
        path.relative_to(run_directory).as_posix(): sha256_file(path)
        for path in run_directory.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def _envelope(
    *,
    object_id: str,
    object_type: str,
    case_id: str,
    request_id: str,
    created_at: str,
    related_object_refs: list[str],
    provenance_refs: list[str],
    author_or_source: str,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "object_type": object_type,
        "object_version": "V0.1",
        "created_at": created_at,
        "modified_at": created_at,
        "author_or_source": author_or_source,
        "case_id": case_id,
        "parent_object_refs": [request_id],
        "related_object_refs": related_object_refs,
        "provenance_refs": provenance_refs,
        "approval_status": "draft",
        "change_history": [],
    }


def _object_ids(case_id: str, export_key: str) -> dict[str, str]:
    return {
        "artifact_manifest": governed_object_id(
            "AM", case_id, export_key, "MANIFEST"
        ),
        "demonstration_record": governed_object_id(
            "DR", case_id, export_key, "RUN"
        ),
        "experiment_outcome_evidence": governed_object_id(
            "NER", case_id, export_key, "OUTCOME"
        ),
        "integrity_evidence": governed_object_id(
            "NER", case_id, export_key, "INTEGRITY"
        ),
        "evidence_package": governed_object_id(
            "L2EO", case_id, export_key, "PACKAGE"
        ),
    }


def _experiment_observation(
    *,
    package: RunPackage,
    run_directory: Path,
) -> tuple[dict[str, Any], list[str]]:
    retrieval_path = run_directory / "retrieval_results.json"
    error_path = run_directory / "run_error.json"
    unresolved_gaps: list[str] = []
    if retrieval_path.is_file():
        retrieval = _read_json(retrieval_path)
        observed = {
            "run_status": package.status,
            "experiment_result": package.experiment_result,
            "scenario_results": retrieval.get("scenarios", []),
        }
        for scenario in retrieval.get("scenarios", []):
            if not isinstance(scenario, dict):
                continue
            boundary_status = scenario.get("retrieval", {}).get(
                "boundary_verification_status"
            )
            if boundary_status == "indeterminate":
                unresolved_gaps.append(
                    f"Boundary verification remained indeterminate for "
                    f"scenario {scenario.get('scenario_id', 'unknown')}."
                )
    elif error_path.is_file():
        observed = {
            "run_status": package.status,
            "experiment_result": package.experiment_result,
            "run_error": _read_json(error_path),
        }
        unresolved_gaps.append("The experiment did not complete normal retrieval execution.")
    else:
        observed = {
            "run_status": package.status,
            "experiment_result": package.experiment_result,
        }
    if package.experiment_result in {"indeterminate", "not_evaluated"}:
        unresolved_gaps.append(
            f"Experiment result remained {package.experiment_result}."
        )
    return observed, unresolved_gaps


def _outcome_evidence_status(package: RunPackage) -> str:
    if package.experiment_result == "pass":
        return "supportive"
    if package.experiment_result == "fail":
        return "contradictory"
    return "inconclusive"


def _copy_source_run(
    *,
    run_directory: Path,
    export_dir: Path,
    package: RunPackage,
) -> list[dict[str, str]]:
    role_by_path = {
        reference.relative_path: reference.artifact_role
        for reference in package.artifact_references
    }
    role_by_path["run_package.json"] = "authoritative_run_package"
    mappings: list[dict[str, str]] = []
    for source_path in sorted(path for path in run_directory.rglob("*") if path.is_file()):
        if source_path.is_symlink():
            raise ValueError("source run contains a symbolic link")
        relative = source_path.relative_to(run_directory).as_posix()
        export_relative = f"artifacts/source_run/{relative}"
        target = export_dir / export_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"source artifact copy already exists: {target}")
        shutil.copyfile(source_path, target)
        digest = sha256_file(source_path)
        if sha256_file(target) != digest:
            raise RuntimeError(f"source artifact copy hash mismatch: {relative}")
        mappings.append(
            {
                "source_relative_path": relative,
                "export_path": export_relative,
                "artifact_role": role_by_path.get(relative, "unclassified_source_artifact"),
                "sha256": digest,
            }
        )
    return mappings


def _artifact_entry(
    *,
    export_dir: Path,
    relative_path: str,
    artifact_type: str,
    related_evidence_ids: list[str],
    collected_at: str,
    environment: str,
    source: str,
    version_or_commit: str,
    export_key: str,
) -> dict[str, Any]:
    path = export_dir / relative_path
    digest = sha256_file(path)
    artifact_identity = sha256_bytes(
        canonical_json_bytes(
            {
                "export_key": export_key,
                "file_path": relative_path,
                "sha256": digest,
            }
        )
    )
    return {
        "artifact_id": f"ART-{artifact_identity[:16].upper()}",
        "artifact_type": artifact_type,
        "source": source,
        "collection_method": "bounded_governed_export",
        "file_path": relative_path,
        "format": path.suffix.removeprefix(".") or "binary",
        "description": f"Governed export artifact: {artifact_type}",
        "collected_at": collected_at,
        "environment": environment,
        "version_or_commit": version_or_commit,
        "integrity_reference": f"sha256:{digest}",
        "sensitivity": "internal",
        "retention_rule": "Retain with the governed case evidence package.",
        "related_evidence_ids": related_evidence_ids,
    }


def _build_governed_objects(
    *,
    export_dir: Path,
    package: RunPackage,
    configuration: RunConfiguration,
    definition: ExperimentDefinition,
    request: EvidenceRequestReference,
    source_mappings: list[dict[str, str]],
    verifier_payload: dict[str, Any],
    export_key: str,
    components: dict[str, str],
    source_package_hash: str,
    framework_check_output: str,
    author_or_source: str,
    export_timestamp: str,
    target_case_id: str,
) -> dict[str, str]:
    object_ids = _object_ids(target_case_id, export_key)
    observed_result, unresolved_gaps = _experiment_observation(
        package=package,
        run_directory=export_dir / "artifacts/source_run",
    )
    limitations = [definition.mandatory_limitation]
    retrieval_scenarios = observed_result.get("scenario_results", [])
    if any(
        isinstance(scenario, dict)
        and scenario.get("retrieval", {}).get("retrieval_outcome") == "empty"
        for scenario in retrieval_scenarios
    ):
        limitations.append(
            "An empty retrieval in an exclusion-boundary scenario does not establish "
            "retrieval usefulness, recall, answer quality, or availability of "
            "permitted evidence."
        )

    mapping_payload = {
        "sidecar_type": "adapter_mapping_manifest",
        "sidecar_version": "0.1",
        "adapter_version": ADAPTER_VERSION,
        "target_contract_version": TARGET_CONTRACT_VERSION,
        "export_key": export_key,
        "export_key_components": components,
        "source_package_hash": source_package_hash,
        "source_package_path": "artifacts/source_run/run_package.json",
        "target_case_id": target_case_id,
        "source_case_id": package.case_id,
        "evidence_request_ref": request.request_id,
        "evidence_request_path": str(request.path),
        "run_id": package.run_id,
        "experiment_id": package.experiment_id,
        "source_artifacts": source_mappings,
        "source_to_target_object_ids": object_ids,
        "mandatory_limitation": definition.mandatory_limitation,
        "export_timestamp": export_timestamp,
        "governed_v0_3_export_claimed": True,
        "client_system_verified": False,
        "production_isolation_verified": False,
    }
    receipt_payload = {
        "sidecar_type": "export_receipt",
        "sidecar_version": "0.1",
        "status": "created",
        "export_key": export_key,
        "created_at": export_timestamp,
        "governed_object_ids": object_ids,
        "source_package_hash": source_package_hash,
        "source_run_unchanged": True,
    }
    validation_report_payload = {
        "sidecar_type": "validation_report",
        "sidecar_version": "0.1",
        "validation_status": "PASS",
        "export_key": export_key,
        "validated_at": export_timestamp,
        "local_validator": "rag_harness.governed_validation:0.1",
        "v0_3_check_case": "PASS",
        "v0_3_check_case_output": framework_check_output,
        "validated_boundaries": [
            "frozen_v0_3_fields",
            "referential_integrity",
            "artifact_integrity",
            "run_identity",
            "export_key_reproducibility",
            "synthetic_limitation",
            "no_layer_1_or_production_claims",
        ],
    }
    _write_json(export_dir / ADAPTER_SIDECAR_PATHS["verifier_result"], verifier_payload)
    _write_json(
        export_dir / ADAPTER_SIDECAR_PATHS["adapter_mapping_manifest"],
        mapping_payload,
    )
    _write_json(export_dir / ADAPTER_SIDECAR_PATHS["export_receipt"], receipt_payload)
    _write_json(
        export_dir / ADAPTER_SIDECAR_PATHS["validation_report"],
        validation_report_payload,
    )

    manifest_id = object_ids["artifact_manifest"]
    demonstration_id = object_ids["demonstration_record"]
    outcome_evidence_id = object_ids["experiment_outcome_evidence"]
    integrity_evidence_id = object_ids["integrity_evidence"]
    package_id = object_ids["evidence_package"]

    demonstration = {
        **_envelope(
            object_id=demonstration_id,
            object_type="Demonstration Record",
            case_id=target_case_id,
            request_id=request.request_id,
            created_at=export_timestamp,
            related_object_refs=[outcome_evidence_id, integrity_evidence_id, package_id],
            provenance_refs=[manifest_id],
            author_or_source=author_or_source,
        ),
        "demonstration_id": demonstration_id,
        "evidence_request_ref": request.request_id,
        "scenario": {
            "name": definition.title,
            "experiment_reference": package.experiment_id,
            "verification_scope": definition.verification_scope,
        },
        "preconditions": [
            "The run-native package reached a terminal status.",
            "The read-only integrity verifier returned verified.",
            "The target Evidence Request existed before export.",
        ],
        "procedure": [
            "Load the registered controlled experiment definition and configuration.",
            "Execute the configured ingestion and retrieval scenarios.",
            "Finalize the run-native package and artifact hashes.",
            "Verify package integrity before governed export.",
        ],
        "observed_result": observed_result,
        "expected_result": [
            {
                "scenario_id": scenario.scenario_id,
                "expected": scenario.expected.model_dump(mode="json"),
            }
            for scenario in configuration.scenarios
        ],
        "environment": definition.verification_scope,
        "run_references": [
            package.run_id,
            f"run_package_sha256:{source_package_hash}",
            f"adapter_mapping_manifest:{ADAPTER_SIDECAR_PATHS['adapter_mapping_manifest'].as_posix()}",
        ],
        "limitations": limitations,
        "reproducibility_notes": [
            (
                "The export preserves the immutable run package, effective "
                "configuration, experiment definition, artifact hashes, and adapter "
                "version."
            ),
            "The controlled result does not establish client production behavior.",
        ],
    }
    outcome_record = {
        **_envelope(
            object_id=outcome_evidence_id,
            object_type="Evidence Record",
            case_id=target_case_id,
            request_id=request.request_id,
            created_at=export_timestamp,
            related_object_refs=[demonstration_id, package_id],
            provenance_refs=[manifest_id],
            author_or_source=author_or_source,
        ),
        "evidence_id": outcome_evidence_id,
        "source_surface_module": "06_RAG",
        "artifact_manifest_ref": manifest_id,
        "evidence_method": "synthetic_test",
        "observed_fact": [
            (
                f"EXP-RAG-001 run {package.run_id} reached terminal status "
                f"{package.status} with experiment result {package.experiment_result} "
                "under the recorded controlled Harness conditions."
            )
        ],
        "scope": definition.applicability,
        "environment": definition.verification_scope,
        "limitations": limitations,
        "evidence_status": _outcome_evidence_status(package),
        "supports_or_contradicts_refs": [request.request_id],
        "traceability": {
            "artifact_manifest": manifest_id,
            "demonstration_record": demonstration_id,
            "evidence_request": request.request_id,
        },
        "report_mapping": {
            "usable_in_sections": ["Evidence Basis", "Limitations"],
            "not_usable_for": [
                "Finding confirmation",
                "Final readiness conclusion",
                "Client recommendation",
            ],
        },
        "notes": "Normalized experiment outcome; not a finding or readiness judgment.",
    }
    integrity_record = {
        **_envelope(
            object_id=integrity_evidence_id,
            object_type="Evidence Record",
            case_id=target_case_id,
            request_id=request.request_id,
            created_at=export_timestamp,
            related_object_refs=[demonstration_id, package_id],
            provenance_refs=[manifest_id],
            author_or_source=author_or_source,
        ),
        "evidence_id": integrity_evidence_id,
        "source_surface_module": "06_RAG",
        "artifact_manifest_ref": manifest_id,
        "evidence_method": "configuration_review",
        "observed_fact": [
            (
                "The read-only run package verifier reported verified with "
                f"{verifier_payload.get('failure_count', 0)} failed checks before "
                "export."
            )
        ],
        "scope": "run_native_package_integrity",
        "environment": definition.verification_scope,
        "limitations": [
            (
                "Integrity verification establishes technical consistency and "
                "immutability; it does not establish the truth or sufficiency of the "
                "underlying evidence."
            )
        ],
        "evidence_status": "normalized",
        "supports_or_contradicts_refs": [],
        "traceability": {
            "artifact_manifest": manifest_id,
            "evidence_request": request.request_id,
            "verifier_result": ADAPTER_SIDECAR_PATHS["verifier_result"].as_posix(),
        },
        "report_mapping": {
            "usable_in_sections": ["Evidence Basis", "Limitations"],
            "not_usable_for": ["Finding confirmation", "Readiness conclusion"],
        },
        "notes": "Normalized integrity outcome for the exported run-native package.",
    }
    evidence_package = {
        **_envelope(
            object_id=package_id,
            object_type="Evidence Package",
            case_id=target_case_id,
            request_id=request.request_id,
            created_at=export_timestamp,
            related_object_refs=[],
            provenance_refs=[
                manifest_id,
                demonstration_id,
                outcome_evidence_id,
                integrity_evidence_id,
            ],
            author_or_source=author_or_source,
        ),
        "evidence_request_ref": request.request_id,
        "artifact_manifest_refs": [manifest_id],
        "evidence_record_refs": [outcome_evidence_id, integrity_evidence_id],
        "demonstration_record_refs": [demonstration_id],
        "coverage_summary": (
            "The package covers the registered EXP-RAG-001 experiment outcome and "
            "the technical integrity of its run-native evidence under controlled "
            "local synthetic conditions."
        ),
        "unresolved_gaps": unresolved_gaps,
        "package_limitations": limitations,
        "completion_status": (
            "complete_with_unresolved_gaps"
            if unresolved_gaps
            else "complete_with_limitations"
        ),
    }

    governed_documents = {
        GOVERNED_PATHS["Demonstration Record"]: demonstration,
        GOVERNED_PATHS["Experiment Outcome Evidence Record"]: outcome_record,
        GOVERNED_PATHS["Integrity Evidence Record"]: integrity_record,
        GOVERNED_PATHS["Evidence Package"]: evidence_package,
    }
    for relative_path, payload in governed_documents.items():
        _write_json(export_dir / relative_path, payload)

    role_by_export_path = {
        mapping["export_path"]: mapping["artifact_role"]
        for mapping in source_mappings
    }
    role_by_export_path.update(
        {
            relative_path.as_posix(): role
            for role, relative_path in ADAPTER_SIDECAR_PATHS.items()
        }
    )
    artifacts = [
        _artifact_entry(
            export_dir=export_dir,
            relative_path=path.relative_to(export_dir).as_posix(),
            artifact_type=role_by_export_path[path.relative_to(export_dir).as_posix()],
            related_evidence_ids=[outcome_evidence_id, integrity_evidence_id],
            collected_at=export_timestamp,
            environment=definition.verification_scope,
            source=(
                f"RAG Evidence Harness run {package.run_id}"
                if path.is_relative_to(export_dir / "artifacts/source_run")
                else "RAG Evidence Harness governed V0.3 export adapter"
            ),
            version_or_commit=(
                package.run_id
                if path.is_relative_to(export_dir / "artifacts/source_run")
                else ADAPTER_VERSION
            ),
            export_key=export_key,
        )
        for path in sorted(
            file
            for file in (export_dir / "artifacts").rglob("*")
            if file.is_file()
        )
    ]
    manifest = {
        **_envelope(
            object_id=manifest_id,
            object_type="Artifact Manifest",
            case_id=target_case_id,
            request_id=request.request_id,
            created_at=export_timestamp,
            related_object_refs=[
                demonstration_id,
                outcome_evidence_id,
                integrity_evidence_id,
                package_id,
            ],
            provenance_refs=[],
            author_or_source=author_or_source,
        ),
        "source_surface_module": "06_RAG",
        "tool_or_review_route": "RAG Evidence Harness governed V0.3 export adapter",
        "artifacts": artifacts,
        "notes": (
            "Artifact Manifest registers byte-preserving source-run copies and "
            "adapter provenance sidecars; raw evidence is not promoted to judgment."
        ),
    }
    _write_json(export_dir / GOVERNED_PATHS["Artifact Manifest"], manifest)
    return object_ids


def export_verified_run_v03(
    *,
    package_path: Path,
    framework_root: Path,
    framework_case_name: str,
    evidence_request_path: Path,
    output_root: Path,
    author_or_source: str = "RAG Evidence Harness",
    target_case_id: str | None = None,
) -> GovernedExportResult:
    """Create or reuse one deterministic governed V0.3 export set."""

    profile = FrozenV03Profile.load(framework_root)
    request = load_evidence_request(
        profile=profile,
        case_name=framework_case_name,
        path=evidence_request_path,
    )
    framework_check_output = profile.check_case(framework_case_name)
    verification = verify_run_package(package_path)
    if verification.verification_status != "verified":
        raise ValueError("governed export requires a verified run-native package")

    package_path = package_path.resolve()
    run_directory = package_path.parent
    package = RunPackage.model_validate_json(package_path.read_text(encoding="utf-8"))
    if package.status == "initialized":
        raise ValueError("governed export requires a terminal run package")
    governed_case_id = target_case_id or package.case_id
    if governed_case_id is None:
        raise ValueError("case_id is required at the governed evidence boundary")
    if package.case_id is not None and target_case_id not in (None, package.case_id):
        raise ValueError("an existing run package case_id cannot be reassigned")
    if governed_case_id != request.case_id:
        raise ValueError("run package case_id does not match the Evidence Request")

    configuration = RunConfiguration.model_validate_json(
        (run_directory / package.configuration_ref).read_text(encoding="utf-8")
    )
    definition = ExperimentDefinition.model_validate_json(
        (run_directory / "experiment_definition.json").read_text(encoding="utf-8")
    )
    if configuration.case_id != package.case_id:
        raise ValueError("configuration case_id does not match the run package")
    if configuration.experiment_id != package.experiment_id:
        raise ValueError("configuration experiment_id does not match the run package")
    if definition.experiment_id != package.experiment_id:
        raise ValueError("experiment definition does not match the run package")

    before_snapshot = _source_snapshot(run_directory)
    source_package_hash = sha256_file(package_path)
    components = export_key_components(
        source_package_hash=source_package_hash,
        target_case_id=governed_case_id,
        evidence_request_ref=request.request_id,
    )
    export_key = compute_export_key(components)
    final_parent = (
        output_root.resolve()
        / safe_identifier_component(governed_case_id)
        / safe_identifier_component(request.request_id)
    )
    final_dir = final_parent / f"export-{export_key}"
    if final_dir.exists():
        validation = validate_governed_export(export_dir=final_dir, profile=profile)
        if validation.validation_status != "PASS":
            raise FileExistsError(
                "existing export has the same identity but failed validation"
            )
        mapping = _read_json(
            final_dir / ADAPTER_SIDECAR_PATHS["adapter_mapping_manifest"]
        )
        return GovernedExportResult(
            export_status="existing",
            export_key=export_key,
            export_path=str(final_dir),
            case_id=governed_case_id,
            evidence_request_ref=request.request_id,
            run_id=package.run_id,
            experiment_id=package.experiment_id,
            source_package_hash=source_package_hash,
            governed_object_ids=mapping["source_to_target_object_ids"],
            validation=validation,
        )

    final_parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = final_parent / f".tmp-{uuid4()}"
    temporary_dir = temporary_parent / f"export-{export_key}"
    temporary_dir.mkdir(parents=True)
    try:
        source_mappings = _copy_source_run(
            run_directory=run_directory,
            export_dir=temporary_dir,
            package=package,
        )
        export_timestamp = datetime.now(UTC).isoformat()
        object_ids = _build_governed_objects(
            export_dir=temporary_dir,
            package=package,
            configuration=configuration,
            definition=definition,
            request=request,
            source_mappings=source_mappings,
            verifier_payload=verification.model_dump(mode="json"),
            export_key=export_key,
            components=components,
            source_package_hash=source_package_hash,
            framework_check_output=framework_check_output,
            author_or_source=author_or_source,
            export_timestamp=export_timestamp,
            target_case_id=governed_case_id,
        )
        validation = validate_governed_export(
            export_dir=temporary_dir,
            profile=profile,
        )
        if validation.validation_status != "PASS":
            failures = [
                check.check_id
                for check in validation.checks
                if check.status == "fail"
            ]
            raise ValueError(
                "local governed export validation failed: " + ", ".join(failures)
            )
        after_snapshot = _source_snapshot(run_directory)
        if before_snapshot != after_snapshot:
            raise RuntimeError("source run changed during governed export")
        temporary_dir.rename(final_dir)
        temporary_parent.rmdir()
    except Exception:
        if temporary_parent.exists():
            shutil.rmtree(temporary_parent)
        raise

    final_validation = validate_governed_export(export_dir=final_dir, profile=profile)
    if final_validation.validation_status != "PASS":
        raise RuntimeError("published export failed final validation")
    return GovernedExportResult(
        export_status="created",
        export_key=export_key,
        export_path=str(final_dir),
        case_id=governed_case_id,
        evidence_request_ref=request.request_id,
        run_id=package.run_id,
        experiment_id=package.experiment_id,
        source_package_hash=source_package_hash,
        governed_object_ids=object_ids,
        validation=final_validation,
    )
