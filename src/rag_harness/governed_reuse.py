"""Synthetic-to-client reuse without relabeling original governed evidence."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from rag_harness.contracts import RunPackage
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
from rag_harness.governed_export import (
    GovernedExportResult,
    _artifact_entry,
    _envelope,
    _read_json,
    _source_snapshot,
    _write_json,
)
from rag_harness.governed_reuse_validation import (
    REUSE_GOVERNED_PATHS,
    REUSE_SIDECAR_PATHS,
    validate_synthetic_reuse_export,
)
from rag_harness.governed_validation import validate_governed_export
from rag_harness.v03_contracts import FrozenV03Profile, load_evidence_request

SYNTHETIC_CLIENT_LIMITATION = (
    "Source evidence was produced in a synthetic Harness environment. It "
    "demonstrates only the tested mechanism under recorded conditions and does "
    "not establish implementation, active configuration, runtime enforcement, "
    "tenant isolation, production behavior, production readiness, or autonomy "
    "readiness of the client system."
)


def _tree_hash(directory: Path) -> str:
    entries = [
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and not path.is_symlink()
    ]
    return sha256_bytes(canonical_json_bytes(entries))


def _copy_source_export(
    *, source_export: Path, target_export: Path
) -> tuple[str, list[dict[str, str]]]:
    relative_root = f"artifacts/source_export/{source_export.name}"
    mappings: list[dict[str, str]] = []
    for source in sorted(path for path in source_export.rglob("*") if path.is_file()):
        if source.is_symlink():
            raise ValueError("source governed export contains a symbolic link")
        relative = source.relative_to(source_export).as_posix()
        export_relative = f"{relative_root}/{relative}"
        target = target_export / export_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = sha256_file(source)
        if sha256_file(target) != digest:
            raise RuntimeError(f"source governed export copy mismatch: {relative}")
        mappings.append(
            {
                "source_relative_path": relative,
                "export_path": export_relative,
                "artifact_role": "synthetic_governed_provenance",
                "sha256": digest,
            }
        )
    return relative_root, mappings


def _reuse_object_ids(case_id: str, export_key: str) -> dict[str, str]:
    return {
        "artifact_manifest": governed_object_id(
            "AM", case_id, export_key, "SYNTHETIC-REUSE"
        ),
        "synthetic_reuse_evidence": governed_object_id(
            "NER", case_id, export_key, "SYNTHETIC-REUSE"
        ),
        "evidence_package": governed_object_id(
            "L2EO", case_id, export_key, "SYNTHETIC-REUSE"
        ),
    }


def reuse_synthetic_export_v03(
    *,
    source_export_dir: Path,
    framework_root: Path,
    framework_case_name: str,
    evidence_request_path: Path,
    output_root: Path,
    author_or_source: str = "RAG Evidence Harness",
) -> GovernedExportResult:
    """Create a new client record while preserving synthetic source identity."""

    profile = FrozenV03Profile.load(framework_root)
    request = load_evidence_request(
        profile=profile,
        case_name=framework_case_name,
        path=evidence_request_path,
    )
    framework_check_output = profile.check_case(framework_case_name)
    source_export_dir = source_export_dir.resolve()
    source_validation = validate_governed_export(
        export_dir=source_export_dir,
        profile=profile,
    )
    if source_validation.validation_status != "PASS":
        raise ValueError("synthetic reuse requires a valid governed V0.3 source export")

    source_mapping_path = (
        source_export_dir / "artifacts/adapter/adapter_mapping_manifest.json"
    )
    source_mapping = _read_json(source_mapping_path)
    source_case_id = source_mapping.get("target_case_id")
    source_package_hash = source_mapping.get("source_package_hash")
    source_export_key = source_mapping.get("export_key")
    source_object_ids = source_mapping.get("source_to_target_object_ids")
    run_id = source_mapping.get("run_id")
    experiment_id = source_mapping.get("experiment_id")
    if not all(
        isinstance(value, str) and value
        for value in (
            source_case_id,
            source_package_hash,
            source_export_key,
            run_id,
            experiment_id,
        )
    ) or not isinstance(source_object_ids, dict):
        raise ValueError("synthetic source export mapping is incomplete")
    if source_case_id == request.case_id:
        raise ValueError("synthetic-to-client reuse requires a distinct target case")

    source_package_path = (
        source_export_dir / "artifacts/source_run/run_package.json"
    )
    package = RunPackage.model_validate_json(
        source_package_path.read_text(encoding="utf-8")
    )
    if package.case_id != source_mapping.get("source_case_id"):
        raise ValueError("synthetic source package identity is inconsistent")

    components = export_key_components(
        source_package_hash=source_package_hash,
        target_case_id=request.case_id,
        evidence_request_ref=request.request_id,
    )
    export_key = compute_export_key(components)
    final_parent = (
        output_root.resolve()
        / safe_identifier_component(request.case_id)
        / safe_identifier_component(request.request_id)
    )
    final_dir = final_parent / f"export-{export_key}"
    if final_dir.exists():
        validation = validate_synthetic_reuse_export(
            export_dir=final_dir,
            profile=profile,
        )
        if validation.validation_status != "PASS":
            raise FileExistsError(
                "existing synthetic reuse export has the same identity but failed "
                "validation"
            )
        mapping = _read_json(
            final_dir / REUSE_SIDECAR_PATHS["adapter_mapping_manifest"]
        )
        return GovernedExportResult(
            export_status="existing",
            export_key=export_key,
            export_path=str(final_dir),
            case_id=request.case_id,
            evidence_request_ref=request.request_id,
            run_id=run_id,
            experiment_id=experiment_id,
            source_package_hash=source_package_hash,
            governed_object_ids=mapping["source_to_target_object_ids"],
            validation=validation,
        )

    source_before = _source_snapshot(source_export_dir)
    source_tree_hash = _tree_hash(source_export_dir)
    final_parent.mkdir(parents=True, exist_ok=True)
    temporary_parent = final_parent / f".tmp-{uuid4()}"
    temporary_dir = temporary_parent / f"export-{export_key}"
    temporary_dir.mkdir(parents=True)
    try:
        source_relative, source_artifacts = _copy_source_export(
            source_export=source_export_dir,
            target_export=temporary_dir,
        )
        timestamp = datetime.now(UTC).isoformat()
        object_ids = _reuse_object_ids(request.case_id, export_key)
        manifest_id = object_ids["artifact_manifest"]
        evidence_id = object_ids["synthetic_reuse_evidence"]
        package_id = object_ids["evidence_package"]
        source_manifest_id = source_object_ids.get("artifact_manifest")
        source_demo_id = source_object_ids.get("demonstration_record")
        source_evidence_id = source_object_ids.get("experiment_outcome_evidence")
        source_package_id = source_object_ids.get("evidence_package")
        external_refs = [
            ref
            for ref in (
                source_manifest_id,
                source_demo_id,
                source_evidence_id,
                source_package_id,
            )
            if isinstance(ref, str)
        ]
        if len(external_refs) != 4:
            raise ValueError("synthetic source governed object mapping is incomplete")

        source_record = _read_json(
            source_export_dir
            / "governed/evidence_records/experiment_outcome.json"
        )
        source_status = source_record.get("evidence_status")
        if source_status not in profile.evidence_statuses:
            raise ValueError("synthetic source evidence status is not canonical")

        mapping = {
            "sidecar_type": "adapter_mapping_manifest",
            "sidecar_version": "0.1",
            "export_kind": "synthetic_client_reuse",
            "semantic_relation": "derived_from_synthetic_evidence",
            "adapter_version": ADAPTER_VERSION,
            "target_contract_version": TARGET_CONTRACT_VERSION,
            "export_key": export_key,
            "export_key_components": components,
            "source_package_hash": source_package_hash,
            "source_export_hash": source_tree_hash,
            "source_export_key": source_export_key,
            "source_export_path": source_relative,
            "source_case_id": source_case_id,
            "target_case_id": request.case_id,
            "evidence_request_ref": request.request_id,
            "evidence_request_path": str(request.path),
            "run_id": run_id,
            "experiment_id": experiment_id,
            "source_artifacts": source_artifacts,
            "source_governed_object_ids": source_object_ids,
            "external_provenance_refs": external_refs,
            "source_to_target_object_ids": object_ids,
            "mandatory_limitation": SYNTHETIC_CLIENT_LIMITATION,
            "export_timestamp": timestamp,
            "governed_v0_3_export_claimed": True,
            "client_system_verified": False,
            "production_isolation_verified": False,
        }
        receipt = {
            "sidecar_type": "export_receipt",
            "sidecar_version": "0.1",
            "status": "created",
            "export_key": export_key,
            "created_at": timestamp,
            "source_export_key": source_export_key,
            "governed_object_ids": object_ids,
            "source_evidence_unchanged": True,
        }
        validation_report = {
            "sidecar_type": "validation_report",
            "sidecar_version": "0.1",
            "validation_status": "PASS",
            "export_key": export_key,
            "validated_at": timestamp,
            "local_validator": "rag_harness.governed_reuse_validation:0.1",
            "v0_3_check_case": "PASS",
            "v0_3_check_case_output": framework_check_output,
            "validated_boundaries": [
                "synthetic_client_case_separation",
                "external_provenance_refs",
                "source_export_integrity",
                "mandatory_synthetic_limitation",
                "no_client_or_production_claims",
            ],
        }
        _write_json(
            temporary_dir / REUSE_SIDECAR_PATHS["adapter_mapping_manifest"],
            mapping,
        )
        _write_json(
            temporary_dir / REUSE_SIDECAR_PATHS["export_receipt"], receipt
        )
        _write_json(
            temporary_dir / REUSE_SIDECAR_PATHS["validation_report"],
            validation_report,
        )

        record = {
            **_envelope(
                object_id=evidence_id,
                object_type="Evidence Record",
                case_id=request.case_id,
                request_id=request.request_id,
                created_at=timestamp,
                related_object_refs=[source_evidence_id, source_demo_id, package_id],
                provenance_refs=[manifest_id, source_manifest_id, source_evidence_id],
                author_or_source=author_or_source,
            ),
            "evidence_id": evidence_id,
            "source_surface_module": "06_RAG",
            "artifact_manifest_ref": manifest_id,
            "evidence_method": "synthetic_test",
            "observed_fact": [
                (
                    f"Synthetic governed evidence from Harness run {run_id} was "
                    "reused as bounded mechanism evidence for this client case; it "
                    "does not represent client-runtime observation."
                )
            ],
            "scope": source_record.get("scope"),
            "environment": "synthetic Harness source; client case reuse only",
            "limitations": [SYNTHETIC_CLIENT_LIMITATION],
            "evidence_status": source_status,
            "supports_or_contradicts_refs": [request.request_id],
            "traceability": {
                "artifact_manifest": manifest_id,
                "evidence_request": request.request_id,
                "source_evidence_record": source_evidence_id,
            },
            "report_mapping": {
                "usable_in_sections": ["Evidence Basis", "Limitations"],
                "not_usable_for": [
                    "Client implementation verification",
                    "Production readiness conclusion",
                    "Autonomy readiness conclusion",
                ],
            },
            "notes": "New case-scoped record derived from unchanged synthetic evidence.",
        }
        evidence_package = {
            **_envelope(
                object_id=package_id,
                object_type="Evidence Package",
                case_id=request.case_id,
                request_id=request.request_id,
                created_at=timestamp,
                related_object_refs=[source_package_id],
                provenance_refs=[manifest_id, evidence_id, source_package_id],
                author_or_source=author_or_source,
            ),
            "evidence_request_ref": request.request_id,
            "artifact_manifest_refs": [manifest_id],
            "evidence_record_refs": [evidence_id],
            "demonstration_record_refs": [source_demo_id],
            "coverage_summary": (
                "Bounded synthetic mechanism evidence is available with explicit "
                "provenance to the original governed Harness export."
            ),
            "unresolved_gaps": [
                (
                    "Client implementation, configuration, runtime enforcement, and "
                    "production behavior remain unverified."
                )
            ],
            "package_limitations": [SYNTHETIC_CLIENT_LIMITATION],
            "completion_status": "complete_with_unresolved_gaps",
        }
        _write_json(temporary_dir / REUSE_GOVERNED_PATHS["Evidence Record"], record)
        _write_json(
            temporary_dir / REUSE_GOVERNED_PATHS["Evidence Package"],
            evidence_package,
        )

        artifact_roles = {
            item["export_path"]: item["artifact_role"] for item in source_artifacts
        }
        artifact_roles.update(
            {
                path.as_posix(): role
                for role, path in REUSE_SIDECAR_PATHS.items()
            }
        )
        artifacts = [
            _artifact_entry(
                export_dir=temporary_dir,
                relative_path=path.relative_to(temporary_dir).as_posix(),
                artifact_type=artifact_roles[
                    path.relative_to(temporary_dir).as_posix()
                ],
                related_evidence_ids=[evidence_id],
                collected_at=timestamp,
                environment="synthetic Harness source; client case reuse only",
                source=(
                    "Original synthetic governed V0.3 export"
                    if path.is_relative_to(temporary_dir / "artifacts/source_export")
                    else "RAG Evidence Harness governed V0.3 export adapter"
                ),
                version_or_commit=(
                    source_export_key
                    if path.is_relative_to(temporary_dir / "artifacts/source_export")
                    else ADAPTER_VERSION
                ),
                export_key=export_key,
            )
            for path in sorted(
                file
                for file in (temporary_dir / "artifacts").rglob("*")
                if file.is_file()
            )
        ]
        manifest = {
            **_envelope(
                object_id=manifest_id,
                object_type="Artifact Manifest",
                case_id=request.case_id,
                request_id=request.request_id,
                created_at=timestamp,
                related_object_refs=[evidence_id, package_id, source_manifest_id],
                provenance_refs=[source_manifest_id],
                author_or_source=author_or_source,
            ),
            "source_surface_module": "06_RAG",
            "tool_or_review_route": (
                "RAG Evidence Harness synthetic-to-client V0.3 reuse adapter"
            ),
            "artifacts": artifacts,
            "notes": (
                "Registers a byte-preserving copy of the original synthetic "
                "governed export and reuse provenance sidecars."
            ),
        }
        _write_json(
            temporary_dir / REUSE_GOVERNED_PATHS["Artifact Manifest"], manifest
        )

        validation = validate_synthetic_reuse_export(
            export_dir=temporary_dir,
            profile=profile,
        )
        if validation.validation_status != "PASS":
            failures = [
                check.check_id for check in validation.checks if check.status == "fail"
            ]
            raise ValueError(
                "local synthetic reuse validation failed: " + ", ".join(failures)
            )
        if source_before != _source_snapshot(source_export_dir):
            raise RuntimeError("source synthetic governed evidence changed during reuse")
        temporary_dir.rename(final_dir)
        temporary_parent.rmdir()
    except Exception:
        if temporary_parent.exists():
            shutil.rmtree(temporary_parent)
        raise

    final_validation = validate_synthetic_reuse_export(
        export_dir=final_dir,
        profile=profile,
    )
    if final_validation.validation_status != "PASS":
        raise RuntimeError("published synthetic reuse export failed final validation")
    return GovernedExportResult(
        export_status="created",
        export_key=export_key,
        export_path=str(final_dir),
        case_id=request.case_id,
        evidence_request_ref=request.request_id,
        run_id=run_id,
        experiment_id=experiment_id,
        source_package_hash=source_package_hash,
        governed_object_ids=object_ids,
        validation=final_validation,
    )
