"""Local conformance validator for governed V0.3 Harness export sets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from rag_harness.contracts import RunPackage, StrictModel
from rag_harness.export_identity import compute_export_key, sha256_file
from rag_harness.v03_contracts import (
    LAYER2_OBJECT_TYPES,
    PROHIBITED_AUTOMATIC_FIELDS,
    FrozenV03Profile,
)

GOVERNED_PATHS = {
    "Artifact Manifest": Path("governed/artifact_manifest.json"),
    "Demonstration Record": Path("governed/demonstration_record.json"),
    "Evidence Package": Path("governed/evidence_package.json"),
    "Experiment Outcome Evidence Record": Path(
        "governed/evidence_records/experiment_outcome.json"
    ),
    "Integrity Evidence Record": Path(
        "governed/evidence_records/integrity_verification.json"
    ),
}
ADAPTER_SIDECAR_PATHS = {
    "verifier_result": Path("artifacts/adapter/verifier_result.json"),
    "adapter_mapping_manifest": Path(
        "artifacts/adapter/adapter_mapping_manifest.json"
    ),
    "export_receipt": Path("artifacts/adapter/export_receipt.json"),
    "validation_report": Path("artifacts/adapter/validation_report.json"),
}

ENVELOPE_FIELDS = {
    "object_id",
    "object_type",
    "object_version",
    "created_at",
    "modified_at",
    "author_or_source",
    "case_id",
    "parent_object_refs",
    "related_object_refs",
    "provenance_refs",
    "approval_status",
    "change_history",
}
ALLOWED_OBJECT_FIELDS = {
    "Artifact Manifest": ENVELOPE_FIELDS
    | {
        "source_surface_module",
        "tool_or_review_route",
        "artifacts",
        "notes",
    },
    "Demonstration Record": ENVELOPE_FIELDS
    | {
        "demonstration_id",
        "evidence_request_ref",
        "scenario",
        "preconditions",
        "procedure",
        "observed_result",
        "expected_result",
        "environment",
        "run_references",
        "limitations",
        "reproducibility_notes",
    },
    "Evidence Record": ENVELOPE_FIELDS
    | {
        "evidence_id",
        "source_surface_module",
        "artifact_manifest_ref",
        "evidence_method",
        "observed_fact",
        "scope",
        "environment",
        "limitations",
        "evidence_status",
        "supports_or_contradicts_refs",
        "traceability",
        "report_mapping",
        "notes",
    },
    "Evidence Package": ENVELOPE_FIELDS
    | {
        "evidence_request_ref",
        "artifact_manifest_refs",
        "evidence_record_refs",
        "demonstration_record_refs",
        "coverage_summary",
        "unresolved_gaps",
        "package_limitations",
        "completion_status",
    },
}
ARTIFACT_FIELDS = {
    "artifact_id",
    "artifact_type",
    "source",
    "collection_method",
    "file_path",
    "format",
    "description",
    "collected_at",
    "environment",
    "version_or_commit",
    "integrity_reference",
    "sensitivity",
    "retention_rule",
    "related_evidence_ids",
}


class ExportValidationCheck(StrictModel):
    check_id: str = Field(min_length=1)
    status: Literal["pass", "fail"]
    detail: str = Field(min_length=1)


class ExportValidationResult(StrictModel):
    validation_version: Literal["0.1"] = "0.1"
    validation_status: Literal["PASS", "FAIL"]
    export_path: str
    export_key: str | None = None
    failure_count: int = Field(ge=0)
    checks: list[ExportValidationCheck] = Field(min_length=1)


def _check(
    checks: list[ExportValidationCheck],
    check_id: str,
    condition: bool,
    pass_detail: str,
    fail_detail: str,
) -> None:
    checks.append(
        ExportValidationCheck(
            check_id=check_id,
            status="pass" if condition else "fail",
            detail=pass_detail if condition else fail_detail,
        )
    )


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON object required: {path}")
    return payload


def _asserted_prohibited_claims(payload: object) -> set[str]:
    """Return prohibited fields only when they assert a positive claim."""

    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in PROHIBITED_AUTOMATIC_FIELDS and value not in (
                False,
                None,
                "",
                [],
                {},
            ):
                keys.add(str(key))
            keys.update(_asserted_prohibited_claims(value))
    elif isinstance(payload, list):
        for value in payload:
            keys.update(_asserted_prohibited_claims(value))
    return keys


def _result(
    export_dir: Path,
    checks: list[ExportValidationCheck],
    export_key: str | None,
) -> ExportValidationResult:
    failures = sum(check.status == "fail" for check in checks)
    return ExportValidationResult(
        validation_status="FAIL" if failures else "PASS",
        export_path=str(export_dir.resolve(strict=False)),
        export_key=export_key,
        failure_count=failures,
        checks=checks,
    )


def validate_governed_export(
    *,
    export_dir: Path,
    profile: FrozenV03Profile,
) -> ExportValidationResult:
    """Validate object contracts, references, identity, and artifact integrity."""

    export_dir = export_dir.resolve(strict=False)
    checks: list[ExportValidationCheck] = []
    _check(
        checks,
        "export.directory",
        export_dir.is_dir() and not export_dir.is_symlink(),
        "export directory exists and is not a symlink",
        "export directory is missing or is a symlink",
    )
    if not export_dir.is_dir() or export_dir.is_symlink():
        return _result(export_dir, checks, None)

    documents: dict[str, dict[str, Any]] = {}
    for label, relative_path in GOVERNED_PATHS.items():
        path = export_dir / relative_path
        try:
            documents[label] = _load_json(path)
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
            _check(
                checks,
                f"governed.{label}.readable",
                False,
                "governed object is readable",
                f"governed object is missing or invalid: {str(error)[:500]}",
            )
        else:
            _check(
                checks,
                f"governed.{label}.readable",
                True,
                "governed object is readable",
                "governed object is missing or invalid",
            )

    sidecars: dict[str, dict[str, Any]] = {}
    for role, relative_path in ADAPTER_SIDECAR_PATHS.items():
        try:
            sidecars[role] = _load_json(export_dir / relative_path)
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
            _check(
                checks,
                f"sidecar.{role}.readable",
                False,
                "adapter sidecar is readable",
                f"adapter sidecar is missing or invalid: {str(error)[:500]}",
            )
        else:
            _check(
                checks,
                f"sidecar.{role}.readable",
                True,
                "adapter sidecar is readable",
                "adapter sidecar is missing or invalid",
            )
    if len(documents) != len(GOVERNED_PATHS) or len(sidecars) != len(
        ADAPTER_SIDECAR_PATHS
    ):
        return _result(export_dir, checks, None)

    mapping = sidecars["adapter_mapping_manifest"]
    export_key = mapping.get("export_key")
    components = mapping.get("export_key_components")
    key_reproducible = (
        isinstance(export_key, str)
        and isinstance(components, dict)
        and compute_export_key(components) == export_key
    )
    _check(
        checks,
        "identity.export_key_reproducible",
        key_reproducible,
        "export key is reproducible from canonical components",
        "export key cannot be reproduced",
    )
    _check(
        checks,
        "identity.export_directory_matches",
        isinstance(export_key, str) and export_dir.name == f"export-{export_key}",
        "export directory matches export key",
        "export directory does not match export key",
    )

    target_case_id = mapping.get("target_case_id")
    request_ref = mapping.get("evidence_request_ref")
    object_ids: list[str] = []
    governed_object_types: list[str] = []
    for label, payload in documents.items():
        object_type = payload.get("object_type")
        governed_object_types.append(str(object_type))
        object_required = profile.object_required_fields.get(
            str(object_type), frozenset()
        )
        if object_type == "Artifact Manifest":
            object_required = frozenset({"artifacts"})
        required = profile.envelope_required_fields | object_required
        missing = sorted(required.difference(payload))
        _check(
            checks,
            f"contract.{label}.required_fields",
            not missing,
            "required V0.3 fields are present",
            "missing required fields: " + ", ".join(missing),
        )
        allowed_fields = ALLOWED_OBJECT_FIELDS.get(str(object_type), set())
        extras = sorted(set(payload).difference(allowed_fields))
        _check(
            checks,
            f"contract.{label}.no_non_v03_fields",
            not extras,
            "object contains only frozen V0.3 fields",
            "non-V0.3 fields detected: " + ", ".join(extras),
        )
        _check(
            checks,
            f"contract.{label}.object_type",
            object_type in LAYER2_OBJECT_TYPES,
            "object_type is a governed Layer 2 type",
            "object_type is not allowed for Harness export",
        )
        _check(
            checks,
            f"contract.{label}.object_version",
            payload.get("object_version") == "V0.1",
            "object version matches the frozen V0.3 contract pack",
            "object version does not match the frozen V0.3 contract pack",
        )
        _check(
            checks,
            f"contract.{label}.approval_status",
            payload.get("approval_status") == "draft"
            and payload.get("approval_status") in profile.allowed_approval_statuses,
            "approval status remains draft",
            "Harness export must remain draft",
        )
        _check(
            checks,
            f"identity.{label}.case_id",
            isinstance(target_case_id, str)
            and payload.get("case_id") == target_case_id,
            "object case_id matches target case",
            "object case_id does not match target case",
        )
        if isinstance(payload.get("object_id"), str):
            object_ids.append(payload["object_id"])

    _check(
        checks,
        "contract.object_types_exact",
        sorted(governed_object_types)
        == sorted(
            [
                "Artifact Manifest",
                "Demonstration Record",
                "Evidence Package",
                "Evidence Record",
                "Evidence Record",
            ]
        ),
        "export contains the required governed object set",
        "export governed object set is incomplete or unexpected",
    )
    _check(
        checks,
        "identity.object_ids_unique",
        len(object_ids) == len(set(object_ids)) == len(documents),
        "governed object IDs are unique",
        "governed object IDs are missing or duplicated",
    )

    manifest = documents["Artifact Manifest"]
    demonstration = documents["Demonstration Record"]
    outcome_record = documents["Experiment Outcome Evidence Record"]
    integrity_record = documents["Integrity Evidence Record"]
    evidence_package = documents["Evidence Package"]
    manifest_id = manifest.get("object_id")
    demonstration_id = demonstration.get("demonstration_id")
    evidence_ids = [outcome_record.get("evidence_id"), integrity_record.get("evidence_id")]
    _check(
        checks,
        "identity.evidence_ids_unique",
        all(isinstance(value, str) for value in evidence_ids)
        and len(evidence_ids) == len(set(evidence_ids)),
        "Evidence Record IDs are unique",
        "Evidence Record IDs are missing or duplicated",
    )
    _check(
        checks,
        "identity.demonstration_id_unique",
        isinstance(demonstration_id, str)
        and demonstration_id == demonstration.get("object_id"),
        "Demonstration Record ID is present and unique",
        "Demonstration Record identity is invalid",
    )

    for label, record in (
        ("experiment_outcome", outcome_record),
        ("integrity", integrity_record),
    ):
        _check(
            checks,
            f"registry.{label}.evidence_method",
            record.get("evidence_method") in profile.evidence_methods,
            "evidence method is canonical",
            "evidence method is not in the V0.3 registry",
        )
        _check(
            checks,
            f"registry.{label}.evidence_status",
            record.get("evidence_status") in profile.evidence_statuses,
            "evidence status is canonical",
            "evidence status is not in the V0.3 registry",
        )
        _check(
            checks,
            f"references.{label}.manifest",
            record.get("artifact_manifest_ref") == manifest_id,
            "Evidence Record resolves to Artifact Manifest",
            "Evidence Record Artifact Manifest reference is invalid",
        )

    _check(
        checks,
        "references.demonstration.request",
        demonstration.get("evidence_request_ref") == request_ref,
        "Demonstration Record resolves to Evidence Request",
        "Demonstration Record Evidence Request reference is invalid",
    )
    _check(
        checks,
        "references.package.request",
        evidence_package.get("evidence_request_ref") == request_ref,
        "Evidence Package resolves to Evidence Request",
        "Evidence Package Evidence Request reference is invalid",
    )
    _check(
        checks,
        "references.package.manifest",
        evidence_package.get("artifact_manifest_refs") == [manifest_id],
        "Evidence Package resolves to Artifact Manifest",
        "Evidence Package Artifact Manifest references are invalid",
    )
    _check(
        checks,
        "references.package.evidence_records",
        set(evidence_package.get("evidence_record_refs", [])) == set(evidence_ids),
        "Evidence Package resolves to Evidence Records",
        "Evidence Package Evidence Record references are invalid",
    )
    _check(
        checks,
        "references.package.demonstration",
        evidence_package.get("demonstration_record_refs") == [demonstration_id],
        "Evidence Package resolves to Demonstration Record",
        "Evidence Package Demonstration Record references are invalid",
    )
    _check(
        checks,
        "references.records.request",
        request_ref in outcome_record.get("supports_or_contradicts_refs", [])
        and request_ref in outcome_record.get("parent_object_refs", [])
        and request_ref in integrity_record.get("parent_object_refs", []),
        "Evidence Records remain scoped to the existing Evidence Request",
        "one or more Evidence Records are not scoped to the Evidence Request",
    )

    internal_refs = set(object_ids)
    allowed_external_refs = {request_ref}
    unresolved_refs: set[str] = set()
    for payload in documents.values():
        for field in ("parent_object_refs", "related_object_refs", "provenance_refs"):
            refs = payload.get(field, [])
            if isinstance(refs, list):
                unresolved_refs.update(
                    str(ref)
                    for ref in refs
                    if ref not in internal_refs and ref not in allowed_external_refs
                )
            else:
                unresolved_refs.add(f"{payload.get('object_id')}:{field}")
    _check(
        checks,
        "references.envelope_resolve",
        not unresolved_refs,
        "all envelope references resolve internally or to the Evidence Request",
        "unresolved envelope references: " + ", ".join(sorted(unresolved_refs)),
    )

    artifacts = manifest.get("artifacts")
    artifact_ids: list[str] = []
    artifact_paths: list[str] = []
    artifact_integrity_ok = True
    artifact_shape_ok = isinstance(artifacts, list) and bool(artifacts)
    if isinstance(artifacts, list):
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                artifact_shape_ok = False
                continue
            artifact_shape_ok &= set(artifact) == ARTIFACT_FIELDS
            artifact_id = artifact.get("artifact_id")
            relative_path = artifact.get("file_path")
            if isinstance(artifact_id, str):
                artifact_ids.append(artifact_id)
            if isinstance(relative_path, str):
                artifact_paths.append(relative_path)
                path = export_dir / relative_path
                resolved = path.resolve(strict=False)
                contained = resolved.is_relative_to(export_dir)
                expected_integrity = artifact.get("integrity_reference")
                actual_integrity = (
                    f"sha256:{sha256_file(path)}"
                    if contained and path.is_file() and not path.is_symlink()
                    else None
                )
                artifact_integrity_ok &= (
                    actual_integrity is not None
                    and actual_integrity == expected_integrity
                )
            else:
                artifact_integrity_ok = False
            _check(
                checks,
                f"artifacts.entry_{index}.shape",
                isinstance(artifact, dict) and set(artifact) == ARTIFACT_FIELDS,
                "artifact entry follows the frozen V0.3 profile",
                "artifact entry fields are incomplete or unexpected",
            )
    _check(
        checks,
        "artifacts.manifest_nonempty",
        artifact_shape_ok,
        "Artifact Manifest contains governed artifact entries",
        "Artifact Manifest is empty or malformed",
    )
    _check(
        checks,
        "identity.artifact_ids_unique",
        len(artifact_ids) == len(set(artifact_ids)) == len(artifact_paths),
        "artifact IDs are unique",
        "artifact IDs are missing or duplicated",
    )
    _check(
        checks,
        "integrity.artifact_hashes",
        artifact_integrity_ok,
        "all recorded artifact hashes match actual files",
        "one or more artifact hashes or paths are invalid",
    )

    actual_artifact_files = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "artifacts").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _check(
        checks,
        "artifacts.all_files_registered",
        actual_artifact_files == set(artifact_paths),
        "all files under artifacts are registered exactly once",
        "artifact files and manifest registrations differ",
    )

    source_mapping = mapping.get("source_artifacts")
    mapping_complete = isinstance(source_mapping, list) and bool(source_mapping)
    if isinstance(source_mapping, list):
        for item in source_mapping:
            if not isinstance(item, dict):
                mapping_complete = False
                continue
            copied_path = item.get("export_path")
            expected_hash = item.get("sha256")
            if not isinstance(copied_path, str) or not isinstance(expected_hash, str):
                mapping_complete = False
                continue
            path = export_dir / copied_path
            mapping_complete &= (
                path.is_file()
                and not path.is_symlink()
                and sha256_file(path) == expected_hash
            )
        mapped_paths = {
            item.get("export_path")
            for item in source_mapping
            if isinstance(item, dict)
            and isinstance(item.get("export_path"), str)
        }
        actual_source_paths = {
            path.relative_to(export_dir).as_posix()
            for path in (export_dir / "artifacts/source_run").rglob("*")
            if path.is_file() or path.is_symlink()
        }
        mapping_complete &= mapped_paths == actual_source_paths
    _check(
        checks,
        "provenance.source_mapping_complete",
        mapping_complete,
        "source-to-export artifact mapping is complete",
        "source-to-export artifact mapping is incomplete",
    )

    copied_package_path = export_dir / "artifacts/source_run/run_package.json"
    try:
        copied_package = RunPackage.model_validate_json(
            copied_package_path.read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001 - validator records invalid input
        copied_package = None
    run_id = mapping.get("run_id")
    experiment_id = mapping.get("experiment_id")
    _check(
        checks,
        "identity.run_id_preserved",
        copied_package is not None and copied_package.run_id == run_id,
        "run_id is preserved in adapter provenance",
        "run_id is not preserved",
    )
    _check(
        checks,
        "identity.experiment_id_preserved",
        copied_package is not None
        and copied_package.experiment_id == experiment_id
        and isinstance(demonstration.get("scenario"), dict)
        and demonstration["scenario"].get("experiment_reference") == experiment_id,
        "experiment_id is preserved in adapter provenance",
        "experiment_id is not preserved",
    )
    run_references = demonstration.get("run_references", [])
    _check(
        checks,
        "identity.run_reference_preserved",
        isinstance(run_references, list) and run_id in run_references,
        "run_id is preserved in Demonstration Record run_references",
        "Demonstration Record does not preserve run_id",
    )
    source_case_id = mapping.get("source_case_id")
    _check(
        checks,
        "identity.case_boundary",
        copied_package is not None
        and copied_package.case_id == source_case_id
        and (
            source_case_id is None
            or source_case_id == target_case_id
        ),
        "source case identity is preserved and any synthetic assignment occurs only "
        "for a standalone package",
        "source and governed case identities violate the boundary rule",
    )
    source_package_hash = mapping.get("source_package_hash")
    _check(
        checks,
        "integrity.source_package_hash",
        copied_package_path.is_file()
        and isinstance(source_package_hash, str)
        and sha256_file(copied_package_path) == source_package_hash,
        "run_package SHA-256 is preserved",
        "run_package SHA-256 is missing or incorrect",
    )
    _check(
        checks,
        "provenance.object_mapping_complete",
        mapping.get("source_to_target_object_ids")
        == {
            "artifact_manifest": manifest_id,
            "demonstration_record": demonstration_id,
            "experiment_outcome_evidence": evidence_ids[0],
            "integrity_evidence": evidence_ids[1],
            "evidence_package": evidence_package.get("object_id"),
        },
        "source-to-governed object mapping is complete",
        "source-to-governed object mapping is incomplete or inconsistent",
    )

    verifier = sidecars["verifier_result"]
    _check(
        checks,
        "eligibility.integrity_verified",
        verifier.get("verification_status") == "verified"
        and verifier.get("failure_count") == 0,
        "source run integrity verification passed",
        "source run integrity verification did not pass",
    )
    receipt = sidecars["export_receipt"]
    report = sidecars["validation_report"]
    _check(
        checks,
        "lifecycle.receipt_identity",
        receipt.get("export_key") == export_key
        and receipt.get("status") in {"created", "reused_existing"},
        "export receipt preserves lifecycle identity",
        "export receipt identity or status is invalid",
    )
    _check(
        checks,
        "validation.report_declares_pass",
        report.get("validation_status") == "PASS"
        and report.get("export_key") == export_key,
        "local validation report declares PASS for this export key",
        "local validation report is not PASS for this export key",
    )

    all_payloads: list[object] = [*documents.values(), *sidecars.values()]
    prohibited = set().union(
        *(_asserted_prohibited_claims(payload) for payload in all_payloads)
    )
    _check(
        checks,
        "governance.no_prohibited_claim_fields",
        not prohibited,
        "export contains no Layer 1 or production claim fields",
        "prohibited automatic claim fields detected: " + ", ".join(sorted(prohibited)),
    )
    mandatory_limitation = mapping.get("mandatory_limitation")
    serialized_governed = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    _check(
        checks,
        "governance.synthetic_limitation_visible",
        isinstance(mandatory_limitation, str)
        and mandatory_limitation
        and mandatory_limitation in serialized_governed,
        "mandatory synthetic Harness limitation is visible",
        "mandatory synthetic Harness limitation is absent",
    )

    known_governed_files = {
        relative_path.as_posix() for relative_path in GOVERNED_PATHS.values()
    }
    actual_governed_files = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "governed").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _check(
        checks,
        "governance.governed_file_set_exact",
        actual_governed_files == known_governed_files,
        "governed output contains only the approved V0.3 object files",
        "governed output contains missing or unexpected files",
    )
    return _result(export_dir, checks, export_key if isinstance(export_key, str) else None)
