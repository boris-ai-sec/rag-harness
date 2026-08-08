"""Conformance validator for synthetic-to-client V0.3 evidence reuse."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_harness.contracts import RunPackage
from rag_harness.export_identity import (
    canonical_json_bytes,
    compute_export_key,
    sha256_bytes,
    sha256_file,
)
from rag_harness.governed_validation import (
    ALLOWED_OBJECT_FIELDS,
    ARTIFACT_FIELDS,
    ExportValidationCheck,
    ExportValidationResult,
    _asserted_prohibited_claims,
    _check,
    _load_json,
    _result,
    validate_governed_export,
)
from rag_harness.v03_contracts import LAYER2_OBJECT_TYPES, FrozenV03Profile

REUSE_GOVERNED_PATHS = {
    "Artifact Manifest": Path("governed/artifact_manifest.json"),
    "Evidence Record": Path("governed/evidence_records/synthetic_reuse.json"),
    "Evidence Package": Path("governed/evidence_package.json"),
}
REUSE_SIDECAR_PATHS = {
    "adapter_mapping_manifest": Path(
        "artifacts/adapter/adapter_mapping_manifest.json"
    ),
    "export_receipt": Path("artifacts/adapter/export_receipt.json"),
    "validation_report": Path("artifacts/adapter/validation_report.json"),
}


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


def validate_synthetic_reuse_export(
    *,
    export_dir: Path,
    profile: FrozenV03Profile,
) -> ExportValidationResult:
    """Validate a new client-scoped record derived from synthetic evidence."""

    export_dir = export_dir.resolve(strict=False)
    checks: list[ExportValidationCheck] = []
    _check(
        checks,
        "export.directory",
        export_dir.is_dir() and not export_dir.is_symlink(),
        "reuse export directory exists and is not a symlink",
        "reuse export directory is missing or is a symlink",
    )
    if not export_dir.is_dir() or export_dir.is_symlink():
        return _result(export_dir, checks, None)

    documents: dict[str, dict[str, Any]] = {}
    for label, relative_path in REUSE_GOVERNED_PATHS.items():
        try:
            documents[label] = _load_json(export_dir / relative_path)
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
            _check(
                checks,
                f"governed.{label}.readable",
                False,
                "governed reuse object is readable",
                f"governed reuse object is missing or invalid: {str(error)[:500]}",
            )
        else:
            _check(
                checks,
                f"governed.{label}.readable",
                True,
                "governed reuse object is readable",
                "governed reuse object is missing or invalid",
            )

    sidecars: dict[str, dict[str, Any]] = {}
    for role, relative_path in REUSE_SIDECAR_PATHS.items():
        try:
            sidecars[role] = _load_json(export_dir / relative_path)
        except (json.JSONDecodeError, OSError, UnicodeError, ValueError) as error:
            _check(
                checks,
                f"sidecar.{role}.readable",
                False,
                "reuse sidecar is readable",
                f"reuse sidecar is missing or invalid: {str(error)[:500]}",
            )
        else:
            _check(
                checks,
                f"sidecar.{role}.readable",
                True,
                "reuse sidecar is readable",
                "reuse sidecar is missing or invalid",
            )
    if len(documents) != 3 or len(sidecars) != 3:
        return _result(export_dir, checks, None)

    mapping = sidecars["adapter_mapping_manifest"]
    export_key = mapping.get("export_key")
    components = mapping.get("export_key_components")
    _check(
        checks,
        "identity.export_key_reproducible",
        isinstance(export_key, str)
        and isinstance(components, dict)
        and compute_export_key(components) == export_key,
        "reuse export key is reproducible from canonical components",
        "reuse export key cannot be reproduced",
    )
    _check(
        checks,
        "identity.export_directory_matches",
        isinstance(export_key, str) and export_dir.name == f"export-{export_key}",
        "reuse export directory matches export key",
        "reuse export directory does not match export key",
    )
    _check(
        checks,
        "governance.reuse_kind",
        mapping.get("export_kind") == "synthetic_client_reuse"
        and mapping.get("semantic_relation")
        == "derived_from_synthetic_evidence",
        "adapter mapping explicitly identifies synthetic-to-client reuse",
        "adapter mapping does not explicitly identify synthetic-to-client reuse",
    )

    target_case_id = mapping.get("target_case_id")
    source_case_id = mapping.get("source_case_id")
    request_ref = mapping.get("evidence_request_ref")
    object_ids: list[str] = []
    for label, payload in documents.items():
        object_type = str(payload.get("object_type"))
        object_required = profile.object_required_fields.get(
            object_type, frozenset()
        )
        if object_type == "Artifact Manifest":
            object_required = frozenset({"artifacts"})
        missing = sorted(
            (profile.envelope_required_fields | object_required).difference(payload)
        )
        extras = sorted(
            set(payload).difference(ALLOWED_OBJECT_FIELDS.get(object_type, set()))
        )
        _check(
            checks,
            f"contract.{label}.required_fields",
            not missing,
            "required V0.3 fields are present",
            "missing required fields: " + ", ".join(missing),
        )
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
            "object_type is outside the governed Layer 2 set",
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
            "Harness reuse export must remain draft",
        )
        _check(
            checks,
            f"identity.{label}.case_id",
            isinstance(target_case_id, str)
            and payload.get("case_id") == target_case_id,
            "object case_id matches the client target case",
            "object case_id does not match the client target case",
        )
        if isinstance(payload.get("object_id"), str):
            object_ids.append(payload["object_id"])

    _check(
        checks,
        "contract.object_types_exact",
        sorted(payload.get("object_type") for payload in documents.values())
        == ["Artifact Manifest", "Evidence Package", "Evidence Record"],
        "reuse contains exactly Manifest, Evidence Record, and Evidence Package",
        "reuse governed object set is incomplete or unexpected",
    )
    _check(
        checks,
        "identity.object_ids_unique",
        len(object_ids) == len(set(object_ids)) == 3,
        "governed reuse object IDs are unique",
        "governed reuse object IDs are missing or duplicated",
    )
    _check(
        checks,
        "identity.case_boundary_changed",
        isinstance(source_case_id, str)
        and isinstance(target_case_id, str)
        and source_case_id != target_case_id,
        "source synthetic and target client cases are distinct",
        "synthetic-to-client reuse must cross into a distinct case",
    )

    manifest = documents["Artifact Manifest"]
    record = documents["Evidence Record"]
    package = documents["Evidence Package"]
    manifest_id = manifest.get("object_id")
    evidence_id = record.get("evidence_id")
    _check(
        checks,
        "identity.evidence_id",
        isinstance(evidence_id, str) and evidence_id == record.get("object_id"),
        "new client-scoped Evidence Record ID is valid",
        "new client-scoped Evidence Record ID is invalid",
    )
    _check(
        checks,
        "registry.evidence_method",
        record.get("evidence_method") in profile.evidence_methods,
        "evidence method is canonical",
        "evidence method is not in the V0.3 registry",
    )
    _check(
        checks,
        "registry.evidence_status",
        record.get("evidence_status") in profile.evidence_statuses,
        "evidence status is canonical",
        "evidence status is not in the V0.3 registry",
    )
    _check(
        checks,
        "references.record.manifest",
        record.get("artifact_manifest_ref") == manifest_id,
        "client Evidence Record resolves to its Artifact Manifest",
        "client Evidence Record Artifact Manifest reference is invalid",
    )
    _check(
        checks,
        "references.package",
        package.get("evidence_request_ref") == request_ref
        and package.get("artifact_manifest_refs") == [manifest_id]
        and package.get("evidence_record_refs") == [evidence_id],
        "client Evidence Package resolves to request, manifest, and record",
        "client Evidence Package references are invalid",
    )

    external_refs = mapping.get("external_provenance_refs")
    external_ref_set = (
        set(external_refs)
        if isinstance(external_refs, list)
        and all(isinstance(ref, str) for ref in external_refs)
        else set()
    )
    source_ids = mapping.get("source_governed_object_ids")
    source_demo_id = (
        source_ids.get("demonstration_record")
        if isinstance(source_ids, dict)
        else None
    )
    _check(
        checks,
        "references.package.external_demonstration",
        isinstance(source_demo_id, str)
        and source_demo_id in external_ref_set
        and package.get("demonstration_record_refs") == [source_demo_id],
        "client Evidence Package explicitly references the synthetic demonstration",
        "client Evidence Package synthetic demonstration reference is invalid",
    )
    unresolved: set[str] = set()
    observed_external: set[str] = set()
    for payload in documents.values():
        for field in ("parent_object_refs", "related_object_refs", "provenance_refs"):
            refs = payload.get(field, [])
            if not isinstance(refs, list):
                unresolved.add(f"{payload.get('object_id')}:{field}")
                continue
            for ref in refs:
                if ref in object_ids or ref == request_ref:
                    continue
                if ref in external_ref_set:
                    observed_external.add(ref)
                else:
                    unresolved.add(str(ref))
    _check(
        checks,
        "references.envelope_resolve",
        not unresolved and bool(observed_external),
        "internal refs resolve and external synthetic provenance is explicit",
        "unresolved or missing external provenance refs: "
        + ", ".join(sorted(unresolved)),
    )

    artifacts = manifest.get("artifacts")
    artifact_ids: list[str] = []
    artifact_paths: list[str] = []
    shape_ok = isinstance(artifacts, list) and bool(artifacts)
    hashes_ok = shape_ok
    if isinstance(artifacts, list):
        for index, artifact in enumerate(artifacts):
            entry_ok = isinstance(artifact, dict) and set(artifact) == ARTIFACT_FIELDS
            shape_ok &= entry_ok
            if not isinstance(artifact, dict):
                continue
            artifact_id = artifact.get("artifact_id")
            relative = artifact.get("file_path")
            if isinstance(artifact_id, str):
                artifact_ids.append(artifact_id)
            if isinstance(relative, str):
                artifact_paths.append(relative)
                path = export_dir / relative
                hashes_ok &= (
                    path.is_file()
                    and not path.is_symlink()
                    and path.resolve().is_relative_to(export_dir)
                    and artifact.get("integrity_reference")
                    == f"sha256:{sha256_file(path)}"
                )
            else:
                hashes_ok = False
            _check(
                checks,
                f"artifacts.entry_{index}.shape",
                entry_ok,
                "artifact entry follows the frozen V0.3 profile",
                "artifact entry fields are incomplete or unexpected",
            )
    actual_artifacts = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "artifacts").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _check(
        checks,
        "artifacts.manifest",
        shape_ok
        and len(artifact_ids) == len(set(artifact_ids)) == len(artifact_paths)
        and actual_artifacts == set(artifact_paths),
        "Artifact Manifest is complete and artifact IDs are unique",
        "Artifact Manifest is malformed, incomplete, or duplicated",
    )
    _check(
        checks,
        "integrity.artifact_hashes",
        hashes_ok,
        "all reuse artifact hashes match actual files",
        "one or more reuse artifact hashes are invalid",
    )

    source_relative = mapping.get("source_export_path")
    source_export = export_dir / str(source_relative)
    copied_source_validation = validate_governed_export(
        export_dir=source_export,
        profile=profile,
    )
    _check(
        checks,
        "provenance.source_export_valid",
        copied_source_validation.validation_status == "PASS",
        "copied original synthetic governed export remains valid",
        "copied original synthetic governed export is invalid",
    )
    source_artifacts = mapping.get("source_artifacts")
    mapped_source_paths: set[str] = set()
    source_mapping_ok = isinstance(source_artifacts, list) and bool(source_artifacts)
    if isinstance(source_artifacts, list):
        for item in source_artifacts:
            if not isinstance(item, dict):
                source_mapping_ok = False
                continue
            relative = item.get("export_path")
            digest = item.get("sha256")
            if not isinstance(relative, str) or not isinstance(digest, str):
                source_mapping_ok = False
                continue
            mapped_source_paths.add(relative)
            copied = export_dir / relative
            source_mapping_ok &= (
                copied.is_file()
                and not copied.is_symlink()
                and sha256_file(copied) == digest
            )
    actual_source_paths = {
        path.relative_to(export_dir).as_posix()
        for path in source_export.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    source_mapping_ok &= mapped_source_paths == actual_source_paths
    _check(
        checks,
        "provenance.source_mapping_complete",
        source_mapping_ok,
        "source governed export mapping is complete",
        "source governed export mapping is incomplete",
    )
    _check(
        checks,
        "integrity.source_export_hash",
        source_export.is_dir()
        and mapping.get("source_export_hash") == _tree_hash(source_export),
        "copied source governed export tree hash is preserved",
        "copied source governed export tree hash is missing or incorrect",
    )
    copied_mapping_path = (
        source_export / "artifacts/adapter/adapter_mapping_manifest.json"
    )
    copied_package_path = source_export / "artifacts/source_run/run_package.json"
    try:
        source_mapping = _load_json(copied_mapping_path)
        copied_package = RunPackage.model_validate_json(
            copied_package_path.read_text(encoding="utf-8")
        )
    except Exception:  # noqa: BLE001 - validator records malformed provenance
        source_mapping = {}
        copied_package = None
    _check(
        checks,
        "identity.source_identity_preserved",
        copied_package is not None
        and copied_package.run_id == mapping.get("run_id")
        and copied_package.experiment_id == mapping.get("experiment_id")
        and copied_package.case_id == source_case_id
        and source_mapping.get("export_key") == mapping.get("source_export_key")
        and source_mapping.get("source_package_hash")
        == mapping.get("source_package_hash"),
        "source run, experiment, package, case, and export identities are preserved",
        "source identity is not preserved through reuse",
    )

    report = sidecars["validation_report"]
    receipt = sidecars["export_receipt"]
    _check(
        checks,
        "lifecycle.sidecars",
        report.get("validation_status") == "PASS"
        and report.get("export_key") == export_key
        and receipt.get("export_key") == export_key
        and receipt.get("status") in {"created", "reused_existing"},
        "reuse validation report and receipt preserve lifecycle identity",
        "reuse validation report or receipt is invalid",
    )
    prohibited = set().union(
        *(
            _asserted_prohibited_claims(payload)
            for payload in [*documents.values(), *sidecars.values()]
        )
    )
    _check(
        checks,
        "governance.no_prohibited_claim_fields",
        not prohibited,
        "reuse contains no Layer 1 or production claims",
        "prohibited claims detected: " + ", ".join(sorted(prohibited)),
    )
    limitation = mapping.get("mandatory_limitation")
    serialized = json.dumps(documents, ensure_ascii=False, sort_keys=True)
    _check(
        checks,
        "governance.synthetic_limitation_visible",
        isinstance(limitation, str) and bool(limitation) and limitation in serialized,
        "mandatory synthetic-to-client limitation is visible",
        "mandatory synthetic-to-client limitation is absent",
    )
    actual_governed = {
        path.relative_to(export_dir).as_posix()
        for path in (export_dir / "governed").rglob("*")
        if path.is_file() or path.is_symlink()
    }
    _check(
        checks,
        "governance.governed_file_set_exact",
        actual_governed
        == {path.as_posix() for path in REUSE_GOVERNED_PATHS.values()},
        "reuse output contains only approved frozen V0.3 object files",
        "reuse governed output contains missing or unexpected files",
    )
    return _result(
        export_dir,
        checks,
        export_key if isinstance(export_key, str) else None,
    )
