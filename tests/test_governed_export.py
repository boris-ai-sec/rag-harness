import hashlib
import json
from pathlib import Path

import pytest

from rag_harness.contracts import RunConfiguration
from rag_harness.export_identity import compute_export_key, export_key_components
from rag_harness.governed_export import export_verified_run_v03
from rag_harness.governed_reuse import (
    SYNTHETIC_CLIENT_LIMITATION,
    reuse_synthetic_export_v03,
)
from rag_harness.governed_validation import validate_governed_export
from rag_harness.registry import EXP_RAG_001
from rag_harness.run_storage import (
    finalize_run_package,
    initialize_run_package,
    write_json_artifact,
)
from rag_harness.v03_contracts import FrozenV03Profile

CONFIG_PATH = Path("configs/exp_rag_001.json")
FIXED_RUN_ID = "run-2742eadb-2d12-4d6a-bbe7-9026485c20f1"
SECOND_RUN_ID = "run-3fa85f64-5717-4562-b3fc-2c963f66afa6"


def test_export_key_uses_stable_canonical_serialization() -> None:
    components = export_key_components(
        source_package_hash="a" * 64,
        target_case_id="case_alpha",
        evidence_request_ref="ER-001",
    )

    assert compute_export_key(components) == (
        "b19264579be353f3d253f9b7bdb88807228ee1f9f8838b8b95b9a0bfaca90366"
    )
    assert compute_export_key(dict(reversed(list(components.items())))) == (
        compute_export_key(components)
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _framework_root(
    tmp_path: Path,
    *,
    case_id: str,
    case_name: str = "CASE-EXP-RAG-001",
    request_id: str = "ER-EXP-RAG-001",
) -> tuple[Path, Path]:
    root = tmp_path / "framework-v03"
    _write_json(
        root / "05_Contracts/common_object_envelope.yaml",
        {
            "contract_id": "common_object_envelope",
            "required_fields": [
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
            ],
            "field_rules": {
                "approval_status": {
                    "allowed_values": [
                        "draft",
                        "proposed",
                        "reviewed",
                        "approved",
                        "canonical",
                        "superseded",
                        "rejected",
                    ]
                }
            },
        },
    )
    _write_json(
        root / "05_Contracts/object_contracts.yaml",
        {
            "contract_pack_id": "technical_prototype_object_contracts",
            "contract_pack_version": "V0.1",
            "objects": {
                "Evidence Request": {
                    "required_fields": [
                        "request_id",
                        "request_version",
                        "assessment_surface",
                        "evidence_need",
                        "target_component",
                        "preferred_evidence_type",
                        "available_artifacts_and_tooling",
                        "scope_boundaries",
                        "expected_output",
                        "constraints",
                        "status",
                    ]
                },
                "Artifact Manifest": {
                    "required_fields": [
                        "artifact_id",
                        "artifact_type",
                        "source",
                        "collection_method",
                        "collected_at",
                        "environment",
                        "version_or_commit",
                        "integrity_reference",
                        "sensitivity",
                        "retention_rule",
                    ]
                },
                "Demonstration Record": {
                    "required_fields": [
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
                    ]
                },
                "Evidence Record": {
                    "required_fields": [
                        "evidence_id",
                        "artifact_manifest_ref",
                        "evidence_method",
                        "observed_fact",
                        "scope",
                        "environment",
                        "limitations",
                        "evidence_status",
                        "supports_or_contradicts_refs",
                    ]
                },
                "Evidence Package": {
                    "required_fields": [
                        "evidence_request_ref",
                        "artifact_manifest_refs",
                        "evidence_record_refs",
                        "demonstration_record_refs",
                        "coverage_summary",
                        "unresolved_gaps",
                        "package_limitations",
                        "completion_status",
                    ]
                },
            },
        },
    )
    _write_json(
        root / "06_Registries/canonical_registries.yaml",
        {
            "registry_pack_version": "V0.1",
            "registries": {
                "evidence_method": {
                    "values": [
                        "document_review",
                        "interview",
                        "configuration_review",
                        "code_review",
                        "log_or_trace_review",
                        "bounded_demonstration",
                        "synthetic_test",
                        "runtime_observation",
                    ]
                },
                "evidence_status": {
                    "values": [
                        "requested",
                        "unavailable",
                        "received",
                        "normalized",
                        "reviewed",
                        "partially_supportive",
                        "supportive",
                        "contradictory",
                        "inconclusive",
                        "superseded",
                    ]
                },
            },
        },
    )
    assess_script = root / "04_Scripts/assess.py"
    assess_script.parent.mkdir(parents=True, exist_ok=True)
    assess_script.write_text(
        "import sys\nprint('Case structure check: OK')\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    case_root = root / "03_Test_Cases" / case_name
    case_root.mkdir(parents=True, exist_ok=True)
    request_path = case_root / f"layer2/evidence_requests/{request_id}.md"
    request_path.parent.mkdir(parents=True)
    request_path.write_text(
        "# Layer 2 Evidence Request\n\n"
        f"## Request ID\n\n{request_id}\n\n"
        "## Request Version\n\nV0.1\n\n"
        f"## Case ID\n\n{case_id}\n\n"
        "## Assessment Surface\n\nRAG retrieval boundary\n\n"
        "## Evidence Need\n\nControlled mechanism evidence\n\n"
        "## Target System / Component\n\nRAG retrieval component\n\n"
        "## Requested Evidence Type\n\n- evaluation\n\n"
        "## Available Artifacts and Tooling\n\nRAG Evidence Harness\n\n"
        "## Scope Boundaries\n\nSynthetic controlled environment only.\n\n"
        "## Expected Output\n\nNormalized Evidence Record.\n\n"
        "## Constraints\n\nNo production claim.\n\n"
        "## Request Status\n\ndraft\n",
        encoding="utf-8",
    )
    return root, request_path


def _run_package(
    tmp_path: Path,
    *,
    status: str = "completed",
    result: str = "pass",
    case_id: str | None = "case_exp_rag_001_synthetic",
    run_id: str = FIXED_RUN_ID,
) -> Path:
    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    ).model_copy(update={"case_id": case_id})
    workspace, package = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
        run_id=run_id,
    )
    if status == "initialized":
        return workspace.package_path
    if status == "blocked":
        artifact_path = write_json_artifact(
            workspace.directory / "run_error.json",
            {
                "stage": "service_preflight",
                "error_type": "RuntimeError",
                "message": "controlled block",
            },
        )
        artifacts = {"run_error": artifact_path}
    else:
        artifact_path = write_json_artifact(
            workspace.directory / "retrieval_results.json",
            {
                "run_id": run_id,
                "experiment_id": "EXP-RAG-001",
                "case_id": case_id,
                "experiment_result": result,
                "scenarios": [],
            },
        )
        artifacts = {"retrieval_results": artifact_path}
    finalize_run_package(
        workspace=workspace,
        package=package,
        status=status,
        experiment_result=result,
        artifact_paths=artifacts,
    )
    return workspace.package_path


def _export(
    *,
    package_path: Path,
    framework_root: Path,
    request_path: Path,
    output_root: Path,
    target_case_id: str | None = None,
):
    return export_verified_run_v03(
        package_path=package_path,
        framework_root=framework_root,
        framework_case_name="CASE-EXP-RAG-001",
        evidence_request_path=request_path,
        output_root=output_root,
        target_case_id=target_case_id,
    )


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def test_verified_pass_creates_valid_v03_export_without_mutating_source(
    tmp_path: Path,
) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    before = _snapshot(package_path.parent)

    result = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=tmp_path / "exports",
    )

    assert result.export_status == "created"
    assert result.validation.validation_status == "PASS"
    assert result.source_run_unchanged is True
    assert result.governed_v0_3_export_claimed is True
    assert result.client_system_verified is False
    assert result.production_isolation_verified is False
    assert before == _snapshot(package_path.parent)
    export_path = Path(result.export_path)
    assert (export_path / "governed/artifact_manifest.json").is_file()
    assert (export_path / "governed/demonstration_record.json").is_file()
    assert (export_path / "governed/evidence_package.json").is_file()


@pytest.mark.parametrize(
    ("status", "experiment_result", "evidence_status"),
    [
        ("completed", "fail", "contradictory"),
        ("failed", "indeterminate", "inconclusive"),
        ("blocked", "not_evaluated", "inconclusive"),
    ],
)
def test_verified_terminal_nonpass_outcomes_are_exported_without_promotion(
    tmp_path: Path,
    status: str,
    experiment_result: str,
    evidence_status: str,
) -> None:
    package_path = _run_package(
        tmp_path,
        status=status,
        result=experiment_result,
    )
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )

    result = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=tmp_path / "exports",
    )

    evidence = json.loads(
        (
            Path(result.export_path)
            / "governed/evidence_records/experiment_outcome.json"
        ).read_text()
    )
    assert result.validation.validation_status == "PASS"
    assert evidence["evidence_status"] == evidence_status
    assert "final_finding" not in evidence
    assert "readiness_conclusion" not in evidence


def test_same_export_key_is_idempotent_and_does_not_create_duplicate(
    tmp_path: Path,
) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    output_root = tmp_path / "exports"
    created = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=output_root,
    )
    before = _snapshot(Path(created.export_path))

    existing = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=output_root,
    )

    assert existing.export_status == "existing"
    assert existing.export_key == created.export_key
    assert before == _snapshot(Path(existing.export_path))
    assert len(list(output_root.rglob("export-*"))) == 1


def test_changed_valid_source_creates_new_revision_without_overwrite(
    tmp_path: Path,
) -> None:
    first_package = _run_package(tmp_path / "first", run_id=FIXED_RUN_ID)
    second_package = _run_package(tmp_path / "second", run_id=SECOND_RUN_ID)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    output_root = tmp_path / "exports"

    first = _export(
        package_path=first_package,
        framework_root=framework_root,
        request_path=request_path,
        output_root=output_root,
    )
    second = _export(
        package_path=second_package,
        framework_root=framework_root,
        request_path=request_path,
        output_root=output_root,
    )

    assert first.export_key != second.export_key
    assert Path(first.export_path).is_dir()
    assert Path(second.export_path).is_dir()
    assert len(list(output_root.rglob("export-*"))) == 2


def test_invalid_existing_export_is_not_silently_overwritten(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    output_root = tmp_path / "exports"
    created = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=output_root,
    )
    manifest_path = Path(created.export_path) / "governed/artifact_manifest.json"
    manifest_path.write_text('{"tampered": true}\n', encoding="utf-8")
    tampered = manifest_path.read_bytes()

    with pytest.raises(FileExistsError, match="failed validation"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=output_root,
        )

    assert manifest_path.read_bytes() == tampered


def test_missing_case_id_is_rejected(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path, case_id=None)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )

    with pytest.raises(ValueError, match="case_id is required"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=tmp_path / "exports",
        )


def test_explicit_synthetic_case_id_exports_standalone_without_source_rewrite(
    tmp_path: Path,
) -> None:
    package_path = _run_package(tmp_path, case_id=None)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    before = _snapshot(package_path.parent)

    result = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=tmp_path / "exports",
        target_case_id="case_exp_rag_001_synthetic",
    )

    mapping = json.loads(
        (
            Path(result.export_path)
            / "artifacts/adapter/adapter_mapping_manifest.json"
        ).read_text()
    )
    assert result.case_id == "case_exp_rag_001_synthetic"
    assert mapping["source_case_id"] is None
    assert mapping["target_case_id"] == "case_exp_rag_001_synthetic"
    assert _snapshot(package_path.parent) == before


def test_nonterminal_package_is_rejected(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path, status="initialized")
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )

    with pytest.raises(ValueError, match="verified run-native package"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=tmp_path / "exports",
        )


def test_hash_mismatch_is_rejected_without_output(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    package_path.with_name("retrieval_results.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    output_root = tmp_path / "exports"

    with pytest.raises(ValueError, match="verified run-native package"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=output_root,
        )

    assert not output_root.exists()


def test_configuration_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    configuration_path = package_path.parent / "run_configuration.json"
    configuration = json.loads(configuration_path.read_text())
    configuration["case_id"] = "different_case"
    configuration_path.write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    package = json.loads(package_path.read_text())
    for reference in package["artifact_references"]:
        if reference["relative_path"] == "run_configuration.json":
            reference["sha256"] = hashlib.sha256(
                configuration_path.read_bytes()
            ).hexdigest()
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )

    with pytest.raises(ValueError, match="verified run-native package"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=tmp_path / "exports",
        )


def test_missing_evidence_request_is_rejected(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    request_path.unlink()

    with pytest.raises(ValueError, match="Evidence Request"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=tmp_path / "exports",
        )


def test_incomplete_evidence_request_is_rejected(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    request_path.write_text(
        "# Layer 2 Evidence Request\n\n"
        "## Request ID\n\nER-EXP-RAG-001\n\n"
        "## Case ID\n\ncase_exp_rag_001_synthetic\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Evidence Request has no resolved value"):
        _export(
            package_path=package_path,
            framework_root=framework_root,
            request_path=request_path,
            output_root=tmp_path / "exports",
        )


def test_validator_detects_governed_object_tampering(tmp_path: Path) -> None:
    package_path = _run_package(tmp_path)
    framework_root, request_path = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
    )
    result = _export(
        package_path=package_path,
        framework_root=framework_root,
        request_path=request_path,
        output_root=tmp_path / "exports",
    )
    package_object_path = Path(result.export_path) / "governed/evidence_package.json"
    package_object = json.loads(package_object_path.read_text())
    package_object["readiness_decision"] = "ready"
    package_object_path.write_text(json.dumps(package_object), encoding="utf-8")

    validation = validate_governed_export(
        export_dir=Path(result.export_path),
        profile=FrozenV03Profile.load(framework_root),
    )

    assert validation.validation_status == "FAIL"
    assert "contract.Evidence Package.no_non_v03_fields" in {
        check.check_id for check in validation.checks if check.status == "fail"
    }


def test_synthetic_to_client_reuse_creates_new_record_and_preserves_source(
    tmp_path: Path,
) -> None:
    synthetic_case = "case_exp_rag_001_synthetic"
    package_path = _run_package(tmp_path / "run", case_id=synthetic_case)
    framework_root, synthetic_request = _framework_root(
        tmp_path,
        case_id=synthetic_case,
        case_name="CASE-SYNTHETIC",
        request_id="ER-SYNTHETIC",
    )
    synthetic = export_verified_run_v03(
        package_path=package_path,
        framework_root=framework_root,
        framework_case_name="CASE-SYNTHETIC",
        evidence_request_path=synthetic_request,
        output_root=tmp_path / "exports",
    )
    client_case = "client_case_alpha"
    _, client_request = _framework_root(
        tmp_path,
        case_id=client_case,
        case_name="CASE-CLIENT-ALPHA",
        request_id="ER-CLIENT-ALPHA",
    )
    source_before = _snapshot(Path(synthetic.export_path))

    reused = reuse_synthetic_export_v03(
        source_export_dir=Path(synthetic.export_path),
        framework_root=framework_root,
        framework_case_name="CASE-CLIENT-ALPHA",
        evidence_request_path=client_request,
        output_root=tmp_path / "exports",
    )

    reused_path = Path(reused.export_path)
    record = json.loads(
        (
            reused_path / "governed/evidence_records/synthetic_reuse.json"
        ).read_text()
    )
    mapping = json.loads(
        (
            reused_path / "artifacts/adapter/adapter_mapping_manifest.json"
        ).read_text()
    )
    assert reused.validation.validation_status == "PASS"
    assert record["case_id"] == client_case
    assert record["object_id"] != synthetic.governed_object_ids[
        "experiment_outcome_evidence"
    ]
    assert synthetic.governed_object_ids["experiment_outcome_evidence"] in record[
        "provenance_refs"
    ]
    assert SYNTHETIC_CLIENT_LIMITATION in record["limitations"]
    assert mapping["semantic_relation"] == "derived_from_synthetic_evidence"
    assert mapping["client_system_verified"] is False
    assert _snapshot(Path(synthetic.export_path)) == source_before


def test_synthetic_reuse_in_two_client_cases_creates_separate_records(
    tmp_path: Path,
) -> None:
    package_path = _run_package(
        tmp_path / "run", case_id="case_exp_rag_001_synthetic"
    )
    framework_root, synthetic_request = _framework_root(
        tmp_path,
        case_id="case_exp_rag_001_synthetic",
        case_name="CASE-SYNTHETIC",
        request_id="ER-SYNTHETIC",
    )
    synthetic = export_verified_run_v03(
        package_path=package_path,
        framework_root=framework_root,
        framework_case_name="CASE-SYNTHETIC",
        evidence_request_path=synthetic_request,
        output_root=tmp_path / "exports",
    )

    records: list[dict[str, object]] = []
    for suffix in ("ALPHA", "BETA"):
        case_id = f"client_case_{suffix.lower()}"
        case_name = f"CASE-CLIENT-{suffix}"
        _, request = _framework_root(
            tmp_path,
            case_id=case_id,
            case_name=case_name,
            request_id=f"ER-CLIENT-{suffix}",
        )
        reused = reuse_synthetic_export_v03(
            source_export_dir=Path(synthetic.export_path),
            framework_root=framework_root,
            framework_case_name=case_name,
            evidence_request_path=request,
            output_root=tmp_path / "exports",
        )
        records.append(
            json.loads(
                (
                    Path(reused.export_path)
                    / "governed/evidence_records/synthetic_reuse.json"
                ).read_text()
            )
        )

    assert records[0]["case_id"] != records[1]["case_id"]
    assert records[0]["object_id"] != records[1]["object_id"]
    assert records[0]["provenance_refs"][1:] == records[1]["provenance_refs"][1:]
