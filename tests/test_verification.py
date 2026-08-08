import hashlib
import json
from pathlib import Path

from rag_harness.contracts import RunConfiguration
from rag_harness.registry import EXP_RAG_001
from rag_harness.run_storage import (
    finalize_run_package,
    initialize_run_package,
    write_json_artifact,
)
from rag_harness.verification import verify_run_package

CONFIG_PATH = Path("configs/exp_rag_001.json")
FIXED_RUN_ID = "run-2742eadb-2d12-4d6a-bbe7-9026485c20f1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finalized_package(tmp_path: Path) -> Path:
    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    workspace, package = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
        run_id=FIXED_RUN_ID,
    )
    retrieval_path = write_json_artifact(
        workspace.directory / "retrieval_results.json",
        {
            "run_id": FIXED_RUN_ID,
            "experiment_id": "EXP-RAG-001",
            "case_id": configuration.case_id,
            "experiment_result": "pass",
        },
    )
    finalize_run_package(
        workspace=workspace,
        package=package,
        status="completed",
        experiment_result="pass",
        artifact_paths={"retrieval_results": retrieval_path},
    )
    return workspace.package_path


def _failed_checks(package_path: Path) -> set[str]:
    result = verify_run_package(package_path)
    return {check.check_id for check in result.checks if check.status == "fail"}


def test_verifier_accepts_untampered_finalized_package_without_modifying_it(
    tmp_path: Path,
) -> None:
    package_path = _finalized_package(tmp_path)
    before = {
        path.relative_to(package_path.parent).as_posix(): path.read_bytes()
        for path in package_path.parent.iterdir()
        if path.is_file()
    }

    result = verify_run_package(package_path)

    after = {
        path.relative_to(package_path.parent).as_posix(): path.read_bytes()
        for path in package_path.parent.iterdir()
        if path.is_file()
    }
    assert result.verification_status == "verified"
    assert result.failure_count == 0
    assert result.run_id == FIXED_RUN_ID
    assert result.artifact_count == 3
    assert result.read_only is True
    assert result.governed_v0_3_export_claimed is False
    assert before == after


def test_verifier_detects_artifact_tampering(tmp_path: Path) -> None:
    package_path = _finalized_package(tmp_path)
    retrieval_path = package_path.with_name("retrieval_results.json")
    retrieval_path.write_text('{"tampered": true}\n', encoding="utf-8")

    failed = _failed_checks(package_path)

    assert "artifact.2.retrieval_results.digest_matches" in failed


def test_verifier_rejects_initialized_package_as_unfinalized_evidence(
    tmp_path: Path,
) -> None:
    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    workspace, _ = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
        run_id=FIXED_RUN_ID,
    )

    failed = _failed_checks(workspace.package_path)

    assert "package.terminal_status" in failed


def test_verifier_detects_missing_referenced_artifact(tmp_path: Path) -> None:
    package_path = _finalized_package(tmp_path)
    package_path.with_name("retrieval_results.json").unlink()

    failed = _failed_checks(package_path)

    assert "artifact.2.retrieval_results.exists" in failed


def test_verifier_rejects_symlink_even_when_target_hash_matches(
    tmp_path: Path,
) -> None:
    package_path = _finalized_package(tmp_path)
    retrieval_path = package_path.with_name("retrieval_results.json")
    external_path = tmp_path / "external-retrieval.json"
    external_path.write_bytes(retrieval_path.read_bytes())
    retrieval_path.unlink()
    retrieval_path.symlink_to(external_path)

    failed = _failed_checks(package_path)

    assert "artifact.2.retrieval_results.contained" in failed
    assert "artifact.2.retrieval_results.not_symlink" in failed


def test_verifier_detects_identity_mismatch_even_with_updated_digest(
    tmp_path: Path,
) -> None:
    package_path = _finalized_package(tmp_path)
    configuration_path = package_path.with_name("run_configuration.json")
    configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
    configuration["case_id"] = "case_other"
    configuration_path.write_text(
        json.dumps(configuration, indent=2) + "\n",
        encoding="utf-8",
    )
    package = json.loads(package_path.read_text(encoding="utf-8"))
    package["artifact_references"][0]["sha256"] = _sha256(configuration_path)
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")

    failed = _failed_checks(package_path)

    assert "artifact.0.effective_run_configuration.case_id_matches" in failed
    assert "identity.configuration_matches" in failed


def test_verifier_detects_unreferenced_file(tmp_path: Path) -> None:
    package_path = _finalized_package(tmp_path)
    package_path.with_name("unreferenced.json").write_text("{}\n", encoding="utf-8")

    failed = _failed_checks(package_path)

    assert "artifacts.no_unreferenced_files" in failed


def test_verifier_rejects_malformed_package(tmp_path: Path) -> None:
    package_path = tmp_path / "run_package.json"
    package_path.write_text('{"run_id":', encoding="utf-8")

    result = verify_run_package(package_path)

    assert result.verification_status == "failed"
    assert result.run_id is None
    assert "package.contract" in {
        check.check_id for check in result.checks if check.status == "fail"
    }
