import os
import shutil
from pathlib import Path

import pytest

from rag_harness.contracts import RunConfiguration
from rag_harness.governed_export import export_verified_run_v03
from rag_harness.registry import EXP_RAG_001
from rag_harness.run_storage import (
    finalize_run_package,
    initialize_run_package,
    write_json_artifact,
)

CONFIG_PATH = Path("configs/exp_rag_001.json")
CASE_NAME = "AR-2026-001_CRM_AI_Assistant"
CASE_ID = "AR-2026-001"


@pytest.mark.framework
def test_export_conforms_to_extracted_frozen_v03_and_check_case(
    tmp_path: Path,
) -> None:
    configured_root = os.environ.get("FRAMEWORK_V03_ROOT")
    if not configured_root:
        pytest.skip("FRAMEWORK_V03_ROOT is not configured")
    authoritative_root = Path(configured_root)
    if not authoritative_root.is_dir():
        pytest.skip("FRAMEWORK_V03_ROOT does not exist")

    framework_root = tmp_path / "frozen-v03"
    shutil.copytree(authoritative_root, framework_root)
    request_path = (
        framework_root
        / "03_Test_Cases"
        / CASE_NAME
        / "layer2/evidence_requests/ER-EXP-RAG-001.md"
    )
    request_path.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(
        "# Layer 2 Evidence Request\n\n"
        "## Request ID\n\nER-EXP-RAG-001\n\n"
        "## Request Version\n\nV0.1\n\n"
        f"## Case ID\n\n{CASE_ID}\n\n"
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

    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    ).model_copy(update={"case_id": None})
    workspace, package = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
    )
    artifact = write_json_artifact(
        workspace.directory / "retrieval_results.json",
        {
            "run_id": workspace.run_id,
            "experiment_id": "EXP-RAG-001",
            "case_id": None,
            "experiment_result": "pass",
            "scenarios": [],
        },
    )
    finalize_run_package(
        workspace=workspace,
        package=package,
        status="completed",
        experiment_result="pass",
        artifact_paths={"retrieval_results": artifact},
    )

    result = export_verified_run_v03(
        package_path=workspace.package_path,
        framework_root=framework_root,
        framework_case_name=CASE_NAME,
        evidence_request_path=request_path,
        output_root=tmp_path / "exports",
        target_case_id=CASE_ID,
    )

    assert result.validation.validation_status == "PASS"
    assert result.validation.failure_count == 0
    assert result.case_id == CASE_ID
