import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_harness.contracts import (
    ExperimentScenario,
    RunConfiguration,
    RunPackage,
)
from rag_harness.registry import (
    EXP_RAG_001,
    get_experiment,
    validate_experiment_configuration,
)
from rag_harness.run_storage import initialize_run_package


CONFIG_PATH = Path("configs/exp_rag_001.json")


def load_configuration() -> RunConfiguration:
    return RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    )


def test_exp_rag_001_configuration_round_trips_as_json() -> None:
    configuration = load_configuration()

    serialized = configuration.model_dump_json()
    restored = RunConfiguration.model_validate_json(serialized)

    assert restored == configuration
    assert restored.case_id == "case_exp_rag_001_synthetic"
    assert restored.top_k == 10


def test_standalone_configuration_allows_missing_case_id() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload.pop("case_id")

    configuration = RunConfiguration.model_validate(payload)

    assert configuration.case_id is None


def test_configuration_rejects_unknown_fields() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["unsafe_fallback"] = True

    with pytest.raises(ValidationError, match="Extra inputs"):
        RunConfiguration.model_validate(payload)


def test_configuration_rejects_empty_boundary_values() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["tenant_id"] = ""

    with pytest.raises(ValidationError, match="at least 1 character"):
        RunConfiguration.model_validate(payload)


def test_configuration_rejects_non_loopback_qdrant() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["qdrant_url"] = "https://qdrant.example.test"

    with pytest.raises(ValidationError, match="loopback"):
        RunConfiguration.model_validate(payload)


def test_configuration_validates_default_qdrant_url() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload.pop("qdrant_url")

    configuration = RunConfiguration.model_validate(payload)

    assert configuration.qdrant_url.host == "127.0.0.1"


def test_unfiltered_scenario_must_be_explicit_control() -> None:
    with pytest.raises(ValidationError, match="control_only"):
        ExperimentScenario(
            scenario_id="unsafe_unfiltered",
            retrieval_mode="unfiltered_control",
            control_only=False,
        )


def test_registry_returns_exp_rag_001_definition() -> None:
    definition = get_experiment("EXP-RAG-001")

    assert definition == EXP_RAG_001
    assert definition.client_system_verified is False
    assert definition.production_isolation_verified is False


def test_registry_fails_closed_for_unknown_experiment() -> None:
    with pytest.raises(ValueError, match="unknown experiment_id"):
        get_experiment("EXP-RAG-999")


def test_registry_allows_client_scoped_case_identity() -> None:
    configuration = load_configuration().model_copy(
        update={"case_id": "case_client_assessment_001"}
    )

    validate_experiment_configuration(EXP_RAG_001, configuration)


def test_run_package_cannot_claim_governed_export() -> None:
    with pytest.raises(ValidationError, match="False"):
        RunPackage(
            run_id="run-2742eadb-2d12-4d6a-bbe7-9026485c20f1",
            experiment_id="EXP-RAG-001",
            case_id="case_exp_rag_001_synthetic",
            governed_v0_3_export_claimed=True,
        )


def test_initialize_run_package_creates_non_overwriting_workspace(
    tmp_path: Path,
) -> None:
    configuration = load_configuration()
    run_id = "run-2742eadb-2d12-4d6a-bbe7-9026485c20f1"

    workspace, package = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
        run_id=run_id,
    )

    assert workspace.directory.name == run_id
    assert workspace.configuration_path.is_file()
    assert workspace.definition_path.is_file()
    assert workspace.package_path.is_file()
    assert package.status == "initialized"
    assert package.governed_v0_3_export_claimed is False
    assert len(package.artifact_references) == 2
    assert all(
        reference.sha256 is not None
        for reference in package.artifact_references
    )

    persisted_package = json.loads(
        workspace.package_path.read_text(encoding="utf-8")
    )
    assert persisted_package["run_id"] == run_id
    assert persisted_package["observability_status"] == "pending"

    with pytest.raises(FileExistsError):
        initialize_run_package(
            runs_root=tmp_path / "runs",
            definition=EXP_RAG_001,
            configuration=configuration,
            run_id=run_id,
        )


def test_initialize_standalone_run_package_preserves_missing_case_id(
    tmp_path: Path,
) -> None:
    configuration = load_configuration().model_copy(update={"case_id": None})

    workspace, package = initialize_run_package(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
    )

    assert package.case_id is None
    persisted_configuration = json.loads(
        workspace.configuration_path.read_text(encoding="utf-8")
    )
    persisted_package = json.loads(
        workspace.package_path.read_text(encoding="utf-8")
    )
    assert persisted_configuration["case_id"] is None
    assert persisted_package["case_id"] is None


def test_initialize_run_package_rejects_path_like_run_id(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="run_id"):
        initialize_run_package(
            runs_root=tmp_path / "runs",
            definition=EXP_RAG_001,
            configuration=load_configuration(),
            run_id="../escape",
        )
