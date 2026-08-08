import json
from pathlib import Path

import pytest

from rag_harness import cli
from rag_harness.contracts import RunConfiguration
from rag_harness.services.vector_store import QdrantVectorStore

pytestmark = pytest.mark.service
CONFIG_PATH = Path("configs/exp_rag_001.json")


def test_cli_executes_exp_rag_001_with_real_services(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    vector_store = QdrantVectorStore.from_configuration(configuration)
    collection: str | None = None

    try:
        exit_code = cli.main(
            [
                "run",
                "--config",
                str(CONFIG_PATH),
                "--runs-root",
                str(tmp_path / "runs"),
                "--standalone",
            ]
        )
        output = json.loads(capsys.readouterr().out)
        package_path = Path(output["run_package_path"])
        ingestion_path = package_path.with_name("ingestion_result.json")
        collection = json.loads(ingestion_path.read_text(encoding="utf-8"))[
            "collection"
        ]

        assert exit_code == cli.EXIT_PASS
        assert output["status"] == "completed"
        assert output["experiment_result"] == "pass"
        assert output["case_id"] is None
        assert output["governed_v0_3_export_claimed"] is False
        assert package_path.is_file()
    finally:
        if collection and vector_store.client.collection_exists(collection):
            vector_store.client.delete_collection(collection)
