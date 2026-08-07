import json
from pathlib import Path

import pytest

from rag_harness.contracts import RunConfiguration
from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.registry import EXP_RAG_001
from rag_harness.runner import execute_experiment
from rag_harness.services.embeddings import OllamaEmbeddingService
from rag_harness.services.vector_store import QdrantVectorStore

pytestmark = pytest.mark.service
CONFIG_PATH = Path("configs/exp_rag_001.json")


def test_unified_runner_against_local_qdrant_and_ollama(tmp_path: Path) -> None:
    configuration = RunConfiguration.model_validate_json(
        CONFIG_PATH.read_text(encoding="utf-8")
    )
    embeddings = OllamaEmbeddingService(
        model=configuration.embedding_model,
        dimension=configuration.embedding_dimension,
    )
    vector_store = QdrantVectorStore.from_configuration(configuration)
    collection: str | None = None

    try:
        workspace, package = execute_experiment(
            runs_root=tmp_path / "runs",
            definition=EXP_RAG_001,
            configuration=configuration,
            corpus=CONTROLLED_CORPUS,
            embeddings=embeddings,
            vector_store=vector_store,
        )
        if (ingestion_path := workspace.directory / "ingestion_result.json").is_file():
            collection = json.loads(ingestion_path.read_text(encoding="utf-8"))[
                "collection"
            ]

        assert package.status == "completed"
        assert package.experiment_result == "pass"
        assert package.observability_status == "disabled"
        assert collection is not None
    finally:
        if collection and vector_store.client.collection_exists(collection):
            vector_store.client.delete_collection(collection)
