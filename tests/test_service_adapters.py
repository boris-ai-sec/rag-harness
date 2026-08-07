from pathlib import Path

import pytest

from rag_harness.contracts import RunConfiguration
from rag_harness.services.embeddings import OllamaEmbeddingService
from rag_harness.services.vector_store import QdrantVectorStore

CONFIG_PATH = Path("configs/exp_rag_001.json")


def load_configuration() -> RunConfiguration:
    return RunConfiguration.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))


class FakeQdrantClient:
    def __init__(self) -> None:
        self.created: list[dict] = []

    def get_collections(self) -> object:
        return object()

    def collection_exists(self, collection: str) -> bool:
        return False

    def create_collection(self, **kwargs) -> None:
        self.created.append(kwargs)


def test_qdrant_configuration_disables_environment_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_client(**kwargs):
        captured.update(kwargs)
        return FakeQdrantClient()

    monkeypatch.setattr(
        "rag_harness.services.vector_store.QdrantClient",
        fake_client,
    )

    QdrantVectorStore.from_configuration(load_configuration())

    assert captured["trust_env"] is False
    assert captured["url"].startswith("http://127.0.0.1:6333")


def test_qdrant_collection_is_unique_and_never_deleted() -> None:
    client = FakeQdrantClient()
    store = QdrantVectorStore(client=client)  # type: ignore[arg-type]

    collection = store.prepare_collection(
        base_name="metadata_ingestion_baseline",
        run_id="run-2742eadb-2d12-4d6a-bbe7-9026485c20f1",
        vector_size=768,
    )

    assert collection.endswith("2742eadb2d124d6abbe79026485c20f1")
    assert len(client.created) == 1
    assert not hasattr(client, "delete_collection")


def test_ollama_adapter_rejects_non_loopback_endpoint() -> None:
    with pytest.raises(ValueError, match="loopback"):
        OllamaEmbeddingService(
            model="nomic-embed-text:v1.5",
            dimension=768,
            endpoint="http://ollama.example.test:11434/api/embed",
        )
