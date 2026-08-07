"""Qdrant adapter preserving explicit boundary-filter behavior."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_harness.contracts import ExperimentScenario, RunConfiguration
from rag_harness.models import Chunk
from rag_harness.retrieval import (
    RetrievalRun,
    filtered_retrieval,
    unfiltered_retrieval,
)


@dataclass
class QdrantVectorStore:
    client: QdrantClient

    @classmethod
    def from_configuration(
        cls,
        configuration: RunConfiguration,
    ) -> QdrantVectorStore:
        return cls(
            QdrantClient(
                url=str(configuration.qdrant_url),
                trust_env=False,
            )
        )

    def preflight(self) -> None:
        self.client.get_collections()

    def prepare_collection(
        self,
        *,
        base_name: str,
        run_id: str,
        vector_size: int,
    ) -> str:
        suffix = run_id.removeprefix("run-").replace("-", "")
        collection = f"{base_name}_{suffix}"
        if self.client.collection_exists(collection):
            raise FileExistsError(f"run collection already exists: {collection}")
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
        return collection

    def index_chunks(
        self,
        *,
        collection: str,
        run_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("each chunk requires exactly one embedding")
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, f"{run_id}:{chunk.metadata.chunk_id}")),
                vector=vector,
                payload=chunk.to_qdrant_payload(),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(
            collection_name=collection,
            points=points,
            wait=True,
        )

    def retrieve(
        self,
        *,
        collection: str,
        run_id: str,
        scenario: ExperimentScenario,
        query_vector: list[float],
        limit: int,
    ) -> RetrievalRun:
        if scenario.retrieval_mode == "unfiltered_control":
            return unfiltered_retrieval(
                client=self.client,
                collection=collection,
                query_vector=query_vector,
                scenario=scenario.scenario_id,
                limit=limit,
                run_id=run_id,
            )
        return filtered_retrieval(
            client=self.client,
            collection=collection,
            query_vector=query_vector,
            tenant_id=scenario.filter_parameters.tenant_id,
            source_id=scenario.filter_parameters.source_id,
            scenario=scenario.scenario_id,
            limit=limit,
            payload_fields=list(scenario.payload_projection),
            run_id=run_id,
        )
