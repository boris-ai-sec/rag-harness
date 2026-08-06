"""Filtered retrieval contracts for LAB-RH-03B."""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class RetrievalRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(
        default_factory=lambda: f"run-{uuid4()}"
    )
    collection: str
    scenario: str | None = None
    retrieval_mode: str
    filter_parameters: dict[str, str]
    returned_chunk_ids: list[str]
    payload_references: list[dict[str, Any]]


from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)


def filtered_retrieval(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    tenant_id: str,
    source_id: str,
    scenario: str | None = None,
    limit: int = 10,
) -> RetrievalRun:
    query_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=tenant_id),
            ),
            FieldCondition(
                key="source_id",
                match=MatchValue(value=source_id),
            ),
        ]
    )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    ).points

    return RetrievalRun(
        collection=collection,
        scenario=scenario,
        retrieval_mode="filtered",
        filter_parameters={
            "tenant_id": tenant_id,
            "source_id": source_id,
        },
        returned_chunk_ids=[
            result.payload["chunk_id"]
            for result in results
        ],
        payload_references=[
            {
                "point_id": result.id,
                "document_id": result.payload["document_id"],
                "chunk_id": result.payload["chunk_id"],
                "tenant_id": result.payload["tenant_id"],
                "source_id": result.payload["source_id"],
            }
            for result in results
        ],
    )


def unfiltered_retrieval(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    scenario: str | None = None,
    limit: int = 10,
) -> RetrievalRun:
    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points

    return RetrievalRun(
        collection=collection,
        scenario=scenario,
        retrieval_mode="unfiltered_control",
        filter_parameters={},
        returned_chunk_ids=[
            result.payload["chunk_id"]
            for result in results
        ],
        payload_references=[
            {
                "point_id": result.id,
                "document_id": result.payload["document_id"],
                "chunk_id": result.payload["chunk_id"],
                "tenant_id": result.payload["tenant_id"],
                "source_id": result.payload["source_id"],
            }
            for result in results
        ],
    )
