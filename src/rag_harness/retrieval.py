"""Filtered retrieval contracts for LAB-RH-03B."""

from typing import Any, Literal
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
    retrieval_outcome: Literal[
        "records_returned",
        "empty",
    ]
    boundary_verification_status: str = "indeterminate"
    verification_reason: str | None = None
    returned_chunk_ids: list[str]
    payload_references: list[dict[str, Any]]


from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
)



def verify_boundary(
    *,
    filter_parameters: dict[str, str],
    payload_references: list[dict[str, Any]],
) -> tuple[str, str]:
    required_fields = {
        "tenant_id",
        "source_id",
        "chunk_id",
    }

    if not payload_references:
        return (
            "passed",
            "no records returned; no boundary violation observed",
        )

    for payload in payload_references:
        missing_fields = {
            field
            for field in required_fields
            if payload.get(field) is None
        }

        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            return (
                "indeterminate",
                f"missing required payload metadata: {missing}",
            )

        for key, expected_value in filter_parameters.items():
            if payload.get(key) != expected_value:
                return (
                    "failed",
                    f"payload {key} does not match filter",
                )

    return (
        "passed",
        "all returned payloads match the boundary filter",
    )



def validate_boundary_filter(
    *,
    tenant_id: str | None,
    source_id: str | None,
) -> dict[str, str]:
    missing_fields = [
        field_name
        for field_name, field_value in {
            "tenant_id": tenant_id,
            "source_id": source_id,
        }.items()
        if not field_value
    ]

    if missing_fields:
        missing = ", ".join(missing_fields)
        raise ValueError(
            f"missing required boundary filter: {missing}"
        )

    return {
        "tenant_id": tenant_id,
        "source_id": source_id,
    }


def filtered_retrieval(
    client: QdrantClient,
    collection: str,
    query_vector: list[float],
    tenant_id: str | None,
    source_id: str | None,
    scenario: str | None = None,
    limit: int = 10,
    payload_fields: list[str] | None = None,
) -> RetrievalRun:
    filter_parameters = validate_boundary_filter(
        tenant_id=tenant_id,
        source_id=source_id,
    )

    query_filter = Filter(
        must=[
            FieldCondition(
                key="tenant_id",
                match=MatchValue(value=filter_parameters["tenant_id"]),
            ),
            FieldCondition(
                key="source_id",
                match=MatchValue(value=filter_parameters["source_id"]),
            ),
        ]
    )

    results = client.query_points(
        collection_name=collection,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=payload_fields or True,
    ).points

    payload_references = [
        {
            "point_id": result.id,
            "document_id": result.payload.get("document_id"),
            "chunk_id": result.payload.get("chunk_id"),
            "tenant_id": result.payload.get("tenant_id"),
            "source_id": result.payload.get("source_id"),
        }
        for result in results
    ]

    verification_status, verification_reason = verify_boundary(
        filter_parameters=filter_parameters,
        payload_references=payload_references,
    )

    return RetrievalRun(
        collection=collection,
        scenario=scenario,
        retrieval_mode="filtered",
        filter_parameters=filter_parameters,
        retrieval_outcome=(
            "records_returned"
            if results
            else "empty"
        ),
        boundary_verification_status=verification_status,
        verification_reason=verification_reason,
        returned_chunk_ids=[
            payload["chunk_id"]
            for payload in payload_references
            if isinstance(payload.get("chunk_id"), str)
        ],
        payload_references=payload_references,
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
        retrieval_outcome=(
            "records_returned"
            if results
            else "empty"
        ),
        returned_chunk_ids=[
            result.payload["chunk_id"]
            for result in results
        ],
        payload_references=[
            {
                "point_id": result.id,
                "document_id": result.payload.get("document_id"),
                "chunk_id": result.payload.get("chunk_id"),
                "tenant_id": result.payload.get("tenant_id"),
                "source_id": result.payload.get("source_id"),
            }
            for result in results
        ],
    )
