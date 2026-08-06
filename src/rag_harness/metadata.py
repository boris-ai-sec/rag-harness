"""Boundary metadata contract for controlled RAG ingestion."""

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


REQUIRED_BOUNDARY_FIELDS = (
    "tenant_id",
    "case_id",
    "document_id",
    "source_id",
    "chunk_id",
    "document_type",
    "access_level",
)


class BoundaryMetadata(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    tenant_id: str
    case_id: str
    document_id: str
    source_id: str
    chunk_id: str
    document_type: str
    access_level: str

    @field_validator("*")
    @classmethod
    def reject_empty_values(cls, value: str) -> str:
        if not value:
            raise ValueError("boundary metadata values must not be empty")
        return value


def validate_boundary_metadata(payload: dict[str, Any]) -> BoundaryMetadata:
    return BoundaryMetadata.model_validate(payload)
