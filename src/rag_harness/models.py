"""Core document and chunk models."""

from pydantic import BaseModel, ConfigDict, field_validator

from rag_harness.metadata import BoundaryMetadata


class Document(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    tenant_id: str
    case_id: str
    document_id: str
    source_id: str
    document_type: str
    access_level: str
    text: str

    @field_validator("*")
    @classmethod
    def reject_empty_values(cls, value: str) -> str:
        if not value:
            raise ValueError("document fields must not be empty")
        return value


class Chunk(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    text: str
    metadata: BoundaryMetadata

    @field_validator("text")
    @classmethod
    def reject_empty_text(cls, value: str) -> str:
        if not value:
            raise ValueError("chunk text must not be empty")
        return value

    def to_qdrant_payload(self) -> dict[str, str]:
        return {
            "text": self.text,
            **self.metadata.model_dump(),
        }
