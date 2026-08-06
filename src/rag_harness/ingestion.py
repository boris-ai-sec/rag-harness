"""Validation-before-ingestion helpers."""

from typing import Any

from rag_harness.metadata import BoundaryMetadata
from rag_harness.models import Chunk, Document


def prepare_record(raw_record: dict[str, Any]) -> Chunk:
    document = Document.model_validate(raw_record)

    metadata = BoundaryMetadata(
        tenant_id=document.tenant_id,
        case_id=document.case_id,
        document_id=document.document_id,
        source_id=document.source_id,
        chunk_id=f"{document.document_id}-chunk-0001",
        document_type=document.document_type,
        access_level=document.access_level,
    )

    return Chunk(
        text=document.text,
        metadata=metadata,
    )


def validation_diagnostic(
    raw_record: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "status": "rejected",
        "stage": "validation_before_ingestion",
        "document_id": raw_record.get("document_id"),
        "error_type": type(error).__name__,
        "error": str(error),
    }


def prepare_corpus(
    raw_records: list[dict[str, Any]],
) -> tuple[list[Chunk], list[dict[str, Any]]]:
    chunks: list[Chunk] = []
    diagnostics: list[dict[str, Any]] = []

    for raw_record in raw_records:
        try:
            chunks.append(prepare_record(raw_record))
        except Exception as error:
            diagnostics.append(
                validation_diagnostic(raw_record, error)
            )

    return chunks, diagnostics
