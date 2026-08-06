import pytest
from pydantic import ValidationError

from rag_harness.metadata import validate_boundary_metadata
from rag_harness.models import Chunk


def valid_metadata() -> dict[str, str]:
    return {
        "tenant_id": "tenant_a",
        "case_id": "case_001",
        "document_id": "doc_a_001",
        "source_id": "source_permitted",
        "chunk_id": "chunk_a_001",
        "document_type": "controlled_synthetic",
        "access_level": "permitted",
    }


def test_valid_metadata_is_accepted() -> None:
    metadata = validate_boundary_metadata(valid_metadata())

    assert metadata.tenant_id == "tenant_a"
    assert metadata.source_id == "source_permitted"


def test_missing_tenant_id_is_rejected() -> None:
    payload = valid_metadata()
    payload.pop("tenant_id")

    with pytest.raises(ValidationError):
        validate_boundary_metadata(payload)


def test_missing_source_id_is_rejected() -> None:
    payload = valid_metadata()
    payload.pop("source_id")

    with pytest.raises(ValidationError):
        validate_boundary_metadata(payload)


def test_chunk_payload_preserves_all_required_fields() -> None:
    chunk = Chunk(
        text="Controlled synthetic metadata boundary record.",
        metadata=validate_boundary_metadata(valid_metadata()),
    )

    payload = chunk.to_qdrant_payload()

    assert payload["text"] == chunk.text
    assert payload["tenant_id"] == "tenant_a"
    assert payload["source_id"] == "source_permitted"


def test_chunk_payload_preserves_all_required_fields() -> None:
    chunk = Chunk(
        text="Controlled synthetic metadata boundary record.",
        metadata=validate_boundary_metadata(valid_metadata()),
    )

    payload = chunk.to_qdrant_payload()

    assert payload["text"] == chunk.text
    assert payload["tenant_id"] == "tenant_a"
    assert payload["source_id"] == "source_permitted"
