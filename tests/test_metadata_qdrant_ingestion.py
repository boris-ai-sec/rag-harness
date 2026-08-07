import pytest

from scripts.metadata_ingestion_baseline import (
    COLLECTION,
    build_collection,
    index_valid_chunks,
)

pytestmark = pytest.mark.service


def test_valid_chunks_are_indexed_with_complete_payload() -> None:
    client = build_collection()
    chunks, diagnostics = index_valid_chunks(client)

    records, _ = client.scroll(
        collection_name=COLLECTION,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    assert len(chunks) == 3
    assert len(diagnostics) == 1
    assert len(records) == 3

    required_fields = {
        "tenant_id",
        "case_id",
        "document_id",
        "source_id",
        "chunk_id",
        "document_type",
        "access_level",
    }

    for record in records:
        assert required_fields.issubset(record.payload)
