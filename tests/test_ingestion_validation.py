import pytest
from pydantic import ValidationError

from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.ingestion import prepare_record


def test_valid_record_is_prepared_as_chunk() -> None:
    chunk = prepare_record(CONTROLLED_CORPUS[0])

    assert chunk.metadata.tenant_id == "tenant_a"
    assert chunk.metadata.chunk_id == "doc_a_001-chunk-0001"


def test_malformed_record_is_rejected_before_ingestion() -> None:
    with pytest.raises(ValidationError):
        prepare_record(CONTROLLED_CORPUS[3])


def test_validation_failure_produces_diagnostic() -> None:
    malformed = CONTROLLED_CORPUS[3]

    try:
        prepare_record(malformed)
    except ValidationError as error:
        from rag_harness.ingestion import validation_diagnostic

        diagnostic = validation_diagnostic(malformed, error)

    assert diagnostic["status"] == "rejected"
    assert diagnostic["stage"] == "validation_before_ingestion"
    assert diagnostic["document_id"] == "doc_invalid_001"
    assert diagnostic["error_type"] == "ValidationError"


def test_prepare_corpus_separates_valid_and_rejected_records() -> None:
    from rag_harness.ingestion import prepare_corpus

    chunks, diagnostics = prepare_corpus(CONTROLLED_CORPUS)

    assert len(chunks) == 3
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "rejected"
    assert diagnostics[0]["document_id"] == "doc_invalid_001"


def test_controlled_corpus_uses_single_stable_case_id() -> None:
    from rag_harness.corpus import CONTROLLED_CORPUS

    case_ids = {
        record["case_id"]
        for record in CONTROLLED_CORPUS
    }

    assert case_ids == {
        "case_exp_rag_001_synthetic"
    }
