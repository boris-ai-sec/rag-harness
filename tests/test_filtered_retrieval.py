import pytest

from rag_harness.retrieval import filtered_retrieval
from scripts.metadata_ingestion_baseline import (
    COLLECTION,
    build_collection,
    index_valid_chunks,
)
from scripts.semantic_retrieval_baseline import embed


@pytest.mark.service
def test_permitted_tenant_and_source_return_only_allowed_chunk() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
    )

    assert run.retrieval_outcome == "records_returned"
    assert run.returned_chunk_ids == ["doc_a_001-chunk-0001"]


@pytest.mark.service
def test_other_source_in_same_tenant_is_not_returned() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A other source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
    )

    assert "doc_a_002-chunk-0001" not in run.returned_chunk_ids


@pytest.mark.service
def test_other_tenant_is_not_returned() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant B isolated source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
    )

    assert "doc_b_001-chunk-0001" not in run.returned_chunk_ids


@pytest.mark.service
def test_unfiltered_control_exposes_cross_boundary_records() -> None:
    from rag_harness.retrieval import unfiltered_retrieval

    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant B isolated source record.",
        "search_query",
    )

    run = unfiltered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
    )

    assert run.retrieval_mode == "unfiltered_control"
    assert run.filter_parameters == {}
    assert run.retrieval_outcome == "records_returned"
    assert "doc_b_001-chunk-0001" in run.returned_chunk_ids
    assert "doc_a_001-chunk-0001" in run.returned_chunk_ids


@pytest.mark.service
def test_filtered_run_contains_required_run_metadata() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
    )

    assert run.run_id.startswith("run-")
    assert run.collection == COLLECTION
    assert run.filter_parameters == {
        "tenant_id": "tenant_a",
        "source_id": "source_permitted",
    }
    assert run.boundary_verification_status == "passed"
    assert "match" in run.verification_reason
    assert run.returned_chunk_ids
    assert run.payload_references


def test_boundary_verification_passes_for_matching_payload() -> None:
    from rag_harness.retrieval import verify_boundary

    status, reason = verify_boundary(
        filter_parameters={
            "tenant_id": "tenant_a",
            "source_id": "source_permitted",
        },
        payload_references=[
            {
                "chunk_id": "chunk-001",
                "tenant_id": "tenant_a",
                "source_id": "source_permitted",
            }
        ],
    )

    assert status == "passed"
    assert "match" in reason


def test_boundary_verification_fails_for_mismatched_payload() -> None:
    from rag_harness.retrieval import verify_boundary

    status, reason = verify_boundary(
        filter_parameters={
            "tenant_id": "tenant_a",
            "source_id": "source_permitted",
        },
        payload_references=[
            {
                "chunk_id": "chunk-002",
                "tenant_id": "tenant_b",
                "source_id": "source_permitted",
            }
        ],
    )

    assert status == "failed"
    assert "tenant_id" in reason


def test_boundary_verification_is_indeterminate_for_missing_metadata() -> None:
    from rag_harness.retrieval import verify_boundary

    status, reason = verify_boundary(
        filter_parameters={
            "tenant_id": "tenant_a",
            "source_id": "source_permitted",
        },
        payload_references=[
            {
                "chunk_id": "chunk-003",
                "source_id": "source_permitted",
            }
        ],
    )

    assert status == "indeterminate"
    assert "tenant_id" in reason


@pytest.mark.service
def test_filtered_run_is_indeterminate_when_returned_payload_lacks_boundary_metadata() -> (
    None
):
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
        payload_fields=[
            "document_id",
            "chunk_id",
            "source_id",
        ],
    )

    assert run.returned_chunk_ids == ["doc_a_001-chunk-0001"]
    assert run.payload_references[0]["tenant_id"] is None
    assert run.retrieval_outcome == "records_returned"
    assert run.boundary_verification_status == "indeterminate"
    assert "tenant_id" in run.verification_reason


@pytest.mark.service
def test_filtered_retrieval_rejects_missing_boundary_filter() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    with pytest.raises(
        ValueError,
        match="tenant_id, source_id",
    ):
        filtered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=query_vector,
            tenant_id=None,
            source_id=None,
        )


@pytest.mark.service
def test_filtered_retrieval_rejects_partial_boundary_filter() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    with pytest.raises(
        ValueError,
        match="source_id",
    ):
        filtered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=query_vector,
            tenant_id="tenant_a",
            source_id=None,
        )


@pytest.mark.service
def test_incorrect_boundary_values_return_no_records() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_unknown",
        source_id="source_unknown",
    )

    assert run.returned_chunk_ids == []
    assert run.payload_references == []
    assert run.retrieval_outcome == "empty"
    assert run.boundary_verification_status == "passed"
