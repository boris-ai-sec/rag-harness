from scripts.metadata_ingestion_baseline import (
    COLLECTION,
    build_collection,
    index_valid_chunks,
)
from scripts.semantic_retrieval_baseline import embed

from rag_harness.retrieval import filtered_retrieval


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

    assert run.returned_chunk_ids == [
        "doc_a_001-chunk-0001"
    ]


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
    assert "doc_b_001-chunk-0001" in run.returned_chunk_ids
    assert "doc_a_001-chunk-0001" in run.returned_chunk_ids


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
    assert run.returned_chunk_ids
    assert run.payload_references
