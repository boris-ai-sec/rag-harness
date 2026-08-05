from scripts.semantic_retrieval_baseline import (
    build_collection,
    index_documents,
    semantic_search,
)


def test_semantic_retrieval_returns_metadata_filtering_first():
    client = build_collection()
    index_documents(client)

    query = "How can retrieval be restricted to permitted document attributes?"
    results = semantic_search(client, query, limit=3)

    assert len(results) == 3
    assert results[0].payload["topic"] == "metadata_filtering"
    assert results[0].score > results[1].score
