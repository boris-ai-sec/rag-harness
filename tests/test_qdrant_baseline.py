from uuid import uuid4

from qdrant_client import QdrantClient, models


def test_qdrant_baseline_data_path() -> None:
    client = QdrantClient(url="http://127.0.0.1:6333")
    collection = f"rag_harness_test_{uuid4().hex}"

    client.create_collection(
        collection_name=collection,
        vectors_config=models.VectorParams(
            size=4,
            distance=models.Distance.COSINE,
        ),
    )

    try:
        client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=1, vector=[1.0, 0.0, 0.0, 0.0], payload={"text": "metadata filtering"}),
                models.PointStruct(id=2, vector=[0.0, 1.0, 0.0, 0.0], payload={"text": "telemetry tracing"}),
            ],
        )

        result = client.query_points(
            collection_name=collection,
            query=[0.9, 0.1, 0.0, 0.0],
            limit=2,
        )

        assert len(result.points) == 2
        assert result.points[0].id == 1
        assert result.points[0].payload["text"] == "metadata filtering"
        assert result.points[0].score > result.points[1].score
    finally:
        client.delete_collection(collection)
