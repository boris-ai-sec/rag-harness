import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
artifact_path = Path("artifacts/LAB-RH-02B/qdrant_baseline_result.json")
artifact_path.parent.mkdir(parents=True, exist_ok=True)


from qdrant_client import QdrantClient, models

client = QdrantClient(url="http://127.0.0.1:6333")
collection = "rag_harness_baseline"

if client.collection_exists(collection):
    client.delete_collection(collection)

client.create_collection(
    collection_name=collection,
    vectors_config=models.VectorParams(size=4, distance=models.Distance.COSINE),
)

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

for point in result.points:
    print(point.id, round(point.score, 4), point.payload["text"])

artifact = {
    "lab_id": "LAB-RH-02B",
    "created_at": datetime.now(UTC).isoformat(),
    "collection": collection,
    "vector_size": 4,
    "distance": "Cosine",
    "query_vector": [0.9, 0.1, 0.0, 0.0],
    "qdrant_client_version": version("qdrant-client"),
    "results": [
        {
            "id": point.id,
            "score": round(point.score, 6),
            "text": point.payload["text"],
        }
        for point in result.points
    ],
}

artifact_path.write_text(
    json.dumps(artifact, indent=2),
    encoding="utf-8",
)

print(f"ARTIFACT: {artifact_path}")
