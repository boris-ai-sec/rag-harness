import json
from urllib.request import Request, urlopen

OLLAMA_URL = "http://127.0.0.1:11434/api/embed"
MODEL = "nomic-embed-text:v1.5"


def embed(text: str, prefix: str) -> list[float]:
    payload = json.dumps({
        "model": MODEL,
        "input": f"{prefix}: {text}",
    }).encode("utf-8")

    request = Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urlopen(request, timeout=60) as response:
        result = json.load(response)

    vector = result["embeddings"][0]

    if len(vector) != 768:
        raise RuntimeError(f"Unexpected vector size: {len(vector)}")

    return vector

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "semantic_retrieval_baseline"

DOCUMENTS = [
    {
        "id": 1,
        "text": "Metadata filtering restricts retrieval to documents with permitted attributes.",
        "topic": "metadata_filtering",
    },
    {
        "id": 2,
        "text": "Telemetry tracing records spans and helps reconstruct execution paths.",
        "topic": "telemetry",
    },
    {
        "id": 3,
        "text": "Prompt injection can manipulate a model through untrusted instructions.",
        "topic": "prompt_injection",
    },
]


def build_collection() -> QdrantClient:
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

    return client


def index_documents(client: QdrantClient) -> None:
    points = []

    for document in DOCUMENTS:
        vector = embed(document["text"], "search_document")

        points.append(
            PointStruct(
                id=document["id"],
                vector=vector,
                payload={
                    "text": document["text"],
                    "topic": document["topic"],
                },
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )


if __name__ == "__main__":
    client = build_collection()
    index_documents(client)

    info = client.get_collection(COLLECTION)

    print("COLLECTION_BUILD: PASS")
    print("COLLECTION:", COLLECTION)
    print("POINTS_COUNT:", info.points_count)
    print("VECTOR_SIZE: 768")


def semantic_search(client: QdrantClient, query: str, limit: int = 3):
    query_vector = embed(query, "search_query")

    return client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    ).points


if __name__ == "__main__":
    query = "How can retrieval be restricted to permitted document attributes?"
    results = semantic_search(client, query)

    print("SEMANTIC_SEARCH: PASS")
    print("QUERY:", query)

    for rank, result in enumerate(results, start=1):
        print(
            f"RANK {rank}: "
            f"score={result.score:.4f} "
            f"topic={result.payload['topic']}"
        )
        print("TEXT:", result.payload["text"])


if __name__ == "__main__":
    artifact = {
        "lab_id": "LAB-RH-02B",
        "status": "PASS",
        "collection": COLLECTION,
        "embedding_model": MODEL,
        "vector_size": 768,
        "query": query,
        "results": [
            {
                "rank": rank,
                "point_id": result.id,
                "score": round(result.score, 4),
                "topic": result.payload["topic"],
                "text": result.payload["text"],
            }
            for rank, result in enumerate(results, start=1)
        ],
    }

    output_path = (
        "artifacts/LAB-RH-02B/"
        "semantic_retrieval_result.json"
    )

    from pathlib import Path

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    print("ARTIFACT_WRITE: PASS")
    print("ARTIFACT_PATH:", output_path)
