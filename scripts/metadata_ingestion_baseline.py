"""Metadata-aware ingestion baseline for LAB-RH-03A."""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.ingestion import prepare_corpus

try:
    from scripts.semantic_retrieval_baseline import embed
except ModuleNotFoundError:
    from semantic_retrieval_baseline import embed


QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "metadata_ingestion_baseline"


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


from qdrant_client.models import PointStruct


def index_valid_chunks(client: QdrantClient):
    chunks, diagnostics = prepare_corpus(CONTROLLED_CORPUS)
    points = []

    for point_id, chunk in enumerate(chunks, start=1):
        vector = embed(chunk.text, "search_document")

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.to_qdrant_payload(),
            )
        )

    client.upsert(
        collection_name=COLLECTION,
        points=points,
        wait=True,
    )

    return chunks, diagnostics


def main() -> None:
    client = build_collection()
    chunks, diagnostics = index_valid_chunks(client)

    info = client.get_collection(COLLECTION)

    import json
    from datetime import UTC, datetime
    from pathlib import Path

    records, _ = client.scroll(
        collection_name=COLLECTION,
        limit=10,
        with_payload=True,
        with_vectors=False,
    )

    artifact = {
        "lab_id": "LAB-RH-03A",
        "status": "PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "collection": COLLECTION,
        "valid_chunk_count": len(chunks),
        "rejected_record_count": len(diagnostics),
        "stored_point_count": info.points_count,
        "stored_payloads": [record.payload for record in records],
        "diagnostics": diagnostics,
    }

    output_path = Path(
        "artifacts/LAB-RH-03A/metadata_ingestion_result.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    print("METADATA_INGESTION: PASS")
    print("COLLECTION:", COLLECTION)
    print("VALID_CHUNKS:", len(chunks))
    print("REJECTED_RECORDS:", len(diagnostics))
    print("POINTS_COUNT:", info.points_count)


if __name__ == "__main__":
    main()
