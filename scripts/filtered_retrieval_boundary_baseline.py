"""Controlled filtered retrieval baseline for LAB-RH-03B."""

import json
from pathlib import Path

from scripts.metadata_ingestion_baseline import (
    COLLECTION,
    build_collection,
    index_valid_chunks,
)
from scripts.semantic_retrieval_baseline import embed
from rag_harness.retrieval import (
    filtered_retrieval,
    unfiltered_retrieval,
)


ARTIFACT_PATH = Path(
    "artifacts/LAB-RH-03B/filtered_retrieval_runs.json"
)


def main() -> None:
    client = build_collection()
    index_valid_chunks(client)

    permitted_query = embed(
        "Tenant A permitted source record.",
        "search_query",
    )
    other_source_query = embed(
        "Tenant A other source record.",
        "search_query",
    )
    other_tenant_query = embed(
        "Tenant B isolated source record.",
        "search_query",
    )

    runs = [
        filtered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=permitted_query,
            tenant_id="tenant_a",
            source_id="source_permitted",
            scenario="permitted_source",
        ),
        filtered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=other_source_query,
            tenant_id="tenant_a",
            source_id="source_permitted",
            scenario="other_source_same_tenant",
        ),
        filtered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=other_tenant_query,
            tenant_id="tenant_a",
            source_id="source_permitted",
            scenario="other_tenant",
        ),
        unfiltered_retrieval(
            client=client,
            collection=COLLECTION,
            query_vector=other_tenant_query,
            scenario="unfiltered_control",
        ),
    ]

    ARTIFACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "lab_id": "LAB-RH-03B",
        "collection": COLLECTION,
        "runs": [
            run.model_dump(mode="json")
            for run in runs
        ],
    }

    ARTIFACT_PATH.write_text(
        json.dumps(
            artifact,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Artifact written to: {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
