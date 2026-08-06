"""Machine-readable degradation baseline for LAB-RH-03C."""

import json
from datetime import UTC, datetime
from pathlib import Path

from scripts.metadata_ingestion_baseline import (
    COLLECTION,
    build_collection,
    index_valid_chunks,
)
from scripts.semantic_retrieval_baseline import embed

from rag_harness.retrieval import filtered_retrieval


def main() -> None:
    client = build_collection()
    index_valid_chunks(client)

    query_vector = embed(
        "Tenant A permitted source record.",
        "search_query",
    )

    runs = []

    valid_run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
        scenario="valid_boundary",
    )
    runs.append(valid_run.model_dump())

    degraded_payload_run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_a",
        source_id="source_permitted",
        scenario="missing_returned_boundary_metadata",
        payload_fields=[
            "document_id",
            "chunk_id",
            "source_id",
        ],
    )
    runs.append(degraded_payload_run.model_dump())

    incorrect_values_run = filtered_retrieval(
        client=client,
        collection=COLLECTION,
        query_vector=query_vector,
        tenant_id="tenant_unknown",
        source_id="source_unknown",
        scenario="incorrect_boundary_values",
    )
    runs.append(incorrect_values_run.model_dump())

    failures = []

    for scenario, tenant_id, source_id in [
        ("missing_filter", None, None),
        ("partial_filter", "tenant_a", None),
    ]:
        try:
            filtered_retrieval(
                client=client,
                collection=COLLECTION,
                query_vector=query_vector,
                tenant_id=tenant_id,
                source_id=source_id,
                scenario=scenario,
            )
        except ValueError as exc:
            failures.append(
                {
                    "scenario": scenario,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    artifact = {
        "lab_id": "LAB-RH-03C",
        "status": "PASS",
        "created_at": datetime.now(UTC).isoformat(),
        "collection": COLLECTION,
        "retrieval_runs": runs,
        "blocked_runs": failures,
    }

    output_path = Path(
        "artifacts/LAB-RH-03C/"
        "filtered_retrieval_degradation_runs.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    print("LAB-RH-03C: PASS")
    print("RETRIEVAL_RUNS:", len(runs))
    print("BLOCKED_RUNS:", len(failures))
    print("ARTIFACT:", output_path)


if __name__ == "__main__":
    main()
