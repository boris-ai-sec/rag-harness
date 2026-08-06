"""Run controlled corpus validation baseline."""

import json

from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.ingestion import prepare_corpus


def main() -> None:
    chunks, diagnostics = prepare_corpus(CONTROLLED_CORPUS)

    result = {
        "status": "pass",
        "valid_chunk_count": len(chunks),
        "rejected_record_count": len(diagnostics),
        "chunk_payloads": [
            chunk.to_qdrant_payload()
            for chunk in chunks
        ],
        "diagnostics": diagnostics,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
