# RAG Evidence Harness

Local-first test harness for producing reproducible evidence about RAG retrieval behaviour, telemetry, and control boundaries.

This repository is an experimental baseline, not a production RAG application. Its purpose is to make retrieval behaviour observable, testable, and reproducible before higher-level evaluation or judgment.

## Current capabilities

- Python 3.12 environment managed with uv
- Qdrant v1.18.3 through Docker Compose
- Qdrant client 1.18.0
- Local dense embeddings through Ollama and nomic-embed-text:v1.5
- Semantic indexing and retrieval
- Synthetic metadata-filtering baseline
- OpenTelemetry export semantics with graceful fallback
- Machine-readable laboratory artifacts
- Automated integration and telemetry tests

Current test gate: 5 passed.

## Architecture

Text documents
→ Ollama embedding model
→ dense vectors
→ Qdrant collection
→ semantic retrieval results
→ payload text and metadata
→ scores, ranking, and JSON evidence artifact

A Qdrant point contains:

- id
- vector
- payload
  - text
  - metadata fields

The current baseline uses dense semantic retrieval. Sparse retrieval, BM25-style search, hybrid fusion, reranking, and production access-control enforcement are not yet implemented.

## Repository structure

- compose.yaml
- scripts/
  - qdrant_baseline.py
  - semantic_retrieval_baseline.py
  - telemetry_smoke.py
- src/rag_harness/
  - telemetry/
- tests/
- artifacts/
- runs/

Generated contents of artifacts/ and runs/ remain local and are excluded from Git.

## Prerequisites

- Ubuntu under WSL 2 or a compatible Linux environment
- Docker with Compose
- Python 3.12
- uv
- Ollama
- Ollama model nomic-embed-text:v1.5

## Quick start

Create and activate the environment:

    uv venv --python 3.12 .venv
    source .venv/bin/activate

Install project dependencies:

    uv sync --extra rag --extra telemetry --group dev

Start Qdrant:

    docker compose up -d qdrant
    bash scripts/check_qdrant.sh

Install the embedding model:

    ollama pull nomic-embed-text:v1.5

Run the semantic retrieval baseline:

    python scripts/semantic_retrieval_baseline.py

Run all tests:

    pytest -q

## Operator CLI

Run the unified `EXP-RAG-001` experiment with its configured case identity:

    rag-harness run \
      --config configs/exp_rag_001.json \
      --runs-root runs

Use `--case-id CASE_ID` to set a client-scoped case identity, or use
`--standalone` to create run-native laboratory evidence without a `case_id`.
Setting a case identity does not itself perform governed export. The two
options are mutually exclusive. Every execution still receives a required
`run_id`; an explicit non-colliding ID can be supplied with `--run-id`.

The command writes one JSON result to stdout containing the `run_id`, terminal
status, experiment result, and authoritative `run_package.json` path. Exit
codes are stable:

- `0`: completed with the expected experiment result (`pass`)
- `1`: completed with an assertion mismatch (`fail`)
- `2`: CLI input, configuration, dependency, or workspace error
- `3`: execution blocked before evaluation
- `4`: execution failed or remained indeterminate

The CLI produces run-native Layer 2 evidence only. It does not claim governed
V0.3 export or client-system verification.

Verify a finalized package without modifying the run directory:

    rag-harness verify \
      --package runs/run-<uuid>/run_package.json

Verification checks the package contract, terminal status, run-directory
identity, artifact containment, symbolic links, referenced-file presence,
SHA-256 digests, JSON identity fields, and unreferenced files. It returns exit
code `0` only when every integrity check passes; integrity failure returns `5`.
The machine-readable result is written to stdout and does not perform governed
export.

## Laboratory status

### LAB-RH-01

Local environment and Qdrant bootstrap baseline.

### LAB-RH-02A

Telemetry baseline with explicit export semantics:

- export_attempted
- export_succeeded

The harness can complete when the telemetry collector is unavailable and records the limitation instead of claiming successful export.

### LAB-RH-02B

Qdrant and dense semantic retrieval baseline using local Ollama embeddings.

The synthetic query:

    How can retrieval be restricted to permitted document attributes?

returns the expected metadata_filtering document as the top-ranked result.

## Evidence boundary

The included runs are controlled synthetic tests. They demonstrate that the local mechanisms operate as implemented.

They do not by themselves establish:

- production readiness
- retrieval quality across a representative corpus
- tenant isolation
- access-control correctness
- robustness against adversarial input
- a Framework-level risk or readiness judgment

## Safety and local operation

Qdrant REST and gRPC ports are exposed only on loopback:

- 127.0.0.1:6333
- 127.0.0.1:6334

Do not run:

    docker compose down -v

unless deletion of the local Qdrant data volume is intentional.

## Installation details

See INSTALL_LOCAL.md.
