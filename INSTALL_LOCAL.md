# Local installation

Target environment: Ubuntu under WSL 2 or a compatible Linux system.

## 1. Enter the project directory

    cd ~/ai-projects/rag-harness

## 2. Keep Conda out of the Harness environment

If a Conda environment is active:

    conda deactivate

Confirm that CONDA_PREFIX is empty:

    printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-}"

## 3. Install uv and Python 3.12

Install uv:

    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
    sh /tmp/uv-install.sh
    export PATH="$HOME/.local/bin:$PATH"

Install Python 3.12:

    uv python install 3.12
    uv python find 3.12

## 4. Create and activate the virtual environment

    uv venv --python 3.12 .venv
    source .venv/bin/activate

Verify the interpreter:

    python --version
    python -c 'import sys; print(sys.executable)'

Expected result:

- Python 3.12.x
- executable path inside ~/ai-projects/rag-harness/.venv/

## 5. Install project dependencies

    uv sync --extra rag --extra telemetry --group dev

This installs:

- qdrant-client
- OpenTelemetry dependencies
- pytest
- the local rag-evidence-harness package

## 6. Start Qdrant

Validate the Compose configuration:

    docker compose config

Start Qdrant:

    docker compose up -d qdrant

Verify readiness:

    docker compose ps
    bash scripts/check_qdrant.sh

Qdrant is exposed locally at:

- REST: http://127.0.0.1:6333
- gRPC: 127.0.0.1:6334
- Dashboard: http://127.0.0.1:6333/dashboard

To stop Qdrant without deleting its data:

    docker compose stop qdrant

Do not run docker compose down -v unless deleting the local Qdrant volume is intentional.

## 7. Install and verify the Ollama embedding model

Install the model in the same environment where the Ollama server runs:

    ollama pull nomic-embed-text:v1.5

Confirm that it is available:

    ollama list

The current baseline expects:

- model: nomic-embed-text:v1.5
- embedding size: 768

## 8. Run the retrieval baselines

Run the synthetic Qdrant baseline:

    python scripts/qdrant_baseline.py

Run the dense semantic retrieval baseline:

    python scripts/semantic_retrieval_baseline.py

The semantic retrieval run should report:

- COLLECTION_BUILD: PASS
- SEMANTIC_SEARCH: PASS
- metadata_filtering as the top-ranked topic
- ARTIFACT_WRITE: PASS

The machine-readable result is written locally to:

    artifacts/LAB-RH-02B/semantic_retrieval_result.json

Generated artifact contents are excluded from Git.

## 9. Run the automated tests

    pytest -q

The current expected test gate is:

    5 passed

## 10. Optional telemetry smoke run

The telemetry smoke script is available at:

    scripts/telemetry_smoke.py

Telemetry export behaviour is represented by two separate fields:

- export_attempted
- export_succeeded

Collector unavailability must be recorded as a limitation and must not be represented as successful export.

## 11. Local data and cleanup

The following remain local and are excluded from Git:

- .venv/
- .env
- runs/
- artifacts/
- backups/
- Qdrant storage data

To stop Qdrant without deleting its volume:

    docker compose stop qdrant
