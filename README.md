# RAG Harness bootstrap v0.1

Minimal, local-first bootstrap for LAB-RH-01 and the first vertical slice of the RAG Evidence Harness.

This package deliberately contains infrastructure and repository scaffolding only. It does not yet install the RAG framework, embedding models, Phoenix, Ragas, Redis, or application dependencies. Those will be added through reviewed, pinned dependency sets.

## Included

- CPython 3.12 environment instructions using `uv` without modifying system Python or the Miniconda base environment.
- Qdrant Compose service pinned to `qdrant/qdrant:v1.18.3`.
- Loopback-only Qdrant ports and a Docker named volume suitable for WSL 2.
- Initial source, configuration, test, run, and artifact directories.
- Environment and Qdrant verification scripts.

## Start here

Follow [INSTALL_LOCAL.md](INSTALL_LOCAL.md) in order. Do not run `docker compose down -v`: that command deletes the named Qdrant data volume.

## Ownership boundary

Laboratory may use this package to prove local installation and reproducibility. Harness Build owns the canonical Python implementation, dependency lock, configuration model, experiment runner, evidence capture, and Framework export adapter.
