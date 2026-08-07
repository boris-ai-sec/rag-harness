"""Unified execution service for controlled Harness experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from rag_harness.contracts import (
    ExperimentDefinition,
    ExperimentScenario,
    RunConfiguration,
    RunError,
    RunPackage,
)
from rag_harness.ingestion import prepare_corpus
from rag_harness.models import Chunk
from rag_harness.retrieval import RetrievalRun, validate_boundary_filter
from rag_harness.run_storage import (
    RunWorkspace,
    finalize_run_package,
    initialize_run_package,
    write_json_artifact,
)


class EmbeddingService(Protocol):
    def preflight(self) -> None: ...

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]: ...

    def embed_query(self, query_text: str) -> list[float]: ...


class VectorStore(Protocol):
    def preflight(self) -> None: ...

    def prepare_collection(
        self,
        *,
        base_name: str,
        run_id: str,
        vector_size: int,
    ) -> str: ...

    def index_chunks(
        self,
        *,
        collection: str,
        run_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None: ...

    def retrieve(
        self,
        *,
        collection: str,
        run_id: str,
        scenario: ExperimentScenario,
        query_vector: list[float],
        limit: int,
    ) -> RetrievalRun: ...


def _sanitized_error(stage: str, error: Exception) -> RunError:
    message = str(error).replace("\n", " ").strip() or "execution error"
    return RunError(
        stage=stage,
        error_type=type(error).__name__,
        message=message[:1000],
    )


def _scenario_assertions(
    scenario: ExperimentScenario,
    retrieval: RetrievalRun,
    expected_run_id: str,
) -> dict[str, bool]:
    return {
        "retrieval_outcome_matches": (
            retrieval.retrieval_outcome == scenario.expected.retrieval_outcome
        ),
        "boundary_status_matches": (
            retrieval.boundary_verification_status == scenario.expected.boundary_status
        ),
        "unified_run_id_preserved": retrieval.run_id == expected_run_id,
    }


def _execute_scenario(
    *,
    scenario: ExperimentScenario,
    workspace: RunWorkspace,
    configuration: RunConfiguration,
    embeddings: EmbeddingService,
    vector_store: VectorStore,
    collection: str,
) -> dict[str, Any]:
    if scenario.expected.execution == "blocked_by_validation":
        try:
            validate_boundary_filter(
                tenant_id=scenario.filter_parameters.tenant_id,
                source_id=scenario.filter_parameters.source_id,
            )
        except ValueError as error:
            return {
                "scenario_id": scenario.scenario_id,
                "execution": "blocked_by_validation",
                "expected_block": True,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "assertions": {"expected_execution_matches": True},
            }
        return {
            "scenario_id": scenario.scenario_id,
            "execution": "validation_unexpectedly_passed",
            "expected_block": True,
            "assertions": {"expected_execution_matches": False},
        }

    query_vector = embeddings.embed_query(scenario.query_text)
    retrieval = vector_store.retrieve(
        collection=collection,
        run_id=workspace.run_id,
        scenario=scenario,
        query_vector=query_vector,
        limit=configuration.top_k,
    )
    assertions = _scenario_assertions(
        scenario,
        retrieval,
        workspace.run_id,
    )
    return {
        "scenario_id": scenario.scenario_id,
        "execution": "retrieval_executed",
        "expected": scenario.expected.model_dump(mode="json"),
        "retrieval": retrieval.model_dump(mode="json"),
        "assertions": assertions,
    }


def _all_assertions_pass(results: list[dict[str, Any]]) -> bool:
    return all(
        assertion for result in results for assertion in result["assertions"].values()
    )


def execute_experiment(
    *,
    runs_root: Path,
    definition: ExperimentDefinition,
    configuration: RunConfiguration,
    corpus: list[dict[str, Any]],
    embeddings: EmbeddingService,
    vector_store: VectorStore,
    run_id: str | None = None,
) -> tuple[RunWorkspace, RunPackage]:
    """Execute one configured experiment into one authoritative run package."""

    workspace, package = initialize_run_package(
        runs_root=runs_root,
        definition=definition,
        configuration=configuration,
        run_id=run_id,
    )
    artifact_paths: dict[str, Path] = {}
    stage = "service_preflight"

    try:
        if configuration.telemetry.enabled:
            raise RuntimeError(
                "telemetry is not integrated in the Stage 2 runner slice"
            )
        embeddings.preflight()
        vector_store.preflight()

        stage = "validate_corpus"
        chunks, diagnostics = prepare_corpus(corpus)
        if not chunks:
            raise ValueError("controlled corpus produced no valid chunks")

        stage = "embed_documents"
        vectors = embeddings.embed_documents(chunks)

        stage = "prepare_collection"
        collection = vector_store.prepare_collection(
            base_name=configuration.collection,
            run_id=workspace.run_id,
            vector_size=configuration.embedding_dimension,
        )

        stage = "ingest_chunks"
        vector_store.index_chunks(
            collection=collection,
            run_id=workspace.run_id,
            chunks=chunks,
            vectors=vectors,
        )
        ingestion_path = write_json_artifact(
            workspace.directory / "ingestion_result.json",
            {
                "run_id": workspace.run_id,
                "experiment_id": configuration.experiment_id,
                "collection": collection,
                "valid_chunk_count": len(chunks),
                "rejected_record_count": len(diagnostics),
                "accepted_chunk_ids": [chunk.metadata.chunk_id for chunk in chunks],
                "diagnostics": diagnostics,
            },
        )
        artifact_paths["ingestion_result"] = ingestion_path

        scenario_results = []
        for scenario in configuration.scenarios:
            stage = f"execute_scenario:{scenario.scenario_id}"
            scenario_results.append(
                _execute_scenario(
                    scenario=scenario,
                    workspace=workspace,
                    configuration=configuration,
                    embeddings=embeddings,
                    vector_store=vector_store,
                    collection=collection,
                )
            )
        result = "pass" if _all_assertions_pass(scenario_results) else "fail"
        retrieval_path = write_json_artifact(
            workspace.directory / "retrieval_results.json",
            {
                "run_id": workspace.run_id,
                "experiment_id": configuration.experiment_id,
                "collection": collection,
                "experiment_result": result,
                "scenarios": scenario_results,
                "governed_v0_3_export_claimed": False,
            },
        )
        artifact_paths["retrieval_results"] = retrieval_path

        return workspace, finalize_run_package(
            workspace=workspace,
            package=package,
            status="completed",
            experiment_result=result,
            artifact_paths=artifact_paths,
        )
    except Exception as error:  # noqa: BLE001 - execution boundary records failure
        run_error = _sanitized_error(stage, error)
        error_path = write_json_artifact(
            workspace.directory / "run_error.json",
            run_error.model_dump(mode="json"),
        )
        artifact_paths["run_error"] = error_path
        preflight_blocked = stage == "service_preflight"
        finalized = finalize_run_package(
            workspace=workspace,
            package=package,
            status="blocked" if preflight_blocked else "failed",
            experiment_result=(
                "not_evaluated" if preflight_blocked else "indeterminate"
            ),
            artifact_paths=artifact_paths,
            errors=[run_error],
            observability_status=(
                "unavailable"
                if configuration.telemetry.enabled
                else package.observability_status
            ),
        )
        return workspace, finalized
