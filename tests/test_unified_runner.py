import json
from pathlib import Path

from rag_harness.contracts import ExperimentScenario, RunConfiguration
from rag_harness.corpus import CONTROLLED_CORPUS
from rag_harness.models import Chunk
from rag_harness.registry import EXP_RAG_001
from rag_harness.retrieval import RetrievalRun
from rag_harness.runner import execute_experiment

CONFIG_PATH = Path("configs/exp_rag_001.json")
FIXED_RUN_ID = "run-2742eadb-2d12-4d6a-bbe7-9026485c20f1"


def load_configuration() -> RunConfiguration:
    return RunConfiguration.model_validate_json(CONFIG_PATH.read_text(encoding="utf-8"))


class FakeEmbeddings:
    def __init__(
        self,
        *,
        preflight_error: Exception | None = None,
        document_error: Exception | None = None,
    ) -> None:
        self.preflight_error = preflight_error
        self.document_error = document_error

    def preflight(self) -> None:
        if self.preflight_error:
            raise self.preflight_error

    def embed_documents(self, chunks: list[Chunk]) -> list[list[float]]:
        if self.document_error:
            raise self.document_error
        return [[float(index), 0.0, 0.0] for index, _ in enumerate(chunks)]

    def embed_query(self, query_text: str) -> list[float]:
        return [float(len(query_text)), 0.0, 0.0]


class FakeVectorStore:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.mismatch = mismatch
        self.indexed_chunks: list[Chunk] = []
        self.collection: str | None = None

    def preflight(self) -> None:
        return None

    def prepare_collection(
        self,
        *,
        base_name: str,
        run_id: str,
        vector_size: int,
    ) -> str:
        self.collection = f"{base_name}_{run_id.removeprefix('run-')}"
        return self.collection

    def index_chunks(
        self,
        *,
        collection: str,
        run_id: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        assert len(chunks) == len(vectors)
        self.indexed_chunks = chunks

    def retrieve(
        self,
        *,
        collection: str,
        run_id: str,
        scenario: ExperimentScenario,
        query_vector: list[float],
        limit: int,
    ) -> RetrievalRun:
        boundary_status = (
            "indeterminate"
            if scenario.retrieval_mode == "unfiltered_control"
            else "passed"
        )
        if self.mismatch and scenario.scenario_id == "permitted_source":
            boundary_status = "failed"
        return RetrievalRun(
            run_id=run_id,
            collection=collection,
            scenario=scenario.scenario_id,
            retrieval_mode=scenario.retrieval_mode,
            filter_parameters={
                key: value
                for key, value in scenario.filter_parameters.model_dump().items()
                if value is not None
            },
            retrieval_outcome="records_returned",
            boundary_verification_status=boundary_status,
            verification_reason="controlled fake result",
            returned_chunk_ids=["doc_a_001-chunk-0001"],
            payload_references=[],
        )


def test_unified_runner_completes_one_authoritative_package(
    tmp_path: Path,
) -> None:
    vector_store = FakeVectorStore()

    workspace, package = execute_experiment(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=load_configuration(),
        corpus=CONTROLLED_CORPUS,
        embeddings=FakeEmbeddings(),
        vector_store=vector_store,
        run_id=FIXED_RUN_ID,
    )

    assert package.status == "completed"
    assert package.experiment_result == "pass"
    assert package.observability_status == "disabled"
    assert len(vector_store.indexed_chunks) == 3
    assert workspace.directory.is_dir()
    assert (workspace.directory / "ingestion_result.json").is_file()
    assert (workspace.directory / "retrieval_results.json").is_file()
    assert all(reference.sha256 for reference in package.artifact_references)

    retrieval_artifact = json.loads(
        (workspace.directory / "retrieval_results.json").read_text()
    )
    assert len(retrieval_artifact["scenarios"]) == 4
    assert {
        item["retrieval"]["run_id"] for item in retrieval_artifact["scenarios"]
    } == {FIXED_RUN_ID}
    assert retrieval_artifact["governed_v0_3_export_claimed"] is False


def test_preflight_failure_blocks_without_evaluating_experiment(
    tmp_path: Path,
) -> None:
    workspace, package = execute_experiment(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=load_configuration(),
        corpus=CONTROLLED_CORPUS,
        embeddings=FakeEmbeddings(
            preflight_error=ConnectionError("Ollama unavailable")
        ),
        vector_store=FakeVectorStore(),
    )

    assert package.status == "blocked"
    assert package.experiment_result == "not_evaluated"
    assert package.errors[0].stage == "service_preflight"
    assert (workspace.directory / "run_error.json").is_file()


def test_runner_blocks_when_unimplemented_telemetry_is_enabled(
    tmp_path: Path,
) -> None:
    configuration = load_configuration().model_copy(
        update={
            "telemetry": load_configuration().telemetry.model_copy(
                update={"enabled": True}
            )
        }
    )

    _, package = execute_experiment(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=configuration,
        corpus=CONTROLLED_CORPUS,
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(),
    )

    assert package.status == "blocked"
    assert package.experiment_result == "not_evaluated"
    assert package.observability_status == "unavailable"
    assert "not integrated" in package.errors[0].message


def test_runtime_failure_is_indeterminate_not_blocked(tmp_path: Path) -> None:
    _, package = execute_experiment(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=load_configuration(),
        corpus=CONTROLLED_CORPUS,
        embeddings=FakeEmbeddings(document_error=RuntimeError("embedding failed")),
        vector_store=FakeVectorStore(),
    )

    assert package.status == "failed"
    assert package.experiment_result == "indeterminate"
    assert package.errors[0].stage == "embed_documents"


def test_expected_result_mismatch_completes_with_fail(tmp_path: Path) -> None:
    _, package = execute_experiment(
        runs_root=tmp_path / "runs",
        definition=EXP_RAG_001,
        configuration=load_configuration(),
        corpus=CONTROLLED_CORPUS,
        embeddings=FakeEmbeddings(),
        vector_store=FakeVectorStore(mismatch=True),
    )

    assert package.status == "completed"
    assert package.experiment_result == "fail"
