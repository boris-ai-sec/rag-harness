"""Canonical registry for reusable Harness experiment definitions."""

from __future__ import annotations

from rag_harness.contracts import ExperimentDefinition, RunConfiguration


EXP_RAG_001 = ExperimentDefinition(
    experiment_id="EXP-RAG-001",
    title="Metadata Filtering / Source Boundary",
    applicability="metadata_filtering_mechanism_only",
    required_filter_fields=("tenant_id", "source_id"),
    allowed_retrieval_modes=("filtered", "unfiltered_control"),
    approved_completion_wording=(
        "EXP-RAG-001 passed in a controlled local synthetic environment. "
        "The experiment demonstrated validation-before-ingestion, "
        "metadata-preserving Qdrant storage, mandatory tenant/source "
        "retrieval filters, negative cross-boundary tests, and explicit "
        "handling of missing or degraded boundary metadata."
    ),
    mandatory_limitation=(
        "This result verifies the implemented metadata-filtering mechanism "
        "only. It does not establish production tenant isolation, "
        "authentication or authorization enforcement, resistance to "
        "malicious filter manipulation, distributed metadata integrity, "
        "production-scale performance, or client-system security."
    ),
)


EXPERIMENT_REGISTRY: dict[str, ExperimentDefinition] = {
    EXP_RAG_001.experiment_id: EXP_RAG_001,
}


def get_experiment(experiment_id: str) -> ExperimentDefinition:
    """Return a stable experiment definition or fail closed."""

    try:
        return EXPERIMENT_REGISTRY[experiment_id]
    except KeyError as exc:
        raise ValueError(f"unknown experiment_id: {experiment_id}") from exc


def validate_experiment_configuration(
    definition: ExperimentDefinition,
    configuration: RunConfiguration,
) -> None:
    """Reject configuration drift from stable experiment identity/policy."""

    if configuration.experiment_id != definition.experiment_id:
        raise ValueError("configuration experiment_id does not match registry")
    configured_modes = {
        scenario.retrieval_mode for scenario in configuration.scenarios
    }
    unexpected_modes = configured_modes.difference(
        definition.allowed_retrieval_modes
    )
    if unexpected_modes:
        unexpected = ", ".join(sorted(unexpected_modes))
        raise ValueError(f"retrieval mode is not allowed: {unexpected}")
