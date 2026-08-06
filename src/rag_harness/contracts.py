"""Run-native contracts for repeatable Harness experiment execution.

These models are internal Harness objects.  They are deliberately separate
from the governed Framework V0.3 object contracts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
RetrievalMode = Literal["filtered", "unfiltered_control"]
RunStatus = Literal["initialized", "completed", "failed", "blocked"]
ObservabilityStatus = Literal[
    "pending",
    "disabled",
    "connected",
    "unavailable",
    "export_failed",
]


class StrictModel(BaseModel):
    """Base class for run-native contracts with no silent extra fields."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


class ExperimentScenario(StrictModel):
    """One explicitly declared scenario in an experiment configuration."""

    scenario_id: NonEmptyStr
    retrieval_mode: RetrievalMode
    control_only: bool = False

    @model_validator(mode="after")
    def require_explicit_unfiltered_control(self) -> "ExperimentScenario":
        expected_control = self.retrieval_mode == "unfiltered_control"
        if self.control_only != expected_control:
            raise ValueError(
                "control_only must be true only for unfiltered_control scenarios"
            )
        return self


class TelemetryConfiguration(StrictModel):
    """Configuration switch for the optional observability layer."""

    enabled: bool = True
    project_name: NonEmptyStr = "rag-harness"


class RunConfiguration(StrictModel):
    """Effective configuration saved with every Harness experiment run."""

    configuration_version: Literal["0.1"] = "0.1"
    experiment_id: NonEmptyStr
    case_id: NonEmptyStr | None = None
    qdrant_url: AnyHttpUrl = "http://127.0.0.1:6333"
    collection: NonEmptyStr
    embedding_model: NonEmptyStr
    embedding_dimension: int = Field(gt=0)
    distance_metric: Literal["cosine"] = "cosine"
    top_k: int = Field(ge=1, le=100)
    tenant_id: NonEmptyStr
    source_id: NonEmptyStr
    scenarios: list[ExperimentScenario] = Field(min_length=1)
    telemetry: TelemetryConfiguration = Field(
        default_factory=TelemetryConfiguration
    )

    @field_validator("qdrant_url")
    @classmethod
    def require_loopback_qdrant(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if value.host not in {"127.0.0.1", "localhost"}:
            raise ValueError("qdrant_url must use a loopback host")
        return value

    @field_validator("scenarios")
    @classmethod
    def require_unique_scenario_ids(
        cls,
        value: list[ExperimentScenario],
    ) -> list[ExperimentScenario]:
        scenario_ids = [scenario.scenario_id for scenario in value]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario_id values must be unique")
        return value


class ExperimentDefinition(StrictModel):
    """Stable registry record describing one reusable Harness experiment."""

    definition_version: Literal["0.1"] = "0.1"
    experiment_id: NonEmptyStr
    title: NonEmptyStr
    evidence_origin: Literal["synthetic_harness"] = "synthetic_harness"
    verification_scope: Literal["controlled_local_test"] = (
        "controlled_local_test"
    )
    applicability: NonEmptyStr
    client_system_verified: Literal[False] = False
    production_isolation_verified: Literal[False] = False
    mechanism_pattern_only: Literal[True] = True
    required_filter_fields: tuple[Literal["tenant_id", "source_id"], ...]
    allowed_retrieval_modes: tuple[RetrievalMode, ...]
    approved_completion_wording: NonEmptyStr
    mandatory_limitation: NonEmptyStr


class RunArtifactReference(StrictModel):
    """Reference to a file inside one run directory."""

    artifact_role: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_path")
    @classmethod
    def require_safe_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must remain inside the run directory")
        return value


class RunPackage(StrictModel):
    """Authoritative run-native index for one Harness execution."""

    package_version: Literal["0.1"] = "0.1"
    run_id: NonEmptyStr
    run_type: Literal["experiment_run"] = "experiment_run"
    experiment_id: NonEmptyStr
    case_id: NonEmptyStr | None = None
    status: RunStatus = "initialized"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None
    configuration_ref: Literal["run_configuration.json"] = (
        "run_configuration.json"
    )
    artifact_references: list[RunArtifactReference] = Field(
        default_factory=list
    )
    observability_status: ObservabilityStatus = "pending"
    evidence_classification: tuple[
        Literal["run_native_layer_2_evidence"], ...
    ] = ("run_native_layer_2_evidence",)
    governed_v0_3_export_claimed: Literal[False] = False
    client_system_verified: Literal[False] = False
    production_isolation_verified: Literal[False] = False

    @field_validator("run_id")
    @classmethod
    def require_uuid_run_id(cls, value: str) -> str:
        if not value.startswith("run-"):
            raise ValueError("run_id must start with 'run-'")
        try:
            UUID(value.removeprefix("run-"))
        except ValueError as exc:
            raise ValueError("run_id must contain a UUID") from exc
        return value

    @model_validator(mode="after")
    def require_completion_timestamp(self) -> "RunPackage":
        if self.status == "completed" and self.completed_at is None:
            raise ValueError("completed runs require completed_at")
        return self
