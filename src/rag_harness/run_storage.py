"""Safe creation of non-overwriting run-native package directories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

from rag_harness.contracts import (
    ExperimentDefinition,
    RunArtifactReference,
    RunConfiguration,
    RunPackage,
)
from rag_harness.registry import validate_experiment_configuration


def generate_run_id() -> str:
    return f"run-{uuid4()}"


def validate_run_id(run_id: str) -> str:
    if not run_id.startswith("run-"):
        raise ValueError("run_id must start with 'run-'")
    try:
        UUID(run_id.removeprefix("run-"))
    except ValueError as exc:
        raise ValueError("run_id must contain a UUID") from exc
    return run_id


@dataclass(frozen=True)
class RunWorkspace:
    run_id: str
    directory: Path
    configuration_path: Path
    definition_path: Path
    package_path: Path


def _write_json_exclusive(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_workspace(
    runs_root: Path,
    run_id: str | None = None,
) -> RunWorkspace:
    """Create a new run directory and refuse collisions or path traversal."""

    resolved_run_id = validate_run_id(run_id or generate_run_id())
    runs_root.mkdir(parents=True, exist_ok=True)
    run_directory = runs_root / resolved_run_id
    run_directory.mkdir(exist_ok=False)

    return RunWorkspace(
        run_id=resolved_run_id,
        directory=run_directory,
        configuration_path=run_directory / "run_configuration.json",
        definition_path=run_directory / "experiment_definition.json",
        package_path=run_directory / "run_package.json",
    )


def initialize_run_package(
    *,
    runs_root: Path,
    definition: ExperimentDefinition,
    configuration: RunConfiguration,
    run_id: str | None = None,
) -> tuple[RunWorkspace, RunPackage]:
    """Create immutable initial files for a single run-native execution."""

    validate_experiment_configuration(definition, configuration)
    workspace = create_run_workspace(runs_root, run_id)

    _write_json_exclusive(
        workspace.configuration_path,
        configuration.model_dump(mode="json"),
    )
    _write_json_exclusive(
        workspace.definition_path,
        definition.model_dump(mode="json"),
    )

    package = RunPackage(
        run_id=workspace.run_id,
        experiment_id=definition.experiment_id,
        case_id=configuration.case_id,
        artifact_references=[
            RunArtifactReference(
                artifact_role="effective_run_configuration",
                relative_path="run_configuration.json",
                sha256=_sha256(workspace.configuration_path),
            ),
            RunArtifactReference(
                artifact_role="experiment_definition",
                relative_path="experiment_definition.json",
                sha256=_sha256(workspace.definition_path),
            ),
        ],
        observability_status=(
            "pending" if configuration.telemetry.enabled else "disabled"
        ),
    )
    _write_json_exclusive(
        workspace.package_path,
        package.model_dump(mode="json"),
    )
    return workspace, package
