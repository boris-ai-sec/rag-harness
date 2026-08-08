"""Read-only integrity verification for finalized run-native packages."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from rag_harness.contracts import (
    ExperimentDefinition,
    RunConfiguration,
    RunPackage,
    StrictModel,
)


class VerificationCheck(StrictModel):
    """One deterministic integrity assertion."""

    check_id: str = Field(min_length=1)
    status: Literal["pass", "fail"]
    detail: str = Field(min_length=1)


class RunPackageVerification(StrictModel):
    """Machine-readable result of a read-only package verification."""

    verification_version: Literal["0.1"] = "0.1"
    verification_status: Literal["verified", "failed"]
    package_path: str = Field(min_length=1)
    run_id: str | None = None
    experiment_id: str | None = None
    case_id: str | None = None
    artifact_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    checked_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    read_only: Literal[True] = True
    governed_v0_3_export_claimed: Literal[False] = False
    checks: list[VerificationCheck] = Field(min_length=1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detail(error: Exception) -> str:
    return str(error).replace("\n", " ").strip()[:1000] or type(error).__name__


def _append_check(
    checks: list[VerificationCheck],
    *,
    check_id: str,
    passed: bool,
    passed_detail: str,
    failed_detail: str,
) -> None:
    checks.append(
        VerificationCheck(
            check_id=check_id,
            status="pass" if passed else "fail",
            detail=passed_detail if passed else failed_detail,
        )
    )


def _result(
    *,
    package_path: Path,
    checks: list[VerificationCheck],
    package: RunPackage | None = None,
) -> RunPackageVerification:
    failure_count = sum(check.status == "fail" for check in checks)
    return RunPackageVerification(
        verification_status="failed" if failure_count else "verified",
        package_path=str(package_path.resolve(strict=False)),
        run_id=package.run_id if package else None,
        experiment_id=package.experiment_id if package else None,
        case_id=package.case_id if package else None,
        artifact_count=len(package.artifact_references) if package else 0,
        failure_count=failure_count,
        checks=checks,
    )


def verify_run_package(package_path: Path) -> RunPackageVerification:
    """Verify one package and its artifacts without modifying either."""

    checks: list[VerificationCheck] = []
    package_path = package_path.absolute()
    package_directory = package_path.parent.resolve(strict=False)

    _append_check(
        checks,
        check_id="package.path_name",
        passed=package_path.name == "run_package.json",
        passed_detail="package filename is run_package.json",
        failed_detail="package filename must be run_package.json",
    )
    package_is_regular = package_path.is_file() and not package_path.is_symlink()
    _append_check(
        checks,
        check_id="package.regular_file",
        passed=package_is_regular,
        passed_detail="package is a regular file",
        failed_detail="package is missing, unreadable, or a symbolic link",
    )
    if not package_is_regular:
        return _result(package_path=package_path, checks=checks)

    try:
        raw_package = json.loads(package_path.read_text(encoding="utf-8"))
        package = RunPackage.model_validate(raw_package)
    except (
        json.JSONDecodeError,
        OSError,
        TypeError,
        UnicodeError,
        ValidationError,
    ) as error:
        _append_check(
            checks,
            check_id="package.contract",
            passed=False,
            passed_detail="package contract is valid",
            failed_detail=f"package contract is invalid: {_detail(error)}",
        )
        return _result(package_path=package_path, checks=checks)

    _append_check(
        checks,
        check_id="package.contract",
        passed=True,
        passed_detail="package contract is valid",
        failed_detail="package contract is invalid",
    )
    _append_check(
        checks,
        check_id="package.terminal_status",
        passed=package.status != "initialized",
        passed_detail=f"package has terminal status: {package.status}",
        failed_detail="initialized package is not finalized evidence",
    )
    _append_check(
        checks,
        check_id="package.run_directory",
        passed=package_directory.name == package.run_id,
        passed_detail="run directory name matches run_id",
        failed_detail="run directory name does not match run_id",
    )

    relative_paths = [
        reference.relative_path for reference in package.artifact_references
    ]
    artifact_roles = [
        reference.artifact_role for reference in package.artifact_references
    ]
    _append_check(
        checks,
        check_id="artifacts.unique_paths",
        passed=len(relative_paths) == len(set(relative_paths)),
        passed_detail="artifact reference paths are unique",
        failed_detail="duplicate artifact reference paths detected",
    )
    _append_check(
        checks,
        check_id="artifacts.unique_roles",
        passed=len(artifact_roles) == len(set(artifact_roles)),
        passed_detail="artifact roles are unique",
        failed_detail="duplicate artifact roles detected",
    )

    parsed_artifacts: dict[str, object] = {}
    for index, reference in enumerate(package.artifact_references):
        prefix = f"artifact.{index}.{reference.artifact_role}"
        artifact_path = package_directory / reference.relative_path
        try:
            resolved_artifact = artifact_path.resolve(strict=False)
        except (OSError, RuntimeError):
            contained = False
        else:
            contained = resolved_artifact.is_relative_to(package_directory)
        _append_check(
            checks,
            check_id=f"{prefix}.contained",
            passed=contained,
            passed_detail="artifact resolves inside the run directory",
            failed_detail="artifact resolves outside the run directory",
        )
        _append_check(
            checks,
            check_id=f"{prefix}.not_symlink",
            passed=not artifact_path.is_symlink(),
            passed_detail="artifact is not a symbolic link",
            failed_detail="artifact is a symbolic link",
        )
        exists = artifact_path.is_file()
        _append_check(
            checks,
            check_id=f"{prefix}.exists",
            passed=exists,
            passed_detail="artifact exists as a file",
            failed_detail="artifact is missing or not a file",
        )
        has_digest = reference.sha256 is not None
        _append_check(
            checks,
            check_id=f"{prefix}.digest_present",
            passed=has_digest,
            passed_detail="artifact reference includes SHA-256",
            failed_detail="artifact reference has no SHA-256",
        )
        if not contained or not exists:
            continue

        try:
            actual_digest = _sha256(artifact_path)
        except OSError as error:
            _append_check(
                checks,
                check_id=f"{prefix}.digest_matches",
                passed=False,
                passed_detail="artifact SHA-256 matches",
                failed_detail=f"artifact could not be hashed: {_detail(error)}",
            )
            continue
        _append_check(
            checks,
            check_id=f"{prefix}.digest_matches",
            passed=has_digest and actual_digest == reference.sha256,
            passed_detail="artifact SHA-256 matches",
            failed_detail="artifact SHA-256 does not match",
        )

        if artifact_path.suffix == ".json":
            try:
                parsed_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeError) as error:
                _append_check(
                    checks,
                    check_id=f"{prefix}.json_parseable",
                    passed=False,
                    passed_detail="JSON artifact is parseable",
                    failed_detail=f"JSON artifact is invalid: {_detail(error)}",
                )
            else:
                parsed_artifacts[reference.relative_path] = parsed_artifact
                _append_check(
                    checks,
                    check_id=f"{prefix}.json_parseable",
                    passed=True,
                    passed_detail="JSON artifact is parseable",
                    failed_detail="JSON artifact is invalid",
                )
                if isinstance(parsed_artifact, dict):
                    for identity_field, expected_value in (
                        ("run_id", package.run_id),
                        ("experiment_id", package.experiment_id),
                        ("case_id", package.case_id),
                    ):
                        if identity_field not in parsed_artifact:
                            continue
                        _append_check(
                            checks,
                            check_id=f"{prefix}.{identity_field}_matches",
                            passed=parsed_artifact[identity_field] == expected_value,
                            passed_detail=(
                                f"artifact {identity_field} matches the package"
                            ),
                            failed_detail=(
                                f"artifact {identity_field} does not match the package"
                            ),
                        )

    expected_files = set(relative_paths) | {package_path.name}
    try:
        actual_files = {
            path.relative_to(package_directory).as_posix()
            for path in package_directory.rglob("*")
            if path.is_file() or path.is_symlink()
        }
    except OSError as error:
        unexpected_files = []
        directory_inventory_error = _detail(error)
    else:
        unexpected_files = sorted(actual_files.difference(expected_files))
        directory_inventory_error = None
    _append_check(
        checks,
        check_id="artifacts.no_unreferenced_files",
        passed=not unexpected_files and directory_inventory_error is None,
        passed_detail="run directory contains no unreferenced files",
        failed_detail=(
            f"run directory could not be inventoried: {directory_inventory_error}"
            if directory_inventory_error
            else "unreferenced files detected: " + ", ".join(unexpected_files[:10])
        ),
    )

    configuration_payload = parsed_artifacts.get(package.configuration_ref)
    configuration_valid = False
    if configuration_payload is not None:
        try:
            configuration = RunConfiguration.model_validate(configuration_payload)
        except ValidationError as error:
            configuration_detail = (
                f"configuration contract is invalid: {_detail(error)}"
            )
        else:
            configuration_valid = (
                configuration.experiment_id == package.experiment_id
                and configuration.case_id == package.case_id
            )
            configuration_detail = (
                "configuration identity matches the package"
                if configuration_valid
                else "configuration identity does not match the package"
            )
    else:
        configuration_detail = "configuration_ref is missing or unreadable"
    _append_check(
        checks,
        check_id="identity.configuration_matches",
        passed=configuration_valid,
        passed_detail="configuration identity matches the package",
        failed_detail=configuration_detail,
    )

    definition_paths = [
        reference.relative_path
        for reference in package.artifact_references
        if reference.artifact_role == "experiment_definition"
    ]
    definition_payload = (
        parsed_artifacts.get(definition_paths[0])
        if len(definition_paths) == 1
        else None
    )
    definition_valid = False
    if definition_payload is not None:
        try:
            definition = ExperimentDefinition.model_validate(definition_payload)
        except ValidationError as error:
            definition_detail = f"definition contract is invalid: {_detail(error)}"
        else:
            definition_valid = definition.experiment_id == package.experiment_id
            definition_detail = (
                "experiment definition matches the package"
                if definition_valid
                else "experiment definition does not match the package"
            )
    else:
        definition_detail = "exactly one readable experiment definition is required"
    _append_check(
        checks,
        check_id="identity.definition_matches",
        passed=definition_valid,
        passed_detail="experiment definition matches the package",
        failed_detail=definition_detail,
    )

    return _result(package_path=package_path, checks=checks, package=package)
