import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_harness import cli
from rag_harness.contracts import RunPackage
from rag_harness.verification import RunPackageVerification, VerificationCheck

CONFIG_PATH = Path("configs/exp_rag_001.json")
FIXED_RUN_ID = "run-2742eadb-2d12-4d6a-bbe7-9026485c20f1"


def _package(*, status: str, result: str, case_id: str | None) -> RunPackage:
    return RunPackage(
        run_id=FIXED_RUN_ID,
        experiment_id="EXP-RAG-001",
        case_id=case_id,
        status=status,
        experiment_result=result,
        completed_at=datetime.now(UTC),
        observability_status="disabled",
    )


def _install_fake_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = "completed",
    result: str = "pass",
) -> list:
    captured_configurations = []
    monkeypatch.setattr(
        cli, "_build_services", lambda configuration: (object(), object())
    )

    def fake_execute_experiment(**kwargs):
        configuration = kwargs["configuration"]
        captured_configurations.append(configuration)
        package_path = kwargs["runs_root"] / FIXED_RUN_ID / "run_package.json"
        return (
            SimpleNamespace(package_path=package_path),
            _package(
                status=status,
                result=result,
                case_id=configuration.case_id,
            ),
        )

    monkeypatch.setattr(cli, "execute_experiment", fake_execute_experiment)
    return captured_configurations


def test_cli_reports_pass_and_authoritative_package_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_fake_execution(monkeypatch)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(CONFIG_PATH),
            "--runs-root",
            str(tmp_path / "runs"),
            "--run-id",
            FIXED_RUN_ID,
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PASS
    assert output["exit_code"] == cli.EXIT_PASS
    assert output["run_id"] == FIXED_RUN_ID
    assert output["status"] == "completed"
    assert output["experiment_result"] == "pass"
    assert output["run_package_path"].endswith(f"{FIXED_RUN_ID}/run_package.json")
    assert output["governed_v0_3_export_claimed"] is False


def test_cli_case_id_override_is_validated_and_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configurations = _install_fake_execution(monkeypatch)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(CONFIG_PATH),
            "--runs-root",
            str(tmp_path),
            "--case-id",
            "case_client_001",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PASS
    assert configurations[0].case_id == "case_client_001"
    assert output["case_id"] == "case_client_001"


def test_cli_standalone_clears_case_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    configurations = _install_fake_execution(monkeypatch)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(CONFIG_PATH),
            "--runs-root",
            str(tmp_path),
            "--standalone",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PASS
    assert configurations[0].case_id is None
    assert output["case_id"] is None


@pytest.mark.parametrize(
    ("status", "result", "expected_exit_code"),
    [
        ("completed", "fail", cli.EXIT_EXPERIMENT_FAIL),
        ("blocked", "not_evaluated", cli.EXIT_BLOCKED),
        ("failed", "indeterminate", cli.EXIT_EXECUTION_FAILED),
    ],
)
def test_cli_maps_terminal_run_state_to_stable_exit_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    status: str,
    result: str,
    expected_exit_code: int,
) -> None:
    _install_fake_execution(monkeypatch, status=status, result=result)

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(CONFIG_PATH),
            "--runs-root",
            str(tmp_path),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert exit_code == expected_exit_code
    assert output["exit_code"] == expected_exit_code
    assert output["status"] == status
    assert output["experiment_result"] == result


def test_cli_returns_input_error_without_starting_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    invalid_config = tmp_path / "invalid.json"
    invalid_config.write_text('{"experiment_id":', encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "execute_experiment",
        lambda **kwargs: pytest.fail("execution must not start"),
    )

    exit_code = cli.main(
        [
            "run",
            "--config",
            str(invalid_config),
            "--runs-root",
            str(tmp_path / "runs"),
        ]
    )

    captured = capsys.readouterr()
    error = json.loads(captured.err)
    assert captured.out == ""
    assert exit_code == cli.EXIT_INPUT_ERROR
    assert error["status"] == "input_error"
    assert error["exit_code"] == cli.EXIT_INPUT_ERROR


def test_cli_rejects_case_id_and_standalone_together() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            [
                "run",
                "--config",
                str(CONFIG_PATH),
                "--case-id",
                "case_client_001",
                "--standalone",
            ]
        )

    assert error.value.code == cli.EXIT_INPUT_ERROR


def test_cli_verify_reports_verified_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_path = tmp_path / FIXED_RUN_ID / "run_package.json"
    monkeypatch.setattr(
        cli,
        "verify_run_package",
        lambda path: RunPackageVerification(
            verification_status="verified",
            package_path=str(path),
            run_id=FIXED_RUN_ID,
            experiment_id="EXP-RAG-001",
            artifact_count=3,
            failure_count=0,
            checks=[
                VerificationCheck(
                    check_id="package.contract",
                    status="pass",
                    detail="package contract is valid",
                )
            ],
        ),
    )

    exit_code = cli.main(["verify", "--package", str(package_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_PASS
    assert output["exit_code"] == cli.EXIT_PASS
    assert output["verification_status"] == "verified"
    assert output["read_only"] is True
    assert output["governed_v0_3_export_claimed"] is False


def test_cli_verify_maps_integrity_failure_to_exit_code_5(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    package_path = tmp_path / FIXED_RUN_ID / "run_package.json"
    monkeypatch.setattr(
        cli,
        "verify_run_package",
        lambda path: RunPackageVerification(
            verification_status="failed",
            package_path=str(path),
            artifact_count=0,
            failure_count=1,
            checks=[
                VerificationCheck(
                    check_id="package.regular_file",
                    status="fail",
                    detail="package is missing",
                )
            ],
        ),
    )

    exit_code = cli.main(["verify", "--package", str(package_path)])

    output = json.loads(capsys.readouterr().out)
    assert exit_code == cli.EXIT_VERIFICATION_FAILED
    assert output["exit_code"] == cli.EXIT_VERIFICATION_FAILED
    assert output["verification_status"] == "failed"
