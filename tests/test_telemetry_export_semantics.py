import json
from pathlib import Path

from rag_harness.telemetry.manager import TelemetrySession, TelemetrySettings


class FakeProvider:
    def __init__(self, flush_result: bool):
        self.flush_result = flush_result
        self.shutdown_called = False

    def force_flush(self, timeout_millis: int) -> bool:
        return self.flush_result

    def shutdown(self) -> None:
        self.shutdown_called = True


def read_sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_finish_without_provider_records_not_attempted(tmp_path: Path) -> None:
    sidecar = tmp_path / "trace_sidecar.json"
    session = TelemetrySession(
        TelemetrySettings(),
        "run-test-unavailable",
        sidecar,
    )
    session.status = "unavailable"
    session.error_type = "collector_unreachable"

    result = session.finish()
    payload = read_sidecar(sidecar)

    assert result is None
    assert payload["export_attempted"] is False
    assert payload["export_succeeded"] is None


def test_finish_with_successful_flush_records_success(tmp_path: Path) -> None:
    sidecar = tmp_path / "trace_sidecar.json"
    session = TelemetrySession(
        TelemetrySettings(),
        "run-test-success",
        sidecar,
    )
    provider = FakeProvider(True)
    session.provider = provider
    session.status = "connected"

    result = session.finish()
    payload = read_sidecar(sidecar)

    assert result is True
    assert provider.shutdown_called is True
    assert payload["export_attempted"] is True
    assert payload["export_succeeded"] is True


def test_finish_with_failed_flush_records_failure(tmp_path: Path) -> None:
    sidecar = tmp_path / "trace_sidecar.json"
    session = TelemetrySession(
        TelemetrySettings(),
        "run-test-failure",
        sidecar,
    )
    provider = FakeProvider(False)
    session.provider = provider
    session.status = "connected"

    result = session.finish()
    payload = read_sidecar(sidecar)

    assert result is False
    assert provider.shutdown_called is True
    assert payload["export_attempted"] is True
    assert payload["export_succeeded"] is False
    assert payload["observability_status"] == "export_failed"
    assert payload["error_type"] == "otlp_export_failed"
