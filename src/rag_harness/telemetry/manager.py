"""Optional OTLP telemetry with file-sidecar fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


@dataclass(frozen=True)
class TelemetrySettings:
    enabled: bool = True
    ui_endpoint: str = "http://127.0.0.1:6006"
    traces_endpoint: str = "http://127.0.0.1:6006/v1/traces"
    project_name: str = "rag-harness"
    probe_timeout_seconds: float = 0.75
    export_timeout_seconds: float = 2.0


class TelemetrySession:
    """Creates spans only when the local collector is reachable.

    Export failure is an observability limitation, never an experiment exception.
    """

    def __init__(self, settings: TelemetrySettings, run_id: str, sidecar_path: Path):
        self.settings = settings
        self.run_id = run_id
        self.sidecar_path = sidecar_path
        self.provider: TracerProvider | None = None
        self.status = "disabled"
        self.error_type: str | None = None
        self.tracer = trace.get_tracer("rag_harness.telemetry")

    def start(self) -> "TelemetrySession":
        if not self.settings.enabled:
            self.status = "disabled"
            self._write_sidecar()
            return self

        try:
            with urlopen(self.settings.ui_endpoint, timeout=self.settings.probe_timeout_seconds):
                pass
        except (OSError, URLError, TimeoutError):
            self.status = "unavailable"
            self.error_type = "collector_unreachable"
            self._write_sidecar()
            return self

        resource = Resource.create(
            {
                "service.name": "rag-evidence-harness",
                "openinference.project.name": self.settings.project_name,
            }
        )
        self.provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(
            endpoint=self.settings.traces_endpoint,
            timeout=self.settings.export_timeout_seconds,
        )
        self.provider.add_span_processor(BatchSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer("rag_harness.telemetry")
        self.status = "connected"
        self._write_sidecar()
        return self

    def finish(self) -> bool | None:
        export_attempted = self.provider is not None
        export_succeeded: bool | None = None

        if self.provider is not None:
            export_succeeded = bool(self.provider.force_flush(timeout_millis=5000))
            self.provider.shutdown()
            if not export_succeeded:
                self.status = "export_failed"
                self.error_type = "otlp_export_failed"

        self._write_sidecar(
            export_attempted=export_attempted,
            export_succeeded=export_succeeded,
        )
        return export_succeeded

    def attributes(self, values: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in values.items() if value is not None}

    def _write_sidecar(
        self,
        export_attempted: bool = False,
        export_succeeded: bool | None = None,
    ) -> None:
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": self.run_id,
            "observability_backend": "phoenix_otlp_http",
            "observability_status": self.status,
            "error_type": self.error_type,
            "export_attempted": export_attempted,
            "export_succeeded": export_succeeded,
            "recorded_at": datetime.now(UTC).isoformat(),
            "settings": {
                **asdict(self.settings),
                "ui_endpoint": "http://127.0.0.1:6006",
                "traces_endpoint": "http://127.0.0.1:6006/v1/traces",
            },
        }
        self.sidecar_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
