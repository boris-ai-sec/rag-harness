#!/usr/bin/env python3
"""Create a safe OTLP diagnostic trace and authoritative file package."""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import uuid

from rag_harness.telemetry import TelemetrySession, TelemetrySettings


def main() -> int:
    run_id = f"run-{uuid.uuid4()}"
    run_dir = Path("runs") / run_id
    sidecar = Path("artifacts") / "telemetry" / run_id / "trace_sidecar.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    package = {
        "diagnostic_id": "DIAG-OTEL-001",
        "run_type": "telemetry_diagnostic",
        "run_id": run_id,
        "case_id": None,
        "evidence_origin": "synthetic_harness",
        "verification_scope": "controlled_local_test",
        "applicability": "telemetry_transport_diagnostic_only",
        "client_system_verified": False,
        "config_revision": "LAB-RH-02A-v0.2",
        "code_revision": "uncommitted-bootstrap",
        "status": "completed",
        "experiment_evidence": False,
        "exp_rag_001_evidence": False,
        "qdrant_retrieval_executed": False,
        "metadata_filtering_executed": False,
        "source_boundary_validation_executed": False,
        "observability_limitation": None,
        "created_at": datetime.now(UTC).isoformat(),
    }

    telemetry = TelemetrySession(TelemetrySettings(), run_id, sidecar).start()
    attrs = {
        "rag_harness.diagnostic_id": package["diagnostic_id"],
        "rag_harness.run_type": package["run_type"],
        "rag_harness.run_id": run_id,
        "rag_harness.evidence_origin": package["evidence_origin"],
        "rag_harness.verification_scope": package["verification_scope"],
        "rag_harness.applicability": package["applicability"],
        "rag_harness.client_system_verified": False,
        "rag_harness.config_revision": package["config_revision"],
        "rag_harness.code_revision": package["code_revision"],
        "rag_harness.status": "completed",
        "openinference.span.kind": "CHAIN",
    }

    with telemetry.tracer.start_as_current_span("telemetry_diagnostic_run", attributes=attrs):
        with telemetry.tracer.start_as_current_span(
            "emit_diagnostic_child",
            attributes={
                "rag_harness.diagnostic_step": "parent_child_propagation",
                "openinference.span.kind": "CHAIN",
            },
        ):
            pass
        with telemetry.tracer.start_as_current_span(
            "finalize_diagnostic",
            attributes={
                "rag_harness.run_package_ref": f"runs/{run_id}/run_package.json",
                "rag_harness.status": "completed",
                "openinference.span.kind": "CHAIN",
            },
        ):
            pass

    export_succeeded = telemetry.finish()
    if telemetry.status != "connected" or export_succeeded is False:
        package["observability_limitation"] = telemetry.error_type or telemetry.status

    package_path = run_dir / "run_package.json"
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_id": run_id, "run_package": str(package_path), "trace_sidecar": str(sidecar), "telemetry": telemetry.status}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
