"""Loader for the frozen Technical Prototype V0.3 contract baseline."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

V03_CONTRACT_LABEL = "Technical Prototype V0.3 / contract pack V0.1"
LAYER2_OBJECT_TYPES = frozenset(
    {
        "Artifact Manifest",
        "Demonstration Record",
        "Evidence Record",
        "Evidence Package",
    }
)
PROHIBITED_AUTOMATIC_FIELDS = frozenset(
    {
        "client_system_verified",
        "production_isolation_verified",
        "production_readiness",
        "framework_risk_judgment",
        "readiness_decision",
        "client_configuration_verified",
        "client_runtime_enforcement_verified",
        "final_finding",
        "readiness_conclusion",
        "client_recommendation",
        "confirmed_findings",
    }
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing frozen V0.3 file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise TypeError(f"frozen V0.3 file must contain a mapping: {path}")
    return payload


@dataclass(frozen=True)
class FrozenV03Profile:
    root: Path
    envelope_required_fields: frozenset[str]
    object_required_fields: dict[str, frozenset[str]]
    evidence_request_required_fields: frozenset[str]
    allowed_approval_statuses: frozenset[str]
    evidence_methods: frozenset[str]
    evidence_statuses: frozenset[str]
    assess_script: Path

    @classmethod
    def load(cls, root: Path) -> FrozenV03Profile:
        root = root.resolve()
        envelope = _load_yaml(root / "05_Contracts/common_object_envelope.yaml")
        contracts = _load_yaml(root / "05_Contracts/object_contracts.yaml")
        registries = _load_yaml(root / "06_Registries/canonical_registries.yaml")

        if envelope.get("contract_id") != "common_object_envelope":
            raise ValueError("unexpected V0.3 common envelope contract")
        if contracts.get("contract_pack_id") != "technical_prototype_object_contracts":
            raise ValueError("unexpected V0.3 object contract pack")
        if contracts.get("contract_pack_version") != "V0.1":
            raise ValueError("unsupported V0.3 object contract pack version")
        if registries.get("registry_pack_version") != "V0.1":
            raise ValueError("unsupported V0.3 registry pack version")

        objects = contracts.get("objects")
        if not isinstance(objects, dict):
            raise TypeError("V0.3 object contracts are missing")
        missing_objects = LAYER2_OBJECT_TYPES.difference(objects)
        if missing_objects:
            raise ValueError(
                "V0.3 contract pack lacks Layer 2 objects: "
                + ", ".join(sorted(missing_objects))
            )
        if "Evidence Request" not in objects:
            raise ValueError("V0.3 contract pack lacks the Evidence Request object")

        envelope_rules = envelope.get("field_rules", {})
        approvals = (
            envelope_rules.get("approval_status", {}).get("allowed_values", [])
        )
        registry_payload = registries.get("registries", {})
        methods = registry_payload.get("evidence_method", {}).get("values", [])
        statuses = registry_payload.get("evidence_status", {}).get("values", [])
        assess_script = root / "04_Scripts/assess.py"
        if not assess_script.is_file():
            raise ValueError("frozen V0.3 assess.py is missing")

        return cls(
            root=root,
            envelope_required_fields=frozenset(envelope.get("required_fields", [])),
            object_required_fields={
                object_type: frozenset(objects[object_type].get("required_fields", []))
                for object_type in LAYER2_OBJECT_TYPES
            },
            evidence_request_required_fields=frozenset(
                objects.get("Evidence Request", {}).get("required_fields", [])
            ),
            allowed_approval_statuses=frozenset(approvals),
            evidence_methods=frozenset(methods),
            evidence_statuses=frozenset(statuses),
            assess_script=assess_script,
        )

    def case_root(self, case_name: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", case_name):
            raise ValueError("framework case name contains unsupported characters")
        case_root = self.root / "03_Test_Cases" / case_name
        if not case_root.is_dir():
            raise ValueError(f"framework case does not exist: {case_name}")
        return case_root

    def check_case(self, case_name: str) -> str:
        completed = subprocess.run(
            [sys.executable, str(self.assess_script), "check-case", case_name],
            cwd=self.root,
            check=False,
            capture_output=True,
            text=True,
        )
        output = (completed.stdout + completed.stderr).strip()
        if completed.returncode != 0:
            raise ValueError(f"V0.3 check-case failed: {output[:1000]}")
        return output


@dataclass(frozen=True)
class EvidenceRequestReference:
    request_id: str
    case_id: str
    path: Path


def _markdown_sections(text: str) -> dict[str, str]:
    matches = list(re.finditer(r"^##\s+(.+?)\s*$", text, re.MULTILINE))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections[match.group(1).strip().casefold()] = text[match.end() : end].strip()
    return sections


def _resolved_section(sections: dict[str, str], heading: str) -> str:
    value = sections.get(heading.casefold(), "").strip().strip("`")
    normalized = re.sub(r"\s+", " ", value).strip().casefold()
    if not value or normalized in {"tbd", "- tbd"}:
        raise ValueError(f"Evidence Request has no resolved value for: {heading}")
    return value


def load_evidence_request(
    *,
    profile: FrozenV03Profile,
    case_name: str,
    path: Path,
) -> EvidenceRequestReference:
    case_root = profile.case_root(case_name).resolve()
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError("Evidence Request must be a regular non-symlink file")
    if not path.is_relative_to(case_root):
        raise ValueError("Evidence Request must remain inside the target V0.3 case")
    if path.suffix.lower() == ".md":
        text = path.read_text(encoding="utf-8-sig")
        sections = _markdown_sections(text)
        heading_by_contract_field = {
            "request_id": "Request ID",
            "request_version": "Request Version",
            "assessment_surface": "Assessment Surface",
            "evidence_need": "Evidence Need",
            "target_component": "Target System / Component",
            "preferred_evidence_type": "Requested Evidence Type",
            "available_artifacts_and_tooling": "Available Artifacts and Tooling",
            "scope_boundaries": "Scope Boundaries",
            "expected_output": "Expected Output",
            "constraints": "Constraints",
            "status": "Request Status",
        }
        missing_contract_fields = sorted(
            profile.evidence_request_required_fields.difference(
                heading_by_contract_field
            )
        )
        if missing_contract_fields:
            raise ValueError(
                "unsupported frozen Evidence Request fields: "
                + ", ".join(missing_contract_fields)
            )
        resolved = {
            field: _resolved_section(sections, heading_by_contract_field[field])
            for field in profile.evidence_request_required_fields
        }
        request_id = resolved["request_id"].splitlines()[0].strip().strip("`")
        case_id = _resolved_section(sections, "Case ID").splitlines()[0].strip()
    elif path.suffix.lower() in {".yaml", ".yml"}:
        payload = _load_yaml(path)
        request_id = str(payload.get("request_id", "")).strip()
        case_id = str(payload.get("case_id", "")).strip()
        missing = sorted(
            field
            for field in profile.evidence_request_required_fields
            if payload.get(field) in (None, "", "TBD", [])
        )
        if not request_id or not case_id or missing:
            raise ValueError(
                "YAML Evidence Request is incomplete: "
                + ", ".join([*missing, "case_id"] if not case_id else missing)
            )
    else:
        raise ValueError("Evidence Request must be Markdown or YAML")
    return EvidenceRequestReference(
        request_id=request_id,
        case_id=case_id,
        path=path,
    )
