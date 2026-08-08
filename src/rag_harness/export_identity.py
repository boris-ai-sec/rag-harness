"""Deterministic identity helpers for governed V0.3 export sets."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ADAPTER_VERSION = "0.1"
TARGET_CONTRACT_VERSION = "Technical Prototype V0.3 / contract pack V0.1"


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_key_components(
    *,
    source_package_hash: str,
    target_case_id: str,
    evidence_request_ref: str,
    adapter_version: str = ADAPTER_VERSION,
    target_contract_version: str = TARGET_CONTRACT_VERSION,
) -> dict[str, str]:
    return {
        "adapter_version": adapter_version,
        "evidence_request_ref": evidence_request_ref,
        "source_package_hash": source_package_hash,
        "target_case_id": target_case_id,
        "target_contract_version": target_contract_version,
    }


def compute_export_key(components: dict[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(components))


def safe_identifier_component(value: str, *, max_length: int = 40) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-")
    if not normalized:
        raise ValueError("identifier component contains no usable characters")
    return normalized[:max_length]


def governed_object_id(prefix: str, case_id: str, export_key: str, suffix: str) -> str:
    return "-".join(
        (
            prefix,
            safe_identifier_component(case_id),
            export_key[:12].upper(),
            suffix,
        )
    )
