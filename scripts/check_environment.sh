#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
import os
import pathlib
import sys

expected = (3, 12)
actual = sys.version_info[:2]
executable = pathlib.Path(sys.executable).resolve()
prefix = pathlib.Path(sys.prefix).resolve()

print(f"Python: {sys.version.split()[0]}")
print(f"Executable: {executable}")
print(f"Prefix: {prefix}")
print(f"CONDA_PREFIX: {os.environ.get('CONDA_PREFIX', '')}")

if actual != expected:
    raise SystemExit(f"FAIL: expected Python 3.12.x, found {actual[0]}.{actual[1]}")
if prefix.name != ".venv":
    raise SystemExit("FAIL: active environment is not the project .venv")
if os.environ.get("CONDA_PREFIX"):
    raise SystemExit("FAIL: a Conda environment is still active")

print("PASS: isolated Harness Python environment is ready")
PY
