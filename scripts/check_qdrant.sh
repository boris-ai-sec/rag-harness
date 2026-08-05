#!/usr/bin/env bash
set -euo pipefail

endpoint="${QDRANT_URL:-http://127.0.0.1:6333}"
curl --fail --silent --show-error "${endpoint}/readyz"
printf '\nPASS: Qdrant readiness endpoint responded successfully\n'
