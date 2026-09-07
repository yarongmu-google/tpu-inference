#!/usr/bin/env bash
# Recollect a CDK job through its supported API; retain remote outputs.
set -euo pipefail
[[ $# -eq 1 ]] || { echo "Usage: bash tmp/cleanup.sh JOB_ID" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run.sh" --collect-only "$1"
