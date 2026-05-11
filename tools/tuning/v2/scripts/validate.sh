#!/bin/bash
# validate.sh — .workload schema validator.
#
# Usage:
#   tools/tuning/v2/scripts/validate.sh <workload>
#
# Exits 0 iff no errors. Warnings (category-mistake vars like
# MAX_NUM_BATCHED_TOKENS in .workload) don't fail; they nudge.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.cli.validate "$@"
