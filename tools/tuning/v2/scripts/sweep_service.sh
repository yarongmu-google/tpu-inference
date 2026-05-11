#!/bin/bash
# sweep_service.sh — service-sweep entry point.
#
# Usage:
#   tools/tuning/v2/scripts/sweep_service.sh <workload>
#
# Delegates to `python3 -m tools.tuning.v2.service.sweep`. The Python
# module today is a placeholder for the vllm-bench binding; library
# callers can use `run_service_sweep(..., measurement_fn=...)` directly.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.service.sweep "$@"
