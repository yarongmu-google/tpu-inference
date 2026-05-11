#!/bin/bash
# sweep_service.sh — service-sweep entry point.
#
# Usage:
#   tools/tuning/v2/scripts/sweep_service.sh <workload>
#                                            [--timeout SECONDS]
#                                            [--commit-every N]
#
# Delegates to `python3 -m tools.tuning.v2.service.sweep`. The Python
# module drives vllm bench serve via the adapter at
# tools/tuning/v2/service/measurement_bench.py (which shells out to
# tools/benchmark/run_benchmark.sh).
#
# Smoke-only modes:
#   MOCK_BENCH=1               — bypass the real bench subprocess and
#                                emit synthetic deterministic metrics.
#                                Lets the sweep flow run end-to-end
#                                without standing up a vLLM server.
#   KERNEL_TUNER_NO_COMMIT=1   — skip commit + push for the entire run.
#   KERNEL_TUNER_NO_PUSH=1     — commit locally; don't push.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.service.sweep "$@"
