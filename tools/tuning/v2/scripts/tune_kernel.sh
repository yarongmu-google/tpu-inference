#!/bin/bash
# tune_kernel.sh — kernel-tune entry point.
#
# Usage:
#   tools/tuning/v2/scripts/tune_kernel.sh <workload> [--iters N] [--warmup N]
#                                          [--commit-every N]
#
# Delegates to `python3 -m tools.tuning.v2.kernel.tune`. The Python
# module drives the rpa_v3 TPU kernel via the adapter at
# tools/tuning/v2/kernel/measurement_tpu.py. Workload env vars
# (MAX_NUM_SEQS, NUM_Q_HEADS, etc.) are pushed into os.environ
# before the v1 RpaV3KernelTuner constructs.
#
# Auto-commit + auto-push are handled inside the Python module via
# tools.tuning.v2.core.git_atomic. Set KERNEL_TUNER_NO_PUSH=1 to
# skip the push step; KERNEL_TUNER_NO_COMMIT=1 to skip commits too
# (useful for smoke runs that shouldn't pollute the branch).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.kernel.tune "$@"
