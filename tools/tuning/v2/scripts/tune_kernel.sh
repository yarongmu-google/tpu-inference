#!/bin/bash
# tune_kernel.sh — kernel-tune entry point.
#
# Usage:
#   tools/tuning/v2/scripts/tune_kernel.sh <workload>
#
# Delegates to `python3 -m tools.tuning.v2.kernel.tune`. The Python
# module today is a placeholder for the TPU measurement binding;
# library callers can already use `run_kernel_tune(..., measurement_fn=...)`
# directly. Once the pallas_call binding lands, this wrapper picks up
# the change automatically.
#
# Auto-commit + auto-push are handled inside the Python module via
# tools.tuning.v2.core.git_atomic. Set KERNEL_TUNER_NO_PUSH=1 to skip
# the push step.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.kernel.tune "$@"
