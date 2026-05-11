#!/bin/bash
# run_pipeline.sh — orchestrator chaining the five v2 primitives.
#
# Usage:
#   tools/tuning/v2/scripts/run_pipeline.sh <workload>
#
# Flow:
#   1. validate.sh          (fail fast on bad workload)
#   2. tune_kernel.sh       (kernel-tune long run with periodic commits)
#   3. project_kernel.sh    (winners → <workload>.kernel)
#   4. sweep_service.sh     (service-sweep long run)
#   5. project_service.sh   (per-objective winners → <workload>.service)
#   6. aggregate.sh         (per-model production.{kernel,service})
#
# Each step inherits auto-commit + auto-push (KERNEL_TUNER_NO_PUSH=1
# to disable). On any step failure, set -e aborts the orchestrator;
# raw stores already on disk plus partial git commits from earlier
# steps survive — restart re-uses the raw stores via skip-set
# (resume semantics).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload>" >&2
    exit 1
fi

WORKLOAD="$1"
WORKLOAD_DIR=$(dirname "$WORKLOAD")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"

echo "=== Tune-v2 pipeline: $WORKLOAD ==="

echo "--- Step 1/6: validate ---"
"$SCRIPT_DIR/validate.sh" "$WORKLOAD"

echo "--- Step 2/6: tune kernel ---"
"$SCRIPT_DIR/tune_kernel.sh" "$WORKLOAD"

echo "--- Step 3/6: project kernel ---"
"$SCRIPT_DIR/project_kernel.sh" "$WORKLOAD"

echo "--- Step 4/6: sweep service ---"
"$SCRIPT_DIR/sweep_service.sh" "$WORKLOAD"

echo "--- Step 5/6: project service ---"
"$SCRIPT_DIR/project_service.sh" "$WORKLOAD"

echo "--- Step 6/6: aggregate ---"
"$SCRIPT_DIR/aggregate.sh" "$WORKLOAD_DIR"

echo "=== Tune-v2 pipeline complete ==="
