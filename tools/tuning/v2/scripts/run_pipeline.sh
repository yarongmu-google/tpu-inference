#!/bin/bash
# run_pipeline.sh — orchestrator chaining the five v2 primitives.
#
# Usage:
#   tools/tuning/v2/scripts/run_pipeline.sh <workload>
#   tools/tuning/v2/scripts/run_pipeline.sh --from <step> <workload>
#
# Steps (run in order):
#   1. validate          (fail fast on bad workload)
#   2. tune_kernel       (kernel-tune long run with periodic commits)
#   3. project_kernel    (winners → <workload>.kernel)
#   4. sweep_service     (service-sweep long run)
#   5. project_service   (per-objective winners → <workload>.service)
#   6. aggregate         (per-model production.{kernel,service})
#
# Each step inherits auto-commit + auto-push (KERNEL_TUNER_NO_PUSH=1
# to disable). On any step failure, set -e aborts the orchestrator;
# raw stores already on disk plus partial git commits from earlier
# steps survive — restart re-uses the raw stores via skip-set
# (resume semantics).
#
# --from <step>: skip every step before <step>. Useful when an earlier
# step succeeded and you only need to re-run the rest (e.g. fixed a
# projection bug; want to re-project without re-tuning). Accepts the
# step names above. Default: validate (run everything).

set -euo pipefail

START_STEP="validate"

# Parse args. Order: `--from <step>` flags before positional <workload>.
while [ $# -gt 0 ]; do
    case "$1" in
        --from)
            if [ $# -lt 2 ]; then
                echo "--from requires a step name" >&2
                exit 1
            fi
            START_STEP="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# *//' >&2
            exit 0
            ;;
        --*)
            echo "Unknown flag: $1" >&2
            exit 1
            ;;
        *)
            break
            ;;
    esac
done

if [ $# -lt 1 ]; then
    echo "Usage: $0 [--from <step>] <workload>" >&2
    exit 1
fi

WORKLOAD="$1"
WORKLOAD_DIR=$(dirname "$WORKLOAD")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Steps in the same order as the comment header; the START_STEP arg
# names the first one we actually execute. Unknown START_STEP -> fail
# rather than silently skip everything.
STEPS=(validate tune_kernel project_kernel sweep_service project_service aggregate)
start_idx=-1
for i in "${!STEPS[@]}"; do
    if [ "${STEPS[$i]}" = "$START_STEP" ]; then
        start_idx="$i"
        break
    fi
done
if [ "$start_idx" -lt 0 ]; then
    echo "Unknown step for --from: $START_STEP" >&2
    echo "Valid steps: ${STEPS[*]}" >&2
    exit 1
fi

echo "=== Tune-v2 pipeline: $WORKLOAD (from: $START_STEP) ==="

run_step() {
    local idx="$1"; local name="$2"; local script="$3"; local arg="$4"
    if [ "$idx" -lt "$start_idx" ]; then
        echo "--- Step $((idx+1))/6: $name (SKIPPED, --from $START_STEP) ---"
        return 0
    fi
    echo "--- Step $((idx+1))/6: $name ---"
    "$SCRIPT_DIR/$script" "$arg"
}

run_step 0 validate         validate.sh        "$WORKLOAD"
run_step 1 "tune kernel"    tune_kernel.sh     "$WORKLOAD"
run_step 2 "project kernel" project_kernel.sh  "$WORKLOAD"
run_step 3 "sweep service"  sweep_service.sh   "$WORKLOAD"
run_step 4 "project service" project_service.sh "$WORKLOAD"
run_step 5 aggregate        aggregate.sh       "$WORKLOAD_DIR"

echo "=== Tune-v2 pipeline complete ==="
