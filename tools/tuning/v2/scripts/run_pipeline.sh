#!/bin/bash
# run_pipeline.sh — orchestrator chaining the six v2 primitives.
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
# to disable push only; KERNEL_TUNER_NO_COMMIT=1 to also skip the
# commit step entirely — useful for smoke runs). On any step
# failure, set -e aborts the orchestrator; raw stores already on
# disk plus partial git commits from earlier steps survive — restart
# re-uses the raw stores via skip-set (resume semantics).
#
# --from <step>: skip every step before <step>. Useful when an
# earlier step succeeded and you only need to re-run the rest (e.g.
# fixed a projection bug; want to re-project without re-tuning).
# Accepts the step names above. Default: validate (run everything).
#
# Arg passthrough to inner steps is via env vars:
#   EXTRA_VALIDATE_FLAGS       → validate.sh
#   EXTRA_TUNE_FLAGS           → tune_kernel.sh        (e.g. --iters 1)
#   EXTRA_PROJECT_KERNEL_FLAGS → project_kernel.sh
#   EXTRA_SWEEP_FLAGS          → sweep_service.sh      (e.g. --timeout 60)
#   EXTRA_PROJECT_SERVICE_FLAGS → project_service.sh
#   EXTRA_AGGREGATE_FLAGS      → aggregate.sh
#
# Common smoke recipe (one combo, no commits, no real bench):
#   SMOKE_TEST=1 \
#   KERNEL_TUNER_NO_COMMIT=1 \
#   MOCK_BENCH=1 \
#   EXTRA_TUNE_FLAGS="--iters 1 --warmup 0" \
#   tools/tuning/v2/scripts/run_pipeline.sh <workload>
#
# Limitations of today's smoke:
#   - Only the LOGICAL kernel case is enumerated; D/M/P enumerators
#     have not landed yet. The resulting <workload>.kernel has one
#     winner (case=logical); cli/lookup.lookup_env will omit
#     RPA_D_BLOCK_SIZES / RPA_M_BLOCK_SIZES until the missing
#     enumerators land.
#   - MOCK_BENCH=1 stamps `mock=True` on every synthetic row but
#     NO automatic projection filter is wired — a smoke run that
#     shares a .service.raw partition with real bench data will
#     produce a mock winner that lookup_env will return as the
#     deploy env. Operational guidance: run smoke against a
#     throwaway workload directory (e.g. tools/benchmark/cases/smoke/...)
#     so the .service.raw partition stays smoke-only.

set -euo pipefail

START_STEP="validate"

print_help() {
    # Portable: print every line up to the first blank, drop the
    # leading "# " (or "#"). Same idea as `sed -n '2,/^$/p'` but
    # implemented in pure bash so it works on BSD sed (macOS) too.
    local in_header=1
    while IFS= read -r line; do
        # Stop at the first non-comment line after the shebang.
        if [ "$line" = "" ]; then
            break
        fi
        case "$line" in
            "#!"*) continue ;;
            "#"*)  printf '%s\n' "${line#\# }" | sed 's/^#//' ;;
            *)     break ;;
        esac
    done < "$0"
}

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
            print_help
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
    local extra_var="${5:-}"
    local extra=""
    if [ -n "$extra_var" ]; then
        extra="${!extra_var:-}"
    fi
    if [ "$idx" -lt "$start_idx" ]; then
        echo "--- Step $((idx+1))/6: $name (SKIPPED, --from $START_STEP) ---"
        return 0
    fi
    echo "--- Step $((idx+1))/6: $name ---"
    # shellcheck disable=SC2086  # word-split $extra on purpose
    "$SCRIPT_DIR/$script" "$arg" $extra
}

run_step 0 validate         validate.sh         "$WORKLOAD"     EXTRA_VALIDATE_FLAGS
run_step 1 "tune kernel"    tune_kernel.sh      "$WORKLOAD"     EXTRA_TUNE_FLAGS
run_step 2 "project kernel" project_kernel.sh   "$WORKLOAD"     EXTRA_PROJECT_KERNEL_FLAGS
run_step 3 "sweep service"  sweep_service.sh    "$WORKLOAD"     EXTRA_SWEEP_FLAGS
run_step 4 "project service" project_service.sh "$WORKLOAD"     EXTRA_PROJECT_SERVICE_FLAGS
run_step 5 aggregate        aggregate.sh        "$WORKLOAD_DIR" EXTRA_AGGREGATE_FLAGS

echo "=== Tune-v2 pipeline complete ==="
