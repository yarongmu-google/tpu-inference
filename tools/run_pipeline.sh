#!/bin/bash
# End-to-end ML optimization pipeline orchestrator.
#
# Usage:
#   tools/run_pipeline.sh [--smoke] <path/to/.workload>
#
# Architecture:
#   - Sole user input: a .workload file (model + traffic + hardware).
#   - The kernel + service pair (default: rpa_v3 + vllm) determines
#     which knobs are tuned, swept, or fixed. The recipe lives in
#     tools/benchmark/sweep_recipes.py — NOT in any per-workload
#     .service file. Per-workload .service files were a leaky
#     abstraction; they were deleted.
#   - The orchestrator synthesizes a sweep spec at runtime by combining
#     the workload with the (kernel, service) recipe, then runs the
#     standard 3-layer pipeline against it.
#
# Pipeline phases:
#   1. Tune the TPU hardware kernels (VMEM pruning enabled).
#   2. Build the .kernel hardware registry.
#   3. Sweep the vLLM scheduler, auto-linking the tuned kernels.
#   4. Extract the absolute best throughput configuration to a
#      production .service file (located alongside the workload).
#
# To use a different (kernel, service) pair (when more recipes are
# added): set KERNEL_ID and/or SERVICE_ID in the environment before
# invoking. Defaults: rpa_v3, vllm.

set -euo pipefail

SMOKE=0
if [ "${1:-}" == "--smoke" ]; then
    SMOKE=1
    shift
fi

WORKLOAD="${1:-}"
if [ -z "$WORKLOAD" ]; then
    echo "Usage: $0 [--smoke] <workload_file>" >&2
    exit 1
fi
if [ ! -f "$WORKLOAD" ]; then
    echo "Error: Workload file not found: $WORKLOAD" >&2
    exit 1
fi

# Workload files are bash-sourced (`set -a; source "$WORKLOAD"`) by
# run_benchmark.sh and tune_all_cases.sh — so the contents execute as
# bash. To prevent accidental arbitrary-code execution from a typo
# (e.g., passing /tmp/something.sh by mistake), require WORKLOAD to
# resolve under tools/benchmark/cases/. Repo-controlled workload files
# are still trusted; this just blocks paths outside the expected tree.
WORKLOAD_ABS=$(cd "$(dirname "$WORKLOAD")" && pwd -P)/$(basename "$WORKLOAD")
CASES_ROOT_ABS=$(cd "tools/benchmark/cases" && pwd -P)
case "$WORKLOAD_ABS" in
    "$CASES_ROOT_ABS"/*) ;;
    *)
        echo "Error: workload must live under tools/benchmark/cases/ (sourced as bash)." >&2
        echo "Got: $WORKLOAD_ABS" >&2
        exit 1
        ;;
esac

KERNEL_ID="${KERNEL_ID:-rpa_v3}"
SERVICE_ID="${SERVICE_ID:-vllm}"

# Pre-flight: every sub-script the orchestrator invokes must exist and
# be executable, otherwise we get an opaque "command not found" or
# permission error mid-pipeline. Fail fast with an actionable message.
for script in \
    "tools/kernel/tuner/v1/tune_all_cases.sh" \
    "tools/kernel/tuner/v1/build_kernel_registry.sh" \
    "tools/benchmark/sweep.sh"; do
    if [ ! -x "$script" ]; then
        echo "Error: sub-script not executable: $script" >&2
        exit 1
    fi
done

export SMOKE_TEST=$SMOKE

WORKLOAD_BASENAME=$(basename "$WORKLOAD" .workload)
MODEL_DIR=$(dirname "$WORKLOAD")
PROD_KERNEL="${MODEL_DIR}/production.kernel"
PROD_SERVICE="${MODEL_DIR}/production.service"

mkdir -p tmp/log

# Synthesize the sweep spec from (workload, kernel, service) recipe.
# This replaces the hand-curated .service files. Output goes under
# tmp/log so it does not pollute the repo, and gets overwritten on
# each pipeline run (matches the ephemeral-log convention).
SERVICE="tmp/log/synthesized_${WORKLOAD_BASENAME}.service"
python3 -m tools.benchmark.sweep_recipes \
    --workload "$WORKLOAD" \
    --kernel "$KERNEL_ID" \
    --service "$SERVICE_ID" \
    --out "$SERVICE"

# Read sweep_name back out (sweep.py and run_benchmark.sh use it for
# output dir naming; defined inside the synthesized spec).
SWEEP_NAME=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['sweep_name'])" "$SERVICE")

PIPELINE_LOG="tmp/log/pipeline_${WORKLOAD_BASENAME}.txt"
SERVICE_LOG="tmp/log/script_build_service_registry_${SWEEP_NAME}.txt"

# Guarantee logs and the production.service artifact are committed
# even if a python script crashes (set -e) — but commit ONLY the
# specific paths we own. Two correctness rules in this trap:
#
# 1. `git commit -- <path>` (path-restricted commit) is used instead
#    of "stage then commit". The "git add then git commit" form would
#    fold any pre-existing staged files (from the user's manual `git
#    add` outside the script) into our auto-commit. The path-
#    restricted form commits only what was added for that path,
#    leaving the user's pre-staged work intact.
#
# 2. Auto-push is opt-in via PIPELINE_AUTOPUSH=1 (default off). The
#    trap fires on Ctrl-C and `set -e` aborts; pushing partial logs
#    to a shared branch on every interrupt is too aggressive. The
#    full run (which actually wants the push) sets this explicitly.
_should_autopush() {
    [ "${PIPELINE_AUTOPUSH:-0}" = "1" ]
}

_commit_path_only() {
    local path="$1"
    local message="$2"
    [ -f "$path" ] || return 0
    git add -f -- "$path" 2>/dev/null || return 0
    if ! git diff --cached --quiet -- "$path"; then
        git commit -m "$message" -- "$path" || true
    fi
}

commit_logs() {
    _commit_path_only "$SERVICE_LOG" \
        "[Logs] Update build_service_registry script log for $SWEEP_NAME"
    _commit_path_only "$PROD_SERVICE" \
        "[Pipeline] Update production.service for $WORKLOAD_BASENAME"
    _commit_path_only "$PIPELINE_LOG" \
        "[Pipeline] Update master orchestrator runlog for $WORKLOAD_BASENAME"
    if _should_autopush; then
        git push origin "$(git rev-parse --abbrev-ref HEAD)" || \
            echo "WARN: PIPELINE_AUTOPUSH=1 but push failed" >&2
    fi
}
trap commit_logs EXIT

echo "=========================================================="
echo " Starting End-to-End Optimization Pipeline"
echo " Workload:        $WORKLOAD_BASENAME"
echo " Kernel+Service:  ${KERNEL_ID}+${SERVICE_ID}"
echo " Synthesized at:  $SERVICE"
if [ "$SMOKE" -eq 1 ]; then
    echo " MODE:            SMOKE TEST (Truncated search space)"
fi
echo "=========================================================="
echo ""

{
    echo "=== Layer 1: Hardware Tuning ==="
    # Export the registry path so the tuner can skip already-tuned cases
    export RPA_V3_KERNEL_REGISTRY="$PROD_KERNEL"

    # Execute the tuner against the SSoT workload boundaries
    tools/kernel/tuner/v1/tune_all_cases.sh "$WORKLOAD" "$WORKLOAD_BASENAME"

    echo ""
    echo "=== Layer 1: Building Kernel Registry ==="
    RUNLOG="tmp/log/tune_all_${WORKLOAD_BASENAME}.txt"
    tools/kernel/tuner/v1/build_kernel_registry.sh "$RUNLOG" "$PROD_KERNEL"
    echo "Kernel Registry updated at $PROD_KERNEL."

    echo ""
    echo "=== Layer 2: Service Sweeping ==="
    # Run the scheduler combinations, auto-linking from the .kernel file
    tools/benchmark/sweep.sh "$SERVICE"

    echo ""
    echo "=== Layer 3: Building Production Configuration ==="
    SWEEP_DIR="tmp/bench_${WORKLOAD_BASENAME}_${SWEEP_NAME}"

    {
        python3 -m tools.benchmark.build_service_registry "$SWEEP_DIR" \
            --export-production "$PROD_SERVICE" \
            --kernel-id "$KERNEL_ID" \
            --service-id "$SERVICE_ID"
    } 2>&1 | tee "$SERVICE_LOG"

    echo ""
    echo "=========================================================="
    echo " Pipeline Complete."
    echo " Final production configuration saved to:"
    echo " $PROD_SERVICE"
    echo "=========================================================="
} 2>&1 | tee "$PIPELINE_LOG"
