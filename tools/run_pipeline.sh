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

KERNEL_ID="${KERNEL_ID:-rpa_v3}"
SERVICE_ID="${SERVICE_ID:-vllm}"

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
# even if a python script crashes (set -e). production.service is the
# whole point of the pipeline; without committing it, a remote reader
# only sees the kernel registry + log breadcrumbs and has to re-run
# locally to reconstruct the winner.
commit_logs() {
    if [ -f "$SERVICE_LOG" ]; then
        git add -f "$SERVICE_LOG"
        if ! git diff --cached --quiet; then
            git commit -m "[Logs] Update build_service_registry script log for $SWEEP_NAME" || true
        fi
    fi
    if [ -f "$PROD_SERVICE" ]; then
        git add -f "$PROD_SERVICE"
        if ! git diff --cached --quiet; then
            git commit -m "[Pipeline] Update production.service for $WORKLOAD_BASENAME" || true
        fi
    fi
    if [ -f "$PIPELINE_LOG" ]; then
        git add -f "$PIPELINE_LOG"
        if ! git diff --cached --quiet; then
            git commit -m "[Pipeline] Update master orchestrator runlog for $WORKLOAD_BASENAME" || true
            git push origin "$(git rev-parse --abbrev-ref HEAD)" || true
        fi
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
