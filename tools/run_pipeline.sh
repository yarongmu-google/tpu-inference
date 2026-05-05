#!/bin/bash
# End-to-end ML optimization pipeline orchestrator.
#
# Usage:
#   tools/run_pipeline.sh [--smoke] <path/to/.workload> <path/to/.service>
#
# This script executes the full 3-layer architecture:
#   1. Reads the Workload boundaries
#   2. Tunes the TPU hardware kernels (VMEM pruning enabled)
#   3. Builds the `.kernel` hardware registry
#   4. Sweeps the vLLM scheduler, auto-linking the tuned kernels
#   5. Extracts the absolute best throughput configuration to a production `.service` file.

set -euo pipefail

SMOKE=0
if [ "${1:-}" == "--smoke" ]; then
    SMOKE=1
    shift
fi

WORKLOAD="${1:-}"
SERVICE="${2:-}"

if [ -z "$WORKLOAD" ] || [ -z "$SERVICE" ]; then
    echo "Usage: $0 [--smoke] <workload_file> <service_file>"
    exit 1
fi

if [ ! -f "$WORKLOAD" ]; then
    echo "Error: Workload file not found: $WORKLOAD" >&2
    exit 1
fi

if [ ! -f "$SERVICE" ]; then
    echo "Error: Service file not found: $SERVICE" >&2
    exit 1
fi

export SMOKE_TEST=$SMOKE

# Extract identifiers
WORKLOAD_BASENAME=$(basename "$WORKLOAD" .workload)
SERVICE_BASENAME=$(basename "$SERVICE" .service)
# Sweep name is defined inside the .service file
SWEEP_NAME=$(python3 -c "import json; print(json.load(open('$SERVICE'))['sweep_name'])")

PIPELINE_LOG="tmp/log/pipeline_${WORKLOAD_BASENAME}.txt"
SERVICE_LOG="tmp/log/script_build_service_registry_${SWEEP_NAME}.txt"
mkdir -p tmp/log

# Guarantee logs are committed even if a python script crashes (set -e)
commit_logs() {
    if [ -f "$SERVICE_LOG" ]; then
        git add -f "$SERVICE_LOG"
        if ! git diff --cached --quiet; then
            git commit -m "[Logs] Update build_service_registry script log for $SWEEP_NAME" || true
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
echo " Workload: $WORKLOAD_BASENAME"
echo " Service:  $SERVICE_BASENAME"
if [ "$SMOKE" -eq 1 ]; then
    echo " MODE:     SMOKE TEST (Truncated search space)"
fi
echo "=========================================================="
echo ""

{
    echo "=== Layer 1: Hardware Tuning ==="
    MODEL_DIR=$(dirname "$WORKLOAD")
    PROD_KERNEL="${MODEL_DIR}/production.kernel"
    
    # Export the registry path so the tuner can skip already-tuned cases
    export RPA_V3_KERNEL_REGISTRY="$PROD_KERNEL"

    # Execute the tuner against the SSoT workload boundaries
    tools/kernel/tuner/v1/tune_all_cases.sh "$WORKLOAD" "$WORKLOAD_BASENAME"

    echo ""
    echo "=== Layer 1: Building Kernel Registry ==="
    # Pass the exact known runlog name
    RUNLOG="tmp/log/tune_all_${WORKLOAD_BASENAME}.txt"
    tools/kernel/tuner/v1/build_kernel_registry.sh "$RUNLOG" "$PROD_KERNEL"
    echo "Kernel Registry updated at $PROD_KERNEL."

    echo ""
    echo "=== Layer 2: Service Sweeping ==="
    # Run the scheduler combinations, which will auto-link from the .kernel file
    tools/benchmark/sweep.sh "$SERVICE"

    echo ""
    echo "=== Layer 3: Building Production Configuration ==="
    SWEEP_DIR="tmp/bench_${WORKLOAD_BASENAME}_${SWEEP_NAME}"
    PROD_FILE="${MODEL_DIR}/production.service"

    # Extract the #1 ranked result and append to the production artifact, logging the output
    {
        python3 -m tools.benchmark.build_service_registry "$SWEEP_DIR" --export-production "$PROD_FILE"
    } 2>&1 | tee "$SERVICE_LOG"
    
    echo ""
    echo "=========================================================="
    echo " Pipeline Complete."
    echo " Final production configuration saved to:"
    echo " $PROD_FILE"
    echo "=========================================================="
} 2>&1 | tee "$PIPELINE_LOG"
