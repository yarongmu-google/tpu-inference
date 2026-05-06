#!/bin/bash
# End-to-end ML optimization pipeline orchestrator.
#
# Usage:
#   tools/run_pipeline.sh [--smoke] <path/to/.service>
#
# The .service file references its workload via the `case_file` field;
# the orchestrator extracts the workload path from there. Two-arg
# legacy form (workload first, service second) was redundant — both
# inputs effectively named the same workload twice.
#
# This script executes the full 3-layer architecture:
#   1. Reads the Workload boundaries (from the .service files case_file)
#   2. Tunes the TPU hardware kernels (VMEM pruning enabled)
#   3. Builds the .kernel hardware registry
#   4. Sweeps the vLLM scheduler, auto-linking the tuned kernels
#   5. Extracts the absolute best throughput configuration to a production .service file.

set -euo pipefail

SMOKE=0
if [ "${1:-}" == "--smoke" ]; then
    SMOKE=1
    shift
fi

SERVICE="${1:-}"
if [ -z "$SERVICE" ]; then
    echo "Usage: $0 [--smoke] <service_file>" >&2
    exit 1
fi
if [ ! -f "$SERVICE" ]; then
    echo "Error: Service file not found: $SERVICE" >&2
    exit 1
fi

# Extract workload path + sweep_name from the .service file. The
# case_file field in .service is encoded relative to the spec dir, so
# we resolve it the same way sweep.py does at load_spec time.
read -r WORKLOAD SWEEP_NAME < <(python3 - <<PY "$SERVICE"
import json, os, sys
spec_path = sys.argv[1]
with open(spec_path) as f:
    spec = json.load(f)
case_file = spec["case_file"]
if not os.path.isabs(case_file):
    case_file = os.path.normpath(
        os.path.join(os.path.dirname(spec_path), case_file))
print(case_file, spec["sweep_name"])
PY
)

if [ ! -f "$WORKLOAD" ]; then
    echo "Error: case_file referenced by $SERVICE does not exist: $WORKLOAD" >&2
    exit 1
fi

export SMOKE_TEST=$SMOKE

WORKLOAD_BASENAME=$(basename "$WORKLOAD" .workload)
SERVICE_BASENAME=$(basename "$SERVICE" .service)
MODEL_DIR=$(dirname "$WORKLOAD")
PROD_KERNEL="${MODEL_DIR}/production.kernel"

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
echo " Service:  $SERVICE_BASENAME"
echo " Workload: $WORKLOAD_BASENAME (resolved from .service)"
if [ "$SMOKE" -eq 1 ]; then
    echo " MODE:     SMOKE TEST (Truncated search space)"
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
