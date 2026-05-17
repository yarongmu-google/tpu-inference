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
#
# Use python3 os.path.realpath instead of `cd && pwd -P` so the FILE
# itself is symlink-resolved, not just its parent directory. Without
# this, a symlink like cases/v7x/foo.workload -> /tmp/evil.sh would
# pass the prefix check (cases/v7x/foo.workload looks under cases/)
# but `source` would execute the symlink target.
#
# tune_all_cases.sh has its own copy of this guard (for the case where
# its invoked directly, not via the orchestrator).
WORKLOAD_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$WORKLOAD")
CASES_ROOT_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "tools/benchmark/cases")
case "$WORKLOAD_REAL" in
    "$CASES_ROOT_REAL"/*) ;;
    *)
        echo "Error: workload (after symlink resolution) must live under tools/benchmark/cases/ (sourced as bash)." >&2
        echo "Got: $WORKLOAD_REAL" >&2
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
    "tools/benchmark/build_service_registry.sh" \
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
# Layer 4 validation bench writes here. Stable tag (no timestamp) so
# subsequent pipeline runs overwrite the prior validation, giving a
# single canonical "production bench" output per workload that the
# commit_logs trap can find.
PROD_BENCH_TAG="production"
PROD_BENCH_DIR="tmp/bench_${WORKLOAD_BASENAME}_${PROD_BENCH_TAG}"

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

# Rank fields (Layer 3 ranker direction). Both have throughput-style
# defaults so older recipes that predate the field continue to behave
# exactly as before.
RANK_METRIC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('rank_metric','metrics.RequestThroughput'))" "$SERVICE")
RANK_DESCENDING=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('rank_descending', True))" "$SERVICE")

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
# 2. Auto-push is opt-OUT via PIPELINE_AUTOPUSH=0 (default on). The
#    trap fires on Ctrl-C and `set -e` aborts too; that means a
#    partial / killed pipeline still pushes its in-flight commits.
#    Acceptable: the trap only commits paths it owns
#    (production.kernel, production.service, the two logs), and
#    those are recoverable from origin if a crash leaves them in a
#    weird state. Users debugging locally who don't want their
#    branch updated can set PIPELINE_AUTOPUSH=0 to disable.
_should_autopush() {
    [ "${PIPELINE_AUTOPUSH:-1}" = "1" ]
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
    _commit_path_only "$PROD_KERNEL" \
        "[Kernel-Tune] Update production.kernel for $WORKLOAD_BASENAME"
    _commit_path_only "$SERVICE_LOG" \
        "[Logs] Update build_service_registry script log for $SWEEP_NAME"
    _commit_path_only "$PROD_SERVICE" \
        "[Pipeline] Update production.service for $WORKLOAD_BASENAME"
    # Layer 4 validation bench: commit metrics + meta from the
    # canonical production-bench dir. The dir may not exist if Layer 4
    # didn't run (e.g., earlier-layer crash) — _commit_path_only
    # handles missing files gracefully (returns 0).
    _commit_path_only "$PROD_BENCH_DIR/metrics.txt" \
        "[Pipeline] Layer 4 validation bench metrics for $WORKLOAD_BASENAME"
    _commit_path_only "$PROD_BENCH_DIR/meta.txt" \
        "[Pipeline] Layer 4 validation bench meta for $WORKLOAD_BASENAME"
    _commit_path_only "$PIPELINE_LOG" \
        "[Pipeline] Update master orchestrator runlog for $WORKLOAD_BASENAME"
    if _should_autopush; then
        local branch
        branch="$(git rev-parse --abbrev-ref HEAD)"
        # Pull --rebase first to avoid race with concurrent pushes
        # (design-doc edits, sweep auto-commits, tune progress). On
        # conflict / network failure, abandon push but keep local
        # commits; human syncs later. The pipeline keeps running.
        if ! git pull --rebase origin "$branch" >/dev/null 2>&1; then
            git rebase --abort 2>/dev/null || true
            echo "WARN: PIPELINE_AUTOPUSH=1 but pull --rebase failed; push abandoned, local commits kept. Sync manually later." >&2
        elif ! git push origin "$branch"; then
            echo "WARN: PIPELINE_AUTOPUSH=1; pull OK but push failed. Local commits kept. Sync manually later." >&2
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
    # Resume is owned by raw.jsonl in the tune DB dir (per-combo,
    # base-class _load_raw_jsonl_skip_set). production.kernel is the
    # final-winners artifact only — never consulted for resume.

    # Execute the tuner against the SSoT workload boundaries
    tools/kernel/tuner/v1/tune_all_cases.sh "$WORKLOAD" "$WORKLOAD_BASENAME"

    echo ""
    echo "=== Layer 1: Building Kernel Registry ==="
    RUNLOG="tmp/log/tune_all_${WORKLOAD_BASENAME}.txt"
    # build_kernel_registry.sh is pure now (no git). The orchestrators
    # commit_logs trap stages the produced .kernel file alongside
    # production.service and the pipeline log.
    tools/kernel/tuner/v1/build_kernel_registry.sh "$RUNLOG" "$PROD_KERNEL"
    echo "Kernel Registry updated at $PROD_KERNEL."

    echo ""
    echo "=== Layer 2: Service Sweeping ==="
    # Run the scheduler combinations, auto-linking from the .kernel file
    tools/benchmark/sweep.sh "$SERVICE"

    echo ""
    echo "=== Layer 3: Building Production Configuration ==="
    SWEEP_DIR="tmp/bench_${WORKLOAD_BASENAME}_${SWEEP_NAME}"

    # Pre-flight: SWEEP_DIR is the per-combo result dir parent that
    # run_benchmark.sh writes to (tmp/bench_${CASE_NAME}_${TAG}/<combo_id>/).
    # If sweep.py / run_benchmark.sh ever change the naming convention,
    # build_service_registry would walk an empty/absent directory,
    # produce zero results, and silently exit without writing
    # production.service — the user would think Layer 3 succeeded but
    # find no winner. Fail-fast here with the expected path.
    if [ ! -d "$SWEEP_DIR" ]; then
        echo "Error: SWEEP_DIR does not exist after Layer 2: $SWEEP_DIR" >&2
        echo "Expected layout: tmp/bench_<workload>_<sweep_name>/<combo_id>/" >&2
        echo "Either Layer 2 failed silently, or run_benchmark.sh changed its RESULT_DIR template." >&2
        exit 1
    fi
    if [ -z "$(ls -A "$SWEEP_DIR" 2>/dev/null)" ]; then
        echo "Error: SWEEP_DIR is empty: $SWEEP_DIR" >&2
        echo "Layer 2 ran but produced no combo dirs." >&2
        exit 1
    fi

    # Compose the optional --ascending flag. build_service_registry's
    # default direction is descending (throughput); pass --ascending iff
    # the recipe asked for ascending order (latency).
    ASCENDING_FLAG=()
    if [ "$RANK_DESCENDING" = "False" ]; then
        ASCENDING_FLAG=(--ascending)
    fi
    {
        tools/benchmark/build_service_registry.sh "$SWEEP_DIR" \
            --metric "$RANK_METRIC" \
            "${ASCENDING_FLAG[@]}" \
            --export-production "$PROD_SERVICE" \
            --kernel-id "$KERNEL_ID" \
            --service-id "$SERVICE_ID"
    } 2>&1 | tee "$SERVICE_LOG"

    echo ""
    echo "=== Layer 4: Production Validation Bench ==="
    # Re-bench the workload using JUST the winning config from Layer 3's
    # production.service, so we have a clean reproducible perf number
    # for the production config (not just one combo's number buried in
    # the sweep grid). Same kernel, same service, same env vars as the
    # winner — but a fresh invocation into a stable result dir
    # ($PROD_BENCH_DIR) the commit_logs trap knows how to pick up.
    #
    # Subshell so workload env vars don't leak into the parent script
    # after Layer 4 completes (they're sourced via `set -a` to derive
    # the workload key).
    (
        set -a
        source "$WORKLOAD"
        set +a

        # Workload key format matches build_service_registry's
        # _make_workload_key (see tools/benchmark/build_service_registry.py).
        # If that schema changes, this lookup must move in lockstep.
        WORKLOAD_KEY="${KERNEL_ID}__${SERVICE_ID}__${MODEL}__tp${TENSOR_PARALLEL_SIZE}__${INPUT_LEN}_in_${OUTPUT_LEN}_out"
        echo "Validation bench: workload_key=$WORKLOAD_KEY"

        # Extract the winning config from production.service and emit
        # it as exportable bash. Fail loudly if the entry is missing —
        # Layer 3 should have just written it; missing => Layer 3 bug
        # or stale spec, NOT a silent fall-back to defaults (which
        # would invalidate the validation).
        PROD_ENV_FILE="tmp/log/prod_env_${WORKLOAD_BASENAME}.sh"
        python3 - "$PROD_SERVICE" "$WORKLOAD_KEY" "$PROD_ENV_FILE" <<'PYEOF'
import json, sys
prod_service_path, workload_key, out_path = sys.argv[1:4]
with open(prod_service_path) as f:
    data = json.load(f)
cfg = data.get("best_configs_by_workload", {}).get(workload_key)
if cfg is None:
    known = list(data.get("best_configs_by_workload", {}).keys())
    print(f"ERROR: no production config for workload_key={workload_key}",
          file=sys.stderr)
    print(f"Known keys: {known}", file=sys.stderr)
    sys.exit(1)
# Skip metadata + identity fields; export everything else.
SKIP = {"metrics", "kernel_id", "service_id", "model",
        "tensor_parallel_size"}
with open(out_path, "w") as f:
    for k, v in cfg.items():
        if k in SKIP:
            continue
        # json.dumps quotes strings safely for shell `export`.
        f.write(f"export {k}={json.dumps(v)}\n")
PYEOF
        # shellcheck source=/dev/null
        source "$PROD_ENV_FILE"

        # RESULT_DIR is derived as tmp/bench_${CASE_NAME}_${TAG} in
        # run_benchmark.sh; CASE_NAME=basename(workload,.workload)
        # matches WORKLOAD_BASENAME, so PROD_BENCH_DIR matches.
        tools/benchmark/run_benchmark.sh "$WORKLOAD" --result-tag "$PROD_BENCH_TAG"
    )

    echo ""
    echo "=========================================================="
    echo " Pipeline Complete — Summary"
    echo "=========================================================="

    # Automated verification block. The pipeline log captures every
    # output line, so this summary lands in tmp/log/pipeline_*.txt
    # for after-the-fact review without needing the user to run any
    # ls / cat / grep commands manually.

    echo ""
    echo "[Layer 1.5] production.kernel:"
    if [ -f "$PROD_KERNEL" ]; then
        ls -la "$PROD_KERNEL"
        python3 -c "
import json
with open('$PROD_KERNEL') as f:
    d = json.load(f)
print('  last_updated_at:', d.get('metadata', {}).get('last_updated_at', '?'))
results = d.get('results', {})
for case, entries in results.items():
    print(f'  {case}: {len(entries)} tuned entries')
" 2>/dev/null || echo "  (failed to parse)"
    else
        echo "  MISSING: $PROD_KERNEL — Layer 1.5 did not complete"
    fi

    echo ""
    echo "[Layer 3] production.service:"
    if [ -f "$PROD_SERVICE" ]; then
        ls -la "$PROD_SERVICE"
        python3 -c "
import json
with open('$PROD_SERVICE') as f:
    d = json.load(f)
for k in d.get('best_configs_by_workload', {}):
    print('  workload_key:', k)
" 2>/dev/null || echo "  (failed to parse)"
    else
        echo "  MISSING: $PROD_SERVICE — Layer 3 did not complete"
    fi

    echo ""
    echo "[Layer 4] validation bench at $PROD_BENCH_DIR:"
    if [ -f "$PROD_BENCH_DIR/metrics.txt" ]; then
        ls -la "$PROD_BENCH_DIR/metrics.txt" "$PROD_BENCH_DIR/meta.txt" 2>/dev/null
        echo "  --- metrics.txt ---"
        sed 's/^/    /' "$PROD_BENCH_DIR/metrics.txt"
    else
        echo "  MISSING: $PROD_BENCH_DIR/metrics.txt — Layer 4 did not complete"
    fi

    echo ""
    echo "[Git] Recent commits on $(git rev-parse --abbrev-ref HEAD):"
    git log --oneline -10
    echo ""
    echo "[Git] Local vs origin status:"
    git status --short --branch | head -5

    echo ""
    echo "=========================================================="
    echo " End of summary. Production service: $PROD_SERVICE"
    echo "=========================================================="
} 2>&1 | tee "$PIPELINE_LOG"
