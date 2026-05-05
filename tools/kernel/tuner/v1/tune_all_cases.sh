#!/bin/bash
# Tune RPA v3 kernel for all three cases (DECODE, PREFILL, MIXED) sequentially.
#
# Usage:
#   tools/kernel/tuner/v1/tune_all_cases.sh <path_to_case.workload> [runlog_label]
#
#   path_to_case.workload  Required. Path to the .workload file defining the workload.
#   runlog_label      Optional label for output files. Defaults to the case filename.
#
# Outputs:
#   tmp/log/tune_all_<label>_<date>.txt           — runlog (auto-committed per case)
#   /tmp/kernel_tuner_run_<YYYY_MM_DD>/           — local DB (one row per case_set_id)
#
# Time budget (per existing PREFILL trial, async-dispatch timing):
#   DECODE: ~5–10 min  (q_len=1 collapses the search space)
#   PREFILL: ~3 h     (~1740 cases)
#   MIXED:  ~30–60 min
#   Total:  ~3.5–4 h. Fits 12 h with margin.
#
# DECODE goes first deliberately: it's the cheapest, so if anything is
# fundamentally broken (e.g. the env-var override didn't take, or the
# kernel can't compile any combo on this TPU) you find out in 15 min
# rather than 3 h.

set -euo pipefail

CASE_FILE="${1:-}"
if [ -z "$CASE_FILE" ] || [ ! -f "$CASE_FILE" ]; then
    echo "Usage: $0 <path_to_case.workload> [runlog_label]" >&2
    exit 1
fi

# Extract filename without extension for default label
DEFAULT_LABEL=$(basename "$CASE_FILE" .workload)
LABEL="${2:-$DEFAULT_LABEL}"

# Per-run timestamp for case_set_id. The RUNLOG filename is intentionally
# stable across runs (so accumulation in build_kernel_registry merges into
# the same .kernel file — see commit 8d4242e3) but the case_set_id MUST
# be unique per run because LocalDbManager treats it as a primary key in
# /tmp/kernel_tuner_run_*/CaseSet.json. Without DATE, re-running the same
# workload would either overwrite the previous run's DB rows or — under
# `set -u` — crash with `DATE: unbound variable`.
DATE=$(date +%Y%m%d_%H%M%S)

# Load the workload definitions into the environment.
# set -a forces all assigned variables to be exported.
set -a
source "$CASE_FILE"
set +a

mkdir -p tmp/log
RUNLOG="tmp/log/tune_all_${LABEL}.txt"
echo "Runlog: $RUNLOG"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

commit_progress() {
    local note="$1"
    git add -f "$RUNLOG"
    if git diff --cached --quiet; then
        return 0
    fi
    git commit -m "[Kernel-Tune] progress: $note"
    git push origin "$BRANCH" || echo "WARN: push failed for $note" >&2
}

# Disable Python output buffering so the runlog updates in near-real-time.
# Without this, you'd see hours of silence followed by a wall of text when
# Python finally flushes at exit.
export PYTHONUNBUFFERED=1

# Header (written outside the per-case loop so it's only committed once,
# at the first per-case fire).
{
    echo "===== $(date '+%F %T') Starting full kernel tune for $LABEL ====="
    echo "Cases: decode -> prefill -> mixed"
    echo "Branch: $BRANCH"
    echo "Git commit: $(git rev-parse HEAD)"
    echo ""
} | tee -a "$RUNLOG"

# Per-case loop. Each iteration's stdout is tee-appended to the runlog,
# then commit_progress fires AFTER tee finishes — so the runlog on disk
# already contains this case's full output before we git add it.
# Without this structure (per-case tee + commit), the alternative would
# be a single big tee around the whole loop, but then commit_progress
# inside the tee subshell can't see runlog disk state and per-case
# commits wouldn't actually capture per-case progress.
for CASE in decode prefill mixed; do
    CASE_SET_ID="${LABEL}_${CASE}_${DATE}"
    {
        echo ""
        echo "===== $(date '+%F %T') Tuning $CASE (case_set_id=$CASE_SET_ID) ====="
        START_S=$(date +%s)

        RPA_V3_TUNER_CASES="$CASE" python3 \
            -m tools.kernel.tuner.v1.kernel_tuner_runner \
            --kernel_tuner_name=rpa_v3_kernel_tuner \
            --run_locally=true \
            --case_set_id="$CASE_SET_ID" \
            --case_set_desc="$LABEL $CASE kernel tune $(date +%F)"

        DUR=$(( $(date +%s) - START_S ))
        echo "===== $(date '+%F %T') $CASE done in ${DUR}s ====="
    } 2>&1 | tee -a "$RUNLOG"

    commit_progress "$CASE done ($LABEL $DATE)"
done

# Final summary (also committed).
{
    echo ""
    echo "===== $(date '+%F %T') ALL CASES DONE ====="
    echo ""
    DB_PATH=$(ls -td /tmp/kernel_tuner_run_* 2>/dev/null | head -1)
    echo "Local DB: $DB_PATH"
    echo ""
    echo "Inspector commands to extract winners (run on the same TPU VM):"
    for CASE in decode prefill mixed; do
        echo "  python3 -m tools.kernel.tuner.v1.inspect_result_cli \\"
        echo "      --source=local --db-path=$DB_PATH \\"
        echo "      query_min_latency \\"
        echo "      --case_set_id=${LABEL}_${CASE}_${DATE} --run_id=0"
    done
} 2>&1 | tee -a "$RUNLOG"

commit_progress "all done ($LABEL $DATE)"
