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
#   tmp/log/tune_all_<label>.txt                       — runlog (auto-committed per case)
#   tmp/log/kernel_tuner_run/<label>_<case>/           — local DB (one dir per case).
#                                                        Fresh-vs-resume is HUMAN-controlled
#                                                        via KERNEL_TUNER_RESUME:
#                                                          unset (default): wraps wipes the
#                                                            dir before tuning -> fresh tune.
#                                                          =1: dir kept; raw.jsonl skip-set
#                                                            resumes already-done combos.
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

# Workload-path guard — duplicated from run_pipeline.sh because this
# script can be invoked directly (not just via the orchestrator).
# Without it, `set -a; source "$CASE_FILE"` below would execute
# arbitrary bash from any path the user passes. Use python3
# os.path.realpath so SYMLINKS are resolved (a symlink under cases/
# pointing to /tmp/evil.sh would otherwise slip through).
CASE_FILE_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$CASE_FILE")
CASES_ROOT_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "tools/benchmark/cases")
case "$CASE_FILE_REAL" in
    "$CASES_ROOT_REAL"/*) ;;
    *)
        echo "Error: workload (after symlink resolution) must live under tools/benchmark/cases/ (sourced as bash)." >&2
        echo "Got: $CASE_FILE_REAL" >&2
        exit 1
        ;;
esac

# Extract filename without extension for default label
DEFAULT_LABEL=$(basename "$CASE_FILE" .workload)
LABEL="${2:-$DEFAULT_LABEL}"

# case_set_id is stable: ${LABEL}_${CASE} (no date, no commit, no
# auto-discrimination). Fresh-vs-resume is HUMAN-CONTROLLED via
# KERNEL_TUNER_RESUME:
#   unset (default): the per-case loop wipes the DB dir before
#     running the tuner — guaranteed fresh tune.
#   =1:              the DB dir is kept as-is; raw.jsonl resume kicks
#     in via the base class's _load_raw_jsonl_skip_set +
#     _combo_skip_key, skipping combos already SUCCESS / FAILED_OOM /
#     SKIPPED.
DATE_SUFFIX=""
# Date kept as a separate var purely for filename / log purposes
# below; it is no longer wired into CASE_SET_ID.
DATE=$(date +%Y%m%d_%H%M%S)

# Load the workload definitions into the environment.
# set -a forces all assigned variables to be exported.
set -a
source "$CASE_FILE"
set +a

# Validate / derive MAX_MODEL_LEN. Same logic as in run_benchmark.sh —
# duplicated because both consumers need it and a sourced helper
# carries its own fragility. For fixed-shape sonnet workloads
# MAX_MODEL_LEN MUST equal INPUT_LEN + OUTPUT_LEN; otherwise the SMEM
# estimate (which scales as max_num_seqs * max_model_len / page_size)
# is wrong, which can either falsely-prune valid page sizes or pass
# combos that will fail at runtime.
EXPECTED_MML=$(( INPUT_LEN + OUTPUT_LEN ))
if [ -z "${MAX_MODEL_LEN:-}" ]; then
    MAX_MODEL_LEN=$EXPECTED_MML
    export MAX_MODEL_LEN
elif [ "$MAX_MODEL_LEN" -ne "$EXPECTED_MML" ]; then
    echo "Error: MAX_MODEL_LEN=$MAX_MODEL_LEN in $(basename "$CASE_FILE") does not match INPUT_LEN+OUTPUT_LEN=$EXPECTED_MML." >&2
    echo "Set MAX_MODEL_LEN=$EXPECTED_MML or remove the line to auto-compute." >&2
    exit 1
fi

mkdir -p tmp/log
RUNLOG="tmp/log/tune_all_${LABEL}.txt"
echo "Runlog: $RUNLOG"

# Sidecar manifest: machine-readable list of per-case (case_set_id, db_path)
# pairs. build_kernel_registry.py prefers this over regex-scraping the
# runlog — the runlogs prose ("Tuning <case> (case_set_id=...)" /
# "Database initialized at ...") is intended for humans and could change
# without notice. The manifest is the source of truth.
#
# Format: JSONL (one JSON object per line). Append-only writes via `>>`
# are atomic for short payloads on POSIX, so concurrent invocations
# with the same LABEL would interleave entries cleanly rather than
# racing a read-modify-write on a single JSON array.
#
# Path mirrors the runlog basename so consumers can derive one from the
# other. The .jsonl extension makes the format self-documenting.
# Truncated per-run (matches the ephemeral-log convention).
MANIFEST="tmp/log/tune_all_${LABEL}.manifest.jsonl"
: > "$MANIFEST"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

commit_progress() {
    local note="$1"
    git add -f "$RUNLOG"
    if git diff --cached --quiet; then
        return 0
    fi
    git commit -m "[Kernel-Tune] progress: $note"
    # Pull --rebase first so concurrent commits (sweep/bench/design-doc)
    # on the same branch don't reject our push. Race is real: pipeline +
    # human design-doc edits target the same branch. If rebase fails
    # (conflict / network), abandon push; tune keeps running and
    # accumulates local commits. Human syncs later when they next check.
    if ! git pull --rebase origin "$BRANCH" >/dev/null 2>&1; then
        git rebase --abort 2>/dev/null || true
        echo "WARN: pull --rebase failed for $note; push abandoned, local commit kept. Sync manually later." >&2
    elif ! git push origin "$BRANCH"; then
        echo "WARN: push failed for $note despite successful pull; local commit kept. Sync manually later." >&2
    fi
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
: "${CASES_TO_TUNE:=decode prefill mixed}"
# CASES_TO_TUNE env var (space-separated) overrides the default case
# loop. Useful for tuning a subset (e.g. CASES_TO_TUNE="logical" to
# extend an existing decode/prefill/mixed kernel registry with the
# decoupled-K LOGICAL winners). The CI default is unchanged so existing
# pipelines see no behaviour difference.
mkdir -p tmp/log/kernel_tuner_run
for CASE in $CASES_TO_TUNE; do
    CASE_SET_ID="${LABEL}_${CASE}${DATE_SUFFIX}"
    # Authoritative DB path: bash decides where the runner writes,
    # rather than the runner generating a timestamped path that bash
    # then has to discover via `ls -td` (which races under concurrent
    # invocations).
    #
    # Path lives under tmp/log/ (durable across reboot — `/tmp` is
    # tmpfs / cleared at boot and we lost 48h of prefill tuning twice
    # to that). gitignored via tmp/log/kernel_tuner_run/. Stable
    # CASE_SET_ID means a kill-restart REUSES this DB and the runner
    # skips the (tuning_key, tunable_params) combos already in
    # CaseResults.json — true resumable tuning.
    DB_PATH="tmp/log/kernel_tuner_run/${CASE_SET_ID}"
    {
        echo ""
        echo "===== $(date '+%F %T') Tuning $CASE (case_set_id=$CASE_SET_ID) ====="
        echo "DB path: $DB_PATH"
        # Fresh-vs-resume gate. Wiping the whole DB dir (SQLite
        # case_set + kernel.raw.jsonl + manifests) is the only way
        # to guarantee a fresh tune: just rotating raw.jsonl leaves
        # the SQLite case_set populated from a prior run, and the
        # base class reuses it, causing the new run to skip every
        # combo (the bug observed 2026-05-16 at 21:03).
        if [ "${KERNEL_TUNER_RESUME:-0}" != "1" ]; then
            rm -rf "$DB_PATH"
            echo "Fresh tune — wiped $DB_PATH (set KERNEL_TUNER_RESUME=1 to resume)"
        else
            echo "Resume — keeping existing $DB_PATH (KERNEL_TUNER_RESUME=1)"
        fi
        START_S=$(date +%s)

        # RPA_V3_TUNER_CASES set per-case here intentionally — overrides
        # any caller-provided value so this loop drives the case order.
        # If a caller wants to tune a single case, invoke kernel_tuner_runner
        # directly (this loop assumes it owns RPA_V3_TUNER_CASES).
        RPA_V3_TUNER_CASES="$CASE" python3 \
            -m tools.kernel.tuner.v1.kernel_tuner_runner \
            --kernel_tuner_name=rpa_v3_kernel_tuner \
            --run_locally=true \
            --case_set_id="$CASE_SET_ID" \
            --case_set_desc="$LABEL $CASE kernel tune" \
            --db_path="$DB_PATH"

        DUR=$(( $(date +%s) - START_S ))
        echo "===== $(date '+%F %T') $CASE done in ${DUR}s ====="
    } 2>&1 | tee -a "$RUNLOG"

    # Append (case, case_set_id, db_path) to the sidecar manifest.
    # DB_PATH was decided above and passed to the runner via --db_path,
    # so this is the path the runner just wrote — no discovery needed.
    # JSONL append: race-safe under concurrent runs.
    python3 -c "
import json, sys
entry = {'case': sys.argv[2], 'case_set_id': sys.argv[3], 'db_path': sys.argv[4]}
with open(sys.argv[1], 'a') as f:
    f.write(json.dumps(entry) + '\n')
" "$MANIFEST" "$CASE" "$CASE_SET_ID" "$DB_PATH"

    commit_progress "$CASE done ($LABEL $DATE)"
done

# Final summary (also committed). Inspector commands now reference each
# cases own DB path (one per case_set_id), matching how the runner
# was invoked above. Previously this used a single `ls -td` heuristic
# DB_PATH for all three cases, which was wrong as soon as we adopted
# per-case DB paths.
{
    echo ""
    echo "===== $(date '+%F %T') ALL CASES DONE ====="
    echo ""
    echo "Inspector commands to extract winners (run on the same TPU VM):"
    for CASE in $CASES_TO_TUNE; do
        CASE_DB="tmp/log/kernel_tuner_run/${LABEL}_${CASE}${DATE_SUFFIX}"
        echo "  python3 -m tools.kernel.tuner.v1.inspect_result_cli \\"
        echo "      --source=local --db-path=$CASE_DB \\"
        echo "      query_min_latency \\"
        echo "      --case_set_id=${LABEL}_${CASE}_${DATE} --run_id=0"
    done
} 2>&1 | tee -a "$RUNLOG"

commit_progress "all done ($LABEL $DATE)"
