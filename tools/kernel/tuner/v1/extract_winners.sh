#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a single committed report.
#
# Why: tune_all_cases.sh runs each (case, model) in its own python
# invocation, and LocalDbManager creates a fresh /tmp/kernel_tuner_run_*
# folder per invocation — so the inspector for DECODE / PREFILL / MIXED
# each takes a different --db-path. This script parses both the
# `Tuning <case> (case_set_id=<id>)` phase headers and the
# `Database initialized at <path>` lines from the runlog, pairs them in
# order, runs the inspector for each, and concatenates the output.
#
# Usage:
#   tools/kernel/tuner/v1/extract_winners.sh [runlog_path]
#
#   runlog_path   Path to a tune_all_*.txt runlog. Defaults to the most
#                 recent file matching tmp/log/tune_all_*.txt.
#
# Output:
#   tmp/log/tuning_winners_<date>_<time>.txt   (committed + pushed)

set -euo pipefail

RUNLOG="${1:-}"
if [ -z "$RUNLOG" ]; then
    RUNLOG=$(ls -t tmp/log/tune_all_*.txt 2>/dev/null | head -1 || true)
fi
[ -n "$RUNLOG" ] && [ -f "$RUNLOG" ] \
    || { echo "Runlog not found: ${RUNLOG:-<none>}" >&2; exit 1; }
echo "Using runlog: $RUNLOG"

# `Tuning <case> (case_set_id=<id>)` phase headers — one per phase.
# Greedy match grabs just the relevant substring even when wrapped in
# the wrappers `===== <timestamp> Tuning ... =====` decoration.
#
# Using `while read` rather than `mapfile -t < <(...)` for bash 3.2
# portability (mapfile is bash 4+; macOSs system bash is 3.2).
CASE_LINES=()
while IFS= read -r line; do
    CASE_LINES+=("$line")
done < <(grep -oE 'Tuning [a-z]+ \(case_set_id=[^)]+\)' "$RUNLOG")

# `Database initialized at /tmp/kernel_tuner_run_<stamp>` — one per
# python invocation. awks $NF picks just the path from the
# LocalDbManager log line.
DB_PATHS=()
while IFS= read -r line; do
    DB_PATHS+=("$line")
done < <(grep -E 'Database initialized at /tmp/kernel_tuner_run_' "$RUNLOG" \
            | awk '{print $NF}')

# Sanity: order in the runlog must match because phases run sequentially
# and each writes both lines before any other phase starts.
if [ "${#CASE_LINES[@]}" -ne "${#DB_PATHS[@]}" ]; then
    echo "Mismatch: ${#CASE_LINES[@]} case headers vs" \
         "${#DB_PATHS[@]} DB paths in $RUNLOG" >&2
    exit 1
fi
if [ "${#CASE_LINES[@]}" -eq 0 ]; then
    echo "No phases found in $RUNLOG (was the run aborted?)" >&2
    exit 1
fi

OUT="tmp/log/tuning_winners_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p tmp/log

{
    echo "===== Kernel-tune winners ====="
    echo "Source runlog:  $RUNLOG"
    echo "Extracted at:   $(date '+%F %T')"
    echo "Phases found:   ${#CASE_LINES[@]}"
    echo ""

    for i in "${!CASE_LINES[@]}"; do
        line="${CASE_LINES[$i]}"
        CASE=$(echo "$line" | sed -E 's/^Tuning ([a-z]+).*/\1/')
        CASE_SET_ID=$(echo "$line" | sed -E 's/^.*case_set_id=([^)]+)\).*/\1/')
        DB_PATH="${DB_PATHS[$i]}"

        echo ""
        echo "===== $CASE ====="
        echo "case_set_id: $CASE_SET_ID"
        echo "db_path:     $DB_PATH"
        echo ""
        # Inspector failures shouldnt abort the loop — emit a marker and
        # keep going so a partial report is still useful.
        python3 -m tools.kernel.tuner.v1.inspect_result_cli \
            --source=local \
            --db-path="$DB_PATH" \
            query_min_latency \
            --case_set_id="$CASE_SET_ID" \
            --run_id=0 \
            || echo "WARN: inspector failed for $CASE (db=$DB_PATH)" >&2
    done
} 2>&1 | tee "$OUT"

echo ""
echo "Wrote: $OUT"

git add -f "$OUT"
if git diff --cached --quiet; then
    echo "Nothing new to commit."
    exit 0
fi
git commit -m "[Kernel-Tune] Winners extracted from $(basename "$RUNLOG")"
git push origin "$(git rev-parse --abbrev-ref HEAD)" \
    || echo "WARN: push failed; commit is local only" >&2
