#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a single committed .kernel JSON report.
#
# Usage:
#   tools/kernel/tuner/v1/build_kernel_registry.sh [runlog_path]

set -euo pipefail

RUNLOG="${1:-}"
if [ -z "$RUNLOG" ]; then
    RUNLOG=$(ls -t tmp/log/tune_all_*.txt 2>/dev/null | head -1 || true)
fi

# Extract label for the script log
LABEL=$(basename "$RUNLOG" | sed 's/tune_all_//' | sed 's/\.txt//')
SCRIPT_LOG="tmp/log/script_build_kernel_registry_${LABEL}.txt"

{
    echo "=== Executing build_kernel_registry.py ==="
    python3 -m tools.kernel.tuner.v1.build_kernel_registry "$RUNLOG"

    # Find the most recently written .kernel file
    OUT=$(ls -t tmp/log/*.kernel 2>/dev/null | head -1)

    if [ -z "$OUT" ]; then
        echo "Error: No .kernel file produced." >&2
        exit 1
    fi

    git add -f "$OUT"
    if git diff --cached --quiet; then
        echo "Nothing new to commit."
    else
        git commit -m "[Kernel-Tune] Winners extracted to $(basename "$OUT")"
        git push origin "$(git rev-parse --abbrev-ref HEAD)" \
            || echo "WARN: push failed; commit is local only" >&2
    fi
} 2>&1 | tee "$SCRIPT_LOG"

git add -f "$SCRIPT_LOG"
git commit -m "[Logs] Update build_kernel_registry script log for $LABEL" || true
