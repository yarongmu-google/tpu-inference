#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a single committed .kernel JSON report.
#
# Usage:
#   tools/kernel/tuner/v1/extract_winners.sh [runlog_path]

set -euo pipefail

python3 -m tools.kernel.tuner.v1.extract_winners "${1:-}"

# Find the most recently written .kernel file
OUT=$(ls -t tmp/log/*.kernel 2>/dev/null | head -1)

if [ -z "$OUT" ]; then
    echo "Error: No .kernel file produced." >&2
    exit 1
fi

git add -f "$OUT"
if git diff --cached --quiet; then
    echo "Nothing new to commit."
    exit 0
fi
git commit -m "[Kernel-Tune] Winners extracted to $(basename "$OUT")"
git push origin "$(git rev-parse --abbrev-ref HEAD)" \
    || echo "WARN: push failed; commit is local only" >&2
