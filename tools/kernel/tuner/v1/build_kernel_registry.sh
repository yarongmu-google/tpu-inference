#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a .kernel JSON registry. Auto-commits and pushes.
#
# Usage:
#   build_kernel_registry.sh <runlog_path> [out_path]
#
#   runlog_path  Path to a tune_all_*.txt runlog. Required.
#   out_path     Optional path for the .kernel output. Defaults to
#                tmp/log/<runlog-basename>.kernel. The orchestrator passes
#                cases/<topo>/<model>/production.kernel here so the file
#                lands where the sweeps `kernel_registry` field expects
#                it. Without this, the pipeline writes to tmp/log but the
#                sweep reads from cases/, and they never connect.

set -euo pipefail

RUNLOG="${1:-}"
OUT_PATH="${2:-}"

if [ -z "$RUNLOG" ]; then
    RUNLOG=$(ls -t tmp/log/tune_all_*.txt 2>/dev/null | head -1 || true)
fi
[ -n "$RUNLOG" ] && [ -f "$RUNLOG" ] \
    || { echo "Runlog not found: ${RUNLOG:-<none>}" >&2; exit 1; }

LABEL=$(basename "$RUNLOG" | sed 's/tune_all_//' | sed 's/\.txt//')
SCRIPT_LOG="tmp/log/script_build_kernel_registry_${LABEL}.txt"

# Build the python invocation, passing --out only when caller specified
# one. With no --out, the python script defaults to tmp/log/<label>.kernel
# and we keep the legacy commit path.
PY_ARGS=("$RUNLOG")
if [ -n "$OUT_PATH" ]; then
    PY_ARGS+=("--out" "$OUT_PATH")
    OUT="$OUT_PATH"
fi

{
    echo "=== Executing build_kernel_registry.py ==="
    python3 -m tools.kernel.tuner.v1.build_kernel_registry "${PY_ARGS[@]}"

    if [ -z "${OUT:-}" ]; then
        # No explicit out — fall back to "newest .kernel under tmp/log".
        OUT=$(ls -t tmp/log/*.kernel 2>/dev/null | head -1)
        if [ -z "$OUT" ]; then
            echo "Error: No .kernel file produced." >&2
            exit 1
        fi
    elif [ ! -f "$OUT" ]; then
        echo "Error: build_kernel_registry.py did not write $OUT" >&2
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
