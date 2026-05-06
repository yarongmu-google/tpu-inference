#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a .kernel JSON registry. PURE TOOL — no git side
# effects. The orchestrator (run_pipeline.sh) is responsible for any
# commit/push of the produced artifact.
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
#
# Why pure: previously this script tee'd output to a SCRIPT_LOG,
# git-add-committed-pushed both the .kernel file and the log, and
# returned. Mixing artifact production with side effects made the tool
# hard to test in isolation, awkward on rebase divergence (it would
# emit a local-only commit + WARN), and impossible to dry-run from a
# Python wrapper. Now the script just produces the artifact and
# prints its path on stdout; the orchestrator stages it.

set -euo pipefail

RUNLOG="${1:-}"
OUT_PATH="${2:-}"

if [ -z "$RUNLOG" ]; then
    RUNLOG=$(ls -t tmp/log/tune_all_*.txt 2>/dev/null | head -1 || true)
fi
[ -n "$RUNLOG" ] && [ -f "$RUNLOG" ] \
    || { echo "Runlog not found: ${RUNLOG:-<none>}" >&2; exit 1; }

# Build the python invocation, passing --out only when caller specified
# one. With no --out, the python script defaults to tmp/log/<label>.kernel.
PY_ARGS=("$RUNLOG")
if [ -n "$OUT_PATH" ]; then
    PY_ARGS+=("--out" "$OUT_PATH")
fi

python3 -m tools.kernel.tuner.v1.build_kernel_registry "${PY_ARGS[@]}"

# Resolve the actually-written path so the orchestrator (or the user)
# knows what to stage. With --out, we trust that path; without, we
# fall back to "newest .kernel under tmp/log".
if [ -n "$OUT_PATH" ]; then
    OUT="$OUT_PATH"
else
    OUT=$(ls -t tmp/log/*.kernel 2>/dev/null | head -1)
fi

if [ -z "${OUT:-}" ] || [ ! -f "$OUT" ]; then
    echo "Error: build_kernel_registry.py did not produce a .kernel file." >&2
    exit 1
fi

# Print the produced path on stdout's last line so callers can capture
# it via `OUT=$(build_kernel_registry.sh ... | tail -1)`.
echo "$OUT"
