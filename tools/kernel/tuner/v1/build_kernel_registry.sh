#!/bin/bash
# Extract per-case kernel-tuning winners from a tune_all_cases.sh runlog
# and write them to a .kernel JSON registry. Auto-commits and pushes the
# produced artifact, matching tune_all_cases.sh's progress-commit pattern.
# Set KERNEL_TUNER_NO_PUSH=1 to skip the git step (tests, CI, dry-runs).
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
# History note: this script was previously pure (no git) on the
# theory that the orchestrator would commit the artifact. In practice
# that left direct users (Phase 1 fast-track recipe) without an
# auto-commit, while tune_all_cases.sh on the same path DID auto-commit
# its runlog — inconsistent. The auto-commit lives here now; the
# orchestrator's commit_logs trap is idempotent against an already-
# committed file. Use KERNEL_TUNER_NO_PUSH=1 for testing/dry-runs.

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

# Auto-commit + push the produced .kernel. Idempotent against
# orchestrator's commit_logs trap (no-op if file is already committed).
if [ "${KERNEL_TUNER_NO_PUSH:-0}" = "1" ]; then
    exit 0
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
RUNLOG_BASENAME="$(basename "$RUNLOG" .txt)"
git add -f -- "$OUT" 2>/dev/null || true
if ! git diff --cached --quiet -- "$OUT"; then
    git commit -m "[Kernel-Tune] Update $(basename "$OUT") from $RUNLOG_BASENAME" -- "$OUT" || true
    git push origin "$BRANCH" || echo "WARN: push failed for $OUT" >&2
fi
