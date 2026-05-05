#!/bin/bash
# Thin shell wrapper around `python -m tools.benchmark.sweep`.
#
# Usage:
#   tools/benchmark/sweep.sh <spec.json> [extra sweep.py flags...]
#
# What it does for you so the call site is short:
#   - mkdirs tmp/log/ and picks a timestamped runlog filename
#   - sets --auto-commit-every 1 so every combo's metrics.txt + meta.txt
#     + the runlog get committed and pushed as the sweep progresses
#     (so a remote reader can follow along, and a crash leaves the
#     last good state on origin)
#   - tees the console to the runlog so a hang/crash leaves a trail
#
# Any additional flags after <spec.json> are passed through to
# sweep.py (e.g. --no-push, --base-dir, --script). Don't pass
# --runlog or --auto-commit-every here — argparse will reject the
# duplicate. Call the python module directly if you need to override.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <spec.json> [extra sweep.py flags...]

  spec.json   Sweep spec JSON (e.g. tools/benchmark/sweeps/<name>.json).
EOF
    exit 1
}

[ $# -lt 1 ] && usage
case "$1" in
    -h|--help) usage ;;
esac

SPEC="$1"; shift
[ -f "$SPEC" ] || { echo "spec not found: $SPEC" >&2; exit 1; }

mkdir -p tmp/log
SPEC_NAME="$(basename "$SPEC" .json)"
RUNLOG="tmp/log/${SPEC_NAME}_$(date +%Y%m%d_%H%M%S).txt"

# `pipefail` makes the pipeline exit non-zero if python fails;
# otherwise tee's 0 would mask a sweep failure.
echo "Runlog: $RUNLOG"
python3 -m tools.benchmark.sweep "$SPEC" \
    --auto-commit-every 1 \
    --runlog "$RUNLOG" \
    "$@" \
    2>&1 | tee "$RUNLOG"
