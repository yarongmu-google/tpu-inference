#!/bin/bash
# Build a service-registry winners file from a sweep result directory
# and auto-commit + push it. Wrapper around
# `python3 -m tools.benchmark.build_service_registry`; mirrors
# tools/kernel/tuner/v1/build_kernel_registry.sh.
# Set KERNEL_TUNER_NO_PUSH=1 to skip the git step (tests, CI, dry-runs).
#
# Usage:
#   build_service_registry.sh <sweep_dir> [extra build_service_registry flags...]
#
# Requires --export-production <path> in the extra flags so the wrapper
# knows what to commit.

set -euo pipefail

# Find --export-production in the flag list (need it before invoking
# python so we can commit the right file afterward). Preserve all
# args for the python call.
OUT=""
i=1
while [ "$i" -le "$#" ]; do
    eval "arg=\${$i}"
    case "$arg" in
        --export-production)
            j=$((i + 1))
            eval "OUT=\${$j}"
            break
            ;;
        --export-production=*)
            OUT="${arg#--export-production=}"
            break
            ;;
    esac
    i=$((i + 1))
done

python3 -m tools.benchmark.build_service_registry "$@"

if [ -z "$OUT" ] || [ ! -f "$OUT" ]; then
    exit 0
fi

if [ "${KERNEL_TUNER_NO_PUSH:-0}" = "1" ]; then
    exit 0
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
git add -f -- "$OUT" 2>/dev/null || true
if ! git diff --cached --quiet -- "$OUT"; then
    git commit -m "[Service] Update $(basename "$OUT") from service sweep" -- "$OUT" || true
    git push origin "$BRANCH" || echo "WARN: push failed for $OUT" >&2
fi
