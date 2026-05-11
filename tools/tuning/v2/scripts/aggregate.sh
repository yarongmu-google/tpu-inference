#!/bin/bash
# aggregate.sh — production.{kernel,service} builder.
#
# Usage:
#   tools/tuning/v2/scripts/aggregate.sh <model_dir> [--topo T] [--model M] [--no-commit]
#
# Walks <model_dir> for per-workload .kernel and .service files,
# writes production.kernel and production.service. Auto-commits +
# pushes unless --no-commit.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model_dir> [--topo T] [--model M] [--no-commit]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.cli.aggregate "$@"
