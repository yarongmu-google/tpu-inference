#!/bin/bash
# project_kernel.sh — kernel projection entry point.
#
# Usage:
#   tools/tuning/v2/scripts/project_kernel.sh <workload> [--code-revision SHA] [--no-commit]
#
# Reads <workload>.kernel.raw/<sha>.jsonl, projects winners, writes
# <workload>.kernel. Auto-commits + pushes unless --no-commit.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload> [--code-revision SHA] [--no-commit]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.kernel.project "$@"
