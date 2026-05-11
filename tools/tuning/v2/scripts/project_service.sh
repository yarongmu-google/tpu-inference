#!/bin/bash
# project_service.sh — service projection entry point.
#
# Usage:
#   tools/tuning/v2/scripts/project_service.sh <workload> [--service-revision SHA] [--no-commit]
#
# Reads <workload>.service.raw/<sha>.jsonl, projects per-objective
# winners, writes <workload>.service. Auto-commits + pushes unless
# --no-commit.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload> [--service-revision SHA] [--no-commit]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.service.project "$@"
