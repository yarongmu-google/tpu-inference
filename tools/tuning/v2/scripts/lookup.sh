#!/bin/bash
# lookup.sh — top-down deploy lookup.
#
# Usage:
#   tools/tuning/v2/scripts/lookup.sh <workload> [--objective NAME]
#
# Prints K=V lines for the merged deploy-time env. Default objective:
# throughput_max. Use `eval $(tools/tuning/v2/scripts/lookup.sh foo.workload)`
# to set the env in the current shell.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <workload> [--objective NAME]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.cli.lookup "$@"
