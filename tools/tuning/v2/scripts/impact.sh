#!/bin/bash
# impact.sh — bottom-up impact-analysis CLI.
#
# Usage:
#   tools/tuning/v2/scripts/impact.sh by-kernel-key <field> <value>
#   tools/tuning/v2/scripts/impact.sh by-service-combo <field> <value>
#   tools/tuning/v2/scripts/impact.sh stale-tunes <current_sha>
#
# Delegates to `python3 -m tools.tuning.v2.cli.impact`. See that
# module's docstring for full subcommand semantics.

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <subcommand> [args...]" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$REPO_ROOT"
exec python3 -m tools.tuning.v2.cli.impact "$@"
