#!/usr/bin/env bash
# Launch both baseline cases through the logged workflow.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRANCH="$(git -C "$HERE" branch --show-current)"
[[ "$BRANCH" == bench ]] || { echo "Run this launcher from the bench branch" >&2; exit 2; }
exec bash "$HERE/../workflow/run.sh" "$HERE/job.yml" "$@"
