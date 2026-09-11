#!/usr/bin/env bash
# Submit the comparison through the existing logged workflow.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --recover ]]; then
  shift
  exec bash "$HERE/../workflow/run.sh" recover "$HERE/results" "$@"
fi
exec bash "$HERE/../workflow/run.sh" "$HERE/job.yml" "$@"
