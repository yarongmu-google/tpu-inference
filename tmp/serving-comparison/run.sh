#!/usr/bin/env bash
# Submit the comparison through the existing logged workflow.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/../../workflow/run.sh" "$HERE/job.yml" "$@"
