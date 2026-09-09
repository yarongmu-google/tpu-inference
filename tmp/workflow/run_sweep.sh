#!/usr/bin/env bash
# Run the serving sweep through the description-based workflow.
set -euo pipefail
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$WORKFLOW_DIR/run.sh" "$WORKFLOW_DIR/sweep.yml" "$@"
