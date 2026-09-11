#!/usr/bin/env bash
# Run the controller independently of the terminal displaying its log.
set -euo pipefail
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$WORKFLOW_DIR/launcher.py" "$@"
