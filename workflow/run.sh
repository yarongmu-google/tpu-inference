#!/usr/bin/env bash
# One logged entry point for described experiments and saved executions.
set -uo pipefail
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
mkdir -p "$WORKFLOW_DIR/local/logs" || exit 1
LOG="$(mktemp "$WORKFLOW_DIR/local/logs/launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX.log")" || exit 1
echo "Launcher log: $LOG"
(
  set -e
  echo '+ prepare workflow-local Python environment'
  python3 "$WORKFLOW_DIR/bootstrap.py"
  echo '+ start workflow controller'
  PYTHONDONTWRITEBYTECODE=1 "$WORKFLOW_DIR/.venv/bin/python" "$WORKFLOW_DIR/controller.py" "$@"
) 2>&1 | tee "$LOG"
codes=("${PIPESTATUS[@]}")
[[ "${codes[1]}" -eq 0 ]] || exit 1
exit "${codes[0]}"
