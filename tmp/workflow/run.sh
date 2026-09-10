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
result="${codes[0]}"
if [[ "$result" -eq 0 ]]; then
  outcome="Launcher completed (exit 0)"
else
  outcome="Launcher FAILED (exit $result)"
fi
printf '%s\n' "$outcome" | tee -a "$LOG" || exit 1
if gzip -c -- "$LOG" > "$LOG.gz.partial"; then
  mv -- "$LOG.gz.partial" "$LOG.gz" || exit 1
  echo "Compressed launcher log: $LOG.gz"
else
  echo "Launcher log compression failed; raw log retained at $LOG" >&2
  result=1
fi
exit "$result"
