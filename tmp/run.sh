#!/usr/bin/env bash
# One CPU-VM command: build/reuse, publish, submit, poll, and retrieve diagnostics.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
mkdir -p "$SCRIPT_DIR/vllm_logs" || exit 1
LOG="$(mktemp "$SCRIPT_DIR/vllm_logs/jobset-launch-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX.log")" || exit 1
echo "Launcher log: $LOG"
python3 "$SCRIPT_DIR/jobset_workflow.py" "$@" 2>&1 | tee "$LOG"
codes=("${PIPESTATUS[@]}")
[[ "${codes[1]}" -eq 0 ]] || exit 1
exit "${codes[0]}"
