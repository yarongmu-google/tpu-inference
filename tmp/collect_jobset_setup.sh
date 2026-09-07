#!/usr/bin/env bash
# Run with vllm12 active; keep setup diagnostics even when a command fails.
set -uo pipefail
if [[ $# -ne 0 ]]; then
  echo "Usage: bash tmp/collect_jobset_setup.sh (with vllm12 active)" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p "$ROOT/tmp/vllm_logs" || exit 1
OUT="$(mktemp -d "$ROOT/tmp/vllm_logs/jobset-setup-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")" || exit 1
LOG="$OUT/setup.log"
: > "$LOG" || exit 1
failures=0

report() {
  printf '%s\n' "$*" | tee -a "$LOG"
}

capture() {
  local file="$1"
  shift
  local shown
  printf -v shown '%q ' "$@"
  report "+ $shown"
  "$@" 2>&1 | tee "$OUT/$file" | tee -a "$LOG"
  local codes=("${PIPESTATUS[@]}")
  report "Exit status: ${codes[0]}; output capture: ${codes[1]}, ${codes[2]}"
  if [[ "${codes[0]}" -ne 0 || "${codes[1]}" -ne 0 || "${codes[2]}" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  report ""
}

report "Setup diagnostics: $OUT"
report "Active environment: ${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-not identified; using python from PATH}}"
report ""
capture python.txt python -VV
capture packages.txt python -m pip list --format=freeze
capture cdk-create-help.txt cdk job create --help
report "Saved command output and errors to $LOG"
report "$failures setup check(s) failed. All available diagnostics were retained."
printf -v stage_command 'git add -- %q' "tmp/vllm_logs/$(basename "$OUT")"
report "After reviewing the diagnostics: $stage_command"
[[ "$failures" -eq 0 ]]
