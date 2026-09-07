#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Sweep: restart the server per concurrency.
# Usage: bash sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8000}"
READY_TIMEOUT="${READY_TIMEOUT:-5400}"   # 90 min (covers a cold compile)

# Remaining EP points and matched 4i comparisons; successful points are omitted.
# Order: the large 1k/8k pair first.
RUNS=(
  "1024:8192:256:DP8_EP"
  "1024:8192:256:4I"
  "1024:8192:128:4I"
  "1024:1024:256:DP4TP2_EP"
)

stop_server() {
  pkill -TERM -f "vllm serve Qwen/Qwen3.5" 2>/dev/null || true
  pkill -TERM -f "VLLM::EngineCore" 2>/dev/null || true
  sleep 6
  pkill -KILL -f "vllm serve Qwen/Qwen3.5" 2>/dev/null || true
  pkill -KILL -f "VLLM::EngineCore" 2>/dev/null || true
  for _ in $(seq 1 30); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://0.0.0.0:${PORT}/health" 2>/dev/null)" = "200" ] || break
    sleep 2
  done
}

start_server() {  # $1 = SHARDING (CONC/ISL/OSL come from the exported env)
  mkdir -p "${SERVER_LOG_DIR:-/tmp}"
  SERVER_LOG="${SERVER_LOG_DIR:-/tmp}/qwen3.5_sweep_server_${1}_isl${ISL}_osl${OSL}_conc${CONC}_$(date +%Y%m%d-%H%M%S)_${RUN_ATTEMPT:-local}.log"
  echo "--- starting server SHARDING=$1 ISL=$ISL OSL=$OSL CONC=$CONC (log: $SERVER_LOG) ---"
  echo "CFG sharding=$1 commit=$(git -C "$SCRIPT_DIR" rev-parse HEAD)" >> "$SERVER_LOG"
  SHARDING="$1" bash "${SCRIPT_DIR}/server.sh" >> "$SERVER_LOG" 2>&1 &
  SERVER_PID=$!
  local waited=0
  until grep -q "Application startup complete" "$SERVER_LOG" 2>/dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "ERROR: server exited during startup (see $SERVER_LOG)" >&2; return 1; }
    sleep 5; waited=$((waited + 5))
    [ "$waited" -ge "$READY_TIMEOUT" ] && { echo "ERROR: server not ready after ${READY_TIMEOUT}s" >&2; return 1; }
  done
  echo "--- server ready ---"
}

trap stop_server EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

RESULT_ROOT="${RESULT_DIR:-/tmp/qwen3.5-inferencex-bench}"

completed_result() {
  python3 - "$RESULT_DIR" "$ISL" "$OSL" "$CONC" "$SHARDING" <<'PY_RESULT'
import json
import sys
from pathlib import Path

root, isl, osl, conc, sharding = sys.argv[1:]
expected = int(conc) * 10
directories = [Path(root)]
# Only the outstanding DP8 point has unambiguous legacy EP results.
# The legacy 1k/1k C256 directory also contained a separate 4i result.
if sharding == "DP8_EP":
    directories.append(Path(root).parent)
pattern = f"qwen3.5_isl{isl}_osl{osl}_conc{conc}_*.json"
for path in sorted((p for d in directories for p in d.glob(pattern)), reverse=True):
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        continue
    if (isinstance(data, dict)
            and data.get("num_prompts") == expected
            and data.get("completed") == expected
            and data.get("total_output_tokens", 0) > 0):
        print(path)
        sys.exit(0)
sys.exit(1)
PY_RESULT
}

failures=0
for point in "${RUNS[@]}"; do
  IFS=: read -r ISL OSL CONC SHARDING <<< "$point"
  export ISL OSL CONC SHARDING
  export RESULT_DIR="$RESULT_ROOT/$SHARDING"
  if done_file=$(completed_result); then
    echo "########## SKIP (done): $point -> $done_file ##########"
    continue
  fi
  stop_server
  if ! start_server "$SHARDING"; then
    failures=$((failures + 1))
    continue
  fi
  echo "########## ISL=$ISL OSL=$OSL CONC=$CONC SHARDING=$SHARDING ##########"
  if ! bash "${SCRIPT_DIR}/bench.sh"; then
    echo "FAIL: client exited unsuccessfully for $point" >&2
    failures=$((failures + 1))
  elif ! completed_result >/dev/null; then
    echo "FAIL: incomplete or zero-output result for $point" >&2
    failures=$((failures + 1))
  fi
done
[ "$failures" -eq 0 ] || { echo "ERROR: $failures point(s) failed; resume to retry." >&2; exit 1; }
echo "########## sweep complete ##########"
