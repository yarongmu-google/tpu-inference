#!/usr/bin/env bash
# Capture a SHORT serving-time xprof window (a few dozen decode steps)
# from a running vllm server - the tool for the f32 step-time mystery:
# the 832-wide f32 step costs 124ms where component arithmetic predicts
# ~80; this capture splits the step and names the unmodeled ~40ms.
#
# Usage (two terminals + this):
#   1. Boot the server line with a profiler dir prefixed, e.g. line 4e:
#        VLLM_TORCH_PROFILER_DIR=tmp/xprof_832_f32 <line 4e command>
#   2. Start the mix client (CONC=832). Do NOT pass --profile (that
#      would capture the whole 25-min run).
#   3. When the run reaches its full-occupancy plateau, run:
#        bash tmp/capture_step_profile.sh tmp/xprof_832_f32
#      The script WAITS for the plateau on its own (polls the stats CSV
#      for Running >= THRESH), captures SECS seconds, then packs the
#      trace. Safe to start it right after the client.
#
#   THRESH=780 SECS=5 bash tmp/capture_step_profile.sh <dir>
#
# Reference ledger to compare against: the 512-global f32 step from the
# earlier xprof session (46ms: MoE 26.6 / GDN ~7.7 / attn ~1.5 / rest
# ~10). Afterwards: git add tmp/ && commit ("step profile 832.") && push.

set -euo pipefail

DIR=${1:?usage: capture_step_profile.sh <profiler-dir> [port]}
PORT=${2:-8000}
THRESH=${THRESH:-780}
SECS=${SECS:-5}
CSV=tmp/vllm_server_stats.csv

echo "waiting for plateau (Running >= ${THRESH} in ${CSV})..."
while :; do
  run=$(tail -1 "$CSV" 2>/dev/null | cut -d, -f4)
  run=${run%%.*}
  if [ -n "${run:-}" ] && [ "${run:-0}" -ge "$THRESH" ] 2>/dev/null; then
    break
  fi
  printf '  running=%s\r' "${run:-?}"
  sleep 10
done
echo "plateau reached (running=${run}); capturing ${SECS}s..."

curl -sf -X POST "localhost:${PORT}/start_profile" >/dev/null
sleep "$SECS"
curl -sf -X POST "localhost:${PORT}/stop_profile" >/dev/null
sleep 3   # let the trace flush

ls -lh "$DIR" || { echo "ERROR: nothing in $DIR - was the server booted with VLLM_TORCH_PROFILER_DIR=$DIR ?"; exit 1; }
tar -c "$DIR" | xz -9 -T0 > "${DIR%/}.tar.xz"
ls -lh "${DIR%/}.tar.xz"
echo "done: git add ${DIR%/}.tar.xz && git commit -m 'step profile.' && git push"
