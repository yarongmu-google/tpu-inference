#!/bin/bash
# Reproduce + surface the REAL exception behind vllm's
# "Engine core initialization failed" wrapper at large
# MAX_NUM_BATCHED_TOKENS (MNB) — first known failing value on
# Llama 3 8B / TPU v7x is MNB=262144.
#
# Why this script exists: the wrapper at vllm/v1/engine/utils.py:1178
# raises `RuntimeError: Engine core initialization failed` but does
# NOT re-raise the upstream exception from the child engine-core
# process. The real exception is in the child's stderr, which vllm
# logs but the parent never surfaces. We launch vllm ourselves,
# capture full stderr to a file, then grep the Traceback out at the
# end.
#
# Usage (on TPU VM):
#   tools/debug_vllm_engine_init.sh [MNB]
#
# Examples:
#   tools/debug_vllm_engine_init.sh           # default: MNB=262144
#   tools/debug_vllm_engine_init.sh 524288    # next bisection point
#   tools/debug_vllm_engine_init.sh 131072    # known-good (sanity check)
#
# Env overrides (defaults shown):
#   MAX_MODEL_LEN=8192
#   MAX_NUM_SEQS=1000
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct
#   TIMEOUT_S=240   # wait this long before giving up + grabbing log
#   PROD_ENV=tmp/log/prod_env_prefill_heavy.sh   # sourced if exists,
#                                                  so block sizes etc. apply
#
# Output (each run gets its own timestamped dir so runs don't clobber):
#   /tmp/vllm_debug/mnb_<MNB>_<stamp>/vllm.log     — full vllm stdout+stderr
#   /tmp/vllm_debug/mnb_<MNB>_<stamp>/failure.txt  — extracted Traceback

# NOT `set -e`: we EXPECT vllm to fail; -e would abort right after
# the failure instead of letting us extract it.
set -uo pipefail

MNB="${1:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1000}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TIMEOUT_S="${TIMEOUT_S:-240}"
PROD_ENV="${PROD_ENV:-tmp/log/prod_env_prefill_heavy.sh}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="/tmp/vllm_debug/mnb_${MNB}_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/vllm.log"
FAIL="$OUT_DIR/failure.txt"

echo "=== vllm engine init debug ==="
echo "MNB:             $MNB"
echo "MAX_MODEL_LEN:   $MAX_MODEL_LEN"
echo "MAX_NUM_SEQS:    $MAX_NUM_SEQS"
echo "MODEL:           $MODEL"
echo "TIMEOUT_S:       $TIMEOUT_S"
echo "OUT_DIR:         $OUT_DIR"
echo

# Source the prod env first (sets kernel block sizes etc.), THEN
# override the value we're testing so it wins regardless of what
# the prod env had.
if [ -f "$PROD_ENV" ]; then
    echo "Sourcing prod env: $PROD_ENV"
    set -a
    # shellcheck disable=SC1090
    source "$PROD_ENV"
    set +a
else
    echo "(no prod env at $PROD_ENV — using vllm + tpu_inference defaults)"
fi
export MAX_NUM_BATCHED_TOKENS="$MNB"
# Pin LPTT == MNB so a separate LPTT-side guard can't be the cause —
# we want to isolate the MNB cap, not stack two unknowns.
export LONG_PREFILL_TOKEN_THRESHOLD="$MNB"

echo "=== Launching vllm serve ==="
echo "(full stdout+stderr -> $LOG)"
vllm serve "$MODEL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MNB" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vllm PID: $VLLM_PID"

# Poll for one of three outcomes:
#   (a) wrapper logs "Engine core initialization failed"  → got our cause
#   (b) "Uvicorn running on" appears                     → MNB actually OK
#   (c) timeout (vllm hung)                              → also informative
echo "=== Waiting up to ${TIMEOUT_S}s for outcome ==="
WAITED=0
OUTCOME="timeout"
while [ "$WAITED" -lt "$TIMEOUT_S" ]; do
    if grep -q "Engine core initialization failed" "$LOG" 2>/dev/null; then
        OUTCOME="engine_init_failed"
        break
    fi
    if grep -q "Uvicorn running on" "$LOG" 2>/dev/null; then
        OUTCOME="started"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        OUTCOME="process_exited"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "Outcome: $OUTCOME (after ${WAITED}s)"

# Kill the vllm process group. vllm forks worker processes; just
# killing the leader leaks them and they keep the TPU attached,
# which breaks the next run. Walk parent → group → kill -9 fallback.
echo "=== Tearing down vllm ==="
pkill -P "$VLLM_PID" 2>/dev/null || true
kill "$VLLM_PID" 2>/dev/null || true
sleep 2
pkill -9 -P "$VLLM_PID" 2>/dev/null || true
kill -9 "$VLLM_PID" 2>/dev/null || true

# Extract the failure block. The Traceback in the log BEFORE
# the wrapper line is the upstream exception we actually want.
echo "=== Extracting failure -> $FAIL ==="
{
    echo "# === Lines around first Traceback ==="
    grep -n -B 2 -A 80 "^Traceback\|: Traceback" "$LOG" | head -200 || true
    echo
    echo "# === Lines around the engine-init wrapper ==="
    grep -n -B 50 -A 5 "Engine core initialization failed" "$LOG" \
        | head -200 || true
    echo
    echo "# === Last 50 lines of vllm.log ==="
    tail -50 "$LOG"
} > "$FAIL"

echo
echo "=========================================="
cat "$FAIL"
echo "=========================================="
echo
echo "Full log:  $LOG"
echo "Failure:   $FAIL"
echo "Outcome:   $OUTCOME"
