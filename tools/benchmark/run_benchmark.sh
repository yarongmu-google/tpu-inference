#!/bin/bash
# Run a single vLLM serving benchmark on a TPU.
#
# Mirrors the methodology used by https://github.com/QiliangCui/bm-infra
# (the QC repo) for v7x perf runs — same vllm serve flags, same
# `vllm bench serve` driver, same metric set — but simplified for one-shot
# local use (no GCS upload, no binary search on rate, no daemonization).
#
# Usage:
#   tools/benchmark/run_benchmark.sh <case_file> [--result-tag TAG] [--rate RATE]
#
# A case_file is a shell `source`-able file under tools/benchmark/cases/
# that exports MODEL, MAX_NUM_SEQS, MAX_NUM_BATCHED_TOKENS,
# TENSOR_PARALLEL_SIZE, MAX_MODEL_LEN, DATASET, INPUT_LEN, OUTPUT_LEN, and
# optionally NUM_PROMPTS / LONG_PREFILL_TOKEN_THRESHOLD / RPA_P_BLOCK_SIZES.
#
# Required environment (set once per shell):
#   VLLM_DIR        — path to a vLLM source checkout (we read its
#                     benchmarks/sonnet.txt). Defaults to ../vllm.
#   DOWNLOAD_DIR    — HF cache dir. Defaults to $HOME/hf-cache.
#   HF_TOKEN        — HuggingFace token (only needed for gated models).

set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <case_file> [--result-tag TAG] [--rate RATE]

  case_file       Path to a tools/benchmark/cases/<name>.workload file.
  --result-tag    Label appended to the result directory name. Defaults
                  to a timestamp.
  --rate          Request rate (--request-rate). Defaults to "inf"
                  (max throughput, no rate limit).

Examples:
  $0 tools/benchmark/cases/llama3_8b_v7x_balanced.workload
  $0 tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.workload --result-tag baseline
  $0 tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.workload --result-tag opt --rate 5
EOF
    exit 1
}

[ $# -lt 1 ] && usage
CASE_FILE="$1"; shift

RESULT_TAG=""
# REQUEST_RATE may be set in the env (so sweep.py can drive it as a sweep
# axis like any other knob). --rate on the command line still overrides;
# default 'inf' (max throughput, no rate limit).
: "${REQUEST_RATE:=inf}"
while [ $# -gt 0 ]; do
    case "$1" in
        --result-tag) RESULT_TAG="$2"; shift 2 ;;
        --rate)       REQUEST_RATE="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown arg: $1"; usage ;;
    esac
done

[ -f "$CASE_FILE" ] || { echo "case file not found: $CASE_FILE" >&2; exit 1; }

# Workload-path guard — duplicated from run_pipeline.sh and
# tune_all_cases.sh because run_benchmark.sh can be invoked directly
# (its own --help advertises that). Without this, `set -a; source
# "$CASE_FILE"` below would execute arbitrary bash from any path the
# user passes. python3 os.path.realpath resolves SYMLINKS so a
# symlink under cases/ pointing to /tmp/evil.sh is also rejected.
CASE_FILE_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "$CASE_FILE")
CASES_ROOT_REAL=$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "tools/benchmark/cases")
case "$CASE_FILE_REAL" in
    "$CASES_ROOT_REAL"/*) ;;
    *)
        echo "Error: workload (after symlink resolution) must live under tools/benchmark/cases/ (sourced as bash)." >&2
        echo "Got: $CASE_FILE_REAL" >&2
        exit 1
        ;;
esac

set -a
# shellcheck disable=SC1090
source "$CASE_FILE"
set +a

# Validate / derive MAX_MODEL_LEN. For our fixed-shape sonnet
# workloads it MUST equal INPUT_LEN + OUTPUT_LEN — anything bigger
# over-allocates HBM and silently caps effective max_num_seqs (vllm
# reduces concurrency to fit max_num_seqs * cdiv(max_model_len,
# block_size) pages into the pool); anything smaller truncates
# requests. If the workload omits MAX_MODEL_LEN, compute it; if set
# to a different value, fail-fast with a pointed error.
EXPECTED_MML=$(( INPUT_LEN + OUTPUT_LEN ))
if [ -z "${MAX_MODEL_LEN:-}" ]; then
    MAX_MODEL_LEN=$EXPECTED_MML
    export MAX_MODEL_LEN
elif [ "$MAX_MODEL_LEN" -ne "$EXPECTED_MML" ]; then
    echo "Error: MAX_MODEL_LEN=$MAX_MODEL_LEN in $(basename "$CASE_FILE") does not match INPUT_LEN+OUTPUT_LEN=$EXPECTED_MML." >&2
    echo "Set MAX_MODEL_LEN=$EXPECTED_MML or remove the line to auto-compute." >&2
    exit 1
fi

# NUM_PROMPTS comes from the case file (every case file sets it via :=).
# DOWNLOAD_DIR and VLLM_DIR are deployment config, not workload, so their
# defaults live here rather than in case files. Both forms end with the
# variable assigned. The case files use the `: "${VAR:=default}"` idiom
# because they need `set -a` to export the assignment side-effect of `:=`.
# The script uses `VAR="${VAR:-default}"` because there's no exporting
# question and the form is more familiar.
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/hf-cache}"
VLLM_DIR="${VLLM_DIR:-../vllm}"

CASE_NAME="$(basename "$CASE_FILE" .workload)"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${RESULT_TAG:-$TS}"
# RESULT_DIR is overridable from the env so sweep.py can direct each
# combo's output into its hashed combo dir (tmp/bench_<case>_<sweep>/<id>).
: "${RESULT_DIR:=tmp/bench_${CASE_NAME}_${TAG}}"
mkdir -p "$RESULT_DIR"
# Normalize to absolute. Downstream we generate sonnet_4x.txt inside
# $RESULT_DIR and pass it to vllm bench (which `cd`s to $VLLM_DIR), so
# the path must survive a working-directory change.
case "$RESULT_DIR" in
    /*) ;;
    *)  RESULT_DIR="$PWD/$RESULT_DIR" ;;
esac

VLLM_LOG="$RESULT_DIR/vllm.log"
BM_LOG="$RESULT_DIR/bench.log"
META_FILE="$RESULT_DIR/meta.txt"
METRICS_FILE="$RESULT_DIR/metrics.txt"

# Canonicalize case_file to an absolute path so two combos that
# referenced the same file from different CWDs produce identical
# meta.case_file values for downstream comparison.
#
# Two assignments, not one. The natural one-liner form
#   CASE_FILE_ABS="$(cd ... && pwd -P)/$(basename "$CASE_FILE")" || ...
# would NOT catch a cd failure: per POSIX, the exit status of an
# assignment with multiple command substitutions is the status of
# the LAST one performed (here `basename`, always 0). The `||` would
# be dead code. Splitting into a `cd` assignment first and the path
# join second makes the cd-failure case have its own statement, so
# `||` fires correctly. The prior `[ -f "$CASE_FILE" ]` check at
# line 60 makes a cd failure unreachable today, but this is the
# point of the insurance — to hold up if a future edit weakens or
# moves that check.
CASE_DIR_ABS="$(cd "$(dirname "$CASE_FILE")" && pwd -P)" \
    || { echo "ERROR: failed to canonicalize CASE_FILE=$CASE_FILE" >&2; exit 1; }
CASE_FILE_ABS="$CASE_DIR_ABS/$(basename "$CASE_FILE")"

{
    echo "case_file=$CASE_FILE_ABS"
    echo "case_name=$CASE_NAME"
    echo "model=$MODEL"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
    echo "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "dataset=$DATASET"
    echo "input_len=$INPUT_LEN"
    echo "output_len=$OUTPUT_LEN"
    echo "num_prompts=$NUM_PROMPTS"
    echo "request_rate=$REQUEST_RATE"
    # Sentinel 'default' (consistent across knobs) means: not explicitly
    # set by the case file or the caller; vllm picks its built-in default.
    echo "block_size=${BLOCK_SIZE:-default}"
    echo "long_prefill_token_threshold=${LONG_PREFILL_TOKEN_THRESHOLD:-default}"
    echo "rpa_d_block_sizes=${RPA_D_BLOCK_SIZES:-default}"
    echo "rpa_p_block_sizes=${RPA_P_BLOCK_SIZES:-default}"
    echo "rpa_m_block_sizes=${RPA_M_BLOCK_SIZES:-default}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "timestamp=$TS"
} | tee "$META_FILE"
echo "----"
echo "Results: $RESULT_DIR"

if [ "$DATASET" = "sonnet" ]; then
    [ -f "$VLLM_DIR/benchmarks/sonnet.txt" ] \
        || { echo "ERROR: $VLLM_DIR/benchmarks/sonnet.txt not found; set VLLM_DIR." >&2; exit 1; }
    # Generate the sonnet repeats file inside the per-run RESULT_DIR
    # rather than mutating $VLLM_DIR/benchmarks/. Two reasons:
    #   1. We dont own $VLLM_DIR; mutating it is bad hygiene.
    #   2. Concurrent runs (parallel sweeps) would race on the shared
    #      file and silently corrupt each other.
    #
    # Repeat factor: vllm bench sonnet samples a window of INPUT_LEN
    # tokens from this file. The file needs >= INPUT_LEN tokens or
    # bench errors out. Single sonnet.txt is ~1000 tokens, so 16x
    # gives ~16000 tokens — comfortably above any INPUT_LEN we plan
    # to test today.
    #
    # File is named "sonnet_4x.txt" for backwards-compat with the
    # gitignore pattern `tmp/bench_*/*/sonnet_4x.txt`. Variable
    # renamed from the old SONNET_4X to SONNET_INPUT_FILE so the
    # name no longer lies about the repeat count.
    SONNET_REPEATS=16
    SONNET_INPUT_FILE="$RESULT_DIR/sonnet_4x.txt"
    : > "$SONNET_INPUT_FILE"
    for _ in $(seq 1 "$SONNET_REPEATS"); do
        cat "$VLLM_DIR/benchmarks/sonnet.txt" >> "$SONNET_INPUT_FILE"
    done
fi

SERVE_ARGS=(
    --seed 42
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --no-enable-prefix-caching
    --download-dir "$DOWNLOAD_DIR"
    --max-model-len "$MAX_MODEL_LEN"
)
if [ -n "${LONG_PREFILL_TOKEN_THRESHOLD:-}" ]; then
    SERVE_ARGS+=(--long-prefill-token-threshold "$LONG_PREFILL_TOKEN_THRESHOLD")
fi
if [ -n "${BLOCK_SIZE:-}" ]; then
    SERVE_ARGS+=(--block-size "$BLOCK_SIZE")
fi

# Tear down ONLY our vllm — never `pkill -f vllm`, that nukes neighbors.
# We launch in its own process group (via `set -m`) so a single
# negative-PID signal hits the whole tree: the vllm CLI parent, the
# engine subprocess, and the per-worker subprocesses. Otherwise the
# SIGKILL fallback kills the parent only, and the workers get reparented
# to init holding TPU resources — fatal for the next combo in a sweep.
clean_up() {
    if [ -z "${VLLM_PID:-}" ]; then
        return 0
    fi
    # Belt-and-suspenders: TERM the whole process group (if we have a
    # safe PGID) AND the leader by PID. In the happy path (`set -m`
    # took effect, VLLM_PGID is non-empty and != $$), the group signal
    # does the work and the per-PID signal is a no-op (parent already
    # dead). In the silent-no-op path (set -m didnt take, VLLM_PGID
    # cleared to empty above), only the per-PID signal fires —
    # degrading cleanly to pre-process-group behavior (workers may
    # orphan, but parent reliably dies, and crucially we do NOT
    # signal this scripts own process group by mistake).
    if [ -n "${VLLM_PGID:-}" ]; then
        kill -- -"$VLLM_PGID" 2>/dev/null || true
    fi
    kill "$VLLM_PID" 2>/dev/null || true
    # Wait up to 5 s for the tree to wind down. The parent's `wait()`
    # for its workers means leader-exit lags worker-exit, so this
    # really is "tree quiesces" not "leader gone".
    for _ in 1 2 3 4 5; do
        kill -0 "$VLLM_PID" 2>/dev/null || break
        sleep 1
    done
    # KILL — same belt-and-suspenders pair.
    if [ -n "${VLLM_PGID:-}" ]; then
        kill -9 -- -"$VLLM_PGID" 2>/dev/null || true
    fi
    kill -9 "$VLLM_PID" 2>/dev/null || true
}
trap clean_up EXIT

echo "Starting vllm server (log: $VLLM_LOG) ..."
# `set -m` (job control) puts the next backgrounded command into its
# own process group; `$!` is then both PID and PGID of the new group.
# In constrained environments (some container runtimes without /dev/tty)
# `set -m` may emit stderr noise or be silently no-op.
set -m 2>/dev/null || true
VLLM_USE_V1=1 vllm serve "$MODEL" "${SERVE_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
set +m 2>/dev/null || true

# Capture the actual process-group ID via ps. If `set -m` took effect
# this is the same as $VLLM_PID (parent is its own PGID); if `set -m`
# was a silent no-op the captured PGID is the bash shells own group,
# and the cleanup `kill -- -$VLLM_PGID` would signal the wrong group
# (potentially this script itself). Capture VLLM_PGID empty in that
# case so the cleanup falls back to the per-PID kill only.
VLLM_PGID=$(ps -o pgid= -p "$VLLM_PID" 2>/dev/null | tr -d ' ' || true)
if [ -n "$VLLM_PGID" ] && [ "$VLLM_PGID" = "$$" ]; then
    # PGID matches THIS shells group — `set -m` was a no-op. Clear so
    # the cleanup does not signal ourselves.
    VLLM_PGID=""
fi

# Readiness probe: scrape vllm's log for the FastAPI startup-complete
# string. NOT a /health HTTP poll — that would be more robust but
# requires port plumbing and a curl dependency. If vllm changes this
# log message in a future version, this loop silently times out;
# upgrade to a /health probe is the natural follow-up.
#
# Timeout in SECONDS (default 1200 = 20 minutes). Override via
# READINESS_TIMEOUT_SECONDS in the env. The default suits Llama 3 8B
# on v7x; bigger models or first-time JIT compiles may need more.
SERVER_READY_MARKER="Application startup complete"
: "${READINESS_TIMEOUT_SECONDS:=1200}"
READINESS_POLL_INTERVAL_S=10
READINESS_MAX_ITERS=$(( READINESS_TIMEOUT_SECONDS / READINESS_POLL_INTERVAL_S ))
echo "Waiting up to ${READINESS_TIMEOUT_SECONDS}s for server startup ..."
SERVER_READY=0
for i in $(seq 1 "$READINESS_MAX_ITERS"); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vllm exited early. tail of $VLLM_LOG:" >&2
        tail -40 "$VLLM_LOG" >&2
        exit 1
    fi
    if grep -Fq "$SERVER_READY_MARKER" "$VLLM_LOG"; then
        echo "  ready after $((i * READINESS_POLL_INTERVAL_S))s"
        SERVER_READY=1
        break
    fi
    sleep "$READINESS_POLL_INTERVAL_S"
done
if [ "$SERVER_READY" -ne 1 ]; then
    echo "ERROR: vllm did not become ready within ${READINESS_TIMEOUT_SECONDS}s" \
         "(no '$SERVER_READY_MARKER' in $VLLM_LOG)" >&2
    tail -40 "$VLLM_LOG" >&2
    exit 1
fi

BM_ARGS=(
    --backend vllm
    --model "$MODEL"
    --request-rate "$REQUEST_RATE"
    --dataset-name "$DATASET"
    --num-prompts "$NUM_PROMPTS"
    --percentile-metrics ttft,tpot,itl,e2el
    --ignore-eos
)
case "$DATASET" in
    sonnet)
        BM_ARGS+=(
            --dataset-path "$SONNET_INPUT_FILE"
            --sonnet-input-len  "$INPUT_LEN"
            --sonnet-output-len "$OUTPUT_LEN"
        ) ;;
    random)
        BM_ARGS+=(
            --random-input-len  "$INPUT_LEN"
            --random-output-len "$OUTPUT_LEN"
        ) ;;
    *)
        echo "ERROR: unsupported dataset '$DATASET'" >&2; exit 1 ;;
esac

BM_ERR_LOG="$BM_LOG.stderr"
echo "Running benchmark (stdout: $BM_LOG, stderr: $BM_ERR_LOG) ..."
# Keep stdout (metric tables) and stderr (warnings, tracebacks) in
# separate files. parse_bench_log only consumes $BM_LOG so warnings
# never end up in metrics.txt.
BENCH_START_S=$(date +%s)
if ! ( cd "$VLLM_DIR" && vllm bench serve "${BM_ARGS[@]}" ) \
        > "$BM_LOG" 2> "$BM_ERR_LOG"; then
    echo "ERROR: bench failed. tail of $BM_ERR_LOG:" >&2
    tail -40 "$BM_ERR_LOG" >&2
    exit 1
fi
BENCH_DURATION_S=$(( $(date +%s) - BENCH_START_S ))
# Append wall-clock duration to meta so compare.py can show it
# alongside the throughput / latency columns. Useful for spotting
# tail outliers and timeout-prone combos.
#
# Scope: this is the duration of `vllm bench serve` ONLY. It excludes
# server startup (model load + JIT compile + warmup), which is the
# variable, minutes-long part of each combo. So compare.py's dur_s
# column is "bench-time difference between combos" not "total
# wall-clock per combo". For the latter, look at the gap between
# meta.timestamp values across combos in a sweep, or extend this
# script to write end_timestamp.
#
# Granularity is 1s (date +%s). Fine for the 100+ prompt benches we
# run today; an ultra-short stub bench would round to 0–1s and look
# broken in the table. Switch to date +%s.%N + awk if that becomes
# real.
echo "bench_duration_seconds=$BENCH_DURATION_S" >> "$META_FILE"

echo "==== Summary ===="
grep -E "Request throughput|Output token throughput|Total Token throughput|Mean.*\(ms\):|Median.*\(ms\):|P99.*\(ms\):" "$BM_LOG" || true
echo "===="

# Metric extraction lives in Python (tools/benchmark/parse_bench_log.py)
# so it gets unit-tested independently of this shell driver. Invoke
# by absolute path rather than `python3 -m tools.benchmark.parse_bench_log`
# — the -m form requires the repo root to be on sys.path / CWD, which
# isn't guaranteed when sweep.py launches us as a subprocess from an
# arbitrary working directory.
# `pwd -P` resolves symlinks in the CWD path so a dir-symlink (e.g.
# `current` → versioned dir) doesn't strand us next to a non-existent
# parse_bench_log.py. Doesn't handle the file-symlink case
# (/usr/local/bin/run_benchmark.sh → /repo/.../run_benchmark.sh) —
# that'd need `readlink -f "$0"`, GNU-only; out of scope.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
python3 "$SCRIPT_DIR/parse_bench_log.py" "$BM_LOG" > "$METRICS_FILE"

echo "Saved metrics to $METRICS_FILE"
