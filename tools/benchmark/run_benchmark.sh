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

  case_file       Path to a tools/benchmark/cases/<name>.env file.
  --result-tag    Label appended to the result directory name. Defaults
                  to a timestamp.
  --rate          Request rate (--request-rate). Defaults to "inf"
                  (max throughput, no rate limit).

Examples:
  $0 tools/benchmark/cases/llama3_8b_v7x_balanced.env
  $0 tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env --result-tag baseline
  $0 tools/benchmark/cases/llama3_8b_v7x_prefill_heavy.env --result-tag opt --rate 5
EOF
    exit 1
}

[ $# -lt 1 ] && usage
CASE_FILE="$1"; shift

RESULT_TAG=""
REQUEST_RATE="inf"
while [ $# -gt 0 ]; do
    case "$1" in
        --result-tag) RESULT_TAG="$2"; shift 2 ;;
        --rate)       REQUEST_RATE="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            echo "Unknown arg: $1"; usage ;;
    esac
done

[ -f "$CASE_FILE" ] || { echo "case file not found: $CASE_FILE" >&2; exit 1; }
set -a
# shellcheck disable=SC1090
source "$CASE_FILE"
set +a

NUM_PROMPTS="${NUM_PROMPTS:-1000}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-$HOME/hf-cache}"
VLLM_DIR="${VLLM_DIR:-../vllm}"

CASE_NAME="$(basename "$CASE_FILE" .env)"
TS="$(date +%Y%m%d_%H%M%S)"
TAG="${RESULT_TAG:-$TS}"
# RESULT_DIR is overridable from the env so sweep.py can direct each
# combo's output into its hashed combo dir (tmp/bench_<case>_<sweep>/<id>).
: "${RESULT_DIR:=tmp/bench_${CASE_NAME}_${TAG}}"
mkdir -p "$RESULT_DIR"

VLLM_LOG="$RESULT_DIR/vllm.log"
BM_LOG="$RESULT_DIR/bench.log"
META_FILE="$RESULT_DIR/meta.txt"
METRICS_FILE="$RESULT_DIR/metrics.txt"

{
    echo "case_file=$CASE_FILE"
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
    echo "block_size=${BLOCK_SIZE:-default}"
    echo "long_prefill_token_threshold=${LONG_PREFILL_TOKEN_THRESHOLD:-default}"
    echo "rpa_p_block_sizes=${RPA_P_BLOCK_SIZES:-unset}"
    echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "git_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "timestamp=$TS"
} | tee "$META_FILE"
echo "----"
echo "Results: $RESULT_DIR"

if [ "$DATASET" = "sonnet" ]; then
    [ -f "$VLLM_DIR/benchmarks/sonnet.txt" ] \
        || { echo "ERROR: $VLLM_DIR/benchmarks/sonnet.txt not found; set VLLM_DIR." >&2; exit 1; }
    : > "$VLLM_DIR/benchmarks/sonnet_4x.txt"
    for _ in 1 2 3 4; do
        cat "$VLLM_DIR/benchmarks/sonnet.txt" >> "$VLLM_DIR/benchmarks/sonnet_4x.txt"
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

clean_up() {
    pkill -f "vllm serve"        2>/dev/null || true
    pkill -f "vllm.entrypoints"  2>/dev/null || true
}
trap clean_up EXIT

echo "Starting vllm server (log: $VLLM_LOG) ..."
VLLM_USE_V1=1 vllm serve "$MODEL" "${SERVE_ARGS[@]}" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "Waiting up to 20 min for server startup ..."
for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "ERROR: vllm exited early. tail of $VLLM_LOG:" >&2
        tail -40 "$VLLM_LOG" >&2
        exit 1
    fi
    if grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "  ready after ${i}0s"
        break
    fi
    sleep 10
done

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
            --dataset-path "$VLLM_DIR/benchmarks/sonnet_4x.txt"
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

echo "Running benchmark (log: $BM_LOG) ..."
if ! ( cd "$VLLM_DIR" && vllm bench serve "${BM_ARGS[@]}" ) > "$BM_LOG" 2>&1; then
    echo "ERROR: bench failed. tail of $BM_LOG:" >&2
    tail -40 "$BM_LOG" >&2
    exit 1
fi

echo "==== Summary ===="
grep -E "Request throughput|Output token throughput|Total Token throughput|Mean.*\(ms\):|Median.*\(ms\):|P99.*\(ms\):" "$BM_LOG" || true
echo "===="

# Metric extraction lives in Python (tools/benchmark/parse_bench_log.py)
# so it gets unit-tested independently of this shell driver.
python3 -m tools.benchmark.parse_bench_log "$BM_LOG" > "$METRICS_FILE"

echo "Saved metrics to $METRICS_FILE"
