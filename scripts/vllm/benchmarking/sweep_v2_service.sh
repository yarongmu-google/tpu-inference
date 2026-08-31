#!/usr/bin/env bash
# Service-level sweep for the TP-MoE decode kernel serving config.
#
# Sweeps the three scheduler knobs that shape what the MoE kernel
# actually sees per step:
#   --max-model-len          (MML_LIST)  KV footprint per seq; the
#                            workload needs in+out <= MML, so its only
#                            tuning freedom is shrinking KV to admit
#                            more concurrent seqs
#   --max-num-batched-tokens (MNB_LIST)  prefill chunk cap; big = fast
#                            prefill but long decode stalls between
#                            decode steps
#   --max-num-seqs           (MNS_LIST)  decode-step width; the v2
#                            kernel was built and tuned at 512 global
#                            tokens/step - this axis moves its shape
#
# One combo = start server -> wait for /health -> short client run ->
# parse throughput -> kill server -> next. Results accumulate in
# tmp/vllm_logs/sweep/results.tsv and a sorted summary prints at the
# end. Every combo records whether the v2 kernel actually ENGAGED
# (server log grep) - a silent GMM fallback row is marked and must not
# win the sweep.
#
# Defaults are the fp8 397B line-4 config (bench_throughput_qwen_server
# .sh); VARIANT=gmm sweeps the GMM_TP baseline (line 3) instead so the
# baseline gets its own best config - never compare a tuned v2 against
# an untuned baseline.
#
# The client here is a SHORT proxy (default in=1024 out=1024, 512
# prompts, range_ratio like the real client): decode-heavy enough to
# rank configs, short enough that a combo is minutes, not hours.
# Re-run the 1-2 finalists with the full client (out=8192, 2048
# prompts) before believing an absolute number.
#
# Usage (from the repo root, serving env):
#   ./scripts/vllm/benchmarking/sweep_v2_service.sh
#   VARIANT=gmm ./scripts/vllm/benchmarking/sweep_v2_service.sh
#   MNS_LIST="512" MNB_LIST="1024 4096" ./scripts/...sweep_v2_service.sh

set -uo pipefail
cd "$(dirname "$0")/../../.."

VARIANT="${VARIANT:-v2}"                 # v2 | gmm
MODEL="${MODEL:-Qwen/Qwen3.5-397B-A17B-FP8}"
PORT="${PORT:-8000}"
IN_LEN="${IN_LEN:-1024}"
OUT_LEN="${OUT_LEN:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-512}"
RANGE_RATIO="${RANGE_RATIO:-0.8}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-3600}"   # seconds; startup is SLOW
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"

SWEEP_DIR=tmp/vllm_logs/sweep
RESULTS="$SWEEP_DIR/results.tsv"
mkdir -p "$SWEEP_DIR"
BS=scripts/vllm/benchmarking/benchmark_serving.py

# Candidate lists: explicit env wins; otherwise ANCHOR them with the
# chip+model estimator (KV-pool arithmetic -> MNS expected-occupancy
# anchor -> MNB lower bounds) and sweep its x0.5/x1/x2 neighbors for
# estimation error. FULL_OUT_LEN prices the KV pool at the REAL
# workload (the short proxy client would under-fill KV and over-admit
# seqs). The estimator queries the TPU for HBM when HBM_GB is unset -
# it must run before any server is up (which it is, here).
FULL_OUT_LEN="${FULL_OUT_LEN:-8192}"
# PEAK_TFLOPS + HBM_GBPS (optional envs): per-device hardware rates
# enabling the MNB roofline-crossover anchor - see the estimator doc
EST_ARGS=(--model "$MODEL" --in-len "$IN_LEN" --out-len "$FULL_OUT_LEN"
          --gpu-util "$GPU_MEM_UTIL")
[ -n "${PEAK_TFLOPS:-}" ] && EST_ARGS+=(--peak-tflops "$PEAK_TFLOPS")
[ -n "${HBM_GBPS:-}" ] && EST_ARGS+=(--hbm-gbps "$HBM_GBPS")
if [ -z "${MNB_LIST:-}" ] || [ -z "${MNS_LIST:-}" ] \
   || [ -z "${MML_LIST:-}" ]; then
  EST=$(python scripts/vllm/benchmarking/estimate_service_knobs.py \
          "${EST_ARGS[@]}" --emit shell) || {
    echo "estimator failed; set MML_LIST/MNB_LIST/MNS_LIST manually" >&2
    exit 1
  }
  echo "estimator anchors: $EST"
  # human-readable derivation into the sweep log for provenance
  python scripts/vllm/benchmarking/estimate_service_knobs.py \
    "${EST_ARGS[@]}" | tee "$SWEEP_DIR/estimate.txt" \
    2>/dev/null || true
  [ -z "${MML_LIST:-}" ] && eval "$(echo "$EST" | grep '^MML_LIST=')"
  [ -z "${MNB_LIST:-}" ] && eval "$(echo "$EST" | grep '^MNB_LIST=')"
  [ -z "${MNS_LIST:-}" ] && eval "$(echo "$EST" | grep '^MNS_LIST=')"
  # always include the kernel design point (the estimator is
  # deliberately kernel-agnostic; the v2 kernel was tuned at 512
  # global tokens/step)
  DESIGN_MNS="${DESIGN_MNS:-512}"
  case " $MNS_LIST " in
    *" $DESIGN_MNS "*) ;;
    *) MNS_LIST="$MNS_LIST $DESIGN_MNS" ;;
  esac
fi
echo "grid: MML={$MML_LIST} MNB={$MNB_LIST} MNS={$MNS_LIST}"

LIBTPU_FLAGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false'

# doubling num-reqs buckets 8..MNS so decode steps pad to the nearest
# bucket, with the top bucket exactly MNS (the kernel design point)
buckets_for() {
  local mns=$1 b=8 out="8"
  while [ $((b * 2)) -le "$mns" ]; do b=$((b * 2)); out="$out,$b"; done
  [ "$b" -ne "$mns" ] && out="$out,$mns"
  echo "$out"
}

grab() {  # grab <file> <label prefix> -> numeric column
  grep -F "$2" "$1" | tail -1 | sed 's/.*:\s*//' | awk '{print $1}'
}

if [ ! -f "$RESULTS" ]; then
  printf "variant\tmml\tmnb\tmns\tengaged\tstatus\tout_tok_s\ttotal_tok_s\tmedian_tpot_ms\tmedian_ttft_ms\n" > "$RESULTS"
fi

for MML in $MML_LIST; do
  for MNB in $MNB_LIST; do
    for MNS in $MNS_LIST; do
      if [ $((IN_LEN + OUT_LEN)) -gt "$MML" ]; then
        echo "SKIP mml=$MML: workload in+out=$((IN_LEN + OUT_LEN)) exceeds it"
        continue
      fi
      # MNB hard lower bounds (see estimate_service_knobs.py): a
      # prompt must prefill in one step, and a decode step batches
      # one token per running seq - an MNB below either just splits
      # steps, never wins. Skip instead of burning a server start.
      if [ "$MNB" -lt "$IN_LEN" ]; then
        echo "SKIP mnb=$MNB < in_len=$IN_LEN (prefill would chunk)"
        continue
      fi
      if [ "$MNB" -lt "$MNS" ]; then
        echo "SKIP mnb=$MNB < mns=$MNS (decode step would split)"
        continue
      fi
      TAG="${VARIANT}_mml${MML}_mnb${MNB}_mns${MNS}"
      SLOG="$SWEEP_DIR/${TAG}_server.log"
      CLOG="$SWEEP_DIR/${TAG}_client.log"
      BUCKETS=$(buckets_for "$MNS")
      echo
      echo "=== $TAG (buckets=$BUCKETS) ==="

      ENV_COMMON=(MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1
        ATTN_BUCKETIZED_NUM_REQS=true
        "ATTN_CUSTOM_NUM_REQS_BUCKETS=$BUCKETS"
        ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256
        "LIBTPU_INIT_ARGS=$LIBTPU_FLAGS")
      [ "$VARIANT" = "v2" ] && ENV_COMMON+=(USE_MOE_TP_DECODE_KERNEL=1)

      setsid env "${ENV_COMMON[@]}" \
        vllm serve "$MODEL" \
        --max-model-len="$MML" --max-num-batched-tokens="$MNB" \
        --max-num-seqs="$MNS" --no-enable-prefix-caching \
        --gpu-memory-utilization="$GPU_MEM_UTIL" \
        --tensor-parallel-size=8 --async-scheduling --port="$PORT" \
        --language-model-only --enable-auto-tool-choice \
        --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 \
        '--limit-mm-per-prompt={"image":0, "video": 0}' \
        --kv-cache-dtype=fp8 \
        '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' \
        --block-size=256 > "$SLOG" 2>&1 &
      SERVER_PID=$!

      STATUS=up
      waited=0
      until curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
          STATUS=died; break
        fi
        if [ "$waited" -ge "$STARTUP_TIMEOUT" ]; then
          STATUS=timeout; break
        fi
        sleep 15; waited=$((waited + 15))
      done
      echo "server: $STATUS after ${waited}s"

      OUT_TPS=; TOT_TPS=; TPOT=; TTFT=; ENGAGED=n/a
      if [ "$STATUS" = "up" ]; then
        if [ "$VARIANT" = "v2" ]; then
          if grep -q "using the TP decode kernel" "$SLOG"; then
            ENGAGED=yes
          else
            ENGAGED=NO
          fi
        fi
        python "$BS" --model "$MODEL" --dataset-name random \
          --backend vllm --port "$PORT" \
          --random-input-len="$IN_LEN" --random-output-len="$OUT_LEN" \
          --random-range-ratio="$RANGE_RATIO" \
          --num-prompts="$NUM_PROMPTS" --max-concurrency="$MNS" \
          --ignore-eos > "$CLOG" 2>&1
        if [ $? -eq 0 ]; then
          STATUS=ok
          OUT_TPS=$(grab "$CLOG" "Output token throughput")
          TOT_TPS=$(grab "$CLOG" "Total token throughput")
          TPOT=$(grab "$CLOG" "Median TPOT")
          TTFT=$(grab "$CLOG" "Median TTFT")
        else
          STATUS=client_failed
        fi
      fi

      # teardown: whole process group, then wait for the port to free
      kill -TERM -- "-$SERVER_PID" 2>/dev/null
      for _ in $(seq 1 24); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 5
      done
      kill -KILL -- "-$SERVER_PID" 2>/dev/null
      sleep 10

      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$VARIANT" "$MML" "$MNB" "$MNS" "$ENGAGED" "$STATUS" \
        "${OUT_TPS:--}" "${TOT_TPS:--}" "${TPOT:--}" "${TTFT:--}" \
        | tee -a "$RESULTS"
      xz -9 -T0 -f "$SLOG" 2>/dev/null
    done
  done
done

echo
echo "=== sweep summary (by output tok/s; ENGAGED=NO rows are GMM"
echo "=== fallbacks masquerading as v2 - fix before trusting) ==="
{ head -1 "$RESULTS"; tail -n +2 "$RESULTS" | sort -t$'\t' -k7 -rn; } \
  | column -t -s$'\t'
echo
echo "results: $RESULTS   logs: $SWEEP_DIR/"
echo "next: re-run the winner with the FULL client (out=8192, 2048"
echo "prompts) via bench_throughput_qwen_client.sh before quoting it"
