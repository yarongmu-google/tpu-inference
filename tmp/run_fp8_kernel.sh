#!/usr/bin/env bash
# fp8 kernel bring-up loop (v2 decode MoE, w8a8): tests -> bf16
# baseline -> fp8 tune -> fp8 A/B + ablation ladder -> Mosaic/XLA
# dumps. Everything tees into push-sized files under tmp/.
#
# Reading guide (qwen35_fp8 proposal sec 0d / 7):
#   - predicted fp8 device ~380-430 us, bench WALL ~750-800 at the
#     tuned winner (wall = device + ~366 us call envelope; serving
#     amortizes the envelope) vs bf16's 1142 wall / 848 device;
#   - ablate=ffn should collapse toward the ~256 us device stream
#     (+envelope) - the "all memory hidden" check;
#   - (none - quant) and (none - scales) are the new fp8 rows;
#   - the Mosaic histogram must show NO vcvt storm (free-latch
#     lowering held in-kernel) and the XLA grep must show NO
#     f8e4m3 weight-shaped copies outside jit_stage (upload-only).
#
# Tune is the long pole (~15 min). Re-run finalists with FLAGS=...
# from the printed winner: FLAGS="..." ./tmp/run_fp8_kernel.sh
# Afterwards: git add tmp/ && commit ("fp8 run.") && push.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/fp8_kernel.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

run() {
  echo
  echo "+ $*"
  "$@"
  echo "+ exit=$?"
}

export PYTHONPATH=.
export JAX_PLATFORMS=tpu
unset LIBTPU_INIT_ARGS

# fp8 flags: start from the bf16 winner's shape (be=4 bg=4 cap=32
# bd1c/bd2c/bcT mid-range); the tuner below re-decides from scratch.
FLAGS="${FLAGS:---be=4 --bg=4 --capacity=32 --bd1c=256 --bd2c=128 --bcT=256}"
echo "FLAGS: $FLAGS"

run git rev-parse --short HEAD
run python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

# ---- 0) the whole test suite (14 bf16 guard + 7 fp8). Interpret-mode
# ---- pallas uses io_callbacks that only lower on CPU - run the tests
# ---- on the CPU platform with 8 simulated devices, NOT on the TPU.
run env JAX_PLATFORMS=cpu \
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
    python -m pytest tests/kernels/fused_moe_v2_decode_test.py -q

# ---- 1) same-session bf16 baseline (lesson 24: differentials within
# ---- one session/env) at the bf16 tuned winner ----------------------
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v1,v02 --iters=15 --warmup=3 \
    --be=4 --bg=4 --capacity=32 --bd1c=256 --bd2c=128 --bcT=256

# ---- 2) fp8 tuner (blocks are always tuned, lesson 13; fp8's be=8
# ---- family fits VMEM where bf16's never did) -----------------------
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --tune --iters=10 --warmup=3

# ---- 3) fp8 at FLAGS (re-run with the winner via FLAGS=...) ---------
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --variants=v02 --iters=30 --warmup=5 $FLAGS

# ---- 3b) act-scale A/B: tensor mode deletes OHS/s_x (one global
# ---- dynamic scale, amax exchange under routing). (3) vs (3b) = the
# ---- per-token machinery's true price; accuracy ladder judges later.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --act-scale=tensor --variants=v02 --iters=30 --warmup=5 $FLAGS

# ---- 4) envelope + ablation ladder, incl the new fp8 rows -----------
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --variants=env --iters=10 --warmup=3
for a in none masks gather ffn combine weights routing ag quant scales all; do
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --wdtype=fp8 --variants=v02 --iters=10 --warmup=3 $FLAGS --ablate=$a
done

# ---- 4b) combine-form discriminator (review finding 3): does the
# ---- kernel's slot-major concat combine run at flat-buffer speed? ---
run python tmp/probe_combine_kfuse.py

# ---- 5) Mosaic dump of the fp8 kernel: op histogram -----------------
# verdicts: vcvt count ~ 0 (free-latch held); vmatmul/vlatchi counts
# consistent with the per-tile model; the combine's K-fused dot did
# not spawn a copy storm (vector_load/store counts sane).
rm -rf tmp/mosaic_fp8k && mkdir -p tmp/mosaic_fp8k
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_fp8k"
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --variants=v02 --iters=2 --warmup=1 $FLAGS
unset LIBTPU_INIT_ARGS
run bash -c '
  LAST=$(ls tmp/mosaic_fp8k/*post-finalize-llo* 2>/dev/null | tail -1)
  python tmp/dump_histogram.py "$LAST" > tmp/fp8_kernel_histogram.txt
  grep -E "vcvt|vpack|vunpack|vmatmul|vlatchi|vector_load|vector_store" \
    tmp/fp8_kernel_histogram.txt | head -20
  tar -c tmp/mosaic_fp8k | xz -9 -T0 > tmp/mosaic_fp8k.tar.xz
  ls -lh tmp/mosaic_fp8k.tar.xz tmp/fp8_kernel_histogram.txt
  rm -rf tmp/mosaic_fp8k'

# ---- 6) XLA copy-trap check at the kernel shapes (lesson A.2) -------
rm -rf tmp/xla_dump_fp8k && mkdir -p tmp/xla_dump_fp8k
XLA_FLAGS="--xla_dump_to=tmp/xla_dump_fp8k --xla_dump_hlo_as_text" \
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --wdtype=fp8 --variants=v02 --iters=2 --warmup=1 $FLAGS
run bash -c '
  for f in tmp/xla_dump_fp8k/*jit_fn*after_optimizations.txt \
           tmp/xla_dump_fp8k/*jit_wrapped*after_optimizations.txt; do
    grep -nE "%(copy|reshape|transpose)" "$f" 2>/dev/null | grep f8e4m3 | head -5
  done | head -20
  echo "(empty grep above = no per-call fp8 weight relayout: PASS)"
  tar -c tmp/xla_dump_fp8k | xz -9 -T0 > tmp/xla_dump_fp8k.tar.xz
  ls -lh tmp/xla_dump_fp8k.tar.xz
  rm -rf tmp/xla_dump_fp8k'

echo
echo "log: tmp/fp8_kernel.log  histogram: tmp/fp8_kernel_histogram.txt"
echo "next: FLAGS=\"<tuner winner>\" ./tmp/run_fp8_kernel.sh  (steps 3-6 re-run at the winner)"
