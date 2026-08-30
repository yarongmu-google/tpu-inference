#!/usr/bin/env bash
# fp8 pre-kernel probes (qwen35_fp8 proposal sec 8 steps 1-2 +
# kernel_plan.md verify items). NO kernel code exists yet - these
# gate it. ~10 min total. Run in the PROF env if profiling later;
# these probes themselves need only the serving env.
#
#   1) probe_wbw.py           bf16 pair (regression ref) THEN e4m3 pair
#                             -> the fp8 stream floor constant.
#                             Expect wpair_8dev fp8 ~= half the bf16
#                             wall at ~equal GB/s; deviation re-prices
#                             the plan (steps0_3_fp8.md sec 3).
#   2) XLA HLO dump around (1) -> the lesson-A.2 copy trap check:
#                             any copy/reshape/transpose at an f8e4m3
#                             weight shape = XLA re-laying-out the
#                             operand per call -> fix at LOAD time.
#   3) probe_fp8_lowering.py  Mosaic dumps + op histograms:
#                             vmatpush.e4m3/vmatmul vs vcvt storms;
#                             w8a16 free-latch legality; K=128 diag.
#   4) probe_combine_kfuse.py as-coded vs K-fused combine, expect ~2x
#                             (gates the y/ohg buffer restructure).
#
# Afterwards everything in tmp/ is push-sized: git add tmp/ && commit.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/fp8_probes.log
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

run git rev-parse --short HEAD
run python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

# ---- 1) stream floor, bf16 ref + fp8 --------------------------------
run python tmp/probe_wbw.py

# ---- 2) XLA copy-trap check on the fp8 stream path ------------------
# (per-compile dump, so re-running the probe is cheap; grep any
# copy/reshape/transpose touching an f8e4m3 tensor at weight shapes)
rm -rf tmp/xla_dump_fp8 && mkdir -p tmp/xla_dump_fp8
XLA_FLAGS="--xla_dump_to=tmp/xla_dump_fp8 --xla_dump_hlo_as_text" \
  run python tmp/probe_wbw.py
run bash -c '
  ls tmp/xla_dump_fp8 | wc -l
  for f in tmp/xla_dump_fp8/*after_optimizations*.txt; do
    grep -nE "%(copy|reshape|transpose)" "$f" | grep -E "f8e4m3" | head -10
  done 2>/dev/null | head -30
  echo "(empty grep above = no fp8 weight-shaped relayout copies: PASS)"
  tar -c tmp/xla_dump_fp8 | xz -9 -T0 > tmp/xla_dump_fp8.tar.xz
  ls -lh tmp/xla_dump_fp8.tar.xz
  rm -rf tmp/xla_dump_fp8'

# ---- 3) lowering probes + Mosaic histograms -------------------------
rm -rf tmp/mosaic_fp8 && mkdir -p tmp/mosaic_fp8
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_fp8"
run python tmp/probe_fp8_lowering.py
unset LIBTPU_INIT_ARGS
run bash -c '
  for f in tmp/mosaic_fp8/*post-finalize-llo*; do
    echo "== $(basename $f)"
    python tmp/dump_histogram.py "$f"
  done > tmp/fp8_lowering_histograms.txt 2>&1
  # the verdict lines: matmul ops vs convert storms, per kernel
  grep -E "^==|vmatmul|vmatpush|vcvt|vpack|vunpack" \
    tmp/fp8_lowering_histograms.txt | head -60
  tar -c tmp/mosaic_fp8 | xz -9 -T0 > tmp/mosaic_fp8.tar.xz
  ls -lh tmp/mosaic_fp8.tar.xz tmp/fp8_lowering_histograms.txt
  rm -rf tmp/mosaic_fp8'

# ---- 4) combine K-fuse discriminator --------------------------------
run python tmp/probe_combine_kfuse.py

echo
echo "log: tmp/fp8_probes.log  histograms: tmp/fp8_lowering_histograms.txt"
echo "verdicts to read off: (1) fp8 wpair_8dev us + GB/s; (2) empty"
echo "f8e4m3 copy grep; (3) per-kernel vcvt counts ~0; (4) ratio ~2x"
