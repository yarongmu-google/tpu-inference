#!/usr/bin/env bash
# Serving-shape ladder for the v2 TP decode kernel: reproduce the
# serving crash OUTSIDE vllm.
#
# Serving engages the kernel through fused_moe_decode_tp_serving
# (router_fused=False: precomputed gating, serving capacity formula,
# be=4/bg=1) at T = the padded num-reqs buckets - a PATH and SHAPES the
# kernel bench never ran on hardware (v02 rows are router-fused at
# T=512 only). This ladder runs the v02s serving-entry variant at every
# serving decode shape, each T in its OWN process so a hardware halt
# kills one rung, not the ladder.
#
# Reading the result against the line-4 crash
# (RuntimeUnexpectedCoreHalt on request arrival):
#   - some T rung halts here          -> kernel bug at that shape;
#     fix in decode_kernel.py, re-run the rung.
#   - every rung passes               -> the kernel is clean at all
#     serving shapes and the crash lives in the serving INTERACTION
#     (SC-offloaded collectives sharing the step with our in-kernel
#     ICI DMAs, 60 stacked layers, or the runner's bucket plumbing) -
#     next probe is the server log's engaged shapes + an xla dump.
# The v02 row also runs at each rung (substring selection): fused-entry
# vs serving-entry at the same T separates router_fused=False bugs
# from shape bugs.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/serving_shapes.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

export PYTHONPATH=.
export JAX_PLATFORMS=tpu
unset LIBTPU_INIT_ARGS

git rev-parse --short HEAD
python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

for WD in fp8 bf16; do
  for T in 8 16 32 64 128 256 512; do
    echo
    echo "=== wdtype=$WD T=$T (serving entry, be=4 bg=1, serving cap) ==="
    python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --wdtype=$WD --variants=v02s --tokens=$T --iters=5 --warmup=2 \
      --be=4 --bg=1 --capacity=32
    echo "=== exit=$? ==="
  done
done

# ---- Mosaic dump at the T=8 rung (the fixed shape): the routing dot
# ---- must appear as vmatmul work at the padded width, with NO
# ---- broadcast-multiply-reduce chain in the prologue - the dump-level
# ---- confirmation of the fix, not just "it no longer errors".
rm -rf tmp/mosaic_t8 && mkdir -p tmp/mosaic_t8
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_t8"
python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
  --wdtype=fp8 --variants=v02s --tokens=8 --iters=2 --warmup=1 \
  --be=4 --bg=1 --capacity=32
unset LIBTPU_INIT_ARGS
LAST=$(ls tmp/mosaic_t8/*post-finalize-llo* 2>/dev/null | tail -1)
if [ -n "$LAST" ]; then
  python tmp/dump_histogram.py "$LAST" > tmp/t8_histogram.txt
  grep -E "vmatmul|vlatchi|vbcast|vcvt" tmp/t8_histogram.txt | head -10
  tar -c tmp/mosaic_t8 | xz -9 -T0 > tmp/mosaic_t8.tar.xz
  ls -lh tmp/mosaic_t8.tar.xz tmp/t8_histogram.txt
fi
rm -rf tmp/mosaic_t8

echo
echo "log: tmp/serving_shapes.log  t8 histogram: tmp/t8_histogram.txt"
echo "then: git add tmp/ && commit ('serving shapes run.') && push"
