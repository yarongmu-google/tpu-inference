#!/usr/bin/env bash
# Bench + Mosaic pass dump for the fused_moe/v2 decode kernel, on a TPU VM.
# Mirrors every command and its stdout/stderr to the screen AND to
# tmp/bench_dump.log; Mosaic per-pass dumps land in tmp/mosaic_dump/.
# Run from anywhere: ./tmp/run_bench_dump.sh

set -uo pipefail
cd "$(dirname "$0")/.."           # repo root

LOG=tmp/bench_dump.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1      # everything to screen + log

run() {
  echo
  echo "+ $*"
  "$@"
  echo "+ exit=$?"
}

# PYTHONPATH=. so THIS checkout shadows any installed tpu_inference
export PYTHONPATH=.
export JAX_PLATFORMS=tpu

# bf16 matmul operands want 16-row tiles; pipelined windows were coming
# back with 8-row tiles, forcing an unpack/repack of every weight vreg.
# Ask XLA for the large 2nd-minor layout on 16-bit types - but probe it
# first: an unknown flag is fatal to every step that follows.
LAYOUT_FLAG="--xla_tpu_enable_large_2nd_minor_layout_for_x16=true"
if XLA_FLAGS="$LAYOUT_FLAG" python -c "import jax; jax.numpy.zeros(1)" \
     >/dev/null 2>&1; then
  export XLA_FLAGS="${XLA_FLAGS:-} $LAYOUT_FLAG"
  echo "layout flag: ACCEPTED"
else
  echo "layout flag: REJECTED by this build - running without it"
fi

# provenance + sanity: commit, jax, devices, and WHICH decode_kernel loads
run git rev-parse --short HEAD
run python -c "import jax; print(jax.__version__); print(jax.devices())"
run python -c "import tpu_inference.kernels.fused_moe.v2.decode_kernel as m; print(m.__file__)"

# 1) Mosaic pass dump: short run; the dumps include apply-vector-layout
#    (post-relayout), which the local CPU-side dump cannot see.
mkdir -p tmp/mosaic_dump
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_dump"
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=3 --warmup=1
unset LIBTPU_INIT_ARGS
run du -sh tmp/mosaic_dump

# The raw dumps are hundreds of MB. What we actually read is the op
# histogram per named_scope region (a few KB) plus, if it fits, the last
# module in the pass chain. Everything else is discarded.
run bash -c '
  set -x
  ls tmp/mosaic_dump | tail -30
  LAST=$(ls tmp/mosaic_dump/* 2>/dev/null | tail -1)
  python tmp/dump_histogram.py $LAST > tmp/op_histogram.txt
  python tmp/llo_timeline.py $LAST --cols 150 > tmp/timeline.txt
  wc -l tmp/op_histogram.txt tmp/timeline.txt
  # every pass, not just the last: the progression shows WHERE a cost is
  # introduced. xz -9 gets the whole chain into a few MB.
  tar -c tmp/mosaic_dump | xz -9 -T0 > tmp/mosaic_passes.tar.xz
  ls -lh tmp/mosaic_passes.tar.xz
  rm -rf tmp/mosaic_dump'

# 1b) isolate the window-vs-scratch operand question (small, separate
#     module chain so its histogram is unambiguous)
mkdir -p tmp/mosaic_probe
LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_probe" \
  run python tmp/probe_tiling.py
run bash -c '
  for f in tmp/mosaic_probe/*post-finalize-llo*; do
    python tmp/dump_histogram.py "$f"
  done > tmp/probe_histogram.txt 2>&1
  grep -c . tmp/probe_histogram.txt
  rm -rf tmp/mosaic_probe'

# 2) clean timing run (no dump overhead): v0.2 only - the v1 baseline is
#    deliberately out of the bring-up loop; it returns for the final
#    comparison once v0.2 is healthy on hardware.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=10 --warmup=3 --tune

# 3) per-stage spans: named_scope markers -> device trace. This is what
#    says WHERE the time goes (prologue vs gather vs gmms vs combine).
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=3 --warmup=1 --profile-dir=tmp/xprof
run bash -c 'tar -czf tmp/xprof.tgz -C tmp xprof && rm -rf tmp/xprof
             ls -lh tmp/xprof.tgz'

echo
echo "log: tmp/bench_dump.log  hist: tmp/op_histogram.txt  passes: tmp/mosaic_passes.tar.xz  trace: tmp/xprof.tgz"
