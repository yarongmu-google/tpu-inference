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

# No layout flag: xla_tpu_enable_large_2nd_minor_layout_for_x16 is an
# opt-in for PRE-v7 generations only - on this target the large 2nd-minor
# (16,128) layout for 16-bit types is already XLA's default (see
# jax/_src/pallas/mosaic/tpu_info.py get_sublane_tiling), so setting it
# is a no-op. The (8,128)-tiled pipelined windows are Mosaic's own
# memref inference for window buffers, which no XLA flag reaches.
unset LIBTPU_INIT_ARGS

# provenance + sanity: commit, jax, devices, and WHICH decode_kernel loads
run git rev-parse --short HEAD
run python -c "import jax; print(jax.__version__); print(jax.devices())"
run python -c "import tpu_inference.kernels.fused_moe.v2.decode_kernel as m; print(m.__file__)"

# 1) Mosaic pass dump: short run; the dumps include apply-vector-layout
#    (post-relayout), which the local CPU-side dump cannot see.
# DUMP_FLAGS = the config under test. The dump must capture the
# HYPOTHESIS config, not the defaults - otherwise the histogram/timeline
# never reflect the change being measured and cannot be compared against
# tmp/baseline/. Override per run: DUMP_FLAGS="--bg=8" ./tmp/run_bench_dump.sh
DUMP_FLAGS="${DUMP_FLAGS:---bg=4 --capacity=24}"
echo "DUMP_FLAGS: $DUMP_FLAGS"

mkdir -p tmp/mosaic_dump
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_dump"
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=3 --warmup=1 $DUMP_FLAGS
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
# which deeper dump/debug flags does this build accept?
run python tmp/probe_flags.py

# v1 baseline: same weights-per-token ratio (btc=32, 1.6 GB/device) and the
# same expert-sum accumulator, so its number says whether ~3.9 ms is bad or
# just what this shape costs. Separate process so a failure can't take v0.2
# down with it.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v1 --iters=10 --warmup=3

# (No scoped-VMEM flag: the backend reported "Used 78.61M of 63.94M vmem"
#  with the ceiling raised - 64 MiB is the PHYSICAL capacity, so weight
#  windows beyond be=4 are impossible and bg is the amortization lever.)
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=10 --warmup=3 --tune

# 3) per-stage spans: named_scope markers -> device trace. This is what
#    says WHERE the time goes (prologue vs gather vs gmms vs combine).
# (xprof removed: the profiler plugin ABI mismatches this libtpu
#  and segfaults in ProfilerSession::Create)

echo
echo "log: tmp/bench_dump.log  hist: tmp/op_histogram.txt  passes: tmp/mosaic_passes.tar.xz"
