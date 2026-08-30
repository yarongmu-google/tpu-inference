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

# 0) TUNE FIRST: every block size - be/bg/capacity AND the weight-fetch
#    grain bd1c (+ bd2c, bcT) - is swept before anything else, and the
#    dumps + ablate ladder below all run at the WINNER config, not at a
#    stale hand-picked one. (64 MiB is the PHYSICAL VMEM capacity; the
#    tuner's failed rows are the ones that exceed it.)
# --profile-dir arms the tuner's finalist re-rank: sweeps order by
# wall-min, then the top-5 are re-measured on DEVICE time (sub-us
# stable vs ~40us envelope jitter). Needs the prof env (jax>=0.11 with
# matched libtpu) - on the old pair the re-rank is skipped, not fatal.
rm -rf tmp/moe_xprof_tune tmp/moe_xprof_tune.tar.xz
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=10 --warmup=3 --tune \
    --profile-dir=tmp/moe_xprof_tune
run bash -c '
  if [ -d tmp/moe_xprof_tune ]; then
    tar -c tmp/moe_xprof_tune | xz -9 -T0 > tmp/moe_xprof_tune.tar.xz
    ls -lh tmp/moe_xprof_tune.tar.xz
    rm -rf tmp/moe_xprof_tune
  fi'
TUNED_FLAGS=$(grep -o 'WINNER: .* ->' "$LOG" | tail -1 \
              | sed 's/WINNER: //; s/ ->//')
echo "TUNED_FLAGS: $TUNED_FLAGS"

# 1) Mosaic pass dump: short run; the dumps include apply-vector-layout
#    (post-relayout), which the local CPU-side dump cannot see.
# DUMP_FLAGS = the config under test: the TUNED WINNER unless overridden
# for a specific hypothesis (DUMP_FLAGS="--bg=8 ..." ./tmp/run_bench_dump.sh).
DUMP_FLAGS="${DUMP_FLAGS:-$TUNED_FLAGS}"
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

# 1a) SECOND dump: the ablate=all floor config. Measured 2026-08-29:
# 2777us = 78% of the whole kernel with an EMPTY step body (no weight
# fetch, no masks/gather/ffn/combine, no park) - ~21.7us/step of
# per-step machinery. Its op stream IS the floor; prime suspect is the
# 13.4k-op unscoped block (2048 vload/vpack/vbitcast/vunpack + 3072
# vstore = 12 MiB/step of VALU shuffling, IDENTICAL in the window-era
# baseline dump) - this dump says whether it survives with everything
# stubbed.
mkdir -p tmp/mosaic_dump
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_dump"
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=3 --warmup=1 $DUMP_FLAGS --ablate=all
unset LIBTPU_INIT_ARGS
run bash -c '
  LAST=$(ls tmp/mosaic_dump/* 2>/dev/null | tail -1)
  python tmp/dump_histogram.py $LAST > tmp/op_histogram_all.txt
  python tmp/llo_timeline.py $LAST --cols 150 > tmp/timeline_all.txt
  wc -l tmp/op_histogram_all.txt tmp/timeline_all.txt
  tar -c tmp/mosaic_dump | xz -9 -T0 > tmp/mosaic_passes_all.tar.xz
  ls -lh tmp/mosaic_passes_all.tar.xz
  rm -rf tmp/mosaic_dump'

# ---- ANSWERED probes, commented out to keep the loop fast. Uncomment
# ---- to re-measure; results recorded inline.

# 1b) window-vs-scratch operand question. ANSWERED: scratch operand is
#     cheaper (1,744 vs 4,304 ops) - and moot since the manual blocked fetch.
# mkdir -p tmp/mosaic_probe
# LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_probe" \
#   run python tmp/probe_tiling.py
# run bash -c '
#   for f in tmp/mosaic_probe/*post-finalize-llo*; do
#     python tmp/dump_histogram.py "$f"
#   done > tmp/probe_histogram.txt 2>&1
#   grep -c . tmp/probe_histogram.txt
#   rm -rf tmp/mosaic_probe'

# 2) which deeper dump/debug flags does this build accept?
#    ANSWERED: LIBTPU_INIT_ARGS takes log_recorder/scoped_vmem/large_2nd
#    _minor; all llo/asm dump flags rejected; XLA_FLAGS rejects almost all.
# run python tmp/probe_flags.py

# v1 baseline. ANSWERED (2026-08-29): 2682.4 us min / 2745.9 median.
# run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
#     --variants=v1 --iters=10 --warmup=3

# Weight-stream bandwidth probe. Single-device ANSWERED (2026-08-29):
# manual blocked fetch 2150-2300 GB/s regardless of split; window pipeline 494;
# spec 3207. RE-ENABLED for the new concurrent question: the in-kernel
# fetch measured 494 UNCHANGED from the window pipeline, and the kernel
# runs on all 8 devices while the probe ran on 1. wpair_1dev/8dev
# stream the ACTUAL w1+w2 pair (1.5 GiB) through the production blocked-
# fetch structure, solo vs all devices at once - the 8dev time compares 1:1
# against the kernel's (ablate=weights - ablate=all).
run python tmp/probe_wbw.py

# Per-step floor probe: empty-body kernels with the decode kernel's
# window structure vs pure-HBM refs, 128 vs 64 steps. Decomposes the
# 21.7us/step ablate=all floor into window machinery vs grid overhead.
run python tmp/probe_floor.py

# ICI/D2D topology probe (all 56 device pairs, ppermute ping-pong latency +
# 1 GiB unidirectional bandwidth). ~5-10 min and its numbers change only
# with the machine, so it is opt-in: RUN_TOPO=1 ./tmp/run_bench_dump.sh
if [ "${RUN_TOPO:-0}" = "1" ]; then
  run python tmp/topo.py
fi

# Envelope decomposition: what the measurement itself costs with ZERO
# kernel. dispatch_only = jit + shard_map + 8-dev launch + host sync;
# envelope_rs adds the exit reduce-scatter. (rows - envelope) = true
# in-kernel time; v1's 800us-profiler-vs-2682-bench gap says the
# envelope is ~1.9ms on his number too.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=env --iters=10 --warmup=3

# Ablate ladder: differential timing as the profiler substitute. Each
# row stubs ONE stage (output wrong on purpose); (none - X) = stage X's
# true wall-clock share. ablate=weights = all compute stubbed, weight
# fetch still streaming (measured 2026-08-29: 3247us, UNCHANGED from the
# window pipeline's 3266 despite the standalone blocked fetch probing 4.6x
# faster - the stream is NOT the floor). ablate=all additionally stubs
# the weight fetch and dispatch park: the bare per-step floor. (weights - all)
# = the stream's true in-situ cost.
for a in none masks gather ffn combine weights routing ag all; do
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v02 --iters=10 --warmup=3 \
      $DUMP_FLAGS --ablate=$a
done


# 3) per-stage spans: named_scope markers -> device trace. This is what
#    says WHERE the time goes (prologue vs gather vs gmms vs combine).
# (xprof removed: the profiler plugin ABI mismatches this libtpu
#  and segfaults in ProfilerSession::Create)

echo
echo "log: tmp/bench_dump.log  hist: tmp/op_histogram.txt  passes: tmp/mosaic_passes.tar.xz"
