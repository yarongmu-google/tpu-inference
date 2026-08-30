#!/usr/bin/env bash
# Fast floor decomposition (no tuner, ~3 min): splits the ~2.8ms fixed
# per-call cost into harness / reduce-scatter / all-gather+barrier.
#
#   v2_dispatch_only              = harness (jit + shard_map + launch)
#   v2_envelope_rs - dispatch     = the exit reduce-scatter
#   none - ablate=ag              = in-kernel AG + barrier
#   ablate=all - (harness + RS)   = prologue + whatever remains
#
# Runs at the last tuned winner; override: FLAGS="..." ./tmp/run_floor_decomp.sh
#
# Run in the PROF env (jax 0.11.1 + matched libtpu) so the profiler
# works: conda activate prof && ./tmp/run_floor_decomp.sh
# Afterwards everything in tmp/ is push-sized: git add tmp/ && commit.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/floor_decomp.log
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

FLAGS="${FLAGS:---be=4 --bg=4 --capacity=32 --bd2c=1024}"
echo "FLAGS: $FLAGS"

run git rev-parse --short HEAD

run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=env --iters=10 --warmup=3

# same-session none baseline so the ag delta is not cross-run noise
for a in none ag all; do
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v02 --iters=10 --warmup=3 $FLAGS --ablate=$a
done

# ---- the ~2.3ms unaccounted remainder (2026-08-29 decomp: all=2772 but
# harness 354 + RS 53 + AG 29 + routing ~0). Hypothesis: XLA inserts a
# per-call HBM copy/layout-conversion of the 1.5 GiB weight operands
# before the kernel - invisible to Mosaic dumps, unchanged by every
# kernel-internal change, and matching v1's identical ~2ms
# bench-vs-profiler envelope. Discriminators:
# (a) weight-size scaling: E=64 is 8x fewer weight bytes - if
#     ablate=all scales with E, it is a weight-bytes copy.
for ex in 64 128 256; do
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v02 --iters=10 --warmup=3 --experts=$ex $FLAGS --ablate=all
done
# (b) the XLA (not Mosaic) HLO of the ablate=all program: grep for big
#     copies/converts of the f16 weight shapes.
rm -rf tmp/xla_dump && mkdir -p tmp/xla_dump
XLA_FLAGS="--xla_dump_to=tmp/xla_dump --xla_dump_hlo_as_text" \
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v02 --iters=2 --warmup=1 $FLAGS --ablate=all
run bash -c '
  ls tmp/xla_dump | wc -l
  grep -l "512,4096,256" tmp/xla_dump/*after_optimizations*.txt 2>/dev/null | head -3
  for f in tmp/xla_dump/*after_optimizations*.txt; do
    grep -nE "copy|bitcast-convert|transpose" "$f" | grep -E "512,4096|4096,256|128,4096" | head -20
  done 2>/dev/null | head -40
  tar -c tmp/xla_dump | xz -9 -T0 > tmp/xla_dump.tar.xz
  ls -lh tmp/xla_dump.tar.xz
  rm -rf tmp/xla_dump'

# v1 XLA-copy audit: the same trap check that caught our 2.3ms - dump
# v1's compiled program and grep for weight-shaped copy/reshape ops
# between its parameters and the custom call. Clean output here + the
# device-only profile below = the "did we really halve v1" verdict.
rm -rf tmp/xla_dump_v1 && mkdir -p tmp/xla_dump_v1
XLA_FLAGS="--xla_dump_to=tmp/xla_dump_v1 --xla_dump_hlo_as_text" \
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v1 --iters=2 --warmup=1
run bash -c '
  for f in tmp/xla_dump_v1/*jit*.after_optimizations.txt; do
    echo "== $(basename $f)"
    grep -nE "%(copy|reshape|transpose)" "$f" \
      | grep -E "512,2,4096|2,4096,1024|512,1024,4096|4096,1024" | head -10
  done
  tar -c tmp/xla_dump_v1 | xz -9 -T0 > tmp/xla_dump_v1.tar.xz
  ls -lh tmp/xla_dump_v1.tar.xz
  rm -rf tmp/xla_dump_v1'

# Device-only profile (the stacked_rpa recipe: python tracer off,
# device tracer 2, parsed from TensorCore pids): kernel-only numbers
# for v1 AND v2 in one shot - the measurement that would have caught
# the XLA reshape copy on day one.
# pre-clean: start_trace appends session dirs, and the parser globs
# EVERY *.trace.json.gz under the dir - stale traces from an earlier
# (or crashed) run would pollute the parse
rm -rf tmp/moe_xprof tmp/moe_xprof.tar.xz
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v1,v02 --iters=5 --warmup=2 $FLAGS \
    --profile-dir=tmp/moe_xprof
# raw traces are too big to push - the parsed TC/SC rows above are the
# result; the tarball keeps the evidence pushable. (If the profiler
# segfaulted - wrong jax/libtpu pairing, run in the prof env - there is
# no dir and this block just says so.)
run bash -c '
  if [ -d tmp/moe_xprof ]; then
    du -sh tmp/moe_xprof
    tar -c tmp/moe_xprof | xz -9 -T0 > tmp/moe_xprof.tar.xz
    ls -lh tmp/moe_xprof.tar.xz
    rm -rf tmp/moe_xprof
  else
    echo "no traces captured (profiler unavailable in this env?)"
  fi'

echo
echo "log: tmp/floor_decomp.log"
