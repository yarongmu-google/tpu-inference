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

FLAGS="${FLAGS:---be=8 --bg=4 --capacity=32 --bd2c=1024 --wbuf=4}"
echo "FLAGS: $FLAGS"

run git rev-parse --short HEAD

run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=env --iters=10 --warmup=3

# same-session none baseline so the ag delta is not cross-run noise
for a in none ag all; do
  run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
      --variants=v02 --iters=10 --warmup=3 $FLAGS --ablate=$a
done

echo
echo "log: tmp/floor_decomp.log"
