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

# Keep only the passes we actually read (the kernel's own module chain),
# then compress: raw dumps run to hundreds of MB and blow the git limit.
run bash -c '
  mkdir -p tmp/mosaic_keep
  grep -rl "dynamic_rotate\|tpu.matmul\|enqueue_dma" tmp/mosaic_dump \
    | xargs -r -I{} cp {} tmp/mosaic_keep/
  tar -czf tmp/mosaic_dump.tgz -C tmp mosaic_keep
  rm -rf tmp/mosaic_dump tmp/mosaic_keep
  ls -lh tmp/mosaic_dump.tgz'

# 2) clean timing run (no dump overhead): v0.2 only - the v1 baseline is
#    deliberately out of the bring-up loop; it returns for the final
#    comparison once v0.2 is healthy on hardware.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=30 --warmup=5

# 3) per-stage spans: named_scope markers -> device trace. This is what
#    says WHERE the time goes (prologue vs gather vs gmms vs combine).
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=3 --warmup=1 --profile-dir=tmp/xprof
run bash -c 'tar -czf tmp/xprof.tgz -C tmp xprof && rm -rf tmp/xprof
             ls -lh tmp/xprof.tgz'

echo
echo "log: tmp/bench_dump.log  dumps: tmp/mosaic_dump.tgz  trace: tmp/xprof.tgz"
