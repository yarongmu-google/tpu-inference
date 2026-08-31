#!/usr/bin/env bash
# Device-time profile of the fp8 winner vs the bf16 baseline - the
# direct instrument behind the envelope-subtraction inference
# (run 6: winner 750.8 wall - 402.9 envelope ~= 348 us device ~= the
# 357 us modeled MXU floor; this run confirms or corrects it).
#
# MUST run in the PROF env (jax 0.11.1 + matched libtpu; the serving
# env's profiler plugin segfaults in ProfilerSession::Create):
#     conda activate prof && ./tmp/run_fp8_profile.sh
#
# The bench parses device kernel ms/dispatch from the TensorCore pids
# (python_tracer_level=0, device_tracer_level=2; barrier-core and
# trailing-copy edges subtracted) - lesson A.5's recipe.
# Afterwards: git add tmp/ && commit ("fp8 profile.") && push.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/fp8_profile.log
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

# run-6 tuner winner; override with FLAGS=...
FLAGS="${FLAGS:---be=8 --bg=2 --capacity=32 --bd1c=256 --bd2c=128 --bcT=0}"
echo "FLAGS: $FLAGS"

run git rev-parse --short HEAD
run python -c "import jax, jaxlib; print('jax', jax.__version__, 'jaxlib', jaxlib.__version__)"

rm -rf tmp/fp8_xprof tmp/fp8_xprof.tar.xz

# fp8 token + tensor arms and the bf16 baseline, one profiled session
# each; the parser prints device-only kernel time per dispatch.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --variants=v02 --iters=8 --warmup=3 $FLAGS \
    --profile-dir=tmp/fp8_xprof/token
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --act-scale=tensor --variants=v02 --iters=8 --warmup=3 \
    $FLAGS --profile-dir=tmp/fp8_xprof/tensor
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --variants=v02 --iters=8 --warmup=3 \
    --be=4 --bg=4 --capacity=32 --bd1c=256 --bd2c=128 --bcT=256 \
    --profile-dir=tmp/fp8_xprof/bf16

run bash -c '
  if [ -d tmp/fp8_xprof ]; then
    du -sh tmp/fp8_xprof
    tar -c tmp/fp8_xprof | xz -9 -T0 > tmp/fp8_xprof.tar.xz
    ls -lh tmp/fp8_xprof.tar.xz
    rm -rf tmp/fp8_xprof
  else
    echo "no traces captured (wrong env? conda activate prof)"
  fi'

echo
echo "log: tmp/fp8_profile.log"
echo "verdicts: fp8 winner device vs the 357 us MXU-floor model;"
echo "bf16 device vs its historic 848 (the VALU-bound reinterpretation"
echo "+ this session's shared-path op reductions may have moved it)"
