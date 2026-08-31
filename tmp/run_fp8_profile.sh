#!/usr/bin/env bash
# STANDALONE device-time profiler for the v2 MoE kernel (fp8 winner,
# fp8 tensor arm, bf16 baseline) - the direct instrument behind the
# envelope-subtraction inference (run 6: winner 750.8 wall - 402.9
# envelope ~= 348 us device ~= the 357 us modeled MXU floor).
#
# Env requirement (lesson 22 - the pairing is the whole game):
# jax/jaxlib/libtpu must be a MATCHED set, newer than the serving
# pin. The preflight below compiles a trivial pallas kernel and
# aborts with remediation if the pairing is broken (the failure mode
# seen 2026-08-30: jax 0.11.1 with a stale libtpu ->
# "Pallas TPU requires a recent libtpu version (at least 0.0.44)").
#     conda activate prof
#     pip install -U 'jax[tpu]'     # pulls the matched libtpu
#     ./tmp/run_fp8_profile.sh
# Callable standalone, or from the main loop via
#     PROFILE=1 ./tmp/run_fp8_kernel.sh
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

# ---- preflight: pallas + libtpu pairing ----------------------------
run python - <<'EOF'
import jax, jax.numpy as jnp
from jax.experimental import pallas as pl


def _k(o_ref):
    o_ref[...] = jnp.ones_like(o_ref)


try:
    out = pl.pallas_call(
        _k, out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32))()
    jax.block_until_ready(out)
    print("preflight: pallas/libtpu pairing OK")
except Exception as ex:
    raise SystemExit(
        f"preflight FAILED: {type(ex).__name__}: {str(ex)[:200]}\n"
        "jax/jaxlib/libtpu are not a matched set. In THIS env run:\n"
        "    pip install -U 'jax[tpu]'\n"
        "then rerun. (Do NOT touch the serving env's pinned jax.)")
EOF

rm -rf tmp/fp8_xprof tmp/fp8_xprof.tar.xz

# fp8 token + tensor arms and the bf16 baseline, one profiled session
# each; the bench parses device-only kernel time per dispatch from
# the TensorCore pids (python tracer off, device tracer 2 - the
# lesson-A.5 recipe; barrier-core and trailing-copy edges subtracted).
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
# the experimental fused EP kernel (rs), fp8 arm: device-only number
# for the head-to-head with the v02 winner (wall A/B is step 3c of
# the main loop). Separate session so an rs failure costs nothing.
run python -m tpu_inference.kernels.fused_moe.v2.bench_decode \
    --wdtype=fp8 --variants=rs --iters=8 --warmup=3 \
    --profile-dir=tmp/fp8_xprof/rs

run bash -c '
  if [ -d tmp/fp8_xprof ]; then
    du -sh tmp/fp8_xprof
    tar -c tmp/fp8_xprof | xz -9 -T0 > tmp/fp8_xprof.tar.xz
    ls -lh tmp/fp8_xprof.tar.xz
    rm -rf tmp/fp8_xprof
  else
    echo "no traces captured (preflight passed but profiler produced"
    echo "nothing - check ProfileOptions support in this jax)"
  fi'

echo
echo "log: tmp/fp8_profile.log"
echo "verdicts: fp8 winner device vs the 357 us MXU-floor model;"
echo "token-vs-tensor device delta (envelope-free A/B); bf16 device"
echo "vs its historic 848 (the shared-path op deletions may have"
echo "moved it)"
