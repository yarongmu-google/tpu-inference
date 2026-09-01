#!/usr/bin/env bash
# Device-time 3-way: v1 (fused_ep_moe), rs (experimental fused EP),
# v02s (our serving entry, tuned blocks) - one profiled session PER
# VARIANT so a crash (the rs fp8 row dies at trace today) costs only
# its own row. T=512 (the tuned shape) and T=64 (the SERVING decode
# shape on the hybrid model) both run: at MNS=64 serving, T=64 device
# time is the number that matters.
#
# ENV: needs the matched jax/libtpu pair (prof env) - preflight below.
#     conda activate prof && ./tmp/run_kernel_profiles.sh
# Wall rows print too, but the "device-only per dispatch" lines are
# the verdict (TC median; SC column shows any SparseCore involvement).
# Afterwards: git add tmp/ && commit ("kernel profiles.") && push.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/kernel_profiles.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

run() { echo; echo "+ $*"; "$@"; echo "+ exit=$?"; }

export PYTHONPATH=.
export JAX_PLATFORMS=tpu
unset LIBTPU_INIT_ARGS

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

rm -rf tmp/kprof tmp/kprof.tar.xz && mkdir -p tmp/kprof

BENCH="python -m tpu_inference.kernels.fused_moe.v2.bench_decode"

for T in 64 512; do
  # bf16: all three kernels
  for V in v1 rs v02s; do
    run $BENCH --variants=$V --tokens=$T --iters=8 --warmup=3 \
        --profile-dir=tmp/kprof/bf16_${V}_t${T}
  done
  # fp8: v1 has no fp8 wiring in the bench; rs's fp8 row currently
  # dies at trace - keep it so the full error lands in this log
  for V in rs v02s; do
    run $BENCH --wdtype=fp8 --variants=$V --tokens=$T --iters=8 --warmup=3 \
        --profile-dir=tmp/kprof/fp8_${V}_t${T}
  done
done

echo
echo "=== DEVICE-TIME SUMMARY (TC median is the verdict) ==="
grep -h "device-only per dispatch" "$LOG" || true
run bash -c 'tar -c tmp/kprof | xz -9 -T0 > tmp/kprof.tar.xz; ls -lh tmp/kprof.tar.xz; rm -rf tmp/kprof'
echo "log: tmp/kernel_profiles.log"
