#!/usr/bin/env bash
# Device-time 3-way: v1 (fused_ep_moe), rs (experimental fused EP,
# via the jax-0.11 shim in bench_decode), v02s (our serving entry,
# tuned blocks) - one profiled session PER VARIANT so a crash costs
# only its own row. T=512 (the tuned shape) and T=64 (the SERVING
# decode shape on the hybrid model) both run.
#
# The v02s sessions also profile the fused-entry shadow row (substring
# selection); the winner FLAGS below make that row the TUNED fused
# entry (last run it silently used be=4/bg=1 defaults - ignore those
# rows in older logs). bf16's bcT=256 is only legal at T=512.
#
# ENV: needs the matched jax/libtpu pair (prof env) - preflight below.
#     conda activate prof && ./tmp/run_kernel_profiles.sh
# The summary at the end tags every row with T= and dtype; TC median
# is the verdict, the SC column shows SparseCore involvement.
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
FP8_FLAGS="--be=8 --bg=2 --capacity=32 --bd1c=256 --bd2c=128 --bcT=0"

for T in 64 512; do
  BF16_FLAGS="--be=4 --bg=4 --capacity=32 --bd1c=256 --bd2c=128"
  [ "$T" -ge 512 ] && BF16_FLAGS="$BF16_FLAGS --bcT=256"

  # bf16: all three kernels
  run $BENCH --variants=v1 --tokens=$T --iters=8 --warmup=3 \
      --profile-dir=tmp/kprof/bf16_v1_t${T}
  run $BENCH --variants=rs --tokens=$T --iters=8 --warmup=3 \
      --profile-dir=tmp/kprof/bf16_rs_t${T}
  run $BENCH --variants=v02s --tokens=$T --iters=8 --warmup=3 \
      $BF16_FLAGS --profile-dir=tmp/kprof/bf16_v02s_t${T}

  # fp8: v1 has no fp8 wiring in the bench
  run $BENCH --wdtype=fp8 --variants=rs --tokens=$T --iters=8 --warmup=3 \
      --profile-dir=tmp/kprof/fp8_rs_t${T}
  run $BENCH --wdtype=fp8 --variants=v02s --tokens=$T --iters=8 --warmup=3 \
      $FP8_FLAGS --profile-dir=tmp/kprof/fp8_v02s_t${T}

  # the STOCK XLA GMM path - the serving baseline no hand-written
  # kernel table ever included: is the default actually slower?
  for GV in gmm_ep gmm_tp; do
    run $BENCH --variants=$GV --tokens=$T --iters=8 --warmup=3 \
        --profile-dir=tmp/kprof/bf16_${GV}_t${T}
    run $BENCH --wdtype=fp8 --variants=$GV --tokens=$T --iters=8 --warmup=3 \
        --profile-dir=tmp/kprof/fp8_${GV}_t${T}
  done
done

echo
echo "=== DEVICE-TIME SUMMARY (TC median is the verdict) ==="
awk '
  /^\+ python/ {
    t = "?"; d = "bf16"
    for (i = 1; i <= NF; i++) {
      if ($i ~ /^--tokens=/)   { split($i, a, "="); t = a[2] }
      if ($i == "--wdtype=fp8") { d = "fp8" }
    }
  }
  /device-only per dispatch/ { printf "T=%-4s %-5s %s\n", t, d, $0 }
' "$LOG"
run bash -c 'tar -c tmp/kprof | xz -9 -T0 > tmp/kprof.tar.xz; ls -lh tmp/kprof.tar.xz; rm -rf tmp/kprof'
echo "log: tmp/kernel_profiles.log"
