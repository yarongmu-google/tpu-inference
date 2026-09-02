#!/usr/bin/env bash
# Big-T tuner: the MNS=128 operating point runs decode steps at
# T=1024 global and mixed steps at ~1280-1536 (decode + prefill
# riders past the power-of-2 padding). Those shapes currently use the
# serving entry's fallback blocks (be=4/bg=1, whole-D) - the 4c run
# measured ~87ms hill TPOT largely from this. This script tunes the
# kernel at each shape; afterwards, refill the _T_BLOCKS table in
# fused_moe_decode_tp_serving (decode_kernel.py) and re-run the
# per-shape v02s check below with the winners.
#
# Tuner candidates flow through the kernel's VMEM assert, so
# oversized configs (be=8 at T>=1024) are pruned, not fatal.
#
# ENV: serving env (jax 0.10.x). ~15 min per shape for the tune.
# Afterwards: git add tmp/ && commit ("bigT tune.") && push.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/bigT_tune.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

run() { echo; echo "+ $*"; "$@"; echo "+ exit=$?"; }

export PYTHONPATH=.
export JAX_PLATFORMS=tpu
unset LIBTPU_INIT_ARGS

run git rev-parse --short HEAD
JAXV=$(python -c "import jax; print(jax.__version__)" 2>/dev/null || echo none)
case "$JAXV" in
  0.10.*) echo "env preflight: jax $JAXV OK" ;;
  *) echo "preflight FAILED: jax=$JAXV is not the serving env" >&2; exit 1 ;;
esac

BENCH="python -m tpu_inference.kernels.fused_moe.v2.bench_decode"

for T in 1024 1280 1536; do
  echo; echo "########## T=$T fp8 tune ##########"
  run $BENCH --wdtype=fp8 --tune --tokens=$T --iters=10 --warmup=3
  # baseline row at the CURRENT serving-entry blocks for the delta
  run $BENCH --wdtype=fp8 --variants=v02s --tokens=$T --iters=15 --warmup=3
done

echo
echo "log: tmp/bigT_tune.log"
echo "next: paste each T's WINNER into _T_BLOCKS in decode_kernel.py's"
echo "serving entry (and bd1c/bd2c/bcT if the winners moved off"
echo "256/128/0), re-run the v02s rows at the winners, then line 4c."
