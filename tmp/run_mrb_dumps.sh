#!/usr/bin/env bash
# MRB probe dump pair - two one-minute diagnostics:
#   1) variant D (manual MXU + dot_general in one program) under a
#      mosaic dump: the compile FAILS by design ("Low-level MXU
#      operations cannot be mixed..."); the per-pass dump files show
#      how far the pipeline got - the last file names the frontier
#      pass where (or just before) the mixing check lives.
#   2) variant A (chunked dot_general, isolated) under a dump: the
#      histogram answers THE question - does Mosaic MRB-accumulate
#      the acc+=dot chain when nothing else is in the program (few/no
#      vadd.f32), or drain-and-vadd per chunk? Few vadds => the real
#      kernel's measured vadd chain is a CONTEXT artifact and trim #4
#      becomes a jax-level combine restructure, no manual mode needed.
#
# ENV: serving env (jax 0.10.x).
# Afterwards: git add tmp/ && commit ("mrb dumps.") && push.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG=tmp/mrb_dumps.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1

run() { echo; echo "+ $*"; "$@"; echo "+ exit=$?"; }

export PYTHONPATH=.
export JAX_PLATFORMS=tpu
unset LIBTPU_INIT_ARGS

run git rev-parse --short HEAD

# ---- 1) D under a dump: locate the frontier pass -------------------
rm -rf tmp/mosaic_D && mkdir -p tmp/mosaic_D
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_D"
run python tmp/probe_mrb_combine.py --variant D_coexist --iters 1
unset LIBTPU_INIT_ARGS
echo "--- D dump files (last = frontier pass) ---"
ls tmp/mosaic_D | tail -5

# ---- 2) A under a dump: vadd chain or MRB store-add? ---------------
rm -rf tmp/mosaic_A && mkdir -p tmp/mosaic_A
export LIBTPU_INIT_ARGS="--xla_mosaic_dump_to=tmp/mosaic_A"
run python tmp/probe_mrb_combine.py --variant A_dot_chunked --iters 2
unset LIBTPU_INIT_ARGS
LAST=$(ls tmp/mosaic_A/*post-finalize-llo* 2>/dev/null | head -1)
if [ -n "$LAST" ]; then
  run python tmp/dump_histogram.py "$LAST"
  python tmp/dump_histogram.py "$LAST" > tmp/mrb_A_histogram.txt
  echo "--- verdict lines (A: 64 chunks -> ~63 drains+adds if NOT"
  echo "--- fused; few vadd.f32 = Mosaic MRB-accumulates in isolation) ---"
  grep -E "vadd|vmatmul|vmatres|vmatprep" tmp/mrb_A_histogram.txt | head -8
fi

# ---- pack the dumps for the push -----------------------------------
run bash -c 'tar -c tmp/mosaic_D tmp/mosaic_A | xz -9 -T0 > tmp/mrb_dumps.tar.xz; ls -lh tmp/mrb_dumps.tar.xz; rm -rf tmp/mosaic_D tmp/mosaic_A'

echo
echo "log: tmp/mrb_dumps.log  histogram: tmp/mrb_A_histogram.txt"
echo "then: git add tmp/ && git commit -m 'mrb dumps.' && git push"
