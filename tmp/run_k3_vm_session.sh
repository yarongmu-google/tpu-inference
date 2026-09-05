#!/usr/bin/env bash
# The complete K3 VM session - one paste covers everything queued:
#   1. KDA decode kernel: TPU goldens + v0 & slotted benches
#   2. Unit-rate probes (EUP exp/sigmoid, fp4 convert, resident)
#   3. Mosaic dump of the unit probes, packed for commit
# ~15 min total. Run from the repo root on the kdn branch.
set -uo pipefail
cd "$(dirname "$0")/.."
echo "branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"

bash tmp/run_kda_decode_bench.sh

python tmp/probe_unit_rates.py

rm -rf tmp/mosaic_units && mkdir -p tmp/mosaic_units
LIBTPU_INIT_ARGS=--xla_mosaic_dump_to=tmp/mosaic_units \
  python tmp/probe_unit_rates.py --iters 2 >/dev/null 2>&1 || true
python tmp/llo_loop_census.py tmp/mosaic_units | tee tmp/unit_rates_census.log
tar -c tmp/mosaic_units 2>/dev/null | xz -9 > tmp/mosaic_units.tar.xz
rm -rf tmp/mosaic_units

echo
echo "done. commit everything:"
echo "  git add tmp/ && git commit -m 'k3 vm session: kda benches + unit rates.' && git push origin kdn"
