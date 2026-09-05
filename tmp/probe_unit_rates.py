# Isolated unit-rate probes (VMEM-resident, Pallas): the two step-3
# constants the HBM-bound jnp measurements could not isolate.
#   A. exp() throughput on VMEM-resident f32 tiles (EUP rate;
#      theory: 1 op/cycle x 1024 elems = 1.13 Telem/s/core).
#   B. fp4(e2m1) -> e4m3 convert throughput on VMEM-resident data -
#      IF Pallas accepts float4 refs; if not, that refusal is itself
#      the Mosaic-reachability finding and is reported as such.
# Method: fori_loop n passes inside ONE kernel over resident tiles;
# rate = elems * n / (t(n_hi) - t(n_lo)) - dispatch cancels.
#
# Run:  python tmp/probe_unit_rates.py    (self-logs to
#       tmp/unit_rates_probe.log)
# Dump: rm -rf tmp/mosaic_units && mkdir -p tmp/mosaic_units
#       LIBTPU_INIT_ARGS=--xla_mosaic_dump_to=tmp/mosaic_units \
#         python tmp/probe_unit_rates.py --iters 2
#       (grep the LLO for the exp/convert op mnemonics + counts)

import argparse
import functools
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl


class _Tee:
    def __init__(self, path):
        self.f, self.stdout = open(path, "w"), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


CLOCK = 1.1e9
ROWS, LANES = 256, 512          # resident tile: 256x512 = 128K elems


def bench(fn, *args, iters=30):
    jfn = jax.jit(fn)
    jax.block_until_ready(jfn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def loop_kernel(body, n, x_ref, o_ref):
    def step(i, acc):
        return acc + body(x_ref[...])
    o_ref[...] = jax.lax.fori_loop(0, n, step,
                                   jnp.zeros_like(o_ref))


def rate(body, x, out_dtype, n_lo=32, n_hi=256, iters=30):
    ts = {}
    for n in (n_lo, n_hi):
        f = pl.pallas_call(
            functools.partial(loop_kernel, body, n),
            out_shape=jax.ShapeDtypeStruct(x.shape, out_dtype))
        ts[n] = bench(f, x, iters=iters)
    dt = (ts[n_hi] - ts[n_lo]) / (n_hi - n_lo)
    elems = x.size
    return elems / dt, dt * CLOCK / (elems / 1024)   # elem/s, cyc/vreg-ish


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()
    sys.stdout = _Tee("tmp/unit_rates_probe.log")
    print("jax", jax.__version__, jax.devices()[:1])
    rng = np.random.default_rng(0)

    # A. exp on resident f32
    x = jnp.asarray(-np.abs(rng.standard_normal((ROWS, LANES))),
                    jnp.float32)
    r, cyc = rate(lambda t: jnp.exp(t), x, jnp.float32,
                  iters=a.iters)
    print(f"A. exp f32 resident: {r/1e12:.2f} Telem/s "
          f"(~{cyc:.1f} cyc per 1024 elems; EUP theory 1.0)")

    # A2. sigmoid (the gate's actual op)
    r, cyc = rate(lambda t: jax.nn.sigmoid(t), x, jnp.float32,
                  iters=a.iters)
    print(f"A2. sigmoid f32 resident: {r/1e12:.2f} Telem/s "
          f"(~{cyc:.1f} cyc/1024)")

    # B. e2m1 -> e4m3 resident convert
    try:
        xf4 = jnp.asarray(rng.standard_normal((ROWS, LANES)),
                          jnp.float4_e2m1fn)
        r, cyc = rate(lambda t: t.astype(jnp.float8_e4m3fn)
                      .astype(jnp.float32),
                      xf4, jnp.float32, iters=a.iters)
        print(f"B. e2m1->e4m3(->f32 acc) resident: {r/1e12:.2f} "
              f"Telem/s (~{cyc:.1f} cyc/1024) "
              f"[NB includes the f32 upcast for the accumulator - "
              f"an upper bound on convert cost]")
    except Exception as e:
        print(f"B. FINDING: Pallas float4 ref/convert UNSUPPORTED "
              f"at this jax: {type(e).__name__}: {str(e)[:200]}")

    print("\ninterpretation: step-3 needs exp >= ~1 cyc/1024 (EUP "
          "full rate) and convert >= ~2 Telem/s (2 slots x 4096/op "
          "theory 9T; >=2T hides under the 24 ms MoE stream).")


if __name__ == "__main__":
    main()
