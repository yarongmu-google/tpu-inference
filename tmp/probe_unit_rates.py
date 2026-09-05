# Isolated unit-rate probes (VMEM-resident, Pallas): the two step-3
# constants the HBM-bound jnp measurements could not isolate.
#   A.  exp() throughput on resident f32 tiles (EUP rate).
#   A2. sigmoid() throughput (the gate op: exp + reciprocal on the EUP).
#   B.  fp4(e2m1) -> e4m3 -> f32/bf16 convert throughput.
#
# Method: fori_loop n passes inside ONE kernel; the per-iteration cost
# is the SLOPE of a least-squares line through t(n) over an n sweep,
# so dispatch cancels and linearity is visible. Every point prints its
# absolute min/median time; a result is VALID only when the swept
# time span exceeds 10x the observed jitter (median - min). Runs 1/2a
# (n = 32..256, spans of 4-30 us) sat inside a >= 10 us dispatch
# jitter band and printed negative and >6 Telem/s "rates" that no
# slot budget allows; the n sweep is what makes the number a
# measurement.
#
# Loop bodies are loop-VARIANT by construction:
#   chained  - acc = body(acc * 0.999): the value chains iteration to
#              iteration (per-vreg chains; 16..128 independent chains
#              per iteration keep the EUP pipeline full).
#   rotating - acc += body(x_ref[i % T]): a different resident tile
#              each iteration (dynamic leading index), accumulator
#              sized to stay in vregs so the probe is not bound by its
#              own accumulator spill traffic (run 4's 256x512 f32
#              accumulator = 128 vregs -> 128 vst/iteration = the
#              1 cyc/1024 store floor, which is what 1.3 Telem/s was).
# Interpret every rate against the in-loop op census from the Mosaic
# dump (tmp/llo_loop_census.py), never against theory alone.
#
# Run:  python tmp/probe_unit_rates.py    (self-logs to
#       tmp/unit_rates_probe.log)
# Dump: rm -rf tmp/mosaic_units && mkdir -p tmp/mosaic_units
#       LIBTPU_INIT_ARGS=--xla_mosaic_dump_to=tmp/mosaic_units \
#         python tmp/probe_unit_rates.py --iters 2
#       python tmp/llo_loop_census.py tmp/mosaic_units

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


CLOCK = 1.1e9                     # TensorCore clock, GHz (spec sheet)
LANES = 512
NS = (128, 512, 2048, 8192)       # n sweep for the slope fit
VALID_SPAN_OVER_JITTER = 10.0


def bench(fn, *args, iters=30):
    jfn = jax.jit(fn)
    jax.block_until_ready(jfn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*args))
        ts.append(time.perf_counter() - t0)
    ts = np.sort(np.asarray(ts))
    return float(ts[0]), float(np.median(ts))


def chained_kernel(body, n, x_ref, o_ref):
    def step(i, acc):
        return body(acc * 0.999)
    o_ref[...] = jax.lax.fori_loop(
        0, n, step, x_ref[...].astype(o_ref.dtype))


def rotating_kernel(body, n, x_ref, o_ref):
    # x_ref: [T, R, L]; acc: [R, L] in o_ref's dtype
    T = x_ref.shape[0]

    def step(i, acc):
        return acc + body(x_ref[jax.lax.rem(i, T)])
    o_ref[...] = jax.lax.fori_loop(0, n, step, jnp.zeros_like(o_ref))


def measure(name, kernel, body, x, out_shape, out_dtype, iters):
    """Fit t(n) = t0 + slope*n over NS; report elems/s of the body."""
    elems = int(np.prod(out_shape))
    tmin, tmed = {}, {}
    for n in NS:
        f = pl.pallas_call(
            functools.partial(kernel, body, n),
            out_shape=jax.ShapeDtypeStruct(out_shape, out_dtype))
        tmin[n], tmed[n] = bench(f, x, iters=iters)
    ns = np.asarray(NS, dtype=np.float64)
    ts = np.asarray([tmin[n] for n in NS])
    slope, t0 = np.polyfit(ns, ts, 1)
    resid = np.max(np.abs(ts - (t0 + slope * ns))) / max(ts[-1], 1e-12)
    jitter = max(tmed[n] - tmin[n] for n in NS)
    span = tmin[NS[-1]] - tmin[NS[0]]
    valid = span > VALID_SPAN_OVER_JITTER * jitter and slope > 0
    rate = elems / slope
    cyc = slope * CLOCK / (elems / 1024)
    print(f"{name}")
    for n in NS:
        print(f"    n={n:5d}  min {tmin[n]*1e6:9.1f} us  "
              f"median {tmed[n]*1e6:9.1f} us")
    print(f"    slope {slope*1e9:8.1f} ns/iter  fit-resid {resid*100:.1f}%  "
          f"span {span*1e6:.1f} us  jitter {jitter*1e6:.1f} us  "
          f"-> {'VALID' if valid else 'INVALID (span < 10x jitter)'}")
    print(f"    => {rate/1e12:.2f} Telem/s  = {cyc:.2f} cyc per 1024 elems")
    return rate, cyc, valid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()
    sys.stdout = _Tee("tmp/unit_rates_probe.log")
    print("jax", jax.__version__, jax.devices()[:1])
    print(f"n sweep {NS}; clock {CLOCK/1e9:.2f} GHz; "
          f"valid iff span > {VALID_SPAN_OVER_JITTER:.0f}x jitter")
    rng = np.random.default_rng(0)

    # A / A2: chained EUP ops. rows=32 -> 16 vregs (no spill, 16
    # chains); rows=256 -> 128 vregs (run-4 form: spills, 128 chains).
    for rows in (32, 256):
        x = jnp.asarray(-np.abs(rng.standard_normal((rows, LANES))),
                        jnp.float32)
        measure(f"A.  exp f32 chained, tile {rows}x{LANES} "
                f"({rows*LANES//1024} vregs)", chained_kernel,
                jnp.exp, x, x.shape, jnp.float32, a.iters)
        measure(f"A2. sigmoid f32 chained, tile {rows}x{LANES} "
                f"({rows*LANES//1024} vregs)", chained_kernel,
                jax.nn.sigmoid, x, x.shape, jnp.float32, a.iters)

    # B: e2m1 -> e4m3 -> acc-dtype convert on rotating resident tiles.
    # fp4 tile second-minor must be a multiple of 8*packing = 64.
    T, rows = 4, 64
    try:
        xf4 = jnp.asarray(rng.standard_normal((T, rows, LANES)),
                          jnp.float4_e2m1fn)
        for acc_dtype, tag in ((jnp.float32, "f32 acc, 32 vregs"),
                               (jnp.bfloat16, "bf16 acc, 16 vregs")):
            body = lambda t, d=acc_dtype: t.astype(jnp.float8_e4m3fn).astype(d)
            measure(f"B.  e2m1->e4m3->{jnp.dtype(acc_dtype).name} rotating "
                    f"{T}x{rows}x{LANES} ({tag})", rotating_kernel,
                    body, xf4, (rows, LANES), acc_dtype, a.iters)
        # run-4 form for continuity: 256x512 accumulator (128 vregs,
        # spilled) - expected to sit on the 1 vst/vreg store floor.
        xf4_big = jnp.asarray(rng.standard_normal((1, 256, LANES)),
                              jnp.float4_e2m1fn)
        measure("B0. e2m1->e4m3->f32 rotating 1x256x512 (run-4 form, "
                "128-vreg acc spilled: store-floor reference)",
                rotating_kernel,
                lambda t: t.astype(jnp.float8_e4m3fn).astype(jnp.float32),
                xf4_big, (256, LANES), jnp.float32, a.iters)
    except Exception as e:
        print(f"B. FINDING: Pallas float4 ref/convert UNSUPPORTED "
              f"at this jax: {type(e).__name__}: {str(e)[:300]}")

    print("\nread-out: 1 EUP push per vreg = 1.0 cyc/1024 (exp); sigmoid "
          "lowers to vexp + vrecip = 2 pushes; the step-3 gate budget "
          "uses the sigmoid figure. Convert: compare cyc/1024 with the "
          "in-loop vcvt/vunpack/vadd census from tmp/llo_loop_census.py.")


if __name__ == "__main__":
    main()
