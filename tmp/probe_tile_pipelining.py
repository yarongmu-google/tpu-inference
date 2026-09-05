# Tile-pipelining probe: does Mosaic let small matmul tiles run
# back-to-back with fill paid once, or is fill exposed per tile?
#
# A per-chunk linear-attention step is a chain of SMALL matmuls
# ((C,128)x(128,C), (C,C)x(C,C), (C,C)x(C,256)). Scheduled in
# isolation such a chain costs ~300-400 cyc/tile, almost all
# pipeline fill (~20-50x its pure MAC cost). The whole kernel plan
# rests on amortizing that fill across tiles. Two variants isolate
# where fill could be exposed:
#   LOOP: one kernel, fori_loop over the chain on VMEM-resident
#         data - pure MXU chaining, no DMA, no grid.
#   GRID: grid over tiles, each fetching its input block from HBM -
#         adds the fetch/compute overlap question.
# Marginal cost/tile = (t(n2) - t(n1)) / (n2 - n1). Verdict:
#   marginal ~ pure MAC (few cyc)  -> fill amortizes, plan holds
#   marginal ~ isolated (~300 cyc) -> fill exposed, P1 doubles+
#
# Run (serving env):  python tmp/probe_tile_pipelining.py
# Output tees to tmp/tile_pipelining_probe.log.
#
# Codegen inspection (recommended second run):
#   rm -rf tmp/mosaic_tilepipe && mkdir -p tmp/mosaic_tilepipe
#   LIBTPU_INIT_ARGS=--xla_mosaic_dump_to=tmp/mosaic_tilepipe \
#     python tmp/probe_tile_pipelining.py --iters 2
#   tar -c tmp/mosaic_tilepipe | xz -9 > tmp/mosaic_tilepipe.tar.xz
# In the dump, the tell is vmatmul/vmatpush density between tile
# boundaries: long scoreboard-wait gaps between tiles = exposed fill.

import argparse
import sys
import time
from functools import partial

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


C, K, N2 = 256, 128, 256
CLOCK_GHZ = 1.1


def chain(x, y):
    # representative per-chunk chain: kkT -> square transform -> apply
    a = jnp.dot(x, x.T, preferred_element_type=jnp.float32)      # C,C
    b = jnp.dot(a.astype(jnp.bfloat16), a.astype(jnp.bfloat16),
                preferred_element_type=jnp.float32)              # C,C
    return jnp.dot(b.astype(jnp.bfloat16), y,
                   preferred_element_type=jnp.float32)           # C,N2


def loop_kernel(n, x_ref, y_ref, o_ref):
    def body(i, acc):
        return acc + chain(x_ref[...], y_ref[...])
    o_ref[...] = jax.lax.fori_loop(0, n, body, jnp.zeros((C, N2),
                                                         jnp.float32))


def grid_kernel(x_ref, y_ref, o_ref):
    o_ref[...] = chain(x_ref[...], y_ref[...])


def bench(fn, *args, iters=30):
    jfn = jax.jit(fn)
    jax.block_until_ready(jfn(*args))
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        jax.block_until_ready(jfn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=30)
    a = p.parse_args()
    sys.stdout = _Tee("tmp/tile_pipelining_probe.log")
    print("jax", jax.__version__, jax.devices()[:1])
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal((C, K)), jnp.bfloat16)
    y = jnp.asarray(rng.standard_normal((C, N2)), jnp.bfloat16)

    print(f"\nLOOP variant (VMEM-resident, no DMA), chain = "
          f"(C,K)x(K,C) -> (C,C)^2 -> (C,C)x(C,{N2}), C={C}")
    print(f"{'n':>5} {'ms':>9} {'cyc/tile':>9}")
    prev = None
    for n in (1, 2, 4, 8, 16, 32, 64):
        f = pl.pallas_call(
            partial(loop_kernel, n),
            out_shape=jax.ShapeDtypeStruct((C, N2), jnp.float32))
        t = bench(f, x, y, iters=a.iters)
        cyc = t * CLOCK_GHZ * 1e9 / n
        marg = ""
        if prev is not None:
            pn, pt = prev
            marg = f"  marginal {((t-pt)/(n-pn))*CLOCK_GHZ*1e9:8.0f}"
        print(f"{n:>5} {t*1e3:9.3f} {cyc:9.0f}{marg}")
        prev = (n, t)

    print(f"\nGRID variant (per-tile HBM fetch)")
    print(f"{'n':>5} {'ms':>9} {'cyc/tile':>9}")
    prev = None
    for n in (1, 2, 4, 8, 16, 32, 64):
        xs = jnp.asarray(rng.standard_normal((n * C, K)), jnp.bfloat16)
        f = pl.pallas_call(
            grid_kernel,
            grid=(n,),
            in_specs=[pl.BlockSpec((C, K), lambda i: (i, 0)),
                      pl.BlockSpec((C, N2), lambda i: (0, 0))],
            out_specs=pl.BlockSpec((C, N2), lambda i: (i, 0)),
            out_shape=jax.ShapeDtypeStruct((n * C, N2), jnp.float32))
        t = bench(f, xs, y, iters=a.iters)
        cyc = t * CLOCK_GHZ * 1e9 / n
        marg = ""
        if prev is not None:
            pn, pt = prev
            marg = f"  marginal {((t-pt)/(n-pn))*CLOCK_GHZ*1e9:8.0f}"
        print(f"{n:>5} {t*1e3:9.3f} {cyc:9.0f}{marg}")
        prev = (n, t)

    print("\nreference points: isolated scheduled tile ~300-400 cyc; "
          "pure MAC for this chain ~200 cyc (3 matmuls, 33.6M MAC, "
          "x0.5 width util). marginal near ~200-400 = fill amortized; "
          "marginal >> 400 = fill exposed per tile.")


if __name__ == "__main__":
    main()
