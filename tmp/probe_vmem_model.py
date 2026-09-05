# Calibrate the whole-kernel VMEM footprint model.
#
# Compiles a family of trivial Pallas kernels whose operand set is
# varied one axis at a time, catches the OOM message (which reports
# the exact scoped requirement), and prints measured-vs-modeled so
# the buffering depth and per-operand charges can be fitted:
#   - N operands of the same block  -> per-operand charge
#   - one operand, growing block    -> block scaling
#   - index_map constant vs varying -> buffering depth
#   - aliased in/out vs separate    -> is an alias charged twice?
#   - scalar-prefetch present/absent-> does it deepen buffering?
#
# Run: python tmp/probe_vmem_model.py   (self-logs)

import functools
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class _Tee:
    def __init__(self, path):
        self.f, self.stdout = open(path, "w"), sys.stdout

    def write(self, s):
        self.stdout.write(s); self.f.write(s)

    def flush(self):
        self.stdout.flush(); self.f.flush()


def scoped_mb(fn, *args):
    """Compile; return (ok, scoped_MB_required_or_None)."""
    try:
        jax.block_until_ready(jax.jit(fn)(*args))
        return True, None
    except Exception as e:
        m = re.search(r"Scoped allocation with size ([\d.]+)M", str(e))
        if m:
            return False, float(m.group(1))
        m = re.search(r"Used ([\d.]+)M of", str(e))
        return False, float(m.group(1)) if m else None


def build(n_ops, rows, varying=True, prefetch=False, alias=False):
    H, K = 96, 128
    blk = (rows, H, K)

    def body(*refs):
        outs = refs[-1]
        acc = jnp.zeros_like(outs[...] if hasattr(outs, "__getitem__")
                             else outs)
        for r in refs[:-1]:
            if hasattr(r, "shape") or hasattr(r, "__getitem__"):
                try:
                    acc = acc + r[...]
                except Exception:
                    pass
        outs[...] = acc

    idx_map = ((lambda i: (i, 0, 0)) if varying
               else (lambda i: (0, 0, 0)))
    B = rows * 4
    xs = [jnp.zeros((B, H, K), jnp.float32) for _ in range(n_ops)]
    specs = [pl.BlockSpec(blk, idx_map) for _ in range(n_ops)]
    f = pl.pallas_call(
        body, grid=(4,), in_specs=specs,
        out_specs=pl.BlockSpec(blk, idx_map),
        out_shape=jax.ShapeDtypeStruct((B, H, K), jnp.float32),
        **({"input_output_aliases": {0: 0}} if alias else {}))
    return f, xs


def main():
    sys.stdout = _Tee("tmp/vmem_model_probe.log")
    print("jax", jax.__version__, jax.devices()[:1])
    sys.path.insert(0, ".")
    print(f"\n{'case':38} {'measured MB':>12}")
    for n_ops in (1, 2, 4, 8):
        f, xs = build(n_ops, rows=64)
        ok, mb = scoped_mb(f, *xs)
        print(f"{f'{n_ops} operands, rows=64':38} "
              f"{'fits' if ok else f'{mb}':>12}")
    for rows in (16, 32, 64, 128, 256):
        f, xs = build(4, rows=rows)
        ok, mb = scoped_mb(f, *xs)
        print(f"{f'4 operands, rows={rows}':38} "
              f"{'fits' if ok else f'{mb}':>12}")
    for varying in (True, False):
        f, xs = build(4, rows=128, varying=varying)
        ok, mb = scoped_mb(f, *xs)
        print(f"{f'4 ops rows=128 varying={varying}':38} "
              f"{'fits' if ok else f'{mb}':>12}")
    for alias in (False, True):
        f, xs = build(4, rows=128, alias=alias)
        ok, mb = scoped_mb(f, *xs)
        print(f"{f'4 ops rows=128 alias={alias}':38} "
              f"{'fits' if ok else f'{mb}':>12}")
    print("\nfit: per-operand charge = d(MB)/d(n_ops); buffering = "
          "measured / (block_bytes x n_ops); alias delta answers "
          "double-charging.")


if __name__ == "__main__":
    main()
