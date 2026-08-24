"""Does a bf16 matmul operand cost extra when it arrives via a pipelined
window instead of VMEM scratch?

Same matmul twice. A: the pushed operand is read straight from the
pipelined input window. B: it is copied into VMEM scratch at step 0 and
the matmul reads the scratch. If the window tiling is the problem, A
carries a pile of vunpack/vpack that B does not.

    python tmp/probe_tiling.py            # prints op counts per variant
"""

import collections
import re
import sys

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

D, N, M, STEPS = 4096, 256, 32, 4


def _kernel_window(x_ref, w_ref, o_ref):
    with jax.named_scope("probe_matmul"):
        o_ref[...] = jnp.dot(x_ref[...], w_ref[...],
                             preferred_element_type=jnp.float32)


def _kernel_scratch(x_ref, w_ref, o_ref, w_scratch):
    i = pl.program_id(0)

    @pl.when(i == 0)
    def _():
        with jax.named_scope("probe_copy"):
            w_scratch[...] = w_ref[...]

    with jax.named_scope("probe_matmul"):
        o_ref[...] = jnp.dot(x_ref[...], w_scratch[...],
                             preferred_element_type=jnp.float32)


def build(scratch: bool):
    specs = dict(
        grid=(STEPS,),
        in_specs=[pl.BlockSpec((M, D), lambda i: (0, 0)),
                  pl.BlockSpec((D, N), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((M, N), lambda i: (0, 0)),
    )
    if scratch:
        return pl.pallas_call(
            _kernel_scratch,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=0,
                scratch_shapes=[pltpu.VMEM((D, N), jnp.bfloat16)], **specs),
            out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32))
    return pl.pallas_call(
        _kernel_window,
        grid_spec=pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=0, **specs),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32))


def main() -> None:
    x = jnp.zeros((M, D), jnp.bfloat16)
    w = jnp.zeros((D, N), jnp.bfloat16)
    for name, scratch in (("A_window", False), ("B_scratch", True)):
        fn = build(scratch)
        jax.block_until_ready(jax.jit(fn)(x, w))
        print(f"{name}: ran")
    print("\nrerun under the dump flag and compare vunpack/vpack counts in "
          "the probe_matmul region of each module.")


if __name__ == "__main__":
    main()
