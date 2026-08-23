"""Correctness: fused_moe/v2 decode kernel vs a plain-JAX reference MoE.

Interpret mode (runs on CPU, 8 simulated devices); shapes small but
structurally faithful (E > be so the grid iterates, T*k >> E so experts
get multiple rows, and an unpadded-odd expert usage pattern via random
router weights).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference.kernels.fused_moe.v2.decode_kernel import \
    fused_moe_decode_tp_fused

P_DEV = 8


def _skip_unless_devices(n: int) -> None:
    if jax.device_count() < n:
        pytest.skip(f"needs {n} devices; on CPU run with "
                    f"XLA_FLAGS=--xla_force_host_platform_device_count={n}")


def _reference_moe(tokens, w1, w2, gating, top_k, renormalize):
    scores = jax.nn.softmax(gating.astype(jnp.float32), axis=-1)
    top_w, top_i = jax.lax.top_k(scores, top_k)
    if renormalize:
        top_w = top_w / jnp.sum(top_w, axis=-1, keepdims=True)
    t, d = tokens.shape
    out = jnp.zeros((t, d), jnp.float32)
    for k_id in range(top_k):
        e = top_i[:, k_id]                                   # [T]
        w1_t = w1[e]                                         # [T, D, 2I]
        w2_t = w2[e]                                         # [T, I, D]
        h = jnp.einsum("td,tdi->ti", tokens.astype(jnp.float32),
                       w1_t.astype(jnp.float32))
        i_half = h.shape[-1] // 2
        a = jax.nn.silu(h[:, :i_half]) * h[:, i_half:]
        y = jnp.einsum("ti,tid->td", a, w2_t.astype(jnp.float32))
        out = out + top_w[:, k_id][:, None] * y
    return out


@pytest.mark.parametrize("bd1c,bd2c", [(None, None), (128, 128)],
                         ids=["whole_d_dots", "chunked_dots"])
@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_tp_matches_reference(t, d, e, i, k, bd1c, bd2c):
    """The kernel vs the global-batch reference on 8 simulated devices.

    DP-attention serving context: each device holds T/P tokens (attention
    is data-parallel), weights are I-sharded; the kernel computes router
    logits in-kernel from (tokens, router_w), routes its local shard, and
    all-gathers tokens + top-k results as VMEM-direct remote copies
    (simulated by the TPU interpret machine).

    Weight contract: shard the global w1 as [E, D, 2, I] on the LAST axis
    so each device's gate|up slices stay locally concatenated after the
    flatten to [E, D, 2*I/P].
    """
    _skip_unless_devices(P_DEV)
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    w1 = jnp.asarray(rng.standard_normal((e, d, 2, i)) * 0.02, jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((e, i, d)) * 0.02, jnp.float32)
    # router weight in the upstream [out, in] = [E, D] layout; the kernel
    # computes the logits in-kernel, the reference derives them here.
    router_w = jnp.asarray(rng.standard_normal((e, d)) * 0.1, jnp.float32)
    gating = tokens @ router_w.T                             # [T, E]
    mesh = Mesh(np.array(jax.devices()[:P_DEV]), ("x",))

    def fn(tok_l, w1_l, w2_l, r_l):
        w1_flat = w1_l.reshape(w1_l.shape[0], w1_l.shape[1], -1)
        return fused_moe_decode_tp_fused(
            tok_l,
            w1_flat,
            w2_l,
            r_l,          # router_w, replicated
            mesh=mesh,
            axis_name="x",
            top_k=k,
            renormalize_topk_logits=True,
            capacity=t,   # capacity=T: no drops
            be=4,
            bd1c=bd1c,
            bd2c=bd2c,
            interpret=True,
        )

    got = jax.jit(jax.shard_map(
        fn, mesh=mesh,
        in_specs=(P("x", None), P(None, None, None, "x"),
                  P(None, "x", None), P(None, None)),
        out_specs=P("x", None), check_vma=False,
    ))(tokens, w1, w2, router_w)
    want = _reference_moe(tokens, w1.reshape(e, d, 2 * i), w2, gating, k,
                          renormalize=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("bd1c,bd2c", [(None, None), (128, 128)],
                         ids=["whole_d_dots", "chunked_dots"])
@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_tp_fused_lowers_for_tpu_on_cpu(t, d, e, i, k,
                                                      bd1c, bd2c):
    """Mosaic gate, hardware-free: cross-platform lowering via an abstract
    8-device TPU7x mesh runs the full Pallas->Mosaic pipeline (layout
    inference, dialect verification) over the whole kernel - the fused
    router matmul, the [E, T] sublane routing, the remote copy descriptors
    with dynamic dst slices, the barrier semaphore, and MESH device ids.
    This is what caught the dynamic_slice-on-traced-arrays bug that
    interpret mode cannot see."""
    import functools

    from jax._src import mesh as mesh_lib

    rng = np.random.default_rng(0)
    i_loc = i // P_DEV
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    w1_l = jnp.asarray(rng.standard_normal((e, d, 2 * i_loc)) * 0.02,
                       jnp.float32)
    w2_l = jnp.asarray(rng.standard_normal((e, i_loc, d)) * 0.02, jnp.float32)
    router_w = jnp.asarray(rng.standard_normal((e, d)) * 0.1, jnp.float32)

    amesh = mesh_lib.AbstractMesh(
        (P_DEV,), ("x",),
        abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"))
    fn = functools.partial(
        fused_moe_decode_tp_fused,
        mesh=amesh,
        axis_name="x",
        top_k=k,
        renormalize_topk_logits=True,
        capacity=t,
        be=4,
        bd1c=bd1c,
        bd2c=bd2c,
        interpret=False,
    )
    with mesh_lib.use_abstract_mesh(amesh):
        sfn = jax.jit(jax.shard_map(
            fn, mesh=amesh,
            in_specs=(P("x", None), P(None, None, None),
                      P(None, None, None), P(None, None)),
            out_specs=P("x", None), check_vma=False))
        exported = jax.export.export(sfn, platforms=["tpu"])(
            tokens, w1_l, w2_l, router_w)
    assert len(exported.mlir_module_serialized) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
