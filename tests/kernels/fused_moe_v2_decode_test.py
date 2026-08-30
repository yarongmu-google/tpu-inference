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

from tpu_inference.kernels.fused_moe.v2.decode_kernel import (
    fused_moe_decode_tp_fused, fused_moe_decode_tp_serving)

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


@pytest.mark.parametrize("be,bg", [(4, 1), (8, 1), (4, 2), (2, 4)])
@pytest.mark.parametrize("bd1c,bd2c", [(None, None), (128, 128)],
                         ids=["whole_d_dots", "chunked_dots"])
@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_tp_matches_reference(t, d, e, i, k, bd1c, bd2c,
                                            be, bg):
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
            r_l,          # router weight, replicated
            w1_flat,
            w2_l,
            mesh=mesh,
            axis_name="x",
            top_k=k,
            renormalize_topk_logits=True,
            capacity=t,   # capacity=T: no drops
            be=be,
            bg=bg,
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


@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_tp_serving_gating_in(t, d, e, i, k):
    """The serving entry: precomputed router logits (gating_local path),
    shard_map done by the wrapper, weights in the serving arrangement
    (each 2I/P chunk of w1 holds that shard's gate|up pair)."""
    _skip_unless_devices(P_DEV)
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    w1 = jnp.asarray(rng.standard_normal((e, d, 2, i)) * 0.02, jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((e, i, d)) * 0.02, jnp.float32)
    router_w = jnp.asarray(rng.standard_normal((e, d)) * 0.1, jnp.float32)
    gating = tokens @ router_w.T                             # [T, E]
    mesh = Mesh(np.array(jax.devices()[:P_DEV]), ("x",))
    p = P_DEV
    # serving weight arrangement: sharding the last axis into P chunks
    # must hand each shard its own [gate_p | up_p] pair
    w1_serving = jnp.transpose(
        w1.reshape(e, d, 2, p, i // p),
        (0, 1, 3, 2, 4)).reshape(e, d, 2 * i)

    got = jax.jit(lambda tok, g, a, b: fused_moe_decode_tp_serving(
        tok, g, a, b,
        mesh=mesh,
        axis_name="x",
        top_k=k,
        renormalize_topk_logits=True,
        capacity=t,   # capacity=T: no drops
        interpret=True,
    ))(tokens, gating, w1_serving, w2)
    want = _reference_moe(tokens, w1.reshape(e, d, 2 * i), w2, gating, k,
                          renormalize=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("bg", [1, 2])
@pytest.mark.parametrize("bd1c,bd2c", [(None, None), (128, 128)],
                         ids=["whole_d_dots", "chunked_dots"])
@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_tp_fused_lowers_for_tpu_on_cpu(t, d, e, i, k,
                                                      bd1c, bd2c, bg):
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
        bg=bg,
        bd1c=bd1c,
        bd2c=bd2c,
        interpret=False,
    )
    with mesh_lib.use_abstract_mesh(amesh):
        sfn = jax.jit(jax.shard_map(
            fn, mesh=amesh,
            in_specs=(P("x", None), P(None, None),
                      P(None, None, None), P(None, None, None)),
            out_specs=P("x", None), check_vma=False))
        exported = jax.export.export(sfn, platforms=["tpu"])(
            tokens, router_w, w1_l, w2_l)
    assert len(exported.mlir_module_serialized) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))


def test_decode_kernel_tp_serving_tiny_padded_shapes():
    """The serving shape class that broke twice: tiny decode batches
    (2 rows/device - unprovable row0 alignment, E2003) with w13's
    per-shard intermediate padded to the lane tile while w2 is
    unpadded. Checked against a dense reference."""
    if len(jax.devices()) < 8:
        pytest.skip("needs 8 devices; on CPU run with "
                    "XLA_FLAGS=--xla_force_host_platform_device_count=8")
    t, d, e, k, p = 16, 256, 16, 4, 8
    il_r, il_p = 12, 16
    rng = np.random.default_rng(0)
    w13 = np.zeros((e, d, 2 * il_p * p), np.float32)
    gates, ups = [], []
    for s_ in range(p):
        g = rng.standard_normal((e, d, il_r)) * 0.02
        u = rng.standard_normal((e, d, il_r)) * 0.02
        gates.append(g)
        ups.append(u)
        w13[:, :, s_ * 2 * il_p:s_ * 2 * il_p + il_r] = g
        w13[:, :, s_ * 2 * il_p + il_p:s_ * 2 * il_p + il_p + il_r] = u
    gate_full = np.concatenate(gates, -1)
    up_full = np.concatenate(ups, -1)
    w2 = rng.standard_normal((e, il_r * p, d)) * 0.02
    tokens = np.asarray(rng.standard_normal((t, d)), np.float32)
    gating = np.asarray(rng.standard_normal((t, e)), np.float32)
    mesh = jax.sharding.Mesh(np.array(jax.devices()[:p]), ("x", ))
    out = np.asarray(
        fused_moe_decode_tp_serving(
            hidden_states=jnp.asarray(tokens),
            gating_output=jnp.asarray(gating),
            w1=jnp.asarray(w13),
            w2=jnp.asarray(w2),
            mesh=mesh,
            axis_name="x",
            top_k=k,
            renormalize_topk_logits=True,
            capacity=t,
            interpret=True,
        ))

    def softmax(x):
        ex = np.exp(x - x.max())
        return ex / ex.sum()

    ref = np.zeros((t, d))
    for ti in range(t):
        idx = np.argsort(-gating[ti])[:k]
        w = softmax(gating[ti][idx])
        for kk in range(k):
            ee = idx[kk]
            h = tokens[ti] @ gate_full[ee]
            hu = tokens[ti] @ up_full[ee]
            ref[ti] += w[kk] * (((h / (1 + np.exp(-h))) * hu) @ w2[ee])
    err = np.abs(out - ref).max() / np.abs(ref).max()
    assert err < 5e-2, err
