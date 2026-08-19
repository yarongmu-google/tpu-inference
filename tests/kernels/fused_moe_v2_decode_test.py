"""Correctness: fused_moe/v2 decode kernel vs a plain-JAX reference MoE.

Interpret mode (runs on CPU); shapes small but structurally faithful
(E > experts_per_block so the grid iterates, T*k >> E so experts get
multiple rows, and an unpadded-odd expert usage pattern via random gating).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.fused_moe.v2.decode_kernel import fused_moe_decode


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


@pytest.mark.parametrize("t,d,e,i,k", [(32, 256, 16, 128, 4)])
def test_decode_kernel_matches_reference(t, d, e, i, k):
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    w1 = jnp.asarray(rng.standard_normal((e, d, 2 * i)) * 0.02, jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((e, i, d)) * 0.02, jnp.float32)
    gating = jnp.asarray(rng.standard_normal((t, e)), jnp.float32)

    got = fused_moe_decode(
        tokens, w1, w2, gating,
        top_k=k, renormalize=True, capacity=t,  # capacity=T: no drops
        experts_per_block=4, interpret=True,
    )
    want = _reference_moe(tokens, w1, w2, gating, k, renormalize=True)

    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("t,d,e,i,k", [(32, 256, 16, 128, 4)])
def test_decode_kernel_lowers_for_tpu_on_cpu(t, d, e, i, k):
    """Mosaic lowerability gate, hardware-free: cross-platform lowering via
    an abstract mesh runs the full Pallas->Mosaic pipeline (layout
    inference, dialect verification) - this is what caught the
    dynamic_slice-on-traced-arrays bug that interpret mode cannot see."""
    import functools

    from jax._src import mesh as mesh_lib

    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.float32)
    w1 = jnp.asarray(rng.standard_normal((e, d, 2 * i)) * 0.02, jnp.float32)
    w2 = jnp.asarray(rng.standard_normal((e, i, d)) * 0.02, jnp.float32)
    gating = jnp.asarray(rng.standard_normal((t, e)), jnp.float32)

    fn = functools.partial(fused_moe_decode, top_k=k, renormalize=True,
                           capacity=t, experts_per_block=4, interpret=False)
    amesh = mesh_lib.AbstractMesh(
        (1,), ("x",),
        abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"))
    with mesh_lib.use_abstract_mesh(amesh):
        exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(
            tokens, w1, w2, gating)
    assert len(exported.mlir_module_serialized) > 0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q", "-s"]))
