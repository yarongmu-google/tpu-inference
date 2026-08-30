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


# ---------------- fp8 (w8a8) path ----------------


def _quant_per_channel(w: np.ndarray, axis: int):
    """Per-channel e4m3 quantization (in_blocks == 1, the serving
    default requant contract): one scale per output channel over the
    whole contraction `axis`."""
    amax = np.abs(w).max(axis=axis, keepdims=True)
    scale = (amax / 448.0).astype(np.float32)
    q = np.asarray(np.clip(w / scale, -448.0, 448.0),
                   jnp.float8_e4m3fn)
    return q, scale


def _dequant_tokens_like_kernel(tokens: np.ndarray) -> np.ndarray:
    """The kernel's per-token dynamic quantization, round-tripped to
    f32 - feeding this to the f32 reference isolates the kernel's
    MACHINERY (gather restore, scale application points, combine
    bracketing) from the irreducible e4m3 representation error."""
    amax = np.abs(tokens).max(axis=1, keepdims=True)
    sinv = np.where(amax > 0.0, 448.0 / amax, 0.0)
    q = np.asarray(np.clip(tokens * sinv, -448.0, 448.0),
                   jnp.float8_e4m3fn)
    return q.astype(np.float32) * (amax / 448.0)


@pytest.mark.parametrize("bg", [1, 2])
@pytest.mark.parametrize("bd1c,bd2c,bcT", [(None, None, None),
                                           (128, 128, 32)],
                         ids=["whole_dots", "chunked_dots"])
@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_fp8_matches_dequant_reference(t, d, e, i, k,
                                                     bd1c, bd2c, bcT, bg):
    """fp8 fused entry vs the f32 reference run on DEQUANTIZED weights
    and kernel-identically-quantized tokens: every fp8 mechanism
    (prologue quantize + fp8 AG, e4m3 OH gather restore, OHS -> s_x,
    fp8 gmm1, scale-after-full-K, per-n-chunk s2, K-fused combine)
    must reproduce the dequantized math up to summation order.

    Goes through the FUSED entry (not serving) so bd1c/bd2c/bcT are
    real: the chunked case covers gmm1 chunk-accumulate-then-scale,
    the per-n-chunk s2 slice offsets, and the bcT-chunked K-fused
    combine - none of which the whole-width case exercises."""
    _skip_unless_devices(P_DEV)
    rng = np.random.default_rng(0)
    p = P_DEV
    tokens = np.asarray(rng.standard_normal((t, d)), np.float32)
    w1 = rng.standard_normal((e, d, 2, i)) * 0.02       # global [E,D,2,I]
    w2 = rng.standard_normal((e, i, d)) * 0.02
    router_w = np.asarray(rng.standard_normal((e, d)) * 0.1, np.float32)
    gating = tokens @ router_w.T

    # per-channel quantization on the GLOBAL [E, D, 2, I]: sharding the
    # last axis then hands each device its own quantized slice + scale
    w1_q, w1_s = _quant_per_channel(w1, axis=1)          # [E,1,2,I]
    w2_q, w2_s = _quant_per_channel(w2, axis=1)          # [E,1,D]
    mesh = Mesh(np.array(jax.devices()[:P_DEV]), ("x",))

    def fn(tok_l, w1_l, w2_l, r_l, s1_l, s2_l):
        w1_flat = w1_l.reshape(w1_l.shape[0], w1_l.shape[1], -1)
        s1_flat = s1_l.reshape(s1_l.shape[0], -1)        # [E, 2*i/p]
        return fused_moe_decode_tp_fused(
            tok_l,
            r_l,          # router weight, replicated
            w1_flat,
            w2_l,
            s1_flat,
            s2_l.reshape(s2_l.shape[0], -1),             # [E, D]
            mesh=mesh,
            axis_name="x",
            top_k=k,
            renormalize_topk_logits=True,
            capacity=t,   # multiple of 32 (fp8 granule); no drops
            be=4,
            bg=bg,
            bd1c=bd1c,
            bd2c=bd2c,
            bcT=bcT,
            interpret=True,
        )

    got = jax.jit(jax.shard_map(
        fn, mesh=mesh,
        in_specs=(P("x", None), P(None, None, None, "x"),
                  P(None, "x", None), P(None, None),
                  P(None, None, None, "x"), P(None, None, None)),
        out_specs=P("x", None), check_vma=False,
    ))(jnp.asarray(tokens), jnp.asarray(w1_q), jnp.asarray(w2_q),
       jnp.asarray(router_w), jnp.asarray(w1_s), jnp.asarray(w2_s))

    w1_deq = (w1_q.astype(np.float32) * w1_s).reshape(e, d, 2 * i)
    w2_deq = w2_q.astype(np.float32) * w2_s
    tok_deq = _dequant_tokens_like_kernel(tokens)
    want = _reference_moe(jnp.asarray(tok_deq), jnp.asarray(w1_deq),
                          jnp.asarray(w2_deq), jnp.asarray(gating), k,
                          renormalize=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)

    # loose sanity vs the CLEAN f32 reference: documents the
    # end-to-end w8a8 quantization error at these shapes
    clean = _reference_moe(jnp.asarray(tokens),
                           jnp.asarray(w1.reshape(e, d, 2 * i)),
                           jnp.asarray(w2), jnp.asarray(gating), k,
                           renormalize=True)
    rel = (np.abs(np.asarray(got) - np.asarray(clean)).max()
           / np.abs(np.asarray(clean)).max())
    assert rel < 0.15, f"end-to-end w8a8 error {rel:.3f}"


def test_decode_kernel_fp8_tiny_padded_serving():
    """fp8 x the tiny-batch serving class: 2 rows/device forces the
    fp8 row granule (padded_rows 16 -> 32) and the phantom-row
    zero-data path through quantize (s = 0, q = 0)."""
    _skip_unless_devices(P_DEV)
    t, d, e, i, k, p = 16, 256, 16, 128, 4, P_DEV
    rng = np.random.default_rng(1)
    tokens = np.asarray(rng.standard_normal((t, d)), np.float32)
    w1 = rng.standard_normal((e, d, 2, i)) * 0.02
    w1_serving = np.transpose(
        w1.reshape(e, d, 2, p, i // p),
        (0, 1, 3, 2, 4)).reshape(e, d, 2 * i)
    w2 = rng.standard_normal((e, i, d)) * 0.02
    gating = np.asarray(rng.standard_normal((t, e)), np.float32)
    w1_q, w1_s = _quant_per_channel(w1_serving, axis=1)
    w2_q, w2_s = _quant_per_channel(w2, axis=1)
    mesh = Mesh(np.array(jax.devices()[:p]), ("x",))

    got = fused_moe_decode_tp_serving(
        jnp.asarray(tokens), jnp.asarray(gating),
        jnp.asarray(w1_q), jnp.asarray(w2_q),
        jnp.asarray(w1_s[:, None]), jnp.asarray(w2_s[:, None]),
        mesh=mesh,
        axis_name="x",
        top_k=k,
        renormalize_topk_logits=True,
        capacity=t,   # rounds up to 32 inside the serving entry
        interpret=True,
    )
    w1_deq = w1_q.astype(np.float32) * w1_s
    w1_deq_ref = np.transpose(
        w1_deq.reshape(e, d, p, 2, i // p),
        (0, 1, 3, 2, 4)).reshape(e, d, 2 * i)
    w2_deq = w2_q.astype(np.float32) * w2_s
    tok_deq = _dequant_tokens_like_kernel(tokens)
    want = _reference_moe(jnp.asarray(tok_deq), jnp.asarray(w1_deq_ref),
                          jnp.asarray(w2_deq), jnp.asarray(gating), k,
                          renormalize=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("t", [64, 16],
                         ids=["design_rows", "tiny_bucket"])
def test_decode_kernel_fp8_lowers_for_tpu_on_cpu(t):
    """Mosaic gate for the fp8 path at the serving dtype and BOTH row
    buckets (lesson C.21: lowering tests must cover the serving dtype
    and the smallest bucket). Exercises the fp8-only code: e4m3
    tokens_full/x stores at 32-row granularity, the OHS reduce, scale
    slab DMAs riding the weight sems, the e4m3 gather store, and the
    K-fused combine's concatenated K=bg*be*C dot."""
    import functools

    from jax._src import mesh as mesh_lib

    d, e, i, k = 256, 16, 128, 4
    i_loc = i // P_DEV
    rng = np.random.default_rng(0)
    tokens = jnp.asarray(rng.standard_normal((t, d)), jnp.bfloat16)
    gating = jnp.asarray(rng.standard_normal((t, e)), jnp.float32)
    w1_l = np.asarray(rng.standard_normal((e, d, 2 * i_loc)),
                      jnp.float8_e4m3fn)
    w2_l = np.asarray(rng.standard_normal((e, i_loc, d)),
                      jnp.float8_e4m3fn)
    w1s_l = jnp.ones((e, 1, 1, 2 * i_loc), jnp.float32)
    w2s_l = jnp.ones((e, 1, 1, d), jnp.float32)

    amesh = mesh_lib.AbstractMesh(
        (P_DEV,), ("x",),
        abstract_device=mesh_lib.AbstractDevice(
            device_kind="TPU7x", num_cores=1, platform="tpu"))
    fn = functools.partial(
        fused_moe_decode_tp_fused,
        router_fused=False,
        mesh=amesh,
        axis_name="x",
        top_k=k,
        renormalize_topk_logits=True,
        capacity=32,
        be=4,
        bg=2,
        interpret=False,
    )
    with mesh_lib.use_abstract_mesh(amesh):
        sfn = jax.jit(jax.shard_map(
            fn, mesh=amesh,
            in_specs=(P("x", None), P("x", None),
                      P(None, None, None), P(None, None, None),
                      P(None, None, None, None),
                      P(None, None, None, None)),
            out_specs=P("x", None), check_vma=False))
        exported = jax.export.export(sfn, platforms=["tpu"])(
            tokens, gating, jnp.asarray(w1_l), jnp.asarray(w2_l),
            w1s_l, w2s_l)
    assert len(exported.mlir_module_serialized) > 0


def _dequant_tokens_tensor_like_kernel(tokens: np.ndarray) -> np.ndarray:
    """The tensor-mode quantization mirror: ONE global dynamic scale."""
    g = float(np.abs(tokens).max())
    sinv = 448.0 / g if g > 0 else 0.0
    q = np.asarray(np.clip(tokens * sinv, -448.0, 448.0),
                   jnp.float8_e4m3fn)
    return q.astype(np.float32) * (g / 448.0)


@pytest.mark.parametrize("t,d,e,i,k", [(64, 256, 16, 128, 4)])
def test_decode_kernel_fp8_tensor_scale_matches_reference(t, d, e, i, k):
    """act_scale="tensor": the prologue amax exchange must land the
    same GLOBAL scale on every device, and the OHS-free path must
    reproduce the dequantized math (scales fold into the s13 row)."""
    _skip_unless_devices(P_DEV)
    rng = np.random.default_rng(3)
    tokens = np.asarray(rng.standard_normal((t, d)), np.float32)
    w1 = rng.standard_normal((e, d, 2, i)) * 0.02
    w2 = rng.standard_normal((e, i, d)) * 0.02
    router_w = np.asarray(rng.standard_normal((e, d)) * 0.1, np.float32)
    gating = tokens @ router_w.T
    w1_q, w1_s = _quant_per_channel(w1, axis=1)
    w2_q, w2_s = _quant_per_channel(w2, axis=1)
    mesh = Mesh(np.array(jax.devices()[:P_DEV]), ("x",))

    def fn(tok_l, w1_l, w2_l, r_l, s1_l, s2_l):
        return fused_moe_decode_tp_fused(
            tok_l, r_l,
            w1_l.reshape(w1_l.shape[0], w1_l.shape[1], -1),
            w2_l,
            s1_l.reshape(s1_l.shape[0], -1),
            s2_l.reshape(s2_l.shape[0], -1),
            act_scale="tensor",
            mesh=mesh, axis_name="x", top_k=k,
            renormalize_topk_logits=True, capacity=t, be=4, bg=2,
            interpret=True)

    got = jax.jit(jax.shard_map(
        fn, mesh=mesh,
        in_specs=(P("x", None), P(None, None, None, "x"),
                  P(None, "x", None), P(None, None),
                  P(None, None, None, "x"), P(None, None, None)),
        out_specs=P("x", None), check_vma=False,
    ))(jnp.asarray(tokens), jnp.asarray(w1_q), jnp.asarray(w2_q),
       jnp.asarray(router_w), jnp.asarray(w1_s), jnp.asarray(w2_s))

    w1_deq = (w1_q.astype(np.float32) * w1_s).reshape(e, d, 2 * i)
    w2_deq = w2_q.astype(np.float32) * w2_s
    tok_deq = _dequant_tokens_tensor_like_kernel(tokens)
    want = _reference_moe(jnp.asarray(tok_deq), jnp.asarray(w1_deq),
                          jnp.asarray(w2_deq), jnp.asarray(gating), k,
                          renormalize=True)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want),
                               rtol=2e-2, atol=2e-2)
