# SPDX-License-Identifier: Apache-2.0
"""Goldens for the KDA fused recurrent decode kernel (v0) against
the fla-math reference. Runs in Pallas interpret mode on CPU; on a
TPU host the same tests exercise the compiled kernel (interpret is
auto-selected by backend)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tpu_inference.kernels.kda.decode_kernel import kda_decode
from tpu_inference.kernels.kda.reference import (
    kda_decode_step_reference)

INTERPRET = jax.default_backend() != "tpu"


def _inputs(rng, B, H, K, V, dtype):
    q = jnp.asarray(rng.standard_normal((B, H, K)), dtype)
    k = jnp.asarray(rng.standard_normal((B, H, K)), dtype)
    v = jnp.asarray(rng.standard_normal((B, H, V)), dtype)
    g = jnp.asarray(rng.standard_normal((B, H, K)), dtype)
    beta = jnp.asarray(rng.standard_normal((B, H)), dtype)
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, H)), jnp.float32)
    dt_bias = jnp.asarray(rng.standard_normal((H, K)) * 0.1,
                          jnp.float32)
    return q, k, v, g, beta, a_log, dt_bias


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize("B,H,K,V,bb,bh", [
    (8, 8, 128, 128, 8, 8),
    (16, 96, 128, 128, 8, 8),     # K3 head count
    (8, 8, 128, 128, 8, 4),
])
def test_kda_decode_matches_reference(dtype, B, H, K, V, bb, bh):
    rng = np.random.default_rng(0)
    q, k, v, g, beta, a_log, dt_bias = _inputs(rng, B, H, K, V, dtype)
    s0 = jnp.asarray(rng.standard_normal((B, H, K, V)) * 0.1,
                     jnp.float32)

    ref_s, ref_o = kda_decode_step_reference(
        s0, q, k, v, g, beta, a_log, dt_bias)
    got_s, got_o = kda_decode(
        jnp.copy(s0), q, k, v, g, beta, a_log, dt_bias,
        block_b=bb, block_h=bh, interpret=INTERPRET)

    tol = 1e-6 if dtype == jnp.float32 else 2e-2
    np.testing.assert_allclose(np.asarray(got_s), np.asarray(ref_s),
                               rtol=tol, atol=tol)
    np.testing.assert_allclose(
        np.asarray(got_o), np.asarray(ref_o.astype(dtype)),
        rtol=tol, atol=tol)


def test_kda_decode_multi_step_chain():
    """State threads correctly across 5 chained steps."""
    rng = np.random.default_rng(1)
    B, H, K, V = 8, 8, 128, 128
    s_ref = jnp.zeros((B, H, K, V), jnp.float32)
    s_ker = jnp.zeros((B, H, K, V), jnp.float32)
    for step in range(5):
        q, k, v, g, beta, a_log, dt_bias = _inputs(
            rng, B, H, K, V, jnp.float32)
        s_ref, o_ref = kda_decode_step_reference(
            s_ref, q, k, v, g, beta, a_log, dt_bias)
        s_ker, o_ker = kda_decode(
            s_ker, q, k, v, g, beta, a_log, dt_bias,
            interpret=INTERPRET)
        np.testing.assert_allclose(np.asarray(o_ker),
                                   np.asarray(o_ref), rtol=2e-5,
                                   atol=2e-5, err_msg=f"step {step}")


@pytest.mark.skipif(jax.default_backend() != "tpu",
                    reason="jax 0.10 cannot lower Pallas-for-TPU "
                           "without a TPU device (get_tpu_info)")
def test_kda_decode_lowers_for_tpu():
    """Trace/lowering gate (TPU host only on jax 0.10)."""
    rng = np.random.default_rng(2)
    B, H, K, V = 8, 8, 128, 128
    q, k, v, g, beta, a_log, dt_bias = _inputs(
        rng, B, H, K, V, jnp.bfloat16)
    s0 = jnp.zeros((B, H, K, V), jnp.float32)

    def fn(s, q, k, v, g, beta):
        return kda_decode(s, q, k, v, g, beta, a_log, dt_bias)

    exported = jax.export.export(jax.jit(fn), platforms=["tpu"])(
        s0, q, k, v, g, beta)
    assert len(exported.mlir_module_serialized) > 0


def test_kda_decode_slotted_matches_reference():
    """Slot-indirected variant: gathered states updated, untouched
    pool blocks bit-identical."""
    from tpu_inference.kernels.kda.decode_kernel import (
        kda_decode_slotted)
    rng = np.random.default_rng(3)
    B, H, K, V = 16, 96, 128, 128
    nb = 2 * B
    idx_np = rng.permutation(nb)[:B]
    idx = jnp.asarray(idx_np, jnp.int32)
    pool0 = jnp.asarray(rng.standard_normal((nb, H, K, V)) * 0.1,
                        jnp.float32)
    q, k, v, g, beta, a_log, dt_bias = _inputs(
        rng, B, H, K, V, jnp.float32)

    ref_s, ref_o = kda_decode_step_reference(
        pool0[idx_np], q, k, v, g, beta, a_log, dt_bias)
    expected = np.asarray(pool0).copy()
    expected[idx_np] = np.asarray(ref_s)

    new_pool, o = kda_decode_slotted(
        jnp.copy(pool0), idx, q, k, v, g, beta, a_log, dt_bias,
        interpret=INTERPRET)
    np.testing.assert_allclose(np.asarray(o), np.asarray(ref_o),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(new_pool), expected,
                               rtol=1e-6, atol=1e-6)
    untouched = np.setdiff1d(np.arange(nb), idx_np)
    np.testing.assert_array_equal(np.asarray(new_pool)[untouched],
                                  np.asarray(pool0)[untouched])
