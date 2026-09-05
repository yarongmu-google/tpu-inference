# SPDX-License-Identifier: Apache-2.0
"""Reference implementation of the KDA (Kimi Delta Attention)
recurrent decode step - the numerics oracle for the Pallas kernels.

Math follows fla.ops.kda (flash-linear-attention @ 3468279),
naive_recurrent_kda + the lower-bound gate (gate.py), which is what
Kimi-K3 configures (gate_lower_bound=-5, use_beta_sigmoid_in_kernel,
use_qk_l2norm_in_kernel):

    g    = lower_bound * sigmoid(exp(A_log) * (g_raw + dt_bias))
    q, k = l2norm(q), l2norm(k);  q *= K**-0.5
    beta = sigmoid(beta_raw)
    S    = S * exp(g)[..., None]          # channel-wise decay on K
    err  = v - k^T S
    S    = S + (beta * k) (x) err         # rank-1 delta update
    o    = q^T S

Shapes (decode: one token per sequence):
    q, k, g_raw : [B, H, K]     v : [B, H, V]      beta_raw : [B, H]
    A_log       : [H]           dt_bias : [H, K]   S : [B, H, K, V]
All core math in float32.
"""

import jax
import jax.numpy as jnp

LOWER_BOUND = -5.0


def kda_gate(g_raw: jax.Array, a_log: jax.Array, dt_bias: jax.Array,
             lower_bound: float = LOWER_BOUND) -> jax.Array:
    """Per-channel log-decay in (lower_bound, 0)."""
    x = jnp.exp(a_log)[:, None] * (g_raw.astype(jnp.float32)
                                   + dt_bias.astype(jnp.float32))
    return lower_bound * jax.nn.sigmoid(x)


def _l2norm(x: jax.Array, eps: float = 1e-6) -> jax.Array:
    return x * jax.lax.rsqrt(
        jnp.sum(x * x, axis=-1, keepdims=True) + eps)


def kda_decode_step_reference(
    state: jax.Array,     # [B, H, K, V] f32
    q: jax.Array,         # [B, H, K]
    k: jax.Array,         # [B, H, K]
    v: jax.Array,         # [B, H, V]
    g_raw: jax.Array,     # [B, H, K]
    beta_raw: jax.Array,  # [B, H]
    a_log: jax.Array,     # [H] f32
    dt_bias: jax.Array,   # [H, K] f32
    lower_bound: float = LOWER_BOUND,
) -> tuple[jax.Array, jax.Array]:
    """One recurrent step for B sequences. Returns (new_state, o)."""
    f32 = jnp.float32
    q, k, v = (x.astype(f32) for x in (q, k, v))
    K = q.shape[-1]
    g = kda_gate(g_raw, a_log, dt_bias, lower_bound)      # [B, H, K]
    q = _l2norm(q) * (K ** -0.5)
    k = _l2norm(k)
    beta = jax.nn.sigmoid(beta_raw.astype(f32))           # [B, H]

    state = state * jnp.exp(g)[..., None]                 # decay
    err = v - jnp.einsum("bhk,bhkv->bhv", k, state)       # correction
    state = state + jnp.einsum(
        "bhk,bhv->bhkv", beta[..., None] * k, err)        # delta
    o = jnp.einsum("bhk,bhkv->bhv", q, state)
    return state, o


def short_conv_step(cache: jax.Array, x: jax.Array,
                    w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Causal depthwise conv (kernel 4) decode step with SiLU.
    cache [B,H,K,3] (last 3 inputs), x [B,H,K], w [H,K,4].
    Returns (new_cache, y)."""
    full = jnp.concatenate(
        [cache.astype(jnp.float32), x.astype(jnp.float32)[..., None]],
        axis=-1)                                      # [B,H,K,4]
    y = jnp.sum(full * w.astype(jnp.float32)[None], axis=-1)
    y = y * jax.nn.sigmoid(y)                         # SiLU
    return full[..., 1:], y


def gated_rmsnorm(o: jax.Array, go: jax.Array, weight: jax.Array,
                  eps: float = 1e-6) -> jax.Array:
    """FusedRMSNormGated(activation='sigmoid'): per-head RMSNorm of
    o over V, scaled by weight [V], gated by sigmoid(go)."""
    o = o.astype(jnp.float32)
    var = jnp.mean(o * o, axis=-1, keepdims=True)
    return (o * jax.lax.rsqrt(var + eps) * weight.astype(jnp.float32)
            * jax.nn.sigmoid(go.astype(jnp.float32)))


def kda_decode_step_full_reference(
    state, conv_q, conv_k, conv_v,          # pools already gathered
    q_raw, k_raw, v_raw,                    # [B,H,K/V] pre-conv
    g_raw, beta_raw, go,                    # go: output gate [B,H,V]
    a_log, dt_bias, wconv_q, wconv_k, wconv_v, norm_w,
    lower_bound: float = LOWER_BOUND,
):
    """Full fused decode step: conv+SiLU on q/k/v -> core recurrence
    -> gated RMSNorm epilogue. Returns (state, conv_q, conv_k,
    conv_v, o)."""
    conv_q, q = short_conv_step(conv_q, q_raw, wconv_q)
    conv_k, k = short_conv_step(conv_k, k_raw, wconv_k)
    conv_v, v = short_conv_step(conv_v, v_raw, wconv_v)
    state, o = kda_decode_step_reference(
        state, q, k, v, g_raw, beta_raw, a_log, dt_bias, lower_bound)
    o = gated_rmsnorm(o, go, norm_w)
    return state, conv_q, conv_k, conv_v, o


def kda_chunk_reference(
    q, k, v, g_raw, beta_raw,       # [T, H, K/V] single sequence
    a_log, dt_bias,                 # [H], [H, K]
    chunk: int = 64,
    lower_bound: float = LOWER_BOUND,
):
    """Chunked (UT/WY) prefill form, single sequence - ports
    fla naive_chunk_kda with the in-kernel gate/l2norm/beta. All
    f32. Returns (o [T,H,V], final_state [H,K,V]). T % chunk == 0."""
    f32 = jnp.float32
    T, H, K = q.shape
    V = v.shape[-1]
    C = chunk
    n = T // C

    def per_head(q, k, v, g_raw, beta_raw, a_log, dt_bias):
        # [T,K],[T,K],[T,V],[T,K],[T],(),[K]
        g = lower_bound * jax.nn.sigmoid(
            jnp.exp(a_log) * (g_raw.astype(f32) + dt_bias))
        q = _l2norm(q.astype(f32)) * (K ** -0.5)
        k = _l2norm(k.astype(f32))
        beta = jax.nn.sigmoid(beta_raw.astype(f32))
        v = v.astype(f32)

        qc, kc, vc = (x.reshape(n, C, -1) for x in (q, k, v))
        gc = jnp.cumsum(g.reshape(n, C, K), axis=1)
        bc = beta.reshape(n, C)

        # pairwise decay E[i,j,:] = exp(G_i - G_j); [n,C,C,K]
        E = jnp.exp(gc[:, :, None, :] - gc[:, None, :, :])
        strict = jnp.tril(jnp.ones((C, C), bool), -1)
        incl = jnp.tril(jnp.ones((C, C), bool))

        A = jnp.einsum("nik,nijk,njk->nij", kc, E, kc)
        A = A * bc[:, :, None]
        A = jnp.where(strict[None], -A, 0.0)
        # forward substitution: A[i,:i] += A[i,:] @ A[:,:i]
        def fs(i, A):
            row = A[:, i, :] + jnp.einsum("nj,njc->nc", A[:, i, :], A)
            mask = jnp.arange(C) < i
            return A.at[:, i, :].set(jnp.where(mask[None], row,
                                               A[:, i, :]))
        A = jax.lax.fori_loop(1, C, fs, A)
        A = (A + jnp.eye(C)[None]) * bc[:, None, :]

        w = jnp.einsum("nij,njk->nik", A, jnp.exp(gc) * kc)
        u = jnp.einsum("nij,njv->niv", A, vc)
        Aqk = jnp.einsum("nik,nijk,njk->nij", qc, E, kc)
        Aqk = jnp.where(incl[None], Aqk, 0.0)

        def step(S, xs):
            qi, ki, ui, wi, gi, aqk = xs
            v_new = ui - wi @ S
            o = (qi * jnp.exp(gi)) @ S + aqk @ v_new
            g_last = gi[-1]
            S = S * jnp.exp(g_last)[:, None] + (
                ki * jnp.exp(g_last - gi)).T @ v_new
            return S, o

        S, o = jax.lax.scan(step, jnp.zeros((K, V), f32),
                            (qc, kc, u, w, gc, Aqk))
        return o.reshape(T, V), S

    return jax.vmap(per_head, in_axes=(1, 1, 1, 1, 1, 0, 0),
                    out_axes=(1, 0))(q, k, v, g_raw, beta_raw,
                                     a_log.astype(f32),
                                     dt_bias.astype(f32))
