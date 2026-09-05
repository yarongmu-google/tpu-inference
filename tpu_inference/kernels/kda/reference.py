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
