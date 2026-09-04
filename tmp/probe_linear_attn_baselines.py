# Linear-attention (gated delta rule) baseline probes on one TPU core.
#
# Three measurements at large-head shapes (H=96, K=V=128), all
# XLA-jitted reference implementations - the "what the compiler gives
# you for free" floor any custom kernel must beat, plus unit-rate
# calibration:
#   A. chunked prefill form (UT/WY, channel-wise decay) at chunk size
#      C in {64, 128, 256}: ms per 8192 tokens, cyc/token.
#   B. decode recurrent step at batch B in {8, 32, 64}: ms/step with
#      the state forced through HBM (donated buffer), vs the pure
#      state r+w bandwidth bound.
#   C. exp() throughput on a large array (transcendental-unit rate).
#
# Run (serving env):  python tmp/probe_linear_attn_baselines.py
# Output tees to tmp/linear_attn_baselines.log - commit that.

import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


class _Tee:
    def __init__(self, path):
        self.f, self.stdout = open(path, "w"), sys.stdout

    def write(self, s):
        self.stdout.write(s)
        self.f.write(s)

    def flush(self):
        self.stdout.flush()
        self.f.flush()


H, K, V = 96, 128, 128
CLOCK_GHZ = 1.1


def bench(fn, *args, iters=20, donate=None):
    jfn = jax.jit(fn, donate_argnums=donate) if donate else jax.jit(fn)
    out = jax.block_until_ready(jfn(*args))
    ts = []
    for _ in range(iters):
        a = args if donate is None else tuple(
            jnp.copy(x) if i in donate else x for i, x in enumerate(args))
        t0 = time.perf_counter()
        out = jax.block_until_ready(jfn(*a))
        ts.append(time.perf_counter() - t0)
    return min(ts)


# ---- A. chunked prefill (UT/WY form, factorized decay) -------------
# NB the exp(-G) factorization is CLIPPED for range safety - this is a
# THROUGHPUT baseline, not a numerics oracle.
def chunked_delta(q, k, v, g, beta, C):
    # q,k,v: [T,H,K/V]; g: [T,H,K] (log-space, <=0); beta: [T,H]
    T = q.shape[0]
    n = T // C

    def per_head(q, k, v, g, beta):         # [T,K],[T,K],[T,V],[T,K],[T]
        qc = q.reshape(n, C, K)
        kc = k.reshape(n, C, K)
        vc = v.reshape(n, C, V)
        gc = jnp.cumsum(g.reshape(n, C, K), axis=1)
        bc = beta.reshape(n, C, 1)
        kg_f = kc * jnp.exp(gc)             # k * exp(G)
        kg_b = kc * jnp.exp(-jnp.clip(gc, -60, 0))  # k * exp(-G), clipped
        A = jnp.einsum('nik,njk->nij', kg_f * bc, kg_b)
        tri = jnp.tril(jnp.ones((C, C)), -1)
        A = -A * tri
        Tinv = jax.scipy.linalg.solve_triangular(
            jnp.eye(C) - A, jnp.broadcast_to(jnp.eye(C), (n, C, C)),
            lower=True)
        u = jnp.einsum('nij,njv->niv', Tinv, vc * bc)
        w = jnp.einsum('nij,njk->nik', Tinv, kg_f * bc)

        def step(S, x):
            qi, ki, ui, wi, gi = x
            g_last = gi[-1]
            v_new = ui - wi @ S
            o = (qi * jnp.exp(gi)) @ S + jnp.tril(
                jnp.einsum('ik,jk->ij', qi * jnp.exp(gi), ki * jnp.exp(-jnp.clip(gi, -60, 0)))) @ v_new
            S = S * jnp.exp(g_last)[:, None] + (
                ki * jnp.exp(g_last - gi)).T @ v_new
            return S, o

        S0 = jnp.zeros((K, V), jnp.float32)
        _, o = jax.lax.scan(step, S0, (qc, kc, u, w, gc))
        return o.reshape(T, V)

    return jax.vmap(per_head, in_axes=(1, 1, 1, 1, 1), out_axes=1)(
        q, k, v, g, beta)


# ---- B. decode recurrent step --------------------------------------
def decode_step(S, q, k, v, g, beta):       # S:[B,H,K,V]
    S = S * jnp.exp(g)[..., None]
    err = v - jnp.einsum('bhk,bhkv->bhv', k, S)
    S = S + jnp.einsum('bhk,bhv->bhkv', beta[..., None] * k, err)
    o = jnp.einsum('bhk,bhkv->bhv', q, S)
    return S, o


def main():
    sys.stdout = _Tee("tmp/linear_attn_baselines.log")
    print("jax", jax.__version__, jax.devices()[:1])
    rng = np.random.default_rng(0)

    # A: chunked prefill, T=8192
    T = 8192
    q, k = (jnp.asarray(rng.standard_normal((T, H, K)), jnp.bfloat16)
            for _ in range(2))
    v = jnp.asarray(rng.standard_normal((T, H, V)), jnp.bfloat16)
    g = jnp.asarray(-np.abs(rng.standard_normal((T, H, K))) * 0.1,
                    jnp.float32)
    beta = jnp.asarray(rng.random((T, H)), jnp.float32)
    print(f"\nA. chunked prefill XLA baseline, T={T}, H={H}, K=V={K}")
    print(f"{'C':>5} {'ms':>9} {'cyc/token(2-MXU clock)':>24}")
    for C in (64, 128, 256):
        t = bench(partial(chunked_delta, C=C),
                  q.astype(jnp.float32), k.astype(jnp.float32),
                  v.astype(jnp.float32), g, beta)
        print(f"{C:>5} {t*1e3:9.2f} {t*CLOCK_GHZ*1e9/T:24.1f}")

    # B: decode step, state donated so r+w hits HBM every call
    print(f"\nB. decode recurrent step (state f32 through HBM)")
    print(f"{'B':>5} {'ms/step':>9} {'state r+w GB':>13} {'GB/s':>8}")
    for B in (8, 32, 64):
        S = jnp.zeros((B, H, K, V), jnp.float32)
        qd = jnp.asarray(rng.standard_normal((B, H, K)), jnp.float32)
        kd, gd = qd, jnp.asarray(-np.abs(
            rng.standard_normal((B, H, K))) * 0.1, jnp.float32)
        vd = jnp.asarray(rng.standard_normal((B, H, V)), jnp.float32)
        bd = jnp.asarray(rng.random((B, H)), jnp.float32)
        t = bench(decode_step, S, qd, kd, vd, gd, bd, donate=(0,))
        gb = 2 * B * H * K * V * 4 / 1e9
        print(f"{B:>5} {t*1e3:9.3f} {gb:13.2f} {gb/t:8.0f}")

    # C: exp throughput
    n = 1 << 27
    x = jnp.asarray(rng.standard_normal(n), jnp.float32) * -1.0
    t = bench(lambda a: jnp.exp(a), x)
    print(f"\nC. exp() on {n/1e6:.0f}M f32: {t*1e3:.2f} ms = "
          f"{n/t/1e9:.0f} Gelem/s (r+w {2*4*n/t/1e9:.0f} GB/s)")


if __name__ == "__main__":
    main()
