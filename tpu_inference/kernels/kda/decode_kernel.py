# SPDX-License-Identifier: Apache-2.0
"""KDA (Kimi Delta Attention) fused recurrent decode kernel, v0.

One token per sequence per call. Fuses, per (sequence-block,
head-block) grid cell: the lower-bound gate, q/k l2-normalization,
beta sigmoid, the channel-wise-decay delta-rule state update, and
the output readout. State is read and written in place (aliased) -
the whole step is one pass over the state, which is the cost model:
decode is state-bandwidth-bound and every op here rides that stream.

v0 scope (deliberate): contiguous state [B, H, K, V]; no slot
indirection yet (the paged-pool gather/scatter double-buffer is the
next iteration - the v2-GDN async-copy pattern). All math
elementwise + lane-reductions: the decode step needs no MXU
(vector-matrix at these shapes is a VPU reduce over K), keeping the
kernel Mosaic-simple.

Numerics oracle: tpu_inference/kernels/kda/reference.py (fla
@ 3468279 math). Core accumulation in f32 regardless of input dtype.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from tpu_inference.kernels.kda.reference import LOWER_BOUND


def _packing(dtype) -> int:
    """32 / bitwidth: f32=1, bf16=2, fp8/int8=4, fp4=8."""
    return max(1, 32 // (jnp.dtype(dtype).itemsize * 8)) \
        if jnp.dtype(dtype).itemsize else 8


def _check_tpu_block_rules(specs_and_shapes):
    """Enforce Mosaic block/tiling rules at trace time so the CPU
    dev loop catches what only TPU lowering would reject (jax 0.10
    cannot lowering-check off-TPU). Rules per the platform tiled-
    layout documentation: tiles are T(t*packing, 128), packing =
    32/bitwidth
    (f32 1, bf16 2, fp8 4, fp4 8); a packed tile needs >= packing
    rows. Entries: (name, block, array_shape, dtype).
      - rank-1 blocks: whole array, x1024, or pow2 >= 128*packing;
      - rank>=2: last dim multiple of 128 or whole; second-to-last
        a multiple of 8*packing or whole.
    """
    for name, block, shape, dtype in specs_and_shapes:
        p = _packing(dtype)
        if len(block) == 1:
            b, a = block[0], shape[0]
            ok = (b == a or b % 1024 == 0
                  or (b >= 128 * p and (b & (b - 1)) == 0))
            if not ok:
                raise ValueError(
                    f"{name}: rank-1 block {b} over array {a} "
                    f"(packing {p}) violates Mosaic rules (whole, "
                    f"x1024, or pow2 >= {128 * p})")
        else:
            b, a = block[-1], shape[-1]
            if not (b == a or b % 128 == 0):
                raise ValueError(
                    f"{name}: last-dim block {b} over array dim {a} "
                    f"must be whole or a multiple of 128")
            b2, a2 = block[-2], shape[-2]
            if not (b2 == a2 or b2 % (8 * p) == 0):
                raise ValueError(
                    f"{name}: second-to-last block dim {b2} over "
                    f"array dim {a2} must be whole or a multiple of "
                    f"8*packing={8 * p} for {jnp.dtype(dtype).name}")


def _decode_kernel(a_log_ref, dt_bias_ref, q_ref, k_ref, v_ref,
                   g_ref, beta_ref, state_ref, o_ref, state_out_ref,
                   *, lower_bound: float, eps: float):
    f32 = jnp.float32
    q = q_ref[...].astype(f32)          # [bb, bh, K]
    k = k_ref[...].astype(f32)
    v = v_ref[...].astype(f32)          # [bb, bh, V]
    s = state_ref[...].astype(f32)      # [bb, bh, K, V]
    kdim = q.shape[-1]

    # gate: per-channel log decay in (lower_bound, 0)
    x = jnp.exp(a_log_ref[...].astype(f32))[None, :, None] * (
        g_ref[...].astype(f32) + dt_bias_ref[...].astype(f32)[None])
    g = lower_bound * jax.nn.sigmoid(x)                 # [bb, bh, K]

    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + eps)
    q = q * (kdim ** -0.5)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + eps)
    beta = jax.nn.sigmoid(
        beta_ref[...].astype(f32)[:, 0, :])             # [bb, H]

    s = s * jnp.exp(g)[..., None]                       # decay
    err = v - jnp.sum(k[..., None] * s, axis=2)         # [bb, bh, V]
    s = s + (beta[..., None] * k)[..., None] * err[..., None, :]
    o_ref[...] = jnp.sum(q[..., None] * s, axis=2).astype(o_ref.dtype)
    state_out_ref[...] = s.astype(state_out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("block_b", "block_h", "lower_bound", "eps",
                     "interpret"),
    donate_argnums=(0,),
)
def kda_decode(
    state: jax.Array,     # [B, H, K, V] f32 (donated, updated)
    q: jax.Array,         # [B, H, K]
    k: jax.Array,         # [B, H, K]
    v: jax.Array,         # [B, H, V]
    g_raw: jax.Array,     # [B, H, K]
    beta_raw: jax.Array,  # [B, H]
    a_log: jax.Array,     # [H] f32
    dt_bias: jax.Array,   # [H, K] f32
    *,
    block_b: int = 8,
    block_h: int = 8,
    lower_bound: float = LOWER_BOUND,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Returns (new_state, o[B, H, V])."""
    B, H, K, V = state.shape
    del block_h  # Mosaic rank-1/rank-2 block rules require whole-H
    # blocks (a_log (H,), beta (.., H)); grid over sequences only.
    # block_b is a TUNABLE: the scheduler/solver picks it from the
    # kernel's residency plan for the target and the request bucket.
    # The clamp below is only a last-resort guard for ad-hoc callers
    # (probes, tests) that pass a block the buffered state window
    # cannot hold - it keeps a diagnostic run from dying at compile
    # instead of substituting for the solver's choice.
    max_bb = max(1, (12 * 2**20) // (H * K * V * 4))
    if block_b > max_bb:
        block_b = max_bb
    while B % block_b:
        block_b -= 1
    grid = (B // block_b,)

    bspec3 = lambda d: pl.BlockSpec((block_b, H, d),
                                    lambda i: (i, 0, 0))
    bspec4 = pl.BlockSpec((block_b, H, K, V),
                          lambda i: (i, 0, 0, 0))

    _check_tpu_block_rules([
        ("a_log", (H,), (H,), a_log.dtype),
        ("dt_bias", (H, K), (H, K), dt_bias.dtype),
        ("qkvg", (block_b, H, K), (B, H, K), q.dtype),
        ("beta", (block_b, 1, H), (B, 1, H), beta_raw.dtype),
        ("state", (block_b, H, K, V), state.shape, state.dtype),
    ])

    o, new_state = pl.pallas_call(
        functools.partial(_decode_kernel, lower_bound=lower_bound,
                          eps=eps),
        grid=grid,
        in_specs=[
            pl.BlockSpec((H,), lambda i: (0,)),                 # a_log
            pl.BlockSpec((H, K), lambda i: (0, 0)),             # dt_bias
            bspec3(K), bspec3(K), bspec3(V),                    # q k v
            bspec3(K),                                          # g_raw
            pl.BlockSpec((block_b, 1, H), lambda i: (i, 0, 0)),
            bspec4,                                             # state
        ],
        out_specs=[bspec3(V), bspec4],
        out_shape=[
            jax.ShapeDtypeStruct((B, H, V), q.dtype),
            jax.ShapeDtypeStruct((B, H, K, V), state.dtype),
        ],
        input_output_aliases={7: 1},   # state -> state_out, in place
        interpret=interpret,
    )(a_log, dt_bias, q, k, v, g_raw,
      beta_raw.reshape(B, 1, H), state)
    return new_state, o


def _slotted_kernel(idx_ref, a_log_ref, dt_bias_ref, q_ref, k_ref,
                    v_ref, g_ref, beta_ref, pool_ref, o_ref,
                    pool_out_ref, *, lower_bound: float, eps: float):
    del idx_ref  # consumed by the index_maps, not the body
    _decode_kernel(a_log_ref, dt_bias_ref, q_ref, k_ref, v_ref,
                   g_ref, beta_ref, pool_ref, o_ref, pool_out_ref,
                   lower_bound=lower_bound, eps=eps)


@functools.partial(
    jax.jit,
    static_argnames=("lower_bound", "eps", "interpret"),
    donate_argnums=(0,),
)
def kda_decode_slotted(
    pool: jax.Array,       # [num_blocks, H, K, V] f32 (donated)
    idx: jax.Array,        # [B] int32 - slot per sequence
    q: jax.Array,          # [B, H, K]
    k: jax.Array,          # [B, H, K]
    v: jax.Array,          # [B, H, V]
    g_raw: jax.Array,      # [B, H, K]
    beta_raw: jax.Array,   # [B, H]
    a_log: jax.Array,      # [H] f32
    dt_bias: jax.Array,    # [H, K] f32
    *,
    lower_bound: float = LOWER_BOUND,
    eps: float = 1e-6,
    interpret: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Slot-indirected decode step: state gathered from / scattered
    back to a paged pool via scalar-prefetched indices. The gather
    and scatter ride the Pallas pipeline's own double-buffering (the
    same idiom as the paged-KV kernels), which is the attack on the
    ~3x collapse XLA shows on pool[idx] / pool.at[idx].set at these
    block sizes. Returns (new_pool, o[B, H, V])."""
    from jax.experimental.pallas import tpu as pltpu

    B = q.shape[0]
    nb, H, K, V = pool.shape

    def pool_map(n, idx_ref):
        return (idx_ref[n], 0, 0, 0)

    def row_map3(n, idx_ref):
        return (n, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(B,),
        in_specs=[
            pl.BlockSpec((H,), lambda n, i: (0,)),          # a_log
            pl.BlockSpec((H, K), lambda n, i: (0, 0)),      # dt_bias
            pl.BlockSpec((1, H, K), row_map3),              # q
            pl.BlockSpec((1, H, K), row_map3),              # k
            pl.BlockSpec((1, H, V), row_map3),              # v
            pl.BlockSpec((1, H, K), row_map3),              # g_raw
            pl.BlockSpec((1, 1, H), lambda n, i: (n, 0, 0)),  # beta
            pl.BlockSpec((1, H, K, V), pool_map),           # state in
        ],
        out_specs=[
            pl.BlockSpec((1, H, V), row_map3),              # o
            pl.BlockSpec((1, H, K, V), pool_map),           # state out
        ],
    )
    o, new_pool = pl.pallas_call(
        functools.partial(_slotted_kernel, lower_bound=lower_bound,
                          eps=eps),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, V), q.dtype),
            jax.ShapeDtypeStruct((nb, H, K, V), pool.dtype),
        ],
        # operand index counts the scalar-prefetch arg: idx=0,
        # a_log=1, ..., pool=8 -> aliased to output 1.
        input_output_aliases={8: 1},
        interpret=interpret,
    )(idx, a_log, dt_bias, q, k, v, g_raw,
      beta_raw.reshape(B, 1, H), pool)
    return new_pool, o


def _fused_kernel(idx_ref, a_log_ref, dt_bias_ref, wcq_ref, wck_ref,
                  wcv_ref, nw_ref, q_ref, k_ref, v_ref, g_ref,
                  beta_ref, go_ref, pool_ref, cq_ref, ck_ref, cv_ref,
                  o_ref, pool_out_ref, cq_out_ref, ck_out_ref,
                  cv_out_ref, *, lower_bound: float, eps: float):
    del idx_ref
    f32 = jnp.float32

    def conv(cache_ref, x_ref, w_ref, out_cache_ref):
        # cache [1,3,H,K] taps-leading (K on lanes); w [4,H,K]
        full = jnp.concatenate(
            [cache_ref[...].astype(f32),
             x_ref[...].astype(f32)[:, None]], axis=1)   # [1,4,H,K]
        y = jnp.sum(full * w_ref[...].astype(f32)[None], axis=1)
        out_cache_ref[...] = full[:, 1:].astype(out_cache_ref.dtype)
        return y * jax.nn.sigmoid(y)

    q = conv(cq_ref, q_ref, wcq_ref, cq_out_ref)     # [1, H, K]
    k = conv(ck_ref, k_ref, wck_ref, ck_out_ref)
    v = conv(cv_ref, v_ref, wcv_ref, cv_out_ref)
    s = pool_ref[...].astype(f32)                    # [1, H, K, V]
    kdim = q.shape[-1]

    x = jnp.exp(a_log_ref[...].astype(f32))[None, :, None] * (
        g_ref[...].astype(f32) + dt_bias_ref[...].astype(f32)[None])
    g = lower_bound * jax.nn.sigmoid(x)

    q = q * jax.lax.rsqrt(jnp.sum(q * q, -1, keepdims=True) + eps)
    q = q * (kdim ** -0.5)
    k = k * jax.lax.rsqrt(jnp.sum(k * k, -1, keepdims=True) + eps)
    beta = jax.nn.sigmoid(beta_ref[...].astype(f32)[:, 0, :])

    s = s * jnp.exp(g)[..., None]
    err = v - jnp.sum(k[..., None] * s, axis=2)
    s = s + (beta[..., None] * k)[..., None] * err[..., None, :]
    o = jnp.sum(q[..., None] * s, axis=2)            # [1, H, V]

    var = jnp.mean(o * o, axis=-1, keepdims=True)
    o = (o * jax.lax.rsqrt(var + eps) * nw_ref[...].astype(f32)
         * jax.nn.sigmoid(go_ref[...].astype(f32)))
    o_ref[...] = o.astype(o_ref.dtype)
    pool_out_ref[...] = s.astype(pool_out_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=("lower_bound", "eps", "interpret"),
    donate_argnums=(0, 1, 2, 3),
)
def kda_decode_fused(
    pool: jax.Array,      # [nb, H, K, V] f32 (donated)
    conv_q: jax.Array,    # [nb, 3, H, K] taps-leading (donated)
    conv_k: jax.Array,    # [nb, 3, H, K] (donated)
    conv_v: jax.Array,    # [nb, 3, H, K] (donated)
    idx: jax.Array,       # [B] int32
    q_raw: jax.Array,     # [B, H, K] pre-conv
    k_raw: jax.Array,
    v_raw: jax.Array,
    g_raw: jax.Array,
    beta_raw: jax.Array,  # [B, H]
    go: jax.Array,        # [B, H, V] output-gate values
    a_log: jax.Array, dt_bias: jax.Array,
    wconv_q: jax.Array, wconv_k: jax.Array, wconv_v: jax.Array,  # [4,H,K]
    norm_w: jax.Array,    # [V]
    *,
    lower_bound: float = LOWER_BOUND,
    eps: float = 1e-6,
    interpret: bool = False,
):
    """Fully fused slotted decode step: conv+SiLU -> gate/l2norm/
    beta -> channel-wise delta recurrence -> gated RMSNorm, with
    state AND conv caches gathered/scattered by slot index. Returns
    (pool, conv_q, conv_k, conv_v, o)."""
    from jax.experimental.pallas import tpu as pltpu

    B = q_raw.shape[0]
    nb, H, K, V = pool.shape

    pmap4 = lambda n, i: (i[n], 0, 0, 0)
    rmap3 = lambda n, i: (n, 0, 0)

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(B,),
        in_specs=[
            pl.BlockSpec((H,), lambda n, i: (0,)),
            pl.BlockSpec((H, K), lambda n, i: (0, 0)),
            pl.BlockSpec((4, H, K), lambda n, i: (0, 0, 0)),
            pl.BlockSpec((4, H, K), lambda n, i: (0, 0, 0)),
            pl.BlockSpec((4, H, K), lambda n, i: (0, 0, 0)),
            pl.BlockSpec((V,), lambda n, i: (0,)),
            pl.BlockSpec((1, H, K), rmap3),
            pl.BlockSpec((1, H, K), rmap3),
            pl.BlockSpec((1, H, V), rmap3),
            pl.BlockSpec((1, H, K), rmap3),
            pl.BlockSpec((1, 1, H), lambda n, i: (n, 0, 0)),
            pl.BlockSpec((1, H, V), rmap3),
            pl.BlockSpec((1, H, K, V), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
        ],
        out_specs=[
            pl.BlockSpec((1, H, V), rmap3),
            pl.BlockSpec((1, H, K, V), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
            pl.BlockSpec((1, 3, H, K), pmap4),
        ],
    )
    o, pool, conv_q, conv_k, conv_v = pl.pallas_call(
        functools.partial(_fused_kernel, lower_bound=lower_bound,
                          eps=eps),
        grid_spec=grid_spec,
        out_shape=[
            jax.ShapeDtypeStruct((B, H, V), q_raw.dtype),
            jax.ShapeDtypeStruct(pool.shape, pool.dtype),
            jax.ShapeDtypeStruct(conv_q.shape, conv_q.dtype),
            jax.ShapeDtypeStruct(conv_k.shape, conv_k.dtype),
            jax.ShapeDtypeStruct(conv_v.shape, conv_v.dtype),
        ],
        # operand order incl scalar arg: idx=0, a_log=1, dt_bias=2,
        # wcq=3, wck=4, wcv=5, nw=6, q=7, k=8, v=9, g=10, beta=11,
        # go=12, pool=13, cq=14, ck=15, cv=16
        input_output_aliases={13: 1, 14: 2, 15: 3, 16: 4},
        interpret=interpret,
    )(idx, a_log, dt_bias, wconv_q, wconv_k, wconv_v, norm_w,
      q_raw, k_raw, v_raw, g_raw, beta_raw.reshape(B, 1, H), go,
      pool, conv_q, conv_k, conv_v)
    return pool, conv_q, conv_k, conv_v, o
