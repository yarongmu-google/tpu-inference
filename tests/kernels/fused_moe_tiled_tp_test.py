"""VMEM dispatch, skewed expert loads, TP numerics and TPU IR export.

CPU: XLA_FLAGS=--xla_force_host_platform_device_count=8 pytest -q this_file
The TPU export checks require no TPU, but are not device execution tests.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from tpu_inference.kernels.fused_moe.v2 import tiled_tp as kernel


def _quantize(weight):
    # Per-channel quantization of one expert's contraction, matching serving.
    scale = jnp.max(jnp.abs(weight.astype(jnp.float32)), axis=1, keepdims=True) / 448
    inv = jnp.where(scale > 0, 1 / scale, 0)
    return jnp.clip(weight * inv, -448, 448).astype(jnp.float8_e4m3fn), scale


def _reference(x, w1, w2, gates, ids, s1, s2, *, fp8, act_scale="token"):
    # Independent direct gather/einsum reference; no padded packing or row loops.
    xf = x.astype(jnp.float32)
    if fp8:
        scale = jnp.max(jnp.abs(xf), axis=1, keepdims=True) / 448
        if act_scale == "tensor":
            scale = jnp.broadcast_to(jnp.max(scale), scale.shape)
        inv = jnp.where(scale > 0, 1 / scale, 0)
        xf = jnp.clip(xf * inv, -448, 448).astype(jnp.float8_e4m3fn).astype(jnp.float32)
    else:
        scale = jnp.ones((x.shape[0], 1), jnp.float32)
    out = jnp.zeros_like(xf)
    for k in range(ids.shape[1]):
        selected = ids[:, k]
        gu = jnp.einsum('td,tdi->ti', xf, w1[selected].astype(jnp.float32))
        if fp8:
            gu = gu * scale * s1[selected, 0, :]
        half = gu.shape[-1] // 2
        act = (jax.nn.silu(gu[:, :half]) * gu[:, half:]).astype(jnp.bfloat16)
        y = jnp.einsum('ti,tid->td', act.astype(jnp.float32),
                       w2[selected].astype(jnp.float32))
        if fp8:
            y = y * s2[selected, 0, :]
        out += y.astype(jnp.bfloat16).astype(jnp.float32) * gates[:, k:k+1].astype(
            jnp.bfloat16).astype(jnp.float32)
    return out.astype(jnp.bfloat16)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("bf16_rows", [16, 32])
@pytest.mark.parametrize("load", [0, 10, 20, 40, 64, 80, 257])
def test_expert_tiles_match_reference(fp8, bf16_rows, load):
    t, d, e, inter = max(80, load), 128, 4, 128
    rng = np.random.default_rng(7)
    x = jnp.asarray(rng.normal(size=(t, d)), jnp.bfloat16)
    w1 = jnp.asarray(rng.normal(size=(e, d, 2 * inter)) * .02, jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(size=(e, inter, d)) * .02, jnp.bfloat16)
    if fp8:
        w1, s1 = _quantize(w1)
        w2, s2 = _quantize(w2)
    else:
        s1 = s2 = None
    # Empty expert weights are poisoned; they must never contribute.
    w1 = w1.at[2].set(jnp.nan)
    w2 = w2.at[2].set(jnp.nan)
    ids = jnp.where(jnp.arange(t) < load, 0, 1).reshape(t, 1)
    logits = jnp.zeros((t, e)).at[jnp.arange(t), ids[:, 0]].set(10.)
    mesh = Mesh(np.array(jax.devices()[:1]), ('x',))
    actual = kernel.fused_moe_tp_tiled_serving(
        x, logits, w1, w2,
        None if s1 is None else s1[:, None],
        None if s2 is None else s2[:, None], mesh=mesh, axis_name='x',
        top_k=1, renormalize_topk_logits=True, token_tile_size=128,
        bf16_rows=bf16_rows, interpret=True)
    expected = _reference(x, w1, w2, jnp.ones((t, 1)), ids, s1, s2, fp8=fp8)
    np.testing.assert_allclose(np.asarray(actual, np.float32),
                               np.asarray(expected, np.float32), rtol=.04, atol=.01)


def test_routing_ties_and_normalization():
    for renormalize in (False, True):
        gates, ids = kernel._routing(jnp.zeros((3, 8)), top_k=2,
                                     renormalize=renormalize)
        np.testing.assert_array_equal(ids, [[7] * 3, [6] * 3])
        np.testing.assert_allclose(gates, .5 if renormalize else .125)


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("act_scale", ["token", "tensor"])
@pytest.mark.parametrize("width", [2, 8])
@pytest.mark.parametrize("local_tokens", [8, 64, 145])
def test_tp_serving_does_not_retrace_for_routing(fp8, act_scale, width, local_tokens):
    if jax.device_count() < width:
        pytest.skip('requires enough CPU devices or TPUs for the requested mesh')
    mesh = Mesh(np.array(jax.devices()[:width]), ('x',))
    t, d, e, inter = width * local_tokens, 128, 4, width * 128
    rng = np.random.default_rng(9)
    x = jnp.asarray(rng.normal(size=(t, d)), jnp.bfloat16)
    w1 = jnp.asarray(rng.normal(size=(e, d, 2, inter)) * .02, jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(size=(e, inter, d)) * .02, jnp.bfloat16)
    # Arrange each TP shard's own gate/up pair, as the serving loader does.
    packed_w1 = w1.reshape(e, d, 2, width, inter // width).transpose(0, 1, 3, 2, 4).reshape(
        e, d, 2 * inter)
    if fp8:
        packed_w1, scale1 = _quantize(packed_w1)
        w2, scale2 = _quantize(w2)
    else:
        scale1 = scale2 = None
    traces = []

    @jax.jit
    def run(logits):
        traces.append(1)
        return kernel.fused_moe_tp_tiled_serving(
            x, logits, packed_w1, w2,
            None if scale1 is None else scale1[:, None],
            None if scale2 is None else scale2[:, None],
            mesh=mesh, axis_name='x',
            top_k=2, renormalize_topk_logits=True, act_scale=act_scale, token_tile_size=width * 64, interpret=True)

    for hot in (0, 3):
        logits = jnp.zeros((t, e)).at[:, hot].set(10.)
        result = run(logits)
        gates, ids = kernel._routing(logits, top_k=2, renormalize=True)
        # Match each TP shard's intermediate rounding before the final sum.
        reference = jnp.zeros((t, d), jnp.bfloat16)
        for rank in range(width):
            a = packed_w1[:, :, rank * 256:(rank + 1) * 256]
            b = w2[:, rank * 128:(rank + 1) * 128, :]
            s1 = None if scale1 is None else scale1[:, :, rank * 256:(rank + 1) * 256]
            reference += _reference(x, a, b, gates.T, ids.T, s1, scale2, fp8=fp8, act_scale=act_scale)
        np.testing.assert_allclose(np.asarray(result, np.float32),
                                   np.asarray(reference, np.float32), rtol=.04, atol=.01)
    assert len(traces) == 1


@pytest.mark.parametrize('fp8', [False, True])
@pytest.mark.parametrize('tokens', [16, 512, 1024, 8192])
@pytest.mark.parametrize('bf16_rows', [16, 32])
def test_serving_lowers_for_tpu(fp8, tokens, bf16_rows):
    from jax import export
    from jax._src import mesh as mesh_lib
    width, hidden, experts, intermediate, topk = 8, 4096, 512, 1024, 10
    mesh = mesh_lib.AbstractMesh(
        (width,), ('x',), abstract_device=mesh_lib.AbstractDevice(
            device_kind='TPU7x', num_cores=1, platform='tpu'))
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    shapes = [jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16),
              jax.ShapeDtypeStruct((tokens, experts), jnp.bfloat16),
              jax.ShapeDtypeStruct((experts, hidden, 2 * intermediate), dtype),
              jax.ShapeDtypeStruct((experts, intermediate, hidden), dtype)]
    if fp8:
        shapes += [jax.ShapeDtypeStruct((experts, 1, 1, 2 * intermediate), jnp.float32),
                   jax.ShapeDtypeStruct((experts, 1, 1, hidden), jnp.float32)]
    fn = functools.partial(kernel.fused_moe_tp_tiled_serving,
                           mesh=mesh, axis_name='x', top_k=topk,
                           renormalize_topk_logits=True, bf16_rows=bf16_rows)
    with mesh_lib.use_abstract_mesh(mesh):
        compiled = export.export(jax.jit(fn), platforms=['tpu'])(*shapes)
    assert compiled.mlir_module_serialized


@pytest.mark.parametrize('tokens', [512, 2048, 8192])
@pytest.mark.parametrize('modifier', [None, 'e_score_correction_bias', 'defer_all_reduce'])
def test_serving_dispatch_uses_tiling_above_legacy_gate(tokens, modifier):
    # Execute the actual dispatch definitions without importing the unrelated
    # vLLM runtime. Backend spies let this test assert selection and forwarding.
    import ast
    from enum import Enum
    from pathlib import Path
    from types import SimpleNamespace

    path = Path(__file__).resolve().parents[2] / 'tpu_inference/layers/common/moe.py'
    source = ast.parse(path.read_text())
    names = {'MoEBackend', '_tp_decode_kernel_axis', 'moe_apply'}
    definitions = [node for node in source.body if getattr(node, 'name', None) in names]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
                      names=[ast.alias(name='annotations')], level=0)] + definitions,
                      type_ignores=[])
    selected = []
    env = SimpleNamespace(
        USE_MOE_TP_DECODE_KERNEL=True, MOE_TP_DECODE_MAX_TOKENS=512,
        MOE_TP_DECODE_ACT_SCALE='token', MOE_ALL_GATHER_ACTIVATION_DTYPE='',
        MOE_APPROX_TOPK=False, FORCE_MOE_RANDOM_ROUTING=False,
        ENABLE_RS_KERNEL=False, ONEHOT_MOE_PERMUTE_THRESHOLD=32768)
    ns = dict(jax=jax, jnp=jnp, Enum=Enum, envs=env,
              ShardingAxisName=SimpleNamespace(ATTN_DATA='x', MLP_TENSOR='x'),
              logger=SimpleNamespace(warning_once=lambda *args: None),
              fused_moe_tp_tiled_serving=lambda **kw: selected.append(('tiled', kw)),
              fused_moe_func=lambda **kw: selected.append(('gmm', kw)))
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), ns)
    weights = SimpleNamespace(
        w13_weight=jnp.zeros((4, 128, 512), jnp.bfloat16),
        w2_weight=jnp.zeros((4, 256, 128), jnp.bfloat16),
        w13_weight_scale=None, w2_weight_scale=None, w13_bias=None, w2_bias=None)
    layer = SimpleNamespace(activation='silu', scoring_func='softmax', top_k=2,
                            renormalize=True, use_ep=False, _get_name=lambda: 'test')
    kwargs = {'scatter_results': True}
    if modifier:
        kwargs[modifier] = True
    ns['moe_apply'](
        layer, jnp.zeros((tokens, 128), jnp.bfloat16),
        jnp.zeros((tokens, 4), jnp.float32), weights, ns['MoEBackend'].GMM_TP,
        SimpleNamespace(axis_names=('x',), shape={'x': 2}), kwargs)
    assert len(selected) == 1
    assert selected[0][0] == ('gmm' if modifier else 'tiled')
    assert selected[0][1]['hidden_states'].shape[0] == tokens


@pytest.mark.parametrize('fp8', [False, True])
def test_combine_chunks_preserve_outputs(fp8):
    # Exercise both accumulator axes and a final tile with one live token.
    t, d, e, inter = 257, 256, 4, 128
    rng = np.random.default_rng(23)
    x = jnp.asarray(rng.normal(size=(t, d)), jnp.bfloat16)
    w1 = jnp.asarray(rng.normal(size=(e, d, 2 * inter)) * .02, jnp.bfloat16)
    w2 = jnp.asarray(rng.normal(size=(e, inter, d)) * .02, jnp.bfloat16)
    if fp8:
        w1, s1 = _quantize(w1)
        w2, s2 = _quantize(w2)
    else:
        s1 = s2 = None
    logits = jnp.asarray(rng.normal(size=(t, e)), jnp.float32)
    gates, ids = kernel._routing(logits, top_k=2, renormalize=True)
    expected = _reference(x, w1, w2, gates.T, ids.T, s1, s2, fp8=fp8)
    mesh = Mesh(np.array(jax.devices()[:1]), ('x',))
    baseline = None
    for ct, cd in ((128, 128), (128, 256), (256, 128), (256, 256)):
        actual = kernel.fused_moe_tp_tiled_serving(
            x, logits, w1, w2,
            None if s1 is None else s1[:, None],
            None if s2 is None else s2[:, None], mesh=mesh, axis_name='x',
            top_k=2, renormalize_topk_logits=True, token_tile_size=256,
            combine_token_rows=ct, combine_hidden_cols=cd, interpret=True)
        actual = np.asarray(actual, np.float32)
        np.testing.assert_allclose(actual, np.asarray(expected, np.float32),
                                   rtol=.04, atol=.01)
        if baseline is not None:
            np.testing.assert_array_equal(actual, baseline)
        baseline = actual
