"""Validate one TP MoE candidate before measuring synchronized call latency."""
from __future__ import annotations

import argparse
import functools
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import time
import traceback

import diagnostics

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.fused_moe.v2 import tiled_tp


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_inputs(config: dict, mesh: Mesh) -> tuple:
    """Generate shards on device; never construct a full host weight tensor."""
    e, d = config['experts'], config['hidden']
    local_inter = config['intermediate'] // config['devices']
    local_tokens = config['tokens'] // config['devices']

    def generate():
        key = jax.random.fold_in(jax.random.key(config['seed']), lax.axis_index('x'))
        keys = jax.random.split(key, 4)
        x = jax.random.normal(keys[0], (local_tokens, d), dtype=jnp.float32).astype(jnp.bfloat16)
        logits = jax.random.normal(keys[1], (local_tokens, e), dtype=jnp.float32).astype(jnp.bfloat16)
        # Each rank already stores its own contiguous [gate | up] pair.
        a = (jax.random.normal(keys[2], (e, d, 2 * local_inter)) * .02).astype(jnp.bfloat16)
        b = (jax.random.normal(keys[3], (e, local_inter, d)) * .02).astype(jnp.bfloat16)
        sa = jnp.max(jnp.abs(a.astype(jnp.float32)), axis=1, keepdims=True) / 448.
        sb = lax.pmax(jnp.max(jnp.abs(b.astype(jnp.float32)), axis=1, keepdims=True), 'x') / 448.
        qa = jnp.clip(a.astype(jnp.float32) / jnp.where(sa > 0, sa, 1.), -448., 448.).astype(jnp.float8_e4m3fn)
        qb = jnp.clip(b.astype(jnp.float32) / jnp.where(sb > 0, sb, 1.), -448., 448.).astype(jnp.float8_e4m3fn)
        return x, logits, qa, qb, sa[:, None, :, :], sb[:, None, :, :]

    specs = (P('x', None), P('x', None), P(None, None, 'x'),
             P(None, 'x', None), P(None, None, None, 'x'), P())
    generate_sharded = jax.shard_map(generate, mesh=mesh, in_specs=(), out_specs=specs,
                                     check_vma=False)
    return jax.block_until_ready(jax.jit(generate_sharded)())


def reference(x: jax.Array, logits: jax.Array, w1: jax.Array, w2: jax.Array,
              s1: jax.Array, s2: jax.Array, *, top_k: int, mesh: Mesh) -> jax.Array:
    """Independent sampled reference; ordinary XLA gather/einsum, no Pallas."""
    def local(xs, gs, a, b, sa, sb):
        values, reversed_ids = lax.top_k(gs.astype(jnp.float32)[:, ::-1], top_k)
        ids = gs.shape[1] - 1 - reversed_ids
        gates = jax.nn.softmax(values, axis=1).astype(jnp.bfloat16).astype(jnp.float32)
        xf = xs.astype(jnp.float32)
        sx = jnp.max(jnp.abs(xf), axis=1, keepdims=True) / 448.
        qx = jnp.clip(xf / jnp.where(sx > 0, sx, 1.), -448., 448.).astype(
            jnp.float8_e4m3fn).astype(jnp.float32)

        def contribution(k, total):
            selected = ids[:, k]
            gu = jnp.einsum('sd,sdi->si', qx, a[selected].astype(jnp.float32),
                            precision=lax.Precision.HIGHEST)
            gu = gu * sx * sa[selected, 0, 0, :]
            half = gu.shape[1] // 2
            act = (jax.nn.silu(gu[:, :half]) * gu[:, half:]).astype(jnp.bfloat16)
            down = jnp.einsum('si,sid->sd', act.astype(jnp.float32),
                              b[selected].astype(jnp.float32), precision=lax.Precision.HIGHEST)
            down = (down * sb[selected, 0, 0, :]).astype(jnp.bfloat16).astype(jnp.float32)
            return total + down * lax.dynamic_slice_in_dim(gates, k, 1, axis=1)

        total = lax.fori_loop(0, top_k, contribution, jnp.zeros(xs.shape, jnp.float32))
        return lax.psum(total.astype(jnp.bfloat16), 'x')

    fn = jax.shard_map(local, mesh=mesh,
        in_specs=(P(), P(), P(None, None, 'x'), P(None, 'x', None), P(None, None, None, 'x'), P()),
        out_specs=P(), check_vma=False)
    return jax.jit(fn)(x, logits, w1, w2, s1, s2)


def sample_indices(config: dict) -> np.ndarray:
    t, width = config['tokens'], config['devices']
    per_rank = t // width
    samples = np.linspace(0, t - 1, min(t, config['reference_samples']), dtype=np.int32)
    boundaries = np.array([rank * per_rank + offset for rank in range(width)
                           for offset in (0, per_rank - 1)], dtype=np.int32)
    return np.unique(np.concatenate((samples, boundaries)))


def check_output(actual: jax.Array, inputs: tuple, *, config: dict, mesh: Mesh) -> dict:
    if not bool(jnp.all(jnp.isfinite(actual))):
        raise AssertionError('Kernel output contains non-finite values')
    x, logits, a, b, sa, sb = inputs
    indices = sample_indices(config=config)
    replicated = NamedSharding(mesh, P())
    xs = jax.device_put(x[indices], replicated)
    gs = jax.device_put(logits[indices], replicated)
    expected = reference(xs, gs, a, b, sa, sb, top_k=config['top_k'], mesh=mesh)
    expected = np.asarray(expected, dtype=np.float32)
    observed = np.asarray(actual[indices], dtype=np.float32)
    error = observed - expected
    result = {'passed': bool(np.allclose(observed, expected, rtol=.04, atol=.01)),
              'sample_indices': indices.tolist(), 'rtol': .04, 'atol': .01,
              'max_abs_error': float(np.max(np.abs(error))),
              'relative_l2_error': float(np.linalg.norm(error) / max(float(np.linalg.norm(expected)), 1e-12))}
    return result


def run_candidate(config: dict, output: Path, *, interpret: bool = False) -> dict:
    devices = jax.devices()
    if len(devices) != config['devices'] or (not interpret and any(d.platform != 'tpu' for d in devices)):
        raise RuntimeError(f"Expected {config['devices']} TPU devices, got {devices}")
    if hasattr(devices[0], 'coords'):
        devices = sorted(devices, key=lambda d: (d.coords[0],
            (-1 if d.coords[0] % 2 else 1) * d.coords[1], getattr(d, 'core_on_chip', 0)))
    mesh = Mesh(np.array(devices), ('x',))
    kernel_path = Path(tiled_tp.__file__).resolve()
    provenance = {'kernel_sha256': hashlib.sha256(kernel_path.read_bytes()).hexdigest(),
                  'jax': jax.__version__, 'devices': [str(d) for d in devices],
                  'interpret': interpret, 'config': config, 'revisions': {}}
    for name in ('jaxlib', 'libtpu', 'vllm', 'tpu-inference'):
        try:
            provenance[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            provenance[name] = None
    for name in ('vllm', 'tpu-inference'):
        path = Path('/opt') / name / 'JOBSET_REVISION'
        provenance['revisions'][name] = path.read_text().strip() if path.exists() else None
    save(path=output / 'provenance.json', value=provenance)
    print('GENERATE: sharded synthetic FP8 weights and BF16 activations', flush=True)
    inputs = make_inputs(config=config, mesh=mesh)
    candidate = jax.jit(functools.partial(
        tiled_tp.fused_moe_tp_tiled_serving, mesh=mesh, axis_name='x',
        top_k=config['top_k'], renormalize_topk_logits=True, act_scale='token',
        token_tile_size=config['token_tile_size'], bf16_rows=config['bf16_rows'], interpret=interpret))
    print('COMPILE: tiled TP serving entry', flush=True)
    start = time.monotonic()
    dump_root = output / 'compiler-dumps'
    before = diagnostics.snapshot(root=dump_root)
    # Parent can recover this window even if the compiler aborts the process.
    save(path=output / 'compile-window.json', value={'before': before})
    try:
        executable = candidate.lower(*inputs).compile()
    finally:
        try:
            files = diagnostics.freeze_compile(root=dump_root, before=before)
            save(path=output / 'compile-window.json', value={'before': before, 'files': files})
        except Exception:
            # Preserve the original compiler exception and leave raw dumps for
            # the parent if attribution/freezing itself fails.
            (output / 'compiler-freeze-error.txt').write_text(traceback.format_exc())
    compile_seconds = time.monotonic() - start
    # Parent retains raw Mosaic/JF dumps separately, including compile failures.
    if not interpret:
        with gzip.open(output / 'compiled-hlo.txt.gz', 'wt') as stream:
            stream.write(executable.as_text())
        (output / 'compiled-memory.txt').write_text(str(executable.memory_analysis()) + '\n')
    correctness = {}
    for mode in ('uniform', 'skew'):
        x, logits, a, b, sa, sb = inputs
        if mode == 'skew':
            # Every token chooses the same k experts; exercises repeated row
            # loops, empty experts and assignments beyond the former capacity.
            logits = jax.jit(lambda g: jnp.broadcast_to(
                jnp.arange(g.shape[1], dtype=jnp.bfloat16), g.shape))(logits)
        current = (x, logits, a, b, sa, sb)
        print(f'VALIDATE: {mode}', flush=True)
        actual = jax.block_until_ready(executable(*current))
        correctness[mode] = check_output(actual, current, config=config, mesh=mesh)
        save(path=output / 'correctness.json', value=correctness)
        if not correctness[mode]['passed']:
            raise AssertionError(f'Numerical comparison failed: {mode}: {correctness[mode]}')
    print('TIME: uniform routing; compilation and correctness excluded', flush=True)
    for _ in range(config['warmup']):
        jax.block_until_ready(executable(*inputs))
    samples = []
    for _ in range(config['iterations']):
        started = time.perf_counter_ns()
        jax.block_until_ready(executable(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1000.)
    result = {'status': 'ok', 'correctness': correctness, 'compile_seconds': compile_seconds,
              'metric': 'synchronized_call_wall_us_including_collectives',
              'median_us': statistics.median(samples), 'min_us': min(samples),
              'samples_us': samples, 'kernel_sha256': provenance['kernel_sha256']}
    save(path=output / 'result.json', value=result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        run_candidate(config=json.loads(args.config.read_text()), output=args.output)
    except Exception:
        (args.output / 'error.txt').write_text(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
