"""Persistent TPU state and CPU validation for a tuning session."""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import ModuleType
from concurrent.futures import ThreadPoolExecutor
import statistics
import time
import traceback

import diagnostics
from tune import KNOBS
from profile_tools import _device_kernel_ms_per_dispatch_from_trace

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.sharding import Mesh, PartitionSpec as P


def load_kernel(variant: str) -> ModuleType:
    if variant == 'original':
        import control_kernel
        return control_kernel
    if variant == 'occupied':
        from tpu_inference.kernels.fused_moe.v2 import decode_kernel_occupied
        return decode_kernel_occupied
    raise ValueError(f'Unknown variant: {variant}')


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_weights(config: dict, mesh: Mesh) -> tuple:
    """Generate and quantize the shared expert weights once for the session."""
    e, d = config['experts'], config['hidden']
    local_inter = config['intermediate'] // config['devices']

    def generate():
        key = jax.random.fold_in(jax.random.key(config['seed']), lax.axis_index('x'))
        keys = jax.random.split(key, 4)
        # Each rank already stores its own contiguous [gate | up] pair.
        a = (jax.random.normal(keys[2], (e, d, 2 * local_inter)) * .02).astype(jnp.bfloat16)
        b = (jax.random.normal(keys[3], (e, local_inter, d)) * .02).astype(jnp.bfloat16)
        sa = jnp.max(jnp.abs(a.astype(jnp.float32)), axis=1, keepdims=True) / 448.
        sb = lax.pmax(jnp.max(jnp.abs(b.astype(jnp.float32)), axis=1, keepdims=True), 'x') / 448.
        qa = jnp.clip(a.astype(jnp.float32) / jnp.where(sa > 0, sa, 1.), -448., 448.).astype(jnp.float8_e4m3fn)
        qb = jnp.clip(b.astype(jnp.float32) / jnp.where(sb > 0, sb, 1.), -448., 448.).astype(jnp.float8_e4m3fn)
        return qa, qb, sa[:, None, :, :], sb[:, None, :, :]

    fn = jax.shard_map(generate, mesh=mesh, in_specs=(),
        out_specs=(P(None, None, 'x'), P(None, 'x', None), P(None, None, None, 'x'), P()),
        check_vma=False)
    return jax.block_until_ready(jax.jit(fn)())


def make_inputs(config: dict, mesh: Mesh, weights: tuple) -> tuple:
    """Generate activations and both routing cases once per token count."""
    e, d = config['experts'], config['hidden']
    local_tokens = config['tokens'] // config['devices']

    def generate():
        key = jax.random.fold_in(jax.random.key(config['seed']), lax.axis_index('x'))
        keys = jax.random.split(key, 4)
        x = jax.random.normal(keys[0], (local_tokens, d), dtype=jnp.float32).astype(jnp.bfloat16)
        logits = jax.random.normal(keys[1], (local_tokens, e), dtype=jnp.float32).astype(jnp.bfloat16)
        # Select a bounded set of experts with roughly 32 assignments each.
        # Unlike all-to-top-k skew, this exercises empty experts without
        # requiring the separate overflow fix at the default capacity.
        active = min(e, max(config['top_k'], (config['tokens'] * config['top_k'] + 31) // 32))
        rows = jnp.arange(local_tokens) + lax.axis_index('x') * local_tokens
        experts = jnp.arange(e)[None, :]
        selected = ((experts - (e - active) - rows[:, None] * config['top_k']) % active < config['top_k'])
        selected &= experts >= e - active
        skew = jnp.where(selected, 1. + experts / e, -10.).astype(jnp.bfloat16)
        return x, logits, skew

    fn = jax.shard_map(generate, mesh=mesh, in_specs=(),
        out_specs=(P('x', None), P('x', None), P('x', None)), check_vma=False)
    x, logits, skew = jax.block_until_ready(jax.jit(fn)())
    return {'uniform': (x, logits, *weights), 'sparse': (x, skew, *weights)}


def reference(x: jax.Array, logits: jax.Array, w1: jax.Array, w2: jax.Array,
              s1: jax.Array, s2: jax.Array, *, top_k: int, mesh: Mesh) -> jax.Array:
    """Untuned full-output TPU XLA reference; ordinary gather/einsum, no Pallas."""
    def local(xs, gs, a, b, sa, sb):
        xs = lax.all_gather(xs, 'x', axis=0, tiled=True)
        gs = lax.all_gather(gs, 'x', axis=0, tiled=True)
        values, reversed_ids = lax.top_k(gs.astype(jnp.float32)[:, ::-1], top_k)
        ids = gs.shape[1] - 1 - reversed_ids
        exps = jnp.exp2((values - values[:, :1]) * jnp.float32(1.4426950408889634))
        gates = (exps / jnp.sum(exps, axis=1, keepdims=True)).astype(jnp.bfloat16).astype(jnp.float32)
        xf = xs.astype(jnp.float32)
        amax = jnp.max(jnp.abs(xf), axis=1, keepdims=True)
        sx = amax / 448.
        qx = jnp.clip(xf * jnp.where(amax > 0, 448. / amax, 0.), -448., 448.).astype(
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
        return lax.psum_scatter(total.astype(jnp.bfloat16), 'x',
                                scatter_dimension=0, tiled=True)

    fn = jax.shard_map(local, mesh=mesh,
        in_specs=(P('x', None), P('x', None), P(None, None, 'x'), P(None, 'x', None),
                  P(None, None, None, 'x'), P()),
        out_specs=P('x', None), check_vma=False)
    return jax.jit(fn)(x, logits, w1, w2, s1, s2)


def sample_indices(config: dict) -> np.ndarray:
    t, width = config['tokens'], config['devices']
    per_rank = t // width
    samples = np.linspace(0, t - 1, min(t, config['reference_samples']), dtype=np.int32)
    boundaries = np.array([rank * per_rank + offset for rank in range(width)
                           for offset in (0, per_rank - 1)], dtype=np.int32)
    return np.unique(np.concatenate((samples, boundaries)))


def check_output(actual: jax.Array, expected: jax.Array, *, config: dict) -> dict:
    expected = np.asarray(expected, dtype=np.float32)
    observed = np.asarray(actual, dtype=np.float32)
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(expected)):
        return {'passed': False, 'reason': 'Kernel or XLA output contains non-finite values',
                'checked_elements': int(expected.size)}
    assertion = None
    try:
        np.testing.assert_allclose(observed, expected, rtol=.04, atol=.01)
    except AssertionError as error:
        assertion = str(error)
    error = observed - expected
    failed = np.abs(error) > .01 + .04 * np.abs(expected)
    indices = sample_indices(config=config)
    return {'passed': assertion is None, 'comparison': 'all_output_elements',
            'rtol': .04, 'atol': .01, 'assertion': assertion,
            'failed_elements': int(np.count_nonzero(failed)),
            'checked_elements': int(expected.size),
            'max_abs_error': float(np.max(np.abs(error))),
            'relative_l2_error': float(np.linalg.norm(error) /
                                      max(float(np.linalg.norm(expected)), 1e-12)),
            'diagnostic_sample_indices': indices.tolist(),
            'diagnostic_sample_row_max_abs_error': np.max(np.abs(error[indices]), axis=1).tolist()}


def setup(config: dict, output: Path, *, interpret: bool) -> tuple[Mesh, dict]:
    devices = jax.devices()
    if len(devices) != config['devices'] or (not interpret and any(d.platform != 'tpu' for d in devices)):
        raise RuntimeError(f"Expected {config['devices']} TPU devices, got {devices}")
    if hasattr(devices[0], 'coords'):
        devices = sorted(devices, key=lambda d: (d.coords[0],
            (-1 if d.coords[0] % 2 else 1) * d.coords[1], getattr(d, 'core_on_chip', 0)))
    mesh = Mesh(np.array(devices), ('x',))
    provenance = {'tensor_contract': {'weights': 'float8_e4m3fn', 'input_activations': 'bfloat16',
                  'gmm1_activations': 'float8_e4m3fn', 'gmm2_activations': 'bfloat16',
                  'scales': 'float32 per-channel', 'routing': 'precomputed logits',
                  'control_revision': '729a68e09f'}, 'jax': jax.__version__, 'devices': [str(d) for d in devices],
                  'interpret': interpret, 'config': config, 'revisions': {},
                  'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for name in ('jaxlib', 'libtpu', 'vllm', 'tpu-inference'):
        try:
            provenance[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            provenance[name] = None
    for name in ('vllm', 'tpu-inference'):
        path = Path('/opt') / name / 'JOBSET_REVISION'
        provenance['revisions'][name] = path.read_text().strip() if path.exists() else None
    save(path=output / 'provenance.json', value=provenance)
    return mesh, provenance


def freeze_dumps(source: Path, output: Path) -> None:
    """Detach complete dump directories before the next compilation starts."""
    target = output / 'compiler-dumps'
    target.mkdir(parents=True, exist_ok=True)
    for name in ('mosaic', 'jf'):
        folder = source / name
        if folder.exists():
            folder.rename(target / name)
            folder.mkdir()


def compile_program(function, inputs: tuple, output: Path, *, dump_root: Path,
                    interpret: bool):
    start = time.monotonic()
    try:
        with diagnostics.phase(output=output, name='compile'):
            executable = function.lower(*inputs).compile()
    finally:
        # Constant-count directory renames; recursive inventory is a CPU task.
        freeze_dumps(source=dump_root, output=output)
    seconds = time.monotonic() - start
    return executable, seconds


def measure(function, inputs: tuple, *, config: dict) -> dict:
    for _ in range(config['warmup']):
        jax.block_until_ready(function(*inputs))
    samples = []
    for _ in range(config['iterations']):
        started = time.perf_counter_ns()
        jax.block_until_ready(function(*inputs))
        samples.append((time.perf_counter_ns() - started) / 1000.)
    return {'median_us': statistics.median(samples), 'min_us': min(samples),
            'samples_us': samples}



def host_output(value: jax.Array) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise AssertionError('Output contains non-finite values')
    result.setflags(write=False)
    return result


def save_arrays(directory: Path, arrays: dict[str, np.ndarray], *, chunk_bytes: int = 16 * 1024 * 1024) -> dict:
    """Store row chunks below the artifact limit; sealing supplies checksums."""
    outputs = {}
    for mode, array in arrays.items():
        if array.ndim != 2 or array.dtype != np.float32 or array.shape[1] * 4 > chunk_bytes:
            raise ValueError('Expected a float32 matrix with rows fitting the chunk budget')
        rows = max(1, chunk_bytes // (array.shape[1] * array.itemsize))
        folder = directory / f'expected-{mode}'
        folder.mkdir()
        chunks = []
        for start in range(0, array.shape[0], rows):
            stop = min(start + rows, array.shape[0])
            path = folder / f'rows-{start:08d}.npy'
            np.save(path, array[start:stop], allow_pickle=False)
            chunks.append({'file': str(path.relative_to(directory)), 'start': start, 'stop': stop})
        outputs[mode] = {'shape': list(array.shape), 'dtype': str(array.dtype), 'chunks': chunks}
    return outputs


class Engine:
    """One device owner; host threads never compile or invoke TPU programs."""
    def __init__(self, config: dict, output: Path, *, interpret: bool = False):
        self.output, self.interpret = output, interpret
        self.dump_root = output / '.compiler-active'
        self.mesh, self.provenance = setup(config=config, output=output, interpret=interpret)
        self.host = ThreadPoolExecutor(max_workers=2, thread_name_prefix='validation')
        self.writers = ThreadPoolExecutor(max_workers=1, thread_name_prefix='baseline-save')
        self.weights = None
        self.inputs = {}
        self.kernels = {}
        self.controls = {}

    def initialize(self, config: dict) -> None:
        directory = self.output / '.work/shared-weights'
        directory.mkdir(parents=True)
        with diagnostics.phase(output=directory, name='generate_weights'):
            try:
                self.weights = make_weights(config=config, mesh=self.mesh)
            finally:
                freeze_dumps(source=self.dump_root, output=directory)
        save(path=directory / 'config.json', value=config)
        save(path=directory / 'compile-window.json', value={'before': {}, 'files': []})

    def inputs_for(self, config: dict) -> dict:
        tokens = config['tokens']
        if tokens not in self.inputs:
            directory = self.output / '.work' / f'inputs-t{tokens}'
            directory.mkdir()
            with diagnostics.phase(output=directory, name='generate_inputs'):
                try:
                    self.inputs[tokens] = make_inputs(config=config, mesh=self.mesh, weights=self.weights)
                finally:
                    freeze_dumps(source=self.dump_root, output=directory)
            save(path=directory / 'compile-window.json', value={'before': {}, 'files': []})
        return self.inputs[tokens]

    def compile(self, config: dict, directory: Path, *, baseline: bool):
        inputs = self.inputs_for(config=config)
        if baseline:
            fn = functools.partial(reference, top_k=config['top_k'], mesh=self.mesh)
        else:
            variant = config['variant']
            if variant not in self.kernels:
                self.kernels[variant] = load_kernel(variant=variant)
            kernel = self.kernels[variant]
            local = functools.partial(kernel.fused_moe_decode_tp_fused,
                mesh=self.mesh, axis_name='x', top_k=config['top_k'], renormalize_topk_logits=True,
                router_fused=False, act_scale='token', interpret=self.interpret,
                **{key: config[key] or None if key in {'bd1c', 'bd2c', 'bcT'} else config[key] for key in KNOBS})
            fn = jax.shard_map(local, mesh=self.mesh,
                in_specs=(P('x', None), P('x', None), P(None, None, 'x'), P(None, 'x', None),
                          P(None, None, None, 'x'), P()),
                out_specs=P('x', None), check_vma=False)
        save(path=directory / 'config.json', value=config)
        provenance = {**self.provenance, 'config': config}
        if not baseline:
            provenance['kernel_sha256'] = hashlib.sha256(Path(kernel.__file__).read_bytes()).hexdigest()
        save(path=directory / 'provenance.json', value=provenance)
        executable, seconds = compile_program(function=jax.jit(fn), inputs=inputs['uniform'],
            output=directory, dump_root=self.dump_root, interpret=self.interpret)
        return {'executable': executable, 'inputs': inputs, 'compile_seconds': seconds}

    def execute(self, context: dict, directory: Path, *, baseline: bool) -> tuple[dict, dict]:
        outputs, calls = {}, {}
        for mode, inputs in context['inputs'].items():
            with diagnostics.phase(output=directory, name='execute_' + mode):
                started = time.perf_counter_ns()
                value = jax.block_until_ready(context['executable'](*inputs))
                calls[mode] = {('xla_us' if baseline else 'candidate_us'):
                              (time.perf_counter_ns() - started) / 1000.}
            save(path=directory / 'validation-calls.json', value=calls)
            # The main thread requests transfer, then continues TPU work.
            value.copy_to_host_async()
            outputs[mode] = value
        return outputs, calls

    def timed_profiles(self, context: dict, config: dict, directory: Path) -> tuple[dict, dict]:
        timings, captures = {}, {}
        for mode, inputs in context['inputs'].items():
            with diagnostics.phase(output=directory, name='warmup_and_timing_' + mode):
                timings[mode] = measure(function=context['executable'], inputs=inputs, config=config)
            # Persist warmed samples before any profiler failure or process crash.
            save(path=directory / 'timings.json', value=timings)
            trace = directory / 'profiles' / mode
            trace.mkdir(parents=True)
            if config['profile_iterations'] == 0 or self.interpret:
                captures[mode] = {'status': 'disabled'}
                continue
            started = False
            try:
                with diagnostics.phase(output=directory, name='profile_capture_' + mode):
                    options = jax.profiler.ProfileOptions()
                    options.python_tracer_level = 0
                    options.device_tracer_level = 2
                    jax.profiler.start_trace(str(trace), profiler_options=options)
                    started = True
                    for _ in range(config['profile_iterations']):
                        jax.block_until_ready(context['executable'](*inputs))
                captures[mode] = {'status': 'captured'}
            except Exception:
                captures[mode] = {'status': 'failed', 'error': traceback.format_exc()}
            finally:
                if started:
                    try:
                        with diagnostics.phase(output=directory, name='profile_export_' + mode):
                            jax.profiler.stop_trace()
                    except Exception:
                        captures[mode] = {'status': 'failed', 'error': traceback.format_exc()}
            save(path=directory / 'profile-capture.json', value=captures)
        return timings, captures

    def analyze_profiles(self, directory: Path, captures: dict) -> dict:
        profiles = {}
        for mode, capture in captures.items():
            if capture['status'] != 'captured':
                profiles[mode] = capture
                continue
            try:
                rows = _device_kernel_ms_per_dispatch_from_trace(str(directory / 'profiles' / mode), jit_name_prefix='jit_')
                profiles[mode] = ({'status': 'ok', 'tc_median_us': statistics.median(r[0] for r in rows) * 1000.,
                    'sc_median_us': statistics.median(r[1] for r in rows) * 1000.,
                    'tc_samples_us': [r[0] * 1000. for r in rows],
                    'metric': 'historical_TC_dispatch_excluding_barrier_and_trailing_copy'}
                    if rows else {'status': 'failed', 'reason': 'No matching device trace rows'})
            except Exception:
                profiles[mode] = {'status': 'failed', 'error': traceback.format_exc()}
        save(path=directory / 'profiles.json', value=profiles)
        return profiles

    def baseline(self, config: dict, directory: Path) -> dict:
        context = self.compile(config=config, directory=directory, baseline=True)
        outputs, calls = self.execute(context=context, directory=directory, baseline=True)
        expected = self.host.submit(lambda: {mode: host_output(value=value) for mode, value in outputs.items()})
        timings, captures = self.timed_profiles(context=context, config=config, directory=directory)
        record = {'kind': 'xla_baseline', 'status': 'ok', 'config': config,
                  'compile_seconds': context['compile_seconds'], 'validation_calls': calls,
                  'timings': timings,
                  'metric': 'synchronized_call_wall_us_including_collectives', **timings['uniform']}
        def persist():
            chunks = save_arrays(directory=directory, arrays=expected.result())
            record['profiles'] = self.analyze_profiles(directory=directory, captures=captures)
            save(path=directory / 'baseline.json', value={**record, 'outputs': chunks})
            return record
        persisted = self.writers.submit(persist)
        return {'expected': expected, 'persisted': persisted, 'record': record, 'directory': directory}

    def candidate(self, config: dict, directory: Path, baseline: dict) -> dict:
        key = (config['tokens'], *(config[name] for name in KNOBS))
        control = self.controls.pop(key, None) if config['variant'] == 'occupied' else None
        context = self.compile(config=config, directory=directory, baseline=False)
        outputs, calls = self.execute(context=context, directory=directory, baseline=False)
        save(path=directory / 'baseline-ref.json', value={'artifact': baseline['directory'].name})
        host = self.host.submit(lambda: {mode: np.asarray(value, dtype=np.float32) for mode, value in outputs.items()})
        if config['variant'] == 'original':
            self.controls[key] = host
        timings, captures = self.timed_profiles(context=context, config=config, directory=directory)
        def compare():
            actual, expected = host.result(), baseline['expected'].result()
            results = {mode: check_output(actual=value, expected=expected[mode], config=config)
                       for mode, value in actual.items()}
            paired = ({mode: {**check_output(actual=value, expected=control.result()[mode], config=config),
                              'bitwise_equal': bool(np.array_equal(value, control.result()[mode]))}
                       for mode, value in actual.items()} if control is not None else {})
            profiles = self.analyze_profiles(directory=directory, captures=captures)
            # This task runs on CPU while later TPU work proceeds. Outputs make
            # future accuracy debugging possible without rerunning the device.
            chunks = save_arrays(directory=directory, arrays=actual)
            save(path=directory / 'outputs.json', value=chunks)
            save(path=directory / 'correctness.json', value=results)
            save(path=directory / 'paired-correctness.json', value=paired)
            return {'correctness': results, 'paired_correctness': paired, 'profiles': profiles}
        return {**context, 'validation': self.host.submit(compare), 'calls': calls, 'timings': timings,
                'baseline': baseline, 'config': config, 'directory': directory, 'control_key': key,
                'control_available': control is not None}

    def finish_candidate(self, candidate: dict) -> dict:
        checked = candidate['validation'].result()
        config = candidate['config']
        accuracy = all(result['passed'] for result in checked['correctness'].values())
        paired = config['variant'] == 'original' or (candidate['control_available'] and
                 all(result['passed'] for result in checked['paired_correctness'].values()))
        profile_ok = all(result['status'] in {'ok', 'disabled'} for result in checked['profiles'].values())
        return {**checked, **candidate['timings']['uniform'], 'timings': candidate['timings'],
                'compile_seconds': candidate['compile_seconds'], 'validation_calls': candidate['calls'],
                'baseline_artifact': candidate['baseline']['directory'].name,
                'metric': 'synchronized_call_wall_us_including_collectives',
                'status': 'ok' if accuracy and paired and profile_ok else 'failed',
                'checks': {'xla_accuracy': accuracy, 'paired_accuracy': paired, 'profile': profile_ok}}

    def release(self, candidate: dict) -> None:
        candidate.pop('executable', None)
        if candidate['config']['variant'] == 'occupied':
            self.controls.pop(candidate['control_key'], None)

    def close(self) -> None:
        self.host.shutdown(wait=True)
        self.writers.shutdown(wait=True)
        self.controls.clear()
