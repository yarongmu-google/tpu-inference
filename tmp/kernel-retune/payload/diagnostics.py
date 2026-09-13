"""Probe compiler dump flags and archive evidence after each worker exits."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import resource
import shlex
import shutil
import subprocess
import sys
import tarfile

FLAGS = ('--xla_mosaic_dump_to', '--xla_jf_dump_to')


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def environment(root: Path, *, jf: bool) -> dict[str, str]:
    env = dict(os.environ)
    original = iter(shlex.split(env.get('LIBTPU_INIT_ARGS', '')))
    retained = []
    for token in original:
        if token in FLAGS:
            next(original, None)
        elif not any(token.startswith(flag + '=') for flag in FLAGS):
            retained.append(token)
    for flag, folder in zip(FLAGS, ('mosaic', 'jf')):
        if folder == 'jf' and not jf:
            continue
        path = (root / folder).resolve()
        path.mkdir(parents=True, exist_ok=True)
        retained.append(f'{flag}={path}')
    env['LIBTPU_INIT_ARGS'] = shlex.join(retained)
    return env


def snapshot(root: Path) -> dict[str, list[int]]:
    return {str(path.relative_to(root)): [path.stat().st_size, path.stat().st_mtime_ns]
            for path in sorted(root.rglob('*')) if path.is_file() and not path.is_symlink()}


def freeze_compile(root: Path, *, before: dict[str, list[int]]) -> list[str]:
    """Move this compilation's dumps away from later compiler writes."""
    files = []
    for name, state in snapshot(root=root).items():
        if state == before.get(name) or Path(name).parts[0] not in ('mosaic', 'jf'):
            continue
        destination = root / 'candidate-compile' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        (root / name).replace(destination)
        files.append(str(destination.relative_to(root)))
    return files


def disable_core_dumps() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def probe(output: Path) -> bool:
    """Return whether JF is accepted; retry only an explicitly unknown JF flag."""
    output.mkdir(parents=True, exist_ok=True)
    attempts = []
    for jf in (True, False):
        directory = output / ('mosaic-and-jf' if jf else 'mosaic-only')
        directory.mkdir()
        root = directory / 'compiler-dumps'
        env = environment(root=root, jf=jf)
        save(path=directory / 'dump-flags.json', value={
            'mosaic': str((root / 'mosaic').resolve()),
            'jf': str((root / 'jf').resolve()) if jf else None})
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--probe']
        with (directory / 'probe.log').open('w') as stream:
            try:
                result = subprocess.run(args=command, env=env, stdout=stream,
                    stderr=subprocess.STDOUT, timeout=120, preexec_fn=disable_core_dumps)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        log = (directory / 'probe.log').read_text(errors='replace')
        inventory = snapshot(root=root)
        final_llo = [name for name in inventory if 'post-finalize-llo' in name]
        attempt = {'jf_requested': jf, 'returncode': code,
                   'final_llo_files': final_llo,
                   'jf_files': [name for name in inventory if name.startswith('jf/')],
                   'log': f'{directory.name}/probe.log'}
        attempts.append(attempt)
        save(path=output / 'probe-results.json', value={'attempts': attempts})
        collect(directory=directory)
        if code == 0 and final_llo:
            save(path=output / 'probe-results.json', value={
                'attempts': attempts, 'jf_accepted': jf,
                'jf_emitted_probe_files': bool(attempt['jf_files']),
                'mosaic_final_llo_verified': True})
            print(f'DUMP_PREFLIGHT: Mosaic LLO present; JF accepted={jf}; '
                  f'JF probe files={len(attempt["jf_files"])}', flush=True)
            return jf
        if jf and 'unknown flag' in log.lower() and '--xla_jf_dump_to' in log:
            print('DUMP_PREFLIGHT: JF flag rejected; retrying Mosaic only; log retained', flush=True)
            continue
        raise RuntimeError(f'Dump preflight failed (exit={code}, final LLOs={len(final_llo)}); '
                           f'see {directory / "probe.log"}')
    raise RuntimeError('Mosaic dump preflight failed')


def collect(directory: Path) -> dict:
    """Preserve raw dumps plus conservative triage, including partial failures."""
    root = directory / 'compiler-dumps'
    inventory = snapshot(root=root)
    window_path = directory / 'compile-window.json'
    window = json.loads(window_path.read_text()) if window_path.exists() else None
    selected = (window.get('files') if window and 'files' in window else
                [name for name, state in inventory.items()
                 if state != window['before'].get(name)] if window else list(inventory))
    selected = set(selected)
    reports = []
    with (directory / 'compiler-findings.txt').open('w') as findings:
        for name in sorted(selected):
            path = root / name
            if name not in inventory or path.suffix not in ('.txt', '.mlir', '.ll', '.log', '.json', '.pbtxt', '.text', '.hlo', '.ir'):
                continue
            counts = Counter()
            hits = Counter()
            final_llo = 'post-finalize-llo' in name
            with path.open(errors='replace') as stream:
                for number, line in enumerate(stream, start=1):
                    if final_llo:
                        counts.update(re.findall(r'\bllo\.([a-zA-Z_][\w.]*)', line))
                    for label, pattern in (
                        ('spill_or_reload_text', r'\bspill\w*\b|\breload\w*\b'),
                        ('relayout_or_shuffle_text', r'\brelayout\b|\bvxpose\b|\bvperm\w*\b|\bvpack\b|\bvunpack\b'),
                        ('narrow_vector_type', r'vector<\d+x(?:1|10|16|32|64)x')):
                        if re.search(pattern, line, flags=re.IGNORECASE):
                            hits[label] += 1
                            # Retain all raw files; bound the human-readable excerpt only.
                            if hits[label] <= 40:
                                findings.write(f'{name}:{number}: {label}: {line[:600].rstrip()}\n')
            if counts or hits or final_llo:
                reports.append({'file': name, 'final_llo': final_llo,
                                'static_llo_name_occurrences': dict(counts),
                                'text_hits': dict(hits)})
    report = {'scope': 'candidate_compile' if window else 'unattributed',
              'files': inventory, 'selected_files': sorted(selected), 'reports': reports,
              'mosaic_final_llo_files': [name for name in selected if 'post-finalize-llo' in name],
              'jf_file_count': sum('jf' in Path(name).parts[:-1] for name in selected),
              'interpretation': 'Text hits are review leads, not proof of unexpected work. '
                  'Static name occurrences are not dynamic instruction counts. Mosaic post-finalize '
                  'LLO alone does not establish register allocation or spill absence. Inspect JF '
                  'output format and allocation evidence; no hits does not mean spill-free.'}
    save(path=directory / 'compiler-summary.json', value=report)
    if inventory:
        archive = directory / 'compiler-dumps.tar.gz'
        temporary = directory / 'compiler-dumps.tar.gz.new'
        with tarfile.open(temporary, 'w:gz') as stream:
            for name in inventory:
                stream.add(root / name, arcname=name, recursive=False)
        # Verify all payload bytes against the original files before removing raw dumps.
        import hashlib
        with tarfile.open(temporary, 'r:gz') as stream:
            if {member.name for member in stream.getmembers()} != set(inventory):
                raise RuntimeError('Compiler archive inventory mismatch')
            for name in inventory:
                with (root / name).open('rb') as raw, stream.extractfile(name) as stored:
                    if hashlib.file_digest(raw, 'sha256').digest() != hashlib.file_digest(stored, 'sha256').digest():
                        raise RuntimeError(f'Compiler archive checksum mismatch: {name}')
        temporary.replace(archive)
        shutil.rmtree(root)
    print(f'COMPILER_DUMPS: {directory.name}: {len(inventory)} files; '
          f'{len(report["mosaic_final_llo_files"])} candidate final LLOs; '
          f'{report["jf_file_count"]} JF files', flush=True)
    return report


def compile_probe() -> None:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu

    devices = jax.devices('tpu')
    if not devices or any(device.platform != 'tpu' for device in devices):
        raise RuntimeError('Dump preflight requires the actual TPU backend')

    def kernel(a, b, out):
        out[...] = jnp.dot(a[...], b[...], preferred_element_type=jnp.float32).astype(jnp.bfloat16)

    with jax.default_device(devices[0]):
        a = jnp.ones((32, 128), jnp.float8_e4m3fn)
        b = jnp.ones((128, 128), jnp.float8_e4m3fn)
        fn = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct((32, 128), jnp.bfloat16),
                            compiler_params=pltpu.CompilerParams())
        jax.block_until_ready(jax.jit(fn)(a, b))
    print('DUMP_PROBE_COMPILED_ON_TPU', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true', required=True)
    parser.parse_args()
    compile_probe()
