#!/usr/bin/env python3
"""Build or reuse a verified local runtime and publish its immutable reference."""
from __future__ import annotations

from collections.abc import Callable

import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

import prepare_jobset_image as snapshot

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'workflow/local/images'
IMAGE_ID = re.compile(r'sha256:[a-f0-9]{64}')


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def run(argv: list[str], *, capture: bool = False, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    print('+ ' + shlex.join(argv), flush=True)
    return subprocess.run(args=argv, cwd=ROOT, env=env, text=True,
                          stdout=subprocess.PIPE if capture else None, check=check)


def file_entry(path: Path, *, follow: bool) -> dict:
    value = {'mode': path.lstat().st_mode & 0o7777}
    if path.is_symlink():
        value['link'] = os.readlink(path)
        if not follow:
            return value
    if path.is_dir():
        value['directory'] = True
    elif path.is_file():
        with path.open('rb') as stream:
            value['sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    else:
        raise RuntimeError(f'Unsupported build input: {path}')
    return value


def tree(root: Path) -> dict:
    entries = {'.': file_entry(path=root, follow=False)}
    if root.is_symlink():
        return entries
    for base, directories, files in os.walk(root, followlinks=False):
        for name in sorted(directories + files):
            path = Path(base) / name
            entries[path.relative_to(root).as_posix()] = file_entry(path=path, follow=False)
    return entries


def sources(root: Path, keep: Callable[[str], bool]) -> dict:
    paths = [p for p in snapshot.git(root, 'ls-files', '-z').split('\0') if p and keep(p)]
    # Tracked edits and newly added source must never silently disappear from an image.
    changes = [p for p in snapshot.git(root, 'diff', '--name-only', '-z', 'HEAD').split('\0') if p and keep(p)]
    unknown = [p for p in snapshot.git(root, 'ls-files', '--others', '--exclude-standard', '-z').split('\0') if p and keep(p)]
    if changes or unknown:
        raise RuntimeError(f'Commit source changes before building in {root}: ' + ', '.join(sorted(set(changes + unknown))))
    return {'revision': snapshot.git(root, 'rev-parse', 'HEAD'),
            'files': {p: file_entry(path=root / p, follow=True) for p in paths}}


def inventory(vllm: Path, client: Path) -> dict:
    print('Checking source and environment contents for image reuse...', flush=True)
    environment = {name: tree(root=Path(sys.prefix) / name) for name in snapshot.RUNTIME_DIRECTORIES
                   if (Path(sys.prefix) / name).exists() or (Path(sys.prefix) / name).is_symlink()}
    code = {
        'tpu_inference': sources(root=ROOT, keep=snapshot.application_file),
        'vllm': sources(root=vllm, keep=lambda p: p.split('/', 1)[0] not in {'tmp', 'docs', 'examples', '.github', '.buildkite'}),
        'InferenceX': sources(root=client, keep=lambda p: p.startswith('utils/') and p != 'utils/aiperf'),
    }
    native = {p.relative_to(vllm).as_posix(): file_entry(path=p, follow=True)
              for p in (vllm / 'vllm').rglob('*') if p.is_file()
              and (p.name.endswith('.so') or '.so.' in p.name or p.name == 'vllm-rs')}
    scripts = {name: file_entry(path=ROOT / 'tmp' / name, follow=True)
               for name in ('Dockerfile.vllm12', 'prepare_jobset_image.py', 'build_jobset_image.sh', 'prepare_runtime_image.py')}
    return {'format': 1, 'python': sys.version, 'prefix': str(Path(sys.prefix).resolve()),
            'platform': platform.platform(), 'sources': code, 'native': native,
            'environment': environment, 'build_scripts': scripts}


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def prepare(repository: str, diagnostics: Path) -> dict:
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Build on the Linux x86_64 CPU VM')
    if sys.version_info[:2] != (3, 12) or os.environ.get('CONDA_DEFAULT_ENV') != 'vllm12' or not (Path(sys.prefix) / 'conda-meta').is_dir():
        raise RuntimeError('Activate the existing Python 3.12 vllm12 environment on the CPU VM, then rerun the launcher')
    spec = importlib.util.find_spec('vllm')
    if spec is None or spec.origin is None:
        raise RuntimeError('Cannot locate the editable vLLM source in the active environment')
    vllm = Path(snapshot.git(Path(spec.origin).resolve().parent, 'rev-parse', '--show-toplevel'))
    if os.environ.get('INFERENCEX_REPO'):
        client = Path(os.environ['INFERENCEX_REPO']).resolve()
    elif Path('/tmp/InferenceX/.git').exists():
        client = Path('/tmp/InferenceX')
    else:
        client = CACHE / 'InferenceX'
        if not client.exists():
            run(argv=['git', 'clone', '--depth', '1', 'https://github.com/SemiAnalysisAI/InferenceX.git', str(client)])
    before = inventory(vllm=vllm, client=client)
    digest = fingerprint(value=before)
    revisions = {key: value['revision'] for key, value in before['sources'].items()}
    save(path=diagnostics / 'inputs.json', value=before)
    print('Source revisions: ' + json.dumps(revisions, sort_keys=True), flush=True)
    print('Build input SHA256: ' + digest, flush=True)
    cached = CACHE / (digest + '.json')
    record = json.loads(cached.read_text()) if cached.is_file() else {}
    local = record.get('local_image_id', '')
    if not IMAGE_ID.fullmatch(local) or run(argv=['docker', 'image', 'inspect', '--format', '{{.Id}}', local], capture=True, check=False).stdout.strip() != local:
        build = diagnostics / 'build'
        run(argv=['bash', str(ROOT / 'tmp/build_jobset_image.sh')], env={**os.environ,
            'INFERENCEX_REPO': str(client), 'JOBSET_BUILD_DIAGNOSTICS': str(build),
            'JOBSET_BUILD_IMAGE': 'runtime:inputs-' + digest})
        local = (build / 'image-id.txt').read_text().strip()
        if not IMAGE_ID.fullmatch(local):
            raise RuntimeError('Builder did not return a Docker image ID')
        metadata = json.loads((build / 'image-source.json').read_text())
        if metadata['source_revisions'] != revisions:
            raise RuntimeError('Source revisions changed during the image build; rerun after edits finish')
        record = {'local_image_id': local, 'input_sha256': digest, 'source_revisions': revisions,
                  'build_directory': str(build)}
    else:
        print('Reusing verified local image: ' + local, flush=True)
    # Detect concurrent edits or environment installs before caching/publishing.
    if fingerprint(value=inventory(vllm=vllm, client=client)) != digest:
        raise RuntimeError('Build inputs changed during preparation; rerun after edits or installs finish')
    save(path=cached, value=record)
    tag = repository + ':inputs-' + digest
    run(argv=['gcloud', 'auth', 'configure-docker', repository.split('/')[0], '--quiet'])
    run(argv=['docker', 'tag', local, tag])
    run(argv=['docker', 'push', tag])
    response = run(argv=['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', tag], capture=True)
    digests = [d for d in json.loads(response.stdout) if d.startswith(repository + '@sha256:')
               and re.fullmatch(r'[a-z0-9._:/-]+@sha256:[a-f0-9]{64}', d)]
    if len(digests) != 1:
        raise RuntimeError('Cannot resolve one published digest for the prepared image')
    return {**record, 'image': digests[0]}


def main() -> None:
    repository = os.environ.get('IMAGE_REPOSITORY', '')
    if not re.fullmatch(r'[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9._-]+/[a-z0-9._/-]+', repository):
        raise RuntimeError('IMAGE_REPOSITORY must be an Artifact Registry image path without a tag')
    diagnostics = Path(os.environ['IMAGE_BUILD_LOG_DIR']).resolve()
    result = Path(os.environ['IMAGE_RESULT']).resolve()
    CACHE.mkdir(parents=True, exist_ok=True)
    print('Waiting for the local image preparation lock...', flush=True)
    with (CACHE / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        save(path=result, value=prepare(repository=repository, diagnostics=diagnostics))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Image preparation failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
