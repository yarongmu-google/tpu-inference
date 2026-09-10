"""Build and publish one run-owned image from the selected source commits."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
VLLM_REVISION = 'bec0a4ede621bc9f1ec7be39b79ed2e286da4ca1'
TPU_BASE = '4c0dc1cd11a65b0631a9b36696bc0774db5ce15c'


def run(argv: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    print('+ ' + shlex.join(argv), flush=True)
    return subprocess.run(args=argv, cwd=ROOT, text=True, check=True,
                          stdout=subprocess.PIPE if capture else None)


def save(path: Path, value: dict) -> None:
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def git(*args: str) -> str:
    return run(argv=['git', *args], capture=True).stdout.strip()


def runtime_file(path: str) -> bool:
    return path.startswith('tpu_inference/') or path in {
        'requirements.txt', 'setup.py', 'pyproject.toml', 'README.md', 'LICENSE', 'MANIFEST.in'}


def prepare() -> None:
    repository = os.environ['IMAGE_REPOSITORY']
    run_id = os.environ['IMAGE_RUN_ID']
    if (not re.fullmatch(r'[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9-]+/[a-z0-9._-]+/[a-z0-9._/-]+', repository)
            or not re.fullmatch(r'[a-z0-9-]+-[a-f0-9]{24}', run_id)
            or not repository.endswith('/' + run_id)):
        raise ValueError('Image repository must belong to this workflow run')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        raise RuntimeError('Build on the Linux x86_64 CPU VM')
    if git('branch', '--show-current') != 'bench':
        raise RuntimeError('Run from the bench branch')
    revision = git('rev-parse', 'HEAD')
    run(argv=['git', 'merge-base', '--is-ancestor', TPU_BASE, revision])
    files = [p for p in git('ls-tree', '-r', '--name-only', revision).splitlines() if runtime_file(p)]
    # These first baseline jobs must use the exact requested runtime source.
    if git('diff', '--name-only', TPU_BASE, revision, '--', *files):
        raise RuntimeError('Runtime sources differ from the requested baseline commit')
    checked = [*files, 'tmp/baselines', 'scripts/vllm/benchmarking/infx_server.sh',
               'scripts/vllm/benchmarking/infx_client.sh']
    if git('diff', '--name-only', 'HEAD', '--', *checked):
        raise RuntimeError('Commit the reviewed baseline changes before building')
    unknown = git('ls-files', '--others', '--exclude-standard').splitlines()
    if any(runtime_file(p) for p in unknown):
        raise RuntimeError('Untracked runtime source would be missing from the image')
    diagnostics = Path(os.environ['IMAGE_BUILD_LOG_DIR'])
    diagnostics.mkdir(parents=True, exist_ok=True)
    result = Path(os.environ['IMAGE_RESULT'])
    host, project, registry, _ = repository.split('/', 3)
    response = run(argv=['gcloud', '--project', project, 'artifacts', 'repositories', 'describe', registry,
        '--location', host.removesuffix('-docker.pkg.dev'), '--format=value(format)'], capture=True)
    if response.stdout.strip() != 'DOCKER':
        raise ValueError('The configured Artifact Registry repository must use Docker format')
    local_tag, tag = 'runtime:' + run_id, repository + ':run'
    with tempfile.TemporaryDirectory(prefix='baseline-image-') as temporary:
        context = Path(temporary)
        target = context / 'tpu-inference'
        target.mkdir()
        # Export committed files only; no environment, credentials or result logs.
        for name in files:
            destination = target / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(subprocess.check_output(args=['git', 'show', revision + ':' + name], cwd=ROOT))
        (target / 'JOBSET_REVISION').write_text(revision + '\n')
        for name in ('Dockerfile', 'constraints.txt', 'smoke.py'):
            shutil.copyfile(src=HERE / name, dst=context / name)
        record = {'source_revisions': {'tpu_inference': revision, 'vllm': VLLM_REVISION},
                  'tpu_base': TPU_BASE, 'source_branches': {'tpu_inference': 'bench', 'vllm': 'bench'},
                  'torch_build_policy': 'CPU Torch 2.10 for TPU torchvision 0.25; vLLM built without isolation'}
        save(path=context / 'image-source.json', value=record)
        save(path=diagnostics / 'image-source.json', value=record)
        run(argv=['docker', 'build', '--progress=plain', '--platform=linux/amd64',
            '--build-arg', 'VLLM_REVISION=' + VLLM_REVISION, '--iidfile', str(diagnostics / 'image-id.txt'),
            '-t', local_tag, str(context)])
    local_id = (diagnostics / 'image-id.txt').read_text().strip()
    if not re.fullmatch(r'sha256:[a-f0-9]{64}', local_id):
        raise ValueError('Docker did not return an image ID')
    run(argv=['docker', 'run', '--rm', '--network=none', '-e', 'JAX_PLATFORMS=cpu',
              local_id, 'python', '/opt/jobset/smoke.py'])
    response = run(argv=['docker', 'run', '--rm', '--network=none', local_id,
                         'cat', '/opt/jobset/image-source.json'], capture=True)
    save(path=diagnostics / 'image-source.json', value=json.loads(response.stdout))
    record.update(owner_run_id=run_id, local_image_id=local_id, registry_image=repository,
                  local_tags=[local_tag, tag])
    run(argv=['gcloud', 'auth', 'configure-docker', host, '--quiet'])
    run(argv=['docker', 'tag', local_id, tag])
    save(path=diagnostics / 'publication.json', value={**record, 'publication_started': True})
    run(argv=['docker', 'push', tag])
    response = run(argv=['docker', 'image', 'inspect', '--format', '{{json .RepoDigests}}', tag], capture=True)
    digests = [d for d in json.loads(response.stdout) if d.startswith(repository + '@sha256:')
               and re.fullmatch(r'[a-z0-9._:/-]+@sha256:[a-f0-9]{64}', d)]
    if len(digests) != 1:
        raise ValueError('Cannot resolve one published image digest')
    save(path=result, value={**record, 'image': digests[0]})


if __name__ == '__main__':
    prepare()
