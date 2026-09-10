"""Execute one frozen description and publish verifiable artifacts."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

from core import CDK_MOUNT_ROOT, cdk_storage_directory, checksum, json_bytes, safe_path, verify


def write_remote(path: Path, value: dict) -> None:
    # Close the object before publishing any reference to it; no FUSE rename assumptions.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        stream.write(json_bytes(value))


def publish(local: Path, bucket: Path, status: dict, extras: dict[str, str]) -> None:
    entries = []
    roots = {'diagnostics': local, **{f'artifacts/{key}': Path(value) for key, value in extras.items()}}
    object_root = bucket / 'objects'
    object_root.mkdir(exist_ok=True)
    for prefix, root in roots.items():
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f'Output is not a regular directory: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'Output symlink is not collected: {path}')
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if root == local and path.is_relative_to(local / 'artifacts'):
                continue
            safe_path(root=root, relative=relative)
            # Freeze exactly the observed prefix of growing logs.
            with tempfile.TemporaryFile() as frozen:
                remaining = path.stat().st_size
                with path.open('rb') as source:
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            break
                        frozen.write(chunk)
                        remaining -= len(chunk)
                length = frozen.tell()
                frozen.seek(0)
                import hashlib
                digest = hashlib.file_digest(frozen, 'sha256').hexdigest()
                destination = object_root / digest
                if not destination.exists() or destination.stat().st_size != length or checksum(destination) != digest:
                    frozen.seek(0)
                    with destination.open('wb') as output:
                        shutil.copyfileobj(fsrc=frozen, fdst=output)
                entries.append({'path': prefix + '/' + relative, 'sha256': digest, 'bytes': length})
    write_remote(path=bucket / 'manifest.json', value={'format': 'run-artifacts-v1', **status, 'files': entries})


def publish_final(local: Path, bucket: Path, status: dict, extras: dict[str, str]) -> None:
    from artifacts import make_bundle
    files = {}
    roots = {'diagnostics': local, **{f'artifacts/{key}': Path(value) for key, value in extras.items()}}
    for prefix, root in roots.items():
        if not root.exists():
            continue
        if root.is_symlink() or not root.is_dir():
            raise ValueError(f'Invalid output directory: {root}')
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'Output symlink is not collected: {path}')
            if path.is_file() and not (root == local and path.is_relative_to(local / 'artifacts')):
                files[prefix + '/' + path.relative_to(root).as_posix()] = path
    print('COMPRESSING_RESULTS: preparing the final artifact bundle', flush=True)
    with tempfile.TemporaryDirectory(prefix='final-artifacts-') as temporary:
        archive = Path(temporary) / 'artifacts.tar.gz'
        manifest = make_bundle(destination=archive, files=files, metadata={'format': 'run-artifacts-v1', **status})
        digest = checksum(archive)
        destination = bucket / 'archives' / (digest + '.tar.gz')
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(src=archive, dst=destination)
        manifest['archive'] = {'path': 'archives/' + destination.name, 'sha256': digest, 'bytes': archive.stat().st_size}
        write_remote(path=bucket / 'manifest.json', value=manifest)
    print('RESULTS_COMPRESSED: final artifact bundle published', flush=True)


def materialize(bundle: Path, items: list[dict]) -> None:
    for item in items:
        source = safe_path(root=bundle, relative=item['source'])
        # Flat buckets may not expose uploaded directories without markers.
        source.mkdir(parents=True, exist_ok=True)
        for entry in item['files']:
            safe_path(root=source, relative=entry['path']).parent.mkdir(parents=True, exist_ok=True)
        verify(root=source, entries=item['files'])
        target = Path(item['destination'])
        for parent in [target, *target.parents]:
            if parent.is_symlink():
                raise ValueError(f'Destination traverses symlink: {target}')
        target.mkdir(parents=True, exist_ok=True)
        for entry in item['files']:
            dest = safe_path(root=target, relative=entry['path'])
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src=source / entry['path'], dst=dest)
            dest.chmod(entry['mode'])


def expand(value: str, variables: dict[str, str]) -> str:
    def replacement(match: re.Match) -> str:
        key = match.group(1)
        if key not in variables:
            raise ValueError(f'Unknown argument variable: {key}')
        return variables[key]
    return re.sub(r'\$\{([A-Z_][A-Z0-9_]*)\}', replacement, value)


def execute(bucket: Path, local: Path, expected: str, require_mount: bool = True, mount_root: Path | None = None, input_wait_seconds: int = 0) -> int:
    local.mkdir(parents=True, exist_ok=True)
    status = {'run_id': os.environ['RUN_ID'], 'image': os.environ.get('RUN_IMAGE'),
              'config_sha256': expected, 'state': 'starting', 'exit_code': None, 'updated': time.time()}
    storage_ready = False
    child = None
    interrupted = threading.Event()
    signum = [0]
    def stop(number: int, _frame: object) -> None:
        signum[0] = number
        interrupted.set()
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    previous_signals = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    for number in previous_signals:
        signal.signal(number, stop)
    extras = {}
    code = 1
    threads = []
    try:
        if require_mount:
            mounts = Path('/proc/self/mountinfo').read_text().splitlines()
            if not any(line.split()[4] == str(mount_root or bucket) and 'fuse' in line for line in mounts):
                raise RuntimeError('Storage mount is missing; workload will not start')
        bucket.mkdir(parents=True, exist_ok=True)
        storage_ready = True
        (bucket / 'input').mkdir(parents=True, exist_ok=True)
        config_path = bucket / 'input/run.json'
        if input_wait_seconds:
            print('WAITING_FOR_INPUTS: controller is preparing this job folder', flush=True)
            deadline = time.monotonic() + input_wait_seconds
            while not config_path.is_file() or not (bucket / 'owner.json').is_file():
                if interrupted.is_set():
                    raise InterruptedError('Interrupted before input delivery')
                if time.monotonic() >= deadline:
                    raise TimeoutError('No job inputs; inspect upload logs and resume the controller')
                interrupted.wait(1)
        if checksum(config_path) != expected:
            raise ValueError('Run configuration checksum mismatch')
        config = json.loads(config_path.read_text())
        if config.get('format') != 'run-description-v1':
            raise ValueError('Unsupported run description')
        if config['run_id'] != status['run_id']:
            raise ValueError('Wrong run identity')
        status.update(image=config['image'], config_sha256=expected)
        owner = json.loads((bucket / 'owner.json').read_text())
        if owner['run_id'] != config['run_id'] or owner['nonce'] != config['nonce']:
            raise ValueError('Wrong bucket ownership marker')
        proof = {'run_id': config['run_id'], 'config_sha256': expected, 'timestamp': time.time()}
        write_remote(path=bucket / 'control/ready.json', value=proof)
        if json.loads((bucket / 'control/ready.json').read_text()) != proof:
            raise RuntimeError('Dedicated bucket read/write check failed')
        print('STORAGE_READY: job storage verified; waiting for launch authorization', flush=True)
        deadline = time.monotonic() + 900
        while True:
            if interrupted.is_set():
                raise InterruptedError('Interrupted before workload start')
            gate = bucket / 'control/start.json'
            if gate.exists():
                if json.loads(gate.read_text()) != {'run_id': config['run_id'], 'config_sha256': expected}:
                    raise ValueError('Invalid launch authorization')
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('No launch authorization; inspect the rendered recipe and resume controller')
            interrupted.wait(2)
        materialize(bundle=bucket / 'input', items=config['bundles'])
        env = {**os.environ, **config['run']['env'], 'RUN_NAME': config['name'],
               'RUN_ID': config['run_id'], 'OUTPUT_DIR': str(local / 'artifacts')}
        env.update({f'INPUT_{key.upper()}_DIR': item['destination'] for key, item in config['inputs'].items()})
        Path(env['OUTPUT_DIR']).mkdir()
        extras = {'output': env['OUTPUT_DIR'], **config['outputs']['extra']}
        argv = [expand(value=a, variables=env) for a in config['run']['argv']]
        cwd = Path(config['run']['cwd'])
        if not cwd.is_dir():
            raise ValueError(f'Working directory does not exist: {cwd}')
        # Verify module origins using the same cwd/environment as the actual command.
        if config['run']['verify_imports']:
            check = ('import importlib.util,json,sys; from pathlib import Path; '
                     'checks=json.loads(sys.argv[1]); '
                     '[(lambda s,p: s is not None and s.origin is not None and '
                     'Path(s.origin).resolve().is_relative_to(Path(p).resolve()) '
                     'or sys.exit("Unexpected import origin"))(importlib.util.find_spec(k),v) '
                     'for k,v in checks.items()]')
            subprocess.run(args=[sys.executable, '-c', check, json.dumps(config['run']['verify_imports'])],
                           cwd=cwd, env=env, check=True, timeout=60)
        write_remote(path=local / 'execution.json', value={'argv': argv, 'cwd': str(cwd), **status})
        status.update(state='running', started=time.time())
        print('RUNNING: workload started', flush=True)
        child = subprocess.Popen(args=argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        def capture(source, filename: str, target) -> None:
            with (local / filename).open('wb') as output:
                while chunk := source.read1(65536):
                    output.write(chunk)
                    output.flush()
                    try:
                        target.buffer.write(chunk)
                        target.buffer.flush()
                    except (BrokenPipeError, AttributeError):
                        pass
        threads = [threading.Thread(target=capture, args=(child.stdout, 'stdout.log', sys.stdout)),
                   threading.Thread(target=capture, args=(child.stderr, 'stderr.log', sys.stderr))]
        for thread in threads:
            thread.start()
        next_publish = 0.0
        deadline = time.monotonic() + config['timeout_seconds']
        stopped_at = None
        while child.poll() is None:
            now = time.monotonic()
            if (interrupted.is_set() or now >= deadline) and stopped_at is None:
                os.killpg(child.pid, signal.SIGTERM)
                stopped_at = now
            if stopped_at is not None and now - stopped_at >= 20:
                os.killpg(child.pid, signal.SIGKILL)
            if now >= next_publish:
                status['updated'] = time.time()
                try:
                    publish(local=local, bucket=bucket, status=status, extras=extras)
                except Exception as error:
                    print(f'Artifact snapshot failed: {error}', file=sys.stderr, flush=True)
                next_publish = now + config['outputs']['snapshot_seconds']
            time.sleep(0.2)
        code = child.wait()
        code = 128 + signum[0] if signum[0] else (124 if stopped_at is not None else (128 - code if code < 0 else code))
    except BaseException:
        detail = traceback.format_exc()
        (local / 'error.txt').write_text(detail)
        print(detail, file=sys.stderr, flush=True)
        code = 128 + signum[0] if signum[0] else 1
    finally:
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        for thread in threads:
            thread.join(timeout=10)
        if child is not None:
            child.stdout.close()
            child.stderr.close()
        status.update(state='succeeded' if code == 0 else 'failed', exit_code=code, updated=time.time())
        write_remote(path=local / 'status.json', value=status)
        try:
            if storage_ready:
                publish_final(local=local, bucket=bucket, status=status, extras=extras)
            else:
                print('Storage unavailable; startup diagnostics remain in the container log', file=sys.stderr, flush=True)
        except Exception:
            print('Final artifact publication failed:\n' + traceback.format_exc(), file=sys.stderr, flush=True)
            code = code or 1
    for number, handler in previous_signals.items():
        signal.signal(number, handler)
    return code


def main(local: Path = Path('/run-work')) -> int:
    if os.environ.get('WORKFLOW_STORAGE_MODE') == 'cdk':
        # CDK_OUTPUT_DIR is a per-container output directory, not the FUSE mount.
        bucket = cdk_storage_directory(run_id=os.environ['RUN_ID'])
        print(f'RUN_STORAGE: mount={CDK_MOUNT_ROOT}; run directory={bucket}', flush=True)
        return execute(bucket=bucket, mount_root=CDK_MOUNT_ROOT, local=local,
                       expected=os.environ['RUN_CONFIG_SHA256'], input_wait_seconds=900)
    return execute(bucket=Path('/run-storage'), local=local, expected=os.environ['RUN_CONFIG_SHA256'])


if __name__ == '__main__':
    sys.exit(main())
