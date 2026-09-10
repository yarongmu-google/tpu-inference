"""Export saved diagnostics as verified archives while retaining local resume data."""
from __future__ import annotations

import fcntl
import gzip
from pathlib import Path
import shutil
import tempfile

from artifacts import make_bundle
from core import checksum, read_document, save


def export_run(directory: Path, destination: Path | None = None) -> Path | None:
    state = read_document(directory / 'state.json')
    image = directory / 'image'
    image.mkdir(exist_ok=True)
    with (image / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(f'Archive deferred: image preparation is still active at {directory}', flush=True)
            return None
        destination = destination or directory.parent.parent / 'archives'
        archive = destination / (state['run_id'] + '.tar.gz')
        files = {}
        for path in directory.rglob('*'):
            relative = path.relative_to(directory)
            if (relative.parts[0] in {'live', 'cleanup-empty'} or path.name == '.lock'
                    or relative.parts[:2] == ('collected', 'objects')
                    or (state.get('artifacts_verified') and relative.as_posix() == 'collected/artifacts.tar.gz')):
                continue
            if path.is_symlink():
                raise ValueError(f'Cannot archive a symlink: {path}')
            if path.is_file():
                files['run/' + relative.as_posix()] = path
        for name in ('campaign.json', 'description.json', 'profile.json'):
            path = directory.parent / name
            if path.is_file():
                files['campaign/' + name] = path
        print(f'COMPRESSING_LOCAL_RESULTS: {directory}', flush=True)
        make_bundle(destination=archive, files=files, metadata={'format': 'run-export-v1',
            'run_id': state['run_id'], 'phase': state['phase'], 'exit_code': state.get('exit_code'),
            'artifacts_verified': state.get('artifacts_verified', False)})
        save(path=archive.with_suffix('').with_suffix('.json'), value={
            'run_id': state['run_id'], 'phase': state['phase'], 'exit_code': state.get('exit_code'),
            'artifact_exit_code': state.get('artifact_exit_code'),
            'resources_cleaned': state.get('resources_cleaned', False),
            'archive': archive.name, 'sha256': checksum(archive), 'bytes': archive.stat().st_size})
        print(f'Result archive: {archive}', flush=True)
        return archive


def compress_log(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.packing-') as temporary:
        packed = Path(temporary) / 'log.gz'
        with source.open('rb') as incoming, gzip.open(packed, 'wb') as outgoing:
            remaining = source.stat().st_size
            while remaining:
                chunk = incoming.read(min(remaining, 1024 * 1024))
                if not chunk:
                    raise ValueError(f'Log shrank during compression: {source}')
                outgoing.write(chunk)
                remaining -= len(chunk)
        with gzip.open(packed, 'rb') as verified:
            while verified.read(1024 * 1024):
                pass
        packed.replace(destination)


def archive_saved(workflow_root: Path, target: Path | None = None) -> int:
    repository = workflow_root.parents[1]
    if target is not None:
        paths = [target] if (target / 'state.json').is_file() else [
            target / name for name in read_document(target / 'campaign.json')['jobs']]
    else:
        roots = [workflow_root / 'local/results', repository / 'workflow/local/results',
                 repository / 'tmp/serving-comparison/results']
        paths = [path.parent for root in roots for path in sorted(root.glob('*/*/state.json'))]
    destination = workflow_root / 'local/results/archives'
    failures = 0
    for directory in paths:
        with (directory / '.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'Skipping active controller: {directory}', flush=True)
                continue
            try:
                export_run(directory=directory, destination=destination)
            except Exception as error:
                failures += 1
                print(f'Archive failed for {directory}: {error}', flush=True)
    for root in (workflow_root / 'local/logs', repository / 'workflow/local/logs'):
        for source in sorted(root.glob('*.log')):
            # The new launcher compresses its own log after its writer exits.
            if root == workflow_root / 'local/logs':
                continue
            try:
                output = workflow_root / 'local/logs' / (source.name + '.gz')
                compress_log(source=source, destination=output)
                print(f'Legacy launcher log archive: {output}', flush=True)
            except Exception as error:
                failures += 1
                print(f'Log archive failed for {source}: {error}', flush=True)
    print(f'Archives ready under {destination}; raw saved runs remain available for resume', flush=True)
    return int(bool(failures))
