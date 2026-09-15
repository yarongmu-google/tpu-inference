"""Seal independent artifact directories and verify them on the receiving host."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from core import checksum, safe_path, save, link_or_copy

PART_BYTES = 40 * 1024 * 1024


class HashingReader:
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        return data


def seal(source: Path, destination: Path, names: list[str], metadata: dict) -> dict:
    """Read frozen files once, compress quickly, then expose the whole directory."""
    started = time.monotonic()
    if destination.exists():
        raise FileExistsError(destination)
    if len(names) != len(set(names)):
        raise ValueError('Duplicate artifact name')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / '.processing'
    staging.mkdir(exist_ok=True)
    record = {'format': 'candidate-artifacts-v1', 'candidate': destination.name,
              'metadata': metadata, 'files': [], 'parts': []}
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        pending = Path(temporary) / destination.name
        pending.mkdir()
        bundle, current, size = None, None, 0
        try:
            for name in sorted(names):
                path = safe_path(root=source, relative=name)
                if not path.is_file() or path.is_symlink():
                    raise ValueError(f'Artifact is not a regular file: {name}')
                initial = path.stat()
                if initial.st_size > PART_BYTES:
                    raise ValueError(f'Artifact exceeds the 40 MiB part limit: {name}')
                if bundle is None or size + initial.st_size > PART_BYTES:
                    if bundle is not None:
                        bundle.close()
                    current = f'part-{len(record["parts"]):04d}.tar.gz'
                    record['parts'].append({'name': current})
                    bundle = tarfile.open(pending / current, 'w:gz', compresslevel=1)
                    size = 0
                info = tarfile.TarInfo(name)
                info.size = initial.st_size
                with path.open('rb') as raw:
                    reader = HashingReader(raw)
                    bundle.addfile(info, reader)
                final = path.stat()
                if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
                    raise ValueError(f'Artifact changed while packaging: {name}')
                record['files'].append({'path': name, 'part': current, 'bytes': info.size,
                                        'sha256': reader.digest.hexdigest()})
                size += info.size
        finally:
            if bundle is not None:
                bundle.close()
        for entry in record['parts']:
            path = pending / entry['name']
            entry.update(bytes=path.stat().st_size, sha256=checksum(path))
            if entry['bytes'] >= 50 * 1024 * 1024:
                raise ValueError('Compressed part exceeds 50 MiB')
        record['pack_seconds'] = time.monotonic() - started
        save(path=pending / 'candidate.json', value=record)
        pending.rename(destination)
    return record


def materialize(source: Path, destination: Path) -> bool:
    """Verify and unpack one completed candidate without changing its bundle."""
    manifest_path = source / 'candidate.json'
    manifest_hash = checksum(manifest_path)
    if destination.exists():
        receipt = json.loads((destination / 'received.json').read_text())
        if receipt['manifest_sha256'] != manifest_hash:
            raise ValueError('Published candidate changed after receipt')
        return False
    started = time.monotonic()
    record = json.loads(manifest_path.read_text())
    if record.get('format') != 'candidate-artifacts-v1' or record.get('candidate') != source.name:
        raise ValueError('Invalid candidate identity')
    files = {entry['path']: entry for entry in record['files']}
    parts = {entry['name']: entry for entry in record['parts']}
    if len(files) != len(record['files']) or len(parts) != len(record['parts']):
        raise ValueError('Duplicate artifact entry')
    if any(entry['part'] not in parts for entry in files.values()):
        raise ValueError('Artifact refers to an undeclared part')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / '.processing'
    staging.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=staging) as temporary:
        pending = Path(temporary) / destination.name
        pending.mkdir()
        seen = set()
        for part, entry in parts.items():
            if Path(part).name != part or not part.startswith('part-') or not part.endswith('.tar.gz'):
                raise ValueError('Invalid part name')
            path = safe_path(root=source, relative=part)
            if path.stat().st_size != entry['bytes'] or checksum(path) != entry['sha256']:
                raise ValueError('Compressed part checksum mismatch')
            with gzip.open(path, 'rb') as compressed:
                with tarfile.open(fileobj=compressed, mode='r|') as bundle:
                    for member in bundle:
                        expected = files.get(member.name)
                        if (not member.isfile() or member.name in seen or expected is None
                                or expected['part'] != part or expected['bytes'] != member.size):
                            raise ValueError('Unexpected artifact member')
                        target = safe_path(root=pending / 'files', relative=member.name)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with bundle.extractfile(member) as raw, target.open('wb') as output:
                            reader = HashingReader(raw)
                            shutil.copyfileobj(reader, output)
                        if reader.digest.hexdigest() != expected['sha256']:
                            raise ValueError(f'Artifact checksum mismatch: {member.name}')
                        seen.add(member.name)
                        bundle.members.clear()
                while compressed.read(1024 * 1024):
                    pass
            link_or_copy(source=path, destination=pending / part)
        if seen != set(files):
            raise ValueError('Missing artifact members')
        compact = record.get('metadata', {}).get('compact_after_receive', False)
        if type(compact) is not bool:
            raise ValueError('compact_after_receive must be Boolean')
        if compact:
            # Opt-in by the workload. Full member verification above completes
            # before removing unpacked bulk; verified compressed parts remain.
            # Keep small human-readable metadata available without extraction.
            raw = pending / 'files'
            for path in sorted(raw.rglob('*')):
                if (path.is_file() and path.stat().st_size <= 1024**2
                        and path.suffix in {'.json', '.md', '.log', '.txt'}
                        and 'compiler-dumps' not in path.relative_to(raw).parts
                        and 'profiles' not in path.relative_to(raw).parts):
                    target = pending / 'summary' / path.relative_to(raw)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(target)
            if raw.exists():
                shutil.rmtree(raw)
        shutil.copyfile(src=manifest_path, dst=pending / 'candidate.json')
        save(path=pending / 'received.json', value={'complete': True, 'manifest_sha256': manifest_hash,
             'files': len(seen), 'bulk_unpacked_retained': not compact, 'unpack_seconds': time.monotonic() - started})
        pending.rename(destination)
    return True
