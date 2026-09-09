"""Create and verify compressed artifact bundles without trusting archive paths."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

from core import json_bytes, safe_path


def verify_bundle(path: Path, manifest: dict, target: Path | None = None) -> None:
    expected = {entry['path']: entry for entry in manifest['files']}
    if len(expected) != len(manifest['files']) or 'bundle-manifest.json' in expected:
        raise ValueError('Duplicate or reserved artifact path')
    seen = set()
    with gzip.open(path, 'rb') as compressed:
        with tarfile.open(fileobj=compressed, mode='r|') as bundle:
            for member in bundle:
                if not member.isfile() or member.name in seen:
                    raise ValueError('Repeated or non-file archive member')
                seen.add(member.name)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError('Archive member has no data')
                with source:
                    if member.name == 'bundle-manifest.json':
                        if member.size != len(json_bytes(manifest)) or source.read() != json_bytes(manifest):
                            raise ValueError('Embedded manifest differs')
                        continue
                    entry = expected.get(member.name)
                    if entry is None or member.size != entry['bytes']:
                        raise ValueError('Unexpected archive member or size')
                    destination = safe_path(root=target or Path('/archive-verification'), relative=member.name)
                    digest = hashlib.sha256()
                    if target:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                    with destination.open('wb') if target else tempfile.TemporaryFile() as output:
                        while chunk := source.read(1024 * 1024):
                            digest.update(chunk)
                            if target:
                                output.write(chunk)
                    if digest.hexdigest() != entry['sha256']:
                        raise ValueError('Archive member checksum mismatch')
        # Consume the gzip footer even when tar has already reached its end marker.
        while compressed.read(1024 * 1024):
            pass
    if seen != set(expected) | {'bundle-manifest.json'}:
        raise ValueError('Archive is missing declared members')


def make_bundle(destination: Path, files: dict[str, Path], metadata: dict) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix='.packing-') as temporary:
        packed = Path(temporary) / 'bundle.tar.gz'
        with tarfile.open(name=packed, mode='w:gz', compresslevel=6) as bundle:
            for name, path in sorted(files.items()):
                safe_path(root=Path('/archive-verification'), relative=name)
                if name == 'bundle-manifest.json' or path.is_symlink() or not path.is_file():
                    raise ValueError(f'Unsupported archive source: {path}')
                # Freeze the observed prefix of logs that may still be growing.
                with tempfile.TemporaryFile() as frozen, path.open('rb') as source:
                    remaining = path.stat().st_size
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError(f'Archive source shrank: {path}')
                        frozen.write(chunk)
                        remaining -= len(chunk)
                    length = frozen.tell()
                    frozen.seek(0)
                    digest = hashlib.file_digest(frozen, 'sha256').hexdigest()
                    frozen.seek(0)
                    entry = tarfile.TarInfo(name=name)
                    entry.size = length
                    entry.mode = path.stat().st_mode & 0o777
                    bundle.addfile(tarinfo=entry, fileobj=frozen)
                    entries.append({'path': name, 'bytes': length, 'sha256': digest})
            manifest = {**metadata, 'files': entries}
            data = json_bytes(manifest)
            entry = tarfile.TarInfo(name='bundle-manifest.json')
            entry.size = len(data)
            bundle.addfile(tarinfo=entry, fileobj=io.BytesIO(data))
        verify_bundle(path=packed, manifest=manifest)
        packed.replace(destination)
    return manifest
