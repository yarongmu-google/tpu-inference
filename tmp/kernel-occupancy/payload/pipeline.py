"""Queue frozen candidate artifacts while the next TPU worker runs."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback

import diagnostics


class Pipeline:
    def __init__(self, output: Path):
        runtime = os.environ.get('WORKFLOW_RUNTIME_DIR')
        if not runtime:
            raise RuntimeError('Incremental artifacts require the shared workflow runtime')
        sys.path.insert(0, runtime)
        from streams import seal
        self.seal = seal
        self.output = output
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.tasks = []
        self.marker = output / '.incomplete-artifacts'
        self.marker.write_text('Private work is retained until every candidate is packaged.\n')

    def submit(self, directory: Path) -> None:
        if not directory.exists():
            return
        self.tasks.append(self.pool.submit(self.package, directory=directory))
        print(f'ARTIFACTS_QUEUED: {directory.name}; TPU can continue', flush=True)

    def package(self, directory: Path) -> None:
        started = time.monotonic()
        try:
            excluded = {}
            root = directory / 'compiler-dumps'
            window_path = directory / 'compile-window.json'
            if root.exists() and window_path.exists():
                inventory = diagnostics.snapshot(root=root)
                window = json.loads(window_path.read_text())
                selected = set(window.get('files', [name for name, state in inventory.items()
                                                   if state != window['before'].get(name)]))
                if not selected <= set(inventory):
                    raise ValueError('Selected compiler files are missing')
                excluded = {name: state[0] for name, state in inventory.items() if name not in selected}
                diagnostics.save(path=directory / 'compiler-selection.json', value={
                    'selected_files': sorted(selected), 'excluded_files': excluded,
                    'selection': 'Every pass in the candidate compile window; other compilations omitted'})
                # Only generated, explicitly unselected compiler output is discarded.
                for name in excluded:
                    (root / name).unlink()
            fragments = {}
            for path in sorted(directory.rglob('*')):
                if not path.is_file() or path.is_symlink() or path.stat().st_size <= 32 * 1024**2:
                    continue
                relative = str(path.relative_to(directory))
                destination = directory / '.fragments' / str(len(fragments))
                destination.mkdir(parents=True)
                digest, chunks = hashlib.sha256(), []
                with path.open('rb') as original:
                    for index in range((path.stat().st_size + 32 * 1024**2 - 1) // (32 * 1024**2)):
                        block = original.read(32 * 1024**2)
                        digest.update(block)
                        chunk = destination / f'chunk-{index:04d}'
                        chunk.write_bytes(block)
                        chunks.append({'path': str(chunk.relative_to(directory)), 'bytes': len(block)})
                fragments[relative] = {'sha256': digest.hexdigest(), 'bytes': path.stat().st_size, 'chunks': chunks}
            if fragments:
                diagnostics.save(path=directory / 'file-fragments.json', value=fragments)
            names = [str(path.relative_to(directory)) for path in sorted(directory.rglob('*'))
                     if path.is_file() and str(path.relative_to(directory)) not in fragments]
            outcome = directory / 'outcome.json'
            record = self.seal(source=directory,
                destination=self.output / 'candidates' / directory.name, names=names,
                metadata={'compact_after_receive': True, 'outcome': json.loads(outcome.read_text()) if outcome.exists() else None,
                          'excluded_compiler_bytes': sum(excluded.values())})
            for child in ('compiler-dumps', 'profiles', '.fragments'):
                import shutil
                target = directory / child
                if target.is_dir():
                    shutil.rmtree(target)
            for path in directory.glob('expected-*'):
                if path.is_dir():
                    shutil.rmtree(path)
            for name in fragments:
                path = directory / name
                if path.exists():
                    path.unlink()
            print(f'ARTIFACTS_READY: {directory.name}; {len(record["files"])} files; '
                  f'{sum(part["bytes"] for part in record["parts"])} bytes; '
                  f'{time.monotonic() - started:.1f}s packaging', flush=True)
        except BaseException:
            errors = self.output / 'artifact-errors'
            errors.mkdir(exist_ok=True)
            (errors / f'{directory.name}.txt').write_text(traceback.format_exc())
            print(f'ARTIFACTS_FAILED: {directory.name}; private work retained for final recovery', flush=True)
            raise

    def finish(self) -> bool:
        pending = set(self.tasks)
        while pending:
            _, pending = wait(pending, timeout=10)
            if pending:
                print(f'ARTIFACTS_DRAINING: {len(pending)} candidates remain; TPU work finished', flush=True)
        self.pool.shutdown(wait=True)
        failed = any(task.exception() is not None for task in self.tasks)
        if not failed:
            self.marker.unlink()
        diagnostics.save(path=self.output / 'artifact-pipeline.json', value={
            'complete': not failed, 'candidates': len(self.tasks),
            'failed': sum(task.exception() is not None for task in self.tasks)})
        return not failed
