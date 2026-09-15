"""Recover a readable tuning report from existing local results."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
import time

from reporting import KNOBS, canonical, case_id, failure_reasons, render, save

LIMIT = 4 * 1024**2
CASE_FILES = {'outcome.json', 'error.txt', 'config.json', 'timings.json',
              'correctness.json', 'paired-correctness.json', 'profiles.json'}
TOP_FILES = {'results.json', 'baselines.json', 'matrix.json', 'provenance.json',
             'session-error.txt', 'preflight-error.txt', 'artifact-pipeline.json'}
RUN_FILES = {'error.txt', 'collection-error.txt', 'archive-error.txt', 'state.json', 'cleanup.json'}


def safe_name(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and '..' not in path.parts and '\\' not in name


def read_bytes(path: Path) -> bytes | None:
    if not path.is_file() or path.is_symlink():
        return None
    if path.stat().st_size > LIMIT:
        raise ValueError(f'{path.name} exceeds the 4 MiB report-file limit')
    return path.read_bytes()


class RunReport:
    def __init__(self, directory: Path, results: Path):
        self.directory, self.results = directory, results
        self.run_id = directory.name
        self.state: dict = {}
        self.records: dict[str, dict] = {}
        self.auxiliary: dict[str, dict] = {}
        self.expected: list[dict] = []
        self.baselines: list[dict] = []
        self.provenance: dict = {}
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.manifests: dict[str, tuple[Path, dict]] = {}
        self.seen: set[Path] = set()

    def record(self, value: dict, source: str, priority: int = 20) -> None:
        config = value.get('config', {})
        if config.get('variant') not in {'original', 'occupied'}:
            return
        identity = case_id(config=config)
        previous = self.records.get(identity, {})
        # Frozen per-candidate outcomes outrank aggregate snapshots. A completed
        # outcome also replaces an equally ranked partial/missing record.
        newer = priority > previous.get('_priority', -1) or (
            priority == previous.get('_priority') and previous.get('status') in {'missing', 'blocked'}
            and value.get('status') in {'ok', 'failed'})
        if not newer and previous.get('status') in {'ok', 'failed'} and value.get('status') in {'blocked', 'missing'}:
            merged = dict(previous)
        else:
            merged = {**previous, **value} if newer else {**value, **previous}
        if newer and previous.get('status') in {'blocked', 'missing'} and value.get('status') in {'ok', 'failed'}:
            for field in ('reason', 'error', 'failures'):
                if field not in value:
                    merged.pop(field, None)
        merged['_priority'] = max(priority, previous.get('_priority', -1))
        reference = merged.pop('xla_baseline', None)
        if reference and not any(b.get('config', {}).get('tokens') == reference.get('config', {}).get('tokens') for b in self.baselines):
            self.baselines.append(reference)
        merged['sources'] = list(dict.fromkeys([*previous.get('sources', []), source]))
        self.records[identity] = merged

    def accept(self, relative: str, data: bytes, source: str) -> None:
        if relative.startswith('run/'):
            name = relative[4:]
            if name == 'state.json':
                self.state = {**json.loads(data), **self.state}
                for field in ('last_error', 'collection_error', 'cleanup_error'):
                    if self.state.get(field):
                        self.errors.append(f'{field}: {self.state[field]}')
            elif name.endswith('.txt'):
                self.errors.append(data.decode(errors='replace'))
            elif name == 'cleanup.json':
                self.auxiliary['cleanup'] = json.loads(data)
            return
        if relative in TOP_FILES:
            if relative == 'results.json':
                for record in json.loads(data):
                    self.record(value=record, source=source, priority=10)
            elif relative == 'matrix.json':
                self.expected = self.expected or json.loads(data)
            elif relative == 'baselines.json':
                self.baselines = self.baselines or json.loads(data)
            elif relative == 'provenance.json':
                self.provenance = self.provenance or json.loads(data)
            elif relative.endswith('.txt'):
                self.errors.append(data.decode(errors='replace'))
            else:
                self.auxiliary[relative] = json.loads(data)
            return
        parts = PurePosixPath(relative).parts
        if len(parts) != 2:
            return
        case, name = parts
        if name == 'candidate.json':
            manifest = json.loads(data)
            outcome = manifest.get('metadata', {}).get('outcome')
            if outcome:
                self.record(value=outcome, source=source, priority=30)
            else:
                self.auxiliary.setdefault(case, {})['has_outcome'] = False
        elif name == 'outcome.json':
            self.record(value=json.loads(data), source=source, priority=40)
        elif name == 'config.json':
            config = json.loads(data)
            if config.get('variant') in {'original', 'occupied'}:
                identity = case_id(config=config)
                self.records.setdefault(identity, {'config': config, 'status': 'missing', 'sources': []})
        elif case in self.records:
            if name == 'error.txt':
                self.records[case]['error'] = data.decode(errors='replace')
            elif name in CASE_FILES:
                field = name.removesuffix('.json')
                field = {'paired-correctness': 'paired_correctness'}.get(field, field)
                self.records[case].setdefault(field, json.loads(data))
        elif name == 'error.txt':
            self.errors.append(f'{case}: {data.decode(errors="replace")}')

    def read(self, path: Path, relative: str) -> None:
        if path in self.seen:
            return
        self.seen.add(path)
        try:
            data = read_bytes(path=path)
            if data is not None:
                self.accept(relative=relative, data=data, source=str(path))
        except (OSError, ValueError, TypeError, KeyError) as error:
            self.warnings.append(f'{path.name}: {error}')

    def candidate(self, manifest: Path) -> None:
        case = manifest.parent.name
        self.read(path=manifest, relative=case + '/candidate.json')
        try:
            data = read_bytes(path=manifest)
            if data is not None:
                self.manifests.setdefault(case, (manifest.parent, json.loads(data)))
        except (OSError, ValueError) as error:
            self.warnings.append(str(error))
        for folder in (manifest.parent / 'summary', manifest.parent / 'files', manifest.parent):
            for name in ('outcome.json', 'config.json', 'error.txt', 'timings.json',
                         'correctness.json', 'paired-correctness.json', 'profiles.json'):
                self.read(path=folder / name, relative=case + '/' + name)

    def parts(self) -> None:
        for case, (folder, manifest) in self.manifests.items():
            record = self.records.get(case)
            needed = set(CASE_FILES)
            if record:
                needed -= {'config.json', 'outcome.json'}
                for name in CASE_FILES - {'error.txt', 'config.json', 'outcome.json'}:
                    field = name.removesuffix('.json').replace('-', '_')
                    if field in record:
                        needed.discard(name)
                if record.get('status') == 'ok' or record.get('error'):
                    needed.discard('error.txt')
            # Auxiliary artifacts need only their failure text, not their dumps.
            if record is None and not case.startswith(('t',)):
                needed = {'error.txt'}
            grouped: dict[str, dict] = {}
            for entry in manifest.get('files', []):
                name, part = entry['path'], entry['part']
                if name not in needed or entry['bytes'] > LIMIT:
                    continue
                if Path(part).name != part or not part.startswith('part-') or not part.endswith('.tar.gz'):
                    self.warnings.append(f'{case}: invalid part name')
                    continue
                grouped.setdefault(part, {})[name] = entry
            for part, expected in grouped.items():
                path = folder / part
                print(f'READ_DIAGNOSTIC: {case}/{part}', flush=True)
                try:
                    if path.is_symlink() or path.stat().st_size > 50 * 1024**2:
                        raise ValueError('Invalid or oversized candidate part')
                    with tarfile.open(name=path, mode='r|gz') as bundle:
                        for member in bundle:
                            entry = expected.get(member.name)
                            if entry:
                                if not member.isfile() or member.size != entry['bytes']:
                                    raise ValueError('Report member differs from inventory')
                                with bundle.extractfile(member) as stream:
                                    data = stream.read(LIMIT + 1)
                                if hashlib.sha256(data).hexdigest() != entry['sha256']:
                                    raise ValueError('Report checksum mismatch')
                                self.accept(relative=case + '/' + member.name, data=data, source=str(path))
                                del expected[member.name]
                            bundle.members.clear()
                    if expected:
                        raise ValueError('Report member missing from candidate part')
                except (OSError, ValueError, EOFError, tarfile.TarError) as error:
                    self.warnings.append(f'{case}: {error}')

    def archive(self) -> None:
        path = self.results / 'archives' / (self.run_id + '.tar.gz')
        if not path.is_file():
            return
        print(f'READ_SAVED_ARCHIVE: {path.name}; selecting small reports only', flush=True)
        prefix = 'run/collected/files/artifacts/output/'
        try:
            with tarfile.open(name=path, mode='r|gz') as bundle:
                for member in bundle:
                    relative = None
                    if member.name.startswith(prefix):
                        tail = member.name[len(prefix):]
                        parts = PurePosixPath(tail).parts
                        if tail in TOP_FILES:
                            relative = tail
                        elif len(parts) == 3 and parts[0] in {'candidates', '.work'} and parts[2] in CASE_FILES | {'candidate.json'}:
                            relative = '/'.join(parts[1:])
                    elif member.name in {'run/' + name for name in RUN_FILES}:
                        relative = member.name
                    if relative and safe_name(name=relative) and member.isfile() and member.size <= LIMIT:
                        with bundle.extractfile(member) as stream:
                            data = stream.read(LIMIT + 1)
                        try:
                            self.accept(relative=relative, data=data, source=str(path) + ':' + member.name)
                        except (ValueError, KeyError, TypeError) as error:
                            self.warnings.append(f'{member.name}: {error}')
                    bundle.members.clear()
        except (OSError, EOFError, tarfile.TarError) as error:
            self.warnings.append(f'Final archive could not be fully read: {error}')

    def load(self) -> None:
        for name in sorted(RUN_FILES):
            self.read(path=self.directory / name, relative='run/' + name)
        output = self.directory / 'collected/files/artifacts/output'
        for name in sorted(TOP_FILES):
            self.read(path=output / name, relative=name)
        for root in (output / 'candidates', self.results / 'candidates' / self.run_id):
            for manifest in sorted(root.glob('*/candidate.json')):
                self.candidate(manifest=manifest)
        for folder in (output / '.work').glob('*'):
            if folder.is_dir():
                for name in ('outcome.json', 'config.json', 'error.txt', 'timings.json',
                             'correctness.json', 'paired-correctness.json', 'profiles.json'):
                    self.read(path=folder / name, relative=folder.name + '/' + name)
        unknown_failure = any(r.get('status') == 'failed' and not any(r.get(k) for k in ('error', 'correctness', 'profiles'))
                              for r in self.records.values())
        if (not self.records or not self.state or not self.expected or not self.baselines
                or not self.provenance or unknown_failure or len(self.records) < len(self.expected)):
            self.archive()
        self.parts()
        for config in self.expected:
            identity = case_id(config=config)
            self.records.setdefault(identity, {'config': config, 'status': 'missing',
                'reason': 'No collected outcome for this planned configuration'})
        if not self.records:
            self.errors.append('No candidate outcomes were collected. This is a diagnostic report, not a successful tuning result.')
        if not self.state.get('artifacts_verified'):
            self.warnings.append('Full artifact collection is unverified in saved state; original data is retained.')

    def write(self, output: Path) -> Path:
        name = self.state.get('name') or re.sub(r'-[a-f0-9]{24}$', '', self.run_id)
        label = re.sub(r'[^A-Za-z0-9_.-]+', '-', name).strip('.-') or 'kernel-occupancy'
        destination = output / label
        for index in output.glob('*/run.json'):
            if json.loads(index.read_text()).get('run_id') == self.run_id:
                destination = index.parent
                break
        previous = destination / 'run.json'
        if previous.is_file() and json.loads(previous.read_text()).get('run_id') != self.run_id:
            source = self.directory.parent if self.directory.parent.exists() else self.results
            stamp = datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).strftime('%Y%m%d-%H%M%S')
            destination = output / (label + '-' + stamp)
            index = 2
            while destination.exists():
                destination = output / (label + '-' + stamp + '-' + str(index))
                index += 1
        records = [self.records[key] for key in sorted(self.records)]
        complete = bool(self.expected) and len(records) == len(self.expected) and all(r.get('status') in {'ok', 'failed'} for r in records)
        render(output=destination, records=records, baselines=self.baselines, name=name,
               complete=complete, run_errors=list(dict.fromkeys(self.errors + self.warnings)))
        state_keys = ('job_id', 'artifacts_verified', 'resources_cleaned', 'collection_error', 'cleanup_error', 'phase')
        save(path=destination / 'run.json', value={'name': name, 'run_id': self.run_id,
            'expected_candidates': len(self.expected), 'collected_candidates': len(records),
            'complete': complete, 'provenance': self.provenance,
            'collection': {key: self.state.get(key) for key in state_keys},
            'warnings': list(dict.fromkeys(self.warnings)), 'errors': list(dict.fromkeys(self.errors)),
            'reference_file': 'baselines.json' if self.baselines else None, 'source': str(self.directory)})
        print((destination / 'SUMMARY.md').read_text(), flush=True)
        print(f'RESULTS: {destination / "SUMMARY.md"}', flush=True)
        print('Stage the small report:\ngit add -- ' + shlex.quote(str(destination)), flush=True)
        return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'reports')
    parser.add_argument('--run', type=Path, help='Saved job directory; supplied by the workflow automatically')
    args = parser.parse_args()
    results = args.results.resolve()
    if args.run:
        directories = [args.run.resolve()]
        results = directories[0].parents[1]
    else:
        campaigns = sorted(results.glob('*/campaign.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if campaigns:
            source = campaigns[0]
            jobs = json.loads(source.read_text())['jobs']
            if any(not isinstance(job, str) or Path(job).name != job or job in {'.', '..'} for job in jobs):
                parser.error('Invalid job name in saved campaign')
            directories = [source.parent / job for job in jobs]
        else:
            archives = sorted((results / 'archives').glob('*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not archives:
                parser.error('No saved run or final archive was found under ' + str(results))
            run_id = json.loads(archives[0].read_text())['run_id']
            if not safe_name(name=run_id) or Path(run_id).name != run_id:
                parser.error('Invalid saved run identifier')
            directories = [results / 'archive-only' / run_id]
    failed = False
    for directory in directories:
        report = RunReport(directory=directory, results=results)
        report.load()
        report.write(output=args.output.resolve())
        failed |= not bool(report.records)
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
