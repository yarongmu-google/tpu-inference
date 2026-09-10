"""Collect historical client logs and compressed comparison results into one ledger."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import lzma
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

HISTORY = [
    ('baseline', '20260902_163515', 'fp8_gmm_ep_20260902_162423.log', 'repo'),
    ('baseline', '20260903_135542', 'fp8_gmm_ep_20260903_134242.log', 'repo'),
    ('4c', '20260903_105407', 'fp8_v2_tp_128s_20260903_103945.log', 'repo'),
    ('4e', '20260903_122921', 'fp8_v2_tp_104s_20260903_120017.log', 'repo'),
    ('4g', '20260903_151322', 'fp8_v2_tp_64s_riders_20260903_145852.log', 'repo'),
    ('4g', '20260903_155530', 'fp8_v2_tp_64s_riders_20260903_145852.log', 'upstream'),
    ('4h', '20260903_165502', 'fp8_v2_tp_104s_p2buckets_20260903_163230.log.xz', 'repo'),
    ('4h', '20260903_181805', 'fp8_v2_tp_104s_p2buckets_20260903_175724.log', 'repo'),
    ('4i', '20260903_191522', 'fp8_v2_tp_104s_singlespeed_20260903_190544.log', 'repo'),
]


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def result_commit(repo: Path, path: str) -> str:
    return git(repo, 'log', '-1', '--format=%H', '--', path) or 'uncommitted'


def text_file(path: Path) -> str:
    return lzma.open(path, 'rt').read() if path.suffix == '.xz' else path.read_text()


def metric(text: str, label: str) -> float:
    match = re.search(r'^' + re.escape(label) + r':\s*([0-9.]+)', text, flags=re.MULTILINE | re.IGNORECASE)
    if match is None:
        raise ValueError(f'Missing metric: {label}')
    return float(match.group(1))


def historical(repo: Path) -> tuple[list[dict], list[str]]:
    rows, commands = [], []
    for config, stamp, server_name, sampler in HISTORY:
        client = f'tmp/vllm_logs/client_fp8_mix_{stamp}.log.xz'
        server = 'tmp/vllm_logs/' + server_name
        text = text_file(repo / client)
        server_text = text_file(repo / server)
        marker = re.search(r'^CFG label=(\S+) commit=(\S+)', server_text)
        if marker is None:
            raise ValueError(f'No source commit in {server}')
        label, source_commit = marker.groups()
        revision = git(repo, 'rev-parse', source_commit + '^{commit}')
        source = git(repo, 'show', revision + ':scripts/vllm/benchmarking/bench_throughput_qwen_server.sh')
        candidates = [line for line in source.splitlines() if line.startswith('L=tmp/vllm_logs/' + label + '_$(')]
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous historical server command: {label}')
        namespace = next(line for line in text.splitlines() if line.startswith('Namespace('))
        completed = int(metric(text, 'Successful requests'))
        num_prompts = int(re.search(r'num_prompts=(\d+)', namespace).group(1))
        row = {'id': config + '-' + stamp, 'config': config, 'input_length': 1024, 'output_length': 8192,
               'sampler': '[0.2X,X]' if sampler == 'repo' else '[0.8X,X]',
               'client_concurrency': int(re.search(r'max_concurrency=(\d+)', namespace).group(1)),
               'completed': completed, 'num_prompts': num_prompts,
               'status': 'complete' if completed == num_prompts else 'partial',
               'output_tok_s': metric(text, 'Output token throughput (tok/s)'),
               'total_tok_s': metric(text, 'Total token throughput (tok/s)'),
               'mean_tpot_ms': metric(text, 'Mean TPOT (ms)'),
               'input_tokens': int(metric(text, 'Total input tokens')),
               'output_tokens': int(metric(text, 'Total generated tokens')),
               'source_commit': revision, 'result_commit': result_commit(repo=repo, path=client),
               'evidence': client, 'server_log': server}
        rows.append(row)
        commands += ['## ' + row['id'], '', 'Server line at the commit recorded in its log:', '',
                     '```bash', candidates[0], '```', '',
                     'Client arguments recorded in the log (original executable/script path was not recorded):',
                     '', '```text', namespace, '```', '']
    return rows, commands


def archived(repo: Path) -> tuple[list[dict], list[str]]:
    sys.path.insert(0, str(repo / 'tmp/workflow'))
    from artifacts import verify_bundle
    rows, commands = [], []
    root = repo / 'tmp/serving-comparison/results/archives'
    for path in sorted(root.glob('*.tar.gz')):
        index = json.loads(path.with_suffix('').with_suffix('.json').read_text())
        if path.stat().st_size != index['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != index['sha256']:
            raise ValueError('Archive checksum mismatch: ' + path.name)
        with tarfile.open(path) as bundle:
            manifest = json.load(bundle.extractfile('bundle-manifest.json'))
        verify_bundle(path=path, manifest=manifest)
        with tarfile.open(path) as bundle:
            names = set(bundle.getnames())
            def read(name: str) -> dict | list:
                return json.load(bundle.extractfile(name))
            state = read('run/state.json')
            prefix = 'run/collected/files/artifacts/output/'
            summary = prefix + 'summary.json'
            commit = result_commit(repo=repo, path=path.relative_to(repo).as_posix())
            if summary not in names:
                rows.append({'id': state['run_id'], 'config': 'infrastructure', 'status': 'failed-before-results',
                             'source_commit': None, 'result_commit': commit, 'evidence': path.relative_to(repo).as_posix(),
                             'job_id': state.get('job_id'), 'resources_cleaned': state.get('resources_cleaned', False)})
                continue
            metadata = read(prefix + 'metadata/comparison.json')
            image = read(prefix + 'metadata/image-source.json')
            for result in read(summary):
                name = result.get('case', result['config'])
                recorded = read(prefix + name + '/commands.json')
                workload = recorded['workload']
                row = {'id': state['run_id'] + '/' + name, 'config': result['config'],
                       'input_length': workload['input_length'], 'output_length': workload['output_length'],
                       'sampler': f"[{workload['range_ratio']}X,X]", 'client_concurrency': workload['concurrency'],
                       'num_prompts': workload['num_prompts'], 'status': result['status'],
                       'source_commit': image['source_revisions']['tpu_inference'],
                       'vllm_commit': image['source_revisions']['vllm'], 'client_commit': metadata['client_revision'],
                       'result_commit': commit, 'evidence': path.relative_to(repo).as_posix(),
                       'job_id': state['job_id'], 'resources_cleaned': state.get('resources_cleaned', False)}
                if result['status'] == 'complete':
                    client = read(prefix + name + '/client.json')
                    row.update(completed=client['completed'], output_tok_s=result['output_throughput'],
                               total_tok_s=result['total_token_throughput'], mean_tpot_ms=result['mean_tpot_ms'],
                               input_tokens=client['total_input_tokens'], output_tokens=client['total_output_tokens'])
                server_text = bundle.extractfile(prefix + name + '/server.log').read().decode(errors='replace')
                row.update(refusal_log_entries=server_text.count('REFUSED gate='),
                           has_preemption_text='preempt' in server_text.lower())
                stats = prefix + name + '/server-stats.csv'
                if stats in names:
                    data = list(csv.DictReader(io.StringIO(bundle.extractfile(stats).read().decode())))
                    row['peak_kv_percent'] = max(float(item['KV Cache Usage (%)']) for item in data)
                rows.append(row)
                client_argv = ['python3', *recorded['client'][1:]]
                commands += ['## ' + row['id'], '', 'Recorded server command:', '', '```bash',
                             shlex.join(recorded['server']), '```', '',
                             'Recorded client command (environment-specific interpreter path displayed as python3):',
                             '', '```bash', shlex.join(client_argv), '```', '']
    return rows, commands


def render(rows: list[dict]) -> str:
    lines = ['# Serving comparison results', '',
             'Source commit identifies runtime code; result commit identifies the latest Git commit that added or updated the result artifact.',
             'Historical source commits come from server log markers and do not certify a clean historical working tree.',
             'Partial runs are retained for diagnosis and excluded from performance claims. Missing metrics mean no completed benchmark.',
             'Eight attention-DP ranks run on eight TPU cores across four physical chips. Throughput below is per full TPU slice.', '',
             '| Run | I/O | Client C | Sampler | Completed | Output tok/s | Total tok/s | TPOT ms | Source commit | Result commit |',
             '|---|---|---:|---|---|---:|---:|---:|---|---|']
    def number(row: dict, key: str) -> str:
        return f'{row[key]:.2f}' if key in row else '-'
    for row in rows:
        shape = f"{row['input_length']//1024}k/{row['output_length']//1024}k" if 'input_length' in row else '-'
        completed = f"{row.get('completed', 0)}/{row['num_prompts']} ({row['status']})" if 'num_prompts' in row else row['status']
        lines.append(f"| {row['id']} | {shape} | {row.get('client_concurrency', '-')} | {row.get('sampler', '-')} | {completed} | "
                     f"{number(row, 'output_tok_s')} | {number(row, 'total_tok_s')} | {number(row, 'mean_tpot_ms')} | "
                     f"{(row.get('source_commit') or '-')[:10]} | {row['result_commit'][:10]} |")
    lines += ['', 'Full provenance, log locations, client/vLLM revisions and memory observations: [results.json](results.json).',
              'Recorded server commands and client arguments: [COMMANDS.md](COMMANDS.md).',
              'Interpretation and next experiment: [ANALYSIS.md](ANALYSIS.md).', '']
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent / 'summary')
    args = parser.parse_args()
    repo = args.repo.resolve()
    old_rows, old_commands = historical(repo=repo)
    new_rows, new_commands = archived(repo=repo)
    rows = old_rows + new_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'RESULTS.md').write_text(render(rows=rows))
    (args.output_dir / 'COMMANDS.md').write_text('# Command appendix\n\n' + '\n'.join(old_commands + new_commands))
    (args.output_dir / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(f'Collected {len(rows)} records into {args.output_dir}')


if __name__ == '__main__':
    main()
