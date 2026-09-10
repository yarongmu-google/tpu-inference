"""Run described workloads against frozen server commands with one client checkout."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
from urllib.error import URLError
from urllib.request import urlopen

MODEL = 'Qwen/Qwen3.5-397B-A17B-FP8'
CLIENT_URL = 'https://github.com/kimbochen/bench_serving.git'
CLIENT_DIRECTORY = Path('/tmp/serving-comparison-client')
CONFIGS = {'baseline': 'fp8_gmm_ep', '4g': 'fp8_v2_tp_64s_riders',
           '4i': 'fp8_v2_tp_104s_singlespeed'}
WORKLOAD = {'input_length': 1024, 'output_length': 8192, 'concurrency': 512,
            'num_prompts': 2048, 'range_ratio': 0.8, 'seed': 0, 'num_warmups': 0,
            'use_chat_template': False, 'physical_chips': 4}


def save(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_plan(path: Path) -> tuple[list[dict], str]:
    plan = json.loads(path.read_text())
    if set(plan) != {'client_revision', 'concurrency', 'num_prompts', 'range_ratio', 'seed', 'num_warmups', 'cases'}:
        raise ValueError('Unexpected comparison plan keys')
    if not re.fullmatch('[a-f0-9]{40}', plan['client_revision']):
        raise ValueError('Pin the benchmark client to a full commit')
    for key in ('concurrency', 'num_prompts'):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f'Invalid {key}')
    if (type(plan['range_ratio']) not in (int, float) or not 0 < plan['range_ratio'] <= 1
            or type(plan['seed']) is not int or plan['num_warmups'] != 0):
        raise ValueError('Invalid sampling settings; this comparison uses zero warmups')
    if not isinstance(plan['cases'], list) or not plan['cases']:
        raise ValueError('At least one case is required')
    cases = []
    for case in plan['cases']:
        if set(case) != {'name', 'config', 'input_length', 'output_length'}:
            raise ValueError('Unexpected case keys')
        if (not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*', case['name'])
                or case['name'] in {row['name'] for row in cases} or case['config'] not in CONFIGS):
            raise ValueError('Invalid or repeated case name/configuration')
        if any(type(case[k]) is not int or case[k] <= 0 for k in ('input_length', 'output_length')):
            raise ValueError('Invalid sequence length')
        if case['input_length'] + case['output_length'] > 9216:
            raise ValueError('Workload exceeds the frozen server context limit')
        workload = WORKLOAD | {k: plan[k] for k in ('concurrency', 'num_prompts', 'range_ratio', 'seed', 'num_warmups')}
        workload.update({k: case[k] for k in ('input_length', 'output_length')})
        cases.append({'name': case['name'], 'config': case['config'], 'workload': workload})
    return cases, plan['client_revision']


def log_tail(path: Path, limit: int = 8192) -> None:
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            print(f'LOG_TAIL: {path}\n' + stream.read().decode(errors='replace'), file=sys.stderr, flush=True)


def server_commands(source: Path) -> dict[str, list[str]]:
    commands = {}
    for label, prefix in CONFIGS.items():
        lines = [line for line in source.read_text().splitlines()
                 if line.startswith('L=tmp/vllm_logs/' + prefix + '_$(')]
        if len(lines) != 1:
            raise ValueError(f'Expected one server command for {label}')
        match = re.search(r'; ([A-Z][A-Z0-9_]*=.*?) 2>&1 \| tee -a "\$L";', lines[0])
        if match is None:
            raise ValueError(f'Cannot extract environment and command for {label}')
        argv = shlex.split(match[1])
        index = argv.index('vllm')
        if argv[index:index + 3] != ['vllm', 'serve', MODEL]:
            raise ValueError(f'Unexpected server executable or model for {label}')
        if any(not re.fullmatch('[A-Z][A-Z0-9_]*=.*', item, flags=re.DOTALL) for item in argv[:index]):
            raise ValueError(f'Unexpected environment assignment for {label}')
        commands[label] = ['env', *argv]
    return commands


def clean_environment(commands: dict[str, list[str]]) -> dict[str, str]:
    # Do not let one configuration inherit kernel flags from the image or caller.
    names = {'USE_MOE_TP_DECODE_KERNEL', 'USE_MOE_EP_KERNEL', 'MOE_TP_DECODE_MAX_TOKENS',
             'MOE_ROUTE_PADDING_TO_EXPERT0', 'MIN_TOKEN_BUCKET', 'DP_SCHED_BATCH_PREFILL',
             'SLICE_ROPE_CACHE', 'RAGGED_GATED_DELTA_RULE_IMPL', 'MOE_REQUANTIZE_BLOCK_SIZE',
             'DISABLE_WEIGHT_REQUANTIZATION'}
    for argv in commands.values():
        names.update(item.split('=', 1)[0] for item in argv[1:argv.index('vllm')])
    return {key: value for key, value in os.environ.items() if key not in names}


def stop(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)


def run_command(argv: list[str], log: Path, timeout: int, env: dict | None = None) -> None:
    print('+ ' + shlex.join(argv), flush=True)
    with log.open('w') as stream:
        stream.write('+ ' + shlex.join(argv) + '\n')
        stream.flush()
        process = subprocess.Popen(args=argv, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f'Command exceeded {timeout}s; see {log}')
                try:
                    process.wait(timeout=min(30, remaining))
                except subprocess.TimeoutExpired:
                    print(f'RUNNING: {log.name}; log: {log}', flush=True)
            if process.returncode:
                raise RuntimeError(f'Command exited {process.returncode}; see {log}')
        except Exception:
            stream.flush()
            with log.open('rb') as saved:
                saved.seek(max(0, log.stat().st_size - 4096))
                print(saved.read().decode(errors='replace'), file=sys.stderr, flush=True)
            raise
        finally:
            stop(process=process)


def healthy() -> bool:
    try:
        with urlopen(url='http://127.0.0.1:8000/health', timeout=2) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


def wait_ready(process: subprocess.Popen, timeout: int = 5400) -> None:
    deadline = time.monotonic() + timeout
    report = 0.0
    while process.poll() is None:
        if healthy():
            return
        now = time.monotonic()
        if now >= deadline:
            raise TimeoutError('Server startup exceeded its deadline')
        if now >= report:
            print('STARTING_SERVER: waiting for /health; server.log is being collected', flush=True)
            report = now + 30
        time.sleep(min(5, max(0, deadline - now)))
    raise RuntimeError(f'Server exited during startup: {process.returncode}')


def client_command(client: Path, output: Path, workload: dict | None = None, tokenizer: str | None = None) -> list[str]:
    workload = WORKLOAD if workload is None else workload
    argv = [sys.executable, str(client / 'benchmark_serving.py'), '--model', MODEL,
        '--backend', 'vllm', '--host', '127.0.0.1', '--port', '8000', '--dataset-name', 'random',
        '--random-input-len', str(workload['input_length']), '--random-output-len', str(workload['output_length']),
        '--random-range-ratio', str(workload['range_ratio']), '--random-prefix-len', '0',
        '--max-concurrency', str(workload['concurrency']), '--num-prompts', str(workload['num_prompts']),
        '--request-rate', 'inf', '--seed', str(workload['seed']), '--num-warmups', str(workload['num_warmups']), '--ignore-eos',
        '--percentile-metrics', 'ttft,tpot,itl,e2el', '--save-result',
        '--result-dir', str(output), '--result-filename', 'client.json']
    if tokenizer:
        argv += ['--tokenizer', tokenizer]
    return argv


def read_result(path: Path, workload: dict | None = None) -> dict:
    workload = WORKLOAD if workload is None else workload
    value = json.loads(path.read_text())
    if (value.get('completed') != workload['num_prompts'] or value.get('num_prompts') != workload['num_prompts']
            or value.get('max_concurrency') != workload['concurrency'] or value.get('model_id') != MODEL):
        raise ValueError('Client result is incomplete or belongs to a different workload')
    keys = ['duration', 'total_input_tokens', 'total_output_tokens', 'output_throughput',
            'total_token_throughput', 'mean_tpot_ms', 'mean_ttft_ms']
    for key in keys:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (float, int)) or not math.isfinite(item) or item <= 0:
            raise ValueError(f'Invalid client metric: {key}')
    for key, expected in [('output_throughput', value['total_output_tokens'] / value['duration']),
                          ('total_token_throughput', (value['total_input_tokens'] + value['total_output_tokens']) / value['duration'])]:
        if not math.isclose(value[key], expected, rel_tol=1e-5):
            raise ValueError(f'Inconsistent client metric: {key}')
    return {key: value[key] for key in keys} | {'total_per_chip': value['total_token_throughput'] / 4}


def compare(commands: dict[str, list[str]], client: Path, output: Path, env: dict[str, str],
            cases: list[dict] | None = None, tokenizer: str | None = None) -> int:
    cases = cases if cases is not None else [{'name': name, 'config': name, 'workload': WORKLOAD} for name in commands]
    rows = []
    for case in cases:
        label, workload = case['name'], case['workload']
        argv = commands[case['config']]
        folder = output / label
        folder.mkdir()
        row = {'case': label, 'config': case['config'], 'input_length': workload['input_length'],
               'output_length': workload['output_length'], 'concurrency': workload['concurrency'], 'status': 'failed'}
        process = None
        started = time.time()
        with (folder / 'server.log').open('w') as log:
            try:
                if healthy():
                    raise RuntimeError('Port 8000 already serves a model; refusing a contaminated comparison')
                save(path=folder / 'commands.json', value={'server': argv,
                    'client': client_command(client=client, output=folder, workload=workload, tokenizer=tokenizer), 'workload': workload})
                print(f'STARTING_SERVER: {label}\n+ {shlex.join(argv)}', flush=True)
                log.write('+ ' + shlex.join(argv) + '\n')
                log.flush()
                process = subprocess.Popen(args=argv, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(process=process)
                print(f"BENCHMARKING: {label}; {workload['num_prompts']} requests at concurrency {workload['concurrency']}", flush=True)
                run_command(argv=client_command(client=client, output=folder, workload=workload, tokenizer=tokenizer), log=folder / 'client.log',
                            timeout=7200, env=env)
                row.update(read_result(path=folder / 'client.json', workload=workload))
                row['status'] = 'complete'
            except Exception:
                detail = traceback.format_exc()
                (folder / 'error.txt').write_text(detail)
                print(detail, file=sys.stderr, flush=True)
                log.flush()
                log_tail(path=folder / 'server.log')
            finally:
                if process is not None:
                    stop(process=process)
                stats = Path('/opt/tpu-inference/tmp/vllm_server_stats.csv')
                if process is not None and stats.is_file() and stats.stat().st_mtime >= started:
                    shutil.copyfile(src=stats, dst=folder / 'server-stats.csv')
                rows.append(row)
                save(path=output / 'summary.json', value=rows)
        if healthy():
            raise RuntimeError('Server remained alive after shutdown; refusing the next configuration')
    completed = [row for row in rows if row['status'] == 'complete']
    for shape in {(row['input_length'], row['output_length']) for row in completed}:
        pairs = {(row['total_input_tokens'], row['total_output_tokens']) for row in completed
                 if (row['input_length'], row['output_length']) == shape}
        if len(pairs) > 1:
            raise ValueError(f'Completed runs processed different token counts for {shape}; inspect the client logs')
    columns = ['case', 'config', 'input_length', 'output_length', 'concurrency', 'status', 'output_throughput', 'total_token_throughput',
               'total_per_chip', 'mean_tpot_ms', 'mean_ttft_ms']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0 if len(completed) == len(cases) else 1


def main() -> int:
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'metadata'
    metadata.mkdir()
    try:
        source = Path(os.environ['INPUT_SERVER_COMMANDS_DIR']) / 'bench_throughput_qwen_server.sh'
        commands = server_commands(source=source)
        cases, pinned_client = load_plan(path=Path(os.environ['COMPARISON_PLAN'])) if os.environ.get('COMPARISON_PLAN') else (None, None)
        if cases:
            commands = {key: argv for key, argv in commands.items() if key in {case['config'] for case in cases}}
        env = clean_environment(commands=commands)
        shutil.copyfile(src=source, dst=metadata / 'server-source.sh')
        provenance = Path('/opt/jobset/image-source.json')
        if provenance.is_file():
            shutil.copyfile(src=provenance, dst=metadata / 'image-source.json')
        client = CLIENT_DIRECTORY
        print('CLONING_CLIENT: one fresh checkout shared by all cases', flush=True)
        run_command(argv=['git', 'clone', '--depth', '1', CLIENT_URL, str(client)],
                    log=metadata / 'clone.log', timeout=300)
        if pinned_client:
            for args in (['fetch', '--depth', '1', 'origin', pinned_client], ['checkout', '--detach', 'FETCH_HEAD']):
                run_command(argv=['git', '-C', str(client), *args], log=metadata / ('client-' + args[0] + '.log'), timeout=300)
        revision = subprocess.check_output(args=['git', '-C', str(client), 'rev-parse', 'HEAD'], text=True).strip()
        if pinned_client and revision != pinned_client:
            raise ValueError('Cloned client revision differs from the plan')
        archive = metadata / 'client-source'
        archive.mkdir()
        hashes = {}
        for path in sorted(client.iterdir()):
            if path.is_file() and (path.suffix == '.py' or path.name == 'LICENSE'):
                shutil.copyfile(src=path, dst=archive / path.name)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(path=metadata / 'comparison.json', value={'client_url': CLIENT_URL,
            'client_revision': revision, 'client_files': hashes, 'workload': WORKLOAD if cases is None else None,
            'cases': cases, 'run_id': os.environ.get('RUN_ID'),
            'expected_sampling': '[floor(range_ratio * X), X]', 'server_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        text = (client / 'benchmark_serving.py').read_text()
        if (not re.search(r'lower\s*=\s*int\(seq_len\s*\*\s*range_ratio\)', text)
                or not re.search(r'upper\s*=\s*seq_len\s*\n', text)):
            raise ValueError('Upstream sampler changed; inspect the saved client before benchmarking')
        run_command(argv=[sys.executable, str(client / 'benchmark_serving.py'), '--help'],
                    log=metadata / 'client-help.log', timeout=120, env=env)
        preflight = str(Path(__file__).with_name('preflight.py'))
        print('PREFLIGHT: validating client, server arguments and TPU execution before model loading', flush=True)
        tokenizer = None
        if cases:
            print('PREPARING_CHECKPOINT: verify scratch capacity and download once before all servers', flush=True)
            checkpoint = metadata / 'checkpoint.json'
            run_command(argv=[sys.executable, preflight, 'checkpoint', MODEL, str(checkpoint)],
                        log=metadata / 'checkpoint.log', timeout=7200, env=env)
            # Keep the extracted server command unchanged, including its model reference.
        checked = cases if cases is not None else [{'name': 'default', 'workload': WORKLOAD}]
        for case in checked:
            run_command(argv=[sys.executable, preflight, 'client',
                             *client_command(client=client, output=output, workload=case['workload'], tokenizer=tokenizer)[1:]],
                        log=metadata / (case['name'] + '-client-preflight.log'), timeout=300, env=env)
        for label, argv in commands.items():
            index = argv.index('vllm')
            run_command(argv=['env', *argv[1:index], sys.executable, preflight, 'server', *argv[index + 2:]],
                        log=metadata / (label + '-preflight.log'), timeout=180, env=env)
            run_command(argv=['env', *argv[1:index], sys.executable, preflight, 'hardware', '8'],
                        log=metadata / (label + '-hardware-preflight.log'), timeout=300, env=env)
        return compare(commands=commands, client=client, output=output, env=env, cases=cases, tokenizer=tokenizer)
    except Exception:
        detail = traceback.format_exc()
        (output / 'error.txt').write_text(detail)
        print(detail, file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    def interrupted(number: int, frame: object) -> None:
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    raise SystemExit(main())
