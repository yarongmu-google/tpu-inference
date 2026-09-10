"""Run the two committed baseline command pairs and retain every outcome."""
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
CLIENT_DIRECTORY = Path('/run-scratch/InferenceX')
IMAGE_METADATA = Path('/opt/jobset/image-source.json')
CLIENT_URL = 'https://github.com/SemiAnalysisAI/InferenceX.git'
CLIENT_REVISION = 'd089a9138c53d16c6388e4251a078fee8ca7bea6'
WORKLOAD = {'concurrency': 256, 'num_prompts': 2560}
CASES = {'8k/1k': (8192, 1024), '1k/1k': (1024, 1024)}

def save(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def log_tail(path: Path, limit: int = 8192) -> None:
    if path.is_file():
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            print(f'LOG_TAIL: {path}\n' + stream.read().decode(errors='replace'), file=sys.stderr, flush=True)

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

def wait_ready(process: subprocess.Popen, timeout: int = 10800) -> None:
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


def parse_commands(path: Path, executable: str) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    label = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('#'):
            label = line[1:].strip()
            if label not in CASES:
                raise ValueError(f'Unexpected command label in {path}: {label}')
            continue
        if label not in CASES or label in commands:
            raise ValueError(f'Missing or repeated command label in {path}')
        argv = shlex.split(line)
        if executable == 'vllm':
            index = argv.index('vllm')
            if argv[index:index + 3] != ['vllm', 'serve', MODEL]:
                raise ValueError('Unexpected baseline server executable or model')
            if any(not re.fullmatch('[A-Z][A-Z0-9_]*=.*', item, re.DOTALL) for item in argv[:index]):
                raise ValueError('Unexpected server environment assignment')
            argv = ['env', *argv]
        elif argv[:2] != ['python', './bench_serving/benchmark_serving.py']:
            raise ValueError('Unexpected baseline client command')
        commands[label] = argv
    if list(commands) != list(CASES):
        raise ValueError('Expected the committed 8k/1k and 1k/1k command pairs')
    return commands


def option(argv: list[str], key: str) -> str:
    values = [item.split('=', 1)[1] if item.startswith(key + '=') else argv[i + 1]
              for i, item in enumerate(argv) if item == key or item.startswith(key + '=')]
    if len(values) != 1:
        raise ValueError(f'Expected exactly one {key}')
    return values[0]


def client_command(original: list[str], script: Path, output: Path, tokenizer: str) -> list[str]:
    argv = [sys.executable, str(script), *original[2:]]
    index = argv.index('--result-dir')
    argv[index + 1] = str(output)
    return [*argv, '--result-filename', 'client.json', '--tokenizer', tokenizer]


def run_cases(servers: dict[str, list[str]], clients: dict[str, list[str]],
              script: Path, output: Path, env: dict[str, str], tokenizer: str) -> int:
    rows = []
    for label, (isl, osl) in CASES.items():
        folder = output / label.replace('/', '-')
        folder.mkdir()
        row = {'case': label, 'input_length': isl, 'output_length': osl,
               'concurrency': 256, 'status': 'failed'}
        argv = servers[label]
        client = client_command(original=clients[label], script=script, output=folder, tokenizer=tokenizer)
        save(path=folder / 'commands.json', value={'server': argv, 'client': client})
        process = None
        with (folder / 'server.log').open('w') as log:
            try:
                if healthy():
                    raise RuntimeError('Port 8000 already serves a model')
                print(f'STARTING_SERVER: {label}\n+ {shlex.join(argv)}', flush=True)
                log.write('+ ' + shlex.join(argv) + '\n')
                log.flush()
                process = subprocess.Popen(args=argv, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(process=process)
                print(f'BENCHMARKING: {label}; C256, 2560 requests, 512 warmups', flush=True)
                run_command(argv=client, log=folder / 'client.log', timeout=7200, env=env)
                row.update(read_result(path=folder / 'client.json'))
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
                rows.append(row)
                save(path=output / 'summary.json', value=rows)
        if healthy():
            raise RuntimeError('Server remained alive; refusing the next case')
    columns = ['case', 'input_length', 'output_length', 'concurrency', 'status',
               'output_throughput', 'total_token_throughput', 'total_per_chip', 'mean_tpot_ms']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row), flush=True)
    return int(any(row['status'] != 'complete' for row in rows))


def main() -> int:
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'metadata'
    metadata.mkdir()
    try:
        source = Path(os.environ['INPUT_SERVER_COMMANDS_DIR'])
        servers = parse_commands(path=source / 'infx_server.sh', executable='vllm')
        clients = parse_commands(path=source / 'infx_client.sh', executable='python')
        for label, (isl, osl) in CASES.items():
            for key, value in {'--model': MODEL, '--random-input-len': str(isl),
                               '--random-output-len': str(osl), '--max-concurrency': '256',
                               '--num-prompts': '2560', '--num-warmups': '512',
                               '--random-range-ratio': '1.0'}.items():
                if option(argv=clients[label], key=key) != value:
                    raise ValueError(f'Unexpected {label} client setting: {key}')
            if '--use-chat-template' not in clients[label] or '--ignore-eos' not in clients[label]:
                raise ValueError('Baseline client must retain chat templates and ignore EOS')
            if int(option(argv=servers[label], key='--max-model-len')) < isl + osl:
                raise ValueError('Server context is too small')
        for name in ('infx_server.sh', 'infx_client.sh'):
            shutil.copyfile(src=source / name, dst=metadata / name)
        shutil.copyfile(src=IMAGE_METADATA, dst=metadata / 'image-source.json')
        client = CLIENT_DIRECTORY
        print('CLONING_CLIENT: fresh pinned checkout on the TPU host', flush=True)
        run_command(argv=['git', 'clone', '--depth', '1', CLIENT_URL, str(client)],
                    log=metadata / 'clone.log', timeout=300)
        for args in (['fetch', '--depth', '1', 'origin', CLIENT_REVISION], ['checkout', '--detach', 'FETCH_HEAD']):
            run_command(argv=['git', '-C', str(client), *args],
                        log=metadata / ('client-' + args[0] + '.log'), timeout=300)
        revision = subprocess.check_output(args=['git', '-C', str(client), 'rev-parse', 'HEAD'], text=True).strip()
        if revision != CLIENT_REVISION:
            raise ValueError('Client checkout does not match the pinned revision')
        script = client / 'utils/bench_serving/benchmark_serving.py'
        if not script.is_file():
            raise RuntimeError(f'Benchmark script missing: {script}')
        shutil.copytree(src=script.parent, dst=metadata / 'client-source',
                        ignore=shutil.ignore_patterns('__pycache__'))
        save(path=metadata / 'protocol.json', value={'client_url': CLIENT_URL, 'client_revision': revision,
            'client_sha256': hashlib.sha256(script.read_bytes()).hexdigest(), 'run_id': os.environ.get('RUN_ID'),
            'command_source_revision': '60a6ccddb', 'concurrency': 256, 'num_prompts': 2560,
            'num_warmups': 512, 'range_ratio': 1.0, 'use_chat_template': True,
            'prefill_flush_timeout': 'unset; source default is 30000 ms'})
        env = dict(os.environ)
        env['VLLM_ENGINE_READY_TIMEOUT_S'] = '10800'
        preflight = str(Path(__file__).with_name('preflight.py'))
        run_command(argv=[sys.executable, preflight, 'hardware', '8'],
                    log=metadata / 'hardware-preflight.log', timeout=300, env=env)
        for label, argv in servers.items():
            index = argv.index('vllm')
            run_command(argv=['env', *argv[1:index], sys.executable, preflight, 'server', *argv[index + 2:]],
                        log=metadata / (label.replace('/', '-') + '-server-preflight.log'), timeout=180, env=env)
        print('PREPARING_CHECKPOINT: shared scratch cache for both cases', flush=True)
        checkpoint = metadata / 'checkpoint.json'
        run_command(argv=[sys.executable, preflight, 'checkpoint', MODEL, str(checkpoint)],
                    log=metadata / 'checkpoint.log', timeout=7200, env=env)
        weights = json.loads(checkpoint.read_text())
        for label, argv in servers.items():
            argv.extend(['--revision=' + weights['revision'], '--tokenizer=' + weights['snapshot']])
            args = client_command(original=clients[label], script=script, output=output, tokenizer=weights['snapshot'])
            run_command(argv=[sys.executable, preflight, 'client', *args[1:]],
                        log=metadata / (label.replace('/', '-') + '-client-preflight.log'), timeout=300, env=env)
        return run_cases(servers=servers, clients=clients, script=script, output=output,
                         env=env, tokenizer=weights['snapshot'])
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
