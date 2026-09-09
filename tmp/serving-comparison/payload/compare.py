"""Run the same original-client workload against three frozen server commands."""
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


def client_command(client: Path, output: Path) -> list[str]:
    return [sys.executable, str(client / 'benchmark_serving.py'), '--model', MODEL,
        '--backend', 'vllm', '--host', '127.0.0.1', '--port', '8000', '--dataset-name', 'random',
        '--random-input-len', '1024', '--random-output-len', '8192', '--random-range-ratio', '0.8',
        '--random-prefix-len', '0', '--max-concurrency', '512', '--num-prompts', '2048',
        '--request-rate', 'inf', '--seed', '0', '--num-warmups', '0', '--ignore-eos',
        '--percentile-metrics', 'ttft,tpot,itl,e2el', '--save-result',
        '--result-dir', str(output), '--result-filename', 'client.json']


def read_result(path: Path) -> dict:
    value = json.loads(path.read_text())
    if (value.get('completed') != 2048 or value.get('num_prompts') != 2048
            or value.get('max_concurrency') != 512 or value.get('model_id') != MODEL):
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


def compare(commands: dict[str, list[str]], client: Path, output: Path, env: dict[str, str]) -> int:
    rows = []
    for label, argv in commands.items():
        folder = output / label
        folder.mkdir()
        row = {'config': label, 'concurrency': 512, 'status': 'failed'}
        process = None
        started = time.time()
        with (folder / 'server.log').open('w') as log:
            try:
                if healthy():
                    raise RuntimeError('Port 8000 already serves a model; refusing a contaminated comparison')
                save(path=folder / 'commands.json', value={'server': argv,
                    'client': client_command(client=client, output=folder), 'workload': WORKLOAD})
                print(f'STARTING_SERVER: {label}\n+ {shlex.join(argv)}', flush=True)
                log.write('+ ' + shlex.join(argv) + '\n')
                log.flush()
                process = subprocess.Popen(args=argv, env=env, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                wait_ready(process=process)
                print(f'BENCHMARKING: {label}; 2048 requests at concurrency 512', flush=True)
                run_command(argv=client_command(client=client, output=folder), log=folder / 'client.log',
                            timeout=7200, env=env)
                row.update(read_result(path=folder / 'client.json'))
                row['status'] = 'complete'
            except Exception:
                detail = traceback.format_exc()
                (folder / 'error.txt').write_text(detail)
                print(detail, file=sys.stderr, flush=True)
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
    if len({(row['total_input_tokens'], row['total_output_tokens']) for row in completed}) > 1:
        raise ValueError('Completed runs processed different token counts; inspect the client logs')
    columns = ['config', 'concurrency', 'status', 'output_throughput', 'total_token_throughput',
               'total_per_chip', 'mean_tpot_ms', 'mean_ttft_ms']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(json.dumps(row), flush=True)
    return 0 if len(completed) == len(commands) else 1


def main() -> int:
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / 'metadata'
    metadata.mkdir()
    try:
        source = Path(os.environ['INPUT_SERVER_COMMANDS_DIR']) / 'bench_throughput_qwen_server.sh'
        commands = server_commands(source=source)
        env = clean_environment(commands=commands)
        shutil.copyfile(src=source, dst=metadata / 'server-source.sh')
        provenance = Path('/opt/jobset/image-source.json')
        if provenance.is_file():
            shutil.copyfile(src=provenance, dst=metadata / 'image-source.json')
        client = CLIENT_DIRECTORY
        print('CLONING_CLIENT: one fresh checkout for all three configurations', flush=True)
        run_command(argv=['git', 'clone', '--depth', '1', CLIENT_URL, str(client)],
                    log=metadata / 'clone.log', timeout=300)
        revision = subprocess.check_output(args=['git', '-C', str(client), 'rev-parse', 'HEAD'], text=True).strip()
        archive = metadata / 'client-source'
        archive.mkdir()
        hashes = {}
        for path in sorted(client.iterdir()):
            if path.is_file() and (path.suffix == '.py' or path.name == 'LICENSE'):
                shutil.copyfile(src=path, dst=archive / path.name)
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        save(path=metadata / 'comparison.json', value={'client_url': CLIENT_URL,
            'client_revision': revision, 'client_files': hashes, 'workload': WORKLOAD,
            'expected_sampling': '[floor(0.8 * X), X]', 'server_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
        text = (client / 'benchmark_serving.py').read_text()
        if (not re.search(r'lower\s*=\s*int\(seq_len\s*\*\s*range_ratio\)', text)
                or not re.search(r'upper\s*=\s*seq_len\s*\n', text)):
            raise ValueError('Upstream sampler changed; inspect the saved client before benchmarking')
        run_command(argv=[sys.executable, str(client / 'benchmark_serving.py'), '--help'],
                    log=metadata / 'client-help.log', timeout=120, env=env)
        preflight = str(Path(__file__).with_name('preflight.py'))
        print('PREFLIGHT: validating client, server arguments and TPU execution before model loading', flush=True)
        run_command(argv=[sys.executable, preflight, 'client', *client_command(client=client, output=output)[1:]],
                    log=metadata / 'client-preflight.log', timeout=300, env=env)
        for label, argv in commands.items():
            index = argv.index('vllm')
            run_command(argv=['env', *argv[1:index], sys.executable, preflight, 'server', *argv[index + 2:]],
                        log=metadata / (label + '-preflight.log'), timeout=180, env=env)
            run_command(argv=['env', *argv[1:index], sys.executable, preflight, 'hardware', '8'],
                        log=metadata / (label + '-hardware-preflight.log'), timeout=300, env=env)
        return compare(commands=commands, client=client, output=output, env=env)
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
