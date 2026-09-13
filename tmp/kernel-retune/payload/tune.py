"""Run isolated tuning candidates and retain failures as well as timings."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def candidates(plan: dict) -> list[dict]:
    required = {'devices', 'hidden', 'experts', 'intermediate', 'top_k',
                'tokens', 'token_tiles', 'bf16_rows', 'seed', 'warmup',
                'iterations', 'reference_samples', 'candidate_timeout_seconds'}
    if set(plan) != required:
        raise ValueError(f'Plan requires exactly {sorted(required)}')
    for key in required - {'tokens', 'token_tiles', 'bf16_rows'}:
        if type(plan[key]) is not int or plan[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Invalid {key}')
    width = plan['devices']
    if plan['hidden'] % 128 or plan['intermediate'] % (width * 128):
        raise ValueError('Hidden and local intermediate widths must align to 128')
    if not 1 <= plan['top_k'] <= plan['experts']:
        raise ValueError('Invalid top_k')
    for key in ('tokens', 'token_tiles', 'bf16_rows'):
        if (not isinstance(plan[key], list) or not plan[key]
                or any(type(v) is not int or v < 1 for v in plan[key])):
            raise ValueError(f'Invalid {key}')
    result, seen = [], set()
    for tokens in plan['tokens']:
        if tokens % width:
            raise ValueError('Token buckets must divide across devices')
        for tile in plan['token_tiles']:
            if tile % (width * 32):
                raise ValueError('FP8 token tiles must align to devices * 32')
            effective = min(tile, ((tokens // width + 31) // 32) * 32 * width)
            for rows in plan['bf16_rows']:
                if rows % 16:
                    raise ValueError('BF16 row tiles must align to 16')
                identity = (tokens, effective, rows)
                if identity in seen:
                    continue
                seen.add(identity)
                config = {k: v for k, v in plan.items()
                          if k not in {'tokens', 'token_tiles', 'bf16_rows'}}
                config.update(tokens=tokens, token_tile_size=effective, bf16_rows=rows)
                result.append(config)
    return result


def summarize(output: Path, records: list[dict]) -> None:
    winners = {}
    for record in records:
        if record['status'] != 'ok':
            continue
        key = str(record['config']['tokens'])
        if key not in winners or record['median_us'] < winners[key]['median_us']:
            winners[key] = record
    save(path=output / 'results.json', value=records)
    save(path=output / 'winners.json', value={
        'metric': 'synchronized_call_wall_us_including_collectives',
        'provisional': True, 'winners': winners})
    lines = ['# TP MoE retuning', '',
             'FP8 weights; FP8 GMM1 rows = 2 * BF16 rows. Times include dispatch and collectives.',
             'Winners are provisional wall-latency results, not device-only timings or serving throughput.', '',
             '| Tokens | Token tile | FP8/BF16 rows | Status | Median us | Log |',
             '| ---: | ---: | ---: | --- | ---: | --- |']
    for record in records:
        config = record['config']
        timing = f"{record['median_us']:.2f}" if record['status'] == 'ok' else '-'
        rows = config['bf16_rows']
        lines.append(f"| {config['tokens']} | {config['token_tile_size']} | {2 * rows}/{rows} "
                     f"| {record['status']} | {timing} | {record['log']} |")
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def run_one(config: dict, directory: Path, *, worker: Path) -> dict:
    directory.mkdir(parents=True)
    save(path=directory / 'config.json', value=config)
    command = [sys.executable, '-u', str(worker), '--config', str(directory / 'config.json'),
               '--output', str(directory)]
    save(path=directory / 'command.json', value=command)
    print('+ ' + shlex.join(command), flush=True)
    started = time.monotonic()
    timed_out = False
    log_offset = 0
    with (directory / 'worker.log').open('w') as log:
        process = subprocess.Popen(args=command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed >= config['candidate_timeout_seconds']:
                    timed_out = True
                    break
                try:
                    process.wait(timeout=min(30, config['candidate_timeout_seconds'] - elapsed))
                except subprocess.TimeoutExpired:
                    with (directory / 'worker.log').open() as progress:
                        progress.seek(log_offset)
                        updates = progress.read()
                        log_offset = progress.tell()
                    if updates:
                        print(updates[-4000:], end='', flush=True)
                    print(f'RUNNING {directory.name}: {int(time.monotonic() - started)}s; '
                          f'log={directory / "worker.log"}', flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    record = {'config': config, 'returncode': process.returncode,
              'elapsed_seconds': time.monotonic() - started,
              'log': f'{directory.name}/worker.log',
              'status': 'timeout' if timed_out else 'failed'}
    path = directory / 'result.json'
    if path.exists():
        try:
            result = json.loads(path.read_text())
            if process.returncode == 0 and not timed_out and result['status'] == 'ok':
                record.update(result)
                if not (isinstance(record['median_us'], (float, int))
                        and math.isfinite(record['median_us'])
                        and record['median_us'] > 0
                        and record['correctness']['uniform']['passed']
                        and record['correctness']['skew']['passed']):
                    raise ValueError('Incomplete or invalid successful result')
        except (ValueError, KeyError, TypeError) as error:
            record.update(status='failed', result_error=str(error))
    if record['status'] != 'ok':
        with (directory / 'worker.log').open('rb') as stream:
            stream.seek(max(0, (directory / 'worker.log').stat().st_size - 8000))
            print(stream.read().decode(errors='replace'), flush=True)
    save(path=directory / 'outcome.json', value=record)
    print(f"FINISHED {directory.name}: {record['status']}", flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    output = Path(os.environ['OUTPUT_DIR']).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    matrix = candidates(plan=plan)
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupted)
    save(path=output / 'plan.json', value=plan)
    save(path=output / 'matrix.json', value=matrix)
    print(f'RETUNE: {len(matrix)} candidates; output={output}', flush=True)
    records = []
    for index, config in enumerate(matrix, start=1):
        name = f"t{config['tokens']}-tile{config['token_tile_size']}-r{config['bf16_rows']}"
        print(f'[{index}/{len(matrix)}] START {name}', flush=True)
        record = run_one(config=config, directory=output / name,
                         worker=Path(__file__).with_name('worker.py'))
        records.append(record)
        summarize(output=output, records=records)
    return 0 if all(record['status'] == 'ok' for record in records) else 1


if __name__ == '__main__':
    sys.exit(main())
