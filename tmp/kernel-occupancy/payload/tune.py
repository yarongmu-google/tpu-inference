"""Compare matched old/new kernels in a bounded sweep around the measured winner."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

KNOBS = ('be', 'bg', 'capacity', 'bd1c', 'bd2c', 'bcT')


def save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def case_name(config: dict) -> str:
    knobs = '-'.join(f'{key}{config[key]}' for key in KNOBS)
    return f"t{config['tokens']}-{config['variant']}-{knobs}"


def candidates(plan: dict) -> list[dict]:
    required = {'devices', 'hidden', 'experts', 'intermediate', 'top_k', 'tokens',
                'seed', 'warmup', 'iterations', 'profile_iterations', 'reference_samples',
                'candidate_timeout_seconds', 'center', 'sweep'}
    if set(plan) != required or set(plan['center']) != set(KNOBS) or set(plan['sweep']) != set(KNOBS):
        raise ValueError('Plan fields must match the documented schema')
    for key in required - {'tokens', 'center', 'sweep'}:
        minimum = 0 if key in {'seed', 'profile_iterations'} else 1
        if type(plan[key]) is not int or plan[key] < minimum:
            raise ValueError(f'Invalid {key}')
    if not 1 <= plan['top_k'] <= plan['experts']:
        raise ValueError('Invalid top_k')
    if plan['hidden'] % 128 or plan['intermediate'] % (128 * plan['devices']):
        raise ValueError('Hidden and local intermediate dimensions must align to 128')
    if not isinstance(plan['tokens'], list) or not plan['tokens']:
        raise ValueError('tokens must be nonempty')
    configurations = [dict(plan['center'])]
    for knob in KNOBS:
        values = plan['sweep'][knob]
        if not isinstance(values, list) or not values:
            raise ValueError(f'Invalid sweep for {knob}')
        configurations.extend({**plan['center'], knob: value} for value in values)
    result, seen = [], set()
    for tokens in plan['tokens']:
        if type(tokens) is not int or tokens < 1 or tokens % (plan['devices'] * 32):
            raise ValueError('Token counts must align to 32 per device')
        for knobs in configurations:
            if any(type(v) is not int or v < (0 if k == 'bcT' else 1) for k, v in knobs.items()):
                raise ValueError('Invalid kernel knob')
            if not 1 <= knobs['be'] <= 128 or plan['experts'] % (knobs['be'] * knobs['bg']):
                raise ValueError('Expert and group sizes must divide experts')
            if knobs['capacity'] % 32 or any(knobs[k] % 128 or plan['hidden'] % knobs[k] for k in ('bd1c', 'bd2c')):
                raise ValueError('Capacity/weight chunks are misaligned')
            if knobs['bcT'] and tokens % knobs['bcT']:
                raise ValueError('Combine rows must divide tokens')
            identity = (tokens, *(knobs[k] for k in KNOBS))
            if identity in seen:
                continue
            seen.add(identity)
            config = {k: v for k, v in plan.items() if k not in {'tokens', 'center', 'sweep'}}
            for variant in ('original', 'occupied'):
                result.append({**config, 'tokens': tokens, **knobs, 'variant': variant})
    return result


def summarize(output: Path, records: list[dict], baselines: list[dict] | None = None) -> None:
    winners = {}
    for record in records:
        if record['status'] != 'ok' or record['config']['variant'] != 'occupied':
            continue
        key = str(record['config']['tokens'])
        if key not in winners or record['median_us'] < winners[key]['median_us']:
            winners[key] = record
    save(path=output / 'results.json', value=records)
    save(path=output / 'winners.json', value={'metric': 'synchronized_call_wall_us_including_collectives',
         'provisional': True, 'winners': winners})
    lines = ['# Occupancy kernel comparison', '',
        'FP8 E4M3 weights, FP8 GMM1 activations, BF16 GMM2 activations. Full outputs checked against TPU XLA.',
        'Every executable uses the same warmups and synchronized samples. Failed accuracy retains timings and cannot win.',
        'Device TC times use the historical trace extraction; they exclude barrier/trailing-copy edges. SC coverage is saved separately.',
        'Each old/new pair shares inputs and block settings. The original kernel comes from 729a68e09f.', '',
        '| Case | Status | Uniform wall median us | Sparse wall median us | TC device median us | Uniform relative L2 | Sparse relative L2 | Old/new max abs |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    def number(value):
        return '-' if value is None else f'{value:.3f}'
    for record in records:
        correctness = record.get('correctness', {})
        profiles = record.get('profiles', {})
        paired = record.get('paired_correctness', {})
        deltas = [r['max_abs_error'] for r in paired.values() if 'max_abs_error' in r]
        lines.append(f"| {case_name(config=record['config'])} | {record['status']} | "
            f"{number(record.get('median_us'))} | {number(record.get('timings', {}).get('sparse', {}).get('median_us'))} | "
            f"{number(profiles.get('uniform', {}).get('tc_median_us'))} | "
            f"{number(correctness.get('uniform', {}).get('relative_l2_error'))} | {number(correctness.get('sparse', {}).get('relative_l2_error'))} | "
            f"{number(max(deltas) if deltas else None)} |")
    if baselines is not None:
        save(path=output / 'baselines.json', value=baselines)
        lines += ['', '## Untuned XLA references', '', '| Tokens | Uniform median us | Sparse median us |', '| --- | ---: | ---: |']
        for record in baselines:
            lines.append(f"| {record['config']['tokens']} | {number(record.get('median_us'))} | "
                         f"{number(record.get('timings', {}).get('sparse', {}).get('median_us'))} |")
    (output / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    from session import launch
    return launch(plan_path=args.plan, output=Path(os.environ['OUTPUT_DIR']))


if __name__ == '__main__':
    raise SystemExit(main())
