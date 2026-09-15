"""Render tuning measurements and failure reasons without transport metadata."""
from __future__ import annotations

import json
import math
from pathlib import Path

KNOBS = ('be', 'bg', 'capacity', 'bd1c', 'bd2c', 'bcT')
MODES = ('uniform', 'sparse')


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def case_id(config: dict) -> str:
    if config.get('variant') not in {'original', 'occupied'} or any(type(config.get(k)) is not int for k in ('tokens', *KNOBS)):
        raise ValueError('Candidate identity requires integer shape/knobs and a known variant')
    return f"t{config['tokens']}-{config['variant']}-" + '-'.join(f'{k}{config[k]}' for k in KNOBS)


def failure_reasons(record: dict) -> list[str]:
    reasons = []
    for key in ('reason', 'error'):
        if record.get(key):
            reasons.append(str(record[key]))
    for field, label in (('correctness', 'XLA accuracy'),
                         ('paired_correctness', 'original/modified accuracy')):
        for mode, check in record.get(field, {}).items():
            if check.get('passed') is False:
                detail = check.get('assertion') or check.get('reason') or (
                    f"{check.get('failed_elements', '?')}/{check.get('checked_elements', '?')} elements failed; "
                    f"max abs={check.get('max_abs_error', '?')}; relative L2={check.get('relative_l2_error', '?')}")
                reasons.append(f'{mode}: {label}: {detail}')
    for mode, profile in record.get('profiles', {}).items():
        if profile.get('status') == 'failed':
            reasons.append(f"{mode}: profiler: " + str(profile.get('error') or profile.get('reason') or 'failed without detail'))
    if record.get('checks', {}).get('paired_accuracy') is False and not record.get('paired_correctness'):
        reasons.append('Matched original output unavailable; paired accuracy could not be checked')
    if record.get('status') in {'failed', 'blocked', 'missing'} and not reasons:
        reasons.append('Failure detail was not collected; see run-level errors and retained artifacts')
    return list(dict.fromkeys(reasons))


def canonical(record: dict) -> dict:
    fields = ('config', 'status', 'reason', 'error', 'compile_seconds', 'timings',
              'profiles', 'correctness', 'paired_correctness', 'checks', 'validation_calls',
              'baseline_artifact', 'sources')
    result = {key: record[key] for key in fields if key in record}
    if 'timings' not in result and 'median_us' in record:
        result['timings'] = {'uniform': {k: record[k] for k in ('median_us', 'min_us', 'samples_us') if k in record}}
    result['id'] = case_id(config=record['config'])
    result['failures'] = failure_reasons(record=record)
    return result


def accuracy_ok(record: dict, mode: str) -> bool:
    if record.get('status') in {'blocked', 'missing'}:
        return False
    xla = record.get('correctness', {}).get(mode, {}).get('passed') is True
    paired = record['config']['variant'] == 'original' or (
        record.get('paired_correctness', {}).get(mode, {}).get('passed') is True)
    return xla and paired


def metric(record: dict, mode: str, kind: str) -> float | None:
    if kind == 'wall':
        value = record.get('timings', {}).get(mode, {}).get('median_us')
    else:
        profile = record.get('profiles', {}).get(mode, {})
        value = profile.get('tc_median_us') if profile.get('status') == 'ok' else None
    return value if isinstance(value, (int, float)) and math.isfinite(value) and value > 0 else None


def render(output: Path, records: list[dict], *, baselines: list[dict] | None = None,
           name: str = 'Kernel occupancy', complete: bool = False,
           run_errors: list[str] | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    normalized = [canonical(record=r) for r in records]
    save(path=output / 'results.json', value=normalized)
    winners = {}
    for record in normalized:
        for mode in MODES:
            for kind in ('wall', 'device'):
                value = metric(record=record, mode=mode, kind=kind)
                if not accuracy_ok(record=record, mode=mode) or value is None:
                    continue
                config = record['config']
                key = f"t{config['tokens']}/{mode}/{kind}/{config['variant']}"
                if key not in winners or value < winners[key]['median_us']:
                    winners[key] = {'candidate': record['id'], 'config': config, 'median_us': value}
    save(path=output / 'best.json', value={'complete': complete, 'winners': winners,
        'wall_metric': 'warmed synchronized call including collectives',
        'device_metric': 'TC dispatch excluding barrier/trailing-copy edges'})
    lines = [f'# {name}', '',
        f"{len(normalized)} candidate records; " + ('complete results.' if complete else 'partial results; winners provisional.'),
        'Times are microseconds. Speedup = original / modified; >1 means modified is faster.',
        'Wall and device rankings are separate. Failed accuracy cannot win; profiler failure only prevents device ranking.', '']
    if winners:
        lines += ['| Workload / metric / variant | Best median us | be / bg / C / d1 / d2 / cT |',
                  '| --- | ---: | --- |']
        for key, value in sorted(winners.items()):
            knobs = '/'.join(str(value['config'][k]) for k in KNOBS)
            lines.append(f"| {key} | {value['median_us']:.3f} | {knobs} |")
    else:
        lines.append('**No accuracy-qualified winner is available. Measured timings and failure reasons follow.**')
    pairs = {}
    for record in normalized:
        config = record['config']
        pairs.setdefault((config['tokens'], *(config[k] for k in KNOBS)), {})[config['variant']] = record
    def number(value):
        return '-' if value is None else f'{value:.3f}'
    for mode in MODES:
        lines += ['', f'## {mode.capitalize()} routing', '',
            '| T; be/bg/C/d1/d2/cT | Old wall | New wall | Speedup | Old TC | New TC | TC speedup | Accuracy old/new |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
        for key, pair in sorted(pairs.items()):
            old, new = pair.get('original'), pair.get('occupied')
            values = [metric(record=r, mode=mode, kind=k) if r else None
                      for k in ('wall', 'device') for r in (old, new)]
            valid = all(r is not None and accuracy_ok(record=r, mode=mode) for r in (old, new))
            ratios = [values[i] / values[i + 1] if valid and values[i] and values[i + 1] else None for i in (0, 2)]
            checks = '/'.join('missing' if r is None else 'pass' if accuracy_ok(record=r, mode=mode) else 'fail/unchecked' for r in (old, new))
            label = str(key[0]) + '; ' + '/'.join(str(v) for v in key[1:])
            row = [values[0], values[1], ratios[0], values[2], values[3], ratios[1]]
            lines.append('| ' + label + ' | ' + ' | '.join(number(v) for v in row) + ' | ' + checks + ' |')
    failures = [r for r in normalized if r['failures']]
    if failures or run_errors:
        lines += ['', '## Failures', '']
        for error in run_errors or []:
            lines += ['```text', str(error).replace('```', "'''"), '```', '']
        for record in failures:
            detail = '\n\n'.join(record['failures'])
            filename = 'failures/' + record['id'] + '.txt'
            target = output / filename
            target.parent.mkdir(exist_ok=True)
            target.write_text(detail + '\n')
            display = detail
            if 'Traceback (most recent call last)' in detail:
                nonempty = [line for line in detail.splitlines() if line.strip()]
                display = nonempty[0][:100] + ': ' + nonempty[-1][-200:]
            compact = ' '.join(display.split())[:300].replace('|', '\\|')
            lines.append(f"- [{record['id']}]({filename}): {compact}")
    if baselines:
        # Shared reference measurements occur once, never inside each candidate.
        save(path=output / 'baselines.json', value=baselines)
        lines += ['', '## Shared XLA reference', '', '| Tokens | Uniform wall us | Sparse wall us |', '| --- | ---: | ---: |']
        for base in baselines:
            timings = base.get('timings', {})
            lines.append(f"| {base.get('config', {}).get('tokens', '?')} | "
                f"{number(timings.get('uniform', {}).get('median_us', base.get('median_us')))} | "
                f"{number(timings.get('sparse', {}).get('median_us'))} |")
    temporary = output / 'SUMMARY.md.new'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'SUMMARY.md')
