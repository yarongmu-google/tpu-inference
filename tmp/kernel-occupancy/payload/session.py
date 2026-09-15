"""Schedule one TPU owner, CPU validation, and immutable artifact publication."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, wait
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import diagnostics
from tune import KNOBS, candidates, case_name, save, summarize


def ready(directory: Path) -> None:
    save(path=directory / '.ready.json', value={'complete': True})


def schedule(matrix: list[dict], output: Path, engine, *, max_pending: int = 2) -> list[dict]:
    """Run ready timings first; fill CPU-comparison waits with the next candidate."""
    if max_pending < 1:
        raise ValueError('max_pending must be positive')
    work = output / '.work'
    work.mkdir(exist_ok=True)
    records, baselines, pending = [], {}, []

    def progress(name: str) -> None:
        save(path=output / 'progress.json', value={'active': name, 'started': time.monotonic()})
        print(f'TPU_STAGE: {name}', flush=True)

    def record(directory: Path, config: dict, value: dict) -> dict:
        saved = directory / 'timings.json'
        if saved.exists() and 'timings' not in value:
            timings = json.loads(saved.read_text())
            value = {**value, 'timings': timings, **timings.get('uniform', {})}
        result = {**value, 'config': config, 'log': f'{directory.name}/worker.log'}
        save(path=directory / 'result.json', value=result)
        save(path=directory / 'outcome.json', value=result)
        (directory / 'worker.log').write_text(json.dumps(result, indent=2) + '\n')
        ready(directory=directory)
        return result

    def update() -> None:
        summarize(output=output, records=records,
                  baselines=[b['record'] for b in baselines.values()])

    def finish_baselines() -> None:
        for baseline in baselines.values():
            if baseline.get('published') or not baseline['persisted'].done():
                continue
            try:
                baseline['persisted'].result()
            except Exception:
                baseline['record'].update(status='failed', reason='Baseline output persistence failed')
                (baseline['directory'] / 'error.txt').write_text(traceback.format_exc())
            record(directory=baseline['directory'], config=baseline['record']['config'],
                   value=baseline['record'])
            baseline['published'] = True

    def finish_ready(*, block: bool) -> None:
        if block and pending and not any(c['validation'].done() for c in pending):
            progress(name='waiting_for_accuracy')
            wait([c['validation'] for c in pending], return_when=FIRST_COMPLETED)
        for item in pending[:]:
            if not item['validation'].done():
                continue
            progress(name='collect-checks-' + item['directory'].name)
            try:
                result = engine.finish_candidate(candidate=item)
            except Exception:
                (item['directory'] / 'error.txt').write_text(traceback.format_exc())
                result = {'status': 'failed', 'reason': 'Validation or timing failed',
                          'validation_calls': item.get('calls', {})}
            records.append(record(directory=item['directory'], config=item['config'], value=result))
            pending[:] = [p for p in pending if p is not item]
            engine.release(candidate=item)
        finish_baselines()
        update()

    progress(name='generate-shared-weights')
    engine.initialize(config=matrix[0])
    for directory in work.glob('shared-*'):
        ready(directory=directory)
    for config in matrix:
        finish_ready(block=False)
        while len(pending) >= max_pending:
            finish_ready(block=True)
        tokens = config['tokens']
        if tokens not in baselines:
            directory = work / f'xla-t{tokens}'
            directory.mkdir()
            baseline_config = {k: v for k, v in config.items() if k not in set(KNOBS) | {'variant'}}
            progress(name=directory.name)
            try:
                baselines[tokens] = engine.baseline(config=baseline_config, directory=directory)
            except Exception:
                from concurrent.futures import Future
                failure = Future()
                failure.set_exception(RuntimeError('Baseline failed; no automatic retry'))
                (directory / 'error.txt').write_text(traceback.format_exc())
                baselines[tokens] = {'directory': directory, 'expected': failure, 'persisted': failure,
                                    'record': {'config': baseline_config, 'status': 'failed',
                                               'log': f'{directory.name}/worker.log'}}
            for inputs in work.glob('inputs-*'):
                if not (inputs / '.ready.json').exists():
                    ready(directory=inputs)
        baseline = baselines[tokens]
        directory = work / case_name(config=config)
        directory.mkdir()
        if baseline['record']['status'] != 'ok' or (baseline['expected'].done() and baseline['expected'].exception()):
            records.append(record(directory=directory, config=config,
                value={'status': 'blocked', 'reason': 'XLA baseline failed; not retried'}))
            continue
        progress(name=directory.name)
        try:
            pending.append(engine.candidate(config=config, directory=directory, baseline=baseline))
        except Exception:
            (directory / 'error.txt').write_text(traceback.format_exc())
            value = {'status': 'failed', 'reason': 'Candidate compile or execution failed'}
            calls = directory / 'validation-calls.json'
            if calls.exists():
                value['validation_calls'] = json.loads(calls.read_text())
            records.append(record(directory=directory, config=config, value=value))
    while pending:
        finish_ready(block=True)
    progress(name='cpu-output-drain')
    save(path=output / 'tpu-finished.json', value={'time': time.time(),
         'note': 'TPU computation finished; pod remains allocated until durable artifact delivery'})
    engine.close()
    finish_baselines()
    update()
    if any(b['record']['status'] != 'ok' for b in baselines.values()):
        raise RuntimeError('A baseline failed or could not be persisted')
    return records


def launch(plan_path: Path, output: Path) -> int:
    """Supervise the device owner and package ready directories concurrently."""
    output.mkdir(parents=True, exist_ok=True)
    matrix = candidates(plan=json.loads(plan_path.read_text()))
    save(path=output / 'plan.json', value=json.loads(plan_path.read_text()))
    save(path=output / 'matrix.json', value=matrix)
    work = output / '.work'
    work.mkdir()
    diagnostics.disable_core_dumps()
    # Preflight is completed before creating any packaging threads or TPU owner.
    preflight = work / 'dump-preflight'
    try:
        dump_jf = diagnostics.probe(output=preflight, collect_dumps=False)
    except Exception:
        (output / 'preflight-error.txt').write_text(traceback.format_exc())
        from pipeline import Pipeline
        artifacts = Pipeline(output=output)
        artifacts.submit(directory=preflight)
        artifacts.finish()
        return 1
    from pipeline import Pipeline
    artifacts = Pipeline(output=output)
    artifacts.submit(directory=preflight)
    submitted = {preflight}
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--execute',
               '--matrix', str(output / 'matrix.json'), '--output', str(output)]
    save(path=output / 'session-command.json', value=command)
    env = diagnostics.environment(root=output / '.compiler-active', jf=dump_jf)
    process = None
    timed_out = False
    started = time.monotonic()
    previous = signal.getsignal(signal.SIGTERM)
    def stop(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, stop)
    try:
        with (output / 'session.log').open('w') as log:
            process = subprocess.Popen(args=command, stdout=log, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            heartbeat = 0.
            while process.poll() is None:
                for marker in work.glob('*/.ready.json'):
                    if marker.parent not in submitted:
                        artifacts.submit(directory=marker.parent)
                        submitted.add(marker.parent)
                state = {'active': 'startup', 'started': started}
                progress = output / 'progress.json'
                if progress.exists():
                    state = json.loads(progress.read_text())
                elapsed = time.monotonic() - state['started']
                if elapsed >= matrix[0]['candidate_timeout_seconds']:
                    timed_out = True
                    (output / 'session-error.txt').write_text(f'Timed out at {state["active"]}; session stopped; no retry\n')
                    break
                if time.monotonic() - heartbeat >= 30:
                    print(f'RUNNING: {state["active"]}; {elapsed:.0f}s; log={output / "session.log"}', flush=True)
                    heartbeat = time.monotonic()
                time.sleep(.2)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # The writer is dead now: incomplete work can be frozen for recovery.
        active_dumps = output / '.compiler-active'
        if active_dumps.exists():
            detached = work / 'session-final-dumps'
            detached.mkdir(exist_ok=True)
            active_dumps.rename(detached / 'compiler-dumps')
            save(path=detached / 'origin.json', value={
                'progress': json.loads((output / 'progress.json').read_text())
                            if (output / 'progress.json').exists() else None})
        for config in matrix:
            directory = work / case_name(config=config)
            if not (directory / 'outcome.json').exists():
                directory.mkdir(exist_ok=True)
                timings_path = directory / 'timings.json'
                timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}
                save(path=directory / 'outcome.json', value={'config': config, 'status': 'blocked',
                     'reason': 'Session ended before completion; no automatic retry',
                     'timings': timings, **timings.get('uniform', {})})
        recovered = [json.loads((work / case_name(config=c) / 'outcome.json').read_text()) for c in matrix]
        for item in recovered:
            item.setdefault('log', case_name(config=item['config']) + '/worker.log')
        summarize(output=output, records=recovered,
                  baselines=json.loads((output / 'baselines.json').read_text())
                            if (output / 'baselines.json').exists() else [])
        for directory in work.iterdir():
            if directory.is_dir() and directory not in submitted:
                artifacts.submit(directory=directory)
        tail = time.monotonic()
        print('ARTIFACT_TAIL: TPU work ended; retaining pod until packaging and upload finish', flush=True)
        complete = artifacts.finish()
        save(path=output / 'artifact-tail.json', value={'packaging_wait_seconds': time.monotonic() - tail,
             'note': 'Final publication is measured separately by the workflow runtime'})
        signal.signal(signal.SIGTERM, previous)
    return 0 if process is not None and process.returncode == 0 and not timed_out and complete else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    def stop(number, frame):
        raise SystemExit(128 + number)
    signal.signal(signal.SIGTERM, stop)
    engine = None
    try:
        # Runtime defaults must be installed before the first backend initialization.
        importlib.import_module('tpu_inference')
        from worker import Engine
        matrix = json.loads(args.matrix.read_text())
        engine = Engine(config=matrix[0], output=args.output)
        records = schedule(matrix=matrix, output=args.output, engine=engine)
        return int(any(record['status'] != 'ok' for record in records))
    except BaseException:
        (args.output / 'session-error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    sys.exit(main())
