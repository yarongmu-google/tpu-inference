"""Exercise scheduling dependencies without TPU hardware."""
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'payload'))
import session
import tune


def complete(value):
    future = Future()
    future.set_result(value)
    return future


class FakeEngine:
    def __init__(self):
        self.events = []
        self.pending = []
        self.baseline_calls = []
        self.hold = False
        self.fail_baseline = None
        self.fail_candidate = False

    def initialize(self, config):
        self.events.append('weights')

    def baseline(self, config, directory):
        self.baseline_calls.append(config['tokens'])
        if config['tokens'] == self.fail_baseline:
            raise ValueError('baseline failure')
        self.persisted = Future() if self.hold else complete(None)
        return {'record': {'status': 'ok', 'config': config, 'median_us': 10., 'log': 'worker.log'},
                'persisted': self.persisted, 'expected': complete({}), 'directory': directory}

    def candidate(self, config, directory, baseline):
        self.events.append(directory.name)
        if self.fail_candidate:
            self.fail_candidate = False
            raise ValueError('compile failure')
        future = Future() if self.hold and not self.pending else complete({})
        if self.pending:
            for previous in self.pending:
                if not previous.done():
                    previous.set_result({})
            if not self.persisted.done():
                self.persisted.set_result(None)
        self.pending.append(future)
        return {'validation': future, 'config': config, 'directory': directory}

    def finish_candidate(self, candidate):
        assert candidate['validation'].done()
        self.events.append('timed-' + candidate['directory'].name)
        return {'status': 'ok', 'median_us': 1.}

    def release(self, candidate):
        pass

    def close(self):
        pass


class SessionTests(unittest.TestCase):
    def matrix(self):
        return tune.candidates(plan=json.loads((HERE / 'payload/plan.json').read_text()))

    def test_one_weight_generation_and_one_baseline_for_24_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            records = session.schedule(matrix=self.matrix(), output=Path(directory), engine=engine)
            self.assertEqual(engine.events.count('weights'), 1)
            self.assertEqual(engine.baseline_calls, [512])
            self.assertEqual(len(records), 24)
            self.assertEqual(len([e for e in engine.events if e.startswith('timed-')]), 24)

    def test_next_candidate_runs_while_accuracy_and_baseline_save_are_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            engine.hold = True
            matrix = self.matrix()[:2]
            session.schedule(matrix=matrix, output=Path(directory), engine=engine)
            first, second = [session.case_name(config=c) for c in matrix]
            self.assertLess(engine.events.index(second), engine.events.index('timed-' + first))

    def test_failed_baseline_blocks_dependents_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            engine.fail_baseline = 512
            with self.assertRaisesRegex(RuntimeError, 'baseline failed'):
                session.schedule(matrix=self.matrix(), output=Path(directory), engine=engine)
            self.assertEqual(engine.baseline_calls, [512])
            records = json.loads((Path(directory) / 'results.json').read_text())
            self.assertEqual(sum(r['status'] == 'blocked' for r in records), 24)
            self.assertEqual(sum(r['status'] == 'ok' for r in records), 0)

    def test_candidate_exception_preserves_error_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            engine.fail_candidate = True
            matrix = self.matrix()[:2]
            records = session.schedule(matrix=matrix, output=Path(directory), engine=engine)
            self.assertEqual([r['status'] for r in records], ['failed', 'ok'])
            self.assertIn('compile failure', (Path(directory) / '.work' /
                          session.case_name(config=matrix[0]) / 'error.txt').read_text())

class SupervisorTests(unittest.TestCase):
    def test_timeout_preserves_active_dumps_and_marks_unfinished_candidates(self):
        from unittest.mock import patch
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'output'
            plan = json.loads((HERE / 'payload/plan.json').read_text())
            plan.update(tokens=[512], sweep={key:[value] for key,value in plan['center'].items()}, candidate_timeout_seconds=1)
            plan_path = root / 'plan.json'
            plan_path.write_text(json.dumps(plan))
            script = root / 'fake.py'
            script.write_text('import os,pathlib,time\n'
                'root=pathlib.Path(os.environ["OUTPUT_DIR"])\n'
                '(root/".compiler-active/mosaic").mkdir(parents=True,exist_ok=True)\n'
                '(root/".compiler-active/mosaic/partial.mlir").write_text("partial compiler evidence")\n'
                'time.sleep(30)\n')
            original_popen = subprocess.Popen
            def spawn(*, args, **kwargs):
                kwargs['env']['OUTPUT_DIR'] = str(output)
                return original_popen(args=[sys.executable, str(script)], **kwargs)
            def probe(*, output, **kwargs):
                output.mkdir()
                return False
            with patch.object(session.diagnostics, 'probe', side_effect=probe), \
                 patch.object(session.subprocess, 'Popen', side_effect=spawn), \
                 patch.dict(os.environ, {'WORKFLOW_RUNTIME_DIR': str(HERE.parent / 'workflow')}):
                self.assertEqual(session.launch(plan_path=plan_path, output=output), 1)
            self.assertIn('Timed out', (output / 'session-error.txt').read_text())
            records = json.loads((output / 'results.json').read_text())
            self.assertEqual(records[0]['status'], 'blocked')
            inventory = json.loads((output / 'candidates/session-final-dumps/candidate.json').read_text())
            self.assertIn('compiler-dumps/mosaic/partial.mlir', [f['path'] for f in inventory['files']])


if __name__ == '__main__':
    unittest.main()
