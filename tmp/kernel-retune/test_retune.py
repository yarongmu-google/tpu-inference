"""Local checks for tuning validation, failure collection and description wiring."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retune', HERE / 'payload/tune.py')
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)


class RetuneTests(unittest.TestCase):
    def test_matrix_has_no_duplicate_effective_shapes(self):
        plan = json.loads((HERE / 'payload/plan.json').read_text())
        matrix = tune.candidates(plan=plan)
        identities = [(c['tokens'], c['token_tile_size'], c['bf16_rows']) for c in matrix]
        self.assertEqual(len(identities), 22)
        self.assertEqual(len(set(identities)), len(identities))
        self.assertEqual({c['tokens'] for c in matrix}, {512, 1024, 2048, 8192})
        for tokens in plan['tokens']:
            self.assertTrue(any(c['tokens'] == tokens and c['token_tile_size'] == min(tokens, 1024)
                                and c['bf16_rows'] == 32 for c in matrix))

    def test_misaligned_plan_fails_before_spawning(self):
        plan = json.loads((HERE / 'payload/plan.json').read_text())
        for key, value in [('token_tiles', [128]), ('bf16_rows', [10]), ('tokens', [513])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                tune.candidates(plan={**plan, key: value})

    def test_failure_and_timeout_leave_collectible_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text('import sys\nprint("compile failed", flush=True)\nsys.exit(7)\n')
            config = {'candidate_timeout_seconds': 1, 'tokens': 512,
                      'token_tile_size': 512, 'bf16_rows': 32}
            failure = tune.run_one(config=config, directory=root / 'fail', worker=worker)
            self.assertEqual(failure['returncode'], 7)
            self.assertEqual(failure['status'], 'failed')
            self.assertIn('compile failed', (root / 'fail/worker.log').read_text())
            worker.write_text('import time\nprint("starting", flush=True)\ntime.sleep(60)\n')
            timeout = tune.run_one(config=config, directory=root / 'timeout', worker=worker)
            self.assertEqual(timeout['status'], 'timeout')
            tune.summarize(output=root, records=[failure, timeout])
            self.assertEqual(json.loads((root / 'winners.json').read_text())['winners'], {})
            self.assertIn('timeout', (root / 'SUMMARY.md').read_text())

    def test_success_without_correctness_cannot_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text('import json, pathlib, sys\n'
                              'pathlib.Path(sys.argv[-1], "result.json").write_text('
                              'json.dumps({"status":"ok", "median_us":1}))\n')
            outcome = tune.run_one(config={'candidate_timeout_seconds': 10},
                                   directory=root / 'bad', worker=worker)
            self.assertEqual(outcome['status'], 'failed')

    def test_valid_success_is_recorded_and_ranked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text('import json, pathlib, sys\n'
                              'pathlib.Path(sys.argv[-1], "result.json").write_text('
                              'json.dumps({"status":"ok", "median_us":10, '
                              '"correctness":{"uniform":{"passed":True},"skew":{"passed":True}}}))\n')
            config = {'candidate_timeout_seconds': 10, 'tokens': 512,
                      'token_tile_size': 512, 'bf16_rows': 32}
            outcome = tune.run_one(config=config, directory=root / 'good', worker=worker)
            self.assertEqual(outcome['status'], 'ok')
            tune.summarize(output=root, records=[outcome])
            winner = json.loads((root / 'winners.json').read_text())
            self.assertTrue(winner['provisional'])
            self.assertEqual(winner['winners']['512']['median_us'], 10)

    def test_shell_forwarding_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / 'kernel-retune'
            workflow = root / 'workflow'
            job.mkdir()
            workflow.mkdir()
            (job / 'run.sh').write_bytes((HERE / 'run.sh').read_bytes())
            (workflow / 'run.sh').write_text('printf "%s\\n" "$@"\nexit 7\n')
            for argv, first in [(['--name', 'test'], str(job / 'job.yml')),
                                (['--recover'], 'recover')]:
                result = subprocess.run(args=['bash', str(job / 'run.sh'), *argv],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 7)
                self.assertEqual(result.stdout.splitlines()[0], first)


if __name__ == '__main__':
    unittest.main()
