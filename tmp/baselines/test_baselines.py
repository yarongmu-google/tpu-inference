"""Check command fidelity, runtime client placement and retained failures locally."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'payload'))
sys.path.insert(0, str(HERE))
import runner
import build_image


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = ROOT / 'scripts/vllm/benchmarking'

    def test_both_committed_commands_are_preserved(self) -> None:
        servers = runner.parse_commands(path=self.source / 'infx_server.sh', executable='vllm')
        clients = runner.parse_commands(path=self.source / 'infx_client.sh', executable='python')
        for label, lengths in runner.CASES.items():
            server = servers[label]
            self.assertEqual(runner.option(argv=server, key='--max-num-seqs'), '64' if label == '8k/1k' else '128')
            self.assertEqual(runner.option(argv=server, key='--max-model-len'), '9236' if label == '8k/1k' else '2068')
            self.assertIn('USE_BATCHED_RPA_KERNEL=1', server)
            self.assertIn('GDN_BF16_RECURRENT_STATE=1', server)
            self.assertFalse(any(v.startswith('DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS=') for v in server))
            original = clients[label]
            argv = runner.client_command(original=original, script=Path('/run-scratch/InferenceX/utils/bench_serving/benchmark_serving.py'),
                                         output=self.root, tokenizer='/run-scratch/tokenizer')
            self.assertEqual(argv[1], '/run-scratch/InferenceX/utils/bench_serving/benchmark_serving.py')
            changed = list(original)
            changed[0:2] = argv[:2]
            changed[changed.index('--result-dir') + 1] = str(self.root)
            self.assertEqual(argv, changed + ['--result-filename', 'client.json', '--tokenizer', '/run-scratch/tokenizer'])
            for key, value in {'--max-concurrency': '256', '--num-prompts': '2560', '--num-warmups': '512',
                               '--random-range-ratio': '1.0', '--random-input-len': str(lengths[0]),
                               '--random-output-len': str(lengths[1])}.items():
                self.assertEqual(runner.option(argv=argv, key=key), value)
            self.assertIn('--use-chat-template', argv)

    def test_repeated_or_unknown_case_is_rejected(self) -> None:
        path = self.root / 'commands.sh'
        original = (self.source / 'infx_client.sh').read_text()
        for text in (original + original, original.replace('# 1k/1k', '# 1k/8k')):
            path.write_text(text)
            with self.assertRaises(ValueError):
                runner.parse_commands(path=path, executable='python')

    def result(self) -> dict:
        return {'model_id': runner.MODEL, 'completed': 2560, 'num_prompts': 2560,
                'max_concurrency': 256, 'duration': 100, 'total_input_tokens': 20000,
                'total_output_tokens': 10000, 'output_throughput': 100,
                'total_token_throughput': 300, 'mean_tpot_ms': 50, 'mean_ttft_ms': 100}

    def test_failed_first_case_preserves_error_and_allows_second_case(self) -> None:
        servers = {label: [sys.executable, '-c', 'import time; time.sleep(60)'] for label in runner.CASES}
        clients = runner.parse_commands(path=self.source / 'infx_client.sh', executable='python')
        def ready(*, process):
            if ready.first:
                ready.first = False
                raise RuntimeError('fixture startup failure')
        ready.first = True
        def run(*, argv, log, **kwargs):
            log.write_text('fixture client completed')
            (log.parent / 'client.json').write_text(json.dumps(self.result()))
        with patch.object(runner, 'healthy', return_value=False), patch.object(runner, 'wait_ready', side_effect=ready), \
                patch.object(runner, 'run_command', side_effect=run), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            result = runner.run_cases(servers=servers, clients=clients, script=self.root / 'client.py',
                                      output=self.root, env=dict(os.environ), tokenizer='fixture')
        self.assertEqual(result, 1)
        rows = json.loads((self.root / 'summary.json').read_text())
        self.assertEqual([row['status'] for row in rows], ['failed', 'complete'])
        self.assertIn('fixture startup failure', (self.root / '8k-1k/error.txt').read_text())
        self.assertTrue((self.root / '1k-1k/commands.json').is_file())

    def test_partial_result_cannot_be_reported_as_complete(self) -> None:
        path = self.root / 'client.json'
        path.write_text(json.dumps(self.result() | {'completed': 2500}))
        with self.assertRaises(ValueError):
            runner.read_result(path=path)

    def test_clone_is_on_runtime_host_and_uses_nested_script_path(self) -> None:
        output = self.root / 'output'
        client = self.root / 'runtime-client'
        image = self.root / 'image.json'
        image.write_text('{}')
        calls = []
        def run(*, argv, log, **kwargs):
            calls.append(argv)
            if argv[:2] == ['git', 'clone']:
                path = client / 'utils/bench_serving/benchmark_serving.py'
                path.parent.mkdir(parents=True)
                path.write_text('# fixture source\n')
            if 'checkpoint' in argv:
                Path(argv[-1]).write_text(json.dumps({'revision': 'b' * 40, 'snapshot': '/run-scratch/model'}))
            log.write_text('fixture')
        with patch.dict(os.environ, {'OUTPUT_DIR': str(output), 'INPUT_SERVER_COMMANDS_DIR': str(self.source)}), \
                patch.object(runner, 'CLIENT_DIRECTORY', client), patch.object(runner, 'IMAGE_METADATA', image), \
                patch.object(runner, 'run_command', side_effect=run), \
                patch.object(runner.subprocess, 'check_output', return_value=runner.CLIENT_REVISION + '\n'), \
                patch.object(runner, 'run_cases', return_value=0) as cases, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(sum(args[:2] == ['git', 'clone'] for args in calls), 1)
        self.assertEqual(cases.call_args.kwargs['script'], client / 'utils/bench_serving/benchmark_serving.py')
        self.assertEqual(sum('checkpoint' in args for args in calls), 1)
        self.assertTrue(any(runner.CLIENT_REVISION in args for args in calls))
        self.assertNotIn('DP_SCHED_BATCH_PREFILL_FLUSH_TIMEOUT_MS', cases.call_args.kwargs['env'])


class ImageTests(unittest.TestCase):
    def test_build_from_wrong_branch_stops_before_cloud_commands(self) -> None:
        with patch.dict(os.environ, {'IMAGE_REPOSITORY': 'us-central1-docker.pkg.dev/project/images/test-' + 'a' * 24,
                                     'IMAGE_RUN_ID': 'test-' + 'a' * 24}), \
                patch.object(build_image.platform, 'system', return_value='Linux'), \
                patch.object(build_image.platform, 'machine', return_value='x86_64'), \
                patch.object(build_image, 'git', return_value='topk'), patch.object(build_image, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'bench branch'):
                build_image.prepare()
        run.assert_not_called()

    def test_runtime_source_drift_stops_before_build_or_upload(self) -> None:
        def git(*args):
            if args[0] == 'branch': return 'bench'
            if args[0] == 'rev-parse': return 'c' * 40
            if args[0] == 'ls-tree': return 'tpu_inference/envs.py\nsetup.py'
            if args[0] == 'diff': return 'tpu_inference/envs.py'
            raise AssertionError(args)
        with patch.dict(os.environ, {'IMAGE_REPOSITORY': 'us-central1-docker.pkg.dev/project/images/test-' + 'a' * 24,
                                     'IMAGE_RUN_ID': 'test-' + 'a' * 24}), \
                patch.object(build_image.platform, 'system', return_value='Linux'), \
                patch.object(build_image.platform, 'machine', return_value='x86_64'), \
                patch.object(build_image, 'git', side_effect=git), patch.object(build_image, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, 'Runtime sources differ'):
                build_image.prepare()
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs['argv'][:2], ['git', 'merge-base'])


if __name__ == '__main__':
    unittest.main()
