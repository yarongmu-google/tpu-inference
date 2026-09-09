"""Local checks for workload matching, failure retention and process cleanup."""
from __future__ import annotations

import argparse
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
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent / 'payload'))
import compare
import preflight

REPO = Path(os.environ.get('COMPARISON_TEST_REPO', str(Path(__file__).resolve().parents[2])))


class ComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = REPO / 'scripts/vllm/benchmarking/bench_throughput_qwen_server.sh'

    def result(self) -> dict:
        return {'model_id': compare.MODEL, 'completed': 2048, 'num_prompts': 2048,
            'max_concurrency': 512, 'duration': 1000, 'total_input_tokens': 1800000,
            'total_output_tokens': 15000000, 'output_throughput': 15000,
            'total_token_throughput': 16800, 'mean_tpot_ms': 50, 'mean_ttft_ms': 1000}

    def test_extracts_real_commands_without_executing_other_lines(self) -> None:
        commands = compare.server_commands(source=self.source)
        self.assertEqual(list(commands), ['baseline', '4g', '4i'])
        self.assertIn('--enable-expert-parallel', commands['baseline'])
        self.assertNotIn('--enable-expert-parallel', commands['4g'])
        self.assertIn('--max-num-seqs=104', commands['4i'])
        for name in ('4g', '4i'):
            self.assertIn('MOE_TP_DECODE_MAX_TOKENS=1024', commands[name])
            self.assertIn('--max-num-batched-tokens=128', commands[name])
        self.assertFalse(any('tee' in argv or 'xz' in argv for argv in commands.values()))

    def test_ambiguous_source_is_rejected(self) -> None:
        source = self.root / 'server.sh'
        source.write_text(self.source.read_text() * 2)
        with self.assertRaisesRegex(ValueError, 'Expected one'):
            compare.server_commands(source=source)

    def test_environment_does_not_leak_kernel_flags(self) -> None:
        with patch.dict(os.environ, {'USE_MOE_TP_DECODE_KERNEL': '1', 'DP_SCHED_BATCH_PREFILL': 'true'}):
            env = compare.clean_environment(commands=compare.server_commands(source=self.source))
        self.assertNotIn('USE_MOE_TP_DECODE_KERNEL', env)
        self.assertNotIn('DP_SCHED_BATCH_PREFILL', env)
        self.assertIn('PATH', env)

    def test_client_matches_historical_workload(self) -> None:
        argv = compare.client_command(client=self.root / 'client', output=self.root)
        for key, value in {'--max-concurrency': '512', '--num-prompts': '2048', '--seed': '0',
                           '--num-warmups': '0', '--random-range-ratio': '0.8'}.items():
            self.assertEqual(argv[argv.index(key) + 1], value)
        self.assertNotIn('--use-chat-template', argv)
        self.assertIn('--save-result', argv)

    def test_partial_and_inconsistent_metrics_are_rejected(self) -> None:
        path = self.root / 'result.json'
        for change in ({'completed': 1772}, {'max_concurrency': 832}, {'total_token_throughput': 9000}):
            path.write_text(json.dumps(self.result() | change))
            with self.assertRaises(ValueError):
                compare.read_result(path=path)
        path.write_text(json.dumps(self.result()))
        self.assertEqual(compare.read_result(path=path)['total_per_chip'], 4200)

    def test_real_failed_command_preserves_stderr(self) -> None:
        log = self.root / 'failed.log'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'exited 42'):
            compare.run_command(argv=[sys.executable, '-c', 'import sys; print("fixture failure", file=sys.stderr); sys.exit(42)'],
                                log=log, timeout=10)
        self.assertIn('fixture failure', log.read_text())

    def test_timeout_stops_real_child(self) -> None:
        pid = self.root / 'pid'
        script = 'import os,time; from pathlib import Path; Path(' + repr(str(pid)) + ').write_text(str(os.getpid())); time.sleep(60)'
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(TimeoutError):
            compare.run_command(argv=[sys.executable, '-c', script], log=self.root / 'timeout.log', timeout=1)
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pid.read_text()), 0)

    def test_failed_case_does_not_hide_next_results(self) -> None:
        commands = {label: [sys.executable, '-c', 'import time; time.sleep(60)'] for label in compare.CONFIGS}
        def client(*, argv: list[str], log: Path, **kwargs) -> None:
            if log.parent.name == 'baseline':
                log.write_text('fixture client failure')
                raise RuntimeError('fixture failure')
            (log.parent / 'client.json').write_text(json.dumps(self.result()))
        with patch.object(compare, 'healthy', return_value=False), patch.object(compare, 'wait_ready'), \
                patch.object(compare, 'run_command', side_effect=client), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = compare.compare(commands=commands, client=self.root / 'client', output=self.root, env=dict(os.environ))
        self.assertEqual(result, 1)
        rows = json.loads((self.root / 'summary.json').read_text())
        self.assertEqual([row['status'] for row in rows], ['failed', 'complete', 'complete'])
        self.assertIn('fixture failure', (self.root / 'baseline/error.txt').read_text())
        self.assertTrue((self.root / 'summary.csv').is_file())

    def test_one_runtime_clone_is_recorded_and_shared(self) -> None:
        client = self.root / 'fresh-client'
        output = self.root / 'output'
        calls = []
        def run(*, argv: list[str], log: Path, **kwargs) -> None:
            calls.append(argv)
            if argv[:2] == ['git', 'clone']:
                self.assertEqual(argv[-1], str(client))
                client.mkdir()
                (client / 'benchmark_serving.py').write_text('lower = int(seq_len * range_ratio)\nupper = seq_len\n')
            log.write_text('fixture command log')
        with patch.dict(os.environ, {'OUTPUT_DIR': str(output), 'INPUT_SERVER_COMMANDS_DIR': str(self.source.parent)}), \
                patch.object(compare, 'CLIENT_DIRECTORY', client), \
                patch.object(compare, 'run_command', side_effect=run), \
                patch.object(compare.subprocess, 'check_output', return_value='a' * 40 + '\n'), \
                patch.object(compare, 'compare', return_value=0) as comparison, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compare.main(), 0)
        self.assertEqual(sum(argv[:2] == ['git', 'clone'] for argv in calls), 1)
        self.assertEqual(comparison.call_args.kwargs['client'], client)
        modes = [argv[index + 1] for argv in calls for index, value in enumerate(argv)
                 if value.endswith('/preflight.py')]
        self.assertEqual(modes, ['client', 'server', 'hardware', 'server', 'hardware', 'server', 'hardware'])
        metadata = json.loads((output / 'metadata/comparison.json').read_text())
        self.assertEqual(metadata['client_revision'], 'a' * 40)
        self.assertTrue((output / 'metadata/client-source/benchmark_serving.py').is_file())

    def test_clone_failure_is_retained_before_any_server_starts(self) -> None:
        output = self.root / 'output'
        with patch.dict(os.environ, {'OUTPUT_DIR': str(output), 'INPUT_SERVER_COMMANDS_DIR': str(self.source.parent)}), \
                patch.object(compare, 'run_command', side_effect=RuntimeError('fixture clone failure')), \
                patch.object(compare, 'compare') as comparison, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(compare.main(), 1)
        comparison.assert_not_called()
        self.assertIn('fixture clone failure', (output / 'error.txt').read_text())

    def test_description_freezes_runner_and_exact_server_source(self) -> None:
        sys.path.insert(0, str(REPO / 'tmp/workflow'))
        import controller
        import core
        here = self.root / 'tmp/serving-comparison'
        here.mkdir(parents=True)
        shutil.copyfile(src=Path(__file__).parent / 'job.yml', dst=here / 'job.yml')
        shutil.copytree(src=Path(__file__).parent / 'payload', dst=here / 'payload')
        server = self.root / 'scripts/vllm/benchmarking/bench_throughput_qwen_server.sh'
        server.parent.mkdir(parents=True)
        shutil.copyfile(src=self.source, dst=server)
        profile = self.root / 'tmp/workflow/sweep-environment.yml'
        profile.parent.mkdir()
        shutil.copyfile(src=REPO / 'tmp/workflow/sweep-environment.yml', dst=profile)
        with contextlib.redirect_stdout(io.StringIO()):
            root = controller.prepare(description_path=here / 'job.yml', name='comparison check', dry_run=True)
        campaign = core.read_document(root / 'campaign.json')
        self.assertEqual(len(campaign['jobs']), 1)
        state = core.read_document(root / campaign['jobs'][0] / 'state.json')
        self.assertEqual(state['execution']['cleanup'], 'after_collection')
        template = core.read_document(root / campaign['jobs'][0] / 'run-template.json')
        self.assertEqual({entry['path'] for entry in template['bundles'][0]['files']},
                         {'compare.py', 'preflight.py'})
        self.assertEqual({entry['path'] for entry in template['bundles'][1]['files']},
                         {'bench_throughput_qwen_server.sh'})


    def test_client_preflight_parses_exact_arguments_without_running_benchmark(self) -> None:
        script = self.root / 'client.py'
        marker = self.root / 'benchmark-started'
        script.write_text('import argparse; from pathlib import Path\n'
            'p=argparse.ArgumentParser(); p.add_argument("--model",required=True); '
            'p.add_argument("--tokenizer"); args=p.parse_args(); '
            'Path(' + repr(str(marker)) + ').write_text("started")\n')
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1])
        factory = SimpleNamespace(from_pretrained=lambda model: tokenizer)
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoTokenizer=factory)}), \
                contextlib.redirect_stdout(io.StringIO()):
            preflight.client(script=script, argv=['--model', 'fixture'])
        self.assertFalse(marker.exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preflight.client(script=script, argv=['--model', 'fixture', '--unknown'])
        self.assertFalse(marker.exists())

    def test_server_preflight_validates_without_starting_server(self) -> None:
        validated = []
        class Command:
            def subparser_init(self, parsers):
                parser = parsers.add_parser('serve')
                parser.add_argument('model')
                parser.add_argument('--max-model-len', type=int)
            def validate(self, args):
                validated.append(args)
            def cmd(self, args):
                raise AssertionError('Server must not start during preflight')
        modules = {'vllm.entrypoints.cli.serve': SimpleNamespace(ServeSubcommand=Command),
                   'vllm.utils.argparse_utils': SimpleNamespace(FlexibleArgumentParser=argparse.ArgumentParser)}
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            preflight.server(argv=['fixture', '--max-model-len=9216'])
        self.assertEqual(validated[0].max_model_len, 9216)
        with patch.dict(sys.modules, modules), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            preflight.server(argv=['fixture', '--unknown'])

    def test_hardware_preflight_checks_all_devices_and_rejects_cpu(self) -> None:
        devices = [SimpleNamespace(platform='tpu') for _ in range(8)]
        visited = []
        class Value:
            def __add__(self, other):
                return self
            def block_until_ready(self):
                return [2]
        def put(value, device):
            visited.append(device)
            return Value()
        jax = SimpleNamespace(devices=lambda: devices, device_put=put, jit=lambda function: function)
        numpy = SimpleNamespace(array=lambda value, **kwargs: value, asarray=lambda value: value, int32=int)
        with patch.dict(sys.modules, {'jax': jax, 'numpy': numpy}), \
                patch.object(preflight.importlib.metadata, 'version', return_value='fixture'), \
                patch.object(preflight.importlib, 'import_module') as imports, \
                contextlib.redirect_stdout(io.StringIO()):
            preflight.hardware(expected_devices=8)
            self.assertEqual(visited, devices)
            self.assertEqual(imports.call_count, 2)
            devices[0].platform = 'cpu'
            with self.assertRaisesRegex(RuntimeError, 'Expected 8 TPU devices'):
                preflight.hardware(expected_devices=8)


if __name__ == '__main__':
    unittest.main()
