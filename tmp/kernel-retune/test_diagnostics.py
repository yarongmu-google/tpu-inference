"""Dump collection and flag negotiation without a TPU or cloud connection."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'payload'))
import diagnostics
import tune


class DiagnosticsTests(unittest.TestCase):
    def test_environment_preserves_unrelated_flags_and_replaces_dump_paths(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {
                'LIBTPU_INIT_ARGS': '--keep=1 --xla_mosaic_dump_to=/old --xla_jf_dump_to /older'}):
            env = diagnostics.environment(root=Path(tmp), jf=True)
            tokens = shlex.split(env['LIBTPU_INIT_ARGS'])
            self.assertEqual(tokens[0], '--keep=1')
            self.assertEqual(len(tokens), 3)
            self.assertIn(f'--xla_jf_dump_to={Path(tmp).resolve()}/jf', tokens)
            env = diagnostics.environment(root=Path(tmp), jf=False)
            self.assertFalse(any('xla_jf_dump_to' in t for t in shlex.split(env['LIBTPU_INIT_ARGS'])))

    def test_collection_attributes_candidates_and_preserves_all_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = directory / 'compiler-dumps'
            root.mkdir()
            (root / 'generation.txt').write_text('llo.vperm unrelated\n')
            before = diagnostics.snapshot(root=root)
            candidate = 'mosaic-post-finalize-llo.txt'
            (root / candidate).write_text('%0 = llo.vperm %1\nspill slot 4\nvector<128x1xi1>\n')
            (root / 'binary.pb').write_bytes(bytes(range(256)))
            files = [candidate, 'binary.pb']
            (root / 'reference.txt').write_text('llo.vperm reference\n')
            diagnostics.save(path=directory / 'compile-window.json', value={'before': before, 'files': files})
            report = diagnostics.collect(directory=directory)
            self.assertEqual(report['scope'], 'candidate_compile')
            self.assertEqual(report['selected_files'], sorted(files))
            self.assertEqual(report['mosaic_final_llo_files'], [candidate])
            self.assertEqual(report['reports'][0]['text_hits']['spill_or_reload_text'], 1)
            self.assertIn('not proof', report['interpretation'])
            self.assertFalse(root.exists())
            with tarfile.open(directory / 'compiler-dumps.tar.gz') as stream:
                self.assertEqual(len(stream.getnames()), 4)
                self.assertEqual(stream.extractfile('binary.pb').read(), bytes(range(256)))

    def test_abort_window_retains_partial_dumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            root = directory / 'compiler-dumps'
            root.mkdir()
            (root / 'earlier.txt').write_text('input generation')
            before = diagnostics.snapshot(root=root)
            diagnostics.save(path=directory / 'compile-window.json', value={'before': before})
            (root / 'partial.txt').write_text('infer-vector-layout: relayout failed')
            report = diagnostics.collect(directory=directory)
            self.assertEqual(report['selected_files'], ['partial.txt'])
            self.assertEqual(report['mosaic_final_llo_files'], [])
            self.assertTrue((directory / 'compiler-dumps.tar.gz').is_file())

    def test_frozen_compile_survives_later_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jf').mkdir()
            path = root / 'jf/module.txt'
            path.write_text('generation')
            before = diagnostics.snapshot(root=root)
            path.write_text('candidate compilation')
            files = diagnostics.freeze_compile(root=root, before=before)
            path.write_text('reference compilation')
            self.assertEqual(files, ['candidate-compile/jf/module.txt'])
            self.assertEqual((root / files[0]).read_text(), 'candidate compilation')
            self.assertEqual(path.read_text(), 'reference compilation')

    @staticmethod
    def fake_probe(*, reject_jf=False, other_failure=False):
        def run(**kwargs):
            flags = shlex.split(kwargs['env']['LIBTPU_INIT_ARGS'])
            jf = any(flag.startswith('--xla_jf_dump_to=') for flag in flags)
            if other_failure or (reject_jf and jf):
                kwargs['stdout'].write('unrelated compile error' if other_failure else
                                       'Unknown flag in LIBTPU_INIT_ARGS: --xla_jf_dump_to')
                return SimpleNamespace(returncode=-6)
            mosaic = Path(next(flag.split('=', 1)[1] for flag in flags if flag.startswith('--xla_mosaic_dump_to=')))
            (mosaic / 'probe-post-finalize-llo.txt').write_text('llo.vmatmul\n')
            return SimpleNamespace(returncode=0)
        return run

    def test_unknown_jf_retries_once_and_preserves_rejection(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                diagnostics.subprocess, 'run', side_effect=self.fake_probe(reject_jf=True)) as run:
            self.assertFalse(diagnostics.probe(output=Path(tmp)))
            self.assertEqual(run.call_count, 2)
            record = json.loads((Path(tmp) / 'probe-results.json').read_text())
            self.assertFalse(record['jf_accepted'])
            self.assertIn('Unknown flag', (Path(tmp) / 'mosaic-and-jf/probe.log').read_text())

    def test_acceptance_does_not_claim_jf_output(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                diagnostics.subprocess, 'run', side_effect=self.fake_probe()) as run:
            self.assertTrue(diagnostics.probe(output=Path(tmp)))
            self.assertEqual(run.call_count, 1)
            record = json.loads((Path(tmp) / 'probe-results.json').read_text())
            self.assertTrue(record['jf_accepted'])
            self.assertFalse(record['jf_emitted_probe_files'])

    def test_other_probe_failure_does_not_retry(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                diagnostics.subprocess, 'run', side_effect=self.fake_probe(other_failure=True)) as run:
            with self.assertRaises(RuntimeError):
                diagnostics.probe(output=Path(tmp))
            self.assertEqual(run.call_count, 1)

    def test_probe_timeout_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                diagnostics.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 120)) as run:
            with self.assertRaises(RuntimeError):
                diagnostics.probe(output=Path(tmp))
            self.assertEqual(run.call_count, 1)
            record = json.loads((Path(tmp) / 'probe-results.json').read_text())
            self.assertEqual(record['attempts'][0]['returncode'], 'timeout')

    def test_failed_worker_dumps_are_packed_by_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            worker = directory / 'worker.py'
            worker.write_text('import os, pathlib, shlex, sys\n'
                'flags=shlex.split(os.environ["LIBTPU_INIT_ARGS"])\n'
                'root=pathlib.Path(next(f.split("=",1)[1] for f in flags if f.startswith("--xla_mosaic_dump_to=")))\n'
                '(root/"partial.txt").write_text("relayout failed\\n")\n'
                'sys.exit(7)\n')
            outcome = tune.run_one(config={'candidate_timeout_seconds': 10},
                                    directory=directory / 'case', worker=worker, dump_jf=False)
            self.assertEqual(outcome['returncode'], 7)
            self.assertEqual(outcome['status'], 'failed')
            self.assertTrue((directory / 'case/compiler-dumps.tar.gz').is_file())
            self.assertFalse((directory / 'case/compiler-dumps').exists())


if __name__ == '__main__':
    unittest.main()
