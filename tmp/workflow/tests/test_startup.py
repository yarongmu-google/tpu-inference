"""Exercise pre-submission failures through the shell entry point."""
from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core


class StartupTests(unittest.TestCase):
    def test_failed_command_shows_stderr_and_keeps_full_log(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            core.save(path=directory / 'state.json', value={'run_id': 'fixture', 'phase': 'PREPARING'})
            job = controller.Job(directory)
            message = 'fixture remote command rejected'
            error = io.StringIO()
            with patch('sys.stdout', new=io.StringIO()), patch('sys.stderr', new=error):
                code, _ = job.command(args=[sys.executable, '-c',
                    f'import sys; print({message!r}, file=sys.stderr); sys.exit(27)'], check=False)
            self.assertEqual(code, 27)
            self.assertIn('Command FAILED (exit 27)', error.getvalue())
            self.assertIn(message, error.getvalue())
            self.assertIn(message, next((directory / 'commands').glob('*.stderr')).read_text())

    def test_comparison_builder_failure_is_visible_without_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            workflow = repository / 'tmp/workflow'
            shutil.copytree(src=controller.ROOT, dst=workflow,
                            ignore=shutil.ignore_patterns('.venv', 'local', '__pycache__'))
            # Use the test interpreter and its installed YAML dependency; no installs.
            (workflow / 'bootstrap.py').write_text('pass\n')
            (workflow / '.venv').symlink_to(sys.prefix, target_is_directory=True)
            original = controller.ROOT.parent / 'baselines'
            comparison = repository / 'tmp/baselines'
            shutil.copytree(src=original, dst=comparison,
                            ignore=shutil.ignore_patterns('results', '__pycache__'))
            commands = repository / 'scripts/vllm/benchmarking'
            commands.mkdir(parents=True)
            (commands / 'infx_server.sh').write_text('# unused before submission\n')
            description = core.read_document(comparison / 'job.yml')
            description['image_build']['argv'] = [sys.executable, '-c',
                'import sys; print("fixture image build failed", file=sys.stderr); sys.exit(42)']
            description['execution']['cleanup'] = 'manual'
            # JSON is valid YAML, so keep the wrapper's real job.yml argument.
            core.save(path=comparison / 'job.yml', value=description)
            fake_bin = repository / 'bin'
            fake_bin.mkdir()
            (fake_bin / 'git').write_text('#!/bin/sh\necho bench\n')
            (fake_bin / 'git').chmod(0o755)
            marker = repository / 'unexpected-cloud-command'
            for name in ('cdk', 'gcloud'):
                executable = fake_bin / name
                executable.write_text('#!/bin/sh\nprintf called > "$STARTUP_CLOUD_MARKER"\nexit 97\n')
                executable.chmod(0o755)
            result = subprocess.run(args=['bash', str(comparison / 'run.sh'), '--name', 'startup check'],
                env={**os.environ, 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH'],
                     'STARTUP_CLOUD_MARKER': str(marker), 'PYTHONDONTWRITEBYTECODE': '1'},
                text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn('fixture image build failed', result.stdout)
            self.assertIn('CDK job=not submitted', result.stdout)
            self.assertIn('Launcher FAILED (exit 1)', result.stdout)
            self.assertFalse(marker.exists(), 'No cloud command should precede the failed image build')
            archive = next((comparison / 'results/archives').glob('*.tar.gz'))
            with tarfile.open(name=archive, mode='r:gz') as bundle:
                self.assertIn('run/error.txt', bundle.getnames())
                with bundle.extractfile('run/image/command.log') as log:
                    self.assertIn(b'fixture image build failed', log.read())
            launcher = next((workflow / 'local/logs').glob('*.log.gz'))
            with gzip.open(filename=launcher, mode='rt') as log:
                self.assertIn('Launcher FAILED (exit 1)', log.read())


if __name__ == '__main__':
    unittest.main()
