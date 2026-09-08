"""Exercise the concrete description without starting a model or using cloud services."""
from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import configure_profile
import controller
import core


class SweepIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix='sweep-description-test-')
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.workflow = self.base / 'workflow'
        self.workflow.mkdir()
        self.profile = {'version': 1,
            'cloud': {'project': 'fixture-project', 'region': 'us-central1', 'service_account': 'fixture-worker',
                      'workload_iam_member': 'serviceAccount:fixture@example.invalid'},
            'runtime': {'image': 'us-central1-docker.pkg.dev/fixture-project/images/runtime@sha256:' + 'a' * 64},
            'hardware': {'accelerator': 'tpu7x', 'topology': '2x2x1', 'chips_per_host': 4},
            'storage': {'dedicated_bucket': True, 'deletion': 'manual', 'soft_delete_days': 0}}
        self.description = core.read_document(controller.ROOT / 'sweep.yml')
        self.description['code']['directory'] = '../scripts'
        self.description_path = self.workflow / 'sweep.json'
        core.save(path=self.description_path, value=self.description)
        source = self.base / 'scripts'
        source.mkdir()
        for name in ('sweep.sh', 'server.sh', 'bench.sh'):
            (source / name).write_text('#!/bin/bash\nexit 0\n')
        (source / 'private-unselected.txt').write_text('not included')

    def saved_profile(self) -> Path:
        path = self.workflow / 'local/environment.json'
        core.save(path=path, value=self.profile)
        return path

    def test_description_freezes_exact_scripts_and_routes_all_outputs(self) -> None:
        self.saved_profile()
        with patch('sys.stdout', new=io.StringIO()):
            root = controller.prepare(description_path=self.description_path, name='fixture', dry_run=True)
        campaign = json.loads((root / 'campaign.json').read_text())
        self.assertEqual(len(campaign['jobs']), 1)
        self.assertEqual(campaign['max_in_flight'], 1)
        folder = root / campaign['jobs'][0]
        config = json.loads((folder / 'input/run.json').read_text())
        self.assertEqual({e['path'] for e in config['bundles'][0]['files']}, {'sweep.sh', 'server.sh', 'bench.sh'})
        self.assertFalse((folder / 'input/code/private-unselected.txt').exists())
        self.assertEqual(config['image'], self.profile['runtime']['image'])
        self.assertEqual(config['timeout_seconds'], 43200)
        self.assertIn('RESULT_DIR="$OUTPUT_DIR/client"', config['run']['argv'][2])
        self.assertIn('SERVER_LOG_DIR="$OUTPUT_DIR/server"', config['run']['argv'][2])
        self.assertIn('RUN_METADATA_DIR="$OUTPUT_DIR/metadata"', config['run']['argv'][2])
        self.assertIn('JOBSET_INSTALL_SOURCES=0', config['run']['argv'][2])

    def test_adapter_propagates_environment_and_failure(self) -> None:
        output = self.base / 'output with spaces'
        source = self.base / 'image'
        source.mkdir()
        bootstrap = source / 'bootstrap.sh'
        bootstrap.write_text('''#!/bin/bash
set -euo pipefail
test "$JOBSET_INSTALL_SOURCES" = 0
test "$RUN_ATTEMPT" = fixture-id
printf '%s\\n' "$RESULT_DIR" "$SERVER_LOG_DIR" "$RUN_METADATA_DIR" > "$OUTPUT_DIR/paths.txt"
echo failure > "$SERVER_LOG_DIR/server.log"
exit 42
''')
        client = self.base / 'client/utils/bench_serving'
        client.mkdir(parents=True)
        (client / 'benchmark_serving.py').write_text('# fixture\n')
        script = self.description['run']['argv'][2].replace('/opt/tpu-inference/tmp/jobset_bootstrap.sh', str(bootstrap))
        result = subprocess.run(args=['bash', '-c', script],
            env={**os.environ, 'OUTPUT_DIR': str(output), 'RUN_ID': 'fixture-id', 'INFERENCEX_REPO': str(self.base / 'client')},
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 42, result.stderr)
        self.assertEqual((output / 'paths.txt').read_text().splitlines(), [str(output / name) for name in ('client', 'server', 'metadata')])
        self.assertEqual((output / 'server/server.log').read_text(), 'failure\n')

    def test_saved_profile_is_reused_without_prompt_or_rewrite(self) -> None:
        path = self.saved_profile()
        before = path.read_bytes()
        with patch('builtins.input') as prompt, patch('sys.stdout', new=io.StringIO()):
            actual = configure_profile.configure(description=self.description_path, source=None)
        self.assertEqual(actual, path)
        self.assertEqual(path.read_bytes(), before)
        prompt.assert_not_called()

    def test_missing_profile_requires_interactive_setup(self) -> None:
        with patch('sys.stdin.isatty', return_value=False), self.assertRaisesRegex(ValueError, 'interactively'):
            configure_profile.configure(description=self.description_path, source=None)
        self.assertFalse((self.workflow / 'local/environment.json').exists())

    def test_previous_execution_supplies_image_and_account_but_not_guessed_identity(self) -> None:
        folder = self.base / 'previous/run-20260101'
        folder.mkdir(parents=True)
        core.save(path=folder / 'state.json', value={'image': self.profile['runtime']['image']})
        core.save(path=folder / 'submitted-recipe.yml', value={'spec': {'replicatedJobs': [
            {'template': {'spec': {'template': {'spec': {'serviceAccountName': 'fixture-worker'}}}}}]}})
        with patch('sys.stdout', new=io.StringIO()):
            defaults = configure_profile.source_defaults(self.base / 'previous')
        self.assertEqual(defaults['image'], self.profile['runtime']['image'])
        self.assertEqual(defaults['service_account'], 'fixture-worker')
        self.assertEqual(defaults['project'], 'fixture-project')
        self.assertNotIn('workload_iam_member', defaults)

    def test_profile_creation_validates_and_requires_confirmation(self) -> None:
        answers = ['fixture-project', 'us-central1', self.profile['runtime']['image'], 'fixture-worker',
                   'allUsers', 'serviceAccount:fixture@example.invalid', '', '', '', 'yes']
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=answers), patch('sys.stdout', new=io.StringIO()):
            profile_path = configure_profile.configure(description=self.description_path, source=None)
        self.assertEqual(json.loads(profile_path.read_text()), self.profile)
        self.assertEqual(profile_path.stat().st_mode & 0o777, 0o600)
        core.load_description(self.description_path)

    def test_profile_not_written_when_confirmation_declined(self) -> None:
        answers = ['fixture-project', 'us-central1', self.profile['runtime']['image'], 'fixture-worker',
                   'serviceAccount:fixture@example.invalid', '', '', '', 'no']
        with patch('sys.stdin.isatty', return_value=True), patch('builtins.input', side_effect=answers), patch('sys.stdout', new=io.StringIO()):
            with self.assertRaisesRegex(ValueError, 'not saved'):
                configure_profile.configure(description=self.description_path, source=None)
        self.assertFalse((self.workflow / 'local/environment.json').exists())

    def test_wrapper_invokes_generic_launcher_with_concrete_description(self) -> None:
        shutil.copyfile(src=controller.ROOT / 'run_sweep.sh', dst=self.workflow / 'run_sweep.sh')
        output = self.base / 'args.txt'
        (self.workflow / 'run.sh').write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 42\n')
        result = subprocess.run(args=['bash', str(self.workflow / 'run_sweep.sh'), '--dry-run', '--name', 'fixture name'],
                                env={**os.environ, 'CAPTURE': str(output)}, capture_output=True, text=True)
        self.assertEqual(result.returncode, 42)
        args = output.read_text().splitlines()
        self.assertEqual(args[0], str(self.workflow / 'sweep.yml'))
        self.assertEqual(args[1:3], ['--configure-profile', '--profile-source'])
        self.assertEqual(args[-3:], ['--dry-run', '--name', 'fixture name'])


if __name__ == '__main__':
    unittest.main()
