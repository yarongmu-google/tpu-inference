"""Exercise terminal disconnects and archive-before-cleanup recovery locally."""
from __future__ import annotations

import gzip
import io
import json
import os
from pathlib import Path
import pty
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import core
import test_cdk_storage
import test_workflow


class LauncherRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ('launcher.py', 'run.sh'):
            shutil.copyfile(src=controller.ROOT / name, dst=self.root / name)
        (self.root / 'bootstrap.py').write_text('pass\n')
        (self.root / '.venv/bin').mkdir(parents=True)
        (self.root / '.venv/bin/python').symlink_to(sys.executable)

    def disconnect(self, number: int) -> None:
        (self.root / 'controller.py').write_text(
            'from pathlib import Path\nimport time\nPath("started").touch()\n'
            'while not Path("release").exists(): time.sleep(0.05)\nprint("results collected")\n')
        with (self.root / 'viewer.txt').open('w') as output:
            viewer = subprocess.Popen(args=['bash', str(self.root / 'run.sh')], cwd=self.root,
                                      stdout=output, stderr=output, start_new_session=True)
            try:
                deadline = time.monotonic() + 10
                while not (self.root / 'started').exists():
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                os.killpg(viewer.pid, number)
                viewer.wait(timeout=5)
                (self.root / 'release').touch()
                logs = self.root / 'local/logs'
                while not list(logs.glob('*.exit')):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.05)
                self.assertEqual(next(logs.glob('*.exit')).read_text().strip(), '0')
                with gzip.open(next(logs.glob('*.log.gz')), 'rt') as log:
                    text = log.read()
                self.assertIn('results collected', text)
                self.assertIn('Launcher completed (exit 0)', text)
            finally:
                (self.root / 'release').touch()
                if viewer.poll() is None:
                    viewer.kill()
                    viewer.wait(timeout=5)

    def test_hangup_does_not_stop_collection(self) -> None:
        self.disconnect(number=signal.SIGHUP)

    def test_ctrl_c_detaches_viewer_and_keeps_collection(self) -> None:
        self.disconnect(number=signal.SIGINT)

    def test_interactive_configuration_still_reads_terminal(self) -> None:
        (self.root / 'controller.py').write_text(
            'import sys\nassert sys.stdin.isatty()\nprint("name:", flush=True)\n'
            'assert input() == "fixture description"\nprint("configured")\n')
        master, slave = pty.openpty()
        try:
            process = subprocess.Popen(args=['bash', str(self.root / 'run.sh')], cwd=self.root,
                                       stdin=slave, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            os.write(master, b'fixture description\n')
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 0, (stdout, stderr))
            self.assertIn(b'configured', stdout)
        finally:
            os.close(master)
            os.close(slave)


class ResultRecoveryTests(unittest.TestCase):
    def test_archive_failure_keeps_cloud_resources(self) -> None:
        fixture = test_cdk_storage.CdkStorageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        with fixture.mocked(), patch('pack_results.export_run', side_effect=OSError('fixture disk full')):
            with self.assertRaisesRegex(OSError, 'fixture disk full'):
                fixture.job.finish_cleanup()
        self.assertFalse(fixture.job.state.get('deleted'))
        self.assertTrue(fixture.fixture.image_exists)
        self.assertEqual(fixture.commands, [])

    def test_verified_archive_exists_before_remote_deletion(self) -> None:
        fixture = test_cdk_storage.CdkStorageTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.gcloud
        checked = []
        def cloud(*, args, **kwargs):
            if args[:2] == ['storage', 'rsync']:
                archive = Path(fixture.job.state['recovery_archive'])
                self.assertTrue(archive.is_file())
                self.assertEqual(core.checksum(archive), fixture.job.state['recovery_archive_sha256'])
                checked.append(True)
            return original(args=args, **kwargs)
        fixture.gcloud = cloud
        with fixture.mocked():
            fixture.job.finish_cleanup()
        self.assertEqual(checked, [True])

    def test_successful_live_read_clears_stale_error(self) -> None:
        fixture = test_workflow.WorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        job = controller.Job(fixture.prepare())
        job.save(collection_error='old manifest 404')
        manifest = {'format': 'run-artifacts-v1', 'run_id': job.state['run_id'],
                    'image': job.state['image'], 'config_sha256': job.state['config_sha256'],
                    'state': 'running', 'exit_code': None, 'files': []}
        with patch.object(job, 'gcloud', return_value=(0, json.dumps(manifest))), patch('sys.stdout', new=io.StringIO()):
            job.collect(live=True)
        self.assertIsNone(job.state['collection_error'])

    def test_recovery_collects_latest_campaign_without_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, name in enumerate(('old', 'latest')):
                campaign = root / name
                (campaign / 'job').mkdir(parents=True)
                core.save(path=campaign / 'campaign.json', value={'jobs': ['job']})
                core.save(path=campaign / 'job/state.json', value={'job_id': 'j-fixture', 'submission_started': True})
                os.utime(campaign / 'campaign.json', (index + 1, index + 1))
            with patch('controller.locked_job', return_value=0) as collect, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(controller.recover_latest(root=root), 0)
            collect.assert_called_once_with(directory=root / 'latest/job', action='collect')
            core.save(path=root / 'latest/job/state.json', value={'submission_started': False})
            with patch('controller.locked_job') as collect, patch('sys.stdout', new=io.StringIO()):
                self.assertEqual(controller.recover_latest(root=root), 1)
            collect.assert_not_called()
