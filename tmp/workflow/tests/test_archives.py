"""Verify compressed transport, local exports and legacy-result packaging."""
from __future__ import annotations

import fcntl
import gzip
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import artifacts
import controller
import core
import pack_results
import runtime
import test_workflow


class ArchiveTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_round_trip_and_compression(self) -> None:
        source = self.root / 'server.log'
        source.write_text('repeated server diagnostic\n' * 2000)
        archive = self.root / 'results.tar.gz'
        manifest = artifacts.make_bundle(destination=archive, files={'diagnostics/server.log': source},
                                         metadata={'format': 'fixture'})
        self.assertLess(archive.stat().st_size, source.stat().st_size // 5)
        artifacts.verify_bundle(path=archive, manifest=manifest, target=self.root / 'unpacked')
        self.assertEqual((self.root / 'unpacked/diagnostics/server.log').read_bytes(), source.read_bytes())

    def test_corruption_and_missing_members_are_rejected(self) -> None:
        source = self.root / 'error.txt'
        source.write_text('failure')
        archive = self.root / 'results.tar.gz'
        manifest = artifacts.make_bundle(destination=archive, files={'error.txt': source}, metadata={})
        wrong = {**manifest, 'files': [manifest['files'][0] | {'sha256': '0' * 64}]}
        with self.assertRaises(ValueError):
            artifacts.verify_bundle(path=archive, manifest=wrong)
        archive.write_bytes(archive.read_bytes()[:-8])
        with self.assertRaises((EOFError, OSError, tarfile.TarError)):
            artifacts.verify_bundle(path=archive, manifest=manifest)

    def test_archive_links_and_traversal_are_rejected(self) -> None:
        for name, kind in [('../escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
            archive = self.root / 'bad.tar.gz'
            with tarfile.open(archive, 'w:gz') as bundle:
                member = tarfile.TarInfo(name)
                member.type = kind
                member.linkname = '/tmp/other'
                bundle.addfile(member)
            manifest = {'files': [{'path': name, 'bytes': 0, 'sha256': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}]}
            with self.assertRaises(ValueError):
                artifacts.verify_bundle(path=archive, manifest=manifest, target=self.root / 'unpacked')
        self.assertFalse((self.root / 'escape').exists())

    def test_final_failure_uses_one_compressed_download_and_verifies(self) -> None:
        fixture = test_workflow.WorkflowTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        directory = fixture.prepare()
        job = controller.Job(directory)
        local = self.root / 'runtime'
        local.mkdir()
        (local / 'error.txt').write_text('workload failed\n' * 1000)
        bucket = self.root / 'bucket'
        bucket.mkdir()
        status = {'run_id': job.state['run_id'], 'image': job.state['image'],
                  'config_sha256': job.state['config_sha256'], 'state': 'failed', 'exit_code': 42}
        with patch('sys.stdout', new=io.StringIO()):
            runtime.publish_final(local=local, bucket=bucket, status=status, extras={})
        manifest = core.read_document(bucket / 'manifest.json')
        self.assertIn('archive', manifest)
        self.assertFalse((bucket / 'objects').exists())
        downloads = []
        def cloud(*, args: list[str], **kwargs):
            if args[:2] == ['storage', 'cat']:
                return 0, json.dumps(manifest)
            self.assertEqual(args[:2], ['storage', 'cp'])
            downloads.append(args)
            shutil.copyfile(src=bucket / manifest['archive']['path'], dst=args[-1])
            return 0, ''
        with patch.object(job, 'gcloud', side_effect=cloud), patch('sys.stdout', new=io.StringIO()):
            self.assertFalse(job.collect())
        self.assertTrue(job.state['artifacts_verified'])
        self.assertEqual(job.state['artifact_exit_code'], 42)
        self.assertEqual(len(downloads), 1)
        self.assertEqual((directory / 'collected/files/diagnostics/error.txt').read_bytes(), (local / 'error.txt').read_bytes())

    def fixture_run(self) -> Path:
        directory = self.root / 'repository/workflow/local/results/campaign/job'
        directory.mkdir(parents=True)
        core.save(path=directory / 'state.json', value={'run_id': 'fixture-' + 'a' * 24,
            'phase': 'FAILED', 'exit_code': 1})
        (directory / 'error.txt').write_text('fixture diagnostic')
        core.save(path=directory.parent / 'campaign.json', value={'jobs': ['job']})
        return directory

    def test_export_keeps_error_and_raw_resume_data(self) -> None:
        directory = self.fixture_run()
        duplicate = directory / 'collected/objects/duplicate'
        duplicate.parent.mkdir(parents=True)
        duplicate.write_text('not needed')
        with patch('sys.stdout', new=io.StringIO()):
            archive = pack_results.export_run(directory=directory)
        with tarfile.open(archive, 'r:gz') as bundle:
            names = bundle.getnames()
            self.assertIn('run/error.txt', names)
            self.assertIn('campaign/campaign.json', names)
            self.assertNotIn('run/collected/objects/duplicate', names)
        self.assertTrue((directory / 'state.json').exists())
        index = core.read_document(archive.with_suffix('').with_suffix('.json'))
        self.assertEqual(index['exit_code'], 1)
        self.assertEqual(index['sha256'], core.checksum(archive))

    def test_legacy_archive_command_skips_active_controller(self) -> None:
        directory = self.fixture_run()
        workflow = self.root / 'repository/tmp/workflow'
        workflow.mkdir(parents=True)
        old_log = self.root / 'repository/workflow/local/logs/launch.log'
        old_log.parent.mkdir()
        old_log.write_text('old launcher error')
        with (directory / '.lock').open('a') as lock, patch('sys.stdout', new=io.StringIO()):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(pack_results.archive_saved(workflow_root=workflow), 0)
        self.assertFalse((workflow / 'local/results/archives').exists())
        with patch('sys.stdout', new=io.StringIO()):
            self.assertEqual(pack_results.archive_saved(workflow_root=workflow), 0)
        self.assertEqual(len(list((workflow / 'local/results/archives').glob('*.tar.gz'))), 1)
        with gzip.open(workflow / 'local/logs/launch.log.gz', 'rt') as log:
            self.assertEqual(log.read(), 'old launcher error')

    def test_launcher_compresses_early_failure_and_returns_failure(self) -> None:
        workflow = self.root / 'workflow'
        workflow.mkdir()
        shutil.copyfile(src=controller.ROOT / 'run.sh', dst=workflow / 'run.sh')
        shutil.copyfile(src=controller.ROOT / 'launcher.py', dst=workflow / 'launcher.py')
        (workflow / 'bootstrap.py').write_text('print("early setup failure"); raise SystemExit(42)\n')
        result = subprocess.run(args=['bash', str(workflow / 'run.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 42)
        self.assertIn('Launcher FAILED (exit 42)', result.stdout)
        logs = list((workflow / 'local/logs').glob('*.log.gz'))
        self.assertEqual(len(logs), 1)
        with gzip.open(logs[0], 'rt') as log:
            text = log.read()
            self.assertIn('early setup failure', text)
            self.assertIn('Launcher FAILED (exit 42)', text)


if __name__ == '__main__':
    unittest.main()
