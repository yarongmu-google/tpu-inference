"""Keep normal completion and recovery on the same finalization path."""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import controller
import test_cdk_storage


class FinalizationTests(unittest.TestCase):
    def fixture(self):
        value = test_cdk_storage.CdkStorageTests()
        value.setUp()
        self.addCleanup(value.doCleanups)
        return value

    def test_normal_and_recovery_use_same_finalizer(self) -> None:
        for action in ('run', 'collect'):
            with self.subTest(action=action):
                fixture = self.fixture()
                job = fixture.job
                job.save(finished=False, artifacts_verified=False)
                def collected():
                    job.save(artifacts_verified=True)
                    return True
                with fixture.mocked(), patch.object(job, 'collect', side_effect=collected) as collect, \
                        patch.object(job, 'finalize', wraps=job.finalize) as finalize:
                    self.assertEqual(controller.job_action(job=job, action=action, discard=False), 0)
                finalize.assert_called_once()
                collect.assert_called_once_with()
                self.assertTrue(job.state['resources_cleaned'])
                self.assertEqual(job.state['exit_code'], 0)

    def test_recovery_preserves_failed_remote_status(self) -> None:
        fixture = self.fixture()
        job = fixture.job
        fixture.fixture.remote['job_status'] = 'Failed'
        with fixture.mocked(), patch.object(job, 'collect', return_value=True):
            self.assertEqual(controller.job_action(job=job, action='collect', discard=False), 1)
        self.assertTrue(job.state['resources_cleaned'])
        self.assertEqual(job.state['exit_code'], 1)

    def test_cleaned_run_uses_verified_local_results(self) -> None:
        fixture = self.fixture()
        job = fixture.job
        job.save(deleted=True, resources_cleaned=True, exit_code=1)
        with patch.object(job, 'discover') as discover, patch.object(job, 'collect') as collect:
            self.assertEqual(job.finalize(), 1)
        discover.assert_not_called()
        collect.assert_not_called()

    def test_active_job_cannot_be_finalized(self) -> None:
        fixture = self.fixture()
        job = fixture.job
        with patch.object(job, 'discover', return_value={'job_status': 'Running', 'state': 'Active'}), \
                patch.object(job, 'collect') as collect, patch.object(job, 'finish_cleanup') as cleanup:
            with self.assertRaisesRegex(ValueError, 'active, archiving, or unresolved'):
                job.finalize()
        collect.assert_not_called()
        cleanup.assert_not_called()
