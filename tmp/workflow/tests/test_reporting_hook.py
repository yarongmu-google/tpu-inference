"""Local report hooks use frozen scripts and run after failed jobs too."""
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import controller
import test_workflow


class ReportingHookTests(unittest.TestCase):
    def fixture(self):
        fixture=test_workflow.WorkflowTests(); fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    def test_frozen_report_runs_after_job_failure(self):
        fixture=self.fixture()
        source=fixture.base/'code/report.py'
        source.write_text('import argparse\nfrom pathlib import Path\np=argparse.ArgumentParser()\n'
            'p.add_argument("--run");p.add_argument("--output");a=p.parse_args()\n'
            'out=Path(a.output)/"friendly";out.mkdir(parents=True,exist_ok=True)\n'
            '(out/"SUMMARY.md").write_text("actual failure detail")\nprint("REPORT: friendly/SUMMARY.md")\n')
        fixture.description['outputs']['report']='report.py'
        directory=fixture.prepare()
        source.write_text('raise RuntimeError("Should use frozen source")\n')
        output=io.StringIO()
        with patch.object(controller,'job_action',return_value=1),patch('sys.stdout',new=output):
            code=controller.locked_job(directory=directory)
        self.assertEqual(code,1)
        self.assertTrue((fixture.base/'reports/friendly/SUMMARY.md').exists())
        self.assertIn('REPORT: friendly/SUMMARY.md',output.getvalue())
        state=json.loads((directory/'state.json').read_text())
        self.assertIsNone(state['report_error'])

    def test_report_must_be_included_in_snapshot(self):
        fixture=self.fixture()
        (fixture.base/'code/report.py').write_text('print("report")\n')
        fixture.description['code']['include']=['main.py']
        fixture.description['outputs']['report']='report.py'
        with self.assertRaisesRegex(ValueError,'included in the code snapshot'):
            fixture.prepare()

    def test_report_failure_is_visible_and_changes_exit_code(self):
        fixture=self.fixture()
        (fixture.base/'code/report.py').write_text('raise RuntimeError("report failed explicitly")\n')
        fixture.description['outputs']['report']='report.py'
        directory=fixture.prepare()
        with patch.object(controller,'job_action',return_value=0),patch('sys.stdout',new=io.StringIO()):
            self.assertEqual(controller.locked_job(directory=directory),1)
        state=json.loads((directory/'state.json').read_text())
        self.assertIn('failed',state['report_error'])

if __name__=='__main__': unittest.main()
