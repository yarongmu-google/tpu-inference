"""Check useful reports, explicit failure reasons, and legacy result recovery."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent / 'payload'))
from collector import RunReport
from reporting import case_id, render, save

CONFIG = {'tokens':512, 'variant':'original', 'be':8, 'bg':2, 'capacity':32,
          'bd1c':256, 'bd2c':128, 'bcT':0}


def record(variant='original', wall=10., sparse=20., profile='ok'):
    return {'config':{**CONFIG,'variant':variant}, 'status':'ok' if profile=='ok' else 'failed',
        'timings':{'uniform':{'median_us':wall,'samples_us':[wall]*3},
                   'sparse':{'median_us':sparse,'samples_us':[sparse]*3}},
        'correctness':{mode:{'passed':True} for mode in ('uniform','sparse')},
        'paired_correctness':{mode:{'passed':True} for mode in ('uniform','sparse')} if variant=='occupied' else {},
        'profiles':{mode:{'status':profile,'tc_median_us':wall/2,
                         'reason':'No matching device trace rows' if profile=='failed' else None}
                    for mode in ('uniform','sparse')}}


def bundle(path, files):
    path.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(name=path,mode='w:gz') as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name=name); entry.size=len(data)
            archive.addfile(tarinfo=entry,fileobj=io.BytesIO(data))


class ReportingTests(unittest.TestCase):
    def test_matched_speedups_and_separate_winners(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            records=[record(),record('occupied',wall=5.,sparse=40.)]
            render(output=output,records=records,complete=True,name='my-run')
            text=(output/'SUMMARY.md').read_text()
            self.assertIn('# my-run',text)
            self.assertIn('2.000',text)
            self.assertIn('0.500',text)
            best=json.loads((output/'best.json').read_text())
            self.assertTrue(best['complete'])
            self.assertEqual(best['winners']['t512/uniform/wall/occupied']['median_us'],5.)
            self.assertEqual(best['winners']['t512/sparse/wall/occupied']['median_us'],40.)

    def test_profile_failure_keeps_valid_wall_ranking_and_shows_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)
            render(output=output,records=[record('occupied',profile='failed')])
            best=json.loads((output/'best.json').read_text())['winners']
            self.assertIn('t512/uniform/wall/occupied',best)
            self.assertNotIn('t512/uniform/device/occupied',best)
            self.assertIn('No matching device trace rows',(output/'SUMMARY.md').read_text())
            self.assertTrue(list((output/'failures').glob('*.txt')))

    def test_accuracy_failure_has_timings_but_no_winner_for_that_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary); failed=record('occupied')
            failed['status']='failed'
            failed['correctness']['uniform']={'passed':False,'assertion':'accuracy mismatch 42 elements'}
            render(output=output,records=[failed])
            text=(output/'SUMMARY.md').read_text()
            self.assertIn('accuracy mismatch 42 elements',text)
            self.assertIn('10.000',text)
            best=json.loads((output/'best.json').read_text())['winners']
            self.assertNotIn('t512/uniform/wall/occupied',best)
            self.assertIn('t512/sparse/wall/occupied',best)

    def test_deduplicates_compact_and_raw_manifests_and_ignores_auxiliary_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); job=root/'campaign/job'; value=record()
            save(path=job/'state.json',value={'name':'my-run','artifacts_verified':True})
            identity=case_id(config=value['config'])
            for base in (job/'collected/files/artifacts/output/candidates',root/'candidates/job'):
                save(path=base/identity/'candidate.json',value={'metadata':{'outcome':value},'files':[]})
                save(path=base/'shared-weights/candidate.json',value={'metadata':{'outcome':None},'files':[]})
            aggregate=job/'collected/files/artifacts/output'
            save(path=aggregate/'matrix.json',value=[value['config']])
            save(path=aggregate/'baselines.json',value=[{'config':{'tokens':512}}])
            save(path=aggregate/'provenance.json',value={'jax':'fixture'})
            report=RunReport(directory=job,results=root)
            report.archive=lambda: self.fail('Unnecessary archive scan')
            report.load()
            self.assertEqual(len(report.records),1)
            self.assertEqual(len(report.auxiliary),1)
            self.assertEqual(len(next(iter(report.records.values()))['sources']),2)
            destination=report.write(output=root/'reports')
            self.assertEqual(destination.name,'my-run')

    def test_compile_error_recovered_from_part_without_bulk_dump(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); job=root/'campaign/job'
            save(path=job/'state.json',value={'name':'failed-run','artifacts_verified':False})
            value={'config':CONFIG,'status':'failed','reason':'Candidate compile or execution failed'}
            identity=case_id(config=CONFIG)
            folder=root/'candidates/job'/identity
            error=b'RuntimeError: actual compiler failure detail'
            save(path=folder/'candidate.json',value={'metadata':{'outcome':value},'files':[
                {'path':'error.txt','bytes':len(error),'sha256':hashlib.sha256(error).hexdigest(),'part':'part-0001.tar.gz'},
                {'path':'compiler-dumps/huge.mlir','bytes':999999999,'part':'part-9999.tar.gz'}]})
            bundle(path=folder/'part-0001.tar.gz',files={'error.txt':error})
            report=RunReport(directory=job,results=root); report.load()
            self.assertIn('actual compiler failure',report.records[identity]['error'])
            destination=report.write(output=root/'reports')
            self.assertIn('actual compiler failure',(destination/'SUMMARY.md').read_text())

    def test_archive_only_failure_and_missing_planned_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); job=root/'archive-only/job'
            value={'config':CONFIG,'status':'failed','reason':'execution failed','error':'device stopped'}
            prefix='run/collected/files/artifacts/output/'
            bundle(path=root/'archives/job.tar.gz',files={
                'run/state.json':json.dumps({'name':'archived-run','artifacts_verified':True}).encode(),
                prefix+'results.json':json.dumps([value]).encode(),
                prefix+'matrix.json':json.dumps([CONFIG,{**CONFIG,'variant':'occupied'}]).encode()})
            report=RunReport(directory=job,results=root); report.load()
            self.assertEqual(len(report.records),2)
            destination=report.write(output=root/'reports')
            text=(destination/'SUMMARY.md').read_text()
            self.assertIn('device stopped',text)
            self.assertIn('No collected outcome',text)
            self.assertFalse(json.loads((destination/'best.json').read_text())['complete'])

    def test_frozen_outcome_overrides_stale_blocked_aggregate(self):
        report=RunReport(directory=Path('/fixture/run'),results=Path('/fixture'))
        blocked={'config':CONFIG,'status':'blocked','reason':'session ended'}
        report.record(value=blocked,source='aggregate',priority=10)
        report.record(value=record(),source='candidate',priority=30)
        self.assertEqual(report.records[case_id(config=CONFIG)]['status'],'ok')
        self.assertNotIn('reason',report.records[case_id(config=CONFIG)])

    def test_stale_archive_cannot_reintroduce_failure_after_completed_outcome(self):
        report=RunReport(directory=Path('/fixture/run'),results=Path('/fixture'))
        report.record(value=record(),source='current candidate',priority=30)
        report.record(value={'config':CONFIG,'status':'blocked','reason':'session ended'},
                      source='old archive',priority=10)
        value=report.records[case_id(config=CONFIG)]
        self.assertEqual(value['status'],'ok')
        self.assertNotIn('reason',value)

    def test_archive_supplies_missing_shared_data_without_reverting_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); job=root/'campaign/job'; value=record()
            save(path=job/'state.json',value={'name':'my-run','artifacts_verified':True,'resources_cleaned':True})
            save(path=root/'candidates/job'/case_id(config=CONFIG)/'candidate.json',
                 value={'metadata':{'outcome':value},'files':[]})
            prefix='run/collected/files/artifacts/output/'
            bundle(path=root/'archives/job.tar.gz',files={
                'run/state.json':json.dumps({'name':'my-run','artifacts_verified':False,'resources_cleaned':False}).encode(),
                prefix+'matrix.json':json.dumps([CONFIG]).encode(),
                prefix+'baselines.json':json.dumps([{'config':{'tokens':512},'median_us':20.}]).encode(),
                prefix+'provenance.json':b'{"jax":"fixture"}'})
            report=RunReport(directory=job,results=root); report.load()
            self.assertTrue(report.expected)
            self.assertTrue(report.baselines)
            self.assertTrue(report.provenance)
            self.assertTrue(report.state['artifacts_verified'])
            self.assertTrue(report.state['resources_cleaned'])

    def test_non_candidate_artifacts_do_not_claim_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); job=root/'campaign/job'
            save(path=job/'state.json',value={'name':'empty-run'})
            save(path=root/'candidates/job/dump-preflight/candidate.json',value={'metadata':{'outcome':None},'files':[]})
            report=RunReport(directory=job,results=root); report.load()
            self.assertFalse(report.records)
            destination=report.write(output=root/'reports')
            self.assertIn('No candidate outcomes',(destination/'SUMMARY.md').read_text())


if __name__=='__main__': unittest.main()
