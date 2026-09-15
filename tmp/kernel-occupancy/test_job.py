"""CPU checks for paired planning, background delivery, compaction, and scheduling."""
from concurrent.futures import Future
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'payload'),str(HERE.parent/'workflow'),str(HERE)]
import tune
import pipeline
from streams import materialize

class JobTests(unittest.TestCase):
    def test_pairs_and_deduplication(self):
        plan=json.loads((HERE/'payload/plan.json').read_text())
        matrix=tune.candidates(plan)
        self.assertEqual(len(matrix),24)
        self.assertEqual(len({tune.case_name(c) for c in matrix}),24)
        for old,new in zip(matrix[::2],matrix[1::2]):
            self.assertEqual(old['variant'],'original')
            self.assertEqual(new['variant'],'occupied')
            self.assertEqual({k:v for k,v in old.items() if k!='variant'}, {k:v for k,v in new.items() if k!='variant'})
        for key,value in [('capacity',33),('be',3),('bcT',77)]:
            bad=json.loads(json.dumps(plan)); bad['center'][key]=value
            with self.assertRaises(ValueError): tune.candidates(bad)

    def test_failed_accuracy_keeps_summary_timing_but_cannot_win(self):
        config=tune.candidates(json.loads((HERE/'payload/plan.json').read_text()))[1]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            tune.summarize(root,[{'config':config,'status':'failed','median_us':12.5,
                'correctness':{'uniform':{'relative_l2_error':.01}}}])
            self.assertIn('12.500',(root/'SUMMARY.md').read_text())
            self.assertEqual(json.loads((root/'winners.json').read_text())['winners'],{})

    def test_large_file_background_pack_and_verified_compaction(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{'WORKFLOW_RUNTIME_DIR':str(HERE.parent/'workflow')}):
            root=Path(temporary); case=root/'.work/case'; case.mkdir(parents=True)
            big=case/'compiler-dumps/big.txt'; big.parent.mkdir(); big.write_bytes(b'x'*(41*1024**2))
            (case/'outcome.json').write_text(json.dumps({'status':'failed','median_us':13.}))
            artifacts=pipeline.Pipeline(root)
            started,release=threading.Event(),threading.Event(); seal=artifacts.seal
            def delayed(**kwargs):
                started.set()
                if not release.wait(10): raise TimeoutError('release missing')
                return seal(**kwargs)
            artifacts.seal=delayed
            artifacts.submit(case)
            try:
                self.assertTrue(started.wait(10))
                (root/'next-device-work.txt').write_text('independent work can proceed')
                self.assertFalse((root/'candidates/case').exists())
            finally:
                release.set(); self.assertTrue(artifacts.finish())
            source=root/'candidates/case'; destination=root/'received/case'
            self.assertTrue(materialize(source,destination))
            self.assertTrue((destination/'summary/file-fragments.json').exists())
            self.assertTrue((destination/'summary/outcome.json').exists())
            self.assertFalse((destination/'files').exists())
            self.assertTrue(list(destination.glob('part-*.tar.gz')))
            self.assertFalse(big.exists())
            self.assertTrue((case/'outcome.json').exists())

    def test_default_materialization_retains_files(self):
        from streams import seal
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source=root/'source'; source.mkdir()
            (source/'a').write_text('original')
            seal(source,root/'sealed',['a'],{})
            materialize(root/'sealed',root/'received')
            self.assertEqual((root/'received/files/a').read_text(),'original')

    def test_failed_pack_retains_original(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ,{'WORKFLOW_RUNTIME_DIR':str(HERE.parent/'workflow')}):
            root=Path(temporary); case=root/'.work/case'; case.mkdir(parents=True)
            (case/'error.txt').write_text('original device error')
            artifacts=pipeline.Pipeline(root)
            def fail(**kwargs): raise OSError('disk full')
            artifacts.seal=fail; artifacts.submit(case)
            self.assertFalse(artifacts.finish())
            self.assertTrue((root/'.incomplete-artifacts').exists())
            self.assertEqual((case/'error.txt').read_text(),'original device error')

if __name__=='__main__': unittest.main()
