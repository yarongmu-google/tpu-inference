"""Exercise actual old/new kernels in the CPU interpreter and shared harness."""
import ast
from concurrent.futures import Future
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'payload'))
import numpy as np
import worker
from tune import KNOBS

CONFIG={'devices':2,'hidden':128,'experts':8,'intermediate':256,'top_k':1,
        'tokens':64,'seed':17,'warmup':1,'iterations':2,'profile_iterations':0,
        'reference_samples':8,'candidate_timeout_seconds':120,'be':4,'bg':1,
        'capacity':32,'bd1c':128,'bd2c':128,'bcT':0,'variant':'original'}

class EngineTests(unittest.TestCase):
    def test_failed_accuracy_does_not_discard_warmed_samples(self):
        checked=Future(); checked.set_result({'correctness':{'uniform':{'passed':False}},
            'paired_correctness':{},'profiles':{'uniform':{'status':'disabled'}}})
        item={'validation':checked,'config':CONFIG,'control_available':False,
              'timings':{'uniform':{'median_us':5.,'min_us':4.,'samples_us':[4.,6.]}},
              'compile_seconds':1.,'calls':{},'baseline':{'directory':Path('xla'),'record':{}}}
        record=worker.Engine.finish_candidate(None,item)
        self.assertEqual(record['status'],'failed')
        self.assertEqual(record['median_us'],5.)
        self.assertEqual(record['samples_us'],[4.,6.])

    @unittest.skipUnless(len(worker.jax.devices())==2,'Requires two virtual CPU devices')
    def test_actual_pair_reuses_inputs_xla_and_retains_outputs(self):
        # The source under test is supplied explicitly; bypass optional serving
        # package initialization while loading the actual kernel functions.
        import os
        repository=Path(os.environ['KERNEL_SOURCE_ROOT'])
        math_ns={}
        exec(compile((repository/'tpu_inference/kernels/common/math.py').read_text(),'math.py','exec'),math_ns)
        act_source=ast.parse((repository/'tpu_inference/kernels/fused_moe/v1/kernel.py').read_text())
        act=next(n for n in act_source.body if isinstance(n,ast.FunctionDef) and n.name=='apply_act_fn')
        act_ns={'jax':worker.jax}
        exec(compile(ast.Module(body=[act],type_ignores=[]),'activation.py','exec'),act_ns)
        def load(variant):
            path=(HERE/'payload/control_kernel.py' if variant=='original' else
                  repository/'tpu_inference/kernels/fused_moe/v2/decode_kernel_occupied.py')
            tree=ast.parse(path.read_text())
            tree.body=[n for n in tree.body if not(isinstance(n,ast.ImportFrom) and n.module.startswith('tpu_inference'))]
            ns={'__file__':str(path),'apply_act_fn':act_ns['apply_act_fn'],'kmath':SimpleNamespace(**math_ns)}
            exec(compile(tree,str(path),'exec'),ns)
            return SimpleNamespace(**ns)
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); engine=worker.Engine(CONFIG,root,interpret=True)
            try:
                with patch.object(worker,'load_kernel',side_effect=load), \
                     patch.object(worker,'make_weights',wraps=worker.make_weights) as weights, \
                     patch.object(worker,'make_inputs',wraps=worker.make_inputs) as inputs:
                    engine.initialize(CONFIG)
                    baseline_dir=root/'.work/xla'; baseline_dir.mkdir()
                    baseline=engine.baseline(CONFIG,baseline_dir)
                    with patch.object(worker,'reference',side_effect=AssertionError('Reference repeated')):
                        for variant in ('original','occupied'):
                            config={**CONFIG,'variant':variant}; directory=root/'.work'/variant; directory.mkdir()
                            candidate=engine.candidate(config,directory,baseline)
                            record=engine.finish_candidate(candidate)
                            self.assertEqual(len(record['samples_us']),2)
                            self.assertTrue(record['checks']['xla_accuracy'], record['correctness'])
                            self.assertTrue((directory/'outputs.json').exists())
                            if variant=='occupied':
                                self.assertTrue(record['paired_correctness']['uniform']['bitwise_equal'])
                                self.assertTrue(record['paired_correctness']['sparse']['bitwise_equal'])
                            engine.release(candidate)
                    self.assertEqual(weights.call_count,1)
                    self.assertEqual(inputs.call_count,1)
                    self.assertFalse(engine.controls)
                    baseline['persisted'].result()
            finally: engine.close()

    def test_profile_failure_preserves_warmed_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            context={'executable':object(),'inputs':{'uniform':(),'sparse':()}}
            samples={'median_us':7.,'min_us':6.,'samples_us':[6.,8.]}
            with patch.object(worker,'measure',return_value=samples) as measure, \
                 patch.object(worker.jax.profiler,'start_trace',side_effect=RuntimeError('profile unavailable')):
                timings,captures=worker.Engine.timed_profiles(SimpleNamespace(interpret=False),
                    context,{**CONFIG,'profile_iterations':1},directory)
            self.assertEqual(measure.call_count,2)
            self.assertEqual(timings['uniform']['median_us'],7.)
            self.assertEqual(captures['uniform']['status'],'failed')
            self.assertEqual(captures['sparse']['status'],'failed')
            self.assertEqual(json.loads((directory/'timings.json').read_text())['sparse']['median_us'],7.)

    def test_nonfinite_output_is_a_reported_accuracy_failure(self):
        result=worker.check_output(np.array([[np.nan]],np.float32),np.array([[1.]],np.float32),config=CONFIG)
        self.assertFalse(result['passed'])

if __name__=='__main__': unittest.main()
