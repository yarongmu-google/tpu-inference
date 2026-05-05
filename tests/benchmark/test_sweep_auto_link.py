import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.benchmark.sweep import SpecError, enumerate_combos, load_spec

class TestSweepAutoLink(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        
        self.kernel_data = {
            "results": {
                "decode": [
                    {
                        "tuning_key": {"page_size": 128},
                        "tunable_params": {"bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1, "bkv_csz": 1024},
                        "Latency": 100
                    }
                ],
                "mixed": [
                    {
                        "tuning_key": {"page_size": 128},
                        "tunable_params": {"bq_sz": 64, "bkv_sz": 512, "bq_csz": 64, "bkv_csz": 256},
                        "Latency": 200
                    }
                ],
                "prefill": [
                    {
                        "tuning_key": {"page_size": 128, "chunk_prefill_size": 512},
                        "tunable_params": {"bq_sz": 256, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512},
                        "Latency": 1000
                    },
                    {
                        "tuning_key": {"page_size": 128, "chunk_prefill_size": 1024},
                        "tunable_params": {"bq_sz": 512, "bkv_sz": 2048, "bq_csz": 256, "bkv_csz": 512},
                        "Latency": 2000
                    }
                ]
            }
        }
        self.kernel_file = self.tmp_path / "test.kernel"
        with open(self.kernel_file, "w") as f:
            json.dump(self.kernel_data, f)
            
        self.case_file = self.tmp_path / "test.workload"
        self.case_file.touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_auto_linking(self):
        spec_file = self.tmp_path / "test.service"
        spec_data = {
            "sweep_name": "test",
            "case_file": "test.workload",
            "kernel_registry": "test.kernel",
            "fixed": {
                "BLOCK_SIZE": 128
            },
            "sweep_axes": {
                "LONG_PREFILL_TOKEN_THRESHOLD": [0, 512]
            }
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)
            
        spec = load_spec(spec_file)
        combos = enumerate_combos(spec)
        
        self.assertEqual(len(combos), 2)
        
        c0 = next(c for c in combos if c["LONG_PREFILL_TOKEN_THRESHOLD"] == "0")
        self.assertEqual(c0["RPA_D_BLOCK_SIZES"], "1,4096,1,1024")
        self.assertEqual(c0["RPA_M_BLOCK_SIZES"], "64,512,64,256")
        self.assertNotIn("RPA_P_BLOCK_SIZES", c0)
        
        c512 = next(c for c in combos if c["LONG_PREFILL_TOKEN_THRESHOLD"] == "512")
        self.assertEqual(c512["RPA_P_BLOCK_SIZES"], "256,2048,256,512")
        self.assertEqual(c512["RPA_D_BLOCK_SIZES"], "1,4096,1,1024")

    def test_missing_kernel_file_raises(self):
        spec_file = self.tmp_path / "bad.service"
        with open(spec_file, "w") as f:
            json.dump({
                "sweep_name": "test",
                "case_file": "test.workload",
                "kernel_registry": "does_not_exist.kernel",
                "fixed": {}
            }, f)
            
        with self.assertRaisesRegex(SpecError, "kernel_registry does not exist"):
            load_spec(spec_file)
