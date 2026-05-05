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

    def test_missing_prefill_entry_for_swept_K_raises(self):
        # Registry has only K in {512, 1024} for PREFILL (per setUp). Spec
        # sweeps K=2048 — no entry. Auto-link must raise rather than
        # silently leave RPA_P_BLOCK_SIZES unset and let the kernel fall
        # back to (often-OOM) defaults.
        spec_file = self.tmp_path / "test.service"
        spec_data = {
            "sweep_name": "test",
            "case_file": "test.workload",
            "kernel_registry": "test.kernel",
            "fixed": {"BLOCK_SIZE": 128},
            "sweep_axes": {"LONG_PREFILL_TOKEN_THRESHOLD": [2048]},
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        spec = load_spec(spec_file)
        with self.assertRaisesRegex(SpecError,
                                    r"no PREFILL entry at .*K=2048"):
            enumerate_combos(spec)

    def test_missing_decode_entry_at_page_size_raises(self):
        # Registry has DECODE/MIXED/PREFILL only at page_size=128. Spec
        # uses BLOCK_SIZE=64 — no decode entry there. Hard fail.
        spec_file = self.tmp_path / "test.service"
        spec_data = {
            "sweep_name": "test",
            "case_file": "test.workload",
            "kernel_registry": "test.kernel",
            "fixed": {"BLOCK_SIZE": 64},
            "sweep_axes": {"LONG_PREFILL_TOKEN_THRESHOLD": [0]},
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        spec = load_spec(spec_file)
        with self.assertRaisesRegex(SpecError,
                                    r"no DECODE entry at page_size=64"):
            enumerate_combos(spec)

    def test_user_provided_block_sizes_skip_registry_check(self):
        # User-pinned RPA_*_BLOCK_SIZES in fixed should bypass the
        # registry lookup entirely — registry can be missing the
        # corresponding entry and enumeration still succeeds. This is
        # the explicit-override path.
        spec_file = self.tmp_path / "test.service"
        spec_data = {
            "sweep_name": "test",
            "case_file": "test.workload",
            "kernel_registry": "test.kernel",
            "fixed": {
                "BLOCK_SIZE": 64,  # registry has no page=64 entries
                "RPA_D_BLOCK_SIZES": "1,2048,1,256",
                "RPA_M_BLOCK_SIZES": "32,1024,32,128",
            },
            "sweep_axes": {"LONG_PREFILL_TOKEN_THRESHOLD": [0]},
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        spec = load_spec(spec_file)
        combos = enumerate_combos(spec)
        # Should NOT raise; user override wins.
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["RPA_D_BLOCK_SIZES"], "1,2048,1,256")
        self.assertEqual(combos[0]["RPA_M_BLOCK_SIZES"], "32,1024,32,128")
