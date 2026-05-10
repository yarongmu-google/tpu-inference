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

    # ---- Decoupled-K (LOGICAL) auto-link tests ----

    def test_logical_auto_link_via_rpa_kernel_k(self):
        # When RPA_KERNEL_K is set, the auto-link should pull
        # RPA_P_BLOCK_SIZES from the LOGICAL winner (NOT the PREFILL
        # winner) and additionally set RPA_MAX_NUM_SUBSEQS from the
        # LOGICAL winners tunable_params.
        kernel_data = {
            "results": {
                "decode": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1,
                        "bkv_csz": 1024,
                    },
                }],
                "mixed": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 64, "bkv_sz": 512, "bq_csz": 64,
                        "bkv_csz": 256,
                    },
                }],
                "logical": [{
                    "tuning_key": {
                        "page_size": 128, "chunk_prefill_size": 128,
                    },
                    "tunable_params": {
                        "bq_sz": 128, "bkv_sz": 1024, "bq_csz": 128,
                        "bkv_csz": 512, "max_num_subseqs": 256,
                    },
                }],
            }
        }
        kernel_file = self.tmp_path / "logical.kernel"
        with open(kernel_file, "w") as f:
            json.dump(kernel_data, f)

        spec_file = self.tmp_path / "logical.service"
        spec_data = {
            "sweep_name": "test",
            "case_file": "test.workload",
            "kernel_registry": "logical.kernel",
            "fixed": {
                "BLOCK_SIZE": 128,
                "RPA_KERNEL_K": 128,    # decoupled-K active
            },
        }
        with open(spec_file, "w") as f:
            json.dump(spec_data, f)

        spec = load_spec(spec_file)
        combos = enumerate_combos(spec)
        self.assertEqual(len(combos), 1)
        c = combos[0]
        # P_BLOCK_SIZES from LOGICAL winner (NOT prefill — there is no
        # prefill entry in this registry, and the lookup must avoid it).
        self.assertEqual(c["RPA_P_BLOCK_SIZES"], "128,1024,128,512")
        # max_num_subseqs auto-linked from LOGICAL winner.
        self.assertEqual(c["RPA_MAX_NUM_SUBSEQS"], "256")
        # LPTT was not pinned by the spec; sweep derives it from
        # the resolved mnss * kernel_K. The server (tpu_runner.py) will
        # enforce the same invariant at init for non-sweep launches.
        self.assertEqual(c["LONG_PREFILL_TOKEN_THRESHOLD"],
                         str(256 * 128))

    def test_logical_missing_max_num_subseqs_raises(self):
        # A LOGICAL entry that lacks `max_num_subseqs` in its
        # tunable_params (e.g. produced by a pre-A.1 tuner) is invalid
        # for runtime use — the runner needs the tuned value to size
        # its prefetch arrays. Auto-link must raise so the misconfig is
        # caught at spec-load.
        kernel_data = {
            "results": {
                "decode": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1,
                        "bkv_csz": 1024,
                    },
                }],
                "mixed": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 64, "bkv_sz": 512, "bq_csz": 64,
                        "bkv_csz": 256,
                    },
                }],
                "logical": [{
                    "tuning_key": {
                        "page_size": 128, "chunk_prefill_size": 128,
                    },
                    "tunable_params": {
                        "bq_sz": 128, "bkv_sz": 1024, "bq_csz": 128,
                        "bkv_csz": 512,
                        # No max_num_subseqs — pre-A.1 tuner output.
                    },
                }],
            }
        }
        kernel_file = self.tmp_path / "stale_logical.kernel"
        with open(kernel_file, "w") as f:
            json.dump(kernel_data, f)
        spec_file = self.tmp_path / "stale.service"
        with open(spec_file, "w") as f:
            json.dump({
                "sweep_name": "test",
                "case_file": "test.workload",
                "kernel_registry": "stale_logical.kernel",
                "fixed": {
                    "BLOCK_SIZE": 128,
                    "RPA_KERNEL_K": 128,
                },
            }, f)
        spec = load_spec(spec_file)
        with self.assertRaisesRegex(SpecError,
                                    "missing max_num_subseqs"):
            enumerate_combos(spec)

    def test_logical_missing_entry_at_kernel_k_raises(self):
        # Spec sets RPA_KERNEL_K=64, but registry only has a LOGICAL
        # entry at K=128. Auto-link must raise rather than silently
        # fall through.
        kernel_data = {
            "results": {
                "decode": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1,
                        "bkv_csz": 1024,
                    },
                }],
                "mixed": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 64, "bkv_sz": 512, "bq_csz": 64,
                        "bkv_csz": 256,
                    },
                }],
                "logical": [{
                    "tuning_key": {
                        "page_size": 128, "chunk_prefill_size": 128,
                    },
                    "tunable_params": {
                        "bq_sz": 128, "bkv_sz": 1024, "bq_csz": 128,
                        "bkv_csz": 512, "max_num_subseqs": 256,
                    },
                }],
            }
        }
        kernel_file = self.tmp_path / "k128_only.kernel"
        with open(kernel_file, "w") as f:
            json.dump(kernel_data, f)
        spec_file = self.tmp_path / "k64.service"
        with open(spec_file, "w") as f:
            json.dump({
                "sweep_name": "test",
                "case_file": "test.workload",
                "kernel_registry": "k128_only.kernel",
                "fixed": {
                    "BLOCK_SIZE": 128,
                    "RPA_KERNEL_K": 64,
                },
            }, f)
        spec = load_spec(spec_file)
        with self.assertRaisesRegex(SpecError,
                                    r"no LOGICAL entry at .*K_kernel=64"):
            enumerate_combos(spec)

    def test_user_overrides_skip_logical_lookup(self):
        # User pins RPA_P_BLOCK_SIZES and RPA_MAX_NUM_SUBSEQS manually;
        # the LOGICAL lookup should be skipped (registry can be empty).
        kernel_data = {
            "results": {
                "decode": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 1, "bkv_sz": 4096, "bq_csz": 1,
                        "bkv_csz": 1024,
                    },
                }],
                "mixed": [{
                    "tuning_key": {"page_size": 128},
                    "tunable_params": {
                        "bq_sz": 64, "bkv_sz": 512, "bq_csz": 64,
                        "bkv_csz": 256,
                    },
                }],
                # No logical entry — would normally cause the lookup
                # to raise.
            }
        }
        kernel_file = self.tmp_path / "no_logical.kernel"
        with open(kernel_file, "w") as f:
            json.dump(kernel_data, f)
        spec_file = self.tmp_path / "manual.service"
        with open(spec_file, "w") as f:
            json.dump({
                "sweep_name": "test",
                "case_file": "test.workload",
                "kernel_registry": "no_logical.kernel",
                "fixed": {
                    "BLOCK_SIZE": 128,
                    "RPA_KERNEL_K": 128,
                    "RPA_P_BLOCK_SIZES": "128,1024,128,512",
                    "RPA_MAX_NUM_SUBSEQS": "512",
                },
            }, f)
        spec = load_spec(spec_file)
        combos = enumerate_combos(spec)   # must not raise
        self.assertEqual(len(combos), 1)
        self.assertEqual(combos[0]["RPA_P_BLOCK_SIZES"],
                         "128,1024,128,512")
        self.assertEqual(combos[0]["RPA_MAX_NUM_SUBSEQS"], "512")
