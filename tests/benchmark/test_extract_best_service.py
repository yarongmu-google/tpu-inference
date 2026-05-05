import json
import tempfile
import unittest
from pathlib import Path

from tools.benchmark.extract_best_service import export_production_service, _to_float

class TestExtractBestService(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.tmpdir.name)
        self.out_path = self.tmp_path / "production.service"

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_export_accumulates_new_workload(self):
        best_result = {
            "meta": {
                "model": "my_model",
                "tensor_parallel_size": "1",
                "input_len": "2048",
                "output_len": "2048",
                "max_num_seqs": "128",
                "max_num_batched_tokens": "10000"
            },
            "metrics": {
                "RequestThroughput": "10.5"
            }
        }
        
        export_production_service(best_result, self.out_path)
        
        with open(self.out_path) as f:
            data = json.load(f)
            
        self.assertEqual(data["model"], "my_model")
        self.assertEqual(data["tensor_parallel_size"], "1")
        self.assertIn("2048_in_2048_out", data["best_configs_by_workload"])
        self.assertEqual(data["best_configs_by_workload"]["2048_in_2048_out"]["MAX_NUM_SEQS"], "128")
        self.assertEqual(data["best_configs_by_workload"]["2048_in_2048_out"]["metrics"]["RequestThroughput"], "10.5")

    def test_export_overwrites_only_if_better(self):
        # Pre-seed with a good result
        with open(self.out_path, "w") as f:
            json.dump({
                "model": "my_model",
                "best_configs_by_workload": {
                    "1024_in_1024_out": {
                        "MAX_NUM_SEQS": "64",
                        "metrics": {"RequestThroughput": "20.0"}
                    }
                }
            }, f)
            
        # Try to overwrite with a worse result
        worse_result = {
            "meta": {"input_len": "1024", "output_len": "1024", "max_num_seqs": "128"},
            "metrics": {"RequestThroughput": "15.0"}
        }
        export_production_service(worse_result, self.out_path)
        
        with open(self.out_path) as f:
            data = json.load(f)
        self.assertEqual(data["best_configs_by_workload"]["1024_in_1024_out"]["MAX_NUM_SEQS"], "64") # Unchanged
        
        # Try to overwrite with a better result
        better_result = {
            "meta": {"input_len": "1024", "output_len": "1024", "max_num_seqs": "256"},
            "metrics": {"RequestThroughput": "25.0"}
        }
        export_production_service(better_result, self.out_path)
        
        with open(self.out_path) as f:
            data = json.load(f)
        self.assertEqual(data["best_configs_by_workload"]["1024_in_1024_out"]["MAX_NUM_SEQS"], "256") # Updated

