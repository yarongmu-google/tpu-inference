"""Local semantic, ring-ownership, campaign and failure-reporting tests."""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np

from pipeline_probe_campaign import planned_jobs, timing_fits
from pipeline_probe_kernels import NAMES, PipelineConfig, blocked_chart, make_case
from run_layer_probes import capability_rejection, validate_arrays


class PipelineTests(unittest.TestCase):
    def check(self, name: str, config: PipelineConfig) -> None:
        case = make_case(name, config=config, interpret=True)
        validate_arrays(case, jax.jit(case.function)(*case.inputs))
        self.assertIn("Blocked operations", blocked_chart(case, config))

    def test_ring_ownership(self) -> None:
        for depth in (1, 2, 3, 4, 8, 16, 32, 64):
            for unroll in (1, 2, 4, 8):
                for n in sorted(
                    {0, 1, 2, 7, 17, depth - 1, depth, depth + 1, 2 * depth + 3}
                ):
                    slots = [None] * depth
                    launched, consumed = [], []

                    def start(j: int) -> None:
                        self.assertIsNone(slots[j % depth])
                        slots[j % depth] = j
                        launched.append(j)

                    for p in range(depth - 1):
                        if p < n:
                            start(p)
                    for base in range(0, n, unroll):
                        for u in range(unroll):
                            j = base + u
                            if j >= n:
                                continue
                            future = j + depth - 1
                            if future < n:
                                start(future)
                            self.assertEqual(slots[j % depth], j)
                            slots[j % depth] = None
                            consumed.append(j)
                    self.assertEqual(launched, list(range(n)))
                    self.assertEqual(consumed, list(range(n)))
                    self.assertTrue(all(x is None for x in slots))

    def test_all_interpretable_cases(self) -> None:
        for dtype in ("float32", "bfloat16"):
            for name in NAMES:
                if name in ("dma_add", "stream_emitted"):
                    continue  # Explicit interpreter capability records tested below.
                with self.subTest(dtype=dtype, case=name):
                    self.check(name, PipelineConfig(jobs=3, depth=2, dtype=dtype))

    def test_stream_directions_and_tails(self) -> None:
        for direction in ("read", "write", "local", "mixed"):
            for n, d, u in ((1, 4, 2), (5, 3, 4), (3, 1, 2)):
                with self.subTest(direction=direction, n=n, d=d):
                    self.check(
                        "stream_manual",
                        PipelineConfig(jobs=n, depth=d, unroll=u, direction=direction),
                    )

    def test_dispatch_repeated_tokens(self) -> None:
        for name in (
            "dispatch_row_serial",
            "dispatch_row_pipeline",
            "dispatch_parent_pipeline",
        ):
            for dtype in ("float32", "bfloat16"):
                self.check(
                    name,
                    PipelineConfig(
                        jobs=5,
                        hits=3,
                        depth=4,
                        unroll=3,
                        rows=17,
                        pattern="skew",
                        dtype=dtype,
                    ),
                )

    def test_mla_depth_and_tail(self) -> None:
        for name in ("mla_expanded_pipeline", "mla_split_pipeline"):
            for n, depth in ((1, 4), (3, 1), (5, 3)):
                self.check(name, PipelineConfig(jobs=n, depth=depth, dtype="bfloat16"))

    def test_compute_padding_and_unroll(self) -> None:
        for unroll in (1, 2, 4, 8):
            self.check(
                "compute_interleaved",
                PipelineConfig(jobs=5, unroll=unroll, block_rows=8),
            )

    def test_campaign_coverage_and_validation(self) -> None:
        args = SimpleNamespace(set=[], sweep="standard", cases="all")
        jobs = planned_jobs(args)
        self.assertLessEqual(len(jobs), 256)
        self.assertTrue(set(NAMES) <= {j["case"] for j in jobs})
        self.assertTrue(
            {"phase_scoped", "phase_colive", "beta_smem"} <= {j["case"] for j in jobs}
        )
        with self.assertRaises(ValueError):
            planned_jobs(SimpleNamespace(set=["depth=0"], sweep="none", cases="all"))

    def test_worker_capability_records(self) -> None:
        runner = Path(__file__).with_name("run_layer_probes.py")
        for name in ("dma_add", "stream_emitted", "broadcast_smem_vector"):
            with tempfile.TemporaryDirectory(prefix="pipeline-worker-test-") as root:
                job = Path(root) / "job.json"
                job.write_text(
                    json.dumps(
                        dict(
                            case=name,
                            family="pipeline",
                            mode="interpret",
                            repeats=1,
                            profile=False,
                            config=dataclasses.asdict(PipelineConfig()),
                        )
                    )
                )
                p = subprocess.run(
                    [sys.executable, "-B", str(runner), "--worker", str(job)],
                    capture_output=True,
                    text=True,
                    timeout=90,
                    check=False,
                )
                self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads((job.parent / "result.json").read_text())
                self.assertFalse(result["backend_compile_validated"])
                if name != "broadcast_smem_vector":
                    self.assertEqual(result["status"], "unsupported_interpreter")
                else:
                    self.assertEqual(result["status"], "passed")

    def test_noise_never_auto_calibrates(self) -> None:
        records = [
            dict(
                case="stream_manual",
                status="passed",
                config={"jobs": n, "depth": 2},
                host_call_us={"min": 100 + n * 0.01, "p90_minus_p10": 10},
            )
            for n in (1, 2, 3)
        ]
        fit = timing_fits(records)[0]
        self.assertFalse(fit["span_sufficient"])
        self.assertFalse(fit["accepted_calibration"])

    def test_unknown_failure_is_not_capability_evidence(self) -> None:
        self.assertFalse(
            capability_rejection("dma_add", "Not enough memory compiling add=True")
        )
        self.assertFalse(
            capability_rejection("stream_manual", "DMA with add=True is not supported.")
        )
        self.assertTrue(
            capability_rejection("dma_add", "DMA with add=True is not supported.")
        )

    def test_oracle_rejects_corruption(self) -> None:
        case = make_case("state_overlap", config=PipelineConfig(jobs=2), interpret=True)
        bad = tuple(np.zeros_like(x) for x in case.expected)
        with self.assertRaises(AssertionError):
            validate_arrays(case, bad)


if __name__ == "__main__":
    unittest.main()
