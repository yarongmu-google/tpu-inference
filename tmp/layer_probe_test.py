"""CPU-only correctness and runner tests for the standalone probe batch.

Run with the same Python used by run_layer_probes.sh. No serving imports,
network access, TPU discovery, checkpoint data or git mutations are needed.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import numpy as np

from layer_probe_kernels import Config, NAMES, make_case, phase_bytes
from run_layer_probes import parse_overrides, static_census, validate_arrays

HERE = Path(__file__).resolve().parent


class ProbeTests(unittest.TestCase):
    def check(self, name: str, config: Config) -> None:
        case = make_case(name, config=config, interpret=True)
        result = jax.jit(case.function)(*case.inputs)
        validate_arrays(case, result)

    def test_all_variants_f32_bf16(self) -> None:
        for dtype in ("float32", "bfloat16"):
            for name in NAMES:
                with self.subTest(name=name, dtype=dtype):
                    self.check(name, Config(dtype=dtype))

    def test_head_views_partial_rows(self) -> None:
        for rows in (1, 7, 17):
            for name in ("head_flat", "head_grouped", "head_naive"):
                with self.subTest(rows=rows, name=name):
                    self.check(name, Config(rows=rows, dtype="bfloat16", heads=12))

    def test_head_row_block_is_a_parameter(self) -> None:
        for block in (4, 32):
            for name in ("head_flat", "head_grouped", "head_naive"):
                with self.subTest(block=block, name=name):
                    self.check(
                        name, Config(rows=17, dtype="bfloat16", head_row_block=block)
                    )

    def test_router_ties_top16_and_tail(self) -> None:
        for pattern in ("ties", "random"):
            for name in ("router_re", "router_er"):
                with self.subTest(pattern=pattern, name=name):
                    self.check(name, Config(rows=7, top=16, pattern=pattern))

    def test_router_orientation_agreement(self) -> None:
        config = Config(rows=31, top=16, experts=896)
        outputs = []
        for name in ("router_re", "router_er"):
            case = make_case(name, config=config, interpret=True)
            outputs.append(jax.jit(case.function)(*case.inputs))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=2e-6)

    def test_gather_capacity_one_and_multiple_panels(self) -> None:
        for pattern in ("random", "contiguous", "skew"):
            for name in ("gather_aligned", "gather_dma", "gather_row"):
                with self.subTest(pattern=pattern, name=name):
                    self.check(
                        name,
                        Config(
                            rows=7,
                            top=4,
                            hit_block=1,
                            width=256,
                            pattern=pattern,
                            dtype="bfloat16",
                        ),
                    )

    def test_gather_incomplete_last_hit_batch(self) -> None:
        for name in ("gather_aligned", "gather_dma", "gather_row"):
            with self.subTest(name=name):
                self.check(
                    name,
                    Config(
                        rows=5,
                        top=3,
                        hit_block=7,
                        width=3584,
                        feature_block=256,
                        dtype="bfloat16",
                    ),
                )

    def test_smem_beta_updates_and_new_call(self) -> None:
        for seed in (17, 39):
            with self.subTest(seed=seed):
                self.check("beta_smem", Config(rows=7, heads=12, steps=3, seed=seed))

    def test_state_orientations_agree(self) -> None:
        config = Config(heads=12)
        outputs = []
        for name in ("state_kv", "state_vk"):
            case = make_case(name, config=config, interpret=True)
            outputs.append(jax.jit(case.function)(*case.inputs))
        np.testing.assert_allclose(
            outputs[0][0],
            np.asarray(outputs[1][0]).swapaxes(-1, -2),
            rtol=2e-5,
            atol=2e-6,
        )
        np.testing.assert_allclose(outputs[0][1], outputs[1][1], rtol=2e-5, atol=2e-6)

    def test_mla_masked_rows_and_tail(self) -> None:
        for offset in (-2, 0, 11):
            for name in ("mla_expanded", "mla_split"):
                with self.subTest(offset=offset, name=name):
                    self.check(
                        name,
                        Config(
                            keys=23,
                            queries=9,
                            key_block=8,
                            query_block=4,
                            query_offset=offset,
                            dtype="bfloat16",
                        ),
                    )

    def test_residency_candidate_accounting(self) -> None:
        config = Config(
            rows=1024,
            model_dim=7168,
            latent_dim=3584,
            hit_block=64,
            output_block=256,
            router_block=128,
            chunk=64,
            core_reserve_bytes=8 * 2**20,
            auxiliary_reserve_bytes=2**20,
        )
        persistent, phases = phase_bytes(config)
        self.assertEqual(persistent + phases[0], 57 * 2**20)
        self.assertEqual(persistent + phases[1], int(60.25 * 2**20))
        self.assertEqual(persistent + phases[2], int(56.4375 * 2**20))
        self.assertGreater(persistent + sum(phases), 64 * 2**20)
        larger_chunk = dataclasses.replace(config, chunk=1024)
        self.assertGreater(phase_bytes(larger_chunk)[1][1], phases[1])

    def test_invalid_configuration_rejected(self) -> None:
        for changes in (
            dict(rows=0),
            dict(top=33),
            dict(feature_block=64),
            dict(width=129),
            dict(dtype="fp4"),
            dict(hit_block=0),
            dict(core_reserve_bytes=-1),
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(Config(), **changes).validate()
        with self.assertRaises(ValueError):
            make_case("missing_case", config=Config(), interpret=True)
        with self.assertRaises(ValueError):
            parse_overrides(["misspelled=32"])

    def test_oracle_catches_corruption(self) -> None:
        case = make_case("head_flat", config=Config(), interpret=True)
        with self.assertRaises(AssertionError):
            validate_arrays(case, np.asarray(case.expected) + 1)
        with self.assertRaises(AssertionError):
            validate_arrays(case, np.full_like(case.expected, np.nan))

    def test_census_is_per_file_and_not_dynamic(self) -> None:
        with tempfile.TemporaryDirectory(prefix="layer-census-test-") as temporary:
            root = Path(temporary)
            (root / "mosaic").mkdir()
            (root / "mosaic" / "one.txt").write_text(
                "scf.for {\n %0 = llo.vld\n scf.for {\n %1 = llo.vld\n }\n}\n"
            )
            (root / "mosaic" / "two.txt").write_text("%0 = llo.vst\n")
            result = static_census(root)
            self.assertEqual(result["files"][0]["llo_text_occurrences"], {"llo.vld": 2})
            self.assertEqual(result["files"][0]["loop_headers"], 2)
            self.assertEqual(result["files"][1]["llo_text_occurrences"], {"llo.vst": 1})
            self.assertIsNone(result["vmem_high_water_bytes"])

    def test_runner_capture_and_cpu_backend_guard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="layer-runner-test-") as temporary:
            for mode, expected_status, exit_code in (
                ("interpret", "passed", 0),
                ("compile", "unavailable", 1),
            ):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(HERE / "run_layer_probes.py"),
                        "--cases",
                        "beta_smem",
                        "--mode",
                        mode,
                        "--output-parent",
                        temporary,
                        "--timeout",
                        "30",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=60,
                )
                self.assertEqual(completed.returncode, exit_code, completed.stderr)
                outputs = sorted(
                    Path(temporary).glob("layer-probes-*/summary.json"),
                    key=lambda p: p.stat().st_mtime_ns,
                )
                summary = json.loads(outputs[-1].read_text())
                self.assertEqual(summary["results"][0]["status"], expected_status)
                self.assertFalse(summary["interpret_is_backend_validation"])
                self.assertEqual(
                    len(list(Path(temporary).glob("*.tar.gz"))), len(outputs)
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
