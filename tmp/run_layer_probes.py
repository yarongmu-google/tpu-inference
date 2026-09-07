"""Bounded subprocess runner for synthetic layer layout/movement probes.

Default mode is CPU interpretation. TPU modes require a locally attached
TPU and never connect to or provision a remote machine. Each run creates a
new result directory; existing logs, compiler dumps and git state are kept.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from typing import Any

HERE = Path(__file__).resolve().parent


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def source_hashes() -> dict[str, str]:
    names = (
        "layer_probe_kernels.py",
        "run_layer_probes.py",
        "run_layer_probes.sh",
        "layer_probe_test.py",
        "pipeline_probe_kernels.py",
        "pipeline_probe_campaign.py",
        "pipeline_probe_test.py",
    )
    return {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        for name in names
        if (HERE / name).exists()
    }


def versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for package in ("jax", "jaxlib", "libtpu", "numpy", "pytest"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def static_census(directory: Path) -> dict[str, Any]:
    """Textual static census only: no loop-trip or executed-cycle inference."""
    from collections import Counter

    files = []
    for path in sorted(directory.rglob("*.txt")):
        if "mosaic" not in path.parts and "hlo" not in path.parts:
            continue
        content = path.read_text(errors="replace")
        operations = Counter(re.findall(r"\b(llo\.[A-Za-z_0-9.]+)\b", content))
        files.append(
            {
                "path": str(path.relative_to(directory)),
                "llo_text_occurrences": dict(sorted(operations.items())),
                "copy_mentions": len(re.findall(r"\bcopy\(", content)),
                "relayout_mentions": len(re.findall(r"\brelayout\b", content)),
                "loop_headers": len(re.findall(r"\bscf\.(?:for|while)\b", content)),
            }
        )
    return {
        "files": files,
        "qualification": "per-file textual occurrences, not dynamic instruction counts; inspect loop bodies and layouts",
        "vmem_high_water_bytes": None,
        "allocation_review": "inspect backend dump; executable temp_size is not automatically VMEM",
    }


def validate_arrays(case: Any, result: Any) -> dict[str, Any]:
    import jax
    import numpy as np

    actual = jax.tree.leaves(result)
    expected = jax.tree.leaves(case.expected)
    if len(actual) != len(expected):
        raise AssertionError("output arity differs from oracle")
    errors = []
    for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
        got, want = np.asarray(got), np.asarray(want)
        if got.shape != want.shape:
            raise AssertionError(f"output {index} shape: {got.shape} != {want.shape}")
        if np.issubdtype(want.dtype, np.integer):
            np.testing.assert_array_equal(got, want)
        else:
            np.testing.assert_allclose(
                got.astype(np.float32),
                want.astype(np.float32),
                rtol=case.rtol,
                atol=case.atol,
                equal_nan=False,
            )
        errors.append(
            float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))))
        )
    return {
        "max_abs_per_output": errors,
        "rtol": case.rtol,
        "atol": case.atol,
        "integer_outputs": "exact",
        "golden_scope": "synthetic probe, not checkpoint or whole layer",
    }


def blocked_operations(name: str) -> str:
    """Readable symbolic steps accompany concrete boundary shapes and jaxpr."""
    steps = {
        "head": [
            (
                "reshape/transpose candidate",
                "[R,H*128] -> flat/grouped/naive",
                "wrapper; inspect HLO for copy",
            ),
            (
                "load owned head rows",
                "[row_block,128]",
                "VMEM -> VReg; assigned tiling pending",
            ),
            (
                "2*x + head_id; store",
                "[row_block,128] -> [H,R,128]",
                "VReg -> output window",
            ),
        ],
        "state": [
            (
                "exp(g) * S",
                "[128] and [128,128]",
                "owned state tile; no intermediate HBM store",
            ),
            (
                "beta * (v - reduce_K(k*S))",
                "[128,128] -> delta[128]",
                "scalar-prefetched beta; K-axis reduction",
            ),
            ("S += outer(k,delta)", "[128,128]", "orientation-specific broadcasts"),
            (
                "o = reduce_K(q*S); store S,o",
                "state[128,128], output[1,128]",
                "persistent outputs",
            ),
        ],
        "beta": [
            ("sigmoid(logits)", "[R,H] f32", "VMEM producer staging"),
            ("async_copy; wait", "[R,H] f32", "VMEM -> retained SMEM"),
            (
                "scalar beta[r,h] -> broadcast; accumulate",
                "scalar -> [8,128]",
                "SMEM -> vector arithmetic",
            ),
            (
                "store checksum; next grid step replaces beta",
                "[H,8,128]",
                "output per step; not cross-call state",
            ),
        ],
        "router": [
            (
                "sigmoid(logits) + selection bias",
                "[R,E] or [E,R]",
                "vector work; no router GEMM in probe",
            ),
            (
                "max, matching-id minimum, mask winner; repeat top times",
                "expert axis -> ids[R]",
                "deterministic probe tie rule",
            ),
            (
                "gather ORIGINAL scores; renormalize",
                "weights[top,R]",
                "f32; selection bias is not a mixing weight",
            ),
            (
                "store ids and weights",
                "[top,R]",
                "output windows; SMEM list building separate",
            ),
        ],
        "gather": [
            ("prefetch ids/weights", "[padded_hits]", "SMEM; prebuilt list"),
            (
                "gather valid hit batches",
                "[R,z_block] -> [M_padded,z_block]",
                "aligned parent selection, local DMA or row-oriented source",
            ),
            (
                "matrix-native handoff; cast(1.125*x)",
                "[M_padded,z_block]",
                "VMEM; explicit synthetic expert-output rounding",
            ),
            (
                "weighted combine by original token",
                "[M_padded,z_block] -> [R,z_block] f32",
                "serialized RMW; duplicate tokens accumulate; no dropped hits",
            ),
        ],
        "phase": [
            (
                "initialize persistent buffer from input",
                "persistent_shape",
                "VMEM reservation; shape recorded in contract",
            ),
            (
                "fill every chunk in each phase buffer",
                "phase_shapes, chunks[8,128] f32",
                "scoped sequential lifetimes or co-live control",
            ),
            (
                "read all chunks and combine",
                "[8,128] accumulator",
                "observable input-dependent output; inspect backend DCE",
            ),
            (
                "consume persistent buffer and store",
                "output[8,128]",
                "allocation skeleton, not decoder arithmetic",
            ),
        ],
        "mla": [
            (
                "DMA key/value block; wait",
                "K[J,192] or Kp[J,128]+Kr[J,64]; V[J,128]",
                "HBM -> VMEM",
            ),
            (
                "Q @ K.T / sqrt(192); causal/tail mask",
                "Q[Qb,192] -> scores[Qb,J] f32",
                "MXU -> softmax layout; split contraction is a comparator",
            ),
            (
                "update running maximum/normalizer; exp",
                "m,l[Qb], p[Qb,J]",
                "vector reductions and broadcasts",
            ),
            (
                "rescale accumulator; cast(p) @ V",
                "acc[Qb,128] f32",
                "explicit probability operand rounding",
            ),
            (
                "normalize; mask invalid queries; store",
                "[Qb,128]",
                "expanded source values, no cache absorption",
            ),
        ],
    }
    family = name.split("_", 1)[0]
    lines = [
        "",
        "## Blocked operations",
        "",
        "R denotes padded resident rows; M/J/Qb and inner panels are configuration variables.",
        "This is source-level dataflow, not a fabricated emitted-instruction chart.",
        "",
        "| Operation | Shape | Storage / movement |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {op} | {shape} | {storage} |" for op, shape, storage in steps[family]
    )
    return "\n".join(lines) + "\n"


def capability_rejection(case: str, message: str) -> bool:
    """Recognize only known capability messages, never generic compile failures."""
    if case == "dma_add":
        return any(
            text in message
            for text in (
                "DMA with add=True is not supported.",
                "DMA partial discharge add=True not yet implemented.",
            )
        )
    return (
        case == "broadcast_smem_vector" and "Can only load scalars from SMEM" in message
    )


def worker(job_path: Path) -> int:
    job = json.loads(job_path.read_text())
    directory = job_path.parent
    result: dict[str, Any] = {
        "case": job["case"],
        "mode": job["mode"],
        "config": job["config"],
        "versions": versions(),
        "source_sha256": source_hashes(),
        "status": "failed",
        "backend_compile_validated": False,
        "device_time_us": None,
        "vmem_high_water_bytes": None,
        "phase": "environment",
    }
    mode = job["mode"]
    if mode == "interpret":
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        # Set before importing JAX/libtpu. Only these output flags are added;
        # no environment dump, remote operations or destructive cleanup.
        (directory / "hlo").mkdir()
        (directory / "mosaic").mkdir()
        os.environ["XLA_FLAGS"] = (
            os.environ.get("XLA_FLAGS", "")
            + f" --xla_dump_to={directory / 'hlo'} --xla_dump_hlo_as_text"
        )
        os.environ["LIBTPU_INIT_ARGS"] = (
            os.environ.get("LIBTPU_INIT_ARGS", "")
            + f" --xla_mosaic_dump_to={directory / 'mosaic'}"
        )
    try:
        import jax
        import numpy as np

        if job.get("family") == "pipeline":
            from pipeline_probe_kernels import PipelineConfig as Config, make_case
        else:
            from layer_probe_kernels import Config, make_case

        import inspect
        from jax._src.pallas.mosaic import lowering, pipeline

        result["installed_source_sha256"] = {
            module.__name__: hashlib.sha256(
                Path(inspect.getfile(module)).read_bytes()
            ).hexdigest()
            for module in (lowering, pipeline)
        }

        devices = jax.devices()
        result["devices"] = [
            {"platform": d.platform, "kind": d.device_kind, "id": d.id} for d in devices
        ]
        if mode != "interpret" and not any(d.platform == "tpu" for d in devices):
            result["status"] = "unavailable"
            result["reason"] = (
                "No local TPU backend; no CPU fallback or remote connection attempted"
            )
            write_json(directory / "result.json", result)
            return 3
        result["phase"] = "case_build"
        config = Config(**job["config"])
        case = make_case(job["case"], config=config, interpret=(mode == "interpret"))
        result["metadata"] = case.metadata
        result["inputs"] = [
            {"shape": list(x.shape), "dtype": str(x.dtype)} for x in case.inputs
        ]
        result["expected_outputs"] = [
            {"shape": list(x.shape), "dtype": str(x.dtype)}
            for x in jax.tree.leaves(case.expected)
        ]
        chart = [
            f"# {case.name}: source-to-blocked probe record",
            "",
            "Synthetic probe only. Actual physical layout and emitted operation",
            "charts require compiler-dump inspection; shapes alone are not proof.",
            "",
            "## Configuration",
            "",
            "```json",
            json.dumps(job["config"], indent=2),
            "```",
            "",
            "## Tensor boundaries",
            "",
            "```json",
            json.dumps(
                {
                    "inputs": result["inputs"],
                    "outputs": result["expected_outputs"],
                    "contract": case.metadata,
                },
                indent=2,
            ),
            "```",
            "",
            "Pallas body/dataflow: jaxpr.txt. Backend HLO/Mosaic dumps, when",
            "available, are captured separately. Census counts are static text",
            "occurrences, not dynamic instructions or measured resource rates.",
        ]
        if job.get("family") == "pipeline":
            from pipeline_probe_kernels import blocked_chart

            operations = blocked_chart(case, config)
        else:
            operations = blocked_operations(case.name)
        (directory / "source_to_blocked.md").write_text(
            "\n".join(chart) + "\n" + operations
        )
        result["phase"] = "trace"
        (directory / "jaxpr.txt").write_text(
            str(jax.make_jaxpr(case.function)(*case.inputs))
        )
        result["phase"] = "lower"
        lowered = jax.jit(case.function).lower(*case.inputs)
        (directory / "lowered.mlir").write_text(lowered.as_text())
        result["phase"] = "compile"
        executable = lowered.compile()
        if mode != "interpret":
            result["backend_compile_validated"] = True
            (directory / "hlo" / "executable.txt").write_text(executable.as_text())
            memory = executable.memory_analysis()
            result["executable_memory_analysis"] = {
                field: getattr(memory, field, None)
                for field in (
                    "argument_size_in_bytes",
                    "output_size_in_bytes",
                    "alias_size_in_bytes",
                    "temp_size_in_bytes",
                    "generated_code_size_in_bytes",
                )
            }
            result["memory_analysis_qualification"] = (
                "compiler executable memory report, NOT a VMEM high-water certificate"
            )
            result["assigned_input_formats"] = str(
                getattr(executable, "input_formats", None)
            )
            result["assigned_output_formats"] = str(
                getattr(executable, "output_formats", None)
            )
        if mode == "compile":
            result["status"] = "compiled"
            result["correctness"] = "not executed in compile-only mode"
        else:
            result["phase"] = "execute"
            got = jax.block_until_ready(executable(*case.inputs))
            result["correctness"] = validate_arrays(case, got)
            result["status"] = "passed"
            if mode == "bench":
                samples = []
                for _ in range(job["repeats"]):
                    start = time.perf_counter_ns()
                    jax.block_until_ready(executable(*case.inputs))
                    samples.append((time.perf_counter_ns() - start) / 1000)
                result["host_call_us"] = {
                    "samples": samples,
                    "min": min(samples),
                    "median": float(np.median(samples)),
                    "p90_minus_p10": float(
                        np.percentile(samples, 90) - np.percentile(samples, 10)
                    ),
                }
                result["timing_qualification"] = (
                    "host-observed whole-call latency; no unit-rate slope or device-only latency inferred"
                )
                if job["profile"]:
                    try:
                        with jax.profiler.trace(
                            str(directory / "profile"), create_perfetto_link=False
                        ):
                            for _ in range(3):
                                jax.block_until_ready(executable(*case.inputs))
                        result["profile"] = (
                            "captured locally; inspect device spans before quoting device time"
                        )
                    except Exception:
                        result["profile"] = "failed"
                        (directory / "profile_error.txt").write_text(
                            traceback.format_exc()
                        )
                        result["status"] = "failed"
                        result["reason"] = "requested profiler capture failed"
    except Exception as exc:
        result["status"] = "failed"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        (directory / "error.txt").write_text(traceback.format_exc())
        message = str(exc)
        known_rejection = capability_rejection(job["case"], message)
        if (
            result["phase"] in ("case_build", "trace", "lower", "compile")
            and known_rejection
        ):
            result["status"] = (
                "unsupported_interpreter" if mode == "interpret" else "unsupported"
            )
            result["capability_supported"] = False
            result["capability_scope"] = (
                "CPU interpreter" if mode == "interpret" else "TPU-target compilation"
            )
        elif (
            mode == "interpret"
            and job["case"] == "stream_emitted"
            and "Unsupported TPU device kind: cpu" in message
        ):
            result["status"] = "unsupported_interpreter"
        result["rejection_is_performance_pass"] = False
    result["census"] = static_census(directory)
    write_json(directory / "result.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return (
        0
        if result["status"]
        in ("passed", "compiled", "unsupported", "unsupported_interpreter")
        else 1
    )


def parse_overrides(values: list[str]) -> dict[str, Any]:
    from layer_probe_kernels import Config

    fields = {f.name for f in dataclasses.fields(Config)}
    result = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or key not in fields:
            raise ValueError(f"unknown or malformed --set: {item}")
        try:
            result[key] = json.loads(value)
        except json.JSONDecodeError:
            result[key] = value
    return result


def run_parent(args: argparse.Namespace) -> int:
    if args.suite == "pipeline":
        from pipeline_probe_campaign import run_campaign

        return run_campaign(args)
    from layer_probe_kernels import Config, NAMES

    overrides = parse_overrides(args.set)
    if args.suite == "target":
        defaults = dict(
            heads=12,
            width=3584,
            experts=896,
            top=16,
            hit_block=64,
            query_block=16,
            key_block=128,
            queries=31,
            keys=257,
            query_offset=226,
            model_dim=7168,
            latent_dim=3584,
            output_block=256,
            router_block=128,
            chunk=64,
            dtype="bfloat16",
            core_reserve_bytes=8 * 2**20,
            auxiliary_reserve_bytes=2**20,
        )
    else:
        defaults = {}
    defaults.update(overrides)
    names = list(NAMES) if args.cases == "all" else args.cases.split(",")
    if any(name not in NAMES for name in names):
        raise ValueError("unknown case name")
    rows = [int(row) for row in args.rows.split(",")]
    if not 0 < len(rows) * len(names) <= 128:
        raise ValueError("batch must contain 1..128 cases")
    if (
        not 1 <= args.repeats <= 200
        or not 1 <= args.timeout <= 1800
        or not 1 <= args.budget <= 14400
    ):
        raise ValueError("repeats/timeout/budget out of bounded range")
    configs = [Config(**{**defaults, "rows": row}) for row in rows]
    for config in configs:
        config.validate()
    if args.output_parent:
        parent = Path(args.output_parent).resolve()
        if not parent.is_dir():
            raise ValueError("--output-parent must be an existing directory")
    else:
        parent = None
    directory = Path(tempfile.mkdtemp(prefix="layer-probes-", dir=parent))
    manifest = {
        "suite": args.suite,
        "mode": args.mode,
        "source_sha256": source_hashes(),
        "versions": versions(),
        "case_names": names,
        "configs": [dataclasses.asdict(c) for c in configs],
        "budget_seconds": args.budget,
        "timeout_seconds": args.timeout,
        "scope": "local execution only; no network, staging, commit or push",
    }
    write_json(directory / "manifest.json", manifest)
    print(f"RESULT_DIR={directory}", flush=True)
    results = []
    deadline = time.monotonic() + args.budget
    with (directory / "runner.log").open("w") as log:
        for index, (config, name) in enumerate((c, n) for c in configs for n in names):
            child = directory / f"{index:03d}-{name}-r{config.rows}"
            child.mkdir()
            job = {
                "case": name,
                "config": dataclasses.asdict(config),
                "mode": args.mode,
                "repeats": args.repeats,
                "profile": args.profile,
            }
            write_json(child / "job.json", job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {
                    "case": name,
                    "status": "not_run",
                    "reason": "batch time budget exhausted",
                }
                write_json(child / "result.json", result)
            else:
                start = time.monotonic()
                try:
                    completed = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--worker",
                            str(child / "job.json"),
                        ],
                        capture_output=True,
                        text=True,
                        timeout=min(args.timeout, remaining),
                        check=False,
                    )
                    (child / "stdout.log").write_text(completed.stdout)
                    (child / "stderr.log").write_text(completed.stderr)
                    if (child / "result.json").exists():
                        result = json.loads((child / "result.json").read_text())
                    else:
                        result = {
                            "case": name,
                            "status": "failed",
                            "reason": "worker exited without result",
                        }
                    result["exit_code"] = completed.returncode
                    if completed.returncode != 0 and result["status"] in (
                        "passed",
                        "compiled",
                    ):
                        result["status"] = "failed"
                        result["reason"] = "worker nonzero exit despite success record"
                except subprocess.TimeoutExpired as exc:

                    def decoded(value: Any) -> str:
                        return (
                            value.decode(errors="replace")
                            if isinstance(value, bytes)
                            else (value or "")
                        )

                    (child / "stdout.log").write_text(decoded(exc.stdout))
                    (child / "stderr.log").write_text(decoded(exc.stderr))
                    result = {
                        "case": name,
                        "status": "timeout",
                        "reason": "worker killed at bounded timeout",
                    }
                result["elapsed_seconds"] = time.monotonic() - start
                write_json(child / "result.json", result)
            results.append(
                {
                    "case": name,
                    "rows": config.rows,
                    "status": result["status"],
                    "result": str((child / "result.json").relative_to(directory)),
                    "reason": result.get("reason"),
                }
            )
            line = f"{index:03d} {name} R={config.rows}: {result['status']}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    passed = all(result["status"] in ("passed", "compiled") for result in results)
    summary = {
        "complete_without_failures": passed,
        "mode": args.mode,
        "interpret_is_backend_validation": False,
        "results": results,
    }
    write_json(directory / "summary.json", summary)
    archive = directory.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    print(f"SUMMARY={directory / 'summary.json'}\nARCHIVE={archive}", flush=True)
    print(
        "Review results and dump evidence. No git commands or external transfers were performed."
    )
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mode", choices=("interpret", "compile", "bench"), default="interpret"
    )
    parser.add_argument(
        "--suite", choices=("smoke", "target", "pipeline"), default="smoke"
    )
    parser.add_argument(
        "--sweep",
        choices=("none", "standard"),
        default="standard",
        help="pipeline suite only: baseline families or additional bounded sweeps",
    )
    parser.add_argument(
        "--cases", default="all", help="comma-separated variant names or all"
    )
    parser.add_argument(
        "--rows", default="16", help="comma-separated row counts; overrides --set rows"
    )
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--budget", type=float, default=1800)
    parser.add_argument(
        "--output-parent", help="existing directory; a fresh child is always created"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="capture local device traces in bench mode",
    )
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.profile and args.mode != "bench":
        parser.error("--profile requires --mode bench")
    try:
        return run_parent(args)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
