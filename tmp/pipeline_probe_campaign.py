"""One bounded campaign for layout, pipeline, capability and calibration probes."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
from typing import Any

from pipeline_probe_kernels import NAMES, PipelineConfig
from run_layer_probes import source_hashes, versions, write_json


def planned_jobs(args: Any) -> list[dict[str, Any]]:
    overrides = {}
    fields = {f.name for f in dataclasses.fields(PipelineConfig)}
    for item in args.set:
        key, sep, value = item.partition("=")
        if not sep or key not in fields:
            raise ValueError(f"unknown pipeline override: {item}")
        try:
            overrides[key] = json.loads(value)
        except json.JSONDecodeError:
            overrides[key] = value
    jobs = []

    def add(name: str, **values: Any) -> None:
        config = PipelineConfig(**{**dict(dtype="bfloat16"), **values, **overrides})
        config.validate()
        if name == "stream_emitted" and config.direction != "mixed":
            return
        jobs.append(
            dict(family="pipeline", case=name, config=dataclasses.asdict(config))
        )

    for name in NAMES:
        add(name)
    if args.sweep == "standard":
        for depth in (1, 2, 4, 8, 16):
            for name in ("stream_manual", "stream_emitted"):
                add(name, depth=depth, block_rows=128, jobs=64)
        for direction in ("read", "write", "mixed", "local"):
            for block_rows in (8, 16, 32, 128, 512):
                for name in ("stream_serial", "stream_manual"):
                    add(
                        name,
                        block_rows=block_rows,
                        jobs=64,
                        direction=direction,
                        depth=8,
                    )
        for unroll in (1, 2, 4, 8):
            for name in ("compute_serial", "compute_interleaved", "stream_manual"):
                add(name, unroll=unroll, jobs=32)
        for name in (
            "stream_serial",
            "stream_manual",
            "compute_serial",
            "compute_interleaved",
        ):
            for horizon in (32, 128, 512):
                add(name, jobs=horizon, block_rows=16, depth=8, unroll=4)
        for pattern in ("contiguous", "skew"):
            for name in (
                "dispatch_row_serial",
                "dispatch_row_pipeline",
                "dispatch_parent_pipeline",
            ):
                add(name, pattern=pattern, jobs=7, hits=8)
        for name in ("mla_expanded_pipeline", "mla_split_pipeline"):
            for depth in (1, 2, 4):
                add(name, depth=depth, jobs=5)
        for name in NAMES:
            add(name, dtype="float32")

    from layer_probe_kernels import Config, NAMES as BASE_NAMES

    base = Config(
        rows=32,
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
    for name in BASE_NAMES:
        jobs.append(dict(family="base", case=name, config=dataclasses.asdict(base)))
    if args.sweep == "standard":
        for rows in (128, 256, 512, 1024):
            for name in ("phase_scoped", "phase_colive"):
                jobs.append(
                    dict(
                        family="base",
                        case=name,
                        config=dataclasses.asdict(dataclasses.replace(base, rows=rows)),
                    )
                )
    if args.cases != "all":
        selected = set(args.cases.split(","))
        if selected - set(NAMES) - set(BASE_NAMES):
            raise ValueError("unknown campaign case")
        jobs = [job for job in jobs if job["case"] in selected]
    unique = {json.dumps(job, sort_keys=True): job for job in jobs}
    return list(unique.values())


def timing_fits(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostic horizon slopes; attribution/census review remains mandatory."""
    import numpy as np

    groups: dict[str, dict[int, dict[str, Any]]] = {}
    for r in records:
        if (
            r.get("status") != "passed"
            or "host_call_us" not in r
            or "jobs" not in r.get("config", {})
        ):
            continue
        config = dict(r["config"])
        horizon = config.pop("jobs")
        key = json.dumps([r["case"], config], sort_keys=True)
        groups.setdefault(key, {})[horizon] = r
    fits = []
    for key, points in groups.items():
        if len(points) < 3:
            continue
        ns = np.array(sorted(points), dtype=np.float64)
        ys = np.array([points[int(n)]["host_call_us"]["min"] for n in ns])
        jitter = max(points[int(n)]["host_call_us"]["p90_minus_p10"] for n in ns)
        slope, intercept = np.polyfit(ns, ys, 1)
        span = float(ys[-1] - ys[0])
        fits.append(
            dict(
                group=json.loads(key),
                horizons=ns.tolist(),
                min_us=ys.tolist(),
                slope_us_per_job=float(slope),
                intercept_us=float(intercept),
                max_residual_us=float(np.max(np.abs(ys - (slope * ns + intercept)))),
                span_us=span,
                jitter_us=jitter,
                span_sufficient=bool(slope > 0 and span > 10 * jitter),
                accepted_calibration=False,
                pending="dynamic instruction/descriptor census, compiler hoisting/spills, and device-time attribution",
            )
        )
    return fits


def run_campaign(args: Any) -> int:
    if (
        not 1 <= args.timeout <= 1800
        or not 1 <= args.budget <= 14400
        or not 1 <= args.repeats <= 200
    ):
        raise ValueError("invalid timeout/budget/repeats")
    jobs = planned_jobs(args)
    if not 1 <= len(jobs) <= 256:
        raise ValueError("campaign must contain 1..256 cases")
    parent = Path(args.output_parent).resolve() if args.output_parent else None
    if parent is not None and not parent.is_dir():
        raise ValueError("output-parent must exist")
    directory = Path(tempfile.mkdtemp(prefix="pipeline-probes-", dir=parent))
    write_json(
        directory / "manifest.json",
        dict(
            mode=args.mode,
            sweep=args.sweep,
            jobs=jobs,
            versions=versions(),
            source_sha256=source_hashes(),
            budget_seconds=args.budget,
            override_scope="--set applies to pipeline cases only; batch A uses recorded target configuration",
            scope="local execution; no network, installs or git actions",
        ),
    )
    print(f"RESULT_DIR={directory}", flush=True)
    records = []
    deadline = time.monotonic() + args.budget
    with (directory / "runner.log").open("w") as log:
        for index, job in enumerate(jobs):
            child = directory / f"{index:03d}-{job['case']}"
            child.mkdir()
            job = {
                **job,
                "mode": args.mode,
                "repeats": args.repeats,
                "profile": args.profile,
            }
            write_json(child / "job.json", job)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                result = {
                    **job,
                    "status": "not_run",
                    "reason": "campaign time budget exhausted",
                }
            else:
                try:
                    p = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("run_layer_probes.py")),
                            "--worker",
                            str(child / "job.json"),
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=min(remaining, args.timeout),
                    )
                    (child / "stdout.log").write_text(p.stdout)
                    (child / "stderr.log").write_text(p.stderr)
                    path = child / "result.json"
                    result = (
                        json.loads(path.read_text())
                        if path.exists()
                        else {
                            **job,
                            "status": "failed",
                            "reason": "worker did not write result",
                        }
                    )
                    result["exit_code"] = p.returncode
                    if p.returncode and result["status"] in (
                        "passed",
                        "compiled",
                        "unsupported",
                        "unsupported_interpreter",
                    ):
                        result.update(
                            status="failed", reason="worker exit contradicts result"
                        )
                except subprocess.TimeoutExpired as exc:
                    for name, value in (("stdout", exc.stdout), ("stderr", exc.stderr)):
                        (child / f"{name}.log").write_text(
                            value.decode(errors="replace")
                            if isinstance(value, bytes)
                            else value or ""
                        )
                    result = {
                        **job,
                        "status": "timeout",
                        "reason": "bounded worker timeout",
                    }
            result["result_path"] = str((child / "result.json").relative_to(directory))
            write_json(child / "result.json", result)
            records.append(result)
            line = f"{index:03d} {job['case']}: {result['status']}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
    collected = {"passed", "compiled", "unsupported", "unsupported_interpreter"}
    complete = all(r["status"] in collected for r in records)
    write_json(
        directory / "summary.json",
        dict(
            complete_collection=complete,
            all_cases_passed=all(r["status"] == "passed" for r in records),
            unsupported_is_performance_pass=False,
            results=records,
            horizon_fits=timing_fits(records),
        ),
    )
    (directory / "review.md").write_text(
        "# Run review\n\nFor each result: inspect rejection stage or correctness, assigned layouts, "
        "DMA start/wait placement and exact extents, in-loop instruction counts, spills, "
        "and device trace. A timing slope alone is not a calibration. Record accepted, "
        "rejected or unresolved decisions by case/config/hash before changing any plan.\n"
    )
    archive = directory.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(directory, arcname=directory.name)
    print(f"SUMMARY={directory / 'summary.json'}\nARCHIVE={archive}", flush=True)
    return 0 if complete else 1
