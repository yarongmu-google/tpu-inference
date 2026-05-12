# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Aggregate per-workload registries into a per-model `production.*`.

CLI entry: `python3 -m tools.tuning.v2.cli.aggregate <model_dir>`.

For one model directory (e.g.
`tools/benchmark/cases/v7x/llama3_8b/`):

  1. Read all `<workload>.kernel` files in the dir → write
     `production.kernel`.
  2. Read all `<workload>.service` files in the dir → write
     `production.service`.
  3. Commit + push (opt-out via `--no-commit` or
     `KERNEL_TUNER_NO_PUSH=1`).

Idempotent: re-running on unchanged inputs rewrites the same bytes
(modulo the schema-version stamp).
"""

import json
import logging
import sys
from pathlib import Path

from tools.tuning.v2.core.accumulator import build_production
from tools.tuning.v2.core.git_atomic import commit_and_push


def aggregate(
    model_dir: Path,
    *,
    topo: str | None = None,
    model: str | None = None,
) -> tuple[Path | None, Path | None]:
    """Write `production.kernel` and `production.service` for a model.

    Args:
      model_dir: per-model directory (e.g. `cases/v7x/llama3_8b/`).
      topo, model: optional explicit identity. Default inferred from
                   `cases/<topo>/<model>/` path shape.

    Returns:
      `(production_kernel_path, production_service_path)`. Either is
      None if the corresponding `<workload>.<suffix>` set was empty.
    """
    kernel_doc = build_production(model_dir, ".kernel", topo=topo, model=model)
    service_doc = build_production(model_dir, ".service", topo=topo, model=model)

    kernel_path: Path | None = None
    service_path: Path | None = None

    if kernel_doc["by_workload"]:
        kernel_path = model_dir / "production.kernel"
        kernel_path.write_text(
            json.dumps(kernel_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if service_doc["by_workload"]:
        service_path = model_dir / "production.service"
        service_path.write_text(
            json.dumps(service_doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return kernel_path, service_path


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(
        description="Aggregate per-workload registries to production.*.",
    )
    p.add_argument("model_dir", type=Path,
                   help="A per-model directory containing "
                        "<workload>.kernel and <workload>.service files.")
    p.add_argument("--topo", default=None,
                   help="Override the inferred topo (e.g. 'v7x').")
    p.add_argument("--model", default=None,
                   help="Override the inferred model id.")
    p.add_argument("--no-commit", action="store_true")
    args = p.parse_args(argv)

    from tools.tuning.v2.core.logs import configure as configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)

    if not args.model_dir.exists() or not args.model_dir.is_dir():
        logger.error("model_dir not a directory: %s", args.model_dir)
        return 1

    kernel_path, service_path = aggregate(
        args.model_dir, topo=args.topo, model=args.model,
    )
    if kernel_path is None and service_path is None:
        logger.error(
            "no per-workload .kernel or .service files in %s; "
            "nothing to do.", args.model_dir,
        )
        return 1

    written = [p for p in (kernel_path, service_path) if p is not None]
    for p in written:
        print(p)

    if not args.no_commit:
        commit_and_push(
            written,
            f"[Tune-v2] Aggregate production registries in "
            f"{args.model_dir.name}",
        )
    return 0


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
