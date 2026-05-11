# Tuning v2

Rewrite of the kernel-tune + service-sweep + benchmark pipeline. Implements
the design in `docs/tuning_architecture.md`. Migration plan: see
`docs/tuning_v2_migration_plan.md`.

Status: **WIP**. v1 (`tools/kernel/tuner/v1/`, `tools/benchmark/sweep*.py`)
remains the production path during migration. v2 lands incrementally,
test-first.

## Layout

- `core/` — shared infra: raw store, projection, accumulator, sha, keyset, git helper.
- `kernel/` — kernel-tune layer (tune, project, search_space).
- `service/` — service-sweep layer (sweep, project, search_space).
- `cli/` — top-level commands: aggregate, lookup, validate, migrate.
- `scripts/` — thin shell wrappers over the Python entries.

## Running tests

```bash
python3 -m unittest discover tests.tuning_v2 -v
```

100% line coverage target on `tools/tuning/v2/`. TPU runs are mocked at the
`pallas_call` / `vllm bench serve` boundary; tests run on any host.
