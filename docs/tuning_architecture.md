# Tuning architecture

Last updated: 2026-05-10.

## 1. Goals

For each *workload* (e.g. `synthesized_prefill_heavy`, `llama3_3_70b_chat`),
we want to:

1. Tune the attention kernel (block sizes, `static_q_len`, `max_num_subseqs`, …).
2. Sweep vLLM-side parameters (`LONG_PREFILL_TOKEN_THRESHOLD`, `BLOCK_SIZE`, `max_num_seqs`, …) under the tuned kernel.
3. Persist results so any kill / reboot / VM tear-down resumes from the partial state instead of restarting.
4. Look the results up at deploy time so production runs use the best params automatically.
5. Look the results up in reverse — given a kernel improvement, find which workloads benefit — for impact analysis.

Two user-facing paths:

- **Prod**: `.workload` exists → tooling looks up `.service` → `.kernel` → runs vLLM with merged params. No human in the loop.
- **Dev**: user sets env vars by hand. Tooling stays out of the way. Inconsistent values are the user's problem.

## 2. Four files per workload

```
tools/benchmark/cases/<topo>/<model>/<workload>.workload   (input)

  ├──▶ <workload>.kernel.raw/   (per-kernel-SHA JSONL, last 2 kept)
  │      <kernel_sha_a>.jsonl
  │      <kernel_sha_b>.jsonl
  │
  ├──▶ <workload>.kernel        (latest winner, JSON; one entry per pin_key)
  │
  ├──▶ <workload>.service.raw/  (per-vllm-service-SHA JSONL, last 2 kept)
  │      <service_sha_a>.jsonl
  │      <service_sha_b>.jsonl
  │
  └──▶ <workload>.service       (latest winners, JSON; one entry per objective)
```

- `.workload` — bash-variable file declaring `INPUT_LEN`, `OUTPUT_LEN`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM`. Today's format; documented contract (§9).
- `.kernel.raw/` — append-only raw measurements, one file per `kernel.py` SHA. Each row: one `(tuning_key, tunable_params, latency_us)` measurement.
- `.kernel` — projection of the most-recent `.kernel.raw/*.jsonl`. One winner per `tuning_key` (= per pin_key set: page_size, K_kernel, geometry).
- `.service.raw/` — append-only raw measurements, one file per vLLM-service SHA. Each row: one `(combo, {ttft_ms, itl_ms, throughput_tps, p99_ms, …})`.
- `.service` — projection of the most-recent `.service.raw/*.jsonl`. **Multiple winners per workload** — one per objective (`throughput_max`, `ttft_min`, `p99_min`, …) — see §4.

`.kernel.raw/*.jsonl` and `.service.raw/*.jsonl` live under git for review and rollback. The active store during a run is operational-only (SQLite, gitignored — §3).

## 3. Storage layers: operational vs archived

Each tuning or sweep run uses **two** stores:

| | Operational (during run) | Archived (after run) |
|---|---|---|
| Format | SQLite at `tmp/log/{kernel_tuner_run,sweep_run}/<workload>.db` | JSONL files under `<workload>.{kernel,service}.raw/` |
| Lifecycle | Created at run start, written per-result, gitignored | Written at run end (or periodically), git-tracked |
| Why | Atomic transactions, fast random access, in-flight resume | Human-reviewable diffs, durable across machine teardowns, git history |

The active SQLite handles the hot path: per-result writes during a 24h+ tune, per-combo skip lookups during resume. JSONL handles the cold path: archival, PR review, rollback, cross-machine sharing via git.

Promotion (operational → archived) is a CPU-only step: open the SQLite, dump rows into a JSONL named after the kernel-or-service SHA, prune old `.raw/` files past the TTL (§6), recompute the `.kernel` / `.service` projection.

## 4. `.kernel` is single-objective; `.service` is multi-objective

`.kernel.raw` rows carry one metric: per-iter latency. The projection picks `min(latency)` per pin_key — one winner.

`.service.raw` rows carry a metric *bag*: TTFT, ITL, throughput, p50, p99, error rate, … Different deployments want different objectives:

- Chat / interactive: minimize TTFT (or p99).
- Offline batch: maximize throughput.
- SLA-bounded: maximize throughput subject to p99 < threshold.

The projection picks **one winner per objective**. `.service` schema:

```json
{
  "schema_version": 1,
  "kernel_pin_keys": {"page_size": 128, "K_kernel": 128},
  "winners": {
    "throughput_max": {"combo": {...env-vars...}, "metrics": {...}},
    "ttft_min":       {"combo": {...},             "metrics": {...}},
    "p99_min":        {"combo": {...},             "metrics": {...}}
  }
}
```

Adding a new objective is additive (extend the `winners` dict). Default objective at deploy time is `throughput_max` (safe for benchmarking).

## 5. Two lookup directions

### Prod (look down)

```
.workload  ──┐
             ├─▶ .service[objective] ──┐
                                       ├─▶ runtime env (env vars + vLLM config)
             ┌─▶ .kernel[pin_keys] ────┘
.service[objective].kernel_pin_keys ───┘
```

The deployment spec names the workload and the objective. Tooling:

1. Loads `<workload>.service`.
2. Picks `winners[objective]`.
3. Reads `kernel_pin_keys` from that winner.
4. Loads `<workload>.kernel` and looks up the pin_keys.
5. Merges both into a runtime env (env vars + vLLM CLI flags).
6. Launches vLLM.

If `.service` or `.kernel` is missing for the workload, deployment fails loudly. Re-tune / re-sweep is the only recovery.

### Bench / look up

Given a kernel commit:

> "I dropped K=128 latency by 5%. Which workloads use K=128?"

→ grep `<workload>.kernel` files for `K_kernel: 128`, intersect with workloads whose `.service.winners.*.kernel_pin_keys.K_kernel == 128`. Trivially scriptable; no special tooling required.

Given a service combo:

> "Which workloads use `LONG_PREFILL_TOKEN_THRESHOLD=512`?"

→ same shape. The git-tracked text files make every cross-cutting query a `grep -r`.

## 6. Resumability: append-only raw + skip-on-hit

Tuning and sweeping both follow the same pattern:

1. **At start**: read the operational SQLite (creates if absent). Build a skip-set of every `(tuning_key, tunable_params)` (or sweep `combo`) already recorded with status SUCCESS.
2. **For each work unit**: if in skip-set, skip. Otherwise run, append row to SQLite.
3. **On kill** (any reason — kernel OOM, ssh disconnect, reboot, OOM-killer): SQLite has the partial state on disk. Restart re-reads it; the skip-set is already populated.
4. **At completion**: archive SQLite → JSONL under `.raw/<sha>.jsonl`. Prune `.raw/*.jsonl` past the TTL.

**TTL = 2.** Keep the 2 most-recent kernel (or service) SHA's `.raw` files. Older ones are pruned. This survives one rollback (current → previous) without losing data.

**Crash-atomic writes** (§ already in `local_db_manager.py` after [P0 fix](#)): SQLite handles this natively. JSONL is append-only with one row per `write()`, atomic for short rows on POSIX.

**Concurrency** is single-machine-per-workload by assumption. Two TPU VMs working on different workloads don't interact. If we ever need multi-machine on one workload, shard by pin_keys (range-partition the search space across VMs); not in scope today.

## 7. Projection: pure-CPU, idempotent

`.raw → .kernel` and `.raw → .service` are both pure functions. Inputs: the latest `.raw/*.jsonl` (and the current pinning policy). Outputs: a single JSON file. No side effects, no networking, no GPU/TPU.

This means:

- The projection runs on a **laptop** as readily as on the TPU VM. After pulling the SQLite + archived JSONL from the VM, the projection runs locally.
- **CI** can re-run the projection on every PR to validate it picks the same winners (regression guard against projection-logic changes).
- Different teams can ship **different projection logic** (different "best for throughput" definitions) and they all see the same `.raw`. The `.kernel` / `.service` files change; the history stays.
- **Kill any time**, restart any time. The projection is deterministic given inputs.

## 8. Manual / dev path

Dev users set env vars directly:

```bash
LONG_PREFILL_TOKEN_THRESHOLD=512 \
RPA_KERNEL_K=128 \
RPA_MAX_NUM_SUBSEQS=72 \
RPA_P_BLOCK_SIZES=128,1024,128,512 \
RPA_D_BLOCK_SIZES=1,4096,1,1024 \
RPA_M_BLOCK_SIZES=64,512,64,256 \
python3 -m vllm.entrypoints.cli.main serve <model> ...
```

Tooling does **not** validate that these are mutually consistent. If the user sets `RPA_MAX_NUM_SUBSEQS < max_num_seqs` or block sizes that exceed VMEM, the kernel fails loudly at runtime — sufficient feedback. No registry lookup, no fallback. Inconsistency is on the user.

The prod and dev paths share the runner. The runner reads env vars; it doesn't know whether they came from a registry lookup or `export` on a shell.

## 9. `.workload` schema (today's contract, documented)

A `.workload` is a bash file sourced into the env. Required variables:

```bash
INPUT_LEN=2048           # per-request input tokens for the synthetic workload
OUTPUT_LEN=128           # per-request output tokens
MAX_MODEL_LEN=2176       # max prompt + decode (must equal INPUT_LEN + OUTPUT_LEN)
MAX_NUM_SEQS=8           # persistent batch capacity
MAX_NUM_BATCHED_TOKENS=1024
NUM_Q_HEADS=32
NUM_KV_HEADS=8
HEAD_DIM=128
```

Convention: lives at `tools/benchmark/cases/<topo>/<model>/<workload>.workload`. The path implicitly carries `topo` (e.g. `v7x`) and `model` (e.g. `llama3_1_8b`); the `.workload` filename is the workload name used as the lookup key.

A separate `.service` and `.kernel` file lives in the same directory. The lookup is by filename match.

Adding a field to `.workload`: append it. No migration; new fields default to "use today's defaults" until tuning explicitly varies them.

## 10. Migration from today's stack

Phased; no flag day. Each step lands as its own commit and is independently reversible.

1. **SQLite operational store** (P1 in the resumability memory). Replaces `local_db_manager.py`'s JSON-file backend with SQLite. Keeps the same `StorageManager` interface; the existing tuner code is unchanged. *Resume gets faster + crash-atomic.*
2. **Archive step** for kernel tuning. New `archive_kernel.py`: opens the SQLite, dumps to `<workload>.kernel.raw/<kernel_sha>.jsonl`, prunes past TTL=2, projects winners to `<workload>.kernel`. Run at the end of a tune; can also run on demand. *Today's `build_kernel_registry.py` becomes a thin wrapper.*
3. **`.service.raw/` + archive step for sweeps**. Sweep currently writes per-combo metrics.txt + meta.txt. New code path: also append a row to a SQLite at `tmp/log/sweep_run/<workload>.db`. Archive script analogous to (2). *Sweeps become resumable and produce a single durable file.*
4. **`.service` multi-objective winners file**. Projection logic for "best per objective". *Replaces today's informal `.service` files.*
5. **Lookup tooling unification**. `tools/benchmark/lookup.py <workload> <objective>` returns the merged env-var set. `sweep.py:_apply_auto_link` rewires to use this. *Deploy spec gets simpler; manual env-var setting still works.*
6. **`.workload` schema doc + validator**. A small script that loads a `.workload` and asserts the required vars are set. *Catches typos at spec-load time.*

Each step is small enough to ship and review independently. Total work: ~2-3 weeks of focused effort. Order matters: (1) is a prerequisite for (2); (3) and (4) are independent of (1)-(2).

---

## Glossary

- **Workload** — a deployment shape (model, context length, batch capacity, request distribution). Identified by a `.workload` file.
- **Tuning key (`tuning_key`)** — the kernel-side identifier of a problem instance: `(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len, sliding_window, case, K_kernel, code_revision)`. Two workloads with the same tuning_key share `.kernel` rows.
- **Tunable params (`tunable_params`)** — the kernel-side knobs being tuned: `(bq_sz, bkv_sz, bq_csz, bkv_csz, max_num_subseqs)`.
- **Pin keys** — a subset of `tuning_key` fields that the `.service` winner uses to identify which `.kernel` row applies. Today: `(page_size, K_kernel)`. Could expand if more axes start mattering.
- **Combo** — one point in the sweep search space: a dict of vLLM CLI flags + env vars.
- **Objective** — a function from a metrics-bag to a scalar, used to pick a winner from `.service.raw`. Examples: `throughput_max`, `ttft_min`, `p99_min`.
- **Projection** — the pure function `.raw → winners`. Idempotent, deterministic, CPU-only.
