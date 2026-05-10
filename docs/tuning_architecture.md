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

## 3. Storage: append-only JSONL

Each tuning or sweep run writes its results to **one** file: a JSONL under `<workload>.{kernel,service}.raw/<sha>.jsonl`. One row per result, appended at the moment of measurement:

```python
with open(raw_path, "a") as f:
    f.write(json.dumps(row) + "\n")
```

That's the entire write path. The file IS the live store AND the archive — there is no separate operational layer. A kill at any point (Ctrl-C, OOM, ssh drop, VM teardown) leaves the file intact through the last completed row.

Why this works at our scale:

- **Atomicity is free.** POSIX guarantees atomic writes under `PIPE_BUF` (~4 KB); our rows are ~200 bytes each. No partial-row corruption is possible. Reads at restart see exactly the rows that finished writing — no recovery logic.
- **No archive step.** The file is git-trackable as-is. `git add` it when ready.
- **JSONL > JSON.** Human-readable diffs in PR review; `grep`-able for ad-hoc queries ("which workloads use K=128?"); schema-flexible (new fields are additive). Protobuf would optimize for size and parse-speed, but at ~1 MB per tune file with ~thousands of rows, neither matters.
- **Speed is adequate.** Building the skip-set at run start = parse the file once. ~1 MB / ms. Negligible.

Resume is just "open the file, read every row, build a set of `(tuning_key, tunable_params)` pairs, skip those, continue." No SQLite, no archive promotion, no two-store synchronization.

If files ever exceed 50 MB each (very unlikely at our scale), `gzip` per-shard. Git stores `.jsonl.gz`; `zless` / `zcat` work fine for inspection. Don't switch to a binary format.

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

1. **At start**: read `<workload>.{kernel,service}.raw/<current_sha>.jsonl` (create if absent). Build a skip-set of every `(tuning_key, tunable_params)` (or sweep `combo`) already in it with status SUCCESS.
2. **For each work unit**: if in skip-set, skip. Otherwise run, append row to the file.
3. **On kill** (any reason — kernel OOM, ssh disconnect, reboot, OOM-killer): the file already has every row up to the last completed unit. Restart re-reads it; the skip-set is already populated.
4. **At completion** (or any safe point): `git add` + `git commit` the `.raw/<sha>.jsonl`. Prune older `.raw/*.jsonl` past the TTL.

**TTL = 2.** Keep the 2 most-recent kernel (or service) SHA's `.raw` files. Older ones are pruned. This survives one rollback (current → previous) without losing data.

**Crash-atomic writes** are inherent to append-only JSONL on POSIX. One row per `write()` call; rows are ~200 bytes (well under PIPE_BUF). No torn rows, no recovery logic at restart.

**Concurrency** is single-machine-per-workload by assumption. Two TPU VMs working on different workloads don't interact. If we ever need multi-machine on one workload, shard by pin_keys (range-partition the search space across VMs); not in scope today.

## 7. Projection: pure-CPU, idempotent

`.raw → .kernel` and `.raw → .service` are both pure functions. Inputs: the latest `.raw/*.jsonl` (and the current pinning policy). Outputs: a single JSON file. No side effects, no networking, no GPU/TPU.

This means:

- The projection runs on a **laptop** as readily as on the TPU VM. After pulling (or `git pull`-ing) the JSONL from the VM, the projection runs locally.
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

1. **`.kernel.raw/` JSONL store for kernel tuning**. Replaces `local_db_manager.py`'s 4-JSON-file directory with one append-only `<workload>.kernel.raw/<kernel_sha>.jsonl`. Resume reads the file at start, builds the skip-set, continues. *Existing `tmp/log/kernel_tuner_run/<id>/` becomes a fallback for legacy DBs; new tunes write to `.raw/`.*
2. **`.kernel` projection update**. `build_kernel_registry.py` already projects raw measurements to a winners JSON; rewire its input from the JSON-files to the new JSONL. Add TTL=2 pruning. *Today's `production.kernel` becomes the new `.kernel`; same content, same field names.*
3. **`.service.raw/` JSONL for sweeps**. Sweep currently writes per-combo `metrics.txt` + `meta.txt` in scattered dirs. Add one extra append to `<workload>.service.raw/<service_sha>.jsonl` at the moment metrics are recorded. *Sweeps become resumable; pre-existing per-combo dirs untouched.*
4. **`.service` multi-objective winners file**. New projection step: read `.service.raw/<latest_sha>.jsonl`, group by pin_keys, pick winners per objective, write `<workload>.service`. *Replaces today's informal `.service` files.*
5. **Lookup tooling unification**. `tools/benchmark/lookup.py <workload> <objective>` returns the merged env-var set (vLLM flags from `.service`, kernel knobs from `.kernel`). `sweep.py:_apply_auto_link` rewires to use this. *Deploy spec gets simpler; manual env-var setting still works.*
6. **`.workload` schema validator**. A small script that loads a `.workload` and asserts the required vars are set. *Catches typos at spec-load time.*

Each step is small enough to ship and review independently. Total work: ~1-2 weeks of focused effort (substantially less than the dual-store version, since SQLite + archive promotion are both eliminated). Order matters: (1) is a prerequisite for (2); (3) is a prerequisite for (4); (5) requires (4).

---

## Glossary

- **Workload** — a deployment shape (model, context length, batch capacity, request distribution). Identified by a `.workload` file.
- **Tuning key (`tuning_key`)** — the kernel-side identifier of a problem instance: `(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len, sliding_window, case, K_kernel, code_revision)`. Two workloads with the same tuning_key share `.kernel` rows.
- **Tunable params (`tunable_params`)** — the kernel-side knobs being tuned: `(bq_sz, bkv_sz, bq_csz, bkv_csz, max_num_subseqs)`.
- **Pin keys** — a subset of `tuning_key` fields that the `.service` winner uses to identify which `.kernel` row applies. Today: `(page_size, K_kernel)`. Could expand if more axes start mattering.
- **Combo** — one point in the sweep search space: a dict of vLLM CLI flags + env vars.
- **Objective** — a function from a metrics-bag to a scalar, used to pick a winner from `.service.raw`. Examples: `throughput_max`, `ttft_min`, `p99_min`.
- **Projection** — the pure function `.raw → winners`. Idempotent, deterministic, CPU-only.
