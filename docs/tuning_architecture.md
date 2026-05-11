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

## 2. Variable classification (rationale per category)

Every parameter answers exactly one of six questions. Conflating them is the bug pattern we hit repeatedly on the rpa3_2 branch (LPTT, MNB, mns, page_size, kernel_K all straddled categories before they were sorted into one).

| # | Category | Question | When set | By whom | Lives in |
|---|---|---|---|---|---|
| 1 | **Workload key** | What is this deployment? | Pre-tune | Operator | `.workload` |
| 2 | **Measurement var** | What benchmark load measures this? | Pre-tune, specified with workload | Operator | `.workload` |
| 3 | **Kernel-tuned var** | How should the kernel JIT? | Kernel-tune | Search | `.kernel` (`tunable_params`) |
| 4 | **Kernel-derived var** | What env vars implement the kernel's tuned config? | Auto-link, at sweep/deploy time | Deterministic from #3 | `.kernel` (env aliases) |
| 5 | **Service-tuned var** | How should vLLM be configured? | Service-sweep | Search | `.service` |
| 6 | **Service-derived var** | What CLI flags implement the service's tuned config? | Auto-link | Deterministic from #5 | `.service` (env aliases) |

### Why each is where it is

**(1) Workload keys** — parameters that identify a problem instance. Two deployments sharing all workload keys share the same `.kernel` and `.service`. Change one → different problem entirely → re-tune required.

- `MODEL`, `TENSOR_PARALLEL_SIZE`: which model, what sharding.
- `NUM_*_HEADS`, `HEAD_DIM`, `q_dtype`, `kv_dtype`, `sliding_window`: model architecture (kernel JIT shape).
- `MAX_MODEL_LEN`: deployment context cap (memory commitment + kernel JIT shape).
- Hardware (TPU topology): physical platform; implicit via path.
- `code_revision`: kernel-source identity; different SHA = different kernel.

**(2) Measurement vars** — parameters that define the benchmark load used to evaluate optimality. They're **not tuned**, but they're not workload identity either — they're specified WITH the workload because the kernel-tune's synthetic workload and the service-sweep's bench load both need to reflect the deployment's representative traffic.

- `DATASET`, `INPUT_LEN`, `OUTPUT_LEN`: load shape used by bench driver.
- `NUM_PROMPTS`, `REQUEST_RATE`: bench duration / arrival pattern.

Operators pin these in `.workload` to characterize the deployment's expected traffic. The kernel-tune and service-sweep both consume them; they're inputs to both layers' measurement loops. (If an operator wants to compare two traffic shapes, that's a meta-experiment with two `.workload` files, not a sweep over measurement axes within one.)

**(3) Kernel-tuned vars** — parameters baked into the JIT'd kernel artifact at compilation time. Searched by kernel-tune across known-good candidate spaces.

- `bq_sz`, `bkv_sz`, `bq_csz`, `bkv_csz` (per case: D, M, P, L): Pallas block sizes — the kernel's compute-tiling strategy.
- `page_size`: KV cache page granularity (kernel-side).
- `kernel_K`: static-shape Q-axis size for the L kernel's chunked-prefill iter.
- `mnss`: prefetch-array size (= iter capacity per pallas_call). SMEM-driven cap.

Kernel-tune owns these because (a) they require a measurement loop against synthetic load, and (b) they bake into a JIT'd artifact that can't change at runtime. Service-sweep takes them as inputs.

**(4) Kernel-derived (env-var aliases)** — deployment env vars whose values are computed deterministically from kernel-tuned vars. Server validates the chain at runner init.

- `BLOCK_SIZE = page_size` (vLLM CacheConfig).
- `RPA_KERNEL_K = kernel_K`.
- `RPA_MAX_NUM_SUBSEQS = mnss`.
- `LONG_PREFILL_TOKEN_THRESHOLD = mnss × kernel_K` (the per-call q-token ceiling vLLM may schedule).
- `RPA_*_BLOCK_SIZES = per-case block sizes`.

Separate from category (3) because they cross a boundary (the kernel-tune's `tunable_params` representation vs the runtime env-var contract). Same `.kernel` row, two facets.

**(5) Service-tuned vars** — parameters that affect deployment performance but are determined at deployment-config time (not kernel-build time). Searched by service-sweep across objective-relevant candidates.

- `MAX_NUM_BATCHED_TOKENS`: per-step q-token budget. **Interacts** with kernel-tuned `mnss` (per-call iter utilization = packed_prefills × ceil(prompt/K), bounded by both MNS and mnss).
- `MAX_NUM_SEQS`: in-flight request capacity. Different `mns` may require different kernel-tunes; the registry is keyed on `(workload, mns)`.

Service-tune owns these because re-sweep is cheap relative to re-JIT.

**(6) Service-derived (env-var aliases)** — CLI flags / env vars from service-tuned vars; runtime passes through. Today largely identity (`MAX_NUM_BATCHED_TOKENS` → `--max-num-batched-tokens`).

### Today vs proposed end-state

Today, `page_size`, `chunk_prefill_size`, `RPA_KERNEL_K`, and `BLOCK_SIZE` straddle `.workload` (user-pinned) and `.kernel` (kernel-derived). The refactor moves them cleanly into `.kernel`: the kernel tuner sweeps `page_size` and `kernel_K` as part of `tunable_params`, the registry returns ONE winner per workload, and `sweep.py:_apply_auto_link` sets `BLOCK_SIZE` and `RPA_KERNEL_K` from that winner. See §12 migration for the mechanics.

`MAX_NUM_BATCHED_TOKENS` is definitively a service-tuned var. The kernel-tune's `mnss` only realizes its capacity at a specific MNB (per-call iters = `packed_prefills × ceil(prompt/kernel_K)`, where `packed_prefills` is capped by `MNB / prompt_len`). Hand-coding MNB in `.workload` decouples the kernel tune from the workload it actually serves — e.g., the L kernel was tuned with `mnss=4224` (designed for ~4096 iters/call at MNB ≈ `mnss × kernel_K` ≈ 1M) but deployed at MNB=8192 (32 iters/call, 130× over-provisioned, paying SMEM cost for capacity that's never used). The sweep decides MNB; the kernel-tune must be aware of, or recomputed against, the sweep's MNB candidates — a coupling the refactor needs to handle (today the two layers don't talk).

## 3. Kernel ↔ Service symmetry

The kernel layer and the service layer mirror each other. The refactor should make that symmetry explicit and exploit it.

| Aspect | Kernel layer | Service layer |
|---|---|---|
| Raw measurements | `.kernel.raw/<sha>.jsonl` (per kernel-source SHA) | `.service.raw/<sha>.jsonl` (per vLLM/runner SHA) |
| Projection step | `build_kernel_registry` | `build_service_registry` |
| Winners file | `.kernel` (one per `(workload, mns)`) | `.service` (one per `(workload, objective)`) |
| Search space | `tunable_params` (block sizes, mnss, page_size, kernel_K) | sweep axes (MNB, MNS, ...) |
| Auto-link consumer | `sweep.py` reads `.kernel` for `RPA_*` envs | `run_pipeline.sh` / dev path reads `.service` for vLLM CLI flags |
| Workload key | Same set | Same set |
| Search granularity | Per `tunable_params` combo → kernel latency | Per scheduler combo → end-to-end metric (req/s, TTFT, ...) |

Implications for the refactor:
- **Shared infra**: append-only raw store + idempotent projection + registry-keyed auto-link, factored out so both kernel-tune and service-sweep reuse it.
- **Schema-evolution friendliness**: both layers partition raw by source SHA so old measurements survive code changes without manual migration.
- **Symmetric tooling**: `tune_all_cases.sh` and `sweep.sh` are sibling top-level scripts with the same flow (define spec → run measurements → append to raw → project to winners). The benchmark worker (`run_benchmark.sh`) is shared; the tuner has its analogous worker (one pallas_call per combo).
- **Bottom-up impact analysis** (see §7) is the same shape on both layers.

## 4. Four files per workload

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

- `.workload` — bash-variable file declaring `INPUT_LEN`, `OUTPUT_LEN`, `MAX_NUM_SEQS`, `MAX_MODEL_LEN`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `HEAD_DIM`. Today's format; documented contract (§11).
- `.kernel.raw/` — append-only raw measurements, one file per `kernel.py` SHA. Each row: one `(tuning_key, tunable_params, latency_us)` measurement.
- `.kernel` — projection of the most-recent `.kernel.raw/*.jsonl`. One winner per `tuning_key` (= per pin_key set: page_size, K_kernel, geometry).
- `.service.raw/` — append-only raw measurements, one file per vLLM-service SHA. Each row: one `(combo, {ttft_ms, itl_ms, throughput_tps, p99_ms, …})`.
- `.service` — projection of the most-recent `.service.raw/*.jsonl`. **Multiple winners per workload** — one per objective (`throughput_max`, `ttft_min`, `p99_min`, …) — see §6.

`.kernel.raw/*.jsonl` and `.service.raw/*.jsonl` live under git for review and rollback. The active store during a run is operational-only (SQLite, gitignored — §5).

## 5. Storage: append-only JSONL

Each tuning or sweep run writes its results to **one** file: a JSONL under `<workload>.{kernel,service}.raw/<sha>.jsonl`. One row per result, appended at the moment of measurement:

```python
with open(raw_path, "a") as f:
    f.write(json.dumps(row) + "\n")
```

That's the entire write path. The file IS the live store AND the archive — there is no separate operational layer. A kill at any point (Ctrl-C, OOM, ssh drop, VM teardown) leaves the file intact through the last completed row.

Why this works at our scale:

- **Atomicity is robust on modern Linux filesystems.** POSIX's `PIPE_BUF` guarantee is for pipes/FIFOs, not regular files — strictly speaking POSIX makes no atomicity guarantee for `write(2)` on regular files. But ext4 (with the default `data=ordered` journal) and XFS both make a single sub-block `write(2)` atomic in practice: the entry either lands fully or not at all. Our rows (~200 bytes) are well within filesystem block boundaries (typically 4 KB). With `O_APPEND` + one `write(2)` per row, we get atomicity + non-interleaving without any application-level coordination. **Soft-kill survival** (Ctrl-C, SIGKILL, OOM-killer, ssh disconnect): the kernel flushes the page cache to disk on process exit; the file is intact through the last completed write. **Hard-crash survival** (power loss, kernel panic): up to a few seconds of un-flushed rows from the page cache may be lost; the file remains *consistent* through the last successfully-flushed row (no torn rows). For our use case — restart and re-tune at most a handful of combos — this is sufficient. If a deployment ever needs hard-crash durability, add `os.fsync` after each row at the cost of ~1 extra ms/row.
- **No archive step.** The file is git-trackable as-is. `git add` it when ready.
- **JSONL > JSON.** Human-readable diffs in PR review; `grep`-able for ad-hoc queries ("which workloads use K=128?"); schema-flexible (new fields are additive). Protobuf would optimize for size and parse-speed, but at ~1 MB per tune file with ~thousands of rows, neither matters.
- **Speed is adequate.** Building the skip-set at run start = parse the file once. ~1 MB / ms. Negligible.

Resume is just "open the file, read every row, build a set of `(tuning_key, tunable_params)` pairs, skip those, continue." No SQLite, no archive promotion, no two-store synchronization.

If files ever exceed 50 MB each (very unlikely at our scale), `gzip` per-shard. Git stores `.jsonl.gz`; `zless` / `zcat` work fine for inspection. Don't switch to a binary format.

## 6. `.kernel` is single-objective; `.service` is multi-objective

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

## 7. Two lookup directions

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

### Bottom-up (impact analysis / R&D)

Symmetric to the prod direction. Given a low-level change, find the deployments affected.

```
kernel-source change (new code_revision)
   ──▶ which `.kernel` rows partition by old vs new SHA?
       ──▶ re-tune those workloads with the new SHA
           ──▶ for each .kernel row whose winners shifted:
               ──▶ re-sweep .service
                   ──▶ for each .service winner that moved:
                       ──▶ which deployments use this workload+objective?
```

Concrete queries today are `grep -r` against the git-tracked `.kernel` and `.service` JSON files; the refactor formalizes this into tooling. Examples:

> "I dropped K=128 kernel latency by 5%. Which workloads use K=128?"

→ grep `<workload>.kernel` files for `K_kernel: 128`, intersect with workloads whose `.service.winners.*.kernel_pin_keys.K_kernel == 128`.

> "Which deployments are served by `LONG_PREFILL_TOKEN_THRESHOLD=512`?"

→ grep `.service.winners.*.combo.LONG_PREFILL_TOKEN_THRESHOLD == 512`, list workloads.

> "I changed the kernel source. Which workloads need re-tune+re-sweep?"

→ grep `.kernel.tuning_key.code_revision != <new_HEAD_kernel_sha>`. All those workloads' kernel-tunes are stale; service-sweeps depend on them.

The git-tracked text files make every cross-cutting query a `grep -r`. The refactor's "impact-analysis CLI" wraps these patterns into named commands with regression-detection semantics.

## 8. Resumability: append-only raw + skip-on-hit

Tuning and sweeping both follow the same pattern:

1. **At start**: read `<workload>.{kernel,service}.raw/<current_sha>.jsonl` (create if absent). Build a skip-set of every `(tuning_key, tunable_params)` (or sweep `combo`) already in it with status SUCCESS.
2. **For each work unit**: if in skip-set, skip. Otherwise run, append row to the file.
3. **On kill** (any reason — kernel OOM, ssh disconnect, reboot, OOM-killer): the file already has every row up to the last completed unit. Restart re-reads it; the skip-set is already populated.
4. **At completion** (or any safe point): `git add` + `git commit` the `.raw/<sha>.jsonl`. Prune older `.raw/*.jsonl` past the TTL.

**TTL = 2.** Keep the 2 most-recent kernel (or service) SHA's `.raw` files. Older ones are pruned. This survives one rollback (current → previous) without losing data.

**Crash-atomic writes** are inherent to append-only JSONL on POSIX. One row per `write()` call; rows are ~200 bytes (well under PIPE_BUF). No torn rows, no recovery logic at restart.

**Concurrency.** Per-(workload, kernel_sha) for kernel tuning and per-(workload, service_sha) for sweeps is the unit of write isolation: each gets its own `.raw/<sha>.jsonl`. Two TPU VMs running **different** workloads, or **different** kernel/service SHAs against the same workload (e.g. one VM tuning `(rpa_v3, vllm)`-throughput, another tuning `(rpa_v3, vllm_latency)`-latency), don't interact. Running two machines on the **same** (workload, sha) pair is unsupported; if needed in the future, shard by pin_keys (range-partition the search space across VMs).

**Schema-evolution friendliness.** `TunableParams` field additions (e.g. adding `max_num_subseqs` for the LOGICAL case in commit 2536219e) come with a `kernel.py` change, hence a new kernel SHA, hence a fresh `.raw/<new_sha>.jsonl`. Old SHAs' rows stay isolated — no hash collisions in the skip-set, no schema migration. The TTL=2 prune retires the prior schema's data after one more SHA bump. Same property holds for `.service.raw` against vLLM/runner SHA changes.

## 9. Projection: pure-CPU, idempotent

`.raw → .kernel` and `.raw → .service` are both pure functions. Inputs: the latest `.raw/*.jsonl` (and the current pinning policy). Outputs: a single JSON file. No side effects, no networking, no GPU/TPU.

This means:

- The projection runs on a **laptop** as readily as on the TPU VM. After pulling (or `git pull`-ing) the JSONL from the VM, the projection runs locally.
- **CI** can re-run the projection on every PR to validate it picks the same winners (regression guard against projection-logic changes).
- Different teams can ship **different projection logic** (different "best for throughput" definitions) and they all see the same `.raw`. The `.kernel` / `.service` files change; the history stays.
- **Kill any time**, restart any time. The projection is deterministic given inputs.

## 10. Manual / dev path

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

Tooling does **not** cross-validate the env vars against the workload geometry. But mostly-broken combinations still fail loudly, just at three distinct stages depending on the failure mode:

- **Runner init**: shape invariants the runner can check pre-trace. E.g. `tpu_runner.py:432` raises `ValueError` for `RPA_MAX_NUM_SUBSEQS < max_num_seqs`. Earliest, clearest feedback.
- **JIT trace time**: VMEM / SMEM overflow, kernel-shape inconsistency. The pallas_call lowering raises with the exact memory-budget breakdown.
- **Kernel runtime**: only true semantic bugs (e.g. NaN propagation from a corrupt cache page). Rare; means the geometry was internally consistent but produced wrong results.

Inconsistency is on the user; loudness is on the runner. No registry lookup, no fallback.

The prod and dev paths share the runner. The runner reads env vars; it doesn't know whether they came from a registry lookup or `export` on a shell.

## 11. `.workload` schema (today's contract, documented)

A `.workload` is a bash file sourced into the env via `set -a; source <file>; set +a`. Variables, grouped by what consumes them:

**Model identity** (vLLM CLI / model loading):

```bash
MODEL=meta-llama/Llama-3.1-8B-Instruct   # HF model id
TENSOR_PARALLEL_SIZE=4                    # vLLM TP degree
```

**Model shape** (kernel `tuning_key`):

```bash
NUM_Q_HEADS=32
NUM_KV_HEADS=8
HEAD_DIM=128
MAX_MODEL_LEN=8192       # must satisfy MAX_MODEL_LEN >= INPUT_LEN + OUTPUT_LEN;
                         # tuner / sweep enforce this at load time.
```

**Runtime capacity** (vLLM `--max-num-seqs`, etc., AND kernel SMEM sizing):

```bash
MAX_NUM_SEQS=8                # persistent batch capacity
MAX_NUM_BATCHED_TOKENS=1024   # per-step token budget across all seqs
LONG_PREFILL_TOKEN_THRESHOLD= # optional: vLLM K_sched. Empty -> vLLM default.
BLOCK_SIZE=                   # optional: KV-cache page size. Empty -> auto-link
                              # from .kernel (orchestrator) or vLLM default (dev).
```

**Benchmark workload** (used by `run_benchmark.sh` to drive the synthetic load; not a kernel input):

```bash
DATASET=sonnet            # which prompt corpus to sample
INPUT_LEN=2048            # per-request prompt length
OUTPUT_LEN=128            # per-request output length
NUM_PROMPTS=1000          # bench duration / sample count
```

**Convention.** Lives at `tools/benchmark/cases/<topo>/<model>/<workload>.workload`. The path implicitly carries `topo` (e.g. `v7x`) and `model` (e.g. `llama3_1_8b`); the `.workload` filename is the workload name used as the lookup key. `.service` and `.kernel` live alongside in the same directory.

**Default-then-override pattern.** Each variable uses `: "${FOO:=default}"` so callers can override via env without editing the file. Empty defaults (`BLOCK_SIZE=`) signal "filled in by orchestrator from the registry, or pass through to vLLM if dev-path".

**Adding a field.** Append it. New fields default to "use today's behavior" until a tuning sweep explicitly varies them.

## 12. Migration from today's stack

Phased; no flag day. Each step lands as its own commit and is independently reversible.

1. **`.kernel.raw/` JSONL store for kernel tuning**. Replaces `local_db_manager.py`'s 4-JSON-file directory with one append-only `<workload>.kernel.raw/<kernel_sha>.jsonl`. Resume reads the file at start, builds the skip-set, continues. *Existing `tmp/log/kernel_tuner_run/<id>/` becomes a fallback for legacy DBs; new tunes write to `.raw/`.*
2. **`.kernel` projection update**. `build_kernel_registry.py` already projects raw measurements to a winners JSON; rewire its input from the JSON-files to the new JSONL. Add TTL=2 pruning. *Today's `production.kernel` becomes the new `.kernel`; same content, same field names.*
3. **`.service.raw/` JSONL for sweeps**. Sweep currently writes per-combo `metrics.txt` + `meta.txt` in scattered dirs. Add one extra append to `<workload>.service.raw/<service_sha>.jsonl` at the moment metrics are recorded. *Sweeps become resumable; pre-existing per-combo dirs untouched.*
4. **`.service` multi-objective winners file**. New projection step: read `.service.raw/<latest_sha>.jsonl`, group by pin_keys, pick winners per objective, write `<workload>.service`. *Replaces today's informal `.service` files.*
5. **Lookup tooling unification**. `tools/benchmark/lookup.py <workload> <objective>` returns the merged env-var set (vLLM flags from `.service`, kernel knobs from `.kernel`). `sweep.py:_apply_auto_link` rewires to use this. *Deploy spec gets simpler; manual env-var setting still works.*
6. **`.workload` schema validator**. A small script that loads a `.workload` and asserts the required vars are set. *Catches typos at spec-load time.*
7. **Wider, liberal search-space defaults (kernel + service)**. Today's hardcoded narrow candidate lists trap us in pre-tuned regions. Examples discovered the hard way on rpa3_2:
   - Old service `LONG_PREFILL_TOKEN_THRESHOLD` axis stopped at 2048; missed LPTT=8192 (the single-shot-prefill regime for P, architecturally similar to L's mnss=33 latency configuration).
   - Old throughput `MAX_NUM_BATCHED_TOKENS` axis stopped at 65,536; missed MNB=131,072 where L finally beat P (Phase 5, 4.90 vs 4.79 req/s). Without this range, L looked like a throughput regression.
   - Old kernel-tune mnss candidates derived from `mns * (M+1)` for `M ∈ {1, 2, 4, 8, 16, 32}`; cannot reach larger M without manual env override.

   Refactor: candidate-space defaults should be liberal (full octave coverage to the SMEM/HBM ceiling), with combos pruned by infeasibility (OOM, invariant violation) rather than by hardcoded short lists. Pruning is automatic; range narrowing is per-user-intent.
8. **Per-model search spaces** at both kernel and service layers. Today one set of defaults is shared across all models (`RPA_V3_BQ_SZ_LST = [32, …, 8192]`, `MAX_NUM_BATCHED_TOKENS = [8192, …, 65536]`). But the optimal envelope is fundamentally different per model shape: 8B-TP1 vs 70B-TP8 vs MoE Qwen3-30B-A3B have different per-token KV footprints, head counts, HBM caps, saturation regimes. Add `tools/benchmark/cases/<topo>/<model>/ranges.{kernel,service}.json` alongside the `.workload` — inherits from a default, overridable per-model. Once landed, **re-sweep 8B on a much wider grid** as the first end-to-end validation.

Each step is small enough to ship and review independently. Total work: ~2-3 weeks of focused effort (was 1-2 before items 7-8). Order matters: (1) is a prerequisite for (2); (3) is a prerequisite for (4); (5) requires (4); (7-8) can land anytime after the schema settles.

---

## Glossary

- **Workload** — a deployment shape (model, context length, batch capacity, request distribution). Identified by a `.workload` file.
- **Tuning key (`tuning_key`)** — the kernel-side identifier of a problem instance: `(page_size, q_dtype, kv_dtype, num_q_heads, num_kv_heads, head_dim, max_model_len, sliding_window, case, K_kernel, code_revision)`. Two workloads with the same tuning_key share `.kernel` rows.
- **Tunable params (`tunable_params`)** — the kernel-side knobs being tuned: `(bq_sz, bkv_sz, bq_csz, bkv_csz, max_num_subseqs)`.
- **Pin keys** — a subset of `tuning_key` fields that the `.service` winner uses to identify which `.kernel` row applies. Today: `(page_size, K_kernel)`. Could expand if more axes start mattering.
- **Combo** — one point in the sweep search space: a dict of vLLM CLI flags + env vars.
- **Objective** — a function from a metrics-bag to a scalar, used to pick a winner from `.service.raw`. Examples: `throughput_max`, `ttft_min`, `p99_min`.
- **Projection** — the pure function `.raw → winners`. Idempotent, deterministic, CPU-only.
- **`kernel_sha`** — short git SHA (8-char) of `tpu_inference` HEAD at tune time. Same definition as today's `TuningKey.code_revision`. Captures any change to `kernels/ragged_paged_attention/v3/kernel.py` (and conservatively to other files; the over-coverage is fine — it just rotates `.raw` shards more often than strictly necessary).
- **`service_sha`** — pair `(vllm_sha, tpu_inference_sha)` joined as `<8>-<8>`. Either repo's change can shape vLLM scheduling or runner behavior; tracking both is the conservative choice. Today's per-sweep `meta.txt` already captures both via `git rev-parse HEAD` in each repo's worktree.
- **"Latest" `.raw/<sha>.jsonl`** — the file whose `<sha>` matches the **current** `tpu_inference` HEAD at the time of the projection. If no file matches HEAD (e.g. fresh checkout, never tuned), the projection fails loudly rather than picking a stale neighbor — better to be told "no data for this SHA" than to silently use winners from a different kernel version.
