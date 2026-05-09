# RPA v3 Kernel — Decoupled K_sched / K_kernel Design

Working design doc for the kernel-side change in PR3 of the rpa3_2 branch.
Goal: enable the runner to chunk a single physical request into multiple
logical iterations of the kernels `@pl.loop` so the static-K PREFILL
kernel runs on more of the prefill budget per scheduler step.

This doc enumerates every kernel array, classifies its size dependence
(logical seq count vs physical seq count vs token budget vs hardware
pool), and lays out the proposed kernel signature change. Treat this as
a living checklist — discuss line by line, edit in place.

---

## 1. Background & motivation

### 1.1 The TTFT problem

The user-visible goal is **reducing TTFT** (time-to-first-token) for
prefill-heavy workloads. vLLM measures TTFT from `arrival_time` (set
by the renderer when the request lands — `vllm/renderers/base.py:929`)
to the moment the first generated token is emitted
(`vllm/v1/metrics/stats.py:369`), so TTFT counts queue + scheduling +
all prefill steps + decode-step-1. The default benchmark
(`vllm/benchmarks/serve.py`) runs **1000 prompts at unbounded request
rate** — bursty, fully concurrent — so multi-request queue clearing
matters.

Today, the throughput-tuned PREFILL kernel runs at a small static
`q_len = K_kernel` (typically 256 or 512). With chunked prefill on
and `K_sched = K_kernel`, an 8K-token prompt takes
`8192 / 256 = 32` steps to finish prefill. TTFT for that request ≈
32 × per-step latency. **Bad.**

We can't just raise `K_kernel` to 8192 to compress prefill into one
step — that would change the kernel's `static_q_len` specialization,
blow up VMEM tile sizes, and abandon the per-K_kernel tuning sweet
spot. We *also* can't just route prefills through the MIXED kernel
in one step — MIXED is less specialized than PREFILL at the K_kernel
sweet spot (no static_q_len, conservative tile choice, remainder
handling), so throughput is lower.

**Decoupled-K** breaks the equality: keep `K_kernel` at the
throughput-optimal value (PREFILL kernel runs fast on small tight
tiles) AND let `K_sched` scale up to the prompt size (whole prefill
consumed in one step). The runner internally chunks `K_sched` into
`K_sched / K_kernel` PREFILL sub-seqs that all run inside one
`pallas_call` per case per layer. Single-request 8K-prompt TTFT
goes from `32 × per-step` to `~1 × per-step`. Multi-request: the
queue clears proportionally faster.

### 1.2 What the kernel needs to support

Today the kernel iterates `@pl.loop(start_seq_idx, end_seq_idx)` once
per real (physical) request scheduled this step. Each iteration:

  - reads its own `kv_lens_ref[seq_idx]`, `cu_q_lens_ref[seq_idx]`
  - looks up its physical pages via
    `page_indices_ref[seq_idx * pages_per_seq + i]`
  - runs attention against the physical KV cache
  - writes new K/V back into the same physical pages

The runner side (subseq_planner.plan_step) already produces a
StepPlan describing how to split each scheduled requests `n_R` tokens
into PREFILL chunks of size `K_kernel` plus a MIXED remainder. To make
the kernel actually consume that split, we need an indirection from
the loop variable to the underlying physical sequence — so multiple
logical iterations can share the same physical pages.

The reverse problem (synthesize many small physical seqs into one
logical iter — the "Spotify case") is **out of scope** for this PR
but worth keeping in mind, because the physical-vs-logical split has
to be modelled cleanly enough that the synthesize direction can land
without rework. See §7 below.

### 1.3 Design tenet — memory is physical, compute is logical

This is the load-bearing principle that drives every later design
decision. **State it once here and apply it everywhere.**

> Memory-state arrays (`kv_cache`, `page_indices`, `kv_lens`,
> `cu_q_lens`) stay sized and indexed by **physical** seq slot
> regardless of how the kernel iterates. Per-step compute metadata
> for the logical-iter view (`real_seq_idx_per_iter`,
> `q_offset_per_iter`) is added as **small** iter-keyed scalar
> prefetches; the kernel uses these to translate `iter_idx →
> physical seq + chunk position` on-the-fly and reads phys-keyed
> SMEM as the source of truth.

Concretely this means we choose the **kernel-side recomputation
path** rather than pre-computing per-iter `kv_lens` / `cu_q_lens`
runner-side and re-keying them logical:

  - `kv_lens_ref` stays `[max_num_seqs]` (PHYSICAL).
  - `cu_q_lens_ref` stays `[max_num_seqs + 1]` (PHYSICAL).
  - `page_indices_ref` stays `[max_num_seqs * max_pages_per_seq]` (PHYSICAL).
  - **NEW** `real_seq_idx_per_iter_ref: [max_num_subseqs]` — iter → phys slot map.
  - **NEW** `q_offset_per_iter_ref: [max_num_subseqs]` — tokens of the
    physical seq already consumed by previous iters (chunk metadata).
  - Kernel does ~5 SMEM loads + ~7 scalar arithmetic ops per iter to
    derive `(iter_q_start, iter_q_end, iter_q_len, iter_kv_len,
    iter_kv_q_gap)` from the above.

Why this path (vs. iter-keying `kv_lens` / `cu_q_lens`):

  1. **SMEM efficiency at the worst-case decoupling ratio.** With
     `K_sched=8192, K_kernel=256` (the design point that maximises
     TTFT win), `max_num_subseqs / MNS ≈ 33×`. Re-keying both
     `kv_lens` and `cu_q_lens` to iter-sized would cost ~32 KiB extra
     SMEM. Keeping them phys-sized + adding one extra iter-keyed
     `q_offset_per_iter` saves ~16 KiB at worst case — directly
     pushing the SMEM-validator's hard-fail boundary further out.
  2. **Synthesize-compatible.** The iter→phys map is the right
     primitive for the future synthesize case too; only the chunk
     metadata gets richer.
  3. **Cheap kernel arithmetic.** Pallas SMEM is single-cycle scalar.
     The extra ~7 ops/iter are amortised against the rest of the
     attention compute.

---

## 2. Conceptual scaffold: the four layers

This whole effort lives at the boundary between scheduler logic and
kernel logic. Before getting into kernel arrays, it pays to be precise
about *which layer* each variable lives at, so we don't conflate "what
the user wants" with "what the kernel needs" with "what the hardware
actually does."

The four layers, framed as goals — each layer's job is to satisfy the
layer above it:

  1. **Workload** — what the *user* wants done. Per-request, model-
     independent: prompt tokens, target output, deadlines.

  2. **HBM** — what the *scheduler* wants done. Bridge from workload:
     given the user's goals, the scheduler decides "to best achieve
     what the user wants done, I need *these* tokens written into KV
     this step." Materialized as DRAM-resident buffers and per-step
     metadata.

  3. **VMEM** — what the *kernel* wants done. Bridge from HBM: given
     what the scheduler asked for, the kernel author decides "to
     achieve that, I need *these* tiles staged into the TPU's fast
     SRAM in *this* loop nest."

  4. **VReg** — what the *hardware actually does*. Bridge from VMEM:
     given the kernel's tile structure, LLO (Google's low-level
     compiler) decides "to do what VMEM wants done, I issue the
     compute *this way* per cycle."

Each layer's static dimensions are *contracts* with the layer below.
Static dims that change → recompile (or worse, a misshaped buffer).
Dynamic dims (everything in scalar prefetch and the ragged Q axis)
flex at runtime.

Naming convention: at HBM we use vLLM engine-config names (MNS, MNB,
MML, page_size, K_sched). At VMEM we use `b<axis>_sz` block-tile sizes
(bq_sz, bkv_sz, bd_sz, …). At VReg we use `b<axis>_csz` compute-tile
sub-blocks (bq_csz, bkv_csz, bd_csz, …).

Hardware-portability note: the Workload column is fully hardware-
agnostic (pure user intent), and HBM is mostly so — these are vLLM
engine-config and persistent-batch concepts shared across backends;
only the physical KV-cache layout differs by hardware. VMEM and VReg
are TPU-specific in name and mechanism. A GPU port of this table
would copy column 1 verbatim, mostly preserve column 2 (with a
KV-layout swap), and rewrite columns 3–4 in terms of SMEM / warp
registers / MMA fragments. The decoupled-K design itself is *not*
TPU-specific — only its kernel implementation is.

### Variable lifetime across layers

Rows are grouped by *origin* — the leftmost layer where the concept
first exists. `—` means the concept doesn't manifest at that layer.

#### Origin at Workload (cascades all the way down)

| Workload | HBM (Per Step) | VMEM (Per Block) | VReg (Per Tile) |
|---|---|---|---|
| `seq_len[i]` (per-request total context length) | Lifetime cap: MML<br>Per step cap: `q_len[i] ≤ min(`<br>* `K_sched,` ← per-req per-step (chunked prefill on)<br>* `prompt_len[i] − chunks_processed,` ← remaining-prompt cap (prefill only)<br>* `MNB − Σ q_len[j] for j ≠ i,` ← per-step total budget<br>* `MML − kv_len[i - 1]` ← transitive lifetime cap<br>`)`<br>Relationship: kv_len[i] = kv_len[i-1] + q_len[i] | See childern below | See childen below |
| *`seq_len[i]` Stateless:* | `q_len[i]` (new tokens for this request this step; 1=decode, N=prefill chunk)<br>- Runtime: read from `cu_q_lens` (in SMEM) | `bq_sz`<br>= `1` for DECODE<br>= `K_kernel` for PREFILL<br>< `K_kernel` for MIXED | `bq_csz`|
| *`seq_len[i]` Accumulative:* | `kv_lens[i]` (incl. tokens processed in this step) | `bkv_sz` | `bkv_csz` |
| `num_seqs` (batch processing) | Bound: `MNS`.<br>Runtime: distribution `[D, D+P, T]` in `distribution_ref` | sequential `@pl.loop(start, end)` — no block dim | — |
| `total_q_tokens` this step (`Σq_len[i]`) | Bound: `MNB`.<br>Runtime: `cu_q_lens[-1]` | Walked via Q-axis stride; `bq_sz` per iter | `bq_csz` |
| `req_ids[i]` (per-request identity) | CPU mirror: `req_ids: list[str]`, `req_id_to_index: dict[str, int]` (req_id → physical-seq slot in `[0, num_seqs)`).<br>On-device: per-seq metadata arrays in SMEM (scalar prefetch), each indexed by slot — see children below. | `seq_idx` is the `@pl.loop` counter; one iter per physical seq because today `max_num_subseqs == MNS`. See children for per-array use. | — |
| *`req_ids[i]` → `kv_lens`:* | `kv_lens: i32[MNS]` in SMEM — KV length per seq at end of step.<br>Runtime form of `kv_len[i]` from `seq_len`'s Accumulative child. | — | — |
| *`req_ids[i]` → `cu_q_lens`:* | `cu_q_lens: i32[MNS + 1]` in SMEM — cumulative Q offsets.<br>Runtime form of `q_len[i]` from `seq_len`'s Stateless child. | — | — |
| *`req_ids[i]` → `page_indices`:* | `page_indices: i32[MNS * max_pages_per_seq]` in SMEM — flattened physical-page map; static, max-possible page counts; req i's j-th page = `page_indices[i * max_pages_per_seq + j]`. | `page_indices_ref[seq_idx * max_pages_per_seq + j]` resolves logical → physical | — |

#### Origin at HBM (model-architectural or scheduler-allocated)

| Concept | Workload | VMEM | VReg |
|---|---|---|---|
| `num_q_heads` (H_q), `num_kv_heads` (H_kv) | — (model property) | `bh_sz` if relevant | `bh_csz` |
| `head_dim` (D) | — | `bd_sz` (often = D) | `bd_csz` |
| `page_size` | — | `bkv_sz` is typically a multiple of `page_size` | — |
| `max_pages_per_seq` = ceil(MML/page_size) (referred to as `pages_per_seq` in `kernel.py` — same quantity, JIT-time static, max-possible page slots per seq) | — | shape of `page_indices_ref` slot | — |
| `kv_cache` [total_pages, page_size, H_kv, D, 2] | — (user provides tokens, not KV) | `K_tile`, `V_tile` of size `bkv_sz × bd_sz` per iter | `K_subtile`, `V_subtile` |
| `page_indices` (per-seq → physical-page map) | — | scalar prefetch `page_indices_ref[seq_idx * max_pages_per_seq + j]` resolves logical → physical | — |
| `kv_lens` (per-seq KV length at end of step) | — | scalar prefetch — bounds the bkv loop per sub-seq | — |
| `cu_q_lens` (cumulative q lens) | — | scalar prefetch — bounds the bq loop per sub-seq; slices Q from the ragged buffer | — |
| `distribution` `[D, D+P, T]` | — | scalar prefetch — provides `start_seq_idx, end_seq_idx` for `pl.loop` per case | — |
| `K_kernel` (PREFILL static_q_len) | — (decoupled-K only — invisible to workload) | static Q-axis size of the PREFILL pass; upper bound on `bq_sz` | — |

#### Origin at VMEM (kernel author's choice)

| Concept | Workload | HBM | VReg |
|---|---|---|---|
| `bq_sz`, `bkv_sz`, `bh_sz`, `bd_sz` | — | — (baked into the pallas_call at trace time; runner sees them only through tuning records) | sub-tiled into `*_csz` for VReg compute |
| pipeline-stage count (e.g. K/V double-buffering) | — | — | scheduling choice the LLO honors |

#### Origin at VReg (LLO compiler, hinted by kernel)

| Concept | Workload | HBM | VMEM |
|---|---|---|---|
| `bq_csz`, `bkv_csz`, `bd_csz`, `bh_csz` | — | — | inner sub-tile of the corresponding `*_sz` VMEM tile |
| issue width / register pressure | — | — | implicit in the kernel's loop nest, refined by LLO |

### Sanity checks the scaffold buys us

Every concrete number should be assignable to *one* layer. Confusion
is almost always a layer-mixing error:

  - "32K-context model" → Workload wants `seq_len_i ≤ 32K` → HBM sets
    `MML=32768` → VMEM walks up to `ceil(32768 / bkv_sz)` bkv iters →
    VReg does that compute in `bkv_csz` chunks.

  - "Chunked prefill on" → Workload doesn't know this exists → HBM
    scheduler grants `K_sched` per request per step → VMEM receives a
    Q slab of size at most `K_kernel` (today: K_kernel=K_sched) →
    VReg in `bq_csz` chunks.

  - "Decoupled K" → HBM may grant `K_sched > K_kernel`; VMEM still
    specializes to `K_kernel`; the new runner-level scaffolding
    (subseq_planner + real_seq_idx_per_iter indirection) chunks the
    workload-level `q_len_i` into K_kernel-sized sub-seqs *before*
    HBM hands work to VMEM. The scaffold makes clear why the planner
    lives at the HBM↔VMEM boundary and why the kernel only needs an
    indirection table, not new math.

---

## 3. Vocabulary

We will be strict about these terms throughout:

  - **Physical sequence (real seq):** an entry in vLLMs persistent
    batch with a unique `req_id`, a stable slot index in
    `input_batch.req_ids`, and a contiguous range of physical pages in
    the KV pool. Count = `max_num_seqs` (= `MAX_NUM_SEQS` in the vLLM
    scheduler config).

  - **Logical sub-sequence (subseq):** one iteration of the kernels
    `@pl.loop`. Each iteration processes some contiguous slice of one
    physical sequences q-tokens. Count = `max_num_subseqs`. In todays
    coupled-K kernel, `max_num_subseqs == max_num_seqs` (one iter per
    physical seq). Under decoupled-K, `max_num_subseqs > max_num_seqs`
    when chunking is active.

  - **Iteration index (iter_idx, formerly seq_idx):** the loop variable
    of `@pl.loop`. Range = `[start_seq_idx, end_seq_idx)` taken from
    `distribution_ref`. Conceptually a logical sub-seq index.

  - **Physical seq slot (real_seq_idx):** an index into the persistent
    batch ordering. Range = `[0, max_num_seqs)`.

  - **The iter→phys mapping:**
    ```
    real_seq_idx_per_iter_ref: i32[max_num_subseqs]
    real_seq_idx = real_seq_idx_per_iter_ref[iter_idx]    # in [0, max_num_seqs)
    ```

  - **The chunk-position metadata (per iter):**
    ```
    q_offset_per_iter_ref: i32[max_num_subseqs]
    q_offset = q_offset_per_iter_ref[iter_idx]
    # = number of new q-tokens of this physical seq already
    #   consumed by previous iters (chunk index × K_kernel for
    #   PREFILL chunks; can also encode MIXED-tail offsets).
    # First iter of a phys seq: q_offset = 0.
    ```

    Together with the phys-keyed `cu_q_lens_ref` and `kv_lens_ref`,
    this is enough for the kernel to derive every per-iter quantity
    (see §6.2 for the derivation).

Under this design (decoupled-K), `kv_lens_ref` and `cu_q_lens_ref`
**stay sized by `max_num_seqs`** (physical). Only
`real_seq_idx_per_iter_ref` and `q_offset_per_iter_ref` are sized by
`max_num_subseqs` (logical). See §1.3 (design tenet) for rationale.

We will rename the kernel-internal variable currently named
`max_num_seqs` to `max_num_subseqs` **only at the iter-counter
sites** (the `@pl.loop` bound, the iter-keyed prefetch shapes). The
existing phys-keyed array shapes still use `max_num_seqs`. See §8
for the precise rename map.

---

## 4. Inventory of every array in the kernel

Source: `tpu_inference/kernels/ragged_paged_attention/v3/kernel.py`.
Symbols used in shapes:
  - `MNT` = `max_num_tokens` (= `MAX_NUM_BATCHED_TOKENS`)
  - `H_kv` = `actual_num_kv_heads`
  - `D` = `head_dim`
  - `KV2` = `num_kv_heads_x2` (= `2 * H_kv` padded for packing)
  - `Q_per_KV` = `num_q_heads_per_kv_head` (the GQA ratio)
  - `q_pack`, `kv_pack` = dtype packing factors

| # | Name | Tier | Shape | Sized by | Dep class |
|---|---|---|---|---|---|
| 1 | `q_hbm_ref` (input Q) | HBM | `[H_kv, MNT, Q_per_KV/q_pack, q_pack, D]` | `MNT` | TOKEN |
| 2 | `kv_hbm_ref` (input K/V for this step) | HBM | `[MNT, KV2/kv_pack, kv_pack, D]` | `MNT` | TOKEN |
| 3 | `kv_cache_hbm_ref` (in/out aliased) | HBM | `[total_num_pages, page_size, KV2/kv_pack, kv_pack, D]` | `total_num_pages` | PHYSICAL pool |
| 4 | `o_hbm_ref` (output) | HBM | same as q_hbm_ref | `MNT` | TOKEN |
| 5 | **`kv_lens_ref`** | SMEM | `[N_iter]` | iter count | **LOGICAL** |
| 6 | **`page_indices_ref`** | SMEM | `[N_real * pages_per_seq]` flat | physical seq count × `pages_per_seq` | **PHYSICAL** |
| 7 | **`cu_q_lens_ref`** | SMEM | `[N_iter + 1]` | iter count + 1 | **LOGICAL** |
| 8 | `distribution_ref` | SMEM | `[3]` | constant `(D, D+P, T)` iter slice points | step-state (logical slice) |
| 9 | `sem_ids_ref` | SMEM | `[3]` | constant | step-state |
| 10 | `bo_ids_ref` | SMEM | `[4]` | constant | step-state |
| 11 | `bkv_update_ids_ref` | SMEM | `[6]` | constant | step-state |
| 12 | `bkv_x2_ref` | VMEM | `[2, bkv_sz, KV2/kv_pack(+1), kv_pack, D]` | block sizes | scratch |
| 13 | `bq_x2_ref` | VMEM | `[2, H_kv, bq_sz, Q_per_KV/q_pack, q_pack, D]` | block sizes | scratch |
| 14 | `bo_x2_ref` | VMEM | same as bq_x2_ref | block sizes | scratch |
| 15 | `sems` | semaphores | `[4, 2]` | constant | scratch |
| 16 | `l_ref`, `m_ref` | VMEM | `[H_kv, bq_sz × Q_per_KV, 128]` | block sizes | scratch (flash-attn state) |
| 17 | `acc_ref` | VMEM | `[H_kv, bq_sz × Q_per_KV, D]` | block sizes | scratch |

**NEW under decoupled-K:**

| # | Name | Tier | Shape | Sized by | Dep class |
|---|---|---|---|---|---|
| 18 | **`real_seq_idx_per_iter_ref`** | SMEM | `[N_iter]` | iter count | **LOGICAL → physical** map (length logical, values physical) |

(`N_iter = max_num_subseqs`, `N_real = max_num_seqs`. Today they are
equal; under decoupled-K, `N_iter ≥ N_real`.)

---

## 5. The four dependence classes

Re-stated in plain terms:

  - **TOKEN-sized** (q_hbm_ref, kv_hbm_ref, o_hbm_ref). Sized by the
    per-step token budget. Independent of how seqs are organized.
    Indexed by `cu_q_lens_ref[iter_idx]` to find a particular iters
    q tokens.

  - **PHYSICAL pool** (kv_cache_hbm_ref). Sized by the global KV pool
    page count. Independent of seq concept entirely. Pages within a
    seq located via `page_indices_ref` lookup.

  - **PHYSICAL-seq-sized** (page_indices_ref). One row of `pages_per_seq`
    physical page numbers per real seq. **Sized by the count of real
    sequences, NOT iterations.** The whole point of the decoupled-K
    indirection is to keep this small even when iter count blows up.

  - **LOGICAL-seq-sized** (kv_lens_ref, cu_q_lens_ref,
    real_seq_idx_per_iter_ref). One entry per loop iteration. Sized
    by iter count. **Inflates with chunking.**

  - Step-state (distribution, sem ids) and VMEM scratch are constant
    or block-size-bounded; out of scope.

---

## 6. The split case (PR3 — current)

**One big physical seq → multiple logical iters.**

### 6.1 Worked example

Physical seq `R0` has `n_R0 = 800` prefill tokens scheduled this
step, with `K_kernel = 256` and prior_kv_len_R0 = 5000 (R0 already
has 5000 tokens of context from earlier steps). Runner emits 3
PREFILL-bucket iters + 1 MIXED-bucket remainder iter, all referring
to physical seq slot `0`.

The runner fills two iter-keyed prefetches:

| Iter | Bucket | `real_seq_idx_per_iter_ref[iter]` | `q_offset_per_iter_ref[iter]` |
|---|---|---|---|
| 0 | PREFILL chunk 0 | 0 | 0 |
| 1 | PREFILL chunk 1 | 0 | 256 |
| 2 | PREFILL chunk 2 | 0 | 512 |
| 3 | MIXED tail      | 0 | 768 |

The phys-keyed prefetches (sized by `max_num_seqs`) hold the
end-of-step quantities for R0 — same semantics as today's kernel:

| Phys slot | `cu_q_lens_ref[s]` | `cu_q_lens_ref[s+1]` | `kv_lens_ref[s]` |
|---|---|---|---|
| 0 (R0)    | 0 | 800 | 5800 |

`page_indices_ref` stays sized `[max_num_seqs * max_pages_per_seq]`,
and slot 0 holds R0's physical page list. **No duplication** — all
4 iters look up R0's pages via the iter→phys map.

### 6.2 What the kernel derives per iter

At the top of each `@pl.loop` iter (replacing today's reads at
`kernel.py:379-383`):

```python
real_seq_idx = real_seq_idx_per_iter_ref[iter_idx]   # SMEM load
q_offset     = q_offset_per_iter_ref[iter_idx]       # SMEM load

phys_q_start = cu_q_lens_ref[real_seq_idx]           # SMEM load (PHYS)
phys_q_end   = cu_q_lens_ref[real_seq_idx + 1]       # SMEM load (PHYS)
phys_q_len   = phys_q_end - phys_q_start             # arithmetic

# This iter's q range within the ragged Q buffer
iter_q_start = phys_q_start + q_offset                                # arithmetic
# iter_q_len: case-dependent (kernel author knows which case at trace time)
#   PREFILL: iter_q_len = K_kernel       (= static_q_len)
#   DECODE:  iter_q_len = 1
#   MIXED:   iter_q_len = phys_q_len - q_offset      (the leftover)
iter_q_end   = iter_q_start + iter_q_len             # arithmetic

phys_kv_end       = kv_lens_ref[real_seq_idx]        # SMEM load (PHYS)
phys_prior_kv_len = phys_kv_end - phys_q_len         # arithmetic
iter_kv_q_gap     = phys_prior_kv_len + q_offset     # arithmetic — write-start for this iter
iter_kv_len       = iter_kv_q_gap + iter_q_len       # arithmetic — visible kv-end for this iter
```

Once these are computed, the rest of the kernel body is
**byte-identical to today** — it just uses `iter_q_start /
iter_q_end / iter_q_len / iter_kv_len / iter_kv_q_gap` where today's
code uses the same names without the `iter_` prefix.

For our worked example:

| Iter | `q_offset` | iter_q_start | iter_q_len | iter_q_end | iter_kv_q_gap | iter_kv_len |
|---|---|---|---|---|---|---|
| 0 | 0   | 0   | 256 (K_kernel) | 256 | 5000 | 5256 |
| 1 | 256 | 256 | 256            | 512 | 5256 | 5512 |
| 2 | 512 | 512 | 256            | 768 | 5512 | 5768 |
| 3 | 768 | 768 |  32 (leftover) | 800 | 5768 | 5800 |

### 6.3 KV-write correctness invariant: disjoint tiling per phys seq

KV writes compose into disjoint, contiguous ranges within each phys
seq's physical pages **only because** the per-iter derivation above
produces non-overlapping `[iter_kv_q_gap, iter_kv_len)` ranges that
tile `[phys_prior_kv_len, phys_kv_end)`. The invariant relies on:

  1. **Runner-side correctness of `q_offset_per_iter`** — must be
     monotonically increasing across consecutive iters of the same
     phys seq, with `q_offset = 0` at the first iter of each phys
     seq. Specifically: for iters belonging to phys `R`,
     `q_offset[i+1] = q_offset[i] + iter_q_len[i]`.
  2. **`real_seq_idx_per_iter` correctness** — must group iters of
     the same phys seq contiguously in iter-order, so successive
     `_update_kv_cache` calls for the same seq write monotonically
     advancing offsets.

If `q_offset_per_iter` is wrong (e.g. all zeros for R0's chunks
instead of `[0, 256, 512, 768]`), every iter writes at the same
position → **silent KV-cache corruption**.
If `real_seq_idx_per_iter` interleaves iters of different physical
seqs while still sharing pages, the same hazard applies.

The runner-side `subseq_planner.plan_step` already enforces both
invariants (chunks of one real req are emitted contiguously,
`q_offset` is monotonic). Verify this in the implementation tests
(see §11 for the strict-verification checklist).

This section covers WRITES only — that the per-iter writes don't
collide with each other. The orthogonal concern (one iter's READ
racing against an EARLIER iter's WRITE on the same `kv_cache_hbm_ref`
cells) is handled in §6.4.

### 6.4 Same-call cross-iter KV read ordering: route in-step reads through `kv_hbm_ref`

Decoupled-K introduces a NEW concern that coupled-K never had: each
iter's KV READ may want cells that an EARLIER iter just wrote.

For R0 with 800 prefill tokens at K_kernel=256 (3 PREFILL chunks):

  - Iter 0 writes R0's chunk-0 K/V to `kv_cache_hbm_ref` at offsets
    `[prior_kv_len, prior_kv_len + 256)` via `_update_kv_cache`.
  - Iter 1's Q attends against `KV[0, prior_kv_len + 512)`. Today's
    `_fetch_bkv` formula
    (`kv_left_frm_cache = max(kv_left - q_len, 0)`) routes cells
    `[prior_kv_len, prior_kv_len + 256)` through `kv_cache_hbm_ref`
    — i.e., reads what iter 0 just wrote.

The `_update_kv_cache` write is async on `sems.at[3, bkv_sem_idx]`;
the `_fetch_bkv` read is async on `sems.at[0, bkv_sem_idx]`. **Different
semaphore groups, no cross-sync.** Race.

#### Why coupled-K is safe and decoupled-K is not

Today's pallas_call has no read-after-write hazard at the `@pl.loop`
level not because of any sync primitive, but because **no two iters
of one pallas_call read or write overlapping `kv_cache_hbm_ref`
cells**. Different phys seqs at different bucket positions; their
page lists don't overlap. Cross-pallas_call ordering (DECODE →
PREFILL → MIXED, and step → step) is enforced by the pallas_call's
drain-on-exit semantics (within a `@jax.jit`) and the @jax.jit
boundary itself (across calls).

Decoupled-K removes the "different phys seqs at different positions"
property: multiple iters of one pallas_call now touch the SAME phys
seq's `kv_cache_hbm_ref` cells. The same `@jax.jit` and pallas_call
boundaries still hold; only the @pl.loop level is exposed.

`input_output_aliases` does NOT help — it makes the kv_cache write
in-place on HBM, but in-place ≠ ordered. Adding a per-iter
`wait_update_kv_cache` would serialize the writeback against the
next iter's read and kill async-DMA pipelining (the very property
that makes the kernel fast).

#### The fix: source in-step reads from `kv_hbm_ref`, not `kv_cache_hbm_ref`

`kv_hbm_ref` is the kernel's per-step input from upstream Wk/Wv
layers (`kernel.py:1182`, `prepare_inputs`). It is:

  - Populated BEFORE the kernel runs.
  - Read-only inside the kernel — never written, never aliased.
  - Holds R0's full 800 new K/V tokens at
    `[phys_q_start, phys_q_end)` for the entire pallas_call.

So iter 0's K/V exists in TWO HBM locations at the time iter 1 runs:

  - `kv_cache_hbm_ref` at `[prior_kv_len, prior_kv_len + 256)` —
    racy (iter 0's writeback may be in flight).
  - `kv_hbm_ref` at `[phys_q_start, phys_q_start + 256)` —
    race-free (read-only, populated by upstream before kernel start).

Routing iter 1's read of iter 0's chunk to `kv_hbm_ref` eliminates
the race architecturally. No semaphore, no wait, no extra VMEM. The
property becomes structural: `kv_cache_hbm_ref` is effectively
write-only within a single pallas_call (writes for cross-step
persistence), and read-only for genuinely-pre-step content.

#### The formula change

Today's `_fetch_bkv` split (`kernel.py:561-567`):

```python
kv_left = kv_len - kv_len_start
kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
kv_left_frm_new   = kv_left - kv_left_frm_cache
```

Reads "the last `q_len` cells of `kv_left` from `kv_hbm_ref`; the
rest from `kv_cache_hbm_ref`." Under coupled-K, `q_len = phys_q_len`
— all of this seq's step contribution gets routed through
`kv_hbm_ref`. No in-step `kv_cache_hbm_ref` reads. Safe.

Under Path C decoupled-K with the §9.2 rename, `q_len` becomes
`iter_q_len` (per-iter, e.g. K_kernel=256 for PREFILL). The formula
now routes only iter i's OWN chunk through `kv_hbm_ref`; earlier
iters' chunks fall into `kv_left_frm_cache` and get sourced from
`kv_cache_hbm_ref`. **That's the race.**

The fix: read "the last `q_offset + iter_q_len` cells from new"
— i.e., all of THIS PHYS SEQ's tokens this step (cumulative across
iters), not just this iter's:

```python
kv_left = kv_len - kv_len_start
kv_left_frm_cache = jnp.maximum(kv_left - (q_offset + q_len), 0)
kv_left_frm_new   = kv_left - kv_left_frm_cache
```

In coupled mode, `q_offset = 0` and `q_len = phys_q_len` — the
formula reduces to today's. Byte-identical behaviour.

In decoupled mode, `q_offset + q_len` correctly tracks "how many
of this phys seq's tokens are now in `kv_hbm_ref` and ready for
read". For iter i of phys seq R: cells
`[phys_prior_kv_len, phys_prior_kv_len + q_offset + iter_q_len)` are
sourced from `kv_hbm_ref` at offsets
`[phys_q_start, phys_q_start + q_offset + iter_q_len)`. The
`new_kv_len_start` formula at `kernel.py:607`
(`q_end - kv_left_frm_new`) lands at the correct kv_hbm_ref offset
because `q_end` under §9.2 is
`phys_q_start + q_offset + iter_q_len`.

#### What stays the same

  - `_update_kv_cache` (the writeback) is still needed for cross-step
    persistence — the next scheduler step's `_fetch_bkv` sees this
    step's new tokens in `kv_cache_hbm_ref` as "history". The
    pallas_call's drain-on-exit ensures writes complete before the
    next pallas_call runs. Cross-pallas_call ordering: unchanged.
  - All existing async-DMA pipelining: unchanged.
  - `bkv_x2_ref` double-buffering: unchanged.
  - No new VMEM allocation, no new semaphore, no new wait.

#### Reference: the general pattern from `emit_pipeline`

The Pallas TPU team's `emit_pipeline.Scheduler.wait_out`-on-`prev_indices`
pattern (jax-ml/jax `mosaic/pipeline.py:1370-1378`) is the GENERAL
solution for inter-iter HBM ordering: "drain the previous iter's
copy_out at the start of the next iter that reads the same window."
Option A sidesteps the need for it because, for our specific access
shape, there's a read-only HBM mirror (`kv_hbm_ref`) of the cells
that would otherwise need synchronization. If a future case lacks
such a mirror, the `wait_out`-on-`prev_indices` pattern is the
fallback.

---

## 7. The synthesize case (future, e.g. Spotify-style short prefills)

**Multiple short physical seqs → fewer logical iters by batching.**

Example: 8 physical seqs `R0..R7`, each with 32 prefill tokens. Naively
each is its own iter (8 iters total). For better MXU utilization wed
batch them into one or more iters that each cover several physicals.

This case **breaks the iter ↔ single-physical-seq assumption.** A
single iter would need to:

  - Know about *N* physical seqs, not 1.
  - Have a list of `(real_seq_idx_k, q_offset_k, q_len_k, kv_len_k)`
    tuples for each k in `0..N-1`.
  - Write KV to N different physical page sets in one iter.
  - Produce N separate output ranges in `o_hbm_ref`.

`real_seq_idx_per_iter_ref` (one scalar per iter) is **insufficient**
for this case. Wed need either:

  - A list-per-iter prefetch (variable-length, awkward in Pallas), or
  - A new kernel "case" (e.g. `MIXED_SHORT`) with a fundamentally
    different inner-loop structure that processes a fixed batch of
    short seqs per iter via masking.

**Implication for the current PR3:** the indirection scheme we are
building is **specific to the split case**. We must not paint
ourselves into a corner that prevents adding the synthesize case
later. Concretely:

  - Keep `real_seq_idx_per_iter_ref` named generically enough that
    its meaning is "physical seq this iter belongs to" — not
    "the only physical seq this iter touches". For the synthesize
    case wed need a different mechanism layered on top, not a
    rename.
  - Keep page_indices_ref sized strictly by physical seq count
    (not iter count). This already holds.
  - Keep `kv_lens_ref` and `cu_q_lens_ref` per-iter (length
    `N_iter`). For the synthesize case, an iter could have
    cu_q_lens spanning multiple physical seqs token ranges; the
    array is the right shape regardless.

---

## 8. Naming proposal

The kernel-internal variable currently called `max_num_seqs` is
derived from `kv_lens.shape[0]`, which is the **iter count**, not
the physical seq count. We rename for clarity:

| Concept | Old name (today) | New name |
|---|---|---|
| Iter count (logical) | `max_num_seqs` | **`max_num_subseqs`** |
| Physical seq count | n/a (conflated with iter count today) | **`max_num_seqs`** (kwarg) |

In todays coupled-K mode the two are equal, so the rename is a
no-op behaviour-wise. Under decoupled-K they diverge.

Name search-and-replace inside `kernel.py`:

  - Line 131: `max_num_seqs = kv_lens.shape[0]` →
    `max_num_subseqs = kv_lens.shape[0]`
  - Lines 132-134: assertion + `pages_per_seq` derivation use the
    new physical `max_num_seqs` kwarg with fallback.
  - Line 363, 1704: same pattern.

External (caller-facing) name `max_num_seqs` stays — referring to
the physical count. New kernel kwarg `max_num_seqs: int | None =
None` (None → fallback to `max_num_subseqs`).

---

## 9. Proposed kernel signature change

### 9.1 `_ragged_paged_attention_kernel_loop` signature (around line 291)

Insert **two** new scalar prefetches — `real_seq_idx_per_iter_ref`
and `q_offset_per_iter_ref` — after `page_indices_ref`. Crucially,
`kv_lens_ref` and `cu_q_lens_ref` keep their phys-keyed shapes
(unchanged from today). Updated comments make the keying explicit:

```python
def _ragged_paged_attention_kernel_loop(
    iter_idx,                  # was seq_idx — now logical sub-seq idx
    # Prefetch (SMEM)
    kv_lens_ref,                # [max_num_seqs]               PHYSICAL (unchanged shape)
    page_indices_ref,           # [max_num_seqs * max_pages_per_seq]  PHYSICAL (unchanged)
    real_seq_idx_per_iter_ref,  # [max_num_subseqs]            LOGICAL→PHYSICAL  (NEW)
    q_offset_per_iter_ref,      # [max_num_subseqs]            LOGICAL chunk-pos (NEW)
    cu_q_lens_ref,              # [max_num_seqs + 1]           PHYSICAL (unchanged shape)
    distribution_ref,           # [3] iter-slice (D, D+P, T)
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    # ...
```

### 9.2 Loop-body derivation block (replaces lines 379-383, mirrored at line 554-559)

Today the kernel reads its per-iter quantities directly:

```python
q_start = cu_q_lens_ref[seq_idx]
q_end   = cu_q_lens_ref[seq_idx + 1]
q_len   = q_end - q_start
kv_len  = kv_lens_ref[seq_idx]
kv_q_gap = kv_len - q_len
```

Under decoupled-K it derives them from phys + chunk metadata:

```python
real_seq_idx = real_seq_idx_per_iter_ref[iter_idx]   # iter → phys
q_offset     = q_offset_per_iter_ref[iter_idx]       # tokens already consumed by prior iters of this phys

# Phys-keyed reads (unchanged shape; today's array semantics)
phys_q_start = cu_q_lens_ref[real_seq_idx]
phys_q_end   = cu_q_lens_ref[real_seq_idx + 1]
phys_q_len   = phys_q_end - phys_q_start
phys_kv_end  = kv_lens_ref[real_seq_idx]
phys_prior_kv_len = phys_kv_end - phys_q_len

# Per-iter derived (rename what's read elsewhere in kernel: q_start→iter_q_start, etc.)
q_start = phys_q_start + q_offset
# q_len: case-dependent (kernel knows which case statically at trace time)
if case is PREFILL:
    q_len = static_q_len           # = K_kernel
elif case is DECODE:
    q_len = 1
elif case is MIXED:
    q_len = phys_q_len - q_offset  # the leftover after preceding chunks
q_end    = q_start + q_len
kv_q_gap = phys_prior_kv_len + q_offset    # write-start position
kv_len   = kv_q_gap + q_len                # visible kv-end for this iter
```

The variable names `q_start`, `q_end`, `q_len`, `kv_len`, `kv_q_gap`
deliberately mirror today's names, so **the rest of the kernel body
is byte-identical** — every later use of these variables sees the
correct per-iter values without further changes.

### 9.3 Page-index lookup (line 568, line 640)

Both sites currently:
```python
page_indices_offset = seq_idx * pages_per_seq + kv_p_start
```

Become:
```python
# real_seq_idx already in scope from §9.2
page_indices_offset = real_seq_idx * pages_per_seq + kv_p_start
```

### 9.4 `run_rpa_kernel` `scalar_prefetches` tuple (line 1778)

Insert the two new prefetches **after `page_indices`** (so phys
prefetches stay grouped):

```python
scalar_prefetches = (
    kv_lens,                  # PHYS (unchanged shape)
    page_indices,             # PHYS (unchanged shape)
    real_seq_idx_per_iter,    # NEW — iter → phys
    q_offset_per_iter,        # NEW — chunk position
    cu_q_lens,                # PHYS (unchanged shape)
    distribution,
    init_sem_ids,
    init_bo_ids,
    init_bkv_update_ids,
)
```

### 9.5 `run_rpa_kernel` `input_output_aliases` (line 1846)

Two new prefetches shift the `q` / `kv` / `kv_cache` positional args
by 2:

```python
input_output_aliases={
    9:  0,    # q -> output 0  (was index 7)
    11: 1,    # kv_cache -> output 1  (was index 9)
},
```

### 9.6 `ragged_paged_attention()` signature (line 1568)

```python
def ragged_paged_attention(
    queries, keys, values, kv_cache,
    kv_lens,                   # [max_num_seqs]                    (PHYSICAL)
    page_indices,              # [max_num_seqs * max_pages_per_seq] (PHYSICAL)
    real_seq_idx_per_iter,     # [max_num_subseqs]                 NEW (REQUIRED, positional)
    q_offset_per_iter,         # [max_num_subseqs]                 NEW (REQUIRED, positional)
    cu_q_lens,                 # [max_num_seqs + 1]                (PHYSICAL)
    distribution,              # [3]
    *,
    # ... existing kwargs ...
    max_num_seqs: int | None = None,   # NEW (optional, explicit physical count)
    # When None: kernel sets max_num_seqs = kv_lens.shape[0]
    # (today's behaviour, identity mapping where iter==real).
    # When set: phys count differs from iter count (decoupled-K).
):
```

`real_seq_idx_per_iter` and `q_offset_per_iter` are **required
positional** (no default) — breaks all existing callers until the
runner is updated to pass identity (`real_seq_idx_per_iter = arange(max_num_seqs)`,
`q_offset_per_iter = zeros(max_num_seqs)`) in coupled mode.

`max_num_seqs` is **optional kwarg** with default-derive fallback.

### 9.7 Body changes (lines 131-134, 363-365, 1704-1707)

Today:
```python
max_num_seqs = kv_lens.shape[0]
num_page_indices = page_indices.shape[0]
assert num_page_indices % max_num_seqs == 0
pages_per_seq = num_page_indices // max_num_seqs
```

Becomes (note: `kv_lens.shape[0]` is now `max_num_seqs`, not
`max_num_subseqs`, because `kv_lens` is phys-keyed):

```python
# kv_lens is PHYS-keyed; its length IS max_num_seqs by definition.
if max_num_seqs is None:
    max_num_seqs = kv_lens.shape[0]
else:
    assert kv_lens.shape[0] == max_num_seqs, "kv_lens is phys-keyed"

# real_seq_idx_per_iter is iter-keyed; its length is max_num_subseqs.
max_num_subseqs = real_seq_idx_per_iter.shape[0]
assert q_offset_per_iter.shape[0] == max_num_subseqs

num_page_indices = page_indices.shape[0]
assert num_page_indices % max_num_seqs == 0
max_pages_per_seq = num_page_indices // max_num_seqs   # PHYS-keyed denominator
```

### 9.8 `static_argnames` update

Add `max_num_seqs` to `@jax.jit(static_argnames=(...))` so it's baked
into the trace shape (it's a static dim, not a runtime tensor).

### 9.9 Coupled-mode (today) compatibility

When the caller is the un-modified runner (no decoupled-K), pass:
- `real_seq_idx_per_iter = arange(max_num_seqs, dtype=i32)` — identity.
- `q_offset_per_iter = zeros(max_num_seqs, dtype=i32)` — every "iter" is the first chunk.
- `max_num_seqs = None` (let kernel derive from `kv_lens.shape[0]`).

In this mode the derived per-iter values reduce to today's:
- `real_seq_idx = iter_idx`
- `q_offset = 0`
- `q_start = phys_q_start + 0 = today's q_start`
- `q_len = phys_q_len` (case=MIXED) or `K_kernel` / `1` for PREFILL/DECODE — matches today's static_q_len behaviour because every phys req is a single-iter chunk
- `kv_len = phys_prior_kv_len + phys_q_len = phys_kv_end` — matches today
- `kv_q_gap = phys_prior_kv_len = today's kv_q_gap`

So the new kernel is a strict generalization; coupled-mode behaviour is byte-equivalent to today after the rename. Coupled-mode is the validation ground for the decoupled-mode landing — see §11.

---

## 10. SMEM accounting under the new layout

Use `N_real = max_num_seqs`, `N_iter = max_num_subseqs`, `MPL =
max_pages_per_seq`. All SMEM scalar prefetches are `i32`, aligned to
128 lanes.

```
SMEM bits = align(N_real, 128) * 32             # kv_lens_ref          PHYS  (unchanged)
          + align(N_real * MPL, 128) * 32       # page_indices_ref     PHYS  (unchanged)
          + align(N_iter, 128) * 32             # real_seq_idx_per_iter_ref  NEW
          + align(N_iter, 128) * 32             # q_offset_per_iter_ref      NEW
          + align(N_real + 1, 128) * 32         # cu_q_lens_ref        PHYS  (unchanged)
          + 4 × (128 * 32)                      # distribution + sem_ids + bo_ids + bkv_update_ids
```

### 10.1 Worst-case numerical example

`N_real = 128`, `MPL = 128`, `N_iter = 4224` (= 128 × 33, the
`K_sched=8192, K_kernel=256` worst case):

| Term | Bytes |
|---|---|
| `kv_lens_ref` (PHYS) | 0.5 KiB |
| `page_indices_ref` (PHYS) | 64 KiB |
| `real_seq_idx_per_iter_ref` (NEW) | 16.5 KiB |
| `q_offset_per_iter_ref` (NEW) | 16.5 KiB |
| `cu_q_lens_ref` (PHYS) | 0.5 KiB |
| 4 small scratch | 2 KiB |
| **Total** | **~100 KiB** |

Compare to today: ~67.5 KiB. Δ = ~33 KiB.
Compare to Path B (per-iter `kv_lens` + `cu_q_lens`): ~115 KiB.
**Path C savings vs. Path B: ~15 KiB at worst case.**

### 10.2 Why phys-keying `kv_lens` and `cu_q_lens` is the right call

The dominant SMEM term remains `page_indices_ref` at
`N_real * MPL × 4 bytes` — phys-sized regardless of which path. The
per-iter quantities `kv_lens` and `cu_q_lens` were the temptation to
re-key as logical, but doing so would have multiplied them by the
same `N_iter / N_real ≈ 33×` factor that already hurts
`real_seq_idx_per_iter`. By keeping them phys-keyed and adding only
`q_offset_per_iter` (= 1 × N_iter array), we pay the iter-multiplier
on **2 arrays** (real_seq_idx_per_iter + q_offset_per_iter) instead
of **3 arrays** (Path B).

### 10.3 `subseq_planner.compute_smem_bytes` update

The runner-side SMEM estimator at `tpu_inference/runner/subseq_planner.py`
mirrors the kernel's `get_smem_estimate_bytes`. It needs to be
updated to:

  1. **Keep** the `kv_lens` and `cu_q_lens` terms phys-keyed
     (currently iter-keyed by accident — fix as part of PR3).
  2. **Add** the `q_offset_per_iter_ref` term (= `align(N_iter, 128) × 32`).
  3. **Add** the `real_seq_idx_per_iter_ref` term if not already present.

This affects `evaluate_decoupled_k_config`'s SMEM-clamp math.
Existing `achievable_M` numbers shift slightly (Path B → Path C
saves ~15 KiB worst case → `achievable_M` goes up by a small amount
because more iters now fit budget).

---

## 11. Open questions + strict-verification checklist

### 11.1 Open questions for the implementer to confirm

  1. **`pages_per_seq` derivation source.** Today the kernel computes
     `pages_per_seq = page_indices.shape[0] // kv_lens.shape[0]`,
     which works because `kv_lens.shape[0] == max_num_seqs`. Under
     this design `kv_lens` stays phys-keyed so the formula still
     holds — no change needed. (Confirmed: choosing this over a
     2-D `page_indices` reshape, which would be breaking for all
     callers.)

  2. **`static_validate_inputs` and `dynamic_validate_inputs`.**
     They currently assert `num_page_indices % kv_lens.shape[0] == 0`
     — still correct under Path C (kv_lens is phys-keyed). Add new
     asserts: `real_seq_idx_per_iter.shape[0] == q_offset_per_iter.shape[0]`
     and `real_seq_idx_per_iter.shape == q_offset_per_iter.shape`.

  3. **`kernel_hd64.py`.** Same surface change. Currently used only
     by `gpt_oss_attention`. Defer to a follow-up unless trivially
     parallel.

  4. **Synthesize forward-compat.** The split-case `real_seq_idx_per_iter`
     (one scalar per iter) is insufficient for synthesize. That's
     accepted — synthesize is its own future PR with `MIXED_SHORT`
     case + per-iter list machinery, layered on top.

### 11.2 Strict-verification checklist (post-implementation)

When the implementation lands, verify these items in order. **Each
is a hard yes/no, not a "looks ok".** Cite file:line for every
finding.

#### Coupled-mode regression (the safety net)

  - [ ] **Coupled mode is byte-equivalent to today.** With
        `real_seq_idx_per_iter = arange(max_num_seqs)`,
        `q_offset_per_iter = zeros(max_num_seqs)`, and
        `max_num_seqs = None`, the kernel must produce numerically
        identical output to today's kernel for any input. Verify
        by running the existing kernel test suite against the
        new kernel signature (with the runner stubbed to pass
        identity) — all green.
  - [ ] **No new compile.** Coupled-mode trace key must match
        today's (modulo the new prefetches' constant-shape
        contribution). Check `jax.make_jaxpr(...)` output for the
        coupled call before and after — no new dynamic shapes.

#### Phys-keying invariants

  - [ ] **`kv_lens.shape[0] == max_num_seqs`** holds in both modes.
        Not `max_num_subseqs`. Read the kernel-side body and confirm
        this assertion.
  - [ ] **`cu_q_lens.shape[0] == max_num_seqs + 1`** holds.
  - [ ] **`page_indices.shape[0] == max_num_seqs * max_pages_per_seq`** holds.

#### Iter-keying invariants

  - [ ] **`real_seq_idx_per_iter.shape[0] == max_num_subseqs`.**
  - [ ] **`q_offset_per_iter.shape[0] == max_num_subseqs`.**
  - [ ] **`max_num_subseqs >= max_num_seqs`** in all decoupled-mode
        configs (one iter per phys at minimum).

#### Per-iter derivation correctness

  Read the kernel body where `q_start`, `q_end`, `q_len`, `kv_len`,
  `kv_q_gap` are computed. For each, confirm the formula matches §6.2:

  - [ ] `q_start = cu_q_lens_ref[real_seq_idx] + q_offset`
  - [ ] `q_len = K_kernel` (PREFILL) / `1` (DECODE) /
        `cu_q_lens_ref[real_seq_idx+1] - cu_q_lens_ref[real_seq_idx] - q_offset` (MIXED)
  - [ ] `q_end = q_start + q_len`
  - [ ] `kv_q_gap = (kv_lens_ref[real_seq_idx] -
        (cu_q_lens_ref[real_seq_idx+1] - cu_q_lens_ref[real_seq_idx]))
        + q_offset`
  - [ ] `kv_len = kv_q_gap + q_len`

  Common bug to look for: using `iter_idx` instead of `real_seq_idx`
  in any of `cu_q_lens_ref[]`, `kv_lens_ref[]`, or `page_indices_ref[]`
  lookups. Grep the kernel body for these refs and verify each is
  indexed by `real_seq_idx`, not `iter_idx` / `seq_idx`.

#### KV-write composition (the silent-corruption hazard)

  Run a 2-iter split test (R0 with K_kernel = 256, n_R0 = 512, two
  PREFILL chunks). Confirm:

  - [ ] After iter 0, KV pages at offsets `[prior, prior+256)` hold
        the correct K_new tokens for chunk 0.
  - [ ] After iter 1, KV pages at offsets `[prior+256, prior+512)`
        hold chunk 1's K_new tokens. **Crucially, `[prior, prior+256)`
        is unchanged** (iter 1 must not overwrite iter 0).
  - [ ] iter 1's QK matmul attends to `[0, prior+512)`, not just
        `[0, prior+256)` — i.e., its kv_len is 512, not 256.

  This is best verified by a numerical comparison against a pure-JAX
  reference attention impl: run the same workload through both and
  assert `allclose`.

#### KV-read source routing (the §6.4 fix verification)

  Confirms iter 1's read of iter 0's chunk is sourced from `kv_hbm_ref`,
  NOT from `kv_cache_hbm_ref`. Without this property, the kernel hits
  a same-call cross-iter read-after-write race that depends on async-DMA
  timing — passes locally most of the time, fails sporadically under
  load.

  The "evil" test: same 2-iter split workload, but BEFORE invoking the
  kernel, deliberately corrupt `kv_cache_hbm_ref` at the offsets where
  iter 0 will write its chunk. With Option A in place, the kernel will
  never read those cells (it sources iter 0's chunk from `kv_hbm_ref`),
  so the corruption is invisible to the output.

  - [ ] Build pre-corrupted `kv_cache_hbm_ref`: zero-init the cache, then
        write garbage (e.g., NaN, or large random values) into the page
        slots that map to KV positions `[prior, prior+512)` for R0. Leave
        positions `[0, prior)` (genuine pre-step content) intact.
  - [ ] Run the 2-iter split. Compare output against the pure-JAX
        reference. With §6.4's fix in place, output `allclose(reference)`.
        Without the fix, output is corrupted because iter 1 reads the
        garbage from `kv_cache_hbm_ref` instead of `kv_hbm_ref`.
  - [ ] Verify the same property at the formula level by reading the
        kernel at `_fetch_bkv` (post-§6.4 edit) and grepping for
        `kv_left_frm_cache`. Confirm it uses `(q_offset + q_len)`, not
        bare `q_len`. File:line citation in the test docstring.

  This test also acts as a regression for Path C as a whole — if a
  future change accidentally re-introduces the `bare q_len` formula, the
  evil test fires immediately rather than waiting for an under-load
  numerical-equivalence flake.

#### Same-pages, no-replication invariant

  - [ ] **`page_indices_ref.shape[0] == max_num_seqs * max_pages_per_seq`**
        (NOT `max_num_subseqs * ...`). If the array is sized by
        `max_num_subseqs`, the runner is replicating page lists —
        the design intent is violated.
  - [ ] **Iters of the same physical seq look up the same page list.**
        Set `q_offset_per_iter = [0, 256]`, `real_seq_idx_per_iter = [0, 0]`,
        and confirm both iters resolve to `page_indices_ref[0..MPL)`.

#### SMEM accounting

  - [ ] `subseq_planner.compute_smem_bytes` matches the formula in
        §10 — phys terms phys-sized, two iter-keyed terms added.
  - [ ] Existing 38+ subseq_planner unit tests pass without change
        (math affects only the SMEM-fit test results, which should
        get *slightly* more generous).
  - [ ] Add at least one test asserting that swapping Path B → Path C
        reduces total SMEM by `align(N_iter, 128) × 4` bytes (the
        savings from phys-keying `cu_q_lens`).

#### Coupled / decoupled equivalence on identical workload

  - [ ] Run the same single-request 8K-prompt workload twice:
    - Once with `K_sched=K_kernel=256` (today's path)
    - Once with `K_sched=8192, K_kernel=256` (decoupled-K)
  - [ ] Both produce **bit-identical output** (`allclose(rtol=0, atol=0)`
        on the generated tokens for greedy sampling, OR `allclose`
        with default tolerances on logits if comparing pre-softmax).
  - [ ] Decoupled run takes 1 prefill step; coupled run takes 32.
  - [ ] TTFT for decoupled run is meaningfully lower than coupled.

#### Eagle3 / spec-decode forwarding

  - [ ] If `real_seq_idx_per_iter` is a new `data_field` on
        `AttentionMetadata`: confirm `eagle3.py`'s metadata rebuild
        forwards it (same treatment as `chunk_prefill_size` — see
        the prior fix at `eagle3.py:376`). If skipped, warmup-traced
        spec-decode kernel will mismatch runtime.

#### Test-coverage targets

  - [ ] All scalar prefetches' shape annotations in kernel.py match
        the new design exactly (PHYS / LOGICAL labels in comments).
  - [ ] No remaining `seq_idx` references in the loop body where
        `iter_idx` or `real_seq_idx` is intended. Easy grep:
        `grep -nE "seq_idx" tpu_inference/kernels/.../kernel.py` —
        every hit is either renamed to `iter_idx` (loop counter)
        or `real_seq_idx` (phys lookup).
  - [ ] Coupled-mode unit tests still all green.
  - [ ] Decoupled-mode integration test (numerical equivalence vs
        coupled) green.
  - [ ] At least one test exercising the MIXED-tail path (R0 with
        n_R0 = K_kernel*N + r where 0 < r < K_kernel).

---

## 12. Out of scope (deferred)

  - Runner-side changes to populate `real_seq_idx_per_iter` and
    accept `max_num_seqs`. Tracked as R1-R7 in the PR3 plan.
  - Tests under `interpret=pltpu.InterpretParams()`. Verified
    upstream JAX hits a `BitcastTransform` gap with this kernel
    (May 2026, JAX 0.10.0). Fallback: pure-JAX reference impl plus
    TPU integration test. See memory entry
    `feedback_pallas_interpret_testing.md`.
  - `kernel_hd64.py` (gpt_oss path). Same change shape.
  - Synthesize / Spotify case. Future PR with new machinery.
  - Sweep recipe / tuner integration.
