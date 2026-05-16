# RPA v3 Kernel — Synthesized-L (SL) Design

Working design doc for a new kernel specialization that **packs many
short prefills into one bundle** so a single MXU launch covers all
of them, instead of issuing N tiny per-seq launches that each waste
the array. Sibling to
`docs/rpa_v3_decoupled_k_kernel_design.md` (decoupled-K LOGICAL =
"one long phys → many K-sized iters"); SL is the inverse mapping
("many short phys → one bundle iter").

This doc enumerates the motivation, the architectural choice
between L-base and M-base, the load-bearing simplification (no
splitting), and the kernel-side delta. Treat as a living checklist
— discuss inline, edit in place.

**Branch / history.** The L-base direction was decided 2026-05-15
(prior M-base plan preserved as Appendix M below). On 2026-05-16
this doc migrated from the `slm` branch (then off `rpa3_3` at
`60810bc9`) to the new `sll` branch built on top of `rpa3_3`'s
L-kernel-tuning final commits (through `fab1b405`). File was
renamed `rpa_v3_slm_kernel_design.md` → `rpa_v3_sll_kernel_design.md`.
SL in main body refers to the new kernel; SLM in Appendix M is
preserved verbatim from the prior plan.

---

## 1. Why we want SL

### 1.1 The MXU-utilization problem for short prefills

Today, short prefills route through MIXED. Each iter is compiled
with static tile sizes (`bq_sz`, `bkv_sz`, `bq_csz`, `bkv_csz`,
all from `production.kernel` for Llama 8B v7x), walks the iter's
Q and KV through them, and **fires the full tile-shaped matmul
regardless of actual q_len**. Padded rows produce garbage scores
zeroed by the causal/kv-bounds mask before softmax
(`kernel.py:815`); padded K cols are masked to `finfo.min` and
`v` is zeroed at `kernel.py:809`. The MXU still runs full
systolic cycles for those wasted positions.

**The nested loop structure** (`kernel.py:1296-1433`, simplified —
five nested loops, four step-sizes):

```python
@pl.loop(0, num_bq)                                          # L0: Q-VMEM,    step = bq_sz
def compute_with_bq(bq_idx):
    @pl.loop(start_bkv_idx, end_bkv_idx)                     # L1: KV-VMEM,   step = bkv_sz
    def compute_with_bkv(bkv_idx):
        num_loops = cdiv(effective_bkv_sz, bkv_csz)          #     (# of bkv_csz chunks in this bkv_sz fetch)
        @pl.loop(0, num_loops)                               # L2: KV-Vreg,   step = bkv_csz (within bkv_sz)
        def attention_loop(idx):
            bkv_start = idx * bkv_csz                        #     current bkv_csz chunk's offset
            for bq_start in range(                           # L3: Q-Vreg,    step = bq_csz (within bq_sz)
                    0, actual_bq_sz, actual_bq_csz):
                for kv_head_idx in range(                    # L4: KV head fan-out (serial, innermost)
                        actual_num_kv_heads):
                    cur_p, cur_v, cur_exp_m_diff = \
                        flash_attention_step1_qk_softmax(...)   # MXU₀ (Q @ K^T)
                    if prev_lm_slice is not None:
                        flash_attention_step2_pv(...)           # MXU₁ (P @ V) for previous kv head
                                                                # (pipelined with current MXU₀)
            flash_attention_step2_pv(...)                    # final MXU₁ outside kv-head loop
```

**Step-size summary** (the four "block sizes" the tuner sweeps):

| Loop | Step | What it tiles |
|---|---|---|
| L0 outer Q  | `bq_sz`  | Q tokens DMA'd into VMEM per Q-pass through one iter |
| L1 outer KV | `bkv_sz` | KV tokens DMA'd into VMEM per KV-pass through one Q-VMEM |
| L2 inner KV | `bkv_csz` (≤ bkv_sz) | KV tokens loaded into VRegs per MXU compute call |
| L3 inner Q  | `bq_csz` (≤ bq_sz)  | Q tokens loaded into VRegs per MXU compute call |

`*_sz` is the VMEM-resident tile (DMA destination from HBM);
`*_csz` is the VReg-resident sub-tile (operand to the MXU). Per
VMEM tile, the kernel does `sz / csz` VReg-load + compute
sub-iterations — that's how DMA-to-VMEM gets pipelined behind
MXU compute (while one VReg's worth of work is on the MXU, the
next VReg-sized slice is loading from VMEM, and the next
VMEM-sized slice is DMA-ing in from HBM). Inside L3 × L4, two
matmuls fire per (bq_start, kv_head_idx): MXU₀ (Q @ K^T inside
`flash_attention_step1_qk_softmax`) and MXU₁ (P @ V inside
`flash_attention_step2_pv`).

`flash_attention_step1_qk_softmax` and `flash_attention_step2_pv`
are pipelined: the second matmul for the *previous* kv head runs
concurrently with the first matmul for the *current* kv head
(`kernel.py:1399-1403`).

**The two MXU matmuls** (labels 0-indexed in this doc; the code
function names remain `step1_qk_softmax` and `step2_pv`):

- **MXU₀** — `s = q @ k.T` at `kernel.py:776`, inside
  `flash_attention_step1_qk_softmax`
- **MXU₁** — `pv = p @ v` at `kernel.py:848`, inside
  `flash_attention_step2_pv`

**Per-kv-head matmul shapes** (one trip through the innermost loop):

```
              M (rows)              K (contract)      N (cols)
MXU₀ Q @ K^T: bq_csz × GQA    @    head_dim     →    bkv_csz
MXU₁ P @ V:   bq_csz × GQA    @    bkv_csz      →    head_dim
              ^^^^^^^^^^^^         ^^^^^^^^          ^^^^^^^^
              same in both         differs:          differs:
                                   head_dim (128)    bkv_csz (256+)
                                   vs bkv_csz        vs head_dim
```

where `GQA = num_q_heads_per_kv_head`. For **Llama 3.1 8B**: 32
Q-heads, 8 KV-heads → GQA = 4, head_dim = 128.

**Important — the MXU's per-launch K-depth is 256, fixed.** Both
MXUs fire the same systolic-array cycles per (M, N) tile: 256
cycles deep, regardless of the matmul's actual K. So:

  - MXU₀ K = `head_dim` ≤ 256: fits in one K-launch per (M, N)
    tile, **K-padded** (half-utilized in the K direction when
    head_dim=128).
  - MXU₁ K = `bkv_csz`: if ≤ 256, fits in one K-launch
    (K-padded if < 256); if > 256, needs `ceil(bkv_csz / 256)`
    K-launches per (M, N) tile.

**Ratio**: MXU₁ fires `ceil(bkv_csz / 256)` × the MACs per (M, N)
tile as MXU₀. At `bkv_csz = 256` → 1× (equal). At `bkv_csz = 512`
→ 2×. At `bkv_csz = 2048` → 8×.

---

**Concrete walkthrough — M kernel processing a single q_len=16,
kv_len=16 short prefill** (production tune `bq_sz = bq_csz = 128,
bkv_sz = 512, bkv_csz = 256` for the M kernel because this workloads go through the **M** kernel today):

Loop counts at this iter:
- `num_bq = cdiv(16, 128) = 1` (one Q-tile pass)
- `num_loops = cdiv(16, 256) = 1` (one KV sub-tile)
- `bq_start` range: `[0]` (one Q sub-tile)
- `kv_head_idx` loop: 8 (one matmul per KV head)
- → **8 MXU₀ + 8 MXU₁ matmul calls per iter**

Each MXU₀ matmul: `(128 × 4 = 512) @ (128, 256) → (512, 256)`.
- (M, N) tiles: `ceil(512/256) × ceil(256/256) = 2 × 1 = 2 per kv head`
- MACs fired per tile: `256 × 256 × 256 = 16,777,216`
  (MXU K-depth = 256; head_dim=128 is K-padded to 256)
- MACs fired per kv-head MXU₀: `2 × 16,777,216 = 33,554,432`

Each MXU₁ matmul: `(512, 256) @ (256, 128) → (512, 128)`.
- (M, N) tiles: `ceil(512/256) × ceil(128/256) = 2 × 1 = 2 per kv head`
- MACs fired per tile: `256 × 256 × 256 = 16,777,216` (K = bkv_csz = 256 exactly)
- MACs fired per kv-head MXU₁: `2 × 16,777,216 = 33,554,432`

Per-iter totals (× 8 kv heads):
- **MXU₀ fired**: `8 × 33,554,432 = 268,435,456`
- **MXU₁ fired**: `8 × 33,554,432 = 268,435,456`
- **Total fired**: `536,870,912` MACs

(MXU₀ and MXU₁ fire equal MACs at `bkv_csz = 256` because the
systolic K-depth is 256 either way — MXU₀'s K=128 gets padded.)

Useful work per iter:
- Useful Q rows: `q_len × GQA = 16 × 4 = 64` (of 512 padded)
- Useful K cols / depth: `kv_len = 16` (of 256 padded)
- MXU₀ useful MACs per kv head: `(useful_q x useful_k) × head_dim
  = (64 × 16) × 128 = 131,072`
- MXU₁ useful MACs per kv head: `(useful_q x head_dim) × useful_K
  = (64 × 128) × 16 = 131,072`
- Total useful per iter: `2 × 131,072 × 8 = 2,097,152` MACs

**Per-iter MXU efficiency: `2,097,152 / 536,870,912 ≈ 0.39%`**.

For **16 short prefills** (each its own iter under MIXED):
`16 × 536.9 M = 8.59 B MACs fired` to do `16 × 2.10 M = 33.5 M
useful MACs`.

---

**Same workload as one SL bundle of 16 × 16 = 256 tokens** (assumed tune
`bq_sz = bq_csz = 256, bkv_csz = 256`, matching LOGICAL chunk=256
sweet spot from §6 table. This should be routed thorugh the L kernel so these tune numbers are based on the **L** kernel):

Loop counts:
- `num_bq = 1` (256 tokens fit `bq_sz=256` exactly)
- `num_loops = 1`, `bq_start = [0]`, `kv_head_idx = 8`
- → **8 MXU₀ + 8 MXU₁ matmul calls per iter** (same call count as
  M kernel, but each matmul is bigger)

Each MXU₀ matmul: `(256 × 4 = 1024) @ (128, 256) → (1024, 256)`.
- (M, N) tiles: `4 × 1 = 4 per kv head`
- MACs fired per tile: `256 × 256 × 256 = 16,777,216`
  (K=128 K-padded to 256)
- MACs fired per kv-head MXU₀: `4 × 16,777,216 = 67,108,864`

Each MXU₁ matmul: `(1024, 256) @ (256, 128) → (1024, 128)`.
- (M, N) tiles: `4 × 1 = 4 per kv head`
- MACs fired per tile: `256 × 256 × 256 = 16,777,216`
- MACs fired per kv-head MXU₁: `4 × 16,777,216 = 67,108,864`

Per-bundle-iter totals (× 8 kv heads):
- **MXU₀ fired**: `8 × 67,108,864 = 536,870,912`
- **MXU₁ fired**: `8 × 67,108,864 = 536,870,912`
- **Total fired**: `1,073,741,824` MACs (~1.07 B)

Useful work per bundle iter:
- 16 segments, each contributing block-diagonal cells only:
  - MXU₀ per segment per kv head: `(16 × 4) × 16 × 128 = 131,072`
  - MXU₁ per segment per kv head: `(16 × 4) × 128 × 16 = 131,072`
- Per kv head: `16 × 2 × 131,072 = 4,194,304`
- Per iter (× 8 kv heads): `33,554,432` MACs

**Per-iter MXU efficiency: `33,554,432 / 1,073,741,824 ≈ 3.13%`**.

---

**Comparison table for q_len = kv_len = 16, 16 short prefills:**

| | M kernel (per iter) | M kernel (16 iters) | SL bundle (1 iter, 256 tokens) |
|---|---|---|---|
| MXU₀ fired | 268.4 M | 4.29 B | 536.9 M |
| MXU₁ fired | 268.4 M | 4.29 B | 536.9 M |
| **Total fired** | 536.9 M | **8.59 B** | **1.07 B** |
| Total useful | 2.10 M | 33.5 M | 33.5 M |
| MXU efficiency | 0.39% | 0.39% | **3.13%** |
| **SL speedup over M** | — | — | **~8×** |

The ~8× factor comes from `bq_sz / q_len = 128 / 16 = 8`. MXU₀
and MXU₁ fire equal MACs at `bkv_csz = 256` (both K-deep 256 on
the systolic array), so the speedup factor is consistent across
both matmuls.

---

**Speedup curve vs q_len** (M tuned at `bq_sz=128`; SL bundles
pack to ≤256 tokens):

```
SL speedup ≈ bq_sz / q_len     (for q_len < bq_sz)
```

| q_len | M-iter useful / fired | M-iter efficiency | SL bundle pack | SL speedup |
|---|---|---|---|---|
| 8   | 0.52 M / 536.9 M | **0.097%** | 32 × 8  = 256 | **~16×** |
| 16  | 2.10 M / 536.9 M | **0.39%**  | 16 × 16 = 256 | **~8×** |
| 32  | 8.39 M / 536.9 M | **1.56%**  | 8 × 32  = 256 | **~4×** |
| 64  | 33.55 M / 536.9 M | **6.25%** | 4 × 64  = 256 | **~2×** |
| 96  | 75.50 M / 536.9 M | **14.06%** | 2 × 96 = 192 (wastes 64 slot) | **~1× (formula breaks; see note)** |
| 128 | 134.2 M / 536.9 M | **25.0%**  | 2 × 128 = 256 | **1× (no MXU win)** |

**Note on the q_len=96 entry**: the `bq_sz / q_len` formula
predicts 128/96 ≈ 1.33×, but **the actual speedup is closer to
1×** because q_len=96 doesn't divide 256 cleanly — the bundle
packs 2 segments and pads 64 slots, giving the SL kernel the same
2 iters' worth of work that M kernel processes. SL pays the
bundle's full-tile MXU cost regardless of the wasted slots.
General formula:

```
SL speedup = floor(SL_BUNDLE_CAP / q_len) × (bq_sz_M / bq_csz_SL)
           = floor(256 / q_len) × (128 / 256)
           = floor(256 / q_len) / 2
```

Equals `bq_sz / q_len` only when `q_len` divides `SL_BUNDLE_CAP`
exactly (powers of 2 ≤ 128 in our case).

So SL's MXU-fill advantage has teeth only when **per-prefill
q_len is materially smaller than M's tuned `bq_sz`** (which is 128):

  - Strong MXU win at q_len ≤ 16
  - Moderate at q_len ~16–48
  - Marginal at q_len ~48–96
  - Zero at q_len ≥ 96 (with current M tune)

LOGICAL handles workloads where q_len > 256 via chunked prefill;
SL targets the band where `q_len ≪ bq_sz` and short prefills
otherwise route to MIXED.

**Two important caveats:**

1. **The SL win depends on M's tuned `bq_sz`.** If M is re-tuned
   with smaller `bq_sz` (e.g., 32 or 16) for short-prefill
   workloads, the speedup curve shifts and SL's advantage shrinks
   proportionally. Open TODO captured under §7.3.

2. **SL achieves ~4% MXU efficiency, not ~100%.** Even fully
   packed, the block-diagonal mask wastes cross-segment Q×K
   contributions. SL's win is "less waste" not "no waste".
   Per-segment K-tile matmul splits (fire only diagonal block
   launches) could push higher; captured as v2 TODO under §7.3.

### 1.2 The two secondary wins

Even if MXU utilization were not the dominant story, SL exploits
two other inefficiencies of "1 short seq = 1 iter":

  - **Per-iter overhead amortization.** Every iter pays prefetch
    advance, mask construction, sync, accumulator init/spill, and
    iter-prologue arithmetic. That cost is independent of `q_len`.
    For 1000 short prefills under MIXED, we pay it 1000 times.
    Under SL, we pay it once per bundle (≈ once per 256 tokens of
    packed Q).

  - **bq-tile work-per-byte.** MIXED's `bq` is one tuned constant
    for the launch. For seqs with `q_len ≪ bq`, the kernel still
    allocates and *touches* a `bq × HD` Q-tile but only `q_len × HD`
    of it carries useful tokens — the rest is masked. SL keeps the
    tile fully populated by drawing tokens from multiple seqs.

### 1.3 What SL is *not*

  - **Not a re-skin of MIXED with a different name.** The mask
    formula changes (causal → block-diagonal causal) and a new
    per-iter prefetch is added (bundle→seq-range map). Everything
    else is shared with MIXED.

  - **Not a replacement for LOGICAL.** LOGICAL chunks *one* long
    phys into many K-sized iters; SL packs many short phys into
    *one* iter. They solve opposite problems and can coexist in
    the same step.

  - **Not free packing.** The bundle-construction logic lives in
    the runner (data-prep). SL assumes bundles arrive pre-packed
    in the Q tensor and described by the prefetch tables.

---

## L-base v1 plan (work in progress, 2026-05-15)

Rebased from M-base to L-base on 2026-05-15. The L-base plan is
being built up step by step, reading L's code together rather
than translating from the M-base plan. The prior M-base plan is
preserved verbatim below as Appendix M; if we revert for any
reason, restore from there.

### 7.1 Why P/L beats M: the `static_q_len` specialization

Reading `process()` at `kernel.py:1257-1265`:

```python
def process(static_q_len=None):
    if static_q_len is None:           # MIXED / DECODE
        actual_bq_sz = bq_sz
        num_bq = cdiv(q_len, actual_bq_sz)
    else:                              # PREFILL / LOGICAL
        actual_bq_sz = min(bq_sz, static_q_len)
        num_bq = cdiv(static_q_len, actual_bq_sz)
    actual_bq_csz = min(bq_csz, actual_bq_sz)
```

The two branches do the same thing — *walk this iter's Q-tokens
through a VMEM tile of size `actual_bq_sz`*. The difference is
whether the iter's Q-length is known statically (PREFILL/LOGICAL)
or only at runtime (MIXED/DECODE).

**MIXED branch (`static_q_len is None`)**: the kernel doesn't
know the iter's q_len at compile time. It allocates the **full
tuned tile size `bq_sz`** in VMEM (sized for the worst case) and
computes `num_bq = cdiv(q_len, bq_sz)` at runtime. Most of the
allocated tile is wasted when q_len < bq_sz.

**PREFILL/LOGICAL branch (`static_q_len` set)**: the kernel knows
the iter has *exactly* `static_q_len` Q-tokens (= `K_kernel` for
LOGICAL, = `chunk_prefill_size` for PREFILL). Two compile-time
optimizations land:

  1. **VMEM allocation can be shrunk** to `min(bq_sz,
     static_q_len)` — if the iter's data fits in a smaller tile,
     no need to allocate the worst-case tile.
  2. **`num_bq` resolves at compile time**, no runtime cdiv.

Concrete examples:

| `static_q_len` | tuned `bq_sz` | `actual_bq_sz` | `num_bq` | Notes |
|---|---|---|---|---|
| 256 (LOGICAL K_kernel) | 256 | 256 | 1 | One full-tile pass per iter. Sweet spot. |
| 256 | 128 | 128 | 2 | Smaller tile, two passes compile-known. |
| 128 (PREFILL chunk=128) | 128 | 128 | 1 | Tile matches; one pass. |
| 64 (PREFILL chunk=64) | 128 | **64** (shrunk!) | 1 | Tile **smaller than tuned cap** — VMEM saved. |

**This is why P/L wins over M on prefill workloads.** Same MXU
math, but P/L's compile-time-static iter-shape lets the compiler
emit tighter VMEM allocations, statically-unrolled loops, and
no-padding MXU launches.

### 7.2 `SL_BUNDLE_CAP` as a per-variant compile parameter

SL plays the same static-K game as PREFILL/LOGICAL:

  - `SL_BUNDLE_CAP` *is* the SL kernel's `static_q_len` —
    each bundle iter has exactly `SL_BUNDLE_CAP` Q-tokens (padded
    if the packer couldn't fill it).
  - Different bundle caps compile to different SL kernel variants
    (just as `chunk_prefill_size ∈ {128, 256, 512, ..., 8192}`
    today compile multiple PREFILL variants per
    `production.kernel`).
  - Smaller bundle caps shrink VMEM and emit denser-utilization
    MXU launches (per §7.1's `actual_bq_sz` shrink mechanism), at
    the cost of less per-iter overhead amortization.

**v1 starting point**: `SL_BUNDLE_CAP = 256` (matches MXU width
and the LOGICAL chunk=256 sweet spot, see §6 production-tune
table). Other caps (e.g., `SL_BUNDLE_CAP = 128`) become future
variants if the q_len distribution justifies them.

The bundle cap is not the same kind of tunable as `bq_sz` /
`bkv_sz`. The cap defines the kernel's static iter shape (which
variant we compile); `bq_sz` / `bkv_sz` are *within-variant*
VMEM-tile tunables that the tuner sweeps.

### 7.3 Open TODOs and sensitivity questions

  - **TODO: re-tune M kernel with smaller `bq_sz` for short-prefill
    workloads.** Today's production M kernel is tuned at `bq_sz =
    128`, which is what makes the SL win look ~8.5× at q_len=15
    (§1.1). A "short-prefill-tuned M" with `bq_sz = 32` would cut
    M's per-iter fired-row-slots from 512 to 128, raising M's
    matmul efficiency at q_len=15 from 11.7% to ~47% — closing
    most of the per-launch MXU-fill gap. SL would still win on
    per-iter overhead amortization, but the MXU-fill win shrinks
    substantially. **Run this experiment before committing to the
    SL implementation effort.**

  - **TODO: tuner architecture — "conditional best" sweeps.** The
    kernel tuner today pins workload-key vars (max_model_len,
    head config, dtype, etc.) and sweeps **all unpinned tunables
    freely** to find the *unconditional best* config. Useful
    extension: optionally pin a subset of the currently-unpinned
    vars (e.g., force `bq_sz = 32`) and sweep the rest to find
    the *conditional best*. Different artifact, different
    question:
      - Unconditional best: *"What's the kernel's peak perf?"*
      - Conditional best: *"What's the kernel's peak perf
        constrained to bq_sz = X?"*
    Conditional best lets us answer "should I compile a
    short-prefill variant with bq_sz = 32?" without exhaustively
    re-tuning. Also: lets us run sensitivity analysis on any
    tuned var — how much perf do we lose by pinning X to suboptimal
    Y? Engineering scope: small extension to v2 tuner's
    search-space declaration.

  - **(Closed) Padding short prefills with `-inf` is not viable.**
    Investigated 2026-05-16 as a possible MXU-skip mechanism.
    Three failure modes:
    1. Padding Q (or K) values with `-inf` → `Q @ K = -inf × any`
       → `-inf`, `+inf`, or `NaN` depending on K sign. Even
       single-sign K gives row-max = `-inf` → `exp(real − (-inf))
       = exp(+inf) = inf` for the *real* Q rows → softmax NaN.
       Math is wrong.
    2. Padding the score `s` with `-inf` post-matmul ≡ what the
       existing `finfo.min` mask does (`kernel.py:815`); not a new
       win.
    3. *Premise was wrong*: the TPU MXU is a systolic array that
       runs **full 256×256×K cycles regardless of operand
       values**. There's no sparsity-skip path. So even if -inf
       in Q were mathematically valid, it wouldn't reduce MXU
       work. Closing this question so we don't relitigate.

  - **(Closed) Per-segment K-tile matmul splits do not help.**
    Investigated 2026-05-16 as a follow-on to the §1.1 ~4% MXU
    efficiency finding. *Idea was*: replace the single
    `(bq_csz × GQA, bkv_csz)` bundled matmul with N per-segment
    matmuls of shape `(seg_q × GQA, seg_kv)` each — fire only the
    diagonal block launches, "skip" the cross-segment waste.
    Three failure modes:
    1. **MXU fires full 256×256×K cycles per launch regardless
       of tile size.** Issuing a smaller matmul doesn't issue a
       smaller MXU launch — it issues a launch where most of the
       systolic array's output cells are unused. For a segment of
       16 tokens with GQA=4: per-segment matmul rows = 64; MXU
       fires 256 rows worth of cycles, of which only 60 are real.
       Per-launch useful-work density = 60/256 ≈ 23%, vs the
       bundled case's ~6.25%. But:
    2. **N small launches replace 4 big launches.** For 16 segments
       packed into a 256-token bundle, the bundled case fires
       4 MXU launches (bq_csz=256 → ceil(1024/256) = 4). The
       per-segment alternative fires 16 launches (one per segment).
       Net: same useful MAC count, more MXU launches with smaller
       per-launch density. **Strictly worse** by a factor of N/4.
    3. **LLO / Mosaic already handle the right abstraction at the
       SA level** — `bq_csz` and `bkv_csz` ARE the lowering
       interface to the systolic array; the compiler tiles them
       across the available MXUs and pipelines K-depth. Reaching
       below this level from the kernel author's side is
       counter-productive because the SA-level decisions LLO
       makes are already optimal for the (bq_csz, bkv_csz) shape
       we give it. Issuing smaller matmuls is *under-using* the
       LLO/SA contract, not bypassing it.

    **Conclusion**: bundled-matmul + post-softmax-mask is the
    right shape at the kernel-author level. The block-diagonal
    waste (~96% of MXU cells at 16-token segments) is a property
    of the masking pattern, not a kernel-implementation choice.
    Reducing it would require **hardware sparsity support**, which
    TPU MXU doesn't have. Closing this question so we don't
    relitigate.

  - **(Carried forward)** Code-research items for L-base
    implementation (iter→segment mapping shape, mask term under
    L-base, write-path per-segment loop, dispatch gate): to be
    worked through together by reading L's code in subsequent
    sessions.

---


# Appendix M — Original M-base plan (preserved 2026-05-15)

The content below is the M-base (SLM) plan as of commit
`6478cdcf`. §1 (motivation) was moved to the main body above and
its §1.1 MXU-fill math was corrected to be GQA-aware (the
original framing significantly overstated SL's speedup). Sections
§2 through §7 below are the original pre-rebase content,
unchanged.


## 2. Unified framing — every iter processes one *logical* seq

All three RPA-v3 kernel cases follow the same per-iter rhythm:
**load one logical seq's Q + KV → compute attention → write
back**. What differs across M, L, and SL is only the
**logical↔phys multiplicity**, i.e. how the runner maps physical
requests into logical iter units before invoking the kernel.

| | What loads (per iter) | Load DMAs | Compute | Write DMAs | Logical : Phys |
|---|---|---|---|---|---|
| **M** | 1 phys's Q + 1 phys's KV (cache for prior steps + new-KV for this step) | 1 Q DMA + pages-loop over history + 1 new-KV DMA | per-seq causal attn | 1 (write phys's new-KV slice to its pages) | **1 : 1** |
| **L** | 1 K_kernel chunk of 1 phys's Q + **cumulative KV `[0, (i+1)·K_kernel)`** of the phys (cache for prior-step history + `kv_hbm_ref` for cumulative this-step new-content) | 1 Q DMA + pages-loop over history + 1 contiguous new-KV DMA (sized to cumulative in-step content) | chunk-causal attn against growing K | 1 (write *only this chunk's* new-KV to phys's pages — earlier iters wrote their own slices) | **N : 1** (N logicals share 1 phys) |
| **SL** | N phys's full Q + N phys's full new-KV (no prior cache by assumption) | **1 contiguous Q DMA + 1 contiguous new-KV DMA**, no pages-loop | bundle attn with block-diagonal causal mask | **N** (one per segment in the bundle, each writing that short seq's new-KV to its own pages) | **1 : N** (1 logical contains N phys) |

The cumulative-KV row for L is sourced from `kernel.py:887-892`'s
own commentary: *"in-step content (this phys seq's tokens this
step, **cumulative across iters**) is sourced from `kv_hbm_ref`,
NOT `kv_cache_hbm_ref`. Eliminates same-pallas_call cross-iter
read-after-write race"*. The cumulative-history caveat applies
across steps too: only the very first iter of the very first
step of a phys's prefill loads zero KV; every later iter (whether
later chunk within the step, or first chunk of a later step)
has some non-empty cache + new-read mix.

Two corollaries flow from this unification:

  - **Read path is symmetric across M/L/SL.** Each iter loads one
    logical seq's contiguous Q and (with the pages-loop) its KV.
    SL is the *simplest* read of the three because short prefills
    have no prior cache — both Q and KV come from contiguous
    flat-ragged HBM buffers (`q_hbm_ref` and `kv_hbm_ref`), no
    page-gather. **No runner-side layout change is needed for
    SL's reads.**
  - **Write path is where the three diverge.** M writes one full
    slice to one phys's pages. L writes one *narrow* slice (its
    own chunk only) to one phys's pages. **SL writes N slices,
    one per segment, each to that segment's own phys's pages** —
    the only case where the write fans out per iter. The kernel
    body's `_write` extent calculation (`kernel.py:899-908`)
    already exists to handle L's narrow-write semantics; SL adds
    a per-segment loop on top.

### 2.1 v1 starting point: build SL on MIXED (not a final decision)

The unification above shows SL and L are *opposite* specializations
of the logical-seq abstraction, not nested ones. Concretely:

  - LOGICAL's `_IterQuantities` carries scalar `phys_seq_idx` per
    iter (kernel.py §6.2 / `:1051`). Extending LOGICAL to support
    SL means widening that scalar to a per-segment list AND
    keeping all of LOGICAL's existing q_offset / cumulative-KV /
    narrow-write machinery that SL doesn't need at v1. Strictly
    additive complexity for the no-split v1 case.
  - MIXED's mental model — *"one iter = one piece of work, sized
    however you like"* — already matches SL's logical-seq pattern.
    SL just redefines "piece of work" from `one seq's q_len
    tokens` to `K_kernel concatenated tokens from many seqs`. The
    new machinery is one per-iter bundle-bounds prefetch + a
    per-segment loop in `_update_kv_cache`. Nothing else.

**v1 starting point: build SL on MIXED.** This is a *complexity*
choice, not a final *performance* choice. Two reasons we may need
to revisit and rebuild SL on LOGICAL later:

  1. **L is materially faster than M today on prefill-heavy
     workloads.** `docs/k_serving_repro.md` shows the LOGICAL-vs-
     MIXED-baseline gap at +47% throughput / −32% P99 TTFT (rows
     1 → 4 in that doc; tuned LOGICAL with K=256 vs default
     MIXED). If SL inherits MIXED's per-row arithmetic-intensity
     curve rather than LOGICAL's denser static-K curve, SL on
     M-base may still leave significant performance on the table
     vs. an L-base alternative.
  2. **The no-split v1 policy throws away tail-packing
     efficiency** (§3). If measured tail-waste is large on real
     short-prefill arrival patterns, the split-policy v2 needs
     LOGICAL's chunked-KV / q_offset machinery — at which point
     SL is essentially a LOGICAL extension that also handles
     many-phys-per-iter, not a MIXED extension at all.

So treat M-base as "the cheapest path to working code that proves
out the MXU-utilization win", not as the final architecture.
Re-evaluate against both performance and tail-waste data after v1
lands.

Rename for kernel-internal clarity (v1): `RpaCase.MIXED` stays
as-is for the dynamic-q_len case; add `RpaCase.SLM` for the
packed-bundle case. They share the kernel body and most of
MIXED's mask path; SLM adds one mask term (§4) and one write-path
loop. If v2 moves SL to L-base, the case name carries forward
and the dispatch gate becomes presence-based on the bundle
prefetches (analogous to how LOGICAL is dispatch-gated on
`phys_seq_indices` presence today, `kernel.py:2244-2249`).

---

## 3. No-split policy — the load-bearing simplification

A packer faced with three short prefills `[10, 20, 30]` and a
bundle capacity of 32 tokens has a choice:

  - **No-split (v1, this doc)**: bundle 0 = `[10, 20]` (30 tokens
    used, 2 slack); the 30-token seq starts bundle 1. **Drop the
    tail-end slack.**
  - **Split (v2, not in this doc)**: bundle 0 = `[10, 20, 2-of-30]`
    (32 tokens, full); bundle 1 starts with the remaining 28
    tokens of the 30-token seq.

The split policy buys ~5–15% packing efficiency on bursty
short-prefill workloads (back-of-envelope; needs measurement).
The cost it pays:

  - Cross-iter softmax carry: bundle 1's iter must combine its
    partial `(m, l, o)` for the split seq with bundle 0's partial
    stats. Same machinery LOGICAL implements for chunked prefill.
  - KV-cache read-after-write: bundle 1's later tokens of the
    split seq must causally attend to bundle 0's KV writes. Either
    the cache is flushed-and-visible across iters (synchronization
    barrier), or partial outputs are accumulated in a carry
    register (extra HBM traffic per bundle boundary).
  - Per-Q-token `q_offset`: the split seq's intra-seq position
    starts at 2 in bundle 1, not 0. Mask construction needs to
    know this.

**This is exactly LOGICAL's existing machinery.** Adopting split
in SLM = re-implementing LOGICAL with a different name.

The no-split policy throws away the packing efficiency to keep
SLM's machinery flat — each bundle is a self-contained piece of
work with no inter-bundle dependencies. The tail-waste cost is
empirical and acceptable as a v1 trade-off; revisit splitting as
a separate v2 effort if measurements justify it (and at that
point, build it on LOGICAL, not on SLM).

---

## 4. Mask design — block-diagonal causal, derivable from `cu_q_lens`

For a bundle covering short seqs at indices `[s_start, s_end)`,
the score matrix is `bq × bkv` where bq spans the bundle's Q
tokens and bkv spans the bundle's K tokens. The wanted regions
are the diagonal blocks; cross-seq blocks must be masked to `-inf`
before softmax.

```
Bundle = [seq A (10 tok), seq B (20 tok), seq C (30 tok)]  →  60×60 score

         A(10)    B(20)    C(30)
    A  [ causal ] [  -∞  ] [  -∞  ]
    B  [  -∞   ] [causal] [  -∞  ]
    C  [  -∞   ] [  -∞  ] [causal]
```

The boundary information `[0, 10, 30, 60]` is *exactly* what
`cu_q_lens` stores today in MIXED, one entry per short seq.
Nothing new is needed for segment lengths; the kernel reads them
straight out of the existing SMEM prefetch.

**The minimal delta from MIXED's mask construction is one new
boolean term.** Today's MIXED mask combines causal + KV-bounds (+
optional sliding-window). SLM keeps all three terms unchanged and
adds a segment-equality term:

```
MIXED today:  mask = causal AND kv_bounds
SLM:          mask = causal AND kv_bounds AND (seg_id_Q[bq] == seg_id_K[bkv])
```

Why so little changes:

  - `processed_q_len` and `processed_kv_len` (the scalar bq/bkv
    tile offsets the kernel already maintains) stay scalar. They
    refer to the bundle iter's tile-start in *bundle-local
    coordinates*, which is mechanically identical to today's
    *iter-local coordinates* (one iter = one bundle, one iter =
    one seq — same shape).
  - `q_span[bq, bkv]` and `k_span[bq, bkv]` (kernel.py:795,797),
    derived from those scalars via `iota`, give per-row / per-col
    absolute positions in bundle coords. Today they give absolute
    positions in seq coords. Same derivation, just renamed
    mentally.
  - Causal `q_span ≥ k_span` (kernel.py:805) **stays bit-for-bit
    the same**. For short-prefill SLM, each segment's K layout
    mirrors its Q layout (same length, same bundle-local
    placement), so within a segment, `q_span − k_span` is exactly
    the intra-segment causal delta. Across segments, causal might
    spuriously *allow* a (later-segment-Q, earlier-segment-K)
    pair, but the new `seg_id` term zeroes those out.
  - KV-bounds `k_span < effective_kv_len` (kernel.py:808) also
    stays the same shape — we just feed it per-bundle bounds
    instead of per-seq bounds (KV bounds for the full bundle of
    short prefills, which is the sum of segment KV lens).

The **only** new computation in mask construction is
`seg_id_Q[bq]` and `seg_id_K[bkv]`. Both are derived from
`cu_q_lens[s_start:s_end+1]` (already in SMEM) by a bucket lookup:
"which `cu_q_lens` interval does this position fall in?". For
≤32 segments per bundle, that's a short, unrolled chain of
compares per row/col, executed once per iter (not per inner
bq×bkv tile) and stored in a small precompute array.

No per-Q-token req-id tag prefetch. No new seg-lens prefetch. No
change to causal or KV-bounds construction. The pattern is the
standard "document mask" / "varlen packing" pattern from
FlashAttention's varlen path and NeMo seq-packing.

---

### 4.1 Numerical safety — `finfo.min`, not `-inf`

The kernel masks positions to a *very-negative-finite* sentinel,
not to `-inf`. Specifically (`kernel.py:195-197` and `:358-359`):

```python
mask_value = jnp.finfo(out_dtype).min
```

`jnp.finfo(dtype).min` returns the **most-negative representable
finite** value: ≈ −3.39 × 10³⁸ for bf16, ≈ −3.40 × 10³⁸ for f32.
The application (`kernel.py:815`) is a `jnp.where(mask, s,
mask_value)` — a pure select, never an addition.

This is deliberate: it avoids any chance of `(+∞) + (−∞) = NaN`
arithmetic later in the chain. Downstream behavior at masked
positions:

  - `max(s, axis=1)` (kernel.py:817): max-over-finite, never picks
    `finfo.min` over a real score, IEEE-defined, no NaN.
  - `exp(s − m)` (kernel.py:824): at masked positions,
    `s − m ≈ −3.4e38 − finite ≈ −3.4e38`, and `exp(−3.4e38)`
    underflows to **exact** 0 in bf16/f32 (the exponent is far
    below the underflow threshold, ≈ −88 for f32 / −10 for bf16).
  - `sum(p, axis=1)` (kernel.py:826): exact zeros contribute zero.
  - `p @ V` (kernel.py:848): zero rows of p produce zero V
    contributions.

**Implication for SLM**: we inherit this safety for free. Adding
the new `seg_id_Q == seg_id_K` term to the mask boolean — applied
through the same `jnp.where(mask, s, mask_value)` path — produces
the same exact-zero contributions at cross-segment positions. No
new numerical concerns, no `-inf` arithmetic anywhere, no live TPU
verification needed.

Reference: JAX docs for `jnp.finfo`,
<https://docs.jax.dev/en/latest/_autosummary/jax.numpy.finfo.html>.
Pattern matches the FlashAttention masking idiom (also picks
`finfo.min`, not `-inf`, for the same reason).

---

### 4.2 Annotated compute path — where segmentation enters

Walking the M-kernel inner-loop body (`kernel.py:745-855`,
`flash_attention_step1_qk_softmax` + `flash_attention_step2_pv`)
line by line to confirm the claim above: only the mask
*construction* is segment-aware; matmuls and softmax are
shape-agnostic.

Notation: `bq = actual_bq_csz * num_q_heads_per_kv_head`,
`bkv = bkv_csz`, `H = head_dim`.

| Line | Op | Shapes | Unit | Segment-aware? |
|---|---|---|---|---|
| 776 | `s = q @ k.T` | `[bq, H] @ [H, bkv] → [bq, bkv]` | **MXU₁** | No |
| 784 | `s *= s_scale` | `[bq, bkv]` | VALU elem | No |
| 787 | `s = soft_cap·tanh(s/soft_cap)` (opt) | `[bq, bkv]` | VALU elem | No |
| 795 | `q_span = processed_q_len + iota(axis=0)//gqa` | scalar + `[bq, bkv]` | VALU elem | No (scalar stays scalar in SLM) |
| 797 | `k_span = processed_kv_len + iota(axis=1)` | scalar + `[bq, bkv]` | VALU elem | No (scalar stays scalar in SLM) |
| 805 | `mask &= (q_span ≥ k_span)` causal | `[bq, bkv]` | VALU elem | No (unchanged) |
| 808 | `mask &= (k_span < effective_kv_len)` kv-bounds | `[bq, bkv]` | VALU elem | No (unchanged) |
| — | **SLM:** `mask &= (seg_id_Q[bq] == seg_id_K[bkv])` | `[bq, bkv]` | VALU elem | **YES — the one new term** |
| 815 | `s = where(mask, s, finfo.min)` | `[bq, bkv]` | VALU elem | No (just applies mask) |
| 817 | `s_rowmax = max(s, axis=1)` | `[bq, bkv] → [bq, 1]` | **VALU reduction** | Implicit (mask did its job) |
| 821-823 | `m_curr = max(m_prev, s_rowmax); m_ref = m_curr` | `[bq, 1]` | VALU elem | No |
| 824 | `p = exp(s − m_curr)` | `[bq, bkv]` | VALU elem | No (masked → 0) |
| 826 | `p_rowsum = sum(p, axis=1)` | `[bq, bkv] → [bq, 1]` | **VALU reduction** | Implicit |
| 827-829 | `l_ref = exp(m_prev − m_curr)·l_prev + p_rowsum` | `[bq, 1]` | VALU elem | No |
| 848 | `pv = p @ v` | `[bq, bkv] @ [bkv, H] → [bq, H]` | **MXU₂** | No (zero rows → zero contributions) |
| 855 | `o_ref = exp_m_diff·o_prev + pv` | `[bq, H]` | VALU elem | No |

**Conclusion.** Exactly one row in this table changes for SLM: a
new mask term inserted between line 808 and line 815. Everything
else — both MXUs, both reductions, all element-wise ops, the
running `(m, l, o)` flash-style accumulators — operates on the
full `[bq, bkv]` bundle tile and is segment-correct automatically
because cross-segment positions hold zeros after softmax.

**No splitting of the score matrix into N sub-matrices is
required**, neither before VALU nor before MXU₂. The hardware
processes the bundle as one tensor end-to-end.

---

## 5. Delta from MIXED today

| Component | MIXED today | SLM |
|---|---|---|
| `cu_q_lens` | per seq, one entry per iter | **per short seq, unchanged semantics** |
| `kv_lens`, `page_indices`, `distribution` | as today | as today |
| New prefetch | — | **one small array**: per-iter `(s_start, s_end)` bundle→short-seq range |
| Outer iter mapping | iter k = seq k | iter k = bundle k (covers seqs `[s_start_k, s_end_k)`) |
| Q tile per iter | `cu_q_lens[k+1] − cu_q_lens[k]` tokens | `cu_q_lens[s_end_k] − cu_q_lens[s_start_k]` tokens |
| Mask formula | causal over `cu_q_lens[k:k+2]` | block-diagonal causal over `cu_q_lens[s_start_k : s_end_k+1]` |
| `q_offset` arithmetic | not needed | **not needed** |
| Cross-iter softmax carry | not needed | **not needed** |
| KV-cache write race | none | **none** (disjoint pages per bundle) |
| `static_q_len` specialization | none (MIXED is dynamic) | none — bundle's total q_len is dynamic, but ≤ K_kernel by construction |

**The only genuinely new thing is the per-iter bundle-bounds
prefetch.** Concretely a `bundle_seq_range_ref` of shape
`[max_num_bundles, 2]` (or a single cumulative `cu_bundle_seqs`
of length `max_num_bundles + 1`, slightly tighter). Few KB total.

Everything else is MIXED today with the mask formula widened and
the iter-unit redefined from "seq" to "bundle".

---

## 6. Open questions to enrich before coding

These are the gaps to close in this doc before we write kernel
code. Each needs code-snippet research and a concrete answer.

  - ~~**Mask construction in Pallas.** Where does MIXED build its
    causal mask today? Confirm `bucket()` is expressible in Pallas
    semantics without blowing up VMEM.~~ **RESOLVED — see §4 and
    §4.2.** Mask is built at `kernel.py:795-815`. Causal and
    KV-bounds terms stay bit-for-bit identical; only one new
    boolean term (`seg_id_Q == seg_id_K`) is inserted before line
    815. Segment IDs are bucket-lookup over `cu_q_lens` (already
    SMEM-resident), precomputed once per iter into a small
    per-token array. No new tile-shaped allocations. The
    `finfo.min` mask sentinel (kernel.py:195) carries through
    unchanged — see §4.1 for why this is numerically safe without
    `-inf` arithmetic.
  - ~~**`bundle_seq_range_ref` placement.**~~ **RESOLVED.** Same
    class as `phys_seq_indices_ref` today: SMEM-resident
    per-iter prefetch (one-shot copy from HBM before the iter
    loop, accessed scalar-style inside the loop). Shape
    `[max_num_bundles, 2]` int32 (or cumulative-length flat array
    `[max_num_bundles + 1]` if we prefer a single fence-post
    layout). Sized to `max_num_bundles` — the kernel's
    static-shape budget; runner pads unused entries.
  - ~~**Q/K/V tensor layout.**~~ **RESOLVED.** Q is laid out in
    HBM as `q_hbm_ref` shape `(num_kv_heads, max_num_tokens, ...)`
    — *flat-ragged*, contiguous along the token dimension with
    `cu_q_lens` defining boundaries (kernel.py:659-665). New-KV
    (`kv_hbm_ref`) uses the same flat-ragged token coordinate
    system (kernel.py:962 derives `new_kv_len_start` from
    `q_end`). A bundle's Q is one contiguous slice
    `[cu_q_lens[s_start], cu_q_lens[s_end])`; same for its
    new-KV. `_fetch_bq` (`:1088-1093`) and the new-read branch of
    `_fetch_bkv` (`:962-969`) already issue single contiguous
    DMAs of exactly this shape — **zero runner changes for SL's
    read path**. The paged cache (`kv_cache_hbm_ref`) is the only
    truly non-contiguous HBM tensor; SL bypasses it entirely
    because short prefills have no prior cache.
  - ~~**Bundle sizing & packing logic in the runner.**~~
    **RESOLVED.** Lives in `tpu_inference/runner/subseq_planner.py`
    — the same `plan_step` function (`:157-274`) that builds
    today's DECODE / PREFILL / MIXED buckets. SL adds a packing
    pass over short-prefill SubSeqEntries to fold them into
    bundles before they leave `plan_step`; the per-iter
    `bundle_seq_range` prefetch is emitted alongside today's
    `phys_seq_indices` / `q_offsets` in `build_iter_prefetches`
    (`:362-452`), and materialized into device memory in
    `tpu_runner.py:1734-1793`. **Packing runs on the CPU host**
    (see §6.1 "Packing placement"); no on-device VPU/SALU
    prefetch program in v1.
  - ~~**Interaction with DECODE / step composition.**~~
    **RESOLVED for v1 — piggyback on MIXED bucket, presence-gated.**
    Keep the 3-tuple `request_distribution = [D, D+L, T]`
    unchanged (kernel.py:62-72 hard-asserts shape (3,) today; SL
    avoids breaking that). The MIXED bucket's iter count becomes
    `num_bundles` (not `num_short_prefills`), and the kernel-side
    dispatch is **presence-based** on `bundle_seq_range`
    prefetch: when present, the MIXED bucket runs the SL kernel
    body (block-diagonal causal mask, per-segment write); when
    absent, MIXED runs today's per-seq body. This mirrors how
    LOGICAL is dispatch-gated on `phys_seq_indices` presence
    (kernel.py:2244-2249). Under v1's "all short prefills"
    consumer assumption, every step's MIXED bucket is uniformly
    SL — no mixed-mode-within-bucket complication.
    **Long-run end state** (out of v1 scope): 4-bucket
    distribution `[D, D+L, D+L+SLM, T]` with `RpaCase.SLM` as a
    distinct case and `get_range` widened. L absorbs long
    prefills, SL absorbs short prefills, M holds only the
    awkward-remainder tail. M is the slowest, M's share is
    smallest by construction.
  - ~~**Min-prefill-length threshold below which SL applies.**~~
    **RESOLVED for v1.** Per-req routing in the v1 `plan_step`:
      - `n_R == 1` → DECODE (unchanged)
      - `n_R >= K_kernel` → LOGICAL (unchanged: chunked into
        K_kernel-sized PREFILL iters + remainder)
      - `1 < n_R < K_kernel` → **SL-eligible**; the packer folds
        these into bundles ≤ `SL_BUNDLE_CAP = 256` (MXU width).
        No lower-bound threshold in v1 — any multi-token
        short-prefill is a packing candidate.
      - Reqs the packer can't fit (e.g., last odd remainder that
        won't fit the current bundle): emit as MIXED single-iter
        (= today's path).
    Threshold tightening (e.g., bottom-cut at `min_prefill = 4`
    if tiny prefills add overhead beyond MXU gain) is empirical
    and revisited after v1 lands.
  - **`bq` re-tuning** (empirical, **post-v1**). MIXED's `bq`
    was tuned against per-seq `q_len` distributions; SL bundles
    are denser, so the bq sweet spot will shift. Plan: tune
    SL separately from MIXED using the existing v1/v2 tuner once
    the kernel builds.
  - **Tail-waste characterization** (empirical, **post-v1**).
    Measure packing efficiency distribution on the actual
    short-prefill arrival patterns we care about. If tail-waste
    is dominantly large, the split-policy v2 effort gets
    prioritized; if small, no-split stays the long-term policy.

### 6.1 Packing placement: CPU host (v1 decision)

The bundling decision is small, serial-dependent, and data lives
on CPU already. **v1 packs in `plan_step` (Python, host).** A
straight first-fit-decreasing over `n_R` values, bounded by
`SL_BUNDLE_CAP = 256`, runs in microseconds for thousands of
short prefills. Output is one new int32 array
`bundle_seq_range[max_num_bundles, 2]` emitted alongside today's
prefetches and DMA'd to SMEM via the standard mechanism.

Alternatives considered:

  - **SALU on-device** — scalar core looping over N short
    prefills serially. Slow (one number per cycle, N=1000 takes
    ~1000 cycles). Output target debatable (SMEM tight, VMEM
    overkill). Adds latency on the critical path before iter 0
    can start.
  - **VPU cumsum on-device** — vectorized cumsum + serial chop.
    Cycle math works out (~70 VPU cycles total) and *can*
    pipeline with iter-0 MXU work, but only with a forked kernel
    path that handles "wait on VPU result before iter k > 0".
    The kernel has been evolving fast (LOGICAL-aware fixes, KV
    race elimination, OOM classification, validate-step
    reclassification, all in the last weeks); maintaining a
    parallel SL kernel path that mirrors these bug fixes is a
    chronic tax. Defer to v2 if CPU profiling shows a bottleneck.
  - **Hybrid (CPU pushes bundle 0, VPU computes bundles 1+)** —
    technically reduces critical-path latency but requires the
    same forked kernel path as VPU-only. Same maintenance
    concern.

v1 commitment: **pure CPU**. CPU runs concurrently with the prior
step's TPU compute; as long as packing completes before the next
pallas_call dispatch, it's free. Profiling should confirm host
isn't already the bottleneck; today's `plan_step` is a similar
O(N) loop, so adding the bundling pass is marginal.

### 6.2 Sizing constraints: where the binding limit sits

| Constraint | Sizing knob | What it caps in v1 |
|---|---|---|
| **MXU width** (256 on v5e/v6e/v7x) | `SL_BUNDLE_CAP` | tokens per iter — **256 = ~17 short prefills at prefill_len=15** |
| **VMEM** (64 MB) | tile params (bq, bkv, num_heads, head_dim, dtype) via `get_vmem_estimate_bytes` (kernel.py:531-562) | tile-config feasibility; at SL's 256-token bundle, VMEM uses ~12-16 MB — has ~4-5× headroom, **not the binding constraint** |
| **SMEM** (256 KB) | `max_num_seqs`, `pages_per_seq` via `get_smem_estimate_bytes` (kernel.py:498-528) | seqs **per pallas_call** — ~8K at pages_per_seq=4, ~13K at pages_per_seq=1 |
| **HBM** (96 GB on v7x-1) | `max_num_batched_tokens`, model weights, KV history | tokens **per step** per chip — ~16-32K Q tokens for Llama 8B (weights take 16 GB, transient activations dominate for the rest) |

**On the "SMEM starves VPU" concern**: SMEM prefetches are loaded
**once per `pallas_call`** (scalar prefetch DMA before the iter
loop runs), and each iter does O(1) scalar reads from SMEM. The
iter loop runs at full VPU/MXU throughput regardless of how full
SMEM is, as long as everything fits at compile time. SMEM caps
*how many seqs one pallas_call can process*, not the per-iter
compute rate. **No starvation in the compile-feasible region.**

**On bundle size vs. MXU width**: bundles bigger than 256 force
sub-iter MXU launches inside one Pallas iter — each MXU launch
still 256-wide, no MXU-utilization gain. The marginal win from
bigger bundles is Pallas-iter overhead amortization, but at
bundle=256 each iter already amortizes ~17 short prefills' worth
of overhead. So **v1 commits to `SL_BUNDLE_CAP = 256`**, leaving
VMEM 75%-empty by design.

**On workloads exceeding `max_num_seqs` per pallas_call**:
multiple pallas_calls per step. The runner already handles
multi-call dispatching for DP-rank sharding; SL inherits this
pattern transparently. No new infrastructure needed.

**Single-chip v1 envelope** (Llama 8B, v7x-1, bf16, prefill_len=15):
`max_num_seqs ≈ 8K` per pallas_call, `max_num_batched_tokens ≈
16-32K` per step. Workloads above this require DP/TP sharding
context — explicit out-of-scope for v1.

---

## 7. Glossary

  - **Short prefill** — a prefill seq with `q_len ≪ K_kernel`
    (concretely: ≪ 256). Routed through MIXED today, would route
    through SLM with this change.
  - **Bundle** — a packed group of N short prefills whose
    concatenated `q_len` ≤ K_kernel. One bundle = one SLM iter.
  - **Segment** — within a bundle, the contiguous Q-token range
    belonging to one original short prefill. Bundles have
    multiple segments; each segment's score is one diagonal block
    of the bundle's `bq × bkv` matrix.
  - **No-split policy** — packer never splits a short prefill
    across two bundles. The simplification this doc relies on
    throughout.
  - **K_kernel** — the kernel's static Q-tile capacity, today
    typically 256 on v5e/v6e/v7x. Bounds the maximum tokens per
    bundle.
