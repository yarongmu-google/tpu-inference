# v2 decode MoE: pipelining plan

Status: v1 = 2700 us, v2 = 3859 us, weight-stream floor ~500 us.
Tuning bd1c/bd2c/capacity is flat (3871-3921 us), so the cost is not in
the knobs we have.

## What v1 does

Four nested levels, each with the same idiom - *issue next, wait current,
compute current*:

| level | unit of work | double buffer | prefetch |
|---|---|---|---|
| L0 | token block `bt` | gating, output | `start_fetch_gating(bt+1)` before `wait(bt)`; `wait_send_bo(bt-2)` before reusing an output slot |
| L1 | one expert | a2a staging (`e_sem_id`) | `start_a2a_scatter(next)` before `wait_scatter(current)` |
| L2 | weight **tile** `(bf, bd1)` | weight buffers (`bw_sem_id`) | `start_fetch_next_bw(...)` before `wait_fetch_bw(...)` |
| L3 | vreg chunk `btc x bd1c x bfc` | - | small dots: `[btc,bd1c] @ [bd1c,bfc]` |

Also: `num_loops = cdiv(dyn_sz, btc)` - the innermost loop runs over the
expert's ACTUAL token count, not a padded capacity.

## What we should NOT copy

- **L0 token blocking.** v1 blocks tokens because its activations stream.
  Ours are VMEM-resident by design (4 MB); adding `bt` would buy nothing
  and would break the single-gather premise.
- **L1 a2a prefetch.** v1 overlaps per-expert token DMAs. We have no
  per-expert token movement at all - dispatch is a matmul. There is
  nothing to prefetch at this level.
- **Dynamic `dyn_sz` loops.** Ours must stay static (no dynamic
  addressing); `capacity` padding is the price and it is already tuned.

So only L2 and L3 transfer, and L2 is where the value is: weights are the
*only* streamed data in our kernel.

## Change 1 (primary): weight buffering per EXPERT, not per block

Today: one DMA for the whole `[be, D, 2I]` block (8.4 MB) + one for
`[be, I, D]` (4.2 MB), both waited before the FFN loop starts. Expert 0
cannot begin until all four experts' weights have landed.

Change to a per-expert double buffer:

```
w1_buf [2, D, 2I]   w2_buf [2, I, D]        (2 experts, not 2*be)
per grid step, for le in range(be):
    fetch(expert blk*be + le + 1 -> slot (le+1) % 2)   # issue next
    wait(slot le % 2)                                   # wait current
    gmm1/act/gmm2 on slot le % 2
```

Three effects, all favourable:

1. **Finer overlap.** A 3.1 MB transfer overlaps one expert's FFN instead
   of a 12.6 MB transfer overlapping a whole block.
2. **VMEM drops from ~25 MB to ~6.3 MB** (2 experts vs 2*be experts).
3. That freed VMEM **decouples `be` from the weight footprint entirely.**

## Change 2: raise `be` now that it is free

`be` currently costs `2 * be * 3.15 MB` of VMEM, which is why `be=8/16`
failed the budget assert. After change 1 it costs only the OH/x/y buffers
(`be*C*(T + 2D)` bytes; at `be=16, C=32`: ~8.5 MB). Raising `be` helps
three separate things at once:

| quantity | scaling | be=4 | be=16 |
|---|---|---|---|
| accumulator RMW traffic | `(E/be) * T*D*8B` | 2.1 GB | 0.5 GB |
| gather pushes (tokens is the pushed operand, per step) | `512 * E/be` | 65k | 16k |
| gather matmul M (rows per fill) | `be*C` | 128 | 512 |

The accumulator is the single largest per-step region (12k ops/step) and
scales as `1/be`; the gather's fills amortise over 4x more rows.

## Change 3: chunk the remaining stages (L3)

Axes with no knob today:

```
masks     T, C            full-width OH[be*C,T] and OHG_T[T,be*C]
gather    K=T, N=D        one [be*C,T] @ [T,D] expression
gmm1      N=2I            (K has bd1c)
gmm2      K=I             (N has bd2c)
combine   N=D             (M has bcT)
epilogue  T, D            out = acc, full width
```

Add `bgK/bgN` (gather), `bfc` (gmm1 N), `bfc2` (gmm2 K), `bcD` (combine
N), `beT` (epilogue). Every one is a loop plus a knob; the tuner picks up
new stages mechanically.

## The caveat that may outrank all of this

Pipelining hides latency; it does not remove work. Counting MXU pushes
(`vmatprep.subr`, one per 16 rows of a stationary block):

```
                       per step   per layer (128 steps)
gather                      512      65k
gmm1                       1024     131k
gmm2                       1024     131k
combine                     256      33k
                                    360k

v1 (FFN only, 3 matrices x 64 experts)      ~196k
ideal (1.6 GB / 128 KB per block x 16)      ~205k
```

Two findings in that table:

1. **khot's gather+combine add ~98k pushes/layer (27%)** that v1 does not
   pay. Change 2 cuts this to ~32k, which is the strongest argument for
   raising `be`.
2. **gmm2 is intrinsically wasteful under TP**: `w2` is `[I/P=128, D]`, so
   its contraction dim is 128 against a 256-deep systolic array - every
   block is half empty. That is 131k pushes doing 65k pushes' worth of
   work. The ISA's diagonal push (`vmatpush.diag`, "two 128x128 matrix
   multiplications", half the pushes) is exactly the remedy; whether
   Mosaic emits it for a K=128 operand is unverified and worth a probe.

So the order is: change 1 (structure + frees VMEM) -> change 2 (uses it,
and cuts the khot surcharge) -> measure -> change 3 only if still short.
If we remain push-bound after 1+2, the honest conclusion is that khot's
extra matmuls cost more than the DMA gather they replaced, and the
dispatch design should be revisited rather than tuned.
