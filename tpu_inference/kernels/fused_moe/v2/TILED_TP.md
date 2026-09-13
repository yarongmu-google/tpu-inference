# Token-tiled TP MoE

`fused_moe_tp_tiled_serving` accepts the existing TP serving weight layout.
The grid is `(token_tile, expert)`, with a parallel token dimension and an
ordered expert reduction. The accumulator is VMEM scratch, initialized at
expert zero and written out after the last expert. Its size depends on the
static token tile, not on the full scheduler token bucket.

## Residency and communication

- Local hidden states and router logits enter from HBM. Top-k selection,
  routing metadata and activation quantization run inside the kernel.
- Quantized token rows, selected expert ids, gates and activation scales
  are all-gathered directly between symmetric VMEM buffers.
- A barrier before each token tile prevents a fast rank from overwriting
  a peer's preceding tile. Both send and receive DMA credits are drained.
- Membership, prefix slots, one-hot gather/combine operators, the expert
  intermediate and the FP32 output accumulator remain in VMEM.
- One vector-produced expert count is copied to on-chip SMEM for scalar
  loop bounds. This is not an HBM packing buffer.
- Completed BF16 TP output partials are written to HBM in rank-major order,
  then reduced and scattered with the existing JAX collective. Tail rows
  introduced for DMA alignment are masked and removed.

For tensor activation scaling, a single global amax is computed before the
Pallas call and supplied through scalar prefetch. It covers the whole input
bucket, preserving the existing tensor-scale meaning across token tiles.
Per-token scaling is computed inside the kernel.

## Tensor layout audit

The orientations below are source-level contracts. Physical HBM/VMEM tiling,
VREG layouts/counts, packing, replicated broadcasts, spills and matrix-boundary
relayouts remain unverified until the installed TPU backend compiles the kernel.
CPU interpretation and IR export do not supply that evidence.

`P` is the TP width, `L` the local token tile, `T=P*L`, `k` the selected
expert count, `D` the hidden width, `I` the local expert intermediate width,
and `C1/C2` the two compute row tiles. FP8 storage means e4m3 throughout.

| Tensor or expression | Shape / dtype | Producer, consumer and lifetime |
| --- | --- | --- |
| Input tokens, local copy, quantization work | `[L,D]` BF16 / FP32 | HBM-to-VMEM DMA, then row scaling; once per token tile. Feature-minor. |
| Input logits, routing scores and expert iota | `[L,E]` input; `[E,L]` FP32 / int32 | One input transpose; top-k compares and expert-axis reductions with tokens on lanes. |
| Top-k maxima, IDs, softmax intermediates | `[1,L]`, `[k,L]` FP32 / int32 | Retain reduced dimensions, concatenate along k, normalize along k. No `[L,k]` round trip. |
| Tail row iota / validity | `[k,L]` int32 / Boolean | Generated directly in the final orientation, only when padding exists. Both select consumers are 32-bit. |
| Quantized all-gather tokens | `[T,D]` FP8 or BF16 | Local row slices, remote DMA, gather matmul; retained for the token tile. |
| Per-token amax / inverse | `[L,1]` FP32 | Reduction and immediate feature broadcast, once per tile. Inspect replicated layout; no persistent token-column table. |
| Metadata transport | `[P,k,L]` IDs/gates; `[P,1,L]` scales | Full rank slabs for DMA, including L below 128. No remote sub-lane slice into `[k,T]`. |
| Canonical metadata | `[k,T]` IDs/gates; `[1,T]` scales | Concatenate rank slabs along token lanes once per tile. Inspect narrow-L concatenation and transport padding. |
| Membership, gates, prefix, slots | `[k,T]` Boolean; `[1,T]` int32 / FP32 | Expert comparisons, reductions and prefix sums. Retained as lane-major scratch for row-loop access. |
| Expert count | `[1,128]` int32 VMEM-to-SMEM; scalar | Reduction, DMA handoff, runtime loop bound. Replication is deliberate. |
| Gather iota / mask / one-hot | `[C1,T]` int32 / Boolean / FP32 | Direct 2-D iota; both mask selects produce FP32, followed by operand conversion. Verify backend does not introduce incompatible predicate layouts. |
| Gathered tokens and scales | `[C1,D]` operand dtype; `[C1,1]` FP32 | Gather matmul and scale reduction feed GMM1. Scale column is a short-lived feature broadcast. |
| Expert weights and scales | `[D,2I]`, `[I,D]` FP8 or BF16; `[1,2I]`, `[1,D]` FP32 | Per-expert HBM-to-VMEM DMA, reused across row loops. No runtime weight-layout conversion. |
| GMM1, activation, GMM2 staging | `[C1,2I]` FP32, `[C1,I]` BF16, `[C2,D]` BF16 | Feature-minor matrix chain; GMM2 staging supports chunked output consumption. |
| Combine metadata slices and operator | `[1,ct]`, `[C2,ct]` | Aligned lane slices, direct 2-D iota, FP32 gate selection then BF16 conversion. No per-row-loop `[T,1]` metadata transposes. |
| Combine contraction and accumulator | `[C2,ct]` with `[C2,cd]` -> `[ct,cd]` FP32 | Contract dimension zero of both operands. This implies a transposed LHS; its backend conversion is not assumed free. Read/update only one accumulator block. |
| Output staging and partials | `[T,D]` BF16; rank-major HBM rows | Convert the accumulator once per token tile, DMA rank slices, then reduce-scatter and trim. |
| Tensor amax prefetch, indices, semaphores | Scalar prefetch / scalar / DMA arrays | Control data. The wrapper's scalar reshape is outside vector routing. |

Scale-wrapper singleton reshapes preserve the existing serving contract;
verify surrounding HLO for any materialized copies. For every row broadcast,
inspect the actual inferred replication rather than pricing from shape alone.
The remaining `.T` operations in the kernel are the input-logit boundary
and the local quantization-scale boundary, each once per token tile.

Large logical vectors still require backend scrutiny. At the default FP8
shape (L=128, T=1024, D=4096, I=128, C1=64, C2=32), representing the entire
FP32 quantization work, gathered-token result and GMM2 result with `(8,128)`
tilings would take 512, 256 and 128 VREGs respectively. These are representation
counts, not measured simultaneous liveness: the backend must schedule/chunk
or spill those values. The new `[128,128]` FP32 accumulator update represents
16 VREGs; that alone does not prove the complete live set fits. Full expert
weight operands likewise depend on backend staging. Inspect these operations
before claiming register fit or a speedup; this patch does not retile all GMMs.

The optional `combine_token_rows` and `combine_hidden_cols` controls default
to 128. Both are positive multiples of 128; the effective hidden chunk must
divide D. Token tiles divisible by the requested token chunk use a runtime
chunk loop; other token tiles use one whole-token-tile update to avoid
unaligned lane slices. These are initial settings, not measured optima.
The new BF16 GMM2 staging and three lane-major metadata scratch rows stay
in VMEM; include their lifetimes and compiler temporaries in the memory audit.

## Occupancy and row tiles

There is no heuristic per-expert capacity. Within a token tile, the exact
expert count controls runtime loops. Every selected token is processed,
even when all tokens choose the same expert. A zero count skips weight and
scale DMA, gather, GMM1, activation, GMM2 and combine. Membership and count
calculation still execute for each expert grid cell.

The static defaults are `token_tile_size=1024` global rows and
`bf16_rows=32`. FP8 GMM1 uses `2 * bf16_rows`; GMM2 consumes BF16 and uses
`bf16_rows`. All-BF16 weights use `bf16_rows` in both GMMs. These are compute
tile sizes. The minimum DMA row alignment is separately 32 for FP8 and 16
for BF16.

For FP8 weights, work within one token tile is:

| Actual expert load | GMM1 rows, default | GMM2 rows, default | GMM1 rows, bf16_rows=16 | GMM2 rows, bf16_rows=16 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | Skip | Skip | Skip | Skip |
| 10 | 64 | 32 | 32 | 16 |
| 20 | 64 | 32 | 32 | 32 |
| 40 | 64 | 64 | 64 | 48 |
| 64 | 64 | 64 | 64 | 64 |
| 80 | 128 | 96 | 96 | 80 |

The number of rows is rounded by
`((count + rows - 1) // rows) * rows`, implemented as a runtime number of
fixed-shape matmuls. Routing changes do not change array shapes or retrace
the kernel. Changing token buckets or static tuning choices can compile a
new executable, as with other serving kernels.

An expert's weights are loaded once per token tile and reused across its
row loops. A larger batch can therefore reload the same expert weights for
multiple token tiles. Splitting assignments across token tiles can also
increase total padding. These are performance tradeoffs to measure.

## Serving integration

The existing `USE_MOE_TP_DECODE_KERNEL` flag selects this implementation for
supported TP softmax/SwiGLU calls. `MOE_TP_DECODE_MAX_TOKENS` is retained for
launch-script compatibility but no longer gates this path. Neither the old
1024-global-token ceiling nor the average-load capacity formula applies.
The scheduler's `max-num-batched-tokens` is a separate setting and is not
changed here.

Biases, alternate routing modifiers, incompatible dtypes/quantization
scales, and calls requesting unreduced/unscattered results retain the
stock GMM path. The previous decode kernel and standalone tuning interface
remain available, but their capacity and pipeline knobs do not configure
this serving implementation.

This implementation uses one expert weight shard at a time. It does not
carry over the previous kernel's expert-block prefetch pipeline. A speedup
is not established by correctness checks; device profiles must determine
whether to add pipelining or adjust token and row tiles.

## Validation

`tests/kernels/fused_moe_tiled_tp_test.py` covers BF16/FP8 numerics, fully
skewed routing, poisoned unused experts, both row-tile pairs, padded tails,
multiple token tiles, two/eight-device collectives, token/tensor activation
scaling, routing changes without retracing, and serving selection above the
legacy token gate. TPU IR export checks include the 8192-token serving
shape, without allocating the full model on the host.

CPU interpretation and TPU IR export do not validate final TPU backend
layout, scratch allocation, device timing or model-level accuracy. Before
performance comparisons, compile and execute on TPU, compare numerical
results under deliberately skewed routing, inspect compiled VMEM use and
weight-copy behavior, and run model-level accuracy checks. Then measure
both token-tile and row-tile choices against the existing serving path.
