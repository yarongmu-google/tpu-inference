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
