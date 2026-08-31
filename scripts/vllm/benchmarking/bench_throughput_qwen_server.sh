# Run each line FROM THE tpu-inference REPO ROOT (vllm is pip -e
# installed, so the server runs from anywhere; the log path tmp/vllm_logs
# is repo-relative so it can be pushed). When the server exits (ctrl-C),
# the line xz-compresses its own log - then: git add tmp/vllm_logs.
# Every line is standalone copy-paste-able and tees its full output to
# tmp/vllm_logs/<label>_<timestamp>.log - server startup errors (mesh
# asserts, OOM, kernel fallbacks like "[MoE]: using the TP decode
# kernel" / "Only 2D mesh is supported") only show up in these logs.

# ---- fp8 397B (production model). v2 cannot engage until it supports
# ---- fp8 (w13/w2 scales fail its guard -> silent GMM fallback), so this
# ---- model is a 2-way: GMM_EP baseline (line 1) vs v1 fused_ep_moe
# ---- (line 2). NB line 2's NEW_MODEL_DESIGN=1 + enable_dp_attention
# ---- builds a multi-axis mesh, but fused_ep_moe raises
# ---- "Only 2D mesh is supported" (v1/kernel.py) - use the v1 line in
# ---- the bf16 section below as the template for a working v1 config.
# ---- (ENABLE_PALLAS_TP_MOE matches nothing in tpu_inference - dead.)
L=tmp/vllm_logs/fp8_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# ---- fp8 line 2: v1 fused_ep_moe. v1 hard-requires the LEGACY 2D
# ---- (data, model) mesh (kernel.py: "Only 2D mesh is supported",
# ---- all non-EP axes size 1) - so NO NEW_MODEL_DESIGN and NO
# ---- dp_attention here (the old broken variant set both and raised
# ---- at startup). CAVEAT: attention therefore runs TP, not DP - a
# ---- different attention config than lines 1/3/4, so this number is
# ---- a LOOSE comparison; v1's clean head-to-head vs v2 is the
# ---- kernel bench (2642-2668 vs ~828 wall, same mesh).
L=tmp/vllm_logs/fp8_v1_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# ---- fp8 397B TP-MoE pair at the DESIGN POINT (max-num-seqs=512 ->
# ---- decode steps carry 512 global tokens, the shape the v2 fp8
# ---- kernel was built and tuned at; device 431us vs bf16 790 on the
# ---- kernel bench). Line 3 = GMM_TP baseline (identical sharding, no
# ---- v2); line 4 = the v2 fp8 kernel (engagement confirmed by
# ---- "[MoE]: using the TP decode kernel" + the input-contract log
# ---- showing e4m3 w13/w2 WITH scale shapes - the fp8 guard requires
# ---- the per-channel in_blocks==1 contract, i.e. the DEFAULT requant;
# ---- do NOT set MOE_REQUANTIZE_BLOCK_SIZE/DISABLE_WEIGHT_
# ---- REQUANTIZATION on these lines). attn_dp_size=8 forces the
# ---- single-axis mesh the guard needs. KV at 512 seqs x 9216 fp8 is
# ---- the capacity risk - watch gpu-memory-utilization.

# fp8 line 3: GMM_TP baseline
L=tmp/vllm_logs/fp8_gmm_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# fp8 line 4: v2 TP decode kernel (fp8 w8a8, per-token act scale)
L=tmp/vllm_logs/fp8_v2_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# ---- bf16 3-way: Qwen3-30B-A3B (fits 8 chips in bf16) - same decode-heavy
# ---- client, three MoE paths. CAVEAT: representative for PLUMBING, not for
# ---- a fair perf fight - I_moe=768 -> I/P=96, so v2's gate/up split is off
# ---- the 128-lane register boundary (pad + lane shuffles it never pays at
# ---- the Qwen3.5 shape where I/P=128). The comparison that matters stays
# ---- the fp8 397B, which needs v2 fp8 support. All lines carry the
# ---- SAME LIBTPU collective flags + ONEHOT threshold as the proven fp8
# ---- config (the RS-legalizer flag in particular may be load-bearing
# ---- under dp_attention, not just a perf tweak).

# 3-way line 1: GMM_EP baseline (expert-parallel, default 2D sharding)
L=tmp/vllm_logs/bf16_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# 3-way line 2: v1 fused_ep_moe (expert-parallel, MUST stay on the default
# 2D (data, model) mesh: no NEW_MODEL_DESIGN, no dp_attention)
L=tmp/vllm_logs/bf16_v1_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# 3-way line 3: v2 TP decode kernel (NO expert parallel; NEW_MODEL_DESIGN +
# dp_attention). attn_dp_size=8 is REQUIRED on this model: with 4 KV heads
# and bf16 KV the auto formula gives attn_dp=2/model=4 - two >1 mesh axes -
# and the v2 guard rejects multi-axis meshes (silent GMM fallback). Forcing
# attn_dp=8/model=1 gives the single-axis TP-MoE-under-DP-attention mesh.
# Engagement is confirmed by "[MoE]: using the TP decode kernel" plus the
# input-contract log right after it.
L=tmp/vllm_logs/bf16_v2_tp_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# 3-way line 4 (the DEFAULT TP-MoE): GMM_TP - same config as line 3 but
# WITHOUT USE_MOE_TP_DECODE_KERNEL, so MoE runs the stock fused_moe_func
# GMM path (unfused topk, all-gather outside the kernel). This is the
# head-to-head for line 3: identical sharding, only the MoE kernel differs.
L=tmp/vllm_logs/bf16_gmm_tp_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"


# ---- design-point pair: max-num-seqs=512 -> decode steps carry 512 global
# ---- tokens (64/chip), the EXACT shape the kernel was built and tuned at
# ---- (aligned rows, no padding path). At 64 seqs a decode step is tiny and
# ---- throughput is dominated by per-step fixed costs, which is why the 30B
# ---- numbers land near the fp8 397B baseline. Run BOTH lines for the A/B.
# ---- KV cache at 512 seqs x 9216 len is the risk - watch gpu-memory-util.

# 512-seq line A: v2 TP decode kernel
L=tmp/vllm_logs/bf16_v2_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"

# 512-seq line B: GMM_TP default (identical config minus the kernel env)
L=tmp/vllm_logs/bf16_gmm_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee "$L"; xz -9 -T0 "$L"
