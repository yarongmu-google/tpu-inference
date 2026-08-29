# Every line is standalone copy-paste-able and tees its full output to
# ~/vllm_logs/<label>_<timestamp>.log - server startup errors (mesh
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
mkdir -p ~/vllm_logs; ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee ~/vllm_logs/fp8_gmm_ep_$(date +%Y%m%d_%H%M%S).log

mkdir -p ~/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee ~/vllm_logs/fp8_v1_ep_$(date +%Y%m%d_%H%M%S).log

# ---- bf16 3-way: Qwen3-30B-A3B (fits 8 chips in bf16) - same decode-heavy
# ---- client, three MoE paths. CAVEAT: representative for PLUMBING, not for
# ---- a fair perf fight - I_moe=768 -> I/P=96, so v2's gate/up split is off
# ---- the 128-lane register boundary (pad + lane shuffles it never pays at
# ---- the Qwen3.5 shape where I/P=128). The comparison that matters stays
# ---- the fp8 397B, which needs v2 fp8 support.

# 3-way line 1: GMM_EP baseline (expert-parallel, default 2D sharding)
mkdir -p ~/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee ~/vllm_logs/bf16_gmm_ep_$(date +%Y%m%d_%H%M%S).log

# 3-way line 2: v1 fused_ep_moe (expert-parallel, MUST stay on the default
# 2D (data, model) mesh: no NEW_MODEL_DESIGN, no dp_attention)
mkdir -p ~/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee ~/vllm_logs/bf16_v1_ep_$(date +%Y%m%d_%H%M%S).log

# 3-way line 3: v2 TP decode kernel (NO expert parallel; NEW_MODEL_DESIGN +
# dp_attention folds the slice into the attn_dp axis - the one config whose
# single >1 mesh axis passes the v2 guard). Engagement is confirmed by the
# log line "[MoE]: using the TP decode kernel".
mkdir -p ~/vllm_logs; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee ~/vllm_logs/bf16_v2_tp_$(date +%Y%m%d_%H%M%S).log
