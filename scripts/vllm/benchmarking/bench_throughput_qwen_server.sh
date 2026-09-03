# Run each line FROM THE tpu-inference REPO ROOT (vllm is pip -e
# installed, so the server runs from anywhere; the log path tmp/vllm_logs
# is repo-relative so it can be pushed). When the server exits (ctrl-C),
# the line xz-compresses its own log - then: git add tmp/vllm_logs.
#
# MULTI-SLICE note: when any line here grows into a multi-slice
# workload, prepend TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434 so
# each slice's runtime metrics server gets its own port (single-slice
# lines don't need it).
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
# STALE NUMBERS REMOVED: earlier '~10500 / decode ~10280' here were
# measured with the PRE-FIX client (length sampler drew outputs
# averaging the full nominal 8192, and the figure was in+out total).
# Post-fix client (outputs mean ~4915), OUTPUT tok/s at CONC=512-1024:
#   default EP mix 8067-8125 (decode-only 8666); ours: line 4 7385,
#   4g single-speed 7952 (decode-only 8687). Compare only same-client.
L=tmp/vllm_logs/fp8_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_gmm_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; VLLM_ADMISSION_DEBUG=1 ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# ---- fp8 line 2: v1 fused_ep_moe. v1 hard-requires the LEGACY 2D
# ---- (data, model) mesh (kernel.py: "Only 2D mesh is supported",
# ---- all non-EP axes size 1) - so NO NEW_MODEL_DESIGN and NO
# ---- dp_attention here (the old broken variant set both and raised
# ---- at startup). CAVEAT: attention therefore runs TP, not DP - a
# ---- different attention config than lines 1/3/4, so this number is
# ---- a LOOSE comparison; v1's clean head-to-head vs v2 is the
# ---- kernel bench (2642-2668 vs ~828 wall, same mesh).
L=tmp/vllm_logs/fp8_v1_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v1_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# ---- fp8 397B TP-MoE pair at MNS=64: the FEASIBLE operating point,
# ---- NOT the kernel's tuned T=512 design point. Qwen3.5-397B is a
# ---- HYBRID model (15 full-attn + 45 GDN/mamba layers with
# ---- per-SEQUENCE state); vllm's hybrid paging pads every 256-token
# ---- block slot to ~13 MB, the compact-mamba sizing refuses to
# ---- engage at MNS=512, and the pool comes out at ~309k tokens =
# ---- ~33 full-length seqs - MNS=512 preemption-thrashes from the
# ---- first steps (measured 2-6k tok/s oscillation). MNS=64 is why
# ---- the original upstream config used 64. Decode steps carry
# ---- 8-64 global tokens (1-8 rows/device) - the shapes the
# ---- T/P<granule kernel fixes exist for.
# ---- Line 3 = GMM_TP baseline (identical sharding, no v2); line 4 =
# ---- the v2 fp8 kernel (engagement confirmed by "[MoE]: using the
# ---- TP decode kernel" + the input-contract log showing e4m3
# ---- w13/w2 WITH scale shapes - the fp8 guard requires the
# ---- per-channel in_blocks==1 contract, i.e. the DEFAULT requant;
# ---- do NOT set MOE_REQUANTIZE_BLOCK_SIZE/DISABLE_WEIGHT_
# ---- REQUANTIZATION on these lines). attn_dp_size=8 forces the
# ---- single-axis mesh the guard needs.

# fp8 line 3: GMM_TP baseline
L=tmp/vllm_logs/fp8_gmm_tp_64s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_gmm_tp_64s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4: v2 TP decode kernel (fp8 w8a8, per-token act scale)
L=tmp/vllm_logs/fp8_v2_tp_64s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_64s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=512 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4g: line 4 with prefill ALWAYS RIDING ON THE KERNEL.
# Line 4 gates the kernel at 512 tokens, so any step carrying
# prompt riders (512 decode + prompt chunk > 512) falls back to
# the slow stock GMM-TP path - our mix runs two-speed steps
# where the default runs one. Fix: cap the per-step budget at
# the tuned shape (MNB=128/rank = 1024 global) and raise the
# gate to 1024 -> EVERY step is 512 decode + up to 512 rider
# tokens, always on the kernel at its tuned T=1024 blocks.
# Prompts chunk across ~5 steps (TTFT up, throughput is the
# bet). Same provably-safe carve as line 4. Client: CONC=512.
# Decides how much of the ~10% mix gap vs the default was the
# two-speed step structure rather than EP prefill sharding.
L=tmp/vllm_logs/fp8_v2_tp_64s_riders_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_64s_riders commit=$(git rev-parse --short HEAD)" | tee "$L"; VLLM_ADMISSION_DEBUG=1 MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4c: v2 at the SWEEP WINNER (per-rank MNS=128 = 1024
# global, MNB=256) - proxy sweep: 5607 out tok/s vs 3278 at the
# line-4 config (concurrency-saturated client). CAVEAT: at
# MNS=128 the mamba state reservation shrinks the pool to 3.5M
# tokens / 381 full-length seqs (vs 9.4M/1023 at MNS=64); the
# proxy's short seqs fit, but the REAL 1k/8k workload at 1024
# global x ~5.1k avg ctx needs ~5.2M -> watch for "preempt" in
# this log during the full client; if it thrashes, retry
# --max-num-seqs=96. Drive the client at --max-concurrency=1024.
L=tmp/vllm_logs/fp8_v2_tp_128s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_128s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=128 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4e: the f32-state FIXED-POINT config - MNS=104/rank
# (832 global) from solving MNS*avg_ctx*15.4KiB*h = budget -
# slot_cost*MNS: the carve where mamba slots and attention pages
# are CONSISTENT at f32 state. Theory: ~832 concurrent at
# interpolated TPOT ~75-85ms -> ~8.5-9k out tok/s, above the
# default's 8125, with NO bf16-state accuracy caveat and no
# valley (expected KV well under 100%). Client: CONC=832.
# Falsifies or confirms the fixed-point sizing theory directly.
L=tmp/vllm_logs/fp8_v2_tp_104s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=832 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=104,208,416,832 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4h: 4e with POWER-OF-2 BUCKETS ONLY - the step-time
# cliff experiment. 4e (buckets 104/208/416/832 global = per-rank
# 13/26/52/104) measured 124ms steps at 832-wide f32 where
# component arithmetic predicts ~70; 4g at per-rank 8/16/32/64
# (powers of 2) shows no such anomaly at the same 1024-token MoE
# shape. Hypothesis: the per-seq kernels (GDN state, attention)
# hit generic slow paths at non-power-of-2 request buckets.
# This line changes ONLY the ladder: per-rank 16/32/64/128, so
# the plateau decodes at the 128-row shape (104 live + 24 pad);
# gate 1024 tracks the padded token bucket. Same MNS=104, same
# carve, same client (mix, CONC=832, VLLM_ADMISSION_DEBUG=1
# optional). Verdict: TPOT ~70-85ms -> bucket cliff CONFIRMED,
# f32 gets its width back (~9.5-10k); TPOT still ~120ms ->
# hypothesis dead, the step profile is next.
L=tmp/vllm_logs/fp8_v2_tp_104s_p2buckets_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s_p2buckets commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4d: 4c + bf16 GDN SSM state (--mamba-ssm-cache-dtype;
# the flag flows vllm cache_config -> MambaSpec.dtypes -> the TPU
# cache builder verbatim). The SSM state is the dominant per-seq
# cost (f32 4.2MB/layer-group): bf16 halves it, roughly doubling
# the pool - the direct attack on the KV>99% admission-starvation
# valley that eats ~30% of the 4c run. Verify in the log: the
# padded block size and "GPU KV cache size" should both improve,
# mamba_dtype line shows bfloat16. ACCURACY CAVEAT: this changes
# the 45 GDN layers' recurrence numerics; the random-token bench
# cannot detect quality loss - an MMLU-style eval is owed before
# any production claim.
L=tmp/vllm_logs/fp8_v2_tp_128s_bf16state_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_128s_bf16state commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=128 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 --mamba-ssm-cache-dtype=bfloat16 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4f: bf16 state at ITS OWN fixed point - MNS=152/rank
# (1216 global; ceiling 1209, +0.6% inside the h=1.35 margin).
# 4d ran bf16 at 1024 = under its ceiling; this cashes the rest.
# Decode pads 160/rank -> 1280 global (tuned block row exists).
# The step-time question decides it: if the bf16 curve is flat
# enough, 1216 x ~105-115ms -> ~10.5-11k; if it is as steep as
# f32 above 1024, this LOSES to 4d and the bf16 optimum is near
# 1024. Client: CONC=1216. Prefix VLLM_ADMISSION_DEBUG=1 to
# certify refusals ~= 0 at the computed carve (tight tails only
# inferred it for 4e). Same accuracy caveat as 4d.
L=tmp/vllm_logs/fp8_v2_tp_152s_bf16state_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_152s_bf16state commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1216 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=152,304,608,1216 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=152 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 --mamba-ssm-cache-dtype=bfloat16 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# fp8 line 4b - CONTINGENCY, run only if line 4 halts at
# SparseCoreSequencer again: identical to line 4 but with ALL SC
# collective offloads disabled (line 4 only disables reduce-scatter).
# If 4b serves cleanly where 4 halts, the fault is the SC-offloaded
# collectives sharing the step with the kernel's ICI DMAs (keep 4b's
# flags); if 4b halts identically, the SC path is a red herring and
# the kernel's own remote traffic is next.
L=tmp/vllm_logs/fp8_v2_tp_64s_noscoff_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_64s_noscoff commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=512 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_enable_sparse_core_collective_offload_all_gather=false --xla_tpu_enable_sparse_core_collective_offload_all_reduce=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

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
L=tmp/vllm_logs/bf16_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_gmm_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# 3-way line 2: v1 fused_ep_moe (expert-parallel, MUST stay on the default
# 2D (data, model) mesh: no NEW_MODEL_DESIGN, no dp_attention)
L=tmp/vllm_logs/bf16_v1_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_v1_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=8,16,32,64 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --enable-expert-parallel --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# 3-way line 3: v2 TP decode kernel (NO expert parallel; NEW_MODEL_DESIGN +
# dp_attention). attn_dp_size=8 is REQUIRED on this model: with 4 KV heads
# and bf16 KV the auto formula gives attn_dp=2/model=4 - two >1 mesh axes -
# and the v2 guard rejects multi-axis meshes (silent GMM fallback). Forcing
# attn_dp=8/model=1 gives the single-axis TP-MoE-under-DP-attention mesh.
# Engagement is confirmed by "[MoE]: using the TP decode kernel" plus the
# input-contract log right after it.
L=tmp/vllm_logs/bf16_v2_tp_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_v2_tp commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# 3-way line 4 (the DEFAULT TP-MoE): GMM_TP - same config as line 3 but
# WITHOUT USE_MOE_TP_DECODE_KERNEL, so MoE runs the stock fused_moe_func
# GMM path (unfused topk, all-gather outside the kernel). This is the
# head-to-head for line 3: identical sharding, only the MoE kernel differs.
L=tmp/vllm_logs/bf16_gmm_tp_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_gmm_tp commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"


# ---- design-point pair: max-num-seqs=512 -> decode steps carry 512 global
# ---- tokens (64/chip), the EXACT shape the kernel was built and tuned at
# ---- (aligned rows, no padding path). At 64 seqs a decode step is tiny and
# ---- throughput is dominated by per-step fixed costs, which is why the 30B
# ---- numbers land near the fp8 397B baseline. Run BOTH lines for the A/B.
# ---- KV cache at 512 seqs x 9216 len is the risk - watch gpu-memory-util.

# 512-seq line A: v2 TP decode kernel
L=tmp/vllm_logs/bf16_v2_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_v2_tp_512s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"

# 512-seq line B: GMM_TP default (identical config minus the kernel env)
L=tmp/vllm_logs/bf16_gmm_tp_512s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=bf16_gmm_tp_512s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 VLLM_MOE_CHUNK_SIZE=256 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3-30B-A3B --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=512 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
