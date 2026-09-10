# Command appendix

## baseline-20260902_163515

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_gmm_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee -a "$L"; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=1024, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## baseline-20260903_135542

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_gmm_ep_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_gmm_ep commit=$(git rev-parse --short HEAD)" | tee "$L"; ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=512, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4c-20260903_105407

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_128s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_128s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=128 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=1024, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4e-20260903_122921

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_104s_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=832 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=104,208,416,832 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=832, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4g-20260903_151322

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_64s_riders_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_64s_riders commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=512, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4g-20260903_155530

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_64s_riders_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_64s_riders commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=512, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, best_of=1, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, save_result=False, save_detailed=False, metadata=None, result_dir=None, result_filename=None, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, sharegpt_output_len=None, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, use_chat_template=False, hf_subset=None, hf_split=None, hf_output_len=None, tokenizer_mode='auto', served_model_name=None, lora_modules=None, num_warmups=0)
```

## 4h-20260903_165502

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_104s_p2buckets_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s_p2buckets commit=$(git rev-parse --short HEAD)" | tee "$L"; MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=832, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4h-20260903_181805

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_104s_p2buckets_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s_p2buckets commit=$(git rev-parse --short HEAD)" | tee "$L"; VLLM_ADMISSION_DEBUG=1 MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=256 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=832, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## 4i-20260903_191522

Server line at the commit recorded in its log:

```bash
L=tmp/vllm_logs/fp8_v2_tp_104s_singlespeed_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; echo "CFG label=fp8_v2_tp_104s_singlespeed commit=$(git rev-parse --short HEAD)" | tee "$L"; VLLM_ADMISSION_DEBUG=1 MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 LIBTPU_INIT_ARGS=' --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256 2>&1 | tee -a "$L"; cp tmp/vllm_server_stats.csv "${L%.log}_stats.csv" 2>/dev/null; xz -9 -T0 "$L"
```

Client arguments recorded in the log (original executable/script path was not recorded):

```text
Namespace(backend='vllm', base_url=None, host='127.0.0.1', port=8000, endpoint='/v1/completions', dataset_name='random', dataset_path=None, max_concurrency=832, model='Qwen/Qwen3.5-397B-A17B-FP8', tokenizer=None, use_beam_search=False, num_prompts=2048, logprobs=None, request_rate=inf, burstiness=1.0, seed=0, trust_remote_code=False, disable_tqdm=False, profile=False, ignore_eos=True, percentile_metrics='ttft,tpot,itl', metric_percentiles='99', goodput=None, request_id_prefix='benchmark-serving', chat_template_system_prompt='Reasoning effort: high', chat_template_kwargs='{}', mmlu_input_len=None, mmlu_output_len=None, mmlu_num_shots=1, mmlu_method='HELM', mmlu_use_chat_template=False, mlperf_input_len=None, mlperf_output_len=None, gpqa_output_len=2048, gpqa_use_chat_template=False, mmmu_pro_subset='vision', mmmu_pro_output_len=16, sonnet_input_len=550, sonnet_output_len=150, sonnet_prefix_len=200, random_input_len=1024, random_output_len=8192, random_range_ratio=0.8, random_prefix_len=0, top_p=None, top_k=None, min_p=None, temperature=None, tokenizer_mode='auto', served_model_name=None, run_eval=False, warmup_mode='sampled', debug=False)
```

## serving-comp-default-b7d2d6774029493e962cfee6/baseline

Recorded server command:

```bash
env VLLM_ADMISSION_DEBUG=1 ENABLE_PALLAS_TP_MOE=1 MODEL_IMPL_TYPE=vllm USE_MOE_EP_KERNEL=0 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 NEW_MODEL_DESIGN=1 'LIBTPU_INIT_ARGS= --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=1024 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image": 0, "video": 0}' --kv-cache-dtype=fp8 --enable-expert-parallel '--additional_config={"sharding": {"sharding_strategy": {"enable_dp_attention": true}}}' --block-size=256
```

Recorded client command (environment-specific interpreter path displayed as python3):

```bash
python3 /tmp/serving-comparison-client/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --backend vllm --host 127.0.0.1 --port 8000 --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 0.8 --random-prefix-len 0 --max-concurrency 512 --num-prompts 2048 --request-rate inf --seed 0 --num-warmups 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /run-work/artifacts/baseline --result-filename client.json
```

## serving-comp-default-b7d2d6774029493e962cfee6/4g

Recorded server command:

```bash
env VLLM_ADMISSION_DEBUG=1 MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=64,128,256,512 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 'LIBTPU_INIT_ARGS= --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=64 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256
```

Recorded client command (environment-specific interpreter path displayed as python3):

```bash
python3 /tmp/serving-comparison-client/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --backend vllm --host 127.0.0.1 --port 8000 --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 0.8 --random-prefix-len 0 --max-concurrency 512 --num-prompts 2048 --request-rate inf --seed 0 --num-warmups 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /run-work/artifacts/4g --result-filename client.json
```

## serving-comp-default-b7d2d6774029493e962cfee6/4i

Recorded server command:

```bash
env VLLM_ADMISSION_DEBUG=1 MODEL_IMPL_TYPE=vllm USE_MOE_TP_DECODE_KERNEL=1 MOE_TP_DECODE_MAX_TOKENS=1024 NEW_MODEL_DESIGN=1 ATTN_BUCKETIZED_NUM_REQS=true ATTN_CUSTOM_NUM_REQS_BUCKETS=128,256,512,1024 ONEHOT_MOE_PERMUTE_THRESHOLD=32768 VLLM_MOE_CHUNK_SIZE=256 'LIBTPU_INIT_ARGS= --xla_tpu_use_minor_sharding_for_major_trivial_input=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false --xla_tpu_ars_combiner_threshold_in_bytes=0 --xla_tpu_enable_async_collective_merger=false --xla_tpu_check_legacy_constraints_in_reduce_scatter_legalizer=false' vllm serve Qwen/Qwen3.5-397B-A17B-FP8 --max-model-len=9216 --max-num-batched-tokens=128 --max-num-seqs=104 --no-enable-prefix-caching --gpu-memory-utilization=0.88 --tensor-parallel-size=8 --async-scheduling --port=8000 --language-model-only --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 '--limit-mm-per-prompt={"image":0, "video": 0}' --kv-cache-dtype=fp8 '--additional_config={"sharding":{"sharding_strategy": {"enable_dp_attention": true, "attn_dp_size": 8}}}' --block-size=256
```

Recorded client command (environment-specific interpreter path displayed as python3):

```bash
python3 /tmp/serving-comparison-client/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --backend vllm --host 127.0.0.1 --port 8000 --dataset-name random --random-input-len 1024 --random-output-len 8192 --random-range-ratio 0.8 --random-prefix-len 0 --max-concurrency 512 --num-prompts 2048 --request-rate inf --seed 0 --num-warmups 0 --ignore-eos --percentile-metrics ttft,tpot,itl,e2el --save-result --result-dir /run-work/artifacts/4i --result-filename client.json
```
