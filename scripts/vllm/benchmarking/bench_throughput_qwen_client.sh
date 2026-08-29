# https://github.com/kimbochen/bench_serving.git

mkdir -p ~/vllm_logs; ENABLE_PALLAS_TP_MOE=1  python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos  --max-concurrency=512 2>&1 | tee ~/vllm_logs/client_fp8_mix_$(date +%Y%m%d_%H%M%S).log
# bf16 3-way client (same decode-heavy shape, 30B model)
# mkdir -p ~/vllm_logs; python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3-30B-A3B --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=512 2>&1 | tee ~/vllm_logs/client_bf16_mix_$(date +%Y%m%d_%H%M%S).log

# pure-decode variant: input-len=1 makes prefill negligible, so per-token
# latency ~= decode step time - the clean kernel-visibility number (the
# 1024/8192 lines above are the realistic serving mix)
# mkdir -p ~/vllm_logs; python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=512 2>&1 | tee ~/vllm_logs/client_fp8_decode_$(date +%Y%m%d_%H%M%S).log
# mkdir -p ~/vllm_logs; python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3-30B-A3B --dataset-name random --backend vllm --random-input-len=1 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=512 2>&1 | tee ~/vllm_logs/client_bf16_decode_$(date +%Y%m%d_%H%M%S).log
