# https://github.com/kimbochen/bench_serving.git
#
# Runs the REPO copy of benchmark_serving.py (scripts/vllm/benchmarking/),
# NOT ~/bench_serving: the repo copy fixes the random-dataset sampling so
# the specified in/out lens are HARD UPPER BOUNDS ([X*(1-b), X], lower
# bounds clamped >= 1). The old [X*(1-b), X*(1+b)] logic (a) sampled
# 0-token prompts at input-len=1 (int(1*0.2)=0), which the server rejects
# as bad requests, and (b) sampled outputs up to 1.8x the specified len.
# With input-len=1 there is now no sampling: every prompt is exactly 1.
BS=scripts/vllm/benchmarking/benchmark_serving.py
# CONC = client concurrency; must be 8x the server's per-rank
# --max-num-seqs or the client throttles itself (MNS and MNB are
# PER-DP-RANK; global concurrency is 8x). Default 512 matches the
# MNS=64 lines; for line 4c (MNS=128) run: CONC=1024 <paste line>.
CONC=1024
L=tmp/vllm_logs/client_fp8_mix_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; ENABLE_PALLAS_TP_MOE=1  python $BS --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos  --max-concurrency=${CONC:-512} 2>&1 | tee "$L"; xz -9 -T0 "$L"
# bf16 3-way client (same decode-heavy shape, 30B model)
# L=tmp/vllm_logs/client_bf16_mix_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; python $BS --model Qwen/Qwen3-30B-A3B --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=${CONC:-512} 2>&1 | tee "$L"; xz -9 -T0 "$L"

# pure-decode variant: input-len=1 makes prefill negligible, so per-token
# latency ~= decode step time - the clean kernel-visibility number (the
# 1024/8192 lines above are the realistic serving mix)
# L=tmp/vllm_logs/client_fp8_decode_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; python $BS --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=${CONC:-512} 2>&1 | tee "$L"; xz -9 -T0 "$L"
# L=tmp/vllm_logs/client_bf16_decode_$(date +%Y%m%d_%H%M%S).log; mkdir -p tmp/vllm_logs; python $BS --model Qwen/Qwen3-30B-A3B --dataset-name random --backend vllm --random-input-len=1 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=${CONC:-512} 2>&1 | tee "$L"; xz -9 -T0 "$L"
