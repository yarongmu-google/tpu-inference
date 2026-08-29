# https://github.com/kimbochen/bench_serving.git

ENABLE_PALLAS_TP_MOE=1  python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos  --max-concurrency=512
# bf16 3-way client (same decode-heavy shape, 30B model)
# python ~/bench_serving/benchmark_serving.py --model Qwen/Qwen3-30B-A3B --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=8192 --num-prompts=2048 --random-range-ratio=0.8 --ignore-eos --max-concurrency=512
