# 8k/1k
python ./bench_serving/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=8192 --random-output-len=1024 --num-prompts=2560 --random-range-ratio=1.0 --ignore-eos --save-result --result-dir /tmp/vllm_bench_wbt4bp0_ --max-concurrency=256 --num-warmups=512 --percentile-metrics=ttft,tpot,itl,e2el --use-chat-template

# 1k/1k
python ./bench_serving/benchmark_serving.py --model Qwen/Qwen3.5-397B-A17B-FP8 --dataset-name random --backend vllm --random-input-len=1024 --random-output-len=1024 --num-prompts=2560 --random-range-ratio=1.0 --ignore-eos --save-result --result-dir /tmp/vllm_bench_fl3_mubj --max-concurrency=256 --num-warmups=512 --percentile-metrics=ttft,tpot,itl,e2el --use-chat-template
