# Baseline job

Run from the TPU inference `bench` checkout on the CPU VM:

```bash
bash tmp/baselines/run.sh --name serving_baselines
```

The launcher builds an image, submits one TPU job, polls, retrieves the final
results, verifies and compresses them, and cleans the run-owned image and GCS
prefix. It uses the existing environment profile; only the run name is needed.
Docker, gcloud and CDK must be available and authenticated on the CPU VM.
The build does not use the CPU VM's conda environment or editable vLLM install.

The image uses vLLM `bec0a4ede621bc9f1ec7be39b79ed2e286da4ca1` and TPU inference
based on `4c0dc1cd11a65b0631a9b36696bc0774db5ce15c`. The builder rejects runtime
source changes from that TPU base for these initial baseline measurements.
The added launch files do not change either runtime's implementation.

| Input/output | Client concurrency | Requests | Warmups | Server sequences/rank |
|---|---:|---:|---:|---:|
| 8192/1024 | 256 | 2560 | 512 | 64 |
| 1024/1024 | 256 | 2560 | 512 | 128 |

The server and client arguments come directly from the two `infx` scripts in
`scripts/vllm/benchmarking/`, copied unchanged from commit `60a6ccddb`.
Ratio 1.0, chat templates, ignore EOS and every server performance flag remain
unchanged. No flush timeout override is added: the source default is 30000 ms.
The runner adjusts executable/output paths, adds a result filename, and pins
one checkpoint revision and tokenizer snapshot for both cases.

The TPU container freshly clones InferenceX, checks out
`d089a9138c53d16c6388e4251a078fee8ca7bea6` (the client revision selected by the
upstream baseline scripts), then invokes
`utils/bench_serving/benchmark_serving.py`. No local client clone is needed.
Client source, resolved revisions, package versions and exact commands are
saved with the measurements. There is no automatic benchmark/job retry.

The image installs both sources editable. CPU Torch 2.10 and torchvision 0.25
match the selected TPU source requirements. vLLM's build metadata asks for
Torch 2.11, so the build explicitly disables isolation and compiles against
installed Torch 2.10. Dependency checks, CPU imports and CLI checks run before
publication; TPU execution and full-model compilation still need live testing.
No CUDA toolkit is installed. Build logs capture dependency or import failures.

The job mounts disk-backed scratch with a 600 GiB volume size ceiling and no
explicit disk reservation. After scheduling, it checks actual free space for
checkpoint bytes plus 64 GiB headroom before downloading. Scratch does not
provision a separate disk. The cache is excluded from results and removed with
the Pod. Each case starts a fresh server on the same TPUs.

A failed case saves its stderr and the next case still runs. Any failed case
makes the aggregate job fail. Results contain per-case commands, server/client
logs, errors and JSON/CSV summaries. Git-visible output is compressed under
`tmp/baselines/results/archives/`; launcher logs are compressed under
`tmp/workflow/local/logs/`. Unverified final output retains remote resources for
recovery. Ctrl-C detaches the local controller; resume the saved campaign to
finish collection and cleanup.

Local verification:

```bash
python3 -m unittest discover -s tmp/baselines -p 'test_*.py' -v
python3 -m unittest discover -s tmp/workflow/tests -v
```
