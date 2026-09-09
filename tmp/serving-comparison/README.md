# Serving comparison

From the repository root on the CPU VM with the existing Python 3.12 `vllm12`
environment active:

```bash
bash tmp/serving-comparison/run.sh
```

Enter a descriptive run name. This uses the existing workflow environment,
image builder, CDK submission, monitoring, collection and cleanup. It is a
separate campaign from the running sweep. One TPU job runs baseline, 4g and 4i
sequentially with the same image and physical TPUs. It builds one owned image.

Inside the TPU container, immediately before testing, the runner freshly clones
`https://github.com/kimbochen/bench_serving.git`. It records the resolved commit
and saves the client source files. All three configurations use that exact
checkout without changing its sampler or installing dependencies. The checkout
requires outbound GitHub access from the container. Clone and import failures
are saved in the normal run outputs.

The workload is nominal 1024 input / 8192 output, concurrency 512, 2048 measured
requests, seed 0, random range ratio 0.8, no prefix, no chat template, ignore EOS,
and zero extra warmup requests. These are the historical original-client
settings; the original sampler uses `[floor(0.8 * X), X]`. This is different from
the repository client's corrected `[floor(0.2 * X), X]` at ratio 0.8. This job
does not run one-token-input diagnostics, so no zero-length sampler patch is
needed. The upstream client controls its normal initial-request behavior.

The workflow snapshots this folder's `payload/` and the repository benchmarking
directory. Results live separately under this folder's `results/`.
The runner extracts exactly the three named server commands from the frozen
`scripts/vllm/benchmarking/bench_throughput_qwen_server.sh`. It replaces only the
shell logging wrapper with managed process logging. Baseline uses 64 sequences
and 1024 batched tokens per rank; 4g uses 64 and 128; 4i uses 104 and 128 with its
power-of-two bucket ladder. The CLIENT concurrency remains 512 for every case;
this is not the historical 4i C832 operating point. Existing source comments
and unrelated server lines are not executed.

Outputs contain each configuration's server log, client log, JSON metrics,
exact commands and errors, plus `summary.json` and `summary.csv`. The summary
reports output throughput, total throughput, total throughput per physical
chip (divide by 4), mean TPOT and mean TTFT. A result is complete only if all
2048 requests succeeded; incomplete results make the job fail. The runner
continues to the next configuration after a case failure once its server has
been stopped. It refuses to proceed if a server remains on the port.
Completed cases must report the same input and output token totals.

The fresh client's commit, file hashes and source snapshot, server command
source, image source metadata and workload settings are saved under `metadata/`.
For provenance of published images and installed sources, the generic workflow
also retains its per-run build records. Each configuration starts a new server;
Docker and compilation caches may still be reused within the container.

After terminal-state collection and artifact verification, the workflow deletes
this run's image and CDK storage subfolder. Local results and errors remain.
A failed comparison can still have a verified, collected artifact bundle; its
resources can be cleaned while its failed status and error logs are retained.
Incomplete collection retains resources for recovery. Resume reconnects the
saved job; it does not restart completed benchmark configurations. Start a new
campaign to repeat the comparison.

This establishes a same-client comparison between our server configurations.
It does not establish identical sampling, software revisions or warmup settings
to the public graph without its exact benchmark invocation.

Local checks (no clone, model loading or cloud commands):

```bash
python3 -m unittest discover -s tmp/serving-comparison -p test_compare.py -v
```

Use an interpreter with the workflow YAML dependency installed.
