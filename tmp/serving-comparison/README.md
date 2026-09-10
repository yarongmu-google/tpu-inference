# Serving comparison

On the CPU VM, with the existing Python 3.12 vllm12 environment active:

```bash
bash tmp/serving-comparison/run.sh --name serving_baseline
```

This job reruns only the missing Line 1 EP baseline at 1024/8192 input/output.
It uses both repositories' `topk` sources, client concurrency 512, 2,048 requests,
seed 0, ratio 0.8, ignore EOS, no chat template and zero extra warmups.
Previously completed 4g and 4i results remain in the summary; they are not rerun.
There is no automatic retry of a failed benchmark or remote job.

The client is freshly fetched on the TPU host and checked out at the full
commit in the plan. It is unmodified. Its ratio 0.8 samples lengths from
`[floor(0.8 * X), X]`, unlike the repository client's `[floor(0.2 * X), X]`.
Completed configurations for the same I/O shape must report identical token
counts. Counts are not compared across different I/O shapes.

The Line 1 server retains 64 sequences and 1024 batched tokens per attention-DP
rank (512 sequence slots across eight ranks on four physical chips). The server
command comes from the frozen benchmarking script. Client concurrency does not
rewrite server settings. The wrapper does not append server flags or change its literal model reference.
The image builder checks that both source checkouts are on
`topk` and records their exact commits.

The job mounts disk-backed scratch at `/run-scratch` with a 600 GiB volume
size ceiling. It makes no explicit disk reservation, matching the earlier
successful jobs' scheduling policy. All Hugging Face caches are there.
The checkpoint preflight checks actual free space for checkpoint bytes plus
64 GiB headroom before downloading with two workers. It verifies selected files
and records the prefetched revision. The server retains Line 1's literal model
name and default revision selection. Scratch is excluded from results and lasts
for the Pod lifetime. The assigned node must still have enough free disk space.

Client/parser/tokenizer and per-configuration TPU checks run before model
serving. They do not establish full-model compilation or performance. Server
startup failures print the server-log tail into the controller output. Every
case saves commands, server/client logs, errors and JSON metrics; summary CSV
and JSON include I/O shape, client concurrency, throughput and latency.

After completion, the controller downloads and verifies the compressed result
bundle and deletes the run-owned image and GCS prefix. A failed aggregate job
can contain successful benchmark cases. Missing/unverified final results retain
resources for recovery. Ctrl-C detaches the controller and does not cancel or
clean the remote job; resume the saved campaign to continue collection.
Raw local resume files remain ignored. Git-visible archives are under
`results/archives/`, and compressed launcher logs under `tmp/workflow/local/logs/`.

The [results summary](summary/RESULTS.md), [analysis](summary/ANALYSIS.md), and
[command appendix](summary/COMMANDS.md) collect the historical comparisons and
compressed job results. Refresh the generated summary after pulling new results:

```bash
python3 tmp/serving-comparison/summarize.py
```

Source commits and result-import commits are separate columns. Historical
client logs record parsed arguments, not always the original script path; the
appendix labels that limitation. Incomplete runs remain visible and are not
used as completed performance controls. The generated JSON retains full hashes,
artifact paths and per-run observations.

Local checks (use an interpreter with the workflow YAML dependency installed):

```bash
python3 -m unittest discover -s tmp/serving-comparison -p 'test_*.py' -v
python3 -m unittest discover -s tmp/workflow/tests -v
```
