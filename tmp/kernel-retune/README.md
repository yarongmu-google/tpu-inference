# TP MoE kernel retuning

On the CPU VM, pull `topk` and activate the existing Python 3.12 `vllm12`
environment whose editable vLLM checkout is also on `topk`. Run:

```bash
bash tmp/kernel-retune/run.sh --name tp-moe-retune
```

The name is the only required input. This uses the existing environment
profile and branch-checking image builder: build the committed source,
publish the run-owned image, submit one TPU job, monitor, collect verified
archives, then clean up the owned image and run-storage subfolder. The
workload does not download model weights or reserve checkpoint scratch.
The shared runtime builder still prepares its usual dependencies.

The job creates synthetic tensors on TPU, with the FP8 serving layout:
D=4096, E=512, I=1024, top-k=10, eight TP ranks, BF16 input activations and
per-channel FP8 weights. Per-token activation quantization stays fixed.
This isolates the new kernel; it is not a serving or model-accuracy test.

## Matrix and correctness

`payload/plan.json` controls the run. Defaults:

- Global tokens: 512, 1024, 2048, 8192.
- Global token tile: 256, 512, 1024.
- FP8/BF16 compute row pairs: 32/16 and 64/32.
- Five warmups and 30 synchronized timing samples per candidate.
- 900 seconds per candidate, including initialization and compilation.

Equivalent effective token tiles are deduplicated, leaving 22 candidates.
Each runs in a fresh process so compile failures or timeouts cannot silently
supply a winner. The next candidate is attempted after a failure. Any failed
candidate makes the overall workload return nonzero, while retaining all
successful results and provisional winners.

Before timing, each candidate runs uniform random routing and a skewed case
where every token selects the same experts. Outputs must be finite everywhere.
An independent XLA gather/einsum reference checks evenly spaced sample rows
plus each rank's first and last row. It preserves per-rank intermediate
rounding and reduction. Tolerances and numerical errors are saved. This is
sampled kernel correctness, not exhaustive model validation. No failed or
unchecked candidate can enter the winner table.

Weights are generated as device shards and passed as executable arguments;
no full host weight tensor or per-call weight-layout conversion is introduced
by the harness. Each candidate uses the same seed. The compiled HLO is saved
as gzip for inspection of any backend-inserted copies; compiled memory
statistics are also retained, without claiming they certify VMEM occupancy.

Timing includes host dispatch, VMEM collectives, kernel execution and final
reduce-scatter. Compilation, tensor generation, correctness checks and warmup
are excluded. `winners.json` uses median synchronized wall latency and labels
winners provisional. It does not report device-only kernel time or tokens/s.
No serving defaults are changed automatically. Device profiling and a repeat
measurement of finalists should precede adopting close timing differences.

## Outputs and recovery

The live workload produces:

- `SUMMARY.md`, `results.json`, `winners.json`, and the frozen plan/matrix.
- Per candidate: exact command/config, phase logs, source revisions/hash,
  package versions, correctness results, raw timing samples and exit status.
- Compile diagnostics when compilation succeeds; traceback/logs on failure.

The shared controller packages these with build/submission diagnostics under
`tmp/kernel-retune/results/archives/`. Raw workflow state remains ignored;
archives and their JSON indexes are visible to Git. Launcher transcripts
remain under the shared `tmp/workflow/local/logs/` location.

If monitoring or collection was interrupted:

```bash
bash tmp/kernel-retune/run.sh --recover
```

Recovery uses existing job IDs and does not rebuild or submit a duplicate.
Cleanup occurs only after verified local collection. An incomplete collection
retains its resources for recovery. Start another named run for a fresh tune.

For local plan, subprocess failure/timeout and wrapper checks:

```bash
python3 -m unittest discover -s tmp/kernel-retune -p test_retune.py -v
```

These tests require no cloud access. Actual TPU backend compilation,
execution, memory fit and timing are established by the submitted job.
