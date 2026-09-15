# Occupancy kernel comparison and tuning

After committing the reviewed sources and pulling them on the CPU VM:

```bash
bash tmp/kernel-occupancy/run.sh --name moe-occupancy
```

Use the existing CPU-VM environment with editable topk vLLM. The existing
workflow profile supplies the project, image registry and TPU settings; the
run name is the only required input. This builds the committed image, submits
one TPU job, polls it, collects verified artifacts, and cleans up its owned
image package and run storage. It uses synthetic tensors; no model download.

```bash
bash tmp/kernel-occupancy/run.sh --name check --dry-run
bash tmp/kernel-occupancy/run.sh --recover
```

The logged controller survives terminal detachment. Ctrl-C detaches its viewer;
it does not cancel the TPU job. Collection and candidate compaction continue
in that background controller. Recovery uses the generic finalization path.

## First campaign

The plan starts with global T=512, D=4096, E=512, I=1024, top-k=10 and eight
TP ranks. It generates weights once, then activations and routing inputs once
per token count. The plan explicitly supports additional aligned token counts,
but the initial run isolates the occupancy change at the measured decode shape.

There are 12 configurations, each tested as a matched original/occupied pair:
24 candidate executions plus one untuned full-output XLA reference executable.
The first configuration is the measured center: be=8, bg=2, capacity=32,
bd1c=256, bd2c=128, bcT=0. Other configurations change one knob at a time:

| Knob | Values |
| --- | --- |
| be | 2, 4, 8 |
| bg | 1, 2, 4 |
| capacity | 32, 64 |
| bd1c | 128, 256, 512 |
| bd2c | 128, 256, 512 |
| bcT | 0, 128, 256 |

This bounded first sweep reuses the old tuner's knobs and measured center. It
is not its exhaustive or adaptive search, and does not establish a global
optimum. The original is an exact source copy from 729a68e09f; the candidate
imports decode_kernel_occupied.py from the built image. Neither serving
integration nor the original kernel in that image is modified by this job.

Both paths receive the same precomputed logits, FP8 E4M3 weights and per-channel
FP32 scales. GMM1 activations are quantized to FP8 in the kernel; GMM2 uses BF16
activations. This is not FP4. Because the historical profiler run fused the
router projection, absolute historical numbers are context, not a measurement
of this exact boundary. The new old/new comparison uses identical boundaries.

Uniform routing uses seeded random logits. Sparse routing bounds expert loads
at roughly 32 for the default shape, with many empty experts. Capacity remains
fixed and overflow behavior is unchanged; this job tests occupancy skipping,
not the separate no-drop change. Larger token counts may need larger capacity.

## Measurements and checks

Every executable runs both routing cases, then five warmups and 30 synchronized
samples per case. All raw samples, median/minimum wall time, compile duration,
source hashes, versions, and configurations are saved. Compilation and host
output conversion are outside the timed calls; collectives are inside them.
XLA outputs and timings are reused across all candidate pairs at that shape.
The XLA program is a direct numerical reference, not the production GMM backend.

Accuracy compares all output elements with TPU XLA using rtol=0.04/atol=0.01,
reports nonfinite values, and also compares each modified result against its
matched original (including a bitwise-equality flag). Outputs are retained in
16 MiB row chunks, so later debugging does not require a new TPU execution.
Failed accuracy retains warmed timings and prevents selection as a winner.
Compile failures retain their errors and the session tries the next candidate.

Three additional dispatches per routing case are captured with the historical
profiler settings. The report keeps synchronized wall latency separate from
TensorCore device latency and SparseCore coverage. The reused TC extractor
excludes barrier/trailing-copy edges; it is not complete end-to-end device time.
Missing/failed trace extraction is explicit and prevents winner selection.
Samples are flushed before profiling, so a profiler crash still retains timing.

## Pipeline and artifacts

One persistent process owns TPU execution. It retains weights/inputs, compiles
one candidate, executes and times it, requests host copies, and continues with
the next while CPU threads compare outputs, parse traces and save chunks. At
most two pending candidates bound host work. Compilation, device timing and
trace capture are serialized on the TPU owner; there are no concurrent timed
kernels. Profiler stop/export is a synchronous API boundary, not an asynchronous
artifact-upload operation.

Compiler dumps move into the case directory by directory rename. A separate
CPU packer hashes and gzip-compresses completed directories while TPU work
continues. Oversized individual files are fragmented into 32 MiB chunks in this
background stage; file-fragments.json records original names, ordering, sizes
and checksums. Parts stay below the shared 50 MiB compressed limit. Sealed
compiler/profile/output duplicates are removed from remote scratch; failed
packaging retains the originals for recovery.

The runtime publishes in another thread. The CPU VM collects each completed
case while later TPU work runs, verifies parts and members, and retains small
readable metadata under summary/. It removes duplicate unpacked bulk only after
verification, retaining the compressed parts. This compaction happens inside
the persistent collector, including after the terminal detaches.

Results live under results/candidates/<run-id>/<case>/:

- candidate.json: checksummed inventory and the case's full outcome metadata.
- summary/: small outcome, accuracy, timing, profile and provenance reports.
- part-*.tar.gz: compiler dumps, traces and output chunks.

The final diagnostics archive lives under results/archives/ and contains the
aggregate SUMMARY.md/results.json plus run logs, without embedding another copy
of all the candidate parts. Keep both the archive and candidate directories.
For a fragmented file, extract the chunks listed in summary/file-fragments.json,
concatenate them in listed order, and verify the recorded full-file checksum.

TPU_STAGE, RUNNING, ARTIFACTS_READY, CANDIDATE_AVAILABLE and FINAL_UPLOAD_WAIT
show progress. A fatal process error or stage timeout ends that session; no
successful work is replayed. Partial artifacts are retained. Cloud cleanup
requires verified collection, for failed jobs as well as successful ones. The
pod still remains allocated for any final packaging/publication tail.

## Local validation

The job tests cover paired planning, failed-accuracy timing retention,
background packaging, oversized files, collection/compaction, failure recovery,
and baseline/input reuse. CPU interpreter tests execute the actual original
and modified kernels with two virtual devices; they are not TPU measurements.

```bash
python3 -m unittest discover -s tmp/kernel-occupancy -p test_job.py
python3 -m unittest discover -s tmp/kernel-occupancy -p test_session.py
KERNEL_SOURCE_ROOT="$PWD" XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  python3 -m unittest discover -s tmp/kernel-occupancy -p test_engine.py
```
