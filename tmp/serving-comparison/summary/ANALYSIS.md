# Historical conclusions checked against the new run

The original HTML was not found in the available Git branches/history or local
HTML search. This audit checks the committed results ledger and original logs;
it does not claim to have verified every sentence of the unavailable HTML.

## Findings

- The eviction case was 4c, with 128 sequence slots per attention-DP rank and
  client C1024. Its 20260903_103945 server log contains 3,786 `REFUSED gate=`
  entries and requests in PREEMPTED state. It was not the 4g configuration.
- Historical 4g has 64 slots/rank and C512. Historical 4i has 104 slots/rank and
  C832. They select the same custom MoE TP decode kernel but have different
  sequence limits and attention request-bucket ladders. 4h also uses 104 slots,
  but its token budget is 256/rank; 4i reduces this to 128/rank.
- Both new clients were C512. 4g still has 512 slots across the eight ranks;
  4i still has 832 slots. The client cap does not resize server state pools.
  Eight ranks are eight TPU cores across four physical chips, not eight chips.
- The new 4g log allocates 146.10 GiB to attention pages and 93.01 GiB to
  sequence state. 4i allocates 88.86 GiB and 150.25 GiB respectively. These
  match the corresponding historical server carve messages.
- New sampled peak KV usage is 40.48% (4g) and 67.28% (4i). Neither log contains
  refusals or preemption text. Both contain admission-debug messages and custom
  decode-kernel engagement. This supports the absence of logged memory churn;
  it does not prove that all instantaneous memory behavior was sampled.
- Historical 4g with the upstream sampler: 8,611.98 output tok/s, 52.03 ms TPOT.
  New 4g: 8,578.08 output tok/s, 52.00 ms TPOT, only 0.39% lower throughput.
  Both processed exactly 1,887,965 input and 15,068,204 output tokens in 2,048
  requests at C512. The new result closely reproduces that earlier operating
  point, although software revisions differ.
- Historical 4i's 8,603.99 output tok/s used C832 and the repository sampler:
  1,249,141 input and 9,943,674 output tokens. Its new 7,749.33 result used C512
  and the longer upstream sample. These are not the same workload. The old
  result does not predict which configuration wins under the new workload.
- The new ranking does not falsify the memory-carve, bucket-shape, or kernel
  routing findings. It does invalidate treating "4i is best" as universal.
  The specific source of its slower C512 performance is not established by
  these logs alone; attributing it to eviction would contradict the evidence.
- One historical ledger caveat needs correction: the 8,066.78 baseline result
  completed only 1,772/2,048 requests. The complete 8,125.10 baseline result
  used client C1024, even though its server had 512 sequence slots. Neither is
  a complete, same-client-C512 control for the new result. The summary labels
  completion count, client concurrency and sampler separately.

## Next baseline jobs

The `topk` job reruns Line 1 only: 1024/8192, client C512, 2,048 requests,
seed 0, upstream ratio 0.8, zero extra warmups, ignore EOS, no chat template.
The client is freshly fetched at `ee867231de0b268e2810a6e31751b23cf5903fc5`,
the same revision as the last run. Server capacity stays 64 sequences/rank
and 1024 tokens/rank. This completes the missing control for the saved 4g/4i
measurements; it does not isolate the kernel or locate a throughput ceiling.

A separate `bench` branch uses the new `infx` commands at 8192/1024 and
1024/1024, C256, 2,560 requests, 512 warmups, ratio 1.0 and chat templates,
with vLLM based on `bec0a4` and TPU inference based on `4c0dc1`.
Its results are separate operating points, not direct controls for the C512 run.
There are no new measurements yet.

## Workflow repair and limits

The new baseline failed downloading the approximately 378 GiB checkpoint into
its container root filesystem with `No space left on device (os error 28)`.
The old mount error belongs to the previous job. The new job completed both
custom-kernel benchmarks, verified its result bundle and deleted its owned
image and GCS prefix. Earlier failed jobs' unverified resources were retained.

The next description requests 600 GiB of local ephemeral storage, mounts a
bounded disk-backed scratch directory, puts Hugging Face caches there, checks
checkpoint metadata plus 64 GiB free-space headroom, and downloads with two
workers. It pins the model revision and verifies every selected file's size.
This is a capacity/size check, not an independent cryptographic model audit.
Scratch is excluded from result archives and lasts for the Pod lifetime.

An emptyDir does not create a new disk. The request requires sufficient node
allocatable storage; the job can remain Pending if the pool cannot provide it.
Actual pool capacity and this rendered resource request need live validation.
The controller validates that the mount, volume and requested capacity survive
CDK recipe rendering. See the [Kubernetes storage documentation](https://kubernetes.io/docs/concepts/storage/ephemeral-storage/).

Failed server startup now prints its server-log tail immediately. Each case's
logs and results remain separate. Any failed case makes the aggregate job fail,
while completed measurements remain valid and are included in the summary.
