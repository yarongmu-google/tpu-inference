# Qwen3.5-397B-A17B-FP8 serving campaign - results ledger

Model: Qwen/Qwen3.5-397B-A17B-FP8 (hybrid: 15 full-attention + 45
linear-attention layers) on 8 accelerators. All configs are lines in
bench_throughput_qwen_server.sh (labels match); all numbers are from
the closed-loop client with the FIXED length sampler (lengths uniform
on [0.2x, x]; in=1024 out=8192 nominal -> means ~614/~4915), 2048
requests, unless marked "old client". Never compare across client
versions - the old sampler ([0.8x, x], means ~922/~7373) reads ~15-20%
higher on the same server.

Logs: tmp/vllm_logs/<label>_<timestamp>.log(.xz) + client logs +
per-run stats CSV snapshots. Admission instrumentation:
VLLM_ADMISSION_DEBUG=1 (refusal/preemption tracer in the scheduler).

## Scoreboard (realistic mix workload)

| line | config summary | width (global) | mean TPOT | out tok/s | date | notes |
|---|---|---|---|---|---|---|
| 3 gmm_tp | stock GMM, TP sharding, MNS=64/rank | 512 | 94.7 | 4763 | 09-02 | the kernel-less baseline of OUR deployment |
| 4 v2_tp | + v2 decode kernel, gate 512 | 512 | 59.9 | 7385 | 09-01 | kernel +55% over line 3; two-speed steps (rider steps fell back to GMM) |
| 4g riders | 4 + MNB=128/rank, gate 1024 | 512 | 54.65 | 7952 | 09-03 | single-speed: EVERY step on the kernel; 0 refusals certified |
| 4c sweep "winner" | MNS=128/rank f32 | 1024 cfg | 118.2-118.7 | 6333-6639 (5 runs) | 09-02/03 | THE VALLEY: inconsistent carve -> 3786 refusals, 547 preemptions, ~1.02M recomputed tokens |
| 4e fixed point | MNS=104/rank f32 (carve-consistent) | 832 | 116.2 | 6157 | 09-03 | valley eliminated (KV 53%, wait ~5) but step time exposed |
| 4h p2buckets | 4e + power-of-2 req buckets | 832 | 92.7 | 7544 | 09-03 | bucket cliff confirmed: -24ms/step from ladder shape alone; 0 refusals (tracer-certified) |
| 4i singlespeed | 4h + MNB=128/rank | 832 | 78.35 | **8604** | 09-03 | BEATS the default (+6.6%) at pure f32, no caveats; 0 refusals, 0 fallbacks; inside the fixed point's originally predicted 8.5-9k |
| 4d bf16 state | MNS=128/rank + bf16 SSM state | 1024 | 94.8 | 8535 | 09-02 | predates the routing/bucket fixes; ACCURACY EVAL OWED; a 4d+fixes config is the obvious next ceiling probe |
| 1 default | stock EP deployment, MNS=64/rank | 512 | 53.6-56.0 | 8067-8125 | 09-02/03 | the target; 8067 run tracer-certified 0 refusals, prefill riding continuously |

Decode-only (in=1): ours 8687 vs default 8666 vs stock GMM-TP 4970 -
the kernel ties the default at the weight-stream floor and is +75%
over the same deployment without it.

Old-client cross-check (do not mix with the table above): 4g measured
8612 out / 9691 total on the old sampler; historical default folklore
"~10500 total" is from that client generation.

## The campaign in one paragraph

The kernel closed its gap on day one at 512-wide (decode tie with the
default; +55-75% over its own deployment without it) and every gain
since has come from the serving layer with the kernel unchanged:
(1) the memory carve - the hybrid model's per-seq state pool is carved
by max-num-seqs while attention pages get the remainder; sizing them
inconsistently (MNS=128/rank) starves admission into a preemption-churn
valley; the consistent carve solves MNS*(slot + avg_ctx*page*h) =
budget -> 104/rank f32, 151/rank bf16 (estimate_service_knobs.py
hybrid mode); (2) routing - the kernel engagement gate was set when
only T<=512 was tuned; raising it + capping MNB so every step runs the
kernel recovered 8% (7385->7952); (3) bucket shapes - non-power-of-2
request-bucket ladders cost ~24ms/step at 832-wide via per-seq kernel
slow paths (116->92.7ms); the runner now sorts ladders after the
max_num_seqs append (a previously fatal config trap). Remaining at
832-wide f32: ~23ms/step residue vs the component model (~70ms) -
single-speed steps (4i) then removed the two-speed mixing tax
entirely: 78.35ms at 832-wide -> 8604 out tok/s, ABOVE the default
deployment at pure f32. Final residue vs the component model: ~8ms;
step-profile still worthwhile but no longer load-bearing.

## Reproduction

- Server lines + client lines: bench_throughput_qwen_server.sh /
  bench_throughput_qwen_client.sh (CONC = the width being tested).
- Knob anchors: estimate_service_knobs.py (hybrid fixed-point mode
  auto-detects layer_types; derived slot bytes verified byte-exact
  against the server's carve log line).
- Step profile capture: tmp/capture_step_profile.sh (short xprof
  window at the serving plateau).
