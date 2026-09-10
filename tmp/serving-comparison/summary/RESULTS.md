# Serving comparison results

Source commit identifies runtime code; result commit identifies the latest Git commit that added or updated the result artifact.
Historical source commits come from server log markers and do not certify a clean historical working tree.
Partial runs are retained for diagnosis and excluded from performance claims. Missing metrics mean no completed benchmark.
Eight attention-DP ranks run on eight TPU cores across four physical chips. Throughput below is per full TPU slice.

| Run | I/O | Client C | Sampler | Completed | Output tok/s | Total tok/s | TPOT ms | Source commit | Result commit |
|---|---|---:|---|---|---:|---:|---:|---|---|
| baseline-20260902_163515 | 1k/8k | 1024 | [0.2X,X] | 2048/2048 (complete) | 8125.10 | 9145.79 | 56.02 | f03e266b11 | cd78a3f623 |
| baseline-20260903_135542 | 1k/8k | 512 | [0.2X,X] | 1772/2048 (partial) | 8066.78 | 9129.07 | 53.57 | fc21ca4846 | 077dbd26a1 |
| 4c-20260903_105407 | 1k/8k | 1024 | [0.2X,X] | 2048/2048 (complete) | 6629.34 | 7462.13 | 118.69 | 8fffecb5ea | 24a83f74cf |
| 4e-20260903_122921 | 1k/8k | 832 | [0.2X,X] | 2048/2048 (complete) | 6156.61 | 6930.01 | 116.24 | f9d0e6b196 | fc21ca4846 |
| 4g-20260903_151322 | 1k/8k | 512 | [0.2X,X] | 2048/2048 (complete) | 7952.14 | 8951.11 | 54.65 | 077dbd26a1 | 8ca04142c0 |
| 4g-20260903_155530 | 1k/8k | 512 | [0.8X,X] | 2048/2048 (complete) | 8611.98 | 9691.02 | 52.03 | 077dbd26a1 | de204fbab6 |
| 4h-20260903_165502 | 1k/8k | 832 | [0.2X,X] | 490/2048 (partial) | 830.76 | 6633.26 | 147.21 | a086d4c940 | 5bddd9c3dc |
| 4h-20260903_181805 | 1k/8k | 832 | [0.2X,X] | 2048/2048 (complete) | 7543.71 | 8491.36 | 92.70 | bdb438a080 | 602d3dbb46 |
| 4i-20260903_191522 | 1k/8k | 832 | [0.2X,X] | 2048/2048 (complete) | 8603.99 | 9684.84 | 78.35 | a9b418cb63 | 847345e140 |
| serving-comp-default-5552a926f686456c9cc955f4 | - | - | - | failed-before-results | - | - | - | - | 8e53153327 |
| serving-comp-default-ad079a900cb14f3bbaaf5435 | - | - | - | failed-before-results | - | - | - | - | 9a8746915f |
| serving-comp-default-b7d2d6774029493e962cfee6/baseline | 1k/8k | 512 | [0.8X,X] | 0/2048 (failed) | - | - | - | e787ccfff4 | 39da1014ee |
| serving-comp-default-b7d2d6774029493e962cfee6/4g | 1k/8k | 512 | [0.8X,X] | 2048/2048 (complete) | 8578.08 | 9652.86 | 52.00 | e787ccfff4 | 39da1014ee |
| serving-comp-default-b7d2d6774029493e962cfee6/4i | 1k/8k | 512 | [0.8X,X] | 2048/2048 (complete) | 7749.33 | 8720.27 | 61.00 | e787ccfff4 | 39da1014ee |

Full provenance, log locations, client/vLLM revisions and memory observations: [results.json](results.json).
Recorded server commands and client arguments: [COMMANDS.md](COMMANDS.md).
Interpretation and next experiment: [ANALYSIS.md](ANALYSIS.md).
