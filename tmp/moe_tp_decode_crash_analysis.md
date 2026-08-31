# Root Cause Analysis: TPU Hardware Halt in MoE TP Decode Kernel (`USE_MOE_TP_DECODE_KERNEL`)

## Issue Description
When attempting to run `vllm serve Qwen/Qwen3.5-397B-A17B-FP8` with `USE_MOE_TP_DECODE_KERNEL=1` on `tpu7x-8` (TPU v6e) hardware, sending client requests to the server crashes both the server and client.

The server's log reports a TPU hardware halt (`RuntimeUnexpectedCoreHalt` UserFatal error) on Sparse Core / Tensor Core:
```text
(EngineCore pid=933905) ERROR 08-31 14:29:35 [core.py:1233] jax.errors.JaxRuntimeError: INTERNAL: E0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid: (launch_id=644757329)Assertion args: INTERNAL: Fatal error occurred during TPU execution. This indicates an infrastructure failure (hardware/network/power), a compiler bug, or a Pallas/Mosaic kernel bug. Please read https://openxla.org/xla/errors/error_0200 carefully for further details on how to triage and resolve this issue.

Detailed error: Node 1 halted unexpectedly at tag:pc SparseCoreSequencer:0:0x3 (from SparseCoreSequencer:0:0x3): no debugging message found for this tag:pc. ; HLO module: jit_step_fun_impl
```

## Root Cause Analysis
The crash was introduced by a commit on **August 30, 2026** in the MoE TP decode kernel (`tpu_inference/kernels/fused_moe/v2/decode_kernel.py`):
```
commit d094fba5a9849cdb6775d5e49f92f0b8bd030243
Author: yarongmu-google <ymu@google.com>
Date:   Sun Aug 30 20:01:07 2026 -0700

    fused_moe/v2: lane-replication on the MXU - the park's jnp.repeat lowered as ~16k lane-shuffle ops on the saturated VALU/XLU classes (run-5 dump); one iota-built 0/1 block-replication matmul does the gather on the near-empty VEX/VRES classes; bf16 bitwise-identical
```

### The Mechanism of the Crash:
1. To replace a slow `jnp.repeat` call (which previously lowered to over 16,000 slow lane-shuffling instructions on saturated hardware execution units), a custom matrix multiplication (`jnp.dot`) was introduced inside the Pallas TPU kernel to implement lane-replication:
   ```python
   rep = jnp.dot(stack_t, rep_mat, preferred_element_type=jnp.float32)
   ```
2. The `stack_t` matrix has shape `(num_tokens, 3 * be)`. For Qwen3.5-397B, `be` (experts per block) is `8`, making the inner contraction dimension `3 * be = 24`.
3. The TPU's Matrix Multiply Unit (MXU) hardware has very strict alignment and size constraints for general matrix multiplications inside a low-level custom Pallas TPU kernel (expecting multiples of 128 or 128-aligned tiles). 
4. Attempting to execute `jnp.dot` with an unaligned inner contraction dimension of `24` directly violates these MXU hardware alignment expectations. This results in the generation of invalid low-level execution instructions, triggering a TPU hardware halt (`RuntimeUnexpectedCoreHalt` UserFatal error), which terminates the entire TPU runtime process and crashes the server.

## Remediation / Mitigation
The production-stable fallback path is the standard TensorCore-based `fused_moe_func`, which bypasses this custom Pallas decode kernel. 

To run safely without crashing, launch the vLLM server with the experimental TP decode kernel disabled:
```bash
USE_MOE_TP_DECODE_KERNEL=0
```
This forces MoE decode to run on the standard GMM/TensorCore execution pipeline, preventing any TPU hardware halts and resolving the issue.
