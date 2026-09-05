#!/usr/bin/env bash
# KDA decode kernel v0: goldens on TPU + bandwidth bench vs the
# XLA reference and vs the pure state-stream bound. One paste; all
# output lands in tmp/ for commit.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=tmp/kda_decode_bench.log
: > "$LOG"; exec > >(tee -a "$LOG") 2>&1
export PYTHONPATH=. JAX_PLATFORMS=tpu

echo "+ goldens (compiled kernel on TPU)"
python -m pytest tests/kernels/kda_decode_kernel_test.py -q

echo "+ bench"
python - <<'PY'
import time
import jax, jax.numpy as jnp, numpy as np
from tpu_inference.kernels.kda.decode_kernel import kda_decode
from tpu_inference.kernels.kda.reference import kda_decode_step_reference

rng = np.random.default_rng(0)
H, K, V = 96, 128, 128
print(f"{'B':>5} {'kernel ms':>10} {'GB/s':>7} {'xla ms':>8} {'gain':>6}")
for B in (8, 32, 64):
    args = [jnp.asarray(rng.standard_normal((B, H, K)), jnp.bfloat16)
            for _ in range(2)]
    args += [jnp.asarray(rng.standard_normal((B, H, V)), jnp.bfloat16),
             jnp.asarray(rng.standard_normal((B, H, K)), jnp.bfloat16),
             jnp.asarray(rng.standard_normal((B, H)), jnp.bfloat16)]
    a_log = jnp.asarray(np.log(rng.uniform(1, 16, H)), jnp.float32)
    dtb = jnp.asarray(rng.standard_normal((H, K)) * .1, jnp.float32)

    def run(fn, S):
        S, o = fn(S, *args, a_log, dtb)
        jax.block_until_ready(S)
        t0 = time.perf_counter()
        for _ in range(20):
            S, o = fn(S, *args, a_log, dtb)
        jax.block_until_ready(S)
        return (time.perf_counter() - t0) / 20

    tk = run(lambda S, *a: kda_decode(S, *a),
             jnp.zeros((B, H, K, V), jnp.float32))
    tx = run(jax.jit(kda_decode_step_reference, donate_argnums=(0,)),
             jnp.zeros((B, H, K, V), jnp.float32))
    gb = 2 * B * H * K * V * 4 / 1e9
    print(f"{B:>5} {tk*1e3:10.3f} {gb/tk:7.0f} {tx*1e3:8.3f} {tx/tk:6.2f}x")
PY
echo "then: git add tmp/ && git commit -m 'kda decode bench.' && git push"
