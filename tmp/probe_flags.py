"""Which dump/debug flags does this build actually accept?

The Mosaic pass chain stops at finalize-llo: no register allocation,
scheduling or bundling is visible, so we cannot see the real issue order.
This tries candidate flag names (XLA_FLAGS and LIBTPU_INIT_ARGS) and
reports which are accepted, so we can dump further down the pipeline.

    python tmp/probe_flags.py
"""

import os
import subprocess
import sys

CANDIDATES = [
    "--xla_tpu_dump_llo_to=/tmp/flagprobe",
    "--xla_tpu_dump_llo=true",
    "--xla_tpu_llo_dump_to=/tmp/flagprobe",
    "--xla_tpu_enable_log_recorder=true",
    "--xla_tpu_dump_post_scheduling=true",
    "--xla_tpu_dump_bundles_to=/tmp/flagprobe",
    "--xla_tpu_dump_asm_to=/tmp/flagprobe",
    "--xla_dump_disable_metadata=false",
    "--xla_tpu_enable_large_2nd_minor_layout_for_x16=true",
    "--xla_tpu_scoped_vmem_limit_kib=65536",
    "--xla_tpu_use_repeated_instance_for_preferred_prefetch_time=true",
]

PROBE = "import jax, jax.numpy as jnp; jax.block_until_ready(jnp.zeros(8) + 1)"


def try_flag(flag: str, var: str) -> bool:
    env = dict(os.environ)
    env[var] = flag
    env.pop("JAX_PLATFORMS", None)
    r = subprocess.run([sys.executable, "-c", PROBE], env=env,
                       capture_output=True, text=True, timeout=300)
    return r.returncode == 0


def main() -> None:
    for var in ("XLA_FLAGS", "LIBTPU_INIT_ARGS"):
        print(f"\n=== {var}")
        for flag in CANDIDATES:
            ok = try_flag(flag, var)
            print(f"  {'OK    ' if ok else 'reject'}  {flag}")


if __name__ == "__main__":
    main()
