import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
import time
import itertools
from tpu_inference.kernels.experimental.e2e_qwen3.kernel import get_tp_moe_kernel

def run_tuning_sweep():
    print("Starting TP-MoE Parameter Sweep...")
    
    # Static Model Dimensions
    B = 64
    H = 4096
    E = 512
    I_slice = 128
    
    # Dummy Data
    x = jax.device_put(jnp.ones((B, H), dtype=jnp.bfloat16))
    router_logits = jax.device_put(jnp.ones((B, E), dtype=jnp.bfloat16))
    w1_gate = jax.device_put(jnp.ones((E, H, I_slice), dtype=jnp.bfloat16))
    w1_up = jax.device_put(jnp.ones((E, H, I_slice), dtype=jnp.bfloat16))
    w2_down = jax.device_put(jnp.ones((E, I_slice, H), dtype=jnp.bfloat16))
    
    # Tuning Grid
    block_b_options = [32, 64]
    num_buffers_options = [2, 3]
    
    best_time = float('inf')
    best_params = None
    
    results = []
    
    for block_b, num_buffers in itertools.product(block_b_options, num_buffers_options):
        print(f"\n--- Testing BLOCK_B={block_b}, BUFFERS={num_buffers} ---")
        
        # We need to redefine the apply wrapper here to pass the static tuning params
        def apply_tp_moe_tuned(x, router_logits, w1_gate, w1_up, w2_down):
            kernel = get_tp_moe_kernel(E, H, I_slice, block_b, num_buffers)
            
            out_shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
            
            # Update scratch sizes based on blocks and buffers
            scratch_w1 = pltpu.VMEM((num_buffers, H, I_slice), x.dtype)
            scratch_w2 = pltpu.VMEM((num_buffers, I_slice, H), x.dtype)
            scratch_x = pltpu.VMEM((block_b, H), x.dtype)
            scratch_logits = pltpu.VMEM((block_b, E), x.dtype)
            scratch_y = pltpu.VMEM((block_b, H), x.dtype)
            scratch_sem = pltpu.SemaphoreType.DMA((num_buffers, 3))
            
            return pl.pallas_call(
                kernel,
                out_shape=out_shape,
                scratch_shapes=(scratch_w1, scratch_w1, scratch_w2, scratch_x, scratch_logits, scratch_y, scratch_sem),
                in_specs=[
                    pl.BlockSpec(memory_space=pltpu.HBM, block_shape=(block_b, H)), # x
                    pl.BlockSpec(memory_space=pltpu.HBM, block_shape=(block_b, E)), # router
                    pl.BlockSpec(memory_space=pltpu.HBM), # gate
                    pl.BlockSpec(memory_space=pltpu.HBM), # up
                    pl.BlockSpec(memory_space=pltpu.HBM), # down
                ],
                out_specs=pl.BlockSpec(memory_space=pltpu.HBM, block_shape=(block_b, H)),
                grid=(B // block_b,), # Launch multiple PPU blocks if BLOCK_B < B
                compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024)
            )(x, router_logits, w1_gate, w1_up, w2_down)
        
        try:
            jitted_apply = jax.jit(apply_tp_moe_tuned)
            
            # Warmup
            y_out = jitted_apply(x, router_logits, w1_gate, w1_up, w2_down)
            jax.block_until_ready(y_out)
            
            # Timing
            num_iters = 10
            start = time.time()
            for _ in range(num_iters):
                y_out = jitted_apply(x, router_logits, w1_gate, w1_up, w2_down)
            jax.block_until_ready(y_out)
            end = time.time()
            
            avg_time_ms = ((end - start) / num_iters) * 1000
            print(f"Success! Time: {avg_time_ms:.3f} ms")
            
            results.append({"b": block_b, "buf": num_buffers, "time": avg_time_ms})
            
            if avg_time_ms < best_time:
                best_time = avg_time_ms
                best_params = (block_b, num_buffers)
                
        except Exception as e:
            print(f"Failed to compile/run: {e}")

    print("\n=== Tuning Results ===")
    for r in results:
        print(f"BLOCK_B: {r['b']:<3} | BUFFERS: {r['buf']} -> {r['time']:.3f} ms")
    
    if best_params:
        print(f"\nBEST PARAMS: BLOCK_B={best_params[0]}, BUFFERS={best_params[1]} with {best_time:.3f} ms")

if __name__ == '__main__':
    run_tuning_sweep()
