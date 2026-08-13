import functools
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

cdiv = pl.cdiv

def get_tp_moe_kernel(
    num_experts: int,
    hidden_size: int,
    intermediate_size_per_chip: int,
    block_b: int,
    num_buffers: int,
):
    """Generates the parameterized TP-MoE Pallas kernel."""

    def tp_moe_kernel(
        x_ref,               
        router_logits_ref,   
        w1_gate_ref,         
        w1_up_ref,           
        w2_down_ref,         
        y_out_ref,           
        w1_gate_vmem,
        w1_up_vmem,
        w2_down_vmem,
        x_vmem,
        logits_vmem,
        y_vmem,
        local_sems,          
    ):
        def _async_copy(src, dst, sem, wait=False):
            cp = pltpu.make_async_copy(src, dst, sem)
            if wait:
                cp.wait()
            else:
                cp.start()
        
        # 1. DMA X and Router Logits from HBM to VMEM
        _async_copy(x_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    local_sems.at[0, 0], wait=False)
        _async_copy(router_logits_ref.at[pl.ds(0, block_b), pl.ds(0, num_experts)], 
                    logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], 
                    local_sems.at[0, 1], wait=False)
        
        # Halt execution until the transfers finish
        _async_copy(x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    local_sems.at[0, 0], wait=True)
        _async_copy(logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], 
                    logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], 
                    local_sems.at[0, 1], wait=True)

        x = x_vmem[...]
        logits = logits_vmem[...]
        y_accum = jnp.zeros_like(x, dtype=jnp.float32)

        def fetch_weights(e_id, buf_id):
            _async_copy(w1_gate_ref.at[e_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                        w1_gate_vmem.at[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                        local_sems.at[buf_id, 0], wait=False)
            _async_copy(w1_up_ref.at[e_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                        w1_up_vmem.at[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                        local_sems.at[buf_id, 1], wait=False)
            _async_copy(w2_down_ref.at[e_id, pl.ds(0, intermediate_size_per_chip), pl.ds(0, hidden_size)],
                        w2_down_vmem.at[buf_id, pl.ds(0, intermediate_size_per_chip), pl.ds(0, hidden_size)],
                        local_sems.at[buf_id, 2], wait=False)

        def wait_weights(buf_id):
            _async_copy(w1_gate_vmem.at[buf_id], w1_gate_vmem.at[buf_id], local_sems.at[buf_id, 0], wait=True)
            _async_copy(w1_up_vmem.at[buf_id], w1_up_vmem.at[buf_id], local_sems.at[buf_id, 1], wait=True)
            _async_copy(w2_down_vmem.at[buf_id], w2_down_vmem.at[buf_id], local_sems.at[buf_id, 2], wait=True)

        fetch_weights(0, 0)
        
        def run_expert(e_idx, y_acc):
            buf_id = e_idx % num_buffers
            next_buf_id = (e_idx + 1) % num_buffers
            next_e_idx = e_idx + 1
            
            @pl.when(next_e_idx < num_experts)
            def _():
                fetch_weights(next_e_idx, next_buf_id)
                
            wait_weights(buf_id)
            
            block_idx = e_idx // 128
            inner_idx = e_idx % 128
            
            expert_logits_block = logits_vmem[pl.ds(0, block_b), pl.ds(block_idx * 128, 128)]
            one_hot_mask = jax.nn.one_hot(inner_idx, 128, dtype=jnp.bfloat16)
            
            expert_logits = jnp.sum(expert_logits_block * one_hot_mask, axis=1, keepdims=True)
            expert_mask = (expert_logits > 0.0).astype(jnp.bfloat16)
            
            x_act = x_vmem[pl.ds(0, block_b), pl.ds(0, hidden_size)]
            w1_g = w1_gate_vmem[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)]
            w1_u = w1_up_vmem[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)]
            w2_d = w2_down_vmem[buf_id, pl.ds(0, intermediate_size_per_chip), pl.ds(0, hidden_size)]
            
            h_gate = jnp.dot(x_act, w1_g, preferred_element_type=jnp.float32) * expert_mask
            h_up = jnp.dot(x_act, w1_u, preferred_element_type=jnp.float32) * expert_mask
            
            h_gate = h_gate.astype(jnp.bfloat16)
            h_up = h_up.astype(jnp.bfloat16)
            h_act = jax.nn.silu(h_gate) * h_up
            
            y_partial = jnp.dot(h_act, w2_d, preferred_element_type=jnp.float32) * expert_mask
            return y_acc + y_partial
            
        y_accum = lax.fori_loop(0, num_experts, run_expert, y_accum)
        
        y_vmem[pl.ds(0, block_b), pl.ds(0, hidden_size)] = y_accum.astype(x_ref.dtype)
        
        _async_copy(y_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    local_sems.at[0, 2], wait=False)
        _async_copy(y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], 
                    local_sems.at[0, 2], wait=True)

    return tp_moe_kernel

@functools.partial(jax.jit, static_argnames=["top_k"])
def apply_tp_moe(
    x,
    router_logits,
    w1_gate,
    w1_up,
    w2_down,
    top_k: int = 10,
):
    batch_size, hidden_size = x.shape
    num_experts = w1_gate.shape[0]
    intermediate_size_per_chip = w1_gate.shape[2]
    
    # We enforce BLOCK_B = 64 due to VMEM and alignment constraints
    BLOCK_B = 64
    NUM_BUFFERS = 3
    
    padded_batch = cdiv(batch_size, BLOCK_B) * BLOCK_B
    
    if padded_batch != batch_size:
        pad_len = padded_batch - batch_size
        x = jnp.pad(x, ((0, pad_len), (0, 0)))
        router_logits = jnp.pad(router_logits, ((0, pad_len), (0, 0)))
        
    grid = (padded_batch // BLOCK_B,)
    
    kernel = get_tp_moe_kernel(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size_per_chip=intermediate_size_per_chip,
        block_b=BLOCK_B,
        num_buffers=NUM_BUFFERS,
    )
    
    out_shape = jax.ShapeDtypeStruct((padded_batch, hidden_size), x.dtype)
    
    in_specs = [
        pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (i, 0), block_shape=(BLOCK_B, hidden_size)),
        pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (i, 0), block_shape=(BLOCK_B, num_experts)),
        pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (0, 0, 0), block_shape=(num_experts, hidden_size, intermediate_size_per_chip)),
        pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (0, 0, 0), block_shape=(num_experts, hidden_size, intermediate_size_per_chip)),
        pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (0, 0, 0), block_shape=(num_experts, intermediate_size_per_chip, hidden_size)),
    ]
    out_specs = pl.BlockSpec(memory_space=pltpu.ANY, index_map=lambda i: (i, 0), block_shape=(BLOCK_B, hidden_size))
    
    y = pl.pallas_call(
        kernel,
        out_shape=out_shape,
        in_specs=in_specs,
        out_specs=out_specs,
        grid=grid,
        compiler_params=dict(
            mosaic=dict(
                dimension_semantics=("parallel",),
            )
        )
    )(x, router_logits, w1_gate, w1_up, w2_down)
    
    if padded_batch != batch_size:
        y = y[:batch_size, :]
        
    return y
