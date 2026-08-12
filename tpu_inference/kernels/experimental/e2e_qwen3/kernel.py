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
        
        pltpu.make_async_copy(src_ref=x_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], dst_ref=x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], sem=local_sems.at[0, 0]).start()
        pltpu.make_async_copy(src_ref=router_logits_ref.at[pl.ds(0, block_b), pl.ds(0, num_experts)], dst_ref=logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], sem=local_sems.at[0, 1]).start()
        
        pltpu.make_async_copy(src_ref=x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], dst_ref=x_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], sem=local_sems.at[0, 0]).wait()
        pltpu.make_async_copy(src_ref=logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], dst_ref=logits_vmem.at[pl.ds(0, block_b), pl.ds(0, num_experts)], sem=local_sems.at[0, 1]).wait()

        x = x_vmem[...]
        logits = logits_vmem[...]
        y_accum = jnp.zeros_like(x, dtype=jnp.float32)

        def fetch_weights(e_id, buf_id):
            pltpu.make_async_copy(
                src_ref=w1_gate_ref.at[e_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                dst_ref=w1_gate_vmem.at[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                sem=local_sems.at[buf_id, 0]
            ).start()
            pltpu.make_async_copy(
                src_ref=w1_up_ref.at[e_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                dst_ref=w1_up_vmem.at[buf_id, pl.ds(0, hidden_size), pl.ds(0, intermediate_size_per_chip)],
                sem=local_sems.at[buf_id, 1]
            ).start()
            pltpu.make_async_copy(
                src_ref=w2_down_ref.at[e_id, pl.ds(0, intermediate_size_per_chip), pl.ds(0, hidden_size)],
                dst_ref=w2_down_vmem.at[buf_id, pl.ds(0, intermediate_size_per_chip), pl.ds(0, hidden_size)],
                sem=local_sems.at[buf_id, 2]
            ).start()

        def wait_weights(buf_id):
            pltpu.make_async_copy(src_ref=w1_gate_vmem.at[buf_id], dst_ref=w1_gate_vmem.at[buf_id], sem=local_sems.at[buf_id, 0]).wait()
            pltpu.make_async_copy(src_ref=w1_up_vmem.at[buf_id], dst_ref=w1_up_vmem.at[buf_id], sem=local_sems.at[buf_id, 1]).wait()
            pltpu.make_async_copy(src_ref=w2_down_vmem.at[buf_id], dst_ref=w2_down_vmem.at[buf_id], sem=local_sems.at[buf_id, 2]).wait()

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
        
        pltpu.make_async_copy(src_ref=y_vmem.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], dst_ref=y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], sem=local_sems.at[0, 2]).start()
        pltpu.make_async_copy(src_ref=y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], dst_ref=y_out_ref.at[pl.ds(0, block_b), pl.ds(0, hidden_size)], sem=local_sems.at[0, 2]).wait()

    return tp_moe_kernel
