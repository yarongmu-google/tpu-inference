# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
from dataclasses import dataclass, field
from typing import Any

import jax


@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "input_positions",
        "block_tables",
        "seq_lens",
        "query_start_loc",
        "request_distribution",
        "mamba_state_indices",
        # Decoupled-K (RPA_KERNEL_K) sub-chunk indirection. Both fields
        # are None on todays coupled-K path — kernel skips the extra
        # dereference and uses seq_idx directly with phys-keyed
        # cu_q_lens / kv_lens. When set, the kernel reads:
        #   phys_seq_indices[seq_idx] -> physical-request slot for
        #     page_indices_ref / cu_q_lens / kv_lens lookups.
        #   q_offsets[seq_idx]        -> q-token offset into the
        #     physical requests scheduled-token slice where this iters
        #     q-window begins. Pairs with the kernels static_q_len to
        #     define [q_start, q_start + static_q_len).
        # See tpu_inference/runner/subseq_planner.py for the per-step
        # chunking semantics and the SMEM accounting.
        "phys_seq_indices",
        "q_offsets",
    ],
    meta_fields=[
        # Static Python int (or None) — captured into the JIT trace as
        # a constant. Plumbed through to ragged_paged_attention to
        # control whether the kernel runs its static-q-len PREFILL
        # pass. Previously a module-global in attention_interface.py;
        # threading it here lets the runner set it per-step on the
        # metadata and avoids the JIT-trace-capture footgun of a
        # mutable module global.
        "chunk_prefill_size",
    ],
    drop_fields=["query_start_loc_cpu", "seq_lens_cpu"],
)
@dataclass
class AttentionMetadata(object):
    # (padded_total_num_scheduled_tokens,)
    input_positions: jax.Array
    # (max_num_seqs * max_num_blocks_per_req,)
    # None for pooling models that using no KV cache
    block_tables: jax.Array | None = None
    # (max_num_seqs,)
    seq_lens: jax.Array = None
    # (max_num_seqs + 1,)
    query_start_loc: jax.Array = None
    # (3,)
    request_distribution: jax.Array = None
    # (max_num_seqs,) int32 — physical slot id (∈ [0, _mamba_num_blocks))
    # in the mamba kv-cache for the request currently in each persistent-
    # batch position. Used by mamba/GDN ops to read/write recurrent state
    # without going through `block_tables`, since the mamba pool is
    # smaller than the attention pool under compact-mamba sizing.
    # None for models without mamba layers; pure-mamba models would also
    # use this field, only hybrid models exercise it today.
    mamba_state_indices: jax.Array | None = None
    # (max_num_subseqs,) i32 — under decoupled-K, maps each loop iteration
    # (sub-seq slot) to its physical-request slot in the persistent
    # batch. Length of this array sets the kernels iteration count;
    # values are in [0, max_num_seqs). Multiple sub-seqs of the same
    # physical request hold the SAME phys-seq slot value, so they share
    # the corresponding row of page_indices_ref / cu_q_lens / kv_lens —
    # eliminating the SMEM blow-up that naive duplication would cause.
    # See subseq_planner.py for the chunking semantics; kernel.py
    # consumes this together with q_offsets via the LOGICAL dispatch.
    # None when RPA_KERNEL_K is unset (todays coupled-K path); the
    # kernel falls back to seq_idx as the page_indices selector.
    #
    # FORWARD-COMPAT: when adding another iter-keyed prefetch (e.g. a
    # per-iter q_len to remove the MIXED phys-q_len lookup), every site
    # below must be updated in lockstep — silent drift is the failure
    # mode, since the kernel reads positional args:
    #   1. AttentionMetadata data_fields + field def (this file)
    #   2. subseq_planner.IterPrefetches + build_iter_prefetches
    #   3. tpu_runner.py: both _prepare_inputs_dp* DeviceBuffer fills
    #      and both _extract_attn_metadata pulls
    #   4. attention_interface.sharded_ragged_paged_attention args /
    #      in_specs and attention() forwarding
    #   5. kernel.ragged_paged_attention signature + LOGICAL trace
    #   6. compilation_manager.py: both warmup paths (main + eagle3)
    #   7. spec_decode/jax/eagle3.py AttentionMetadata rebuild
    phys_seq_indices: jax.Array | None = None
    # (max_num_subseqs,) i32 — under decoupled-K, the q-token offset into
    # the physical requests scheduled-token slice where this iters
    # q-window begins. 0 for the first sub-seq of a physical request,
    # static_q_len for the second, and so on. Combined with the
    # kernels static_q_len, defines this iters [q_start, q_end)
    # inside cu_q_lens[phys] : cu_q_lens[phys + 1]. None on todays
    # coupled-K path (kernel derives q-bounds from cu_q_lens directly).
    q_offsets: jax.Array | None = None
    # Static (Python int) — when None, ragged_paged_attention skips its
    # PREFILL pass (decode-only + mixed). When set to K > 0, sequences
    # in the request_distribution[0]:request_distribution[1] slice are
    # routed to the static-q-len PREFILL path with q_len == K.
    chunk_prefill_size: int | None = None

    query_start_loc_cpu: Any = field(init=False)
    seq_lens_cpu: Any = field(init=False)
