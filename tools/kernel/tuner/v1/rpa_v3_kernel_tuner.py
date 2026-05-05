# Copyright 2026 Google LLC
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

import dataclasses
import itertools
import logging
import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            TuningCase,
                                                            TuningStatus)
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import (
    dynamic_validate_inputs, get_kv_cache_shape, get_smem_estimate_bytes,
    get_vmem_estimate_bytes, ragged_paged_attention)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


CASE_DECODE = "decode"
CASE_PREFILL = "prefill"
CASE_MIXED = "mixed"


def cdiv(a, b):
    assert b != 0
    return (a + b - 1) // b


def get_dtype_packing(dtype):
    return 32 // jax.dtypes.itemsize_bits(dtype)


def align_to(x, alignment):
    return cdiv(x, alignment) * alignment


def next_power_of_2(x):
    assert x > 0
    return 1 << (x - 1).bit_length()


def get_simplified_raw_key(
    page_size,
    q_dtype,
    kv_dtype,
    actual_num_q_heads,
    actual_num_kv_heads,
    head_dim,
    max_model_len,
    sliding_window,
):
    """Get the simplified key."""
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_kv_heads_x2 = align_to(actual_num_kv_heads * 2, kv_packing)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head,
                                       q_packing)
    assert num_kv_heads_x2 % 2 == 0

    return (
        next_power_of_2(page_size),
        jnp.dtype(q_dtype).name,
        jnp.dtype(kv_dtype).name,
        next_power_of_2(num_q_heads_per_kv_head * actual_num_kv_heads),
        next_power_of_2(num_kv_heads_x2) // 2,
        align_to(head_dim, 128),
        next_power_of_2(max_model_len),
        sliding_window,
    )


VMEM_LIMIT_BYTES = 60 * 1024 * 1024
SMEM_LIMIT_BYTES = 0.9 * 1024 * 1024
jax.config.parse_flags_with_absl()


def get_decode_only_example(actual_num_seqs, max_model_len):
    """All seqs decode (q_len=1). distribution=[N, N, N]."""
    cu_q_lens = list(range(actual_num_seqs + 1))
    kv_lens = [max_model_len] * actual_num_seqs
    return cu_q_lens, kv_lens, actual_num_seqs, actual_num_seqs


def get_prefill_only_example(actual_num_seqs, chunk_prefill_size,
                             max_model_len):
    """All seqs uniform K-prefill (q_len=K). distribution=[0, N, N].

    Each seq has q_len=K and kv_len=max_model_len, modeling steady-state
    chunked prefill where K is one chunk of an already-partially-prefilled
    prompt — so the kernel must read (kv_len - K) tokens from the paged cache
    per seq, exercising the realistic in-production path.
    """
    K = chunk_prefill_size
    assert K > 0
    cu_q_lens = [0]
    for _ in range(actual_num_seqs):
        cu_q_lens.append(cu_q_lens[-1] + K)
    kv_lens = [max_model_len] * actual_num_seqs
    return cu_q_lens, kv_lens, 0, actual_num_seqs


def get_mixed_example(actual_num_seqs, max_num_tokens, max_model_len):
    """Mixed: 1 decode + (N-1) prefill of varied q_len. distribution=[1, 1, N]."""
    if actual_num_seqs == 1:
        cu_q_lens = [0, max_num_tokens]
        decode_end = 0
    else:
        decode_end = 1
        cu_q_lens = [0, 1]
        num_prefill_seqs = actual_num_seqs - 1
        tokens_for_prefill = max_num_tokens - 1
        q_len_per_seq = tokens_for_prefill // num_prefill_seqs
        r = tokens_for_prefill % num_prefill_seqs
        for i in range(num_prefill_seqs):
            q_len = q_len_per_seq + (1 if i < r else 0)
            cu_q_lens.append(cu_q_lens[-1] + q_len)
    kv_lens = []
    for i in range(actual_num_seqs):
        q_len = cu_q_lens[i + 1] - cu_q_lens[i]
        kv_lens.append(max_model_len if q_len == 1 else q_len)
    return cu_q_lens, kv_lens, decode_end, decode_end


@dataclasses.dataclass
class TuningKey:
    page_size: int
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    sliding_window: int
    case: str
    chunk_prefill_size: int  # 0 when not CASE_PREFILL


@dataclasses.dataclass
class TunableParams:
    bq_sz: int
    bkv_sz: int
    bq_csz: int
    bkv_csz: int


class RpaV3KernelTuner(KernelTunerBase):
    """Tunes RPA v3 separately for the DECODE / PREFILL / MIXED cases.

    The PREFILL case sweeps additionally over chunk_prefill_size (K), constrained
    to K % bq_sz == 0 to avoid within-iter padding waste.
    """

    def __init__(self, storage_manager):
        super().__init__(tuning_key_class=TuningKey,
                         tunable_params_class=TunableParams,
                         storage_manager=storage_manager,
                         job_bucket_size=100,
                         kernel_tuner_name="rpa_v3_kernel_tuner")
        # ---- Llama 3 8B preset (first tuning target) ----
        # Override these fields on a tuner instance for other models.
        # Llama 3 8B: num_q_heads=32, num_kv_heads=8 (GQA 4:1), head_dim=128,
        # no sliding window, native context 8192.

        # Workload shape.
        self.max_num_tokens = 2048
        self.max_model_len = 8192
        self.max_num_seqs = 32
        # Sized for max_model_len/page_size pages_per_seq × max_num_seqs.
        self.total_num_pages = 4096

        # Model shape.
        self.page_size = [64, 128]
        self.q_dtype = jnp.bfloat16
        # Switch to [jnp.float8_e4m3fn] if serving with quantized KV cache.
        self.kv_dtype = [jnp.bfloat16]
        self.num_q_heads = [32]
        self.num_kv_heads = [8]
        self.head_dim = [128]
        self.sliding_window = [None]

        # Cases to tune. Default is PREFILL only (the original target);
        # the env var RPA_V3_TUNER_CASES overrides — comma-separated subset of
        # {"decode","prefill","mixed"}. Lets a wrapper script tune one case
        # per python invocation without code edits, and matches the local-DB
        # convention of "one case_set_id per case".
        cases_env = os.getenv("RPA_V3_TUNER_CASES")
        if cases_env:
            self.cases = [c.strip() for c in cases_env.split(",") if c.strip()]
            valid = {CASE_DECODE, CASE_PREFILL, CASE_MIXED}
            unknown = [c for c in self.cases if c not in valid]
            if unknown:
                raise ValueError(
                    f"RPA_V3_TUNER_CASES contains unknown case(s) {unknown}; "
                    f"valid: {sorted(valid)}")
        else:
            self.cases = [CASE_PREFILL]

        # Tunable block-size sweep, anchored at the kernel's v7 default for
        # this model (bq_sz=512, bkv_sz=2048, bq_csz=256, bkv_csz=512 — see
        # get_default_block_sizes in kernel.py).
        self.bq_sz_lst = [32, 64, 128, 256, 512]
        self.bkv_sz_lst = [512, 1024, 2048, 4096]
        self.bq_csz_lst = [32, 64, 128, 256]
        self.bkv_csz_lst = [128, 256, 512, 1024]

        # Chunk prefill sizes to sweep (PREFILL only).
        self.chunk_prefill_size_lst = [128, 256, 512, 1024, 2048]

    def _block_sizes_valid(self, case, page_size, bq_sz, bkv_sz, bq_csz,
                           bkv_csz, K):
        if bq_sz % bq_csz != 0:
            return False
        if bkv_sz % bkv_csz != 0:
            return False
        if bkv_sz % page_size != 0:
            return False
        if bkv_csz % page_size != 0:
            return False
        if case == CASE_DECODE:
            if bq_sz != 1 or bq_csz != 1:
                return False
        if case == CASE_PREFILL:
            if K % bq_sz != 0:
                return False
            if bq_sz > K:
                return False
        return True

    def generate_cases(self) -> list[TuningCase]:
        as_list = lambda x: x if isinstance(x, list) else [x]
        sweep = dict(
            page_size=as_list(self.page_size),
            q_dtype=as_list(self.q_dtype),
            kv_dtype=as_list(self.kv_dtype),
            num_q_heads=as_list(self.num_q_heads),
            num_kv_heads=as_list(self.num_kv_heads),
            head_dim=as_list(self.head_dim),
            max_model_len=as_list(self.max_model_len),
            sliding_window=as_list(self.sliding_window),
            cases=as_list(self.cases),
        )
        bq_sz_lst = as_list(self.bq_sz_lst)
        bkv_sz_lst = as_list(self.bkv_sz_lst)
        bq_csz_lst = as_list(self.bq_csz_lst)
        bkv_csz_lst = as_list(self.bkv_csz_lst)
        K_lst = as_list(self.chunk_prefill_size_lst)

        out: list[TuningCase] = []
        for (ps, qd, kd, nq, nk, hd, mml, sw, case) in itertools.product(
                sweep["page_size"], sweep["q_dtype"], sweep["kv_dtype"],
                sweep["num_q_heads"], sweep["num_kv_heads"],
                sweep["head_dim"], sweep["max_model_len"],
                sweep["sliding_window"], sweep["cases"]):
            if case == CASE_DECODE:
                bq_iter, bqc_iter, K_iter = [1], [1], [0]
            elif case == CASE_PREFILL:
                bq_iter, bqc_iter, K_iter = bq_sz_lst, bq_csz_lst, K_lst
            else:
                bq_iter, bqc_iter, K_iter = bq_sz_lst, bq_csz_lst, [0]

            for bq_sz, bkv_sz, bq_csz, bkv_csz, K in itertools.product(
                    bq_iter, bkv_sz_lst, bqc_iter, bkv_csz_lst, K_iter):
                if not self._block_sizes_valid(case, ps, bq_sz, bkv_sz, bq_csz,
                                               bkv_csz, K):
                    continue
                # bkv_sz expressed in tokens; cap pages-per-block.
                pages_per_seq = cdiv(mml, ps)
                if bkv_sz // ps > pages_per_seq:
                    continue
                if bkv_sz > 8192:
                    continue

                tuning_key = TuningKey(
                    page_size=ps,
                    q_dtype=jnp.dtype(qd).name,
                    kv_dtype=jnp.dtype(kd).name,
                    num_q_heads=nq,
                    num_kv_heads=nk,
                    head_dim=hd,
                    max_model_len=mml,
                    sliding_window=sw,
                    case=case,
                    chunk_prefill_size=K,
                )
                tunable_params = TunableParams(bq_sz=bq_sz,
                                               bkv_sz=bkv_sz,
                                               bq_csz=bq_csz,
                                               bkv_csz=bkv_csz)
                out.append(TuningCase(tuning_key, tunable_params))
        logger.info(f"[Debug] Generated {len(out)} cases")
        return out

    def _build_inputs(self, tuning_key: TuningKey):
        case = tuning_key.case
        max_num_seqs = self.max_num_seqs
        if case == CASE_DECODE:
            actual_num_seqs = min(32, max_num_seqs)
            return get_decode_only_example(actual_num_seqs,
                                           tuning_key.max_model_len) + (
                                               actual_num_seqs, )
        if case == CASE_PREFILL:
            K = tuning_key.chunk_prefill_size
            assert K > 0
            actual_num_seqs = max(
                1, min(self.max_num_tokens // K, max_num_seqs))
            return get_prefill_only_example(actual_num_seqs, K,
                                            tuning_key.max_model_len) + (
                                                actual_num_seqs, )
        actual_num_seqs = min(8, max_num_seqs)
        return get_mixed_example(actual_num_seqs, self.max_num_tokens,
                                 tuning_key.max_model_len) + (
                                     actual_num_seqs, )

    def generate_inputs(self, tuning_key: TuningKey):
        cache_key = (tuning_key.case, tuning_key.chunk_prefill_size,
                     tuning_key.page_size, tuning_key.q_dtype,
                     tuning_key.kv_dtype, tuning_key.num_q_heads,
                     tuning_key.num_kv_heads, tuning_key.head_dim,
                     tuning_key.max_model_len, tuning_key.sliding_window)
        if getattr(self, "_INPUT_CACHE_KEY", None) == cache_key:
            return self._KERNEL_INPUTS_CACHE
        self._INPUT_CACHE_KEY = cache_key

        cu_q_lens, kv_lens, decode_end, prefill_end, actual_num_seqs = (
            self._build_inputs(tuning_key))

        (
            page_size,
            q_dtype_name,
            kv_dtype_name,
            num_q_heads,
            num_kv_heads,
            head_dim,
            max_model_len,
            _,
        ) = get_simplified_raw_key(
            tuning_key.page_size,
            tuning_key.q_dtype,
            tuning_key.kv_dtype,
            tuning_key.num_q_heads,
            tuning_key.num_kv_heads,
            tuning_key.head_dim,
            tuning_key.max_model_len,
            tuning_key.sliding_window,
        )
        q_dtype = jnp.dtype(q_dtype_name)
        kv_dtype = jnp.dtype(kv_dtype_name)
        self.pages_per_seq = cdiv(max_model_len, page_size)

        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, self.max_num_seqs + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.pad(kv_lens, (0, self.max_num_seqs - kv_lens.shape[0]))

        q_shape = (self.max_num_tokens, num_q_heads, head_dim)
        kv_shape = (self.max_num_tokens, num_kv_heads, head_dim)
        kv_cache_shape = get_kv_cache_shape(self.total_num_pages, page_size,
                                            num_kv_heads, head_dim, kv_dtype)

        rng = np.random.default_rng(1234)
        q = jnp.array(rng.random(size=q_shape, dtype=np.float32),
                      dtype=q_dtype)
        k = jnp.array(rng.random(size=kv_shape, dtype=np.float32),
                      dtype=kv_dtype)
        v = jnp.array(rng.random(size=kv_shape, dtype=np.float32),
                      dtype=kv_dtype)
        kv_cache = jnp.array(rng.random(size=kv_cache_shape,
                                        dtype=np.float32),
                             dtype=kv_dtype)
        page_indices = jnp.array(rng.integers(
            0,
            self.total_num_pages,
            size=(self.max_num_seqs * self.pages_per_seq, ),
            dtype=np.int32),
                                 dtype=jnp.int32)

        distribution = jnp.array([decode_end, prefill_end, actual_num_seqs],
                                 dtype=jnp.int32)
        logger.info(
            f"[Debug] case={tuning_key.case} K={tuning_key.chunk_prefill_size}"
            f" actual_num_seqs={actual_num_seqs} {distribution=}")

        self._KERNEL_INPUTS_CACHE = {
            "cu_q_lens": cu_q_lens,
            "kv_lens": kv_lens,
            "q": q,
            "k": k,
            "v": v,
            "kv_cache": kv_cache,
            "page_indices": page_indices,
            "distribution": distribution,
        }
        return self._KERNEL_INPUTS_CACHE

    def _kernel_kwargs(self, tuning_key: TuningKey,
                       tunable_params: TunableParams) -> dict:
        block_tuple = (tunable_params.bq_sz, tunable_params.bkv_sz,
                       tunable_params.bq_csz, tunable_params.bkv_csz)
        # ragged_paged_attention always emits decode + mixed pallas_calls
        # (and prefill if chunk_prefill_size is set). Each must fit in VMEM
        # at compile time, even when its distribution range is empty at
        # runtime. The kernel's default m_block_sizes for this model
        # (bq_sz=512, bkv_sz=2048 on v7) exceeds 64MB VMEM. Pass tiny
        # block sizes for the no-op buckets so they compile cheaply; the
        # runtime cost is ~tens of us of empty-launch overhead per call.
        ps = tuning_key.page_size
        d_minimal = (1, ps, 1, ps)        # decode requires bq_sz=bq_csz=1
        pm_minimal = (16, ps, 16, ps)     # smallest legal block for p/m
        kwargs = {
            "sliding_window": tuning_key.sliding_window,
            "vmem_limit_bytes": VMEM_LIMIT_BYTES,
        }
        if tuning_key.case == CASE_DECODE:
            kwargs["d_block_sizes"] = block_tuple
            kwargs["m_block_sizes"] = pm_minimal
        elif tuning_key.case == CASE_PREFILL:
            kwargs["p_block_sizes"] = block_tuple
            kwargs["chunk_prefill_size"] = tuning_key.chunk_prefill_size
            kwargs["d_block_sizes"] = d_minimal
            kwargs["m_block_sizes"] = pm_minimal
        else:
            kwargs["m_block_sizes"] = block_tuple
            kwargs["d_block_sizes"] = d_minimal
        return kwargs

    def run(self,
            tuning_key: TuningKey,
            tunable_params: TunableParams,
            iters: int = 1) -> tuple[TuningStatus, float, float]:
        logger.info(f"Running rpa_v3 case={tuning_key.case} "
                    f"K={tuning_key.chunk_prefill_size} "
                    f"key={tuning_key} params={tunable_params} iters={iters}")
        inputs = self.generate_inputs(tuning_key)
        args = [
            inputs["q"],
            inputs["k"],
            inputs["v"],
            inputs["kv_cache"],
            inputs["kv_lens"],
            inputs["page_indices"],
            inputs["cu_q_lens"],
            inputs["distribution"],
        ]
        kwargs = self._kernel_kwargs(tuning_key, tunable_params)

        try:
            dynamic_validate_inputs(*args, **kwargs)
        except Exception as err:
            logger.info(f"[Debug] Validate failed: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")

        vmem_estimate = get_vmem_estimate_bytes(
            tuning_key.num_kv_heads,
            tuning_key.num_q_heads // tuning_key.num_kv_heads,
            tuning_key.head_dim,
            tunable_params.bq_sz,
            tunable_params.bkv_sz,
            tuning_key.q_dtype,
            tuning_key.kv_dtype,
        )
        if vmem_estimate > VMEM_LIMIT_BYTES:
            logger.info(f"[Debug] Skip ({tunable_params=}): "
                        f"{vmem_estimate=} > {VMEM_LIMIT_BYTES=}")
            return TuningStatus.SKIPPED, float("inf"), float("inf")

        smem_estimate = get_smem_estimate_bytes(self.max_num_seqs,
                                                self.pages_per_seq)
        if smem_estimate > SMEM_LIMIT_BYTES:
            logger.info(f"[Debug] Skip ({tunable_params=}): "
                        f"{smem_estimate=} > {SMEM_LIMIT_BYTES=}")
            return TuningStatus.SKIPPED, float("inf"), float("inf")

        # ragged_paged_attention donates (queries, keys, values, kv_cache).
        # kv_cache is also returned with matching shape — JAX aliases it to
        # the return, so we chain it iter-to-iter (same trick as the old
        # tuner). queries/keys/values are not returned with matching shape,
        # so their donated buffers are freed after one call; we pre-clone
        # `iters` fresh copies outside the timer (HBM->HBM, stays on TPU)
        # and feed one per iter, so the timed loop contains only the kernel
        # call — no clone overhead in the measurement.
        try:
            prepped_qkv = [(
                jnp.array(inputs["q"]),
                jnp.array(inputs["k"]),
                jnp.array(inputs["v"]),
            ) for _ in range(iters)]
            jax.block_until_ready(prepped_qkv)
        except Exception as err:
            logger.info(f"[Debug] Clone failed: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")

        try:
            # Async-dispatch all iters back-to-back, single block_until_ready
            # at the end. Lets the device pipeline iter N+1's launch with iter
            # N's tail; host (Python+JAX dispatch) overhead is amortized over
            # the whole loop instead of paid per-iter. Per-call latency
            # approaches kernel time as iters grows. Standard JAX benchmark
            # pattern.
            start_ns = time.perf_counter_ns()
            for q_d, k_d, v_d in prepped_qkv:
                args[0], args[1], args[2] = q_d, k_d, v_d
                _, args[3] = ragged_paged_attention(*args, **kwargs)
            jax.block_until_ready(args[3])
            end_ns = time.perf_counter_ns()
            latency_ns = end_ns - start_ns
            # Propagate the latest (live) kv_cache back into the cached
            # inputs so the next run() call (warmup -> measurement, or the
            # next tunable_params on the same key) reads a valid buffer.
            # The original Array object passed in as args[3] is marked
            # deleted by JAX donation; only the returned object is live.
            inputs["kv_cache"] = args[3]
            return TuningStatus.SUCCESS, latency_ns / iters, latency_ns
        except Exception as err:
            logger.info(f"[Debug] Run failed: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
