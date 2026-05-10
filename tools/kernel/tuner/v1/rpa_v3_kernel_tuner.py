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
import json
import logging
import os
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.cache_key_utils import (
    should_skip_commit_cache_check, tuning_key_hash)
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
# Decoupled-K LOGICAL: chunked PREFILL where one phys req emits
# multiple LOGICAL sub-seqs (one per K_kernel-sized chunk). The
# kernel reads the same `p_block_sizes` as PREFILL but additionally
# consumes iter-keyed `phys_seq_indices` / `q_offsets` prefetches.
# `max_num_subseqs` (the prefetch-array sizing bound) is a TunableParam
# rather than a runner-derived geometry — see
# project_tuner_key_max_num_subseqs.md and approach A in the design doc.
CASE_LOGICAL = "logical"


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


def estimate_vmem_for_combo(
    num_kv_heads, num_q_heads, head_dim, bq_sz, bkv_sz,
    q_dtype, kv_dtype,
) -> int:
    """Compute the per-iter VMEM footprint for a (model, block-sizes) combo.

    Single source of truth: previously the same call+threshold check
    was duplicated in `generate_cases` (pre-flight pruning, kwargs form)
    and `measure_one` (runtime check, positional form using TuningKey
    + TunableParams attributes). Two copies are easy to drift —
    one update for a model-arch knob, the other forgotten. Funnel both
    through this helper.
    """
    return get_vmem_estimate_bytes(
        actual_num_kv_heads=num_kv_heads,
        actual_num_q_heads_per_kv_head=num_q_heads // num_kv_heads,
        actual_head_dim=head_dim,
        bq_sz=bq_sz,
        bkv_sz=bkv_sz,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )


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


def get_logical_only_example(actual_num_seqs, K_kernel, M_per_phys,
                             max_model_len):
    """N phys reqs, each producing M LOGICAL chunks of K tokens.

    Models steady-state chunked prefill under decoupled-K: each phys req
    is partway through its prompt (`kv_lens[i] = max_model_len`,
    `q_lens[i] = K * M`), and the kernel processes its q tokens as M
    LOGICAL sub-seqs each of static length K.

    distribution = [0, N*M, N*M] — all iters in the LOGICAL bucket; no
    decode, no MIXED. (A real workload may also have a MIXED tail when
    the per-req scheduled token count is not a multiple of K, but the
    tail interaction is exercised separately in
    `tests/kernels/ragged_paged_attention_kernel_v3_logical_test.py`.)
    """
    assert K_kernel > 0
    assert M_per_phys > 0
    cu_q_lens = [0]
    for _ in range(actual_num_seqs):
        cu_q_lens.append(cu_q_lens[-1] + K_kernel * M_per_phys)
    kv_lens = [max_model_len] * actual_num_seqs
    return cu_q_lens, kv_lens, 0, actual_num_seqs * M_per_phys


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
    # tpu_inference HEAD short-SHA at tune time. Included in the
    # idempotency cache key so a kernel.py change invalidates prior
    # winners — old commits stale entries naturally do not match the
    # current commits hash and get re-tuned. Default empty string for
    # backwards-compat with older entries that pre-date this field.
    code_revision: str = ""


@dataclasses.dataclass
class TunableParams:
    bq_sz: int
    bkv_sz: int
    bq_csz: int
    bkv_csz: int
    # Decoupled-K only: prefetch-array sizing bound for the LOGICAL
    # kernel (`phys_seq_indices` / `q_offsets` shape). The static SMEM
    # model in `subseq_planner.max_M_under_smem` is approximate, so we
    # let the tuner discover the largest workable max_num_subseqs by
    # sweeping. Sentinel `None` means "not applicable" (D / P / M cases
    # don't use it). Stored as `null` in production.kernel JSON so
    # downstream consumers can distinguish "untuned" from "tuned to 0".
    max_num_subseqs: int | None = None


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

        # tpu_inference HEAD short-SHA, baked into every TuningKey so
        # idempotency invalidates when kernel.py (or any tpu_inference
        # code the kernel transitively depends on) changes. Per the
        # users storage policy we do NOT garbage-collect older-commit
        # entries; the registry just grows with code history.
        #
        # Asymmetry note (intentional): the sweep-side cache key
        # (sweep.py is_completed) mixes BOTH tpu_inference and vllm
        # commit SHAs, because the served stack depends on both. The
        # kernel cache key here mixes ONLY tpu_inference, because the
        # tuned artefact is a JAX kernel function — it does not
        # transitively depend on any vllm code. A vllm bump should not
        # invalidate kernel winners; it should only invalidate sweep
        # results. If that assumption ever changes (e.g. the kernel
        # interface starts importing from vllm), add vllm SHA here.
        #
        # SKIP_COMMIT_CACHE_CHECK=1 escape hatch: ignore code_revision
        # when checking idempotency. Mirrors sweep.pys behaviour. Useful
        # for re-using winners from prior commits when iterating on
        # non-kernel code (docs, sweep recipes, runlogs) without paying
        # a 3-4 hour re-tune.
        try:
            self.code_revision = subprocess.run(
                ["git", "rev-parse", "--short=12", "HEAD"],
                capture_output=True, check=True, text=True,
                timeout=5,
            ).stdout.strip()
        except (subprocess.SubprocessError, OSError, FileNotFoundError):
            self.code_revision = ""
            logger.warning(
                "Could not determine tpu_inference HEAD commit; "
                "TuningKey.code_revision will be empty (cache invalidation "
                "across commits will not work). Are we outside a git checkout?")
        self._skip_commit_cache_check = should_skip_commit_cache_check()
        if self._skip_commit_cache_check:
            logger.warning(
                "SKIP_COMMIT_CACHE_CHECK=1: code_revision will be ignored "
                "when checking idempotency; entries from older commits will "
                "be reused.")

        # ---- Idempotency: Load existing registry to skip tuned cases ----
        #
        # Granularity is by `tuning_key` only (page_size, case, K, model
        # geometry, ...) — NOT by `(tuning_key, tunable_params)`. So once
        # ANY winner exists for `(page=128, K=512)` in the registry, the
        # tuner skips ALL `(bq, bkv, bq_csz, bkv_csz)` combos for that
        # key on re-run.
        #
        # This works well for INCREMENTAL coverage (adding new K values
        # or new page sizes to the search) but does NOT support EXPANDING
        # the inner block-size grid (e.g. raising bkv_sz_lst max from
        # 2048 to 8192 and re-running the same K). To re-tune an
        # existing key with a wider inner grid, delete or trim the
        # corresponding entries from production.kernel before running.
        # If you need finer-grained idempotency, hash
        # `(tuning_key, tunable_params)` instead.
        self.completed_tuning_keys = set()
        registry_path = os.environ.get("RPA_V3_KERNEL_REGISTRY")
        if registry_path and os.path.exists(registry_path):
            try:
                with open(registry_path, "r") as f:
                    registry_data = json.load(f)
                for case_name, results in registry_data.get("results", {}).items():
                    for entry in results:
                        tk_dict = entry.get("tuning_key", {})
                        # Skip malformed/empty entries — without this an
                        # entry whose tuning_key is missing (e.g. {} from
                        # a pre-rename schema or an incomplete write)
                        # would absorb its hash ('{}') into the skip set
                        # and silently make EVERY un-keyed combo a no-op.
                        if not tk_dict:
                            continue
                        tk_hash = tuning_key_hash(
                            tk_dict,
                            skip_commit_cache_check=self._skip_commit_cache_check)
                        self.completed_tuning_keys.add(tk_hash)
                logger.info(f"Loaded {len(self.completed_tuning_keys)} completed TuningKeys from {registry_path}")
            except Exception as e:
                logger.warning(f"Failed to read existing kernel registry: {e}")

        # ---- Single Source of Truth: Read from environment ----
        # These must be set by tune_all_cases.sh sourcing a .workload case file.
        self.max_num_tokens = int(os.environ["MAX_NUM_BATCHED_TOKENS"])
        self.max_model_len = int(os.environ["MAX_MODEL_LEN"])
        self.max_num_seqs = int(os.environ["MAX_NUM_SEQS"])

        # Model shape.
        self.page_size = [64, 128, 256]

        # Synthetic KV-cache shape needs enough pages for the WORST-case
        # tuning workload: smallest swept page_size + full max_model_len
        # × max_num_seqs. Hard-coding 4096 was correct only at the
        # original (max_num_seqs=32, max_model_len=8192, page_size=64)
        # = 32 * 128 = 4096 fit. At max_num_seqs=128 the requirement
        # is 16384, and the under-allocated cache made page indices
        # wrap silently. Compute it from the actual sweep parameters.
        smallest_page = min(self.page_size)
        worst_pages_per_seq = cdiv(self.max_model_len, smallest_page)
        self.total_num_pages = worst_pages_per_seq * self.max_num_seqs
        self.q_dtype = jnp.bfloat16
        # Switch to [jnp.float8_e4m3fn] if serving with quantized KV cache.
        self.kv_dtype = [jnp.bfloat16]
        self.num_q_heads = [int(os.environ["NUM_Q_HEADS"])]
        self.num_kv_heads = [int(os.environ["NUM_KV_HEADS"])]
        self.head_dim = [int(os.environ["HEAD_DIM"])]
        self.sliding_window = [None]

        # Cases to tune. Default is PREFILL only (the original target);
        # the env var RPA_V3_TUNER_CASES overrides — comma-separated subset of
        # {"decode","prefill","mixed"}. Lets a wrapper script tune one case
        # per python invocation without code edits, and matches the local-DB
        # convention of "one case_set_id per case".
        cases_env = os.getenv("RPA_V3_TUNER_CASES")
        if cases_env:
            self.cases = [c.strip() for c in cases_env.split(",") if c.strip()]
            valid = {CASE_DECODE, CASE_PREFILL, CASE_MIXED, CASE_LOGICAL}
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
        self.bq_sz_lst = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self.bkv_sz_lst = [512, 1024, 2048, 4096, 8192]
        self.bq_csz_lst = [32, 64, 128, 256, 512, 1024, 2048]
        self.bkv_csz_lst = [128, 256, 512, 1024, 2048]

        # Chunk prefill sizes to sweep (PREFILL and LOGICAL).
        self.chunk_prefill_size_lst = [128, 256, 512, 1024, 2048, 4096, 8192]

        # max_num_subseqs sweep (LOGICAL only). Workload-relative: anchored
        # to the formula `max_num_subseqs = max_num_seqs * (M + 1)` with
        # M ranging across powers of 2. M is the chunks-per-req bound, so
        # M=1 means "single chunk per req" (≈ coupled-K behaviour with the
        # LOGICAL prefetches as a no-op), and larger M corresponds to
        # finer chunking. The +max_num_seqs slack term in the formula
        # accounts for one MIXED tail per req — even though our tuning
        # workload doesnt emit MIXED tails, the kernel still allocates
        # the slot and the SMEM accounting must include it for parity
        # with production runs.
        self.max_num_subseqs_lst = [
            self.max_num_seqs * (M + 1) for M in (1, 2, 4, 8, 16, 32)
        ]

        if os.environ.get("SMOKE_TEST") == "1":
            logger.info("SMOKE_TEST=1 detected: Truncating tuner search space to minimum.")
            self.bq_sz_lst = [128]
            self.bkv_sz_lst = [512]
            self.bq_csz_lst = [128]
            self.bkv_csz_lst = [256]
            self.chunk_prefill_size_lst = [128]
            self.page_size = [128]
            self.max_num_subseqs_lst = [self.max_num_seqs * 2]   # M=1

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
        if case == CASE_LOGICAL:
            # LOGICAL uses the same kernel path as PREFILL with K_kernel
            # as the static_q_len, so the same bq_sz/K constraints apply.
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
                mns_iter = [None]
            elif case == CASE_PREFILL:
                bq_iter, bqc_iter, K_iter = bq_sz_lst, bq_csz_lst, K_lst
                mns_iter = [None]
            elif case == CASE_LOGICAL:
                # LOGICAL uses the PREFILL block-size sweep + K sweep AND
                # additionally sweeps max_num_subseqs.
                bq_iter, bqc_iter, K_iter = bq_sz_lst, bq_csz_lst, K_lst
                mns_iter = as_list(self.max_num_subseqs_lst)
            else:
                bq_iter, bqc_iter, K_iter = bq_sz_lst, bq_csz_lst, [0]
                mns_iter = [None]

            for bq_sz, bkv_sz, bq_csz, bkv_csz, K, mns in itertools.product(
                    bq_iter, bkv_sz_lst, bqc_iter, bkv_csz_lst, K_iter,
                    mns_iter):
                if not self._block_sizes_valid(case, ps, bq_sz, bkv_sz, bq_csz,
                                               bkv_csz, K):
                    continue
                # bkv_sz expressed in tokens; cap pages-per-block.
                pages_per_seq = cdiv(mml, ps)
                if bkv_sz // ps > pages_per_seq:
                    continue

                # LOGICAL combo-specific bounds:
                if case == CASE_LOGICAL:
                    # max_num_subseqs must accommodate at least one slot
                    # per phys req (the formulas +max_num_seqs slack term).
                    if mns < self.max_num_seqs:
                        continue
                    # The synthetic LOGICAL workload uses
                    # M_per_phys = (mns // max_num_seqs) - 1 chunks per
                    # phys at K tokens each. If that exceeds
                    # max_num_tokens // max_num_seqs we cant fit the
                    # workload in the q buffer — skip.
                    M_per_phys = max(1, mns // self.max_num_seqs - 1)
                    per_phys_q = K * M_per_phys
                    if per_phys_q * self.max_num_seqs > self.max_num_tokens:
                        continue

                # ---- Hybrid VMEM-Aware Tuning Pruning (Theoretical bounds) ----
                # Mathematically estimate the kernels per-iteration VMEM
                # footprint and drop combos that wont fit. This saves hours
                # of guaranteed JAX compilation OOM crashes.
                #
                # The pre-flight uses the same VMEM_LIMIT_BYTES constant as
                # the runtime check below (~60 MB on v7x — chosen as a safe
                # margin under v7xs ~64 MB per-core VMEM capacity). A more
                # aggressive pre-flight (e.g. 16 MiB v6e-era estimate) would
                # silently prune valid v7x combos that the runtime check
                # would happily accept.
                #
                # NOTE: VMEM_LIMIT_BYTES is conservative for v7x and may
                # be too LENIENT for v6e (which has ~16 MB VMEM). When
                # tuning targets a smaller-VMEM device, this constant
                # should be auto-detected (e.g. via
                # `pltpu.get_tpu_info().vmem_capacity_bytes`) or set
                # device-specifically. Today the constant is hardcoded.
                vmem_estimate = estimate_vmem_for_combo(
                    num_kv_heads=nk, num_q_heads=nq, head_dim=hd,
                    bq_sz=bq_sz, bkv_sz=bkv_sz,
                    q_dtype=qd, kv_dtype=kd,
                )
                if vmem_estimate > VMEM_LIMIT_BYTES:
                    continue

                # LOGICAL-only pre-flight SMEM check. The two iter-keyed
                # prefetches scale with max_num_subseqs and can push SMEM
                # over the limit at the larger M values; pruning here
                # avoids spending a full kernel-compile + run on a combo
                # the runtime check at `run()` would reject anyway.
                # Non-LOGICAL combos defer to the runtime check
                # (preserving existing tune behaviour).
                if case == CASE_LOGICAL:
                    smem_estimate = get_smem_estimate_bytes(
                        self.max_num_seqs, pages_per_seq,
                        max_num_subseqs=mns)
                    if smem_estimate > SMEM_LIMIT_BYTES:
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
                    code_revision=self.code_revision,
                )

                # Skip if we already successfully tuned this exact environment shape
                tk_dict = dataclasses.asdict(tuning_key)
                tk_hash = tuning_key_hash(
                    tk_dict,
                    skip_commit_cache_check=self._skip_commit_cache_check)
                if tk_hash in self.completed_tuning_keys:
                    continue

                tunable_params = TunableParams(bq_sz=bq_sz,
                                               bkv_sz=bkv_sz,
                                               bq_csz=bq_csz,
                                               bkv_csz=bkv_csz,
                                               max_num_subseqs=mns)
                out.append(TuningCase(tuning_key, tunable_params))
        logger.info(f"[Debug] Generated {len(out)} cases")
        return out

    def _build_inputs(self, tuning_key: TuningKey,
                      tunable_params: TunableParams | None = None):
        """Build the (cu_q_lens, kv_lens, decode_end, prefill_end,
        actual_num_seqs) tuple for the case's synthetic workload.

        For CASE_LOGICAL, `tunable_params` is required because the
        workload shape depends on the tuned `max_num_subseqs` (it
        determines M_per_phys = chunks per req in the synthetic
        workload). For other cases, the workload is tuning-key-only
        and `tunable_params` is ignored.
        """
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
        if case == CASE_LOGICAL:
            assert tunable_params is not None and (
                tunable_params.max_num_subseqs is not None), (
                "LOGICAL case requires tunable_params with max_num_subseqs "
                f"set, got {tunable_params=}")
            K = tuning_key.chunk_prefill_size
            assert K > 0
            mns = tunable_params.max_num_subseqs
            # M_per_phys derives from the formula
            # `max_num_subseqs = max_num_seqs * (M + 1)` -> M = mns/N - 1.
            # Floor at 1 so the workload always has at least one chunk
            # per phys; bounds check in generate_cases ensures the
            # resulting per-phys q_len fits max_num_tokens.
            M_per_phys = max(1, mns // max_num_seqs - 1)
            actual_num_seqs = max_num_seqs
            return get_logical_only_example(
                actual_num_seqs, K, M_per_phys,
                tuning_key.max_model_len) + (actual_num_seqs, )
        actual_num_seqs = min(8, max_num_seqs)
        return get_mixed_example(actual_num_seqs, self.max_num_tokens,
                                 tuning_key.max_model_len) + (
                                     actual_num_seqs, )

    def generate_inputs(self, tuning_key: TuningKey,
                        tunable_params: TunableParams | None = None):
        # For LOGICAL, the workload shape depends on the tunable
        # `max_num_subseqs` (M_per_phys derives from it), so the cache
        # key must include it. For D / P / M cases, tunable_params is
        # ignored and `mns_for_cache` is None — same caching behaviour
        # as before. The base class invokes generate_inputs without
        # tunable_params; the override here only requires it for
        # LOGICAL, which the override-aware run() at the bottom of this
        # file always supplies.
        mns_for_cache = (tunable_params.max_num_subseqs
                         if tuning_key.case == CASE_LOGICAL and
                         tunable_params is not None else None)
        cache_key = (tuning_key.case, tuning_key.chunk_prefill_size,
                     tuning_key.page_size, tuning_key.q_dtype,
                     tuning_key.kv_dtype, tuning_key.num_q_heads,
                     tuning_key.num_kv_heads, tuning_key.head_dim,
                     tuning_key.max_model_len, tuning_key.sliding_window,
                     mns_for_cache)
        if getattr(self, "_INPUT_CACHE_KEY", None) == cache_key:
            return self._KERNEL_INPUTS_CACHE
        # Cache miss = switching to a different shape. Explicitly release
        # the previous keys HBM buffers BEFORE we start allocating the
        # new ones, otherwise peak HBM = old_buffers + new_buffers
        # transiently, which OOMs at 70B / 32K (kv_cache test buffer
        # alone is ~8 GB before sharding). JAX does not auto-free a
        # buffer on Python refcount drop — the device-allocators
        # release pass lags. ``jax.Array.delete()`` releases device
        # memory immediately. The cache only ever holds one keys
        # buffers, so this delete is safe.
        old_cache = getattr(self, "_KERNEL_INPUTS_CACHE", None)
        if old_cache is not None:
            for arr in old_cache.values():
                if hasattr(arr, "delete"):
                    try:
                        arr.delete()
                    except Exception:
                        # Already deleted (e.g. donated to a kernel call
                        # earlier and not refreshed) — fine, nothing to do.
                        pass
            self._KERNEL_INPUTS_CACHE = None
        self._INPUT_CACHE_KEY = cache_key

        cu_q_lens, kv_lens, decode_end, prefill_end, actual_num_seqs = (
            self._build_inputs(tuning_key, tunable_params))

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

        cache_dict = {
            "cu_q_lens": cu_q_lens,
            "kv_lens": kv_lens,
            "q": q,
            "k": k,
            "v": v,
            "kv_cache": kv_cache,
            "page_indices": page_indices,
            "distribution": distribution,
        }

        # LOGICAL: build the iter-keyed prefetch arrays from the
        # synthetic workload. phys_seq_indices: [0]*M + [1]*M + ... +
        # [N-1]*M, padded to mns. q_offsets: [0, K, 2K, ..., (M-1)*K]
        # repeated N times, padded to mns. Padding values are 0 (kernel
        # never reads beyond distribution[2] = N*M).
        if (tuning_key.case == CASE_LOGICAL and tunable_params is not None
                and tunable_params.max_num_subseqs is not None):
            mns = tunable_params.max_num_subseqs
            K = tuning_key.chunk_prefill_size
            M_per_phys = max(1, mns // self.max_num_seqs - 1)
            phys_seq_indices_np = np.zeros(mns, dtype=np.int32)
            q_offsets_np = np.zeros(mns, dtype=np.int32)
            i = 0
            for phys in range(actual_num_seqs):
                for chunk_idx in range(M_per_phys):
                    phys_seq_indices_np[i] = phys
                    q_offsets_np[i] = chunk_idx * K
                    i += 1
            cache_dict["phys_seq_indices"] = jnp.array(phys_seq_indices_np)
            cache_dict["q_offsets"] = jnp.array(q_offsets_np)

        self._KERNEL_INPUTS_CACHE = cache_dict
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
        elif tuning_key.case == CASE_LOGICAL:
            # LOGICAL replaces the PREFILL pass at the same bucket
            # position; reuse `p_block_sizes` and `chunk_prefill_size`.
            # The kernel's `use_logical_prefill = phys_seq_indices is
            # not None` check selects the LOGICAL branch when run()
            # passes the iter-keyed prefetches via positional args.
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
        inputs = self.generate_inputs(tuning_key, tunable_params)
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
        # LOGICAL: append iter-keyed prefetches to args at positions 8,
        # 9 — matches `ragged_paged_attention()` positional signature.
        if tuning_key.case == CASE_LOGICAL:
            args.append(inputs["phys_seq_indices"])
            args.append(inputs["q_offsets"])
        kwargs = self._kernel_kwargs(tuning_key, tunable_params)

        try:
            dynamic_validate_inputs(*args, **kwargs)
        except Exception as err:
            logger.info(f"[Debug] Validate failed: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")

        vmem_estimate = estimate_vmem_for_combo(
            num_kv_heads=tuning_key.num_kv_heads,
            num_q_heads=tuning_key.num_q_heads,
            head_dim=tuning_key.head_dim,
            bq_sz=tunable_params.bq_sz,
            bkv_sz=tunable_params.bkv_sz,
            q_dtype=tuning_key.q_dtype,
            kv_dtype=tuning_key.kv_dtype,
        )
        if vmem_estimate > VMEM_LIMIT_BYTES:
            logger.info(f"[Debug] Skip ({tunable_params=}): "
                        f"{vmem_estimate=} > {VMEM_LIMIT_BYTES=}")
            return TuningStatus.SKIPPED, float("inf"), float("inf")

        # LOGICAL adds two iter-keyed prefetch arrays sized at
        # max_num_subseqs; pass through to the SMEM estimator for an
        # accurate check. Non-LOGICAL: max_num_subseqs=None ->
        # collapses to max_num_seqs (coupled-K identity).
        smem_estimate = get_smem_estimate_bytes(
            self.max_num_seqs,
            self.pages_per_seq,
            max_num_subseqs=tunable_params.max_num_subseqs)
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
        except jax.errors.JaxRuntimeError as err:
            if "ResourceExhausted" in str(err) or "Out of memory" in str(err):
                logger.info(f"[Debug] Run OOM (empirical bound hit): {err=}")
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.info(f"[Debug] Run JAX runtime error: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
        except Exception as err:
            if "ResourceExhausted" in str(err):
                logger.info(f"[Debug] Compilation OOM: {err=}")
                return TuningStatus.FAILED_OOM, float("inf"), float("inf")
            logger.info(f"[Debug] Run failed: {err=}")
            return TuningStatus.UNKNOWN_ERROR, float("inf"), float("inf")
