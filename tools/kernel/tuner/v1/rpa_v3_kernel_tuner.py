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
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from tools.kernel.tuner.v1.common.kernel_tuner_base import (KernelTunerBase,
                                                            TuningCase,
                                                            TuningStatus)
from tools.kernel.tuner.v1.logical_workload_sizing import (
    size_logical_workload)
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
    q_dtype: str
    kv_dtype: str
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    max_model_len: int
    sliding_window: int
    case: str
    code_revision: str = ""


@dataclasses.dataclass
class TunableParams:
    bq_sz: int
    bkv_sz: int
    bq_csz: int
    bkv_csz: int
    page_size: int
    chunk_prefill_size: int = 0
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

        # tpu_inference HEAD short-SHA, baked into every TuningKey.
        # Consumed by `kernel.raw.jsonl`'s resume skip-key (full TuningKey
        # JSON via `_combo_skip_key`), so a code change at the tpu_inference
        # layer invalidates resume hits and forces a re-tune. Per the
        # users storage policy we do NOT garbage-collect older-commit
        # entries; raw.jsonl just grows with code history.
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
        # NOTE: resume behaviour is owned by the base class via
        # `KERNEL_TUNER_RESUME=1` (opt-in). Default is fresh tune —
        # any existing kernel.raw.jsonl is rotated to .bak.<timestamp>.
        # When opted in, the skip-key includes the full TuningKey
        # (including code_revision), so entries from prior commits
        # don't match. There's no commit-stripping path; to reuse
        # winners across commits you currently have to hand-edit
        # raw.jsonl rows. See base-class _flush_raw_jsonl_once and
        # _load_raw_jsonl_skip_set for the mechanism.
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
        # Resume / idempotency is owned by the base class via
        # `kernel.raw.jsonl` (see `_load_raw_jsonl_skip_set` in
        # `kernel_tuner_base.py`). Each combo's outcome is appended
        # per-combo; resume reads it and skips known-permanent rows.
        # `production.kernel` is the final-winners artifact and is
        # deliberately NOT consulted here — keeping them separate
        # avoids the prior conflation where the winners file doubled
        # as a coarse subspace-level skip-set.

        # =========================================================
        # ENV CONTRACT — all `os.environ` reads happen in this block.
        # No env reads occur AFTER this section. Grouped here as a
        # single audit point for the class's input contract.
        # =========================================================

        # --- Tuner-config env vars ---
        _smoke_test = os.environ.get("SMOKE_TEST") == "1"
        _cases_env = os.environ.get("RPA_V3_TUNER_CASES")

        # --- Tuning keys (workload identity; doc §2 row 1) ---
        # Same order as TuningKey dataclass: q_dtype, kv_dtype come
        # from kernel-side constants below; the rest are env-read here.
        _env_num_q_heads = int(os.environ["NUM_Q_HEADS"])
        _env_num_kv_heads = int(os.environ["NUM_KV_HEADS"])
        _env_head_dim = int(os.environ["HEAD_DIM"])
        _env_max_model_len = int(os.environ["MAX_MODEL_LEN"])

        # --- Per-axis search-space overrides (doc §2 row 3 vars) ---
        # Comma-separated ints, e.g. RPA_V3_BQ_SZ_LST=256,512.
        # Empty / unset = use the default below.
        _env_bq_sz_override = os.environ.get("RPA_V3_BQ_SZ_LST")
        _env_bkv_sz_override = os.environ.get("RPA_V3_BKV_SZ_LST")
        _env_bq_csz_override = os.environ.get("RPA_V3_BQ_CSZ_LST")
        _env_bkv_csz_override = os.environ.get("RPA_V3_BKV_CSZ_LST")
        _env_k_override = os.environ.get("RPA_V3_K_LST")
        _env_mnss_override = os.environ.get("RPA_V3_MAX_NUM_SUBSEQS_LST")
        _env_page_size_override = os.environ.get("RPA_V3_PAGE_SIZE_LST")

        # =========================================================
        # Step A — Tuning keys (workload identity; doc §2 row 1).
        # Order matches TuningKey dataclass fields. Service-tuned
        # vars (MAX_NUM_BATCHED_TOKENS, MAX_NUM_SEQS — doc category
        # 5) are deliberately NOT read at all; synthetic workload
        # sizing derives per-combo in generate_inputs.
        # =========================================================
        self.q_dtype = jnp.bfloat16
        # Switch to [jnp.float8_e4m3fn] if serving with quantized KV cache.
        self.kv_dtype = jnp.bfloat16
        self.num_q_heads = [_env_num_q_heads]
        self.num_kv_heads = [_env_num_kv_heads]
        self.head_dim = [_env_head_dim]
        self.max_model_len = _env_max_model_len
        self.sliding_window = [None]

        # =========================================================
        # Step B — Tuner control: which kernel cases to tune.
        # =========================================================
        # Default = PREFILL only (the original target);
        # RPA_V3_TUNER_CASES env var overrides — comma-separated
        # subset of {"decode","prefill","mixed","logical"}.
        if _cases_env:
            self.cases = [c.strip() for c in _cases_env.split(",") if c.strip()]
            valid = {CASE_DECODE, CASE_PREFILL, CASE_MIXED, CASE_LOGICAL}
            unknown = [c for c in self.cases if c not in valid]
            if unknown:
                raise ValueError(
                    f"RPA_V3_TUNER_CASES contains unknown case(s) {unknown}; "
                    f"valid: {sorted(valid)}")
        else:
            self.cases = [CASE_PREFILL]

        # =========================================================
        # Step D — Tuning params search space (doc §2 row 3).
        # block sizes, page_size, kernel_K (= chunk_prefill_size),
        # mnss. Cross-producted by generate_cases() to yield
        # (tuning_key, tunable_params) combos.
        # =========================================================
        # Tunable block-size defaults, anchored at the kernel's v7 default
        # for this model (bq_sz=512, bkv_sz=2048, bq_csz=256, bkv_csz=512 —
        # see get_default_block_sizes in kernel.py).
        self.bq_sz_lst = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        self.bkv_sz_lst = [512, 1024, 2048, 4096, 8192]
        self.bq_csz_lst = [32, 64, 128, 256, 512, 1024, 2048]
        self.bkv_csz_lst = [128, 256, 512, 1024, 2048]
        self.chunk_prefill_size_lst = [128, 256, 512, 1024, 2048, 4096, 8192]
        self.page_size = [16, 32, 64, 128, 256, 512, 1024, 2048]

        # max_num_subseqs sweep (LOGICAL only).
        # 2026-05-16: doc-compliant SMEM-bound enumeration. mnss is
        # category-3 (kernel-tuned) per doc §2 row 3, with a SMEM-
        # driven cap. Previously derived as MNS * (M+1) which tied the
        # kernel-tune search to a service-tuned var.
        #
        # The upper bound here (65536) is deliberately set ABOVE any
        # SMEM-feasible value (SMEM ~0.9 MiB caps mnss around ~14000-
        # 15000 at the best page_size; less at narrower pages). The
        # runtime `validate_smem_fits()` filters infeasible (page_size,
        # mnss) combos before measurement, so enumerating clearly-too-
        # large mnss values is harmless — they get pruned, not measured.
        # Going wide here avoids hardcoding a hardware-specific number
        # that could be wrong if SMEM or kernel internals change.
        _MNSS_ENUMERATION_CEILING = 65536  # 2^16, safely past any feasible value
        self.max_num_subseqs_lst = [
            2**i for i in range(5, 17) if 2**i <= _MNSS_ENUMERATION_CEILING
        ]
        # Result: [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

        # SMOKE_TEST truncation — fastest possible smoke (one combo per axis).
        # Takes the first value of each list (no magic numbers, so future
        # widening of any axis automatically extends what smoke covers).
        if _smoke_test:
            logger.info("SMOKE_TEST=1 detected: Truncating tuner search space to first value of each axis.")
            self.bq_sz_lst = self.bq_sz_lst[:1]
            self.bkv_sz_lst = self.bkv_sz_lst[:1]
            self.bq_csz_lst = self.bq_csz_lst[:1]
            self.bkv_csz_lst = self.bkv_csz_lst[:1]
            self.chunk_prefill_size_lst = self.chunk_prefill_size_lst[:1]
            self.page_size = self.page_size[:1]
            self.max_num_subseqs_lst = self.max_num_subseqs_lst[:1]

        # Apply per-axis overrides last. Useful for narrowing a sweep
        # to the neighborhood of an existing winner (e.g., pin block
        # sizes from the PREFILL winner and sweep only max_num_subseqs
        # to compare L kernel latency to P kernel latency).
        def _apply_override(raw, default_lst, env_name):
            if not raw:
                return default_lst
            parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
            logger.info("%s override: %s -> %s", env_name, default_lst, parsed)
            return parsed
        self.bq_sz_lst = _apply_override(_env_bq_sz_override, self.bq_sz_lst,
                                         "RPA_V3_BQ_SZ_LST")
        self.bkv_sz_lst = _apply_override(_env_bkv_sz_override, self.bkv_sz_lst,
                                          "RPA_V3_BKV_SZ_LST")
        self.bq_csz_lst = _apply_override(_env_bq_csz_override, self.bq_csz_lst,
                                          "RPA_V3_BQ_CSZ_LST")
        self.bkv_csz_lst = _apply_override(_env_bkv_csz_override, self.bkv_csz_lst,
                                           "RPA_V3_BKV_CSZ_LST")
        self.chunk_prefill_size_lst = _apply_override(
            _env_k_override, self.chunk_prefill_size_lst, "RPA_V3_K_LST")
        self.max_num_subseqs_lst = _apply_override(
            _env_mnss_override, self.max_num_subseqs_lst,
            "RPA_V3_MAX_NUM_SUBSEQS_LST")
        self.page_size = _apply_override(
            _env_page_size_override, self.page_size, "RPA_V3_PAGE_SIZE_LST")

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
        if case in (CASE_PREFILL, CASE_LOGICAL):
            # LOGICAL uses the same kernel path as PREFILL with K_kernel
            # (= chunk_prefill_size) as the static_q_len, so the same
            # bq_sz / K constraints apply to both.
            if K % bq_sz != 0:
                return False
            if bq_sz > K:
                return False
        return True

    def generate_cases(self) -> list[TuningCase]:
        # Workload-key axes (cartesian-producted into every case).
        # q_dtype + max_model_len are scalars by design (used as scalar
        # elsewhere); wrap explicitly so the cartesian product walks one
        # element. The other axes are already lists from __init__.
        workload_axes = (
            [self.q_dtype],
            [self.kv_dtype],
            self.num_q_heads,
            self.num_kv_heads,
            self.head_dim,
            [self.max_model_len],
            self.sliding_window,
        )
        # Per-case kernel-tuned axes. Cases differ in which axes they
        # search vs pin: DECODE forces bq_sz=bq_csz=1 and skips K/mnss;
        # PREFILL/MIXED don't sweep mnss; LOGICAL is the only case that
        # sweeps mnss (= TunableParams.max_num_subseqs).
        bq, bkv = self.bq_sz_lst, self.bkv_sz_lst
        bqc, bkvc = self.bq_csz_lst, self.bkv_csz_lst
        K, mns = self.chunk_prefill_size_lst, self.max_num_subseqs_lst
        ps = self.page_size
        per_case_axes = {
            CASE_DECODE:  (ps, [1],  bkv, [1],  bkvc, [0], [None]),
            CASE_PREFILL: (ps, bq,   bkv, bqc,  bkvc, K,   [None]),
            CASE_LOGICAL: (ps, bq,   bkv, bqc,  bkvc, K,   mns),
            CASE_MIXED:   (ps, bq,   bkv, bqc,  bkvc, [0], [None]),
        }

        # Resume skip-set: combos already recorded in `kernel.raw.jsonl`
        # with a permanent status (SUCCESS / FAILED_OOM / SKIPPED) are
        # known-done. Loading here lets us short-circuit BEFORE running
        # the pre-flight checks again — saves work on resume and avoids
        # appending duplicate SKIPPED rows for combos pre-flight-rejected
        # on a prior run.
        jsonl_skip = self._load_raw_jsonl_skip_set()

        out: list[TuningCase] = []
        n_resumed = 0
        n_rejected = 0
        for case in self.cases:
            axes = per_case_axes[case]
            for (qd, kd, nq, nk, hd, mml, sw,
                 ps, bq_sz, bkv_sz, bq_csz, bkv_csz, K, mns
                 ) in itertools.product(*workload_axes, *axes):
                # Construct keys FIRST so any reject branch can record
                # the combo in raw.jsonl (resume-completeness).
                tuning_key = TuningKey(
                    q_dtype=jnp.dtype(qd).name,
                    kv_dtype=jnp.dtype(kd).name,
                    num_q_heads=nq, num_kv_heads=nk, head_dim=hd,
                    max_model_len=mml, sliding_window=sw,
                    case=case, code_revision=self.code_revision,
                )
                tunable_params = TunableParams(
                    bq_sz=bq_sz, bkv_sz=bkv_sz,
                    bq_csz=bq_csz, bkv_csz=bkv_csz,
                    page_size=ps, chunk_prefill_size=K,
                    max_num_subseqs=mns,
                )

                # Resume short-circuit: known-permanent combo from prior
                # run. Skip without re-evaluating pre-flight or
                # appending a duplicate row.
                if (jsonl_skip and
                        self._combo_skip_key(tuning_key, tunable_params)
                        in jsonl_skip):
                    n_resumed += 1
                    continue

                # Pre-flight rejects: log SKIPPED so subsequent resumes
                # see them as known-done. Without this, every resume
                # re-evaluates these checks (cheap but leaves
                # raw.jsonl an incomplete record of "what's done").
                def _reject(reason: str):
                    nonlocal n_rejected
                    n_rejected += 1
                    logger.debug("Pre-flight skip (%s): %s / %s",
                                 reason, tuning_key, tunable_params)
                    self._append_raw_jsonl(
                        tuning_key=tuning_key,
                        tunable_params=tunable_params,
                        status=TuningStatus.SKIPPED,
                        case_id=-1,
                    )

                if not self._block_sizes_valid(case, ps, bq_sz, bkv_sz,
                                               bq_csz, bkv_csz, K):
                    _reject("block_sizes_invalid")
                    continue

                # bkv_sz expressed in tokens; cap pages-per-block.
                pages_per_seq = cdiv(mml, ps)
                if bkv_sz // ps > pages_per_seq:
                    _reject("bkv_exceeds_pages_per_seq")
                    continue

                # LOGICAL combo-specific bounds: route through the same
                # sizing helper that _build_inputs uses, so the
                # pre-flight check and the workload construction stay
                # in lockstep. See logical_workload_sizing.py.
                if case == CASE_LOGICAL:
                    eff_max_num_tokens_pre = mns * K
                    eff_max_num_seqs_pre = max(
                        1, 1 + eff_max_num_tokens_pre // mml)
                    sizing = size_logical_workload(
                        max_num_seqs=eff_max_num_seqs_pre,
                        max_num_tokens=eff_max_num_tokens_pre,
                        K=K, max_num_subseqs=mns)
                    if not sizing.valid:
                        _reject("logical_sizing_invalid")
                        continue

                # ---- VMEM pre-flight (mathematical pruning) ----
                # The pre-flight uses the same VMEM_LIMIT_BYTES constant
                # as the runtime check (~60 MB on v7x — safe margin
                # under v7xs ~64 MB per-core VMEM capacity). A more
                # aggressive pre-flight (e.g. 16 MiB v6e-era estimate)
                # would silently prune valid v7x combos that the runtime
                # check would happily accept.
                #
                # NOTE: VMEM_LIMIT_BYTES is conservative for v7x and may
                # be too LENIENT for v6e (~16 MB VMEM). When tuning
                # targets a smaller-VMEM device, this constant should be
                # auto-detected (e.g. `pltpu.get_tpu_info().vmem_capacity_bytes`)
                # or set device-specifically. Today the constant is hardcoded.
                vmem_estimate = get_vmem_estimate_bytes(
                    actual_num_kv_heads=nk,
                    actual_num_q_heads_per_kv_head=nq // nk,
                    actual_head_dim=hd,
                    bq_sz=bq_sz, bkv_sz=bkv_sz,
                    q_dtype=qd, kv_dtype=kd,
                )
                if vmem_estimate > VMEM_LIMIT_BYTES:
                    _reject("vmem_overflow")
                    continue

                # LOGICAL-only pre-flight SMEM check. The two iter-keyed
                # prefetches scale with max_num_subseqs and can push SMEM
                # over the limit at the larger M values; pruning here
                # avoids spending a full kernel-compile + run on a combo
                # the runtime check at `run()` would reject anyway.
                if case == CASE_LOGICAL:
                    eff_max_num_seqs_pre = max(1, (mns * K) // mml)
                    smem_estimate = get_smem_estimate_bytes(
                        eff_max_num_seqs_pre, pages_per_seq,
                        max_num_subseqs=mns)
                    if smem_estimate > SMEM_LIMIT_BYTES:
                        _reject("smem_overflow")
                        continue

                out.append(TuningCase(tuning_key, tunable_params))
        logger.info(
            "[generate_cases] %d enumerated, %d resumed-skip, %d pre-flight-rejected",
            len(out), n_resumed, n_rejected)
        return out

    def _build_inputs(self, tuning_key: TuningKey,
                      tunable_params: TunableParams | None = None):
        """Build the synthetic workload + sizing for one combo.

        Single source of truth for per-combo sizing — derives
        `eff_max_num_tokens` / `eff_max_num_seqs` here (per doc §2
        row 5: kernel-tune does not read MAX_NUM_BATCHED_TOKENS /
        MAX_NUM_SEQS from env). For LOGICAL, also computes the
        `LogicalWorkloadSize` so callers can drive the iter-keyed
        prefetch fill without re-invoking `size_logical_workload`.

        Returns:
            (cu_q_lens, kv_lens, decode_end, prefill_end,
             actual_num_seqs, eff_max_num_tokens, eff_max_num_seqs,
             logical_sizing)

            `actual_num_seqs` semantics: PHYS count for D/P/M;
            ITER count (= phys * m_per_phys) for LOGICAL, since
            distribution[2] = num_total and the kernel's
            dynamic_validate_inputs requires i <= j <= k.

            `logical_sizing` is the full `LogicalWorkloadSize`
            for CASE_LOGICAL, None for D/P/M.
        """
        case = tuning_key.case
        mml = tuning_key.max_model_len

        if case == CASE_DECODE:
            # Synthetic: 32 decode seqs, 1 token each.
            eff_max_num_seqs = 32
            eff_max_num_tokens = eff_max_num_seqs  # 1 token per seq
            actual_num_seqs = eff_max_num_seqs
            cu_q_lens, kv_lens, decode_end, prefill_end = (
                get_decode_only_example(actual_num_seqs, mml))
            return (cu_q_lens, kv_lens, decode_end, prefill_end,
                    actual_num_seqs, eff_max_num_tokens,
                    eff_max_num_seqs, None)

        if case == CASE_PREFILL:
            K = tunable_params.chunk_prefill_size
            assert K > 0
            # Synthetic: pack ≤32 K-sized chunks per call.
            eff_max_num_seqs = max(1, min(mml // K, 32))
            eff_max_num_tokens = eff_max_num_seqs * K
            actual_num_seqs = eff_max_num_seqs
            cu_q_lens, kv_lens, decode_end, prefill_end = (
                get_prefill_only_example(actual_num_seqs, K, mml))
            return (cu_q_lens, kv_lens, decode_end, prefill_end,
                    actual_num_seqs, eff_max_num_tokens,
                    eff_max_num_seqs, None)

        if case == CASE_LOGICAL:
            assert tunable_params is not None and (
                tunable_params.max_num_subseqs is not None), (
                "LOGICAL case requires tunable_params with max_num_subseqs "
                f"set, got {tunable_params=}")
            K = tunable_params.chunk_prefill_size
            mns = tunable_params.max_num_subseqs
            assert K > 0
            eff_max_num_tokens = mns * K
            # Per user 2026-05-16: max_num_seqs derived from token
            # budget / per-prompt size (= max_model_len), not from
            # mnss directly.
            eff_max_num_seqs = max(1, eff_max_num_tokens // mml)
            sizing = size_logical_workload(
                max_num_seqs=eff_max_num_seqs,
                max_num_tokens=eff_max_num_tokens,
                K=K, max_num_subseqs=mns)
            assert sizing.valid, (
                "LOGICAL combo passed generate_cases pre-flight but "
                f"failed at _build_inputs sizing. {tuning_key=} "
                f"{tunable_params=} "
                f"eff_max_num_seqs={eff_max_num_seqs} "
                f"eff_max_num_tokens={eff_max_num_tokens}. This is "
                "a logic bug — both sites must use size_logical_workload.")
            cu_q_lens, kv_lens, decode_end, prefill_end = (
                get_logical_only_example(sizing.actual_num_seqs, K,
                                         sizing.m_per_phys, mml))
            # actual_num_seqs in the returned tuple is the ITER count
            # for LOGICAL (= sizing.total_iters): distribution[2] =
            # num_total, and the kernel's dynamic_validate_inputs
            # requires i <= j <= k. Phys count is in logical_sizing.
            return (cu_q_lens, kv_lens, decode_end, prefill_end,
                    sizing.total_iters, eff_max_num_tokens,
                    eff_max_num_seqs, sizing)

        # CASE_MIXED: 1 decode + (N-1) prefill chunks; total tokens
        # = one full prompt (max_model_len).
        eff_max_num_seqs = 8
        eff_max_num_tokens = mml
        actual_num_seqs = eff_max_num_seqs
        cu_q_lens, kv_lens, decode_end, prefill_end = (
            get_mixed_example(actual_num_seqs, eff_max_num_tokens, mml))
        return (cu_q_lens, kv_lens, decode_end, prefill_end,
                actual_num_seqs, eff_max_num_tokens,
                eff_max_num_seqs, None)

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
        # 2026-05-16: page_size moved from TuningKey to TunableParams.
        # Required for cache_key correctness — same page_size with
        # different block sizes is a different cached input set, AND
        # different page_size with same block sizes is also different.
        # Our local run() (line 935) always passes tunable_params; if
        # the base class fallback ever calls without it, fail loud.
        assert tunable_params is not None, (
            "generate_inputs requires tunable_params (page_size + block "
            "sizes are tunable_params fields per docs/tuning_architecture.md "
            "§2 row 3). Local run() at the bottom of this file always passes "
            "tunable_params; this assertion catches base-class fallback paths.")
        cache_key = (tuning_key.case, tunable_params.chunk_prefill_size,
                     tunable_params.page_size, tuning_key.q_dtype,
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

        # Per-combo synthetic-workload sizing lives entirely in
        # _build_inputs (single source of truth — no env-derived
        # MNS/MNB; per doc §2 row 5).
        (cu_q_lens, kv_lens, decode_end, prefill_end, actual_num_seqs,
         eff_max_num_tokens, eff_max_num_seqs,
         logical_sizing) = self._build_inputs(tuning_key, tunable_params)

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
            tunable_params.page_size,
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

        # Tensor allocations use the per-combo eff_* from _build_inputs
        # for every case. KV cache pool sized at
        # (eff_max_num_seqs × pages_per_seq) — was previously
        # self.total_num_pages (init-time worst-case).
        eff_total_num_pages = eff_max_num_seqs * self.pages_per_seq

        cu_q_lens = jnp.array(cu_q_lens, dtype=jnp.int32)
        kv_lens = jnp.array(kv_lens, dtype=jnp.int32)
        cu_q_lens = jnp.pad(cu_q_lens,
                            (0, eff_max_num_seqs + 1 - cu_q_lens.shape[0]))
        kv_lens = jnp.pad(kv_lens, (0, eff_max_num_seqs - kv_lens.shape[0]))

        q_shape = (eff_max_num_tokens, num_q_heads, head_dim)
        kv_shape = (eff_max_num_tokens, num_kv_heads, head_dim)
        kv_cache_shape = get_kv_cache_shape(eff_total_num_pages, page_size,
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
            eff_total_num_pages,
            size=(eff_max_num_seqs * self.pages_per_seq, ),
            dtype=np.int32),
                                 dtype=jnp.int32)

        distribution = jnp.array([decode_end, prefill_end, actual_num_seqs],
                                 dtype=jnp.int32)
        logger.info(
            f"[Debug] case={tuning_key.case} K={tunable_params.chunk_prefill_size}"
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
        # repeated N times, padded to mns. Padding values are 0
        # (kernel never reads beyond distribution[2] = N*M). Sizing
        # comes from _build_inputs (no second size_logical_workload
        # invocation here).
        if logical_sizing is not None:
            mns = tunable_params.max_num_subseqs
            K = tunable_params.chunk_prefill_size
            phys_seq_indices_np = np.zeros(mns, dtype=np.int32)
            q_offsets_np = np.zeros(mns, dtype=np.int32)
            i = 0
            for phys in range(logical_sizing.actual_num_seqs):
                for chunk_idx in range(logical_sizing.m_per_phys):
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
        ps = tunable_params.page_size
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
            kwargs["chunk_prefill_size"] = tunable_params.chunk_prefill_size
            kwargs["d_block_sizes"] = d_minimal
            kwargs["m_block_sizes"] = pm_minimal
        elif tuning_key.case == CASE_LOGICAL:
            # LOGICAL replaces the PREFILL pass at the same bucket
            # position; reuse `p_block_sizes` and `chunk_prefill_size`.
            # The kernel's `use_logical_prefill = phys_seq_indices is
            # not None` check selects the LOGICAL branch when run()
            # passes the iter-keyed prefetches via positional args.
            kwargs["p_block_sizes"] = block_tuple
            kwargs["chunk_prefill_size"] = tunable_params.chunk_prefill_size
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
                    f"K={tunable_params.chunk_prefill_size} "
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

        vmem_estimate = get_vmem_estimate_bytes(
            actual_num_kv_heads=tuning_key.num_kv_heads,
            actual_num_q_heads_per_kv_head=(
                tuning_key.num_q_heads // tuning_key.num_kv_heads),
            actual_head_dim=tuning_key.head_dim,
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
        # 2026-05-16: per-case derivation — kernel-tune no longer reads
        # env MAX_NUM_SEQS (doc §2 row 5). For LOGICAL, derive from
        # (mnss × K) / max_model_len. For D/P/M, use the same per-case
        # constants generate_inputs uses for synthetic workload sizing.
        case = tuning_key.case
        mml = tuning_key.max_model_len
        if (case == CASE_LOGICAL
                and tunable_params.max_num_subseqs is not None):
            _mns = tunable_params.max_num_subseqs
            _K = tunable_params.chunk_prefill_size
            _smem_check_mns = max(1, (_mns * _K) // mml)
        elif case == CASE_DECODE:
            _smem_check_mns = 32
        elif case == CASE_PREFILL:
            _smem_check_mns = max(1, min(mml // tunable_params.chunk_prefill_size, 32))
        else:  # CASE_MIXED
            _smem_check_mns = 8
        smem_estimate = get_smem_estimate_bytes(
            _smem_check_mns,
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
