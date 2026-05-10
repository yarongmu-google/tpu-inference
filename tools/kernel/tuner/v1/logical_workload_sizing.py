# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sizing helper for the synthetic LOGICAL tuning workload.

Pure-stdlib so it can be unit-tested without importing the JAX/TPU
runtime stack. Both the pre-flight pruning in `generate_cases` and
the workload construction in `_build_inputs` of
`tools.kernel.tuner.v1.rpa_v3_kernel_tuner` route through this single
helper, so they stay in lockstep by construction. The regression test
in `tests/benchmark/test_kernel_tuner_workload_sizing.py` covers the
specific (MNS, MNB, K, mns) combos that motivated the helper.

Why centralise: the previous version had the bound check (`per_phys_q
* max_num_seqs > max_num_tokens`) inconsistent with the workload-size
calculation (which hardcoded `actual_num_seqs = max_num_seqs` instead
of scaling). Result: every LOGICAL combo got pruned and the tune
generated zero cases. See commit c5ade1bb for the fix.
"""

import dataclasses


@dataclasses.dataclass(frozen=True)
class LogicalWorkloadSize:
    """Sizing of the synthetic LOGICAL tuning workload for one combo.

    Attributes:
      valid: True iff this combo can produce a non-empty workload.
      actual_num_seqs: number of phys reqs in the synthetic workload
        (0 when ``valid`` is False; >= 1 otherwise). Scales down from
        ``max_num_seqs`` to fit ``max_num_tokens``.
      m_per_phys: chunks per phys req under this combo's
        ``max_num_subseqs``. Always >= 1 when valid.
      per_phys_q: q-tokens per phys req for the synthetic workload
        (= ``K * m_per_phys``).
      total_iters: ``actual_num_seqs * m_per_phys`` — the number of
        LOGICAL iters the kernel will execute in one pallas_call for
        this combo.
    """
    valid: bool
    actual_num_seqs: int
    m_per_phys: int
    per_phys_q: int
    total_iters: int


def size_logical_workload(
    *,
    max_num_seqs: int,
    max_num_tokens: int,
    K: int,
    max_num_subseqs: int,
) -> LogicalWorkloadSize:
    """Return the synthetic LOGICAL workload sizing for a tuning combo.

    Args:
      max_num_seqs: persistent-batch capacity for this tune
        (= ``MAX_NUM_SEQS`` env from the .workload).
      max_num_tokens: q-token budget for one kernel call
        (= ``MAX_NUM_BATCHED_TOKENS`` env).
      K: kernel ``static_q_len`` (= ``chunk_prefill_size`` /
        ``RPA_KERNEL_K``).
      max_num_subseqs: this combos prefetch-array sizing
        (= ``TunableParams.max_num_subseqs``).

    Returns:
      A `LogicalWorkloadSize`.

    Validity (returns ``valid=False`` with zeros when):
      - ``max_num_subseqs < max_num_seqs``: the prefetch arrays cant
        even fit one slot per phys req.
      - ``per_phys_q > max_num_tokens``: even ONE phys reqs full
        chunked q (= ``K * m_per_phys``) overflows the q buffer.

    Sizing (when valid):
      - ``m_per_phys = max(1, max_num_subseqs // max_num_seqs - 1)``
        from the formula ``max_num_subseqs = max_num_seqs * (M + 1)``.
      - ``actual_num_seqs = max(1, min(max_num_tokens // per_phys_q,
        max_num_seqs))`` — scales the synthetic batch to fit the q
        buffer, mirroring how PREFILL builds its workload.

    Note on representativeness: the runtime ``max_num_subseqs`` is
    SMEM-driven (set from ``envs.RPA_MAX_NUM_SUBSEQS`` at deploy time),
    independent of how many phys reqs the synthetic tune workload uses.
    Scaling ``actual_num_seqs`` down for the tune doesnt under-tune
    anything that matters at deploy time.
    """
    if max_num_subseqs < max_num_seqs:
        return LogicalWorkloadSize(valid=False, actual_num_seqs=0,
                                   m_per_phys=0, per_phys_q=0,
                                   total_iters=0)

    m_per_phys = max(1, max_num_subseqs // max_num_seqs - 1)
    per_phys_q = K * m_per_phys

    if per_phys_q > max_num_tokens:
        return LogicalWorkloadSize(valid=False, actual_num_seqs=0,
                                   m_per_phys=m_per_phys,
                                   per_phys_q=per_phys_q,
                                   total_iters=0)

    actual_num_seqs = max(1, min(max_num_tokens // per_phys_q,
                                 max_num_seqs))
    return LogicalWorkloadSize(
        valid=True,
        actual_num_seqs=actual_num_seqs,
        m_per_phys=m_per_phys,
        per_phys_q=per_phys_q,
        total_iters=actual_num_seqs * m_per_phys,
    )
