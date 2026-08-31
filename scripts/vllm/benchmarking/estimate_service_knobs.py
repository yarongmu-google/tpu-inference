# Estimate vLLM scheduler knobs from chip + model arithmetic.
#
# The sweep axes should be ANCHORED by first-principles estimates and
# then swept over neighbors for estimation error - not hand-picked
# octaves (the origin/tune search space missed a peak by 16x precisely
# because its liberal-but-arbitrary grid stopped early).
#
# The causal chain (vLLM semantics):
#
#   1. KV pool (bytes/device) = HBM x gpu_util - weights/device
#                               - activation reserve.
#      The pool caps TOKENS IN FLIGHT (stored KV), and it is the same
#      total under DP or TP attention: DP keeps full per-token KV on
#      one device but splits tokens P ways; TP shards each token's KV
#      P ways but stores every token everywhere. (DP additionally
#      avoids the KV-head < P replication waste, which is why the
#      TP-MoE lines force attn_dp_size = P.)
#   2. max-num-seqs is what the pool constrains, THROUGH max-model-len:
#        MNS_worst    = pool_tokens / max_model_len      (all seqs full)
#        MNS_expected = pool_tokens / (in + out/2)       (steady state:
#      uniformly aged decodes average half the output written).
#      Above MNS_expected the scheduler preempts/queues - throughput
#      does not OOM, it collapses.
#   3. max-num-batched-tokens is NOT a KV knob - it is the per-step
#      compute budget. Its hard bounds:
#        MNB >= input_len   (a prompt must prefill in one step unless
#                            you want chunked prefill in the picture)
#        MNB >= MNS         (a decode step batches one token per seq;
#                            smaller MNB splits the decode step)
#      Its IDEAL sits at the roofline crossover: a step must stream
#      the activated weights from HBM once regardless of how many
#      tokens ride it (with MNS x top_k >> num_experts the whole
#      shard streams), so step time = max(C(t), M) with M fixed.
#      Tokens are FREE until compute C(t) catches the stream M:
#        MNB* = W_bytes x peak_flops / (hbm_bw x 2 x active_params)
#      (per-device W/P and per-device flops both scale 1/P, so P
#      cancels). Below MNB* the stream is underused; above it steps
#      lengthen and TPOT degrades. gpu_util plays no role here - it
#      is a capacity knob and this is a bandwidth law. Pass
#      --peak-tflops and --hbm-gbps to compute it (hardware rates
#      are never baked in); without them the estimate falls back to
#      the max of the two lower bounds, and the sweep probes upward.
#
# Emits shell-eval-able candidate lists: each anchor and its x2 / x0.5
# neighbors (clamped to the hard bounds), rounded to scheduler-friendly
# multiples.
#
# Usage:
#   python estimate_service_knobs.py --model Qwen/Qwen3.5-397B-A17B-FP8 \
#       --in-len 1024 --out-len 8192 [--hbm-gb N] [--emit shell]
#
# HBM per device comes from --hbm-gb, else the HBM_GB env, else a jax
# device query (grabs the TPU briefly - do not run while a server is
# up).

import argparse
import os
import sys


def _text_config(model: str):
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    # multimodal checkpoints nest the language model config
    return getattr(cfg, "text_config", None) or cfg


def _weight_bytes(cfg, weight_byte: float) -> int:
    """Parameter bytes from config arithmetic (MoE experts dominate).

    weight_byte: bytes/param for the bulk weights (1 for fp8; the
    per-channel scales and the bf16 norms/router are counted at their
    own widths). Good to ~5%, which is enough for a KV-pool estimate;
    override with --weight-gb when exactness matters.
    """
    d = cfg.hidden_size
    layers = cfg.num_hidden_layers
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", d // heads)
    vocab = cfg.vocab_size
    experts = getattr(cfg, "num_experts", 0) or getattr(
        cfg, "num_routed_experts", 0)
    i_moe = getattr(cfg, "moe_intermediate_size", 0)
    i_dense = getattr(cfg, "intermediate_size", 0)

    attn = (d * heads * head_dim          # q
            + 2 * d * kv_heads * head_dim  # k, v
            + heads * head_dim * d)        # o
    if experts and i_moe:
        ffn = experts * 3 * d * i_moe + experts * d   # gate/up/down + router
    else:
        ffn = 3 * d * i_dense
    body = layers * (attn + ffn) * weight_byte
    embed = 2 * vocab * d * 2.0            # in+out embeddings, bf16
    return int(body + embed)


def _active_params(cfg) -> int:
    """Active (per-token) parameters: attention + top_k experts."""
    d = cfg.hidden_size
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", d // heads)
    attn = (d * heads * head_dim + 2 * d * kv_heads * head_dim
            + heads * head_dim * d)
    experts = getattr(cfg, "num_experts", 0) or getattr(
        cfg, "num_routed_experts", 0)
    i_moe = getattr(cfg, "moe_intermediate_size", 0)
    k_top = getattr(cfg, "num_experts_per_tok", 0)
    if experts and i_moe and k_top:
        ffn = k_top * 3 * d * i_moe + experts * d   # experts + router
    else:
        ffn = 3 * d * cfg.intermediate_size
    return cfg.num_hidden_layers * (attn + ffn)


def _kv_bytes_per_token(cfg, kv_byte: int) -> int:
    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // heads)
    return 2 * kv_heads * head_dim * cfg.num_hidden_layers * kv_byte


def _hbm_bytes_per_device(args) -> int:
    if args.hbm_gb:
        return int(args.hbm_gb * 2**30)
    env = os.environ.get("HBM_GB")
    if env:
        return int(float(env) * 2**30)
    import jax   # brief TPU grab; do not run while a server is up
    stats = jax.devices()[0].memory_stats()
    return int(stats["bytes_limit"])


def _round_to(x: float, mult: int, at_least: int) -> int:
    return max(int(round(x / mult)) * mult, at_least)


def _neighbors(anchor: int, mult: int, lo: int) -> list[int]:
    cands = {_round_to(anchor * f, mult, lo) for f in (0.5, 1.0, 2.0)}
    return sorted(cands)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--in-len", type=int, required=True)
    p.add_argument("--out-len", type=int, required=True)
    p.add_argument("--tp", type=int, default=8, help="devices")
    p.add_argument("--gpu-util", type=float, default=0.88)
    p.add_argument("--kv-byte", type=int, default=1,
                   help="KV cache bytes/elem (1 = fp8)")
    p.add_argument("--weight-byte", type=float, default=1.0,
                   help="bulk weight bytes/param (1 = fp8)")
    p.add_argument("--weight-gb", type=float, default=0.0,
                   help="override the config-arithmetic weight size")
    p.add_argument("--hbm-gb", type=float, default=0.0,
                   help="HBM per device; else HBM_GB env, else jax query")
    p.add_argument("--reserve-gb", type=float, default=4.0,
                   help="activation/workspace reserve per device")
    p.add_argument("--peak-tflops", type=float, default=0.0,
                   help="per-device peak TFLOP/s at the serving dtype "
                   "(enables the MNB roofline-crossover anchor)")
    p.add_argument("--hbm-gbps", type=float, default=0.0,
                   help="per-device HBM GB/s (enables the MNB "
                   "roofline-crossover anchor)")
    p.add_argument("--emit", choices=("human", "shell"), default="human")
    args = p.parse_args()

    cfg = _text_config(args.model)
    mml = args.in_len + args.out_len
    hbm = _hbm_bytes_per_device(args)
    weights = (int(args.weight_gb * 2**30) if args.weight_gb
               else _weight_bytes(cfg, weight_byte=args.weight_byte))
    kv_tok = _kv_bytes_per_token(cfg, kv_byte=args.kv_byte)

    pool_dev = hbm * args.gpu_util - weights / args.tp \
        - args.reserve_gb * 2**30
    if pool_dev <= 0:
        print(f"ERROR: no KV pool left (hbm={hbm/2**30:.0f}G util="
              f"{args.gpu_util} weights/dev={weights/args.tp/2**30:.1f}G "
              f"reserve={args.reserve_gb}G)", file=sys.stderr)
        return 1
    pool_tokens = int(args.tp * pool_dev / kv_tok)

    mns_worst = pool_tokens // mml
    mns_expected = int(pool_tokens / (args.in_len + args.out_len / 2))
    # MNS anchored at expected occupancy; worst case is the floor the
    # operator should know about (guaranteed-no-preemption point)
    mns_list = _neighbors(mns_expected, mult=64, lo=64)
    # MNB anchor: roofline crossover when the hardware rates are
    # given, else the max of the two hard lower bounds. The sweep's
    # per-combo skip enforces the pairwise MNB >= MNS rule.
    mnb_cross = 0
    if args.peak_tflops and args.hbm_gbps:
        active = _active_params(cfg)
        mnb_cross = int(weights * args.peak_tflops * 1e12
                        / (args.hbm_gbps * 1e9 * 2 * active))
        mnb_anchor = max(mnb_cross, args.in_len, max(mns_list))
    else:
        mnb_anchor = max(args.in_len, max(mns_list))
    mnb_list = _neighbors(mnb_anchor, mult=256, lo=max(args.in_len, 256))

    if args.emit == "shell":
        print(f'MML_LIST="{mml}"')
        print(f'MNB_LIST="{" ".join(str(x) for x in mnb_list)}"')
        print(f'MNS_LIST="{" ".join(str(x) for x in mns_list)}"')
        return 0

    print(f"model                {args.model}")
    print(f"layers/kv_heads/hd   {cfg.num_hidden_layers}/"
          f"{cfg.num_key_value_heads}/"
          f"{getattr(cfg, 'head_dim', '?')}")
    print(f"HBM/device           {hbm / 2**30:8.1f} GiB")
    print(f"weights (total)      {weights / 2**30:8.1f} GiB "
          f"-> {weights / args.tp / 2**30:.1f}/device")
    print(f"KV bytes/token       {kv_tok / 2**10:8.1f} KiB (full copy; "
          f"same total under DP or TP attention)")
    print(f"KV pool/device       {pool_dev / 2**30:8.1f} GiB "
          f"(util={args.gpu_util}, reserve={args.reserve_gb}G)")
    print(f"pool capacity        {pool_tokens:8d} tokens across "
          f"{args.tp} devices")
    print(f"max-model-len        {mml:8d} (= in {args.in_len} + out "
          f"{args.out_len}; workload-pinned)")
    print(f"MNS worst-case       {mns_worst:8d} (all seqs at full len; "
          f"no-preemption guarantee)")
    print(f"MNS expected         {mns_expected:8d} (steady state, "
          f"avg len = in + out/2)")
    print(f"MNB lower bounds     max(in_len={args.in_len}, MNS)")
    if mnb_cross:
        print(f"MNB roofline cross   {mnb_cross:8d} (compute catches "
              f"the fixed weight stream; tokens below this ride free, "
              f"beyond it TPOT pays)")
    else:
        print("MNB roofline cross   n/a (pass --peak-tflops and "
              "--hbm-gbps to compute; sweep probes upward instead)")
    print(f"MNS_LIST             {mns_list}")
    print(f"MNB_LIST             {mnb_list}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
