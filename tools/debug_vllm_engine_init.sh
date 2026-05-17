#!/bin/bash
# Reproduce + surface the REAL exception behind vllm's
# "Engine core initialization failed" wrapper at large
# MAX_NUM_BATCHED_TOKENS (MNB) — first known failing value on
# Llama 3 8B / TPU v7x is MNB=262144.
#
# Why this script exists: the wrapper at vllm/v1/engine/utils.py:1178
# raises `RuntimeError: Engine core initialization failed` but does
# NOT re-raise the upstream exception from the child engine-core
# process. The real exception is in the child's stderr, which vllm
# logs but the parent never surfaces. We launch vllm ourselves,
# capture full stderr to a file, then grep the Traceback out at the
# end.
#
# Usage (on TPU VM):
#   tools/debug_vllm_engine_init.sh [MNB]
#
# Examples:
#   tools/debug_vllm_engine_init.sh           # default: MNB=262144
#   tools/debug_vllm_engine_init.sh 524288    # next bisection point
#   tools/debug_vllm_engine_init.sh 131072    # known-good (sanity check)
#
# Env overrides (defaults shown):
#   MAX_MODEL_LEN=8192
#   MAX_NUM_SEQS=1000
#   MODEL=meta-llama/Meta-Llama-3-8B-Instruct
#   TIMEOUT_S=240   # wait this long before giving up + grabbing log
#   PROD_ENV=tmp/log/prod_env_prefill_heavy.sh   # sourced if exists,
#                                                  so block sizes etc. apply
#
# Output (each run gets its own timestamped dir so runs don't clobber).
# Dir lives INSIDE the repo so the script can auto-commit the artifacts
# locally (no push) — that way the failure mode of each bisection point
# is preserved in git history for offline review.
#   tmp/debug_vllm_engine_init/mnb_<MNB>_<stamp>/vllm.log     — full vllm stdout+stderr
#                                                              (read this directly to find
#                                                              the upstream JAX/XLA error)
#   tmp/debug_vllm_engine_init/mnb_<MNB>_<stamp>/meta.txt     — params + outcome + git/vllm SHAs
#   tmp/debug_vllm_engine_init/mnb_<MNB>_<stamp>/script.log   — this script's own
#                                                               stdout+stderr (every echo,
#                                                               source error, missing-cmd
#                                                               crash); always present even
#                                                               when vllm.log is missing

# NOT `set -e`: we EXPECT vllm to fail; -e would abort right after
# the failure instead of letting us extract it.
set -uo pipefail

MNB="${1:-262144}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1000}"
MODEL="${MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
TIMEOUT_S="${TIMEOUT_S:-240}"
PROD_ENV="${PROD_ENV:-tmp/log/prod_env_prefill_heavy.sh}"
# gpu_memory_utilization is the fraction of HBM vllm reserves for
# weights+KV cache. The remaining (1-util) is headroom for XLA compile
# scratch and other runtime allocations. vllm's TPU default is ~0.92
# (94.75 GiB total -> 87.17 GiB cap), leaving 7.58 GiB headroom — not
# enough for the q_len=262144 bucket's 9.02 GiB HLO temp. 0.9 leaves
# ~9.5 GiB headroom (just fits 9.02 with ~0.5 GiB margin); lower if
# probing even larger MNB. Override:  GPU_MEM_UTIL=0.85 tools/...
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"
# When SKIP_BUCKET_AUTOGEN=1, tpu_inference's runner skips its default
# bucket auto-generation (16, 32, 64, ..., MNB) and uses ONLY the
# buckets passed via vllm's --additional-config compilation_sizes.
# Useful for fixed-shape benchmarks where the auto-buckets are pure
# waste (compile time + persistent compiled-program HBM).
# Implemented in tpu_inference/runner/tpu_runner.py + tpu_inference/envs.py.
SKIP_BUCKET_AUTOGEN="${SKIP_BUCKET_AUTOGEN:-0}"
# COMPILATION_SIZES: comma-separated bucket sizes vllm pre-compiles
# when SKIP_BUCKET_AUTOGEN=1. Defaults to just MNB (the workload's
# only shape). Override for multi-bucket testing, e.g.
# COMPILATION_SIZES=131072,262144 to compile both sides of a transition.
COMPILATION_SIZES="${COMPILATION_SIZES:-$MNB}"

STAMP="$(date +%Y%m%d_%H%M%S)"
# In-repo path so the final auto-commit step can stage it. The repo's
# .gitignore for tmp/bench_*/**/vllm.log doesn't match tmp/debug_*/
# so the full log is committable.
OUT_DIR="tmp/debug_vllm_engine_init/mnb_${MNB}_${STAMP}"
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/vllm.log"
META="$OUT_DIR/meta.txt"
SCRIPT_LOG="$OUT_DIR/script.log"

# Redirect this script's own stdout+stderr through tee so script.log
# captures every echo, source-time failure, missing-cmd error, etc.
# If vllm itself never gets launched (e.g., `source $PROD_ENV` crashes
# under `set -u`), vllm.log will be missing — but script.log will still
# have the actual cause. The user also sees output live in the terminal
# because of `tee`, not a bare `>`.
exec > >(tee -a "$SCRIPT_LOG") 2>&1

echo "=== vllm engine init debug ==="
echo "MNB:                  $MNB"
echo "MAX_MODEL_LEN:        $MAX_MODEL_LEN"
echo "MAX_NUM_SEQS:         $MAX_NUM_SEQS"
echo "MODEL:                $MODEL"
echo "GPU_MEM_UTIL:         $GPU_MEM_UTIL"
echo "SKIP_BUCKET_AUTOGEN:  $SKIP_BUCKET_AUTOGEN"
if [ "$SKIP_BUCKET_AUTOGEN" = "1" ]; then
    echo "COMPILATION_SIZES:    $COMPILATION_SIZES"
fi
echo "TIMEOUT_S:            $TIMEOUT_S"
echo "OUT_DIR:              $OUT_DIR"
echo

# Source the prod env first (sets kernel block sizes etc.), THEN
# override the value we're testing so it wins regardless of what
# the prod env had.
if [ -f "$PROD_ENV" ]; then
    echo "Sourcing prod env: $PROD_ENV"
    set -a
    # shellcheck disable=SC1090
    source "$PROD_ENV"
    set +a
else
    echo "(no prod env at $PROD_ENV — using vllm + tpu_inference defaults)"
fi
export MAX_NUM_BATCHED_TOKENS="$MNB"
# Pin LPTT == MNB so a separate LPTT-side guard can't be the cause —
# we want to isolate the MNB cap, not stack two unknowns.
export LONG_PREFILL_TOKEN_THRESHOLD="$MNB"

# Wire the fixed-workload bucket-skip if requested.
# - Export SKIP_BUCKET_AUTOGEN so the vllm child + the EngineCore
#   grandchild inherit it (tpu_inference.envs reads os.environ).
# - Build the --additional-config JSON only when we actually want to
#   pin compilation_sizes; otherwise vllm uses its defaults and the
#   tpu_inference auto-bucket path runs as before.
EXTRA_VLLM_ARGS=()
export SKIP_BUCKET_AUTOGEN
if [ "$SKIP_BUCKET_AUTOGEN" = "1" ]; then
    EXTRA_VLLM_ARGS+=(--additional-config \
        "{\"compilation_sizes\":[$COMPILATION_SIZES]}")
fi

echo "=== Launching vllm serve ==="
echo "(full stdout+stderr -> $LOG)"
vllm serve "$MODEL" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MNB" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    "${EXTRA_VLLM_ARGS[@]+"${EXTRA_VLLM_ARGS[@]}"}" \
    > "$LOG" 2>&1 &
VLLM_PID=$!
echo "vllm PID: $VLLM_PID"

# Poll for one of three outcomes:
#   (a) wrapper logs "Engine core initialization failed"  → got our cause
#   (b) vllm V1 logs "Application startup complete"      → MNB actually OK
#   (c) timeout (vllm hung)                              → also informative
# Note on (b): vllm V1 (commit c51df4300+) emits the uvicorn-style
# "Started server process" + "Application startup complete" lines via
# uvicorn's INFO logger, NOT the bare "Uvicorn running on" string. The
# older string never appears in vllm V1 logs, so the previous regex
# never matched a successful start — runs timed out with `outcome=timeout`
# even when the engine was healthy.
echo "=== Waiting up to ${TIMEOUT_S}s for outcome ==="
WAITED=0
OUTCOME="timeout"
while [ "$WAITED" -lt "$TIMEOUT_S" ]; do
    if grep -q "Engine core initialization failed" "$LOG" 2>/dev/null; then
        OUTCOME="engine_init_failed"
        break
    fi
    if grep -q "Application startup complete" "$LOG" 2>/dev/null; then
        OUTCOME="started"
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        OUTCOME="process_exited"
        break
    fi
    sleep 3
    WAITED=$((WAITED + 3))
done
echo "Outcome: $OUTCOME (after ${WAITED}s)"

# Kill the vllm process group. vllm forks worker processes; just
# killing the leader leaks them and they keep the TPU attached,
# which breaks the next run. Walk parent → group → kill -9 fallback.
echo "=== Tearing down vllm ==="
pkill -P "$VLLM_PID" 2>/dev/null || true
kill "$VLLM_PID" 2>/dev/null || true
sleep 2
pkill -9 -P "$VLLM_PID" 2>/dev/null || true
kill -9 "$VLLM_PID" 2>/dev/null || true

# NOTE: we used to extract a failure.txt here via grep. Removed —
# vllm.log is committed locally and we read it directly (the grep
# regex was fragile because vllm prefixes every child-process line
# with "(EngineCore pid=...) ERROR ..." and the extraction missed
# the real upstream exception more often than it helped).

# Write meta.txt: same shape as the sweep's per-combo meta.txt so any
# future tooling that walks tmp/bench_*/**/meta.txt can also walk
# tmp/debug_vllm_engine_init/**/meta.txt with the same parser.
GIT_COMMIT=$(git rev-parse HEAD 2>/dev/null || echo "?")
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "?")
# vllm commit: best-effort from the installed package; falls back to
# `pip show` if the import path lookup fails. "?" if neither works.
VLLM_COMMIT=$(python3 -c "import vllm, os; print(open(os.path.join(os.path.dirname(vllm.__file__), '..', 'commit.txt')).read().strip())" 2>/dev/null \
    || python3 -c "import vllm; print(getattr(vllm, '__commit__', '?'))" 2>/dev/null \
    || echo "?")
{
    echo "stamp=$STAMP"
    echo "mnb=$MNB"
    echo "max_model_len=$MAX_MODEL_LEN"
    echo "max_num_seqs=$MAX_NUM_SEQS"
    echo "model=$MODEL"
    echo "gpu_memory_utilization=$GPU_MEM_UTIL"
    echo "skip_bucket_autogen=$SKIP_BUCKET_AUTOGEN"
    if [ "$SKIP_BUCKET_AUTOGEN" = "1" ]; then
        echo "compilation_sizes=$COMPILATION_SIZES"
    fi
    echo "timeout_s=$TIMEOUT_S"
    echo "prod_env=$PROD_ENV"
    echo "outcome=$OUTCOME"
    echo "waited_s=$WAITED"
    echo "git_branch=$GIT_BRANCH"
    echo "git_commit=$GIT_COMMIT"
    echo "vllm_commit=$VLLM_COMMIT"
    echo "out_dir=$OUT_DIR"
    # Replay any tpu_inference env vars that were active when the
    # test ran (the script may have sourced them from PROD_ENV).
    # Useful for re-running this exact configuration later.
    for v in MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS MAX_MODEL_LEN BLOCK_SIZE \
             LONG_PREFILL_TOKEN_THRESHOLD RPA_P_BLOCK_SIZES \
             RPA_D_BLOCK_SIZES RPA_M_BLOCK_SIZES RPA_KERNEL_K; do
        # Indirect expansion: "${!v}" → the value of the var named in v.
        eval "val=\${$v:-}"
        # Lowercase via `tr` rather than `${v,,}` (bash 4+); the TPU
        # VM has bash 5 but local dry-run on macOS uses bash 3.2 and
        # would silently fail this line, hiding bugs in the dry-run.
        [ -n "$val" ] && echo "$(echo "$v" | tr 'A-Z' 'a-z')=$val"
    done
} > "$META"

# Print the run summary BEFORE the git commit. script.log is
# captured by `tee -a`, but the commit step snapshots the file at
# `git add` time — any output AFTER the add (commit hash, push
# results) won't be in the committed script.log. By printing the
# Full log: / Meta: / Outcome: lines BEFORE the commit, the
# committed script.log captures the final summary; only the
# commit's own diagnostic lines are lost, and those are
# recoverable from outside (git log -1, origin/sll rev).
echo
echo "Full log:  $LOG"
echo "Meta:      $META"
echo "Script:    $SCRIPT_LOG"
echo "Outcome:   $OUTCOME"
echo

# Auto-commit + push. Path-restricted commit: only our three files,
# even if the working tree has other staged or unstaged changes
# from concurrent work. The `|| true` on each step keeps the
# script exit code clean if git fails (not in a repo, hooks, etc.).
echo "=== Commit (NO_PUSH=1 to skip push at end) ==="
if git rev-parse --git-dir >/dev/null 2>&1; then
    # NOTE: -f is REQUIRED. The repo's top-level .gitignore has a
    # global '*.log' rule, so without -f, `git add` silently skips
    # vllm.log + script.log and only meta.txt makes it into the
    # commit. -f is safe here because the path list is narrow and
    # explicit — no risk of sweeping in unrelated ignored files.
    # The path-restricted `git commit -- <paths>` below provides
    # the same guarantee at commit time.
    git add -f "$META" "$LOG" "$SCRIPT_LOG" 2>/dev/null || true
    if git commit -m "[Debug] vllm engine init: MNB=$MNB outcome=$OUTCOME stamp=$STAMP" \
        -- "$META" "$LOG" "$SCRIPT_LOG" 2>/dev/null; then
        echo "Committed locally as $(git rev-parse --short HEAD)"
        # Auto-push (opt-out via NO_PUSH=1). Pull --rebase first
        # because the push race has bitten this repo before; on
        # conflict, abort the rebase and keep the local commit
        # rather than crash. User syncs manually in that case.
        if [ "${NO_PUSH:-0}" != "1" ]; then
            echo "=== Pushing to origin (pull --rebase first) ==="
            if git pull --rebase 2>&1; then
                if git push 2>&1; then
                    echo "Pushed OK."
                else
                    echo "WARN: push failed. Local commit kept; sync manually."
                fi
            else
                echo "WARN: pull --rebase failed. Local commit kept; sync manually."
                git rebase --abort 2>/dev/null || true
            fi
        else
            echo "(NO_PUSH=1 — skipping push; commit is local only)"
        fi
    else
        echo "(commit skipped — nothing new to commit, or git refused)"
    fi
else
    echo "(not in a git repo — skipping auto-commit)"
fi
