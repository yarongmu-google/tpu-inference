#!/usr/bin/env bash
# Build and check a local image; retain setup/build failures for review.
set -uo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash tmp/build_jobset_image.sh (with vllm12 active)" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p "$ROOT/tmp/vllm_logs" || exit 1
OUT="$(mktemp -d "$ROOT/tmp/vllm_logs/jobset-build-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")" || exit 1
LOG="$OUT/build.log"

run() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

(
  set -euo pipefail
  echo "Build diagnostics: $OUT"
  # Read the tool's agent instructions; do not perform other CDK actions here.
  if run cdk agent-letter > "$OUT/cdk-agent-letter.txt" 2>&1; then
    cat "$OUT/cdk-agent-letter.txt"
  else
    cat "$OUT/cdk-agent-letter.txt"
    echo "CDK agent letter could not be read; it is still needed before CDK submission."
  fi
  run docker version
  run python -VV
  BUILD_CONTEXT="$(mktemp -d "${TMPDIR:-/tmp}/vllm12-image-context-XXXXXXXX")"
  trap 'rm -rf -- "$BUILD_CONTEXT"' EXIT
  mkdir -p "$BUILD_CONTEXT/image"
  CLIENT="${INFERENCEX_REPO:-/tmp/InferenceX}"
  if [[ ! -d "$CLIENT/.git" && ! -f "$CLIENT/.git" ]]; then
    [[ -z "${INFERENCEX_REPO:-}" ]] || { echo "Configured InferenceX checkout is missing" >&2; exit 2; }
    CLIENT="$BUILD_CONTEXT/InferenceX"
    run git clone --depth 1 https://github.com/SemiAnalysisAI/InferenceX.git "$CLIENT"
  fi
  run python "$SCRIPT_DIR/prepare_jobset_image.py" "$BUILD_CONTEXT/image" "$ROOT" "$CLIENT" "$OUT"
  PREFIX="$(cat "$OUT/environment-prefix.txt")"
  REVISION="$(git rev-parse HEAD)"
  IMAGE="vllm12:topk-${REVISION:0:12}"
  printf '%s\n' "$IMAGE" > "$OUT/image-tag.txt"
  run docker build --progress=plain --platform=linux/amd64 \
    --build-arg "ENV_PREFIX=$PREFIX" --build-arg "SOURCE_REVISION=$REVISION" \
    --iidfile "$OUT/image-id.txt" -t "$IMAGE" "$BUILD_CONTEXT/image"
  run docker run --rm --network=none -e JAX_PLATFORMS=cpu "$IMAGE" \
    bash /opt/tpu-inference/tmp/jobset_smoke.sh
  echo "Local image built and CPU checks passed: $IMAGE"
) 2>&1 | tee "$LOG"
codes=("${PIPESTATUS[@]}")
result="${codes[0]}"
[[ "${codes[1]}" -eq 0 ]] || result=1
printf 'Build exit status: %s\n' "$result" | tee -a "$LOG"
printf '%s\n' "$result" > "$OUT/exit-code.txt"
printf 'Diagnostics retained at %s\n' "$OUT" | tee -a "$LOG"
printf -v stage_command 'git add -- %q' "tmp/vllm_logs/$(basename "$OUT")"
printf 'After reviewing the diagnostics: %s\n' "$stage_command" | tee -a "$LOG"
exit "$result"
