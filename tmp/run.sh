#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: bash tmp/run.sh IMAGE@sha256:DIGEST [SCRIPT] [RESUME_MOUNT_PATH]" >&2
  exit 2
fi
IMAGE="$1"
SCRIPT="${2:-tests/e2e/benchmarking/inferencex/qwen3.5/sweep.sh}"
RESUME_DIR="${3:-}"
[[ "$IMAGE" =~ ^[a-zA-Z0-9._:/-]+@sha256:[a-f0-9]{64}$ ]] || { echo "Use an immutable image digest" >&2; exit 2; }
[[ "$SCRIPT" =~ ^[a-zA-Z0-9_./-]+$ && "$SCRIPT" != /* && "/$SCRIPT/" != */../* ]] || { echo "Invalid script path" >&2; exit 2; }
[[ -z "$RESUME_DIR" || "$RESUME_DIR" =~ ^/[a-zA-Z0-9_./-]+$ ]] || { echo "Invalid mounted resume path" >&2; exit 2; }
command -v cdk >/dev/null || { echo "cdk must be installed and configured" >&2; exit 2; }
cd "$SCRIPT_DIR"
# The image contains the reviewed code. CDK supplies the persistent output mount.
# Confirm recipe discovery and mount injection with the installed CDK version.
exec cdk job create v7-8-meta "IMAGE=$IMAGE" "SCRIPT=$SCRIPT" "RESUME_DIR=$RESUME_DIR"
