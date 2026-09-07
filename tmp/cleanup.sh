#!/usr/bin/env bash
# Collect successful or failed JobSet attempts; retain the remote copy.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ ( $# -ne 1 && $# -ne 3 ) || "$1" != gs://*/* || "$1" == *'<'* || "$1" == *'*'* ]]; then
  echo "Usage: bash tmp/cleanup.sh gs://BUCKET/JOB_OUTPUT_PREFIX [NAMESPACE JOBSET_NAME]" >&2
  exit 2
fi
mkdir -p "$ROOT/tmp/vllm_logs/jobsets"
DEST="$(mktemp -d "$ROOT/tmp/vllm_logs/jobsets/collection-XXXXXXXX")"
echo "Collecting all available diagnostics to $DEST"
if [[ $# -eq 3 ]]; then
  mkdir -p "$DEST/kubernetes"
  if command -v kubectl >/dev/null; then
    selector="jobset.sigs.k8s.io/jobset-name=$3"
    kubectl get pods -n "$2" -l "$selector" -o 'custom-columns=NAME:.metadata.name,PHASE:.status.phase,REASON:.status.reason,MESSAGE:.status.message,WAITING:.status.containerStatuses[*].state.waiting.reason,EXIT:.status.containerStatuses[*].state.terminated.exitCode' > "$DEST/kubernetes/pods.txt" 2>&1 || true
    if kubectl get pods -n "$2" -l "$selector" -o name > "$DEST/kubernetes/pod_names.txt" 2> "$DEST/kubernetes/discovery.log"; then
      while IFS= read -r pod; do
        [[ -n "$pod" ]] || continue
        name="${pod#pod/}"
        kubectl logs -n "$2" "$pod" --all-containers=true --timestamps=true > "$DEST/kubernetes/$name.log" 2>&1 || true
        kubectl logs -n "$2" "$pod" --all-containers=true --timestamps=true --previous=true > "$DEST/kubernetes/$name.previous.log" 2>&1 || true
        kubectl get events -n "$2" --field-selector "involvedObject.name=$name" > "$DEST/kubernetes/$name.events.txt" 2>&1 || true
      done < "$DEST/kubernetes/pod_names.txt"
    fi
  else
    echo "kubectl unavailable; cluster diagnostics could not be collected" | tee "$DEST/kubernetes/error.txt" >&2
  fi
fi
if ! gcloud storage rsync --recursive "$1" "$DEST/artifacts"; then
  echo "Download failed; partial diagnostics retained at $DEST. Remote data retained." >&2
  exit 1
fi
python3 - "$DEST" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
count = 0
for path in sorted(root.rglob("manifest.json")):
    data = json.loads(path.read_text())
    if data.get("format") != "sweep-jobset-v1":
        continue
    count += 1
    for entry in data["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid artifact path")
        artifact = path.parent / relative
        if artifact.is_symlink() or artifact.stat().st_size != entry["bytes"]:
            raise ValueError(f"Artifact size mismatch: {relative}")
        with artifact.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {relative}; retry collection after the job stops")
    print(f"{data['attempt']}: {data['state']}, exit={data['exit_code']}, {len(data['files'])} verified files")
    if data["state"] == "running":
        print("No final status was published; these are partial diagnostics, not a successful run.")
if not count:
    raise ValueError("No result manifest found; downloaded diagnostics retained for inspection")
PY
echo "Collected and verified. Remote artifacts retained. Review these files before staging:"
printf 'git add -- %q\n' "tmp/vllm_logs/jobsets/$(basename "$DEST")"
echo "After staging, commit the reviewed results locally."
