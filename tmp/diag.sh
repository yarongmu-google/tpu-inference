#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
LOG="$(mktemp "$ROOT/tmp/diag-XXXXXXXX.txt")" || exit 1
exec > >(tee "$LOG") 2>&1
echo "Diagnostic report: $LOG"
run() {
  printf '\n+'
  printf ' %q' "$@"
  printf '\n'
  "$@" || true
}
run git rev-parse --show-toplevel
run git branch --show-current
run git log -1 --format='%h %s'
run git worktree list
run git status --short --untracked-files=all
run git diff --cached --stat
for DIR in tmp/baselines/results/archives tmp/serving-comparison/results/archives tmp/workflow/local/logs; do
  run ls -lt "$DIR"
  run git ls-files -- "$DIR"
done
echo "Launcher outcomes:"
zgrep -hE 'Result archive:|Results and diagnostics:|compression failed|Launcher completed|Launcher FAILED' tmp/workflow/local/logs/*.gz | tail -30 || true
python3 - <<'PY'
import json
import shlex
import subprocess
from pathlib import Path

roots = ("tmp/baselines/results", "tmp/serving-comparison/results",
         "tmp/workflow/local/results", "workflow/local/results")
keys = ("run_id", "job_id", "phase", "exit_code", "artifact_exit_code",
        "artifacts_verified", "deleted", "resources_cleaned", "uri",
        "collection_error", "last_error")
for root in roots:
    paths = sorted(Path(root).glob("*/*/state.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths[:5]:
        print(f"\nSaved state: {path}")
        state = json.loads(path.read_text())
        print(json.dumps({k: state.get(k) for k in keys}, indent=2))
        print("Local collected files exist:", (path.parent / "collected/files").is_dir())
        uri = state.get("uri")
        project = state.get("profile", {}).get("cloud", {}).get("project")
        if not uri or not uri.startswith("gs://") or not project:
            print("GCS: UNKNOWN - saved project or storage URI missing")
            continue
        prefix = uri.rstrip("/")
        probes = (
            ["storage", "ls", prefix + "/**"],
            ["storage", "objects", "describe", prefix + "/manifest.json"],
        )
        for probe in probes:
            command = ["gcloud", "--project", project, *probe]
            print("\n+", shlex.join(command), flush=True)
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=60)
                print("Exit code:", result.returncode)
                print("stdout:", result.stdout[:12000])
                print("stderr:", result.stderr[:4000])
                if result.returncode:
                    print("Probe failed; inspect the error to distinguish "
                          "missing objects from access or network failures.")
            except (OSError, subprocess.TimeoutExpired) as error:
                print("GCS: UNKNOWN -", error)
PY
