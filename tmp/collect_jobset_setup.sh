#!/usr/bin/env bash
# Run with vllm12 active; keep setup diagnostics even when a command fails.
set -uo pipefail
if [[ $# -ne 0 ]]; then
  echo "Usage: bash tmp/collect_jobset_setup.sh (with vllm12 active)" >&2
  exit 2
fi
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
cd "$ROOT" || exit 1
mkdir -p "$ROOT/tmp/vllm_logs" || exit 1
OUT="$(mktemp -d "$ROOT/tmp/vllm_logs/jobset-setup-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")" || exit 1
LOG="$OUT/setup.log"
: > "$LOG" || exit 1
failures=0

report() {
  printf '%s\n' "$*" | tee -a "$LOG"
}

capture() {
  local file="$1"
  shift
  local shown
  printf -v shown '%q ' "$@"
  report "+ $shown"
  "$@" 2>&1 | tee "$OUT/$file" | tee -a "$LOG"
  local codes=("${PIPESTATUS[@]}")
  report "Exit status: ${codes[0]}; output capture: ${codes[1]}, ${codes[2]}"
  if [[ "${codes[0]}" -ne 0 || "${codes[1]}" -ne 0 || "${codes[2]}" -ne 0 ]]; then
    failures=$((failures + 1))
  fi
  report ""
}

report "Setup diagnostics: $OUT"
report "Active environment: ${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-not identified; using python from PATH}}"
report ""
capture python.txt python -VV
capture packages.txt python -m pip list --format=freeze
capture source-info.json python - <<'PY_SOURCE'
import ast
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys

report = {"python_executable": sys.executable, "prefix": sys.prefix,
          "platform": platform.platform(), "machine": platform.machine(),
          "conda_environment": (Path(sys.prefix) / "conda-meta").is_dir(),
          "packages": {}}
failed = False
for name in ("vllm", "tpu_inference"):
    info = {}
    report["packages"][name] = info
    try:
        info["distribution_version"] = importlib.metadata.version(name)
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise RuntimeError("Cannot locate installed package source")
        package = Path(spec.origin).resolve().parent
        info["package_directory"] = str(package)
        version_file = package / "_version.py"
        if version_file.is_file():
            for node in ast.parse(version_file.read_text()).body:
                if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "__version__"
                        for target in node.targets):
                    try:
                        info["source_version"] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        info["source_version"] = "not a literal; inspect at build time"
        for key, args in (
                ("git_root", ["rev-parse", "--show-toplevel"]),
                ("git_revision", ["rev-parse", "HEAD"]),
                ("tracked_changes", ["status", "--porcelain", "--untracked-files=no"])):
            result = subprocess.run(args=["git", "-C", str(package), *args],
                                    capture_output=True, text=True, timeout=15)
            info[key] = result.stdout.strip() if result.returncode == 0 else None
        info["editable_source_available"] = bool(info["git_root"])
    except Exception as error:
        info["error"] = str(error)
        failed = True
print(json.dumps(report, indent=2))
sys.exit(1 if failed else 0)
PY_SOURCE
capture pip-check.txt python -m pip check
capture docker-version.txt docker version
capture cdk-help.txt cdk --help
capture cdk-create-help.txt cdk job create --help
report "Saved command output and errors to $LOG"
report "$failures setup check(s) failed. All available diagnostics were retained."
printf -v stage_command 'git add -- %q' "tmp/vllm_logs/$(basename "$OUT")"
report "After reviewing the diagnostics: $stage_command"
[[ "$failures" -eq 0 ]]
