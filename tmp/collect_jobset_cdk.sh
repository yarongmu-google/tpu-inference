#!/usr/bin/env bash
# Collect CDK documentation needed to implement submission and result retrieval.
set -uo pipefail
[[ $# -eq 0 ]] || { echo "Usage: bash tmp/collect_jobset_cdk.sh" >&2; exit 2; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" || exit 1
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)" || exit 1
mkdir -p "$ROOT/tmp/vllm_logs" || exit 1
OUT="$(mktemp -d "$ROOT/tmp/vllm_logs/jobset-cdk-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXXXX")" || exit 1
export CDK_DOC_OUTPUT="$OUT"
(
  set -euo pipefail
  echo "CDK documentation: $OUT"
  echo "+ cdk agent-letter"
  cdk agent-letter
  echo "+ collect Markdown documentation from the cloud-devkit checkout"
  python3 - <<'PY_DOCS'
import hashlib
import json
import os
from pathlib import Path
import subprocess

root = Path(os.environ.get("CDK_SOURCE_DIR", str(Path.home() / "cloud-devkit"))).resolve()
if not (root / ".claude").is_dir():
    raise SystemExit(f"CDK skills directory missing: {root / '.claude'}; set CDK_SOURCE_DIR to the checkout")
paths = {root / name for name in ("README.md", "CLAUDE.md", "AGENTS.md") if (root / name).is_file()}
for directory in (root / ".claude", root / "docs"):
    if directory.is_dir():
        paths.update(directory.rglob("*.md"))
files = []
for path in sorted(paths):
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
        continue
    content = path.read_bytes()
    relative = path.relative_to(root).as_posix()
    print(f"\n===== {relative} =====\n{content.decode('utf-8')}", flush=True)
    files.append({"path": relative, "sha256": hashlib.sha256(content).hexdigest()})
if not any(entry["path"].startswith(".claude/") for entry in files):
    raise SystemExit("No Markdown CDK skills found")
revision = subprocess.check_output(args=["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
metadata = {"source_revision": revision, "files": files}
(Path(os.environ["CDK_DOC_OUTPUT"]) / "docs-manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(f"\nCollected {len(files)} documentation files; source revision: {revision}")
PY_DOCS
) 2>&1 | tee "$OUT/cdk-docs.log"
codes=("${PIPESTATUS[@]}")
result="${codes[0]}"
[[ "${codes[1]}" -eq 0 ]] || result=1
printf '%s\n' "$result" > "$OUT/exit-code.txt"
printf 'Exit status: %s; documentation/errors retained at %s\n' "$result" "$OUT"
printf 'Review before sharing: git add -- %q\n' "tmp/vllm_logs/$(basename "$OUT")"
exit "$result"
