import argparse
import json
import os
import re
import sys
from datetime import datetime

from tools.kernel.tuner.v1.inspect_result_cli import local_query_min_latency

def main():
    parser = argparse.ArgumentParser(description="Extract kernel tuning winners into a .kernel file.")
    parser.add_argument("runlog", nargs="?", help="Path to runlog. Defaults to latest tmp/log/tune_all_*.txt")
    args = parser.parse_args()

    runlog = args.runlog
    if not runlog:
        import glob
        logs = sorted(glob.glob("tmp/log/tune_all_*.txt"), key=os.path.getmtime)
        if not logs:
            print("No tune_all_*.txt runlogs found in tmp/log/", file=sys.stderr)
            sys.exit(1)
        runlog = logs[-1]

    if not os.path.isfile(runlog):
        print(f"Runlog not found: {runlog}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading runlog: {runlog}")
    with open(runlog) as f:
        content = f.read()

    case_matches = list(re.finditer(r"Tuning ([a-z]+) \(case_set_id=([^)]+)\)", content))
    db_matches = list(re.finditer(r"Database initialized at (/tmp/kernel_tuner_run_[^\s]+)", content))

    if len(case_matches) != len(db_matches):
        print(f"Mismatch: {len(case_matches)} case headers vs {len(db_matches)} DB paths.", file=sys.stderr)
        sys.exit(1)
    
    if not case_matches:
        print("No phases found in runlog.", file=sys.stderr)
        sys.exit(1)

    # Structure: 
    # {
    #   "metadata": {"runlog": runlog, "timestamp": ...},
    #   "results": {
    #       "prefill": [ { tuning_key: ..., tunable_params: ..., latency: ... }, ... ],
    #       "decode": [...],
    #       "mixed": [...]
    #   }
    # }
    
    out_data = {
        "metadata": {
            "source_runlog": runlog,
            "extracted_at": datetime.now().isoformat(),
        },
        "results": {}
    }

    for case_m, db_m in zip(case_matches, db_matches):
        case = case_m.group(1)
        case_set_id = case_m.group(2)
        db_path = db_m.group(1)
        print(f"Extracting {case} from {db_path}")

        try:
            winners = local_query_min_latency(db_path, case_set_id, "0")
            out_data["results"][case] = winners
        except Exception as e:
            print(f"WARN: Failed to extract {case} from {db_path}: {e}", file=sys.stderr)

    # Generate output path
    basename = os.path.basename(runlog).replace("tune_all_", "").replace(".txt", "")
    out_path = f"tmp/log/{basename}.kernel"

    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nWrote: {out_path}")

if __name__ == "__main__":
    main()
