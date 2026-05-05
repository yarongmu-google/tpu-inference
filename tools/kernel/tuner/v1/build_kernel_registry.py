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
    parser.add_argument("--out", default=None,
                        help="Output .kernel path. Defaults to "
                             "tmp/log/<runlog-basename>.kernel. Required when "
                             "the orchestrator wants the registry to land at "
                             "cases/<topo>/<model>/production.kernel — without "
                             "this, the sweep reads a different file than the "
                             "pipeline writes.")
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

    if args.out:
        out_path = args.out
    else:
        basename = os.path.basename(runlog).replace("tune_all_", "").replace(".txt", "")
        out_path = f"tmp/log/{basename}.kernel"

    # ---- Accumulation Logic ----
    # If the file already exists, load it so we can merge best-of-all-time.
    out_data = {
        "metadata": {
            "last_updated_from_runlog": runlog,
            "last_updated_at": datetime.now().isoformat(),
        },
        "results": {"decode": [], "mixed": [], "prefill": []}
    }
    
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                existing_data = json.load(f)
                if "results" in existing_data:
                    out_data["results"] = existing_data["results"]
        except Exception as e:
            print(f"WARN: Could not read existing {out_path}, starting fresh. ({e})", file=sys.stderr)

    for case_m, db_m in zip(case_matches, db_matches):
        case = case_m.group(1)
        case_set_id = case_m.group(2)
        db_path = db_m.group(1)
        print(f"Extracting {case} from {db_path}")

        try:
            new_winners = local_query_min_latency(db_path, case_set_id, "0")
            
            # Merge logic: keep the lowest latency per TuningKey.
            existing_list = out_data["results"].get(case, [])
            
            # Map existing by TuningKey hash
            best_map = {}
            for r in existing_list:
                # Use a sorted JSON string of the dict as a stable hash key
                tk_hash = json.dumps(r.get("tuning_key", {}), sort_keys=True)
                best_map[tk_hash] = r
                
            # Compare and update with new winners
            for r in new_winners:
                tk_hash = json.dumps(r.get("tuning_key", {}), sort_keys=True)
                new_lat = r.get("Latency", float("inf"))
                
                if tk_hash not in best_map or new_lat < best_map[tk_hash].get("Latency", float("inf")):
                    best_map[tk_hash] = r
                    
            # Rebuild the flat list
            out_data["results"][case] = list(best_map.values())
            
        except Exception as e:
            print(f"WARN: Failed to extract {case} from {db_path}: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nWrote (Accumulated): {out_path}")

if __name__ == "__main__":
    main()
