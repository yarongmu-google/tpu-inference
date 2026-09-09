"""A small execution and storage check before submitting a larger experiment."""
import argparse
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output-dir', required=True, type=Path)
args = parser.parse_args()
args.output_dir.mkdir(parents=True, exist_ok=True)
result = {'run_id': os.environ['RUN_ID'], 'name': os.environ['RUN_NAME'], 'python': sys.version}
(args.output_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print('Result written successfully', flush=True)
