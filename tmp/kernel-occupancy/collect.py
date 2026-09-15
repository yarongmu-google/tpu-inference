"""Collect the latest saved run into a small report named after the run."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'payload'))
from collector import main

if __name__ == '__main__':
    raise SystemExit(main())
