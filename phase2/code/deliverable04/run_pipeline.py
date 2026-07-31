#!/usr/bin/env python
"""
D4 timeline pipeline — single canonical full-corpus run, in the correct order.

ONE command, nothing to remember. `04b`/`05b`/`05c` are baked in: skipping `05b` is what corrupted
CE (stale `ranking_score`). Runs on the FULL pool (all processes); writes to data/analysis/timeline/.

Usage:
    CONDA_DEFAULT_ENV=nepa python run_pipeline.py             # full 02 -> 08 (02/03 parallel)
    CONDA_DEFAULT_ENV=nepa python run_pipeline.py --select    # selection-only (05b -> 05 -> 05c -> 08), minutes
    CONDA_DEFAULT_ENV=nepa python run_pipeline.py --workers 1 # force the serial 02/03 path (debug)

This is the source of truth for run order. The old sharded runner (`_run.py`) is retired (in git history).
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa' (CONDA_DEFAULT_ENV=nepa).")

HERE = Path(__file__).resolve().parent
PY = sys.executable
DEFAULT_WORKERS = min((os.cpu_count() or 4), 8)

# (script, args). 08 is R and is run separately at the end. `{workers}` placeholders in
# 02/03 are filled with --workers at run time (parallel by default; pass --workers 1 for serial).
FULL = [
    ("02_retrieve.py", ["--force", "--workers", "{workers}"]),
    ("03_extract_candidates.py", ["--force", "--workers", "{workers}"]),
    ("04_classify_candidates.py", ["--force"]),
    ("04b_calibrate.py", ["--apply"]),
    ("05b_rank.py", ["--apply"]),
    ("05_select_dates.py", []),
    ("05c_inject_ground_truth.py", ["--scope", "all"]),
    ("07_validate.py", ["--validate"]),
]
SELECT = [
    ("05b_rank.py", ["--apply"]),
    ("05_select_dates.py", []),
    ("05c_inject_ground_truth.py", ["--scope", "all"]),
]
# 07 may no-op when the gold sample is unfilled; never fatal.
ALLOW_FAIL = {"07_validate.py"}


def run(stages: list[tuple[str, list[str]]], workers: int) -> None:
    for script, args in stages:
        args = [a.format(workers=workers) for a in args]
        print(f"\n=== {datetime.now():%H:%M:%S}  {script} {' '.join(args)} ===", flush=True)
        rc = subprocess.run([PY, str(HERE / script)] + args).returncode
        if rc != 0:
            if script in ALLOW_FAIL:
                print(f"  ({script} exited {rc}; non-fatal, continuing)")
            else:
                raise SystemExit(f"{script} failed (exit {rc}) — stopping.")
    print(f"\n=== {datetime.now():%H:%M:%S}  08_create_figures.R ===", flush=True)
    subprocess.run(["Rscript", str(HERE / "08_create_figures.R")])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="D4 timeline pipeline (full or selection-only).")
    ap.add_argument("--select", action="store_true",
                    help="Selection-only: 05b -> 05 -> 05c -> 08 (minutes).")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"Worker processes for 02/03 (default {DEFAULT_WORKERS}; "
                         "1 = serial). Other stages are unaffected.")
    a = ap.parse_args()
    run(SELECT if a.select else FULL, a.workers)
    print(f"\n=== {'selection-only' if a.select else 'FULL'} pipeline complete "
          f"({datetime.now():%H:%M:%S}) ===")
