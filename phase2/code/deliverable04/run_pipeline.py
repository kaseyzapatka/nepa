#!/usr/bin/env python
"""
D4 timeline pipeline — single canonical full-corpus run, in the correct order.

ONE command, nothing to remember. `04b`/`05b`/`05c` are baked in: skipping `05b` is what corrupted
CE (stale `ranking_score`). Runs on the FULL pool (all processes); writes to data/analysis/timeline/.

Usage:
    CONDA_DEFAULT_ENV=nepa python run_pipeline.py            # full 02 -> 08
    CONDA_DEFAULT_ENV=nepa python run_pipeline.py --select   # selection-only (05b -> 05 -> 05c -> 08), minutes

This is the source of truth for run order. The sharded runner `_run.py` is retired (_archived/).
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa' (CONDA_DEFAULT_ENV=nepa).")

HERE = Path(__file__).resolve().parent
PY = sys.executable

# (script, args). 08 is R and is run separately at the end.
FULL = [
    ("02_retrieve.py", ["--force"]),
    ("03_extract_candidates.py", ["--force"]),
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


def run(stages: list[tuple[str, list[str]]]) -> None:
    for script, args in stages:
        print(f"\n=== {datetime.now():%H:%M:%S}  {script} {' '.join(args)} ===", flush=True)
        rc = subprocess.run([PY, str(HERE / script)] + args).returncode
        if rc != 0:
            if script in ALLOW_FAIL:
                print(f"  ({script} exited {rc}; non-fatal, continuing)")
            else:
                raise SystemExit(f"{script} failed (exit {rc}) — stopping.")
    print(f"\n=== {datetime.now():%H:%M:%S}  08_analyze.R ===", flush=True)
    subprocess.run(["Rscript", str(HERE / "08_analyze.R")])


if __name__ == "__main__":
    select_only = "--select" in sys.argv[1:]
    run(SELECT if select_only else FULL)
    print(f"\n=== {'selection-only' if select_only else 'FULL'} pipeline complete "
          f"({datetime.now():%H:%M:%S}) ===")
