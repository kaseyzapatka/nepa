"""D6 v2 (narrow-first) orchestrator: run n01 -> n05.

The v2 pipeline reuses existing Phase 2 artifacts as read-only inputs
(fonsi_project_inventory / fonsi_project_packets / fonsi_evidence_spans /
fonsi_document_sections / fonsi_conditions / ce_explorer_snapshot, plus the D3
review + CE-citation tables). It does not rebuild those.

The superseded v1 scripts (01/03/04/05/06/07/08/09) remain in place but are no
longer orchestrated; they will be archived to `_archived_v1/` after v2 is
validated (see phase2/plans/deliverable06.md, Definition of Done #6).

Usage:
  CONDA_DEFAULT_ENV=nepa python _run.py            # deterministic Stage A
  CONDA_DEFAULT_ENV=nepa python _run.py --use-llm  # enable the gated LLM pass (Gate 3)
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent

STEPS = (
    "n01_select_candidate_corpus.py",
    "n02_assemble_candidate_evidence.py",
    "n03_extract_candidate_facts.py",
    "n04_base_rates_and_ce.py",
    "n05_build_report_tables.py",
)


def run(script: str, *extra: str) -> None:
    command = [sys.executable, str(CODE_DIR / script), *extra]
    print("\n+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the D6 v2 narrow-first Stage A pipeline.")
    ap.add_argument("--use-llm", action="store_true", help="enable the gated LLM pass in n03 (Gate 3)")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--only", nargs="*", help="run only these step scripts (by filename)")
    args = ap.parse_args()

    for script in STEPS:
        if args.only and script not in args.only:
            continue
        if script == "n03_extract_candidate_facts.py" and args.use_llm:
            run(script, "--use-llm", "--model", args.model)
        else:
            run(script)
    print("\n[_run] D6 v2 Stage A pipeline complete.")


if __name__ == "__main__":
    main()
