"""D6 v2 (narrow-first) orchestrator: n01 -> n08.

Runs the linear chain in numeric order (each step depends only on lower numbers):
  n01 select corpus -> n02 assemble evidence -> n03 extract facts
  -> n04 base rates + existing-CE match/bounds -> n05 mitigation & boundary (Track B)
  -> n06 CE landscape (Track C) -> n07 classify & rank (new/expand/adopt + tables)
  -> n08 analyze (R: report figures)
then phase2/reports/deliverable06.qmd embeds the n07 tables + n08 figures.

Standalone (NOT in this chain): benchmark_models.py (model selection, run once
before --use-llm), extract_ce_catalog.py (renders the CE catalog .md), and the
ce_source/candidates/bounds/embeddings/common helpers.

The superseded v1 scripts (01/03/04/05/06/07/08/09) remain in place for now;
archive to `_archived_v1/` after validation.

Usage:
  CONDA_DEFAULT_ENV=nepa python _run.py            # deterministic Stage A
  CONDA_DEFAULT_ENV=nepa python _run.py --use-llm  # enable the gated LLM pass in n03 (Gate 3)
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent

PY_STEPS = (
    "n01_select_candidate_corpus.py",
    "n02_assemble_candidate_evidence.py",
    "n03_extract_candidate_facts.py",
    "n04_base_rates_and_ce.py",
    "n05_mitigation_and_boundary.py",
    "n06_ce_landscape.py",
    "n07_classify_and_rank.py",
)
R_STEP = "n08_analyze.R"


def run(*cmd: str) -> None:
    print("\n+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the D6 v2 narrow-first pipeline (n01-n08).")
    ap.add_argument("--use-llm", action="store_true", help="enable the gated LLM pass in n03 (Gate 3)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--skip-figures", action="store_true", help="skip the n08 R figures step")
    args = ap.parse_args()

    for script in PY_STEPS:
        if script == "n03_extract_candidate_facts.py" and args.use_llm:
            run(sys.executable, CODE_DIR / script, "--use-llm", "--model", args.model)
        else:
            run(sys.executable, CODE_DIR / script)

    if not args.skip_figures:
        rscript = shutil.which("Rscript")
        if rscript:
            run(rscript, CODE_DIR / R_STEP)
        else:
            print("\n[_run] Rscript not found — skipping n08 figures "
                  "(run `Rscript phase2/code/deliverable06/n08_analyze.R` manually).")

    print("\n[_run] D6 v2 pipeline complete (n01-n08). "
          "Render phase2/reports/deliverable06.qmd for the report.")


if __name__ == "__main__":
    main()
