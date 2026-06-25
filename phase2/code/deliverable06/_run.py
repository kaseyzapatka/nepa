"""D6 v2 (narrow-first) orchestrator: 01 -> 08.

Runs the linear chain in numeric order (each step depends only on lower numbers):
  01 select corpus -> 02 assemble evidence -> 03 extract facts
  -> 04 base rates + existing-CE match/bounds -> 05 mitigation & boundary (Track B)
  -> 06 CE landscape (Track C) -> 07 classify & rank (new/expand/adopt + tables)
  -> 08 analyze (R: report figures)
then phase2/reports/deliverable06.qmd embeds the 07 tables + 08 figures.

Standalone (NOT in this chain): benchmark_models.py (model selection, run once
before --use-llm), extract_ce_catalog.py (renders the CE catalog .md), and the
ce_source/candidates/bounds/embeddings/common helpers.

The superseded v1 scripts (01/03/04/05/06/07/08/09) remain in place for now;
archive to `_archived_v1/` after validation.

Usage:
  CONDA_DEFAULT_ENV=nepa python _run.py            # deterministic Stage A
  CONDA_DEFAULT_ENV=nepa python _run.py --use-llm  # OLD narrow facts LLM pass in 03_extract_candidate_facts (NOT the new 37-field enrichment)
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
    "01_select_candidate_corpus.py",
    "02_assemble_candidate_evidence.py",
    "03_extract_candidate_facts.py",
    "04_base_rates_and_ce.py",
    "05_mitigation_and_boundary.py",
    "06_ce_landscape.py",
    "09_wire_enrichment.py",   # overwrites 03/05 facts+mitigation with LLM enrichment (if present)
    "07_classify_and_rank.py",
)
ENRICHMENT = Path(__file__).resolve().parent.parent.parent / "data" / "analysis" / "deliverable06" / "fonsi_enrichment.parquet"
R_STEP = "08_analyze.R"


def run(*cmd: str) -> None:
    print("\n+ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the D6 v2 narrow-first pipeline (01-08).")
    ap.add_argument("--use-llm", action="store_true",
                    help="OLD narrow facts LLM pass in 03_extract_candidate_facts (NOT the new enrichment)")
    ap.add_argument("--model", default="claude-sonnet-4-6")
    ap.add_argument("--skip-figures", action="store_true", help="skip the 08 R figures step")
    args = ap.parse_args()

    if args.use_llm:
        print("[_run] NOTE: --use-llm runs the OLD narrow facts pass in 03_extract_candidate_facts.py "
              "(only action_definition / mitigation_dependence / mitigation_summary). The new 37-field "
              "enrichment pass is 03_enrich_llm.py — standalone, NOT yet wired into this pipeline.")

    for script in PY_STEPS:
        if script == "09_wire_enrichment.py" and not ENRICHMENT.exists():
            print(f"\n[_run] skipping 09_wire_enrichment.py — no enrichment at {ENRICHMENT.name} "
                  "(run 03_enrich_llm.py first to LLM-back the report). Using deterministic 03/05 facts.")
            continue
        if script == "03_extract_candidate_facts.py" and args.use_llm:
            run(sys.executable, CODE_DIR / script, "--use-llm", "--model", args.model)
        else:
            run(sys.executable, CODE_DIR / script)

    if not args.skip_figures:
        rscript = shutil.which("Rscript")
        if rscript:
            run(rscript, CODE_DIR / R_STEP)
        else:
            print("\n[_run] Rscript not found — skipping 08 figures "
                  "(run `Rscript phase2/code/deliverable06/08_analyze.R` manually).")

    print("\n[_run] D6 v2 pipeline complete (01-08). "
          "Render phase2/reports/deliverable06.qmd for the report.")


if __name__ == "__main__":
    main()
