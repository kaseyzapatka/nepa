"""D2 orchestrator — deterministic, gold-free, API-free stage (runnable now).

Runs the foundation that the POC spike + Gate 1/2 need:
    00_resolve_framework_regime  ->  01_build_d2_inventory

Downstream stages are gated and built separately (see notes/deliverable02/runbook.md):
  - 02_extract_significance  : needs the 30-doc spike + LLM-budget go-ahead
  - 03_build_gold_set_queue  : produces the labeling worksheet (gold is hand-coded)
  - 04_extract_eis_significance / 05_validate / 06_analyze : gated on gold + frozen schema

Run:  CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable02/_run.py
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

STAGE = ["00_resolve_framework_regime.py", "01_build_d2_inventory.py"]


def main() -> None:
    for script in STAGE:
        print(f"\n{'='*70}\n{script}\n{'='*70}")
        runpy.run_path(str(HERE / script), run_name="__main__")
    print("\nDeterministic stage complete. Next: eyeball "
          "output/deliverable02/corpus_membership_review.csv (Gate 1/2).")


if __name__ == "__main__":
    main()
