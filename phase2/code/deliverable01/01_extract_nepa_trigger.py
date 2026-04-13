import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# DELIVERABLE 1: NEPA TRIGGERED — Trigger Classification
# --------------------------
# Four-tier classification of what triggered NEPA review per project:
#   Tier 1 — Metadata heuristics (agency → trigger type)
#   Tier 2 — Regex on first 30 pages (DuckDB scan)
#   Tier 3 — SetFit sentence classifier (recommended) or spaCy NER
#   Tier 4 — Claude Haiku fallback for low-confidence cases
#
# [SELF-CONTAINED] — requires only projects_combined.parquet and CE/EA/EIS pages.
#
# Usage:
#   python 01_extract_nepa_trigger.py --sample 50   # test on 50 projects
#   python 01_extract_nepa_trigger.py               # full run
#
# Output: data/analysis/nepa_trigger/projects_nepa_trigger.parquet

# TODO: Implement — see phase2/plans/deliverable01.md for full specification
