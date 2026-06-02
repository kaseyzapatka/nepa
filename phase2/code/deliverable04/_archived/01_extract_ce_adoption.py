import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# DELIVERABLE 4: TIMELINES — CE Adoption Post-FRA
# --------------------------
# Pull Federal Register notices for BLM and DOE categorical exclusion adoptions
# post-FRA (post-2023). Match to NEPATEC projects and correlate with review speed.
# Reuses federal_register.py API pattern.
#
# Usage:
#   python 01_extract_ce_adoption.py
#
# Output: data/analysis/ce_adoption/ce_adoption_fr.parquet

# TODO: Implement — see phase2/plans/deliverable04.md for full specification
# Key reuse: federal_register.py search/match pattern
