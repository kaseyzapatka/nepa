#!/usr/bin/env python
"""
D5 / 02 — Build normalized CE category codes  [SELF-CONTAINED]

"What types of CEs were used" is already structured metadata: every CE document in
phase2/data/processed/ce/documents.parquet carries a `ce_category` array of code+description
strings, e.g. ["B5.1"], ["A9, B3.6"], ["516 DM 11.9, D. Rangeland Management ..."].

This script explodes that array to one row per (project_id, normalized code), tags the agency
schedule the code belongs to, and attaches a short human-readable description for the codes that
matter. Fast (metadata only, no page text).

Output: phase2/data/analysis/deliverable05/ce_categories.parquet
Grain : one row per (project_id, code_norm)
Cols  : project_id, code_raw, code_norm, schedule, code_description, ce_categories_extraction_run_at

Usage: python phase2/code/deliverable05/02_build_ce_categories.py
"""

import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

import re
from pathlib import Path
from datetime import datetime, timezone

import duckdb
import pandas as pd

CE_DOCS_PATH = Path("phase2/data/processed/ce/documents.parquet")
OUTPUT_PATH = Path("phase2/data/analysis/deliverable05/ce_categories.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# --- Code normalization patterns ------------------------------------------------------------
# DOE: 10 CFR 1021 Subpart D, Appendices A & B  -> leading token like A9, B5.1, B1.31
DOE_RE = re.compile(r"^\s*([AB]\d+(?:\.\d+)?)")
# DOI / BLM: 516 DM 11 (and 516 DM 6) -> token like "516 DM 11.9"
DOI_RE = re.compile(r"^\s*(516\s*DM\s*\d+(?:\.\d+)?)", re.IGNORECASE)
# Energy Policy Act of 2005, Section 390 (oil/gas; fossil contrast)
EPACT_RE = re.compile(r"(section\s*390|energy\s*policy\s*act\s*of\s*2005)", re.IGNORECASE)

# --- Short descriptions for the codes that drive the analysis -------------------------------
# Source: DOE NEPA categorical exclusions, 10 CFR 1021 Subpart D, Appendices A & B.
# Curated for the verified high-frequency / ARRA-relevant codes; uncurated codes fall back to
# the code string itself. Verify/extend against the current 10 CFR 1021 text before publication.
DOE_DESC = {
    "B5.1": "Actions to conserve energy or water",
    "B1.3": "Routine maintenance activities",
    "B3.6": "Small-scale research, development & demonstration projects",
    "B3.1": "Site characterization and environmental monitoring",
    "A9":  "Information gathering, data analysis & document preparation",
    "A1":  "Technical/financial assistance, advice, training & education",
    "A11": "Technical advice and planning assistance",
    "B1.31": "Installation of fences, gates, and signs",
}


def normalize(raw: str):
    """Return (code_norm, schedule, description) for one ce_category element, or (None, ...)."""
    if raw is None:
        return None, None, None
    s = str(raw).strip()
    if s == "" or s.lower() == "nan":
        return None, None, None

    m = DOE_RE.match(s)
    if m:
        code = m.group(1).upper()
        return code, "DOE (10 CFR 1021)", DOE_DESC.get(code, code)

    m = DOI_RE.match(s)
    if m:
        code = re.sub(r"\s+", " ", m.group(1)).upper().replace("DM", "DM")
        return code, "DOI (516 DM 11)", "DOI/BLM departmental categorical exclusion"

    if EPACT_RE.search(s):
        return "EPAct §390", "EPAct 2005 §390", "Energy Policy Act of 2005 §390 (oil & gas)"

    return None, None, None  # long-tail / uncoded


def main():
    run_at = datetime.now(timezone.utc).isoformat()
    con = duckdb.connect()

    # Pull project_id + ce_category as python lists. fetchdf turns VARCHAR[] into list objects.
    # NB: project_id is stored as STRUCT("value" VARCHAR) in the processed documents file, so we
    # extract .value back to a plain string.
    df = con.execute(f"""
        SELECT project_id.value AS project_id, ce_category
        FROM read_parquet('{CE_DOCS_PATH.as_posix()}')
        WHERE ce_category IS NOT NULL AND len(ce_category) > 0
    """).fetchdf()
    print(f"Loaded {len(df):,} CE document rows with a ce_category array")

    rows = []
    for project_id, cats in zip(df["project_id"], df["ce_category"]):
        if cats is None:
            continue
        for element in cats:
            # one array element can pack several codes: "A9, B3.6, B3.15"
            for token in re.split(r"[;,]", str(element)):
                code_norm, schedule, desc = normalize(token)
                if code_norm is not None:
                    rows.append((project_id, token.strip()[:200], code_norm, schedule, desc))

    out = (
        pd.DataFrame(rows, columns=["project_id", "code_raw", "code_norm", "schedule",
                                    "code_description"])
        .drop_duplicates(subset=["project_id", "code_norm"])
        .reset_index(drop=True)
    )
    out["ce_categories_extraction_run_at"] = run_at
    out.to_parquet(OUTPUT_PATH, index=False)

    print(f"Wrote {len(out):,} (project_id, code) rows to {OUTPUT_PATH}")
    print(f"  distinct projects with >=1 normalized code: {out['project_id'].nunique():,}")
    print("\nTop 15 normalized codes by project count:")
    print(out.groupby("code_norm")["project_id"].nunique().sort_values(ascending=False).head(15)
          .to_string())
    print("\nSchedule mix (projects):")
    print(out.drop_duplicates(["project_id", "schedule"]).groupby("schedule")["project_id"]
          .nunique().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
