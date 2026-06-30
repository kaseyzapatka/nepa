"""D6 — CE coverage verification worksheet (reviewer aid; run after 07).

The 'adopt' verdicts rest on a TEXT-SIMILARITY match between each candidate action type and an
existing CE (candidate_ce_comparison, from 04) — every match is manual_verification_status =
'pending'. This builds a reviewer worksheet: per candidate category, the best-match CE (full
text, extraordinary circumstances, eCFR URL), the candidate action profile (count, agencies,
scope), and a FIRST-PASS coverage assessment to confirm against the CURRENT eCFR text.

The claude_* columns are a first read of the snapshot CE text — NOT a legal determination.
Confirm each against live eCFR before acting.

Output: output/deliverable06/ce_verification_worksheet.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa'.")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs

# First-pass coverage read of the snapshot CE text (review against current eCFR before acting).
ASSESS = {
    "transmission_upgrade": dict(
        verdict="LIKELY COVERS (qualitative)",
        bound_to_verify="TVA #17 is qualitative ('routine modification ... minor upgrade of existing "
        "transmission') with no mileage cap. Confirm the candidate rebuilds read as 'minor'; large "
        "rebuilds (>~25 mi) already fall to expand, not adopt.",
        reasoning="TVA #17 explicitly covers modification/repair/maintenance/minor upgrade of EXISTING "
        "transmission infrastructure; the CE-shaped candidates are within-ROW modify-existing lines. "
        "Cross-agency: BLM/BOR/DOE/PMA would ADOPT a TVA CE."),
    "solar": dict(
        verdict="PARTIAL — category contamination, re-examine first",
        bound_to_verify="B5.16 covers commercially available solar PV SYSTEMS. But 6 of 9 CE-shaped "
        "'solar' rows are GEN-TIE / interconnection TRANSMISSION lines (the gen-tie precedence rule). "
        "B5.16 does NOT cover gen-tie lines. Decide whether gen-ties belong in solar, transmission, or "
        "other BEFORE treating this match as adopt.",
        reasoning="B5.16 covers solar PV generation; it does not cover gen-tie transmission lines. The "
        "match is valid only for the ~3 actual solar-generation candidates."),
    "geothermal_exploration": dict(
        verdict="LIKELY COVERS",
        bound_to_verify="B3.1 site characterization is qualitative; confirm exploratory drilling "
        "depth / well count is within DOE B3.1 scope.",
        reasoning="DOE B3.1 (site characterization & environmental monitoring) covers exploratory "
        "drilling / geophysical survey — a close match to geothermal exploration."),
    "temporary_resource_assessment": dict(
        verdict="LIKELY COVERS",
        bound_to_verify="B3.1 site characterization is qualitative; confirm the met-tower / boring / "
        "survey scope is within it.",
        reasoning="DOE B3.1 covers temporary site characterization & monitoring — a direct match to "
        "temporary resource assessment."),
    "wind_onshore": dict(
        verdict="CONDITIONAL — turbine-count bound",
        bound_to_verify="B5.18 caps at a SMALL NUMBER (generally <=2) of commercially available wind "
        "turbines. Candidates above ~2 turbines EXCEED it -> expand, not adopt. Confirm turbine counts.",
        reasoning="DOE B5.18 covers <=2 commercially available wind turbines; only small wind adopts, "
        "larger wind farms are expand."),
}


def _agencies(series) -> str:
    vals = set()
    for v in series.dropna():
        s = str(v).strip().strip("[]").replace("'", "").replace('"', "")
        if s:
            vals.add(s)
    return "; ".join(sorted(vals))[:140]


def main() -> None:
    ensure_d6_dirs()
    f = pd.read_parquet(D6_ANALYSIS_DIR / "candidate_facts.parquet")
    ce = pd.read_parquet(D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet")
    shaped = f[f["is_ce_shaped"]].copy()
    top = ce.sort_values("retrieval_rank").groupby("candidate_category").head(1).set_index("candidate_category")

    rows = []
    for cat, g in shaped.groupby("candidate_category"):
        c = top.loc[cat].to_dict() if cat in top.index else {}
        miles, mw = g["max_miles"].dropna(), g["max_megawatts"].dropna()
        scope = []
        if len(miles):
            scope.append(f"line {miles.min():.0f}-{miles.max():.0f} mi (n={len(miles)})")
        if len(mw):
            scope.append(f"{mw.min():.0f}-{mw.max():.0f} MW (n={len(mw)})")
        a = ASSESS.get(cat, {})
        rows.append({
            "candidate_category": cat,
            "n_ce_shaped": g["project_id"].nunique(),
            "candidate_agencies": _agencies(g["lead_agency_harmonized"]),
            "candidate_scope": "; ".join(scope) or "n/a",
            "best_ce_id": c.get("structured_id", ""),
            "best_ce_agency": c.get("agency_unit", ""),
            "match_score": c.get("retrieval_score", ""),
            "ce_verification_status": c.get("manual_verification_status", ""),
            "best_ce_text": str(c.get("ce_description", ""))[:600],
            "best_ce_extraordinary_circumstances": str(c.get("extraordinary_circumstances", ""))[:300],
            "best_ce_url": c.get("canonical_source_url", ""),
            "claude_coverage_verdict": a.get("verdict", ""),
            "bound_to_verify": a.get("bound_to_verify", ""),
            "claude_reasoning": a.get("reasoning", ""),
            "reviewer_confirms_covers": "",      # for you: yes / no / partial
            "reviewer_notes": "",
        })
    out = pd.DataFrame(rows).sort_values("n_ce_shaped", ascending=False)
    path = D6_OUTPUT_DIR / "ce_verification_worksheet.csv"
    out.to_csv(path, index=False)
    print(f"[ce-verify] wrote {len(out)} categories -> {path}")
    print(out[["candidate_category", "n_ce_shaped", "best_ce_id", "claude_coverage_verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
