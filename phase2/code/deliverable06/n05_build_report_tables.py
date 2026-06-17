"""D6 v2 — n05: build the narrow, client-facing report tables.

Produces the small set of CSVs that `phase2/reports/deliverable06.qmd` reads:
  - d6_shortlist.csv          ranked candidate shortlist + recommendation
  - d6_comparison_table.csv   single at-a-glance comparison
  - d6_candidate_evidence_<category>.csv  representative profile projects w/ cited limits

These are review aids, not legal-sufficiency determinations.
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now

BASE = D6_ANALYSIS_DIR / "candidate_base_rates.parquet"
FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
CE = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
SHORTLIST = D6_OUTPUT_DIR / "d6_shortlist.csv"
COMPARISON = D6_OUTPUT_DIR / "d6_comparison_table.csv"

PROFILE_MIN = 5  # profile subset projects needed to stand alone


def limit_summary(values: pd.Series) -> str:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return ""
    return f"n={len(v)} median={round(v.median(), 1)} max={round(v.max(), 1)}"


def recommend(role: str, n_profile: int, n_fonsi: int) -> tuple[str, str]:
    if role == "contrast":
        return "contrast", "Kept as a contrast case: large footprint + case-specific wildlife mitigation typically disqualify a CE."
    if n_profile >= PROFILE_MIN:
        return "profile", f"{n_profile} CE-shaped FONSI projects with bounded limits — profile for CATF review."
    if n_fonsi >= PROFILE_MIN:
        return "profile_broadened", f"CE-shaped subset thin ({n_profile}); profile the broader {n_fonsi}-project set with siting as a reported dimension."
    return "thin_fold_or_drop", f"Only {n_fonsi} observed FONSI projects — too thin to stand alone; fold into a related candidate or drop."


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    base = pd.read_parquet(BASE)
    facts = pd.read_parquet(FACTS)
    ce = pd.read_parquet(CE) if CE.exists() else pd.DataFrame()

    base = base.set_index("candidate_category")

    # multi-category projects: a project can map to >1 candidate (e.g. solar + its
    # gen-tie transmission). Track so report counts aren't read as fully independent.
    cats_by_project = facts.groupby("project_id")["candidate_category"].agg(lambda s: sorted(set(s)))
    multi = cats_by_project[cats_by_project.map(len) > 1]

    shortlist_rows, comparison_rows = [], []
    for cat, brow in base.iterrows():
        cat_facts = facts.loc[facts["candidate_category"].eq(cat)]
        prof = cat_facts.loc[cat_facts["is_profile_subtype"]]
        focus = prof if not prof.empty else cat_facts
        n_profile = int(brow["n_profile_fonsi_projects"])
        n_fonsi = int(brow["n_observed_fonsi_projects"])
        rec, rationale = recommend(brow["candidate_role"], n_profile, n_fonsi)

        # mitigation dependence over the focus set
        mit = focus["mitigation_dependence"].value_counts().to_dict()
        n_focus = max(len(focus), 1)
        case_specific_share = round(mit.get("case_specific_dependent", 0) / n_focus, 2)

        # best existing CE (lexical rank 1)
        best_ce = ""
        if not ce.empty:
            top = ce.loc[ce["candidate_category"].eq(cat)].sort_values("retrieval_rank")
            if not top.empty:
                r = top.iloc[0]
                best_ce = f"{r['agency_name']} — {str(r['ce_description'])[:120]} (lexical rank; UNVERIFIED)"

        shortlist_rows.append({
            "candidate_category": cat,
            "candidate_label": brow["candidate_label"],
            "role": brow["candidate_role"],
            "ce_story": brow["ce_story"],
            "n_observed_fonsi": n_fonsi,
            "n_profile_fonsi": n_profile,
            "recommendation": rec,
            "rationale": rationale,
        })
        comparison_rows.append({
            "candidate_category": cat,
            "candidate_label": brow["candidate_label"],
            "role": brow["candidate_role"],
            "recommendation": rec,
            "universe_CE": int(brow["n_ce_universe"]),
            "universe_EA": int(brow["n_ea_universe"]),
            "universe_EIS": int(brow["n_eis_universe"]),
            "observed_fonsi": n_fonsi,
            "profile_fonsi": n_profile,
            "typical_acres": limit_summary(focus["max_acres"]),
            "typical_miles": limit_summary(focus["max_miles"]),
            "typical_mw": limit_summary(focus["max_megawatts"]),
            "no_new_road_projects": int(focus["no_new_access_road"].sum()),
            "within_existing_row_projects": int(focus["within_existing_row"].sum()),
            "previously_disturbed_projects": int(focus["previously_disturbed_land"].sum()),
            "case_specific_mitigation_share": case_specific_share,
            "mitigation_breakdown": "; ".join(f"{k}={v}" for k, v in sorted(mit.items())),
            "best_existing_ce_unverified": best_ce,
        })

        # per-candidate evidence table (profile subset, representative projects)
        ev = focus.sort_values(["max_acres"], ascending=False)[[
            "project_id", "subtype", "action_definition", "max_acres", "max_miles",
            "max_megawatts", "n_wells", "no_new_access_road", "within_existing_row",
            "previously_disturbed_land", "has_sensitive_resource", "extraordinary_circumstances",
            "mitigation_dependence", "confidence", "citation_document_id", "citation_page",
        ]].copy()
        ev["also_in_categories"] = ev["project_id"].map(
            lambda p: ", ".join(c for c in cats_by_project.get(p, []) if c != cat))
        ev.to_csv(D6_OUTPUT_DIR / f"d6_candidate_evidence_{cat}.csv", index=False)

    order = {"profile": 0, "profile_broadened": 1, "thin_fold_or_drop": 2, "contrast": 3}
    shortlist = pd.DataFrame(shortlist_rows)
    shortlist["_o"] = shortlist["recommendation"].map(order).fillna(9)
    shortlist = shortlist.sort_values(["_o", "n_observed_fonsi"], ascending=[True, False]).drop(columns="_o")
    shortlist.to_csv(SHORTLIST, index=False)

    comparison = pd.DataFrame(comparison_rows)
    comparison["_o"] = comparison["recommendation"].map(order).fillna(9)
    comparison = comparison.sort_values(["_o", "observed_fonsi"], ascending=[True, False]).drop(columns="_o")
    comparison.to_csv(COMPARISON, index=False)

    print(f"[n05] shortlist -> {SHORTLIST}")
    print(shortlist.to_string(index=False))
    print(f"\n[n05] comparison -> {COMPARISON}")
    print(comparison[["candidate_label", "recommendation", "observed_fonsi", "profile_fonsi",
                      "typical_acres", "no_new_road_projects", "case_specific_mitigation_share"]].to_string(index=False))
    # multi-category overlap summary (so counts aren't read as fully independent)
    overlap = pd.DataFrame({
        "project_id": list(multi.index),
        "categories": [", ".join(v) for v in multi.values],
    })
    overlap.to_csv(D6_OUTPUT_DIR / "d6_multicategory_overlap.csv", index=False)

    print(f"\n[n05] per-candidate evidence CSVs written for {len(comparison_rows)} candidates. run_at={run_at}")
    print(f"[n05] multi-category projects: {len(multi)} (appear under >1 candidate; "
          f"see d6_multicategory_overlap.csv)")


if __name__ == "__main__":
    main()
