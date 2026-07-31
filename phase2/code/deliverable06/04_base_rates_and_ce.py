"""D6 v2 — 04: candidate base rates + existing-CE comparison.

Base rates (per candidate, three explicit counts — never one ambiguous "share"):
  1. full clean candidate universe by process_type (CE / EA / EIS);
  2. candidate EA projects; and
  3. observed EA-source FONSI projects.

CE comparison: lexical ranking of the pinned CE Explorer snapshot against each
candidate's query terms — a *ranking aid only*; every match is left
`manual_verification_status = pending`. Also reuses D3 `ce_citations` for any
project-level CE-use evidence among candidate projects.

Outputs:
  - data/analysis/deliverable06/candidate_base_rates.parquet
  - data/analysis/deliverable06/candidate_ce_comparison.parquet
  - output/deliverable06/candidate_ce_comparison_review.csv
"""

import os
import re

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

import bounds
import ce_source
import embeddings
from common import (
    D03_CE_CITATIONS,
    D6_ANALYSIS_DIR,
    D6_REVIEW_DIR,
    ensure_d6_dirs,
    normalize_space,
    utc_now,
    write_parquet,
)
from candidates import TAXONOMY_VERSION

CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
# Existing-CE source: canonical ce.json (CE Explorer) via ce_source — see ce_source.py
BASE_OUT = D6_ANALYSIS_DIR / "candidate_base_rates.parquet"
CE_OUT = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
CE_REVIEW = D6_REVIEW_DIR / "candidate_ce_comparison_review.csv"
DESC_OUT = D6_ANALYSIS_DIR / "candidate_descriptive.parquet"
DESC_REVIEW = D6_REVIEW_DIR / "candidate_descriptive_review.csv"

ANALYSIS_VERSION = "d6_stage_a_v2"
TOP_CE = 8

QUERY_TERMS = {
    "transmission_upgrade": "transmission line upgrade rebuild reconductor power existing right of way corridor",
    "geothermal_exploration": "geothermal exploration temperature gradient well geophysical survey",
    "solar": "solar photovoltaic energy generation facility disturbed land",
    "temporary_resource_assessment": "site characterization meteorological tower geotechnical monitoring survey investigation",
    "wind_onshore": "wind turbine energy meteorological tower testing",
}

_token = re.compile(r"[a-z]{3,}")
_STOP = {"the", "and", "for", "with", "from", "this", "that", "are", "any", "all", "may",
         "right", "way", "land", "energy", "facility", "generation", "power"}


def tokens(text: str) -> set[str]:
    return {t for t in _token.findall((text or "").lower()) if t not in _STOP}


def ce_text(row) -> str:
    return " ".join(normalize_space(getattr(row, c, "")) for c in
                    ("agency_name", "agency_unit", "ce_description", "context",
                     "additional_context", "extraordinary_circumstances"))


def descriptive_breakdowns(fonsi: pd.DataFrame) -> pd.DataFrame:
    """Long-form per-candidate distributions by agency, geography, and decision year.

    Decision year is preliminary (from the inventory's BLM/DOE decision dates);
    fuller timeline integration awaits the D4 deliverable.
    """
    rows = []
    for cat, grp in fonsi.groupby("candidate_category"):
        for dim, col in (("agency", "lead_agency_harmonized"), ("state", "project_state")):
            vals = grp[col].dropna().astype(str).replace({"[]": "", "": None}).dropna()
            for value, n in vals.value_counts().items():
                rows.append({"candidate_category": cat, "dimension": dim,
                             "value": value, "n_fonsi_projects": int(n)})
        if "decision_year" in grp.columns:
            yrs = grp["decision_year"].dropna()
            for value, n in yrs.value_counts().sort_index().items():
                rows.append({"candidate_category": cat, "dimension": "decision_year",
                             "value": str(int(value)), "n_fonsi_projects": int(n)})
    return pd.DataFrame(rows)


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    corpus = pd.read_parquet(CORPUS)
    corpus["project_id"] = corpus["project_id"].astype(str)

    # --- D3 CE-use evidence among candidate projects ---
    ce_use_ids: set[str] = set()
    if D03_CE_CITATIONS.exists():
        cites = pd.read_parquet(D03_CE_CITATIONS)
        ce_use_ids = set(cites["project_id"].astype(str))

    # --- base rates ---
    base_rows = []
    for cat, grp in corpus.groupby("candidate_category"):
        by_proc = grp.groupby("process_type")["project_id"].nunique().to_dict()
        fonsi = grp.loc[grp["is_fonsi"]]
        profile_fonsi = fonsi.loc[fonsi["is_profile_subtype"], "project_id"].nunique()
        cand_ids = set(grp["project_id"])
        base_rows.append({
            "candidate_category": cat,
            "candidate_label": grp["candidate_label"].iloc[0],
            "candidate_role": grp["candidate_role"].iloc[0],
            "ce_story": grp["ce_story"].iloc[0],
            "n_universe_projects": grp["project_id"].nunique(),
            "n_ce_universe": int(by_proc.get("CE", 0)),
            "n_ea_universe": int(by_proc.get("EA", 0)),
            "n_eis_universe": int(by_proc.get("EIS", 0)),
            "n_observed_fonsi_projects": int(fonsi["project_id"].nunique()),
            "n_profile_fonsi_projects": int(profile_fonsi),
            "n_projects_with_ce_citation": len(cand_ids & ce_use_ids),
            "base_rate_caveat": ("Counts are distinct projects; a project may map to >1 candidate. "
                                 "'observed EA-source FONSI projects' is not a general FONSI rate."),
            "taxonomy_version": TAXONOMY_VERSION,
            "analysis_version": ANALYSIS_VERSION,
            "analysis_run_at": run_at,
        })
    base = pd.DataFrame(base_rows).sort_values("n_observed_fonsi_projects", ascending=False)
    write_parquet(base, BASE_OUT)

    # --- descriptive breakdowns (agency, geography; preliminary decision year) ---
    obs = corpus.loc[corpus["is_fonsi"]].copy()
    if INVENTORY.exists():
        inv = pd.read_parquet(INVENTORY, columns=["project_id", "blm_decision_date", "doe_decision_date"])
        inv["project_id"] = inv["project_id"].astype(str)
        dt = pd.to_datetime(inv["blm_decision_date"].fillna(inv["doe_decision_date"]), errors="coerce")
        inv["decision_year"] = dt.dt.year
        obs = obs.merge(inv[["project_id", "decision_year"]], on="project_id", how="left")
    desc = descriptive_breakdowns(obs)
    write_parquet(desc, DESC_OUT)
    if not desc.empty:
        desc.sort_values(["candidate_category", "dimension", "n_fonsi_projects"],
                         ascending=[True, True, False]).to_csv(DESC_REVIEW, index=False)

    # --- CE comparison per tech_group x action CELL (refactor: cells from enrichment + 10 labels) ---
    # Each cell's query is its tech + action + a few representative member summaries — replacing the
    # old hardcoded QUERY_TERMS. Ranking is the same blended embedding+lexical aid; never decides coverage.
    ce_rows = []
    if ce_source.CE_JSON.exists():
        ce = ce_source.load_ce_catalog().reset_index(drop=True)
        ce["cetext"] = [ce_text(r) for r in ce.itertuples(index=False)]
        ce["cetok"] = ce["cetext"].map(tokens)
        use_emb = embeddings.available()
        if not use_emb:
            print(f"[04] WARNING: embeddings unavailable ({embeddings.MODEL_NAME}); CE scores fall back "
                  "to lexical-only and are NOT comparable to the report's blended scores.")
        ce_emb = embeddings.embed(ce["cetext"].tolist()) if use_emb else None

        # build cells (tech_group x action) from the enrichment + the 10 action labels
        en = pd.read_parquet(D6_ANALYSIS_DIR / "fonsi_enrichment.parquet")
        en = en[en["action_summary"].notna()].copy()
        en["project_id"] = en["project_id"].astype(str)
        _lab = pd.read_parquet(D6_ANALYSIS_DIR / "fonsi_action_labels.parquet")
        _lab["project_id"] = _lab["project_id"].astype(str)
        en = en.merge(_lab[["project_id", "action"]], on="project_id", how="left")
        en["action"] = en["action"].fillna("other").astype(str)
        en["tech_group"] = en["tech_group"].fillna("(missing)").astype(str)
        en["_cell"] = en["tech_group"] + "__" + en["action"]
        _bnd = en["is_bounded_low_impact"].map(lambda v: v is True)
        _tx = en["tech_group"].eq("Transmission")
        _wr = en["within_existing_row"].map(lambda v: v is True)
        _nr = ~(en["new_access_road"] == True)
        en["_shaped"] = _bnd & (~_tx | ((en["action"] == "upgrade") & _wr & _nr))

        import numpy as np
        for cell, grp in en.groupby("_cell"):
            focus = grp[grp["_shaped"]] if grp["_shaped"].any() else grp
            summaries = focus["action_summary"].dropna().astype(str).tolist()
            if not summaries or not use_emb:
                continue
            # Per-member matching (avoids the long-query dilution): the cell->CE score is the MEDIAN
            # over the cell's members of each member's cosine to that CE, so the best CE is the one the
            # cell's members most CONSISTENTLY match. Robust + honest (mirrors the per-FONSI diagnostic);
            # 07 applies the develop/adopt threshold to this score.
            m_emb = embeddings.embed(summaries)                       # (n_mem, 384), L2-normalized
            cosm = np.asarray(embeddings.cosine(m_emb, ce_emb))       # (n_mem, n_ce)
            cell_ce = np.median(cosm, axis=0)                         # (n_ce,) robust cell->CE score
            for rank, ci in enumerate(np.argsort(-cell_ce)[:TOP_CE], start=1):
                rd = ce.iloc[int(ci)].to_dict()
                ret = float(cell_ce[int(ci)])
                ce_rows.append({
                    "candidate_category": cell,
                    "retrieval_rank": rank,
                    "retrieval_score": round(ret, 4),
                    "embedding_cosine": round(ret, 4),
                    "embedding_model": embeddings.MODEL_NAME,
                    "lexical_score": None,
                    "token_overlap": None,
                    "match_method": "per_member_median_cosine",
                    "manual_verification_status": "pending",
                    "match_type": "unverified_candidate",
                    "ce_id": rd.get("ce_id", ""),
                    "structured_id": rd.get("structured_id", ""),
                    "agency_name": rd.get("agency_name", ""),
                    "agency_unit": rd.get("agency_unit", ""),
                    "ce_description": normalize_space(rd.get("ce_description", ""))[:400],
                    "canonical_source_url": rd.get("canonical_source_url", ""),
                    "extraordinary_circumstances": normalize_space(rd.get("extraordinary_circumstances", ""))[:200],
                    "source_version": rd.get("source_version", ""),
                    "source_version_date": rd.get("source_version_date", ""),
                    "ce_comparison_run_at": run_at,
                    "taxonomy_version": TAXONOMY_VERSION,
                    **{f"bound_{m}": v for m, v in bounds.parse_bounds(
                        normalize_space(rd.get("ce_description", "")) + " " +
                        normalize_space(rd.get("extraordinary_circumstances", ""))).items()},
                })
    ce_cmp = pd.DataFrame(ce_rows)
    write_parquet(ce_cmp, CE_OUT)
    if not ce_cmp.empty:
        ce_cmp.sort_values(["candidate_category", "retrieval_rank"]).to_csv(CE_REVIEW, index=False)

    print(f"[04] base rates -> {BASE_OUT}")
    print(base[["candidate_category", "n_ce_universe", "n_ea_universe", "n_eis_universe",
                "n_observed_fonsi_projects", "n_profile_fonsi_projects",
                "n_projects_with_ce_citation"]].to_string(index=False))
    print(f"\n[04] descriptive breakdown rows={len(desc)} (agency/state/decision_year) -> {DESC_OUT}")
    print(f"[04] CE comparison rows={len(ce_cmp)} embeddings={embeddings.available()} "
          f"(all pending manual verification) -> {CE_OUT}")


if __name__ == "__main__":
    main()
