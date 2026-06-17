"""D6 v2 — n04: candidate base rates + existing-CE comparison.

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

import embeddings
from common import (
    D03_CE_CITATIONS,
    D6_ANALYSIS_DIR,
    D6_OUTPUT_DIR,
    ensure_d6_dirs,
    normalize_space,
    utc_now,
    write_parquet,
)
from candidates import TAXONOMY_VERSION

CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
CE_SNAPSHOT = D6_ANALYSIS_DIR / "ce_explorer_snapshot.parquet"
BASE_OUT = D6_ANALYSIS_DIR / "candidate_base_rates.parquet"
CE_OUT = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
CE_REVIEW = D6_OUTPUT_DIR / "candidate_ce_comparison_review.csv"
DESC_OUT = D6_ANALYSIS_DIR / "candidate_descriptive.parquet"
DESC_REVIEW = D6_OUTPUT_DIR / "candidate_descriptive_review.csv"

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

    # --- CE comparison (lexical ranking aid only) ---
    ce_rows = []
    if CE_SNAPSHOT.exists():
        ce = pd.read_parquet(CE_SNAPSHOT).reset_index(drop=True)
        ce["cetext"] = [ce_text(r) for r in ce.itertuples(index=False)]
        ce["cetok"] = ce["cetext"].map(tokens)
        # Embedding similarity is a semantic ranking aid (blended with lexical);
        # it never decides coverage. Falls back to lexical-only if unavailable.
        use_emb = embeddings.available()
        ce_emb = embeddings.embed(ce["cetext"].tolist()) if use_emb else None
        for cat in corpus["candidate_category"].unique():
            qterms = QUERY_TERMS.get(cat, cat.replace("_", " "))
            q = tokens(qterms)
            cos = embeddings.cosine(embeddings.embed([qterms]), ce_emb)[0] if use_emb else None
            scored = []
            for i, r in enumerate(ce.itertuples(index=False)):
                overlap = len(q & r.cetok)
                lex = overlap / max(len(q), 1)
                c = float(cos[i]) if cos is not None else 0.0
                ret = (0.65 * c + 0.35 * lex) if use_emb else lex
                if ret > 0:
                    scored.append((ret, c, lex, overlap, r))
            scored.sort(key=lambda x: (-x[0], -x[3]))
            for rank, (ret, c, lex, overlap, r) in enumerate(scored[:TOP_CE], start=1):
                ce_rows.append({
                    "candidate_category": cat,
                    "retrieval_rank": rank,
                    "retrieval_score": round(ret, 4),
                    "embedding_cosine": round(c, 4) if use_emb else None,
                    "embedding_model": embeddings.MODEL_NAME if use_emb else "",
                    "lexical_score": round(lex, 4),
                    "token_overlap": overlap,
                    "manual_verification_status": "pending",
                    "match_type": "unverified_candidate",
                    "ce_id": getattr(r, "ce_id", ""),
                    "structured_id": getattr(r, "structured_id", ""),
                    "agency_name": getattr(r, "agency_name", ""),
                    "agency_unit": getattr(r, "agency_unit", ""),
                    "ce_description": normalize_space(getattr(r, "ce_description", ""))[:400],
                    "canonical_source_url": getattr(r, "canonical_source_url", ""),
                    "extraordinary_circumstances": normalize_space(
                        getattr(r, "extraordinary_circumstances", ""))[:200],
                    "source_version": getattr(r, "source_version", ""),
                    "source_version_date": getattr(r, "source_version_date", ""),
                    "ce_comparison_run_at": run_at,
                    "taxonomy_version": TAXONOMY_VERSION,
                })
    ce_cmp = pd.DataFrame(ce_rows)
    write_parquet(ce_cmp, CE_OUT)
    if not ce_cmp.empty:
        ce_cmp.sort_values(["candidate_category", "retrieval_rank"]).to_csv(CE_REVIEW, index=False)

    print(f"[n04] base rates -> {BASE_OUT}")
    print(base[["candidate_category", "n_ce_universe", "n_ea_universe", "n_eis_universe",
                "n_observed_fonsi_projects", "n_profile_fonsi_projects",
                "n_projects_with_ce_citation"]].to_string(index=False))
    print(f"\n[n04] descriptive breakdown rows={len(desc)} (agency/state/decision_year) -> {DESC_OUT}")
    print(f"[n04] CE comparison rows={len(ce_cmp)} embeddings={embeddings.available()} "
          f"(all pending manual verification) -> {CE_OUT}")


if __name__ == "__main__":
    main()
