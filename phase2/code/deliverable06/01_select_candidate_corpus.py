"""D6 v2 — 01: select the candidate corpus (narrow-first).

Applies the `candidates.py` membership rules over:
  - the full clean-energy project universe (projects_nepa_reviews) for base-rate
    denominators by process_type; and
  - the observed clean EA-source FONSI projects (fonsi_project_inventory).

Also: splits transmission and solar into subtypes, runs the temporary
resource-assessment (#4) prevalence + de-overlap screen against geothermal
exploration, and runs the storage-deployment scan (Gate 2 evidence only).

Outputs:
  - data/analysis/deliverable06/candidate_corpus.parquet
  - output/deliverable06/candidate_membership_review.csv
  - output/deliverable06/candidate_storage_scan_review.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import (
    D03_REVIEWS,
    D6_ANALYSIS_DIR,
    D6_REVIEW_DIR,
    PROJECTS_COMBINED,
    ensure_d6_dirs,
    input_hashes,
    utc_now,
    write_parquet,
)
from candidates import (
    RESOURCE_ASSESSMENT,
    STORAGE_SCAN_EXCLUDE,
    STORAGE_SCAN_INCLUDE,
    TAXONOMY_VERSION,
    TECH_CANDIDATES,
    text_blob,
)

FONSI_INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
CORPUS_OUT = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
MEMBERSHIP_REVIEW = D6_REVIEW_DIR / "candidate_membership_review.csv"
STORAGE_REVIEW = D6_REVIEW_DIR / "candidate_storage_scan_review.csv"

GEO_EXPLORATION = ("geothermal_exploration", "exploration")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["project_energy_type"].astype(str).eq("Clean")].copy()


def _blob_series(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    present = [c for c in cols if c in df.columns]
    return df.apply(lambda r: text_blob(*(r.get(c) for c in present)), axis=1)


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    reviews = _clean(pd.read_parquet(D03_REVIEWS))
    inv = _clean(pd.read_parquet(FONSI_INVENTORY))

    # Enrich the universe with title/description so denominator subtype matching
    # is not starved (projects_nepa_reviews has only project_type).
    pc = pd.read_parquet(PROJECTS_COMBINED, columns=["project_id", "project_title", "project_description"])
    pc["project_id"] = pc["project_id"].astype(str)
    reviews["project_id"] = reviews["project_id"].astype(str)
    reviews = reviews.merge(pc, on="project_id", how="left")

    # universe blob = project_type + title + description; fonsi blob = same fields.
    reviews["blob"] = _blob_series(reviews, ["project_type", "project_title", "project_description"])
    inv["blob"] = _blob_series(inv, ["project_title", "project_description", "project_type"])

    universe_rows: list[dict] = []
    fonsi_rows: list[dict] = []

    # --- tech-group candidates ---
    for cand in TECH_CANDIDATES:
        tg = set(cand.tech_groups)
        u = reviews.loc[reviews["tech_group"].isin(tg)]
        for r in u.itertuples(index=False):
            sub = cand.subtype_for(r.blob)
            universe_rows.append({
                "project_id": str(r.project_id), "candidate_category": cand.category,
                "subtype": sub, "process_type": r.process_type, "tech_group": r.tech_group,
            })
        f = inv.loc[inv["tech_group"].isin(tg)]
        for r in f.itertuples(index=False):
            sub = cand.subtype_for(r.blob)
            fonsi_rows.append({
                "project_id": str(r.project_id), "candidate_category": cand.category,
                "subtype": sub, "tech_group": r.tech_group, "project_title": r.project_title,
                "canonical_fonsi_document_id": r.canonical_fonsi_document_id,
                "lead_agency_harmonized": getattr(r, "lead_agency_harmonized", None),
                "project_state": getattr(r, "project_state", None),
            })

    # --- #4 temporary resource assessment (cross-tech, keyword) ---
    ra = RESOURCE_ASSESSMENT
    geo_expl_ids = {
        row["project_id"] for row in fonsi_rows + universe_rows
        if (row["candidate_category"], row.get("subtype")) == GEO_EXPLORATION
    }
    u = reviews.loc[reviews["blob"].apply(lambda b: bool(ra.include.search(b)))]
    for r in u.itertuples(index=False):
        if str(r.project_id) in geo_expl_ids:  # de-overlap with #2
            continue
        universe_rows.append({
            "project_id": str(r.project_id), "candidate_category": ra.category,
            "subtype": "resource_assessment", "process_type": r.process_type, "tech_group": r.tech_group,
        })
    f = inv.loc[inv["blob"].apply(lambda b: bool(ra.include.search(b)))]
    for r in f.itertuples(index=False):
        if str(r.project_id) in geo_expl_ids:
            continue
        fonsi_rows.append({
            "project_id": str(r.project_id), "candidate_category": ra.category,
            "subtype": "resource_assessment", "tech_group": r.tech_group, "project_title": r.project_title,
            "canonical_fonsi_document_id": r.canonical_fonsi_document_id,
            "lead_agency_harmonized": getattr(r, "lead_agency_harmonized", None),
            "project_state": getattr(r, "project_state", None),
        })

    universe = pd.DataFrame(universe_rows).drop_duplicates(["project_id", "candidate_category"])
    fonsi = pd.DataFrame(fonsi_rows).drop_duplicates(["project_id", "candidate_category"])

    # --- merge into one candidate_corpus (universe = denominator; flag observed FONSI) ---
    fonsi_keyed = fonsi.set_index(["project_id", "candidate_category"])
    fonsi_pairs = set(fonsi_keyed.index)

    meta = {c.category: c for c in TECH_CANDIDATES}
    profile_subs = {c.category: set(c.profile_subtypes) for c in TECH_CANDIDATES}

    def label_role_story(cat: str):
        if cat in meta:
            return meta[cat].label, meta[cat].role, meta[cat].ce_story
        return ra.label, ra.role, ra.ce_story  # resource_assessment

    rows: list[dict] = []
    # start from union of (project_id, candidate_category) across universe + fonsi
    all_pairs = set(zip(universe["project_id"], universe["candidate_category"])) | fonsi_pairs
    uni_keyed = universe.set_index(["project_id", "candidate_category"])
    for pid, cat in sorted(all_pairs):
        is_fonsi = (pid, cat) in fonsi_pairs
        label, role, ce_story = label_role_story(cat)
        if is_fonsi:
            frow = fonsi_keyed.loc[(pid, cat)]
            subtype = frow["subtype"]
            tech_group = frow["tech_group"]
            title = frow.get("project_title")
            doc_id = frow.get("canonical_fonsi_document_id")
            agency = frow.get("lead_agency_harmonized")
            state = frow.get("project_state")
        else:
            urow = uni_keyed.loc[(pid, cat)]
            subtype = urow["subtype"]
            tech_group = urow["tech_group"]
            title = doc_id = agency = state = None
        # process_type from universe if present, else EA (FONSI projects are EA-source)
        proc = uni_keyed.loc[(pid, cat)]["process_type"] if (pid, cat) in uni_keyed.index else "EA"
        if cat in profile_subs and profile_subs[cat]:
            is_profile = subtype in profile_subs[cat]
        else:
            is_profile = role == "profile"
        rows.append({
            "project_id": pid, "candidate_category": cat, "candidate_label": label,
            "candidate_role": role, "ce_story": ce_story, "subtype": subtype,
            "is_profile_subtype": is_profile, "process_type": proc, "tech_group": tech_group,
            "is_fonsi": is_fonsi, "project_title": title, "canonical_fonsi_document_id": doc_id,
            "lead_agency_harmonized": agency, "project_state": state,
            "taxonomy_version": TAXONOMY_VERSION, "corpus_run_at": run_at,
        })

    corpus = pd.DataFrame(rows)
    corpus["input_hashes"] = input_hashes([D03_REVIEWS, FONSI_INVENTORY, PROJECTS_COMBINED])
    write_parquet(corpus, CORPUS_OUT)

    # --- membership review (observed FONSI only) ---
    review = corpus.loc[corpus["is_fonsi"]].copy()
    review = review[[
        "candidate_category", "subtype", "is_profile_subtype", "candidate_role",
        "project_id", "project_title", "lead_agency_harmonized", "project_state",
        "tech_group", "ce_story",
    ]].sort_values(["candidate_category", "subtype", "project_id"])
    review.to_csv(MEMBERSHIP_REVIEW, index=False)

    # --- storage-deployment scan (Gate 2 evidence only) ---
    scan = inv.loc[
        inv["blob"].apply(lambda b: bool(STORAGE_SCAN_INCLUDE.search(b)))
        & ~inv["blob"].apply(lambda b: bool(STORAGE_SCAN_EXCLUDE.search(b)))
    ].copy()
    scan_out = scan[[
        "project_id", "project_title", "tech_group", "lead_agency_harmonized", "project_state",
    ]].sort_values("tech_group")
    scan_out.to_csv(STORAGE_REVIEW, index=False)

    # --- console summary ---
    print(f"[01] taxonomy={TAXONOMY_VERSION} run_at={run_at}")
    print(f"[01] wrote candidate_corpus rows={len(corpus):,} -> {CORPUS_OUT}")
    print("\n[01] candidate summary (distinct projects):")
    g = corpus.groupby("candidate_category")
    for cat, sub in g:
        n_univ = sub["project_id"].nunique()
        n_fonsi = sub.loc[sub["is_fonsi"], "project_id"].nunique()
        n_prof = sub.loc[sub["is_fonsi"] & sub["is_profile_subtype"], "project_id"].nunique()
        by_proc = sub.groupby("process_type")["project_id"].nunique().to_dict()
        role = sub["candidate_role"].iloc[0]
        print(f"  {cat:32s} role={role:8s} universe={n_univ:5d} by_proc={by_proc}  "
              f"fonsi={n_fonsi:4d} profile_fonsi={n_prof:4d}")
    print("\n[01] subtype breakdown (observed FONSI):")
    fb = corpus.loc[corpus["is_fonsi"]].groupby(["candidate_category", "subtype"])["project_id"].nunique()
    for (cat, sub), n in fb.items():
        print(f"  {cat:32s} {sub:22s} {n:4d}")
    print(f"\n[01] storage-deployment scan (non-manufacturing) rows={len(scan_out)} -> {STORAGE_REVIEW}")
    if len(scan_out):
        print(scan_out.groupby("tech_group")["project_id"].nunique().to_string())


if __name__ == "__main__":
    main()
