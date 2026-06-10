"""
Phase A — read-only EA decision-coverage audit (ea_audit.md §7, §14.1).

Reproduces the EA failure funnel from the production D4 parquets and freezes the review/holdout
sets needed by Phase B. READ-ONLY: reads timeline parquets + ea pages/documents; writes ONLY under
phase2/output/deliverable04/ea_audit/. Does not touch any production parquet, candidates, or flags.

Run:
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_audit_ea_decision_recall.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
TIMELINE = PHASE2 / "data" / "analysis" / "timeline"
EA_PROC = PHASE2 / "data" / "processed" / "ea"
OUT = PHASE2 / "output" / "deliverable04" / "ea_audit"
OUT.mkdir(parents=True, exist_ok=True)

PDATES = TIMELINE / "timeline_project_dates.parquet"
CANDS = TIMELINE / "timeline_candidates.parquet"
INDEX = TIMELINE / "timeline_document_index.parquet"
PAGES = EA_PROC / "pages.parquet"
RANKER = PHASE2 / "training" / "deliverable04" / "ranker.csv"

SEED = 42
POOL_ROLES = ("clear_decision", "proxy_decision", "body_text")
# Eligibility gate replicated from 05_select_dates.py:607-629
GATE_CLEAR = 0.0     # clear_decision needs ranking_score (==learned_decision_score) > 0
GATE_PROXY = -2.0    # proxy/body need > -2
PDC_STRONG = 0.5     # classifier "high-confidence" threshold (171-cohort)
PDC_UNCERTAIN = 0.25
STRONG_DOC = 4.5     # decision_doc_score threshold for a "strong decision document"
IMAGE_ONLY_CHARS = 200  # total extracted chars below this => image-only/empty

DAY_DATE_RE = re.compile(
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}"
    r"|\b\d{1,2}/\d{1,2}/\d{4}\b"
    r"|\b\d{4}-\d{1,2}-\d{1,2}\b"
    r"|\b\d{1,2}\.\d{1,2}\.\d{4}\b",
    re.IGNORECASE,
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def main() -> None:
    con = duckdb.connect()
    log("=== Phase A EA audit ===")

    # ---- 1. Missing EA projects (authoritative: project_dates, NOT candidate flags) ----
    pdates = con.execute(
        f"SELECT * FROM read_parquet('{PDATES}') WHERE process_type='EA'"
    ).fetchdf()
    n_ea = len(pdates)
    missing = pdates[(pdates["decision_date"].isna()) | (pdates["decision_date"] == "")].copy()
    miss_ids = set(missing["project_id"].astype(str))
    n_missing = len(miss_ids)
    n_selected = n_ea - n_missing
    log(f"EA projects={n_ea}  selected={n_selected}  missing={n_missing}  "
        f"coverage={n_selected/n_ea:.4f}")

    # ---- 2. EA candidates for missing projects ----
    cand = con.execute(
        f"""SELECT * FROM read_parquet('{CANDS}')
            WHERE process_type='EA' AND project_id IN (
                SELECT project_id FROM read_parquet('{PDATES}')
                WHERE process_type='EA' AND (decision_date IS NULL OR decision_date=''))"""
    ).fetchdf()
    cand["lds"] = _num(cand["learned_decision_score"])
    cand["pdc"] = _num(cand["p_dec_cal"])
    cand["is_pool"] = cand["candidate_role"].isin(POOL_ROLES)
    cand["is_reg"] = (cand["source_tier"] == "metadata") & (cand["retrieval_tier"] == "tier_a")
    cand["is_case_year"] = cand["rule_ids"].astype(str).str.contains("nepa_case_year", na=False)
    cand["rcs"] = _num(cand["role_confidence_score"])

    def gate_pass(row) -> bool:
        if row["candidate_role"] == "clear_decision":
            return pd.notna(row["lds"]) and row["lds"] > GATE_CLEAR
        if row["candidate_role"] in ("proxy_decision", "body_text"):
            return pd.notna(row["lds"]) and row["lds"] > GATE_PROXY
        return False

    cand["gate_pass"] = cand.apply(gate_pass, axis=1)

    # ---- 3. Per-project aggregate flags ----
    cand["_day"] = cand["date_granularity"].eq("day")
    cand["_month"] = cand["date_granularity"].eq("month")
    cand["_clear"] = cand["candidate_role"].eq("clear_decision")
    g = cand.groupby("project_id")
    pp = pd.DataFrame({"project_id": list(miss_ids)})
    agg = pd.DataFrame({
        "has_pool": g["is_pool"].any(),
        "pool_day": g.apply(lambda d: bool((d["is_pool"] & d["_day"]).any()), include_groups=False),
        "pool_month": g.apply(lambda d: bool((d["is_pool"] & d["_month"]).any()), include_groups=False),
        "reg_day_clear": g.apply(lambda d: bool((d["is_reg"] & d["_clear"] & d["_day"]).any()), include_groups=False),
        "day_clear": g.apply(lambda d: bool((d["_clear"] & d["_day"]).any()), include_groups=False),
        # strong-cue day clear_decision (CLEAR_DECISION_STRONG => role_confidence_score==5.0)
        "strong_day_clear": g.apply(lambda d: bool((d["_clear"] & d["_day"] & d["rcs"].ge(5.0)).any()), include_groups=False),
        "max_pdc_day_pool": g.apply(lambda d: _num(d.loc[d["is_pool"] & d["_day"], "pdc"]).max(), include_groups=False),
        "month_gate_selectable": g.apply(lambda d: bool((d["is_pool"] & d["_month"] & d["gate_pass"]).any()), include_groups=False),
        "day_gate_selectable": g.apply(lambda d: bool((d["is_pool"] & d["_day"] & d["gate_pass"]).any()), include_groups=False),
        "has_case_year": g["is_case_year"].any(),
        "n_cands": g.size(),
    }).reset_index()
    pp = pp.merge(agg, on="project_id", how="left")
    # projects with zero candidates
    for col in ["has_pool", "pool_day", "pool_month", "reg_day_clear", "day_clear", "strong_day_clear",
                "month_gate_selectable", "day_gate_selectable", "has_case_year"]:
        pp[col] = pp[col].fillna(False).astype(bool)
    pp["n_cands"] = pp["n_cands"].fillna(0).astype(int)
    pp["max_pdc_day_pool"] = _num(pp["max_pdc_day_pool"])

    # ---- 4. Mutually exclusive cohort funnel (fixed order; first match wins) ----
    def assign(r) -> str:
        # 1. month would be recovered under current gate (no eligible day, eligible month)
        if (not r["day_gate_selectable"]) and r["month_gate_selectable"]:
            return "1_month_current_gate"
        # 2. day pool candidate with strong classifier support, blocked by ranker gate
        if r["pool_day"] and pd.notna(r["max_pdc_day_pool"]) and r["max_pdc_day_pool"] >= PDC_STRONG:
            return "2_day_pdc05_blocked"
        # 3. strong day clear_decision blocked by ranker (register, or strong-cue doc-text)
        if r["reg_day_clear"] or r["strong_day_clear"]:
            return "3_strong_day_clear_blocked"
        # 4. day pool candidate, uncertain classifier
        if r["pool_day"] and pd.notna(r["max_pdc_day_pool"]) and PDC_UNCERTAIN <= r["max_pdc_day_pool"] < PDC_STRONG:
            return "4_day_uncertain"
        # 5. day pool candidate, weak classifier
        if r["pool_day"]:
            return "5_day_weak"
        # 6. month-only pool evidence
        if r["pool_month"]:
            return "6_month_only"
        # 7. case-number year only
        if r["has_case_year"]:
            return "7_case_year_only"
        # 8. no decision-pool role
        return "8_no_pool_role"

    pp["cohort"] = pp.apply(assign, axis=1)
    # register sub-split inside cohort 3
    pp["reg_in_cohort3"] = (pp["cohort"] == "3_strong_day_clear_blocked") & pp["reg_day_clear"]

    funnel = pp["cohort"].value_counts().sort_index()
    log("\n--- Mutually exclusive cohort funnel ---")
    for k, v in funnel.items():
        log(f"  {k:32s} {v}")
    log(f"  {'TOTAL':32s} {funnel.sum()}  (expect {n_missing})")

    # ---- 5. Headline funnel (audit §0.2 / §2) reproduction ----
    any_pool = int(pp["has_pool"].sum())
    explicit_role = int(g.apply(lambda d: bool(d["candidate_role"].isin(["clear_decision", "proxy_decision"]).any()), include_groups=False).reindex(pp["project_id"]).fillna(False).sum())
    day_pool = int(pp["pool_day"].sum())
    day_pool_pdc05 = int(((pp["max_pdc_day_pool"] >= PDC_STRONG)).sum())
    no_pool = int((~pp["has_pool"]).sum())
    reg_total = int(pp["reg_day_clear"].sum())
    reg_in_c3 = int(pp["reg_in_cohort3"].sum())
    headline = pd.DataFrame([
        ("missing_ea_decisions", n_missing, 988),
        ("any_pool_role", any_pool, 786),
        ("explicit_role", explicit_role, 732),
        ("day_pool", day_pool, 461),
        ("day_pool_pdc05", day_pool_pdc05, 171),
        ("no_pool_role", no_pool, 202),
        ("register_blocked_total", reg_total, 55),
        ("register_blocked_in_cohort3", reg_in_c3, 31),
    ], columns=["metric", "audit_value", "ea_audit_md_stated"])
    log("\n--- Headline reproduction (mine vs ea_audit.md) ---")
    log(headline.to_string(index=False))

    # ---- 6. Document inventory + register queues (audit §4.1/§4.2) ----
    idx = con.execute(
        f"""SELECT project_id, document_id, decision_doc_score, document_type_clean,
                   blm_match_status, blm_decision_date, doe_match_status, doe_decision_date,
                   doe_cx_decision_date, file_id
            FROM read_parquet('{INDEX}')
            WHERE process_type='EA' AND project_id IN ({','.join(["'%s'" % i for i in miss_ids])})"""
    ).fetchdf()
    gi = idx.groupby("project_id")
    doc_inv = pd.DataFrame({
        "max_decision_doc_score": gi["decision_doc_score"].max(),
        "has_strong_decision_doc": gi["decision_doc_score"].max() >= STRONG_DOC,
        "blm_status": gi["blm_match_status"].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
        "blm_has_decision_date": gi["blm_decision_date"].apply(lambda s: s.notna().any() and (s.astype(str) != "").any()),
        "doe_status": gi["doe_match_status"].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    }).reset_index()
    doc_inv = doc_inv.merge(pp[["project_id", "cohort", "reg_day_clear"]], on="project_id", how="right")
    blm_unresolved = int(doc_inv["blm_status"].isin(["unmatched", "no_accepted_match"]).sum())
    doe_not_found = int((doc_inv["doe_status"] == "accepted_not_found").sum())
    log(f"\n--- Register queues ---  BLM unresolved={blm_unresolved}  DOE accepted_not_found={doe_not_found}")

    # ---- 7. Full-text gap audit over strong decision docs of missing projects ----
    log("\n--- Full-text gap scan (strong decision docs) ---")
    strong_docs = idx[idx["decision_doc_score"] >= STRONG_DOC][["project_id", "document_id"]].drop_duplicates()
    log(f"  strong decision documents to scan: {len(strong_docs)} (projects: {strong_docs['project_id'].nunique()})")
    doc_ids = strong_docs["document_id"].astype(str).tolist()
    gap_rows = []
    if doc_ids:
        in_list = ",".join("'%s'" % d for d in doc_ids)
        pages = con.execute(
            f"""SELECT document_id, SUM(LENGTH(COALESCE(page_text,''))) AS total_chars,
                       STRING_AGG(COALESCE(page_text,''), ' ') AS full_text
                FROM read_parquet('{PAGES}')
                WHERE document_id IN ({in_list})
                GROUP BY document_id"""
        ).fetchdf()
        ptext = {str(r.document_id): (int(r.total_chars or 0), str(r.full_text or "")) for r in pages.itertuples()}
        # day candidate per document among missing-EA candidates
        day_cand_docs = set(cand.loc[cand["date_granularity"].eq("day"), "document_id"].astype(str))
        for r in strong_docs.itertuples():
            did = str(r.document_id)
            chars, txt = ptext.get(did, (0, ""))
            has_day_cand = did in day_cand_docs
            if chars < IMAGE_ONLY_CHARS:
                cls = "image_only_or_empty"
            elif DAY_DATE_RE.search(txt):
                cls = "has_day_cand" if has_day_cand else "day_in_text_no_candidate"
            else:
                cls = "no_day_date_in_text"
            gap_rows.append({"project_id": r.project_id, "document_id": did,
                             "total_chars": chars, "has_day_candidate": has_day_cand, "gap_class": cls})
    gap = pd.DataFrame(gap_rows)
    if not gap.empty:
        # project-level: worst-case classification per project (one strong doc resolving is enough)
        log(gap["gap_class"].value_counts().to_string())

    # ---- 8. Freeze review / holdout sets ----
    rng = np.random.default_rng(SEED)
    # 8a. register promotion review (all register-blocked, the B-1 verification list)
    reg_review = cand[cand["is_reg"] & cand["candidate_role"].eq("clear_decision") & cand["date_granularity"].eq("day")][
        ["project_id", "candidate_id", "document_id", "page_number", "raw_date_text", "parsed_date",
         "context_text", "lds", "pdc", "retrieval_tier", "candidate_source_type"]].copy()
    reg_review.to_csv(OUT / "ea_register_promotion_review.csv", index=False)
    # 8b. strong doc-text (cohort 3, non-register) review (the ~39)
    c3_ids = set(pp.loc[(pp["cohort"] == "3_strong_day_clear_blocked") & (~pp["reg_day_clear"]), "project_id"])
    text_review = cand[cand["project_id"].isin(c3_ids) & cand["candidate_role"].eq("clear_decision") & cand["date_granularity"].eq("day")][
        ["project_id", "candidate_id", "document_id", "page_number", "raw_date_text", "parsed_date",
         "context_text", "lds", "pdc", "document_type_clean"]].copy()
    text_review.to_csv(OUT / "ea_explicit_text_review.csv", index=False)
    # 8c. month review sample (>=30 if available)
    month_pool = cand[cand["is_pool"] & cand["date_granularity"].eq("month")]
    nm = min(40, len(month_pool))
    month_sample = month_pool.sample(n=nm, random_state=SEED) if nm else month_pool
    month_sample[["project_id", "candidate_id", "document_id", "page_number", "raw_date_text",
                  "context_text", "candidate_role", "lds", "pdc"]].to_csv(OUT / "ea_month_review_sample.csv", index=False)
    # 8d. promotion holdout: frozen 50% of register+cohort3 projects, sampled BEFORE any tuning
    promo_universe = sorted(set(pp.loc[pp["cohort"].isin(["2_day_pdc05_blocked", "3_strong_day_clear_blocked"]), "project_id"]))
    holdout = set(rng.choice(promo_universe, size=len(promo_universe) // 2, replace=False)) if promo_universe else set()
    (OUT / "ea_promotion_holdout_ids.txt").write_text("\n".join(sorted(holdout)))
    # 8e. promotion review sample (stratified: what the proposed selector would newly promote)
    promo_sample = pp[pp["cohort"].isin(["1_month_current_gate", "2_day_pdc05_blocked", "3_strong_day_clear_blocked"])]
    promo_sample.to_csv(OUT / "ea_promotion_review_sample.csv", index=False)

    # ---- 9. Persist core artifacts ----
    pp.to_parquet(OUT / "ea_missing_projects.parquet", index=False)
    cand.to_parquet(OUT / "ea_candidate_inventory.parquet", index=False)
    doc_inv.to_parquet(OUT / "ea_document_inventory.parquet", index=False)
    if not gap.empty:
        gap.to_parquet(OUT / "ea_full_text_gap_audit.parquet", index=False)
    (OUT / "ea_audit_ids.txt").write_text("\n".join(sorted(miss_ids)))

    funnel_out = funnel.rename_axis("cohort").reset_index(name="projects")
    funnel_out.to_csv(OUT / "ea_failure_funnel.csv", index=False)
    headline.to_csv(OUT / "ea_headline_reproduction.csv", index=False)

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "thresholds": {"GATE_CLEAR": GATE_CLEAR, "GATE_PROXY": GATE_PROXY, "PDC_STRONG": PDC_STRONG,
                       "PDC_UNCERTAIN": PDC_UNCERTAIN, "STRONG_DOC": STRONG_DOC,
                       "IMAGE_ONLY_CHARS": IMAGE_ONLY_CHARS},
        "ea_projects": int(n_ea), "selected": int(n_selected), "missing": int(n_missing),
        "coverage": round(n_selected / n_ea, 4),
        "cohort_funnel": {k: int(v) for k, v in funnel.items()},
        "register_blocked_total": reg_total, "register_blocked_in_cohort3": reg_in_c3,
        "blm_unresolved": blm_unresolved, "doe_accepted_not_found": doe_not_found,
        "gap_classes": gap["gap_class"].value_counts().to_dict() if not gap.empty else {},
    }
    (OUT / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"\nWrote artifacts to {OUT}")
    log("Cohort sum == missing: " + str(int(funnel.sum()) == n_missing))


if __name__ == "__main__":
    main()
