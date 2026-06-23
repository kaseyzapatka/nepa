"""D6 v2 — n07: classify candidates (new / expand / adopt) + rank + report tables.

Integrates the tracks into the deliverable's three outputs:
  - **NEW**    — best ce.json match is weak/absent (a FONSI class with no existing CE).
  - **EXPAND** — matched CE, but our FONSIs exceed its parsed numeric bound.
  - **ADOPT**  — matched CE, but some of our FONSI agencies don't have it.
  - (drop) already_covered — matched, within bounds, same agency.

Inputs (lower-numbered scripts only):
  candidate_base_rates / candidate_facts / candidate_ce_comparison(+bounds) /
  candidate_mitigation_summary / candidate_corpus.

Outputs:
  - data/analysis/deliverable06/candidate_verdicts.parquet
  - output/deliverable06/d6_new.csv, d6_expand.csv, d6_adopt.csv  (the three lists)
  - output/deliverable06/d6_comparison_table.csv
  - output/deliverable06/d6_candidate_evidence_<category>.csv

NOTE: deterministic first pass. Verdicts use the (rough) deterministic CE match +
bound parse; LLM verification (Gate 3) firms them up. With the current 5 candidate
categories — all of which already map to a CE — expect expand/adopt, not NEW;
surfacing real NEW needs the broadened candidate generation (Track A / 6C).
"""

import json
import os
import re

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now, write_parquet
from candidates import TAXONOMY_VERSION

BASE = D6_ANALYSIS_DIR / "candidate_base_rates.parquet"
FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
CE = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
MIT = D6_ANALYSIS_DIR / "candidate_mitigation_summary.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
VERDICTS_OUT = D6_ANALYSIS_DIR / "candidate_verdicts.parquet"
COMPARISON = D6_OUTPUT_DIR / "d6_comparison_table.csv"

MATCH_THRESHOLD = 0.40            # below this, the candidate has no real CE → NEW
EXPAND_METRICS = {"acres": "max_acres", "miles": "max_miles",
                  "kv": "max_kilovolts", "mw": "max_megawatts"}

# coarse agency-token aliases (our FONSI lead_agency vs ce.json unit codes)
OUR_AGENCY_ALIASES = {
    "land management": "BLM", "department of energy": "DOE", "power marketing": "PMA",
    "forest service": "USFS", "reclamation": "BOR", "nuclear security": "NNSA",
    "indian affairs": "BIA", "ocean energy": "BOEM", "engineers": "USACE",
    "fish and wildlife": "USFWS", "national park": "NPS", "agriculture": "USDA",
}


def our_agency_tokens(agency_field) -> set[str]:
    s = str(agency_field).lower()
    toks = {code for kw, code in OUR_AGENCY_ALIASES.items() if kw in s}
    if "department of energy" in s or "power marketing" in s:
        toks.add("DOE")
    return toks


def ce_agency_tokens(unit: str) -> set[str]:
    return {t.upper() for t in re.split(r"[^A-Za-z]+", str(unit)) if len(t) >= 2}


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    base = pd.read_parquet(BASE).set_index("candidate_category")
    facts = pd.read_parquet(FACTS)
    ce = pd.read_parquet(CE) if CE.exists() else pd.DataFrame()
    mit = pd.read_parquet(MIT).set_index("candidate_category") if MIT.exists() else pd.DataFrame()
    corpus = pd.read_parquet(CORPUS)
    fonsi = corpus[corpus["is_fonsi"]].copy()

    rows = []
    for cat, brow in base.iterrows():
        cat_facts = facts[facts["candidate_category"].eq(cat)]
        prof = cat_facts[cat_facts["is_profile_subtype"]]
        focus = prof if not prof.empty else cat_facts
        n_focus = len(focus)
        role = brow["candidate_role"]

        # our FONSI agencies/states (profile subset where possible)
        cfon = fonsi[fonsi["candidate_category"].eq(cat)]
        cprof = cfon[cfon["is_profile_subtype"]] if cfon["is_profile_subtype"].any() else cfon
        our_tokens: set[str] = set()
        for a in cprof["lead_agency_harmonized"].dropna():
            our_tokens |= our_agency_tokens(a)
        n_agencies = cprof["lead_agency_harmonized"].astype(str).nunique()
        n_states = cprof["project_state"].astype(str).nunique()

        # best CE match
        best = {}
        if not ce.empty:
            top = ce[ce["candidate_category"].eq(cat)].sort_values("retrieval_rank")
            if not top.empty:
                best = top.iloc[0].to_dict()
        match_score = float(best.get("retrieval_score", 0) or 0)
        ce_units = ce_agency_tokens(best.get("agency_unit", "")) if best else set()

        # EXPAND test: our FONSIs exceeding the matched CE's parsed bound
        expand_gaps = []
        for m, fcol in EXPAND_METRICS.items():
            bound = best.get(f"bound_{m}")
            if bound is None or pd.isna(bound):
                continue
            vals = pd.to_numeric(focus.get(fcol), errors="coerce").dropna()
            n_exceed = int((vals > float(bound)).sum())
            if n_exceed >= max(2, round(0.10 * max(n_focus, 1))):
                expand_gaps.append({"metric": m, "ce_bound": float(bound),
                                    "our_max": round(float(vals.max()), 1),
                                    "n_exceeding": n_exceed})

        # ADOPT test: our agencies the matched CE's agency doesn't cover
        adopt_targets = sorted(our_tokens - ce_units) if (best and our_tokens) else []

        # verdict (priority new > expand > adopt > covered).
        # Contrast-role candidates (wind) are never an actionable recommendation.
        if role == "contrast":
            verdict = "contrast"
        elif not best or match_score < MATCH_THRESHOLD:
            verdict = "new"
        elif expand_gaps:
            verdict = "expand"
        elif adopt_targets:
            verdict = "adopt"
        else:
            verdict = "already_covered"

        # mitigation signal
        msum = mit.loc[cat].to_dict() if (not mit.empty and cat in mit.index) else {}
        mit_share = float(msum.get("mitigated_share", 0) or 0)

        # transparent multi-factor rank score (0-1)
        novelty = {"new": 1.0, "expand": 0.66, "adopt": 0.33,
                   "already_covered": 0.0, "contrast": 0.0}[verdict]
        volume = min(n_focus / 50.0, 1.0)
        diversity = min((n_agencies + n_states) / 20.0, 1.0)
        case_specific_penalty = mit_share  # high case-specific mitigation = riskier CE
        has_limits = float(focus[["max_acres", "max_miles", "max_megawatts"]].notna().any(axis=1).mean())
        rank_score = round(0.30 * novelty + 0.20 * volume + 0.15 * diversity +
                           0.15 * has_limits + 0.10 * (1 - case_specific_penalty) +
                           0.10 * (1 if role == "profile" else 0), 4)

        rows.append({
            "candidate_category": cat, "candidate_label": brow["candidate_label"],
            "role": role, "verdict": verdict, "rank_score": rank_score,
            "n_profile_fonsi": int(brow["n_profile_fonsi_projects"]),
            "n_observed_fonsi": int(brow["n_observed_fonsi_projects"]),
            "best_ce_structured_id": best.get("structured_id", ""),
            "best_ce_agency": best.get("agency_name", ""),
            "best_ce_match_score": round(match_score, 4),
            "expand_gaps": json.dumps(expand_gaps),
            "adopt_targets": ", ".join(adopt_targets),
            "our_agencies": ", ".join(sorted(our_tokens)),
            "n_agencies": n_agencies, "n_states": n_states,
            "mitigated_share": round(mit_share, 3),
            "best_ce_description": str(best.get("ce_description", ""))[:200],
            "best_ce_url": best.get("canonical_source_url", ""),
            "verdict_confidence": "low",  # deterministic; LLM verification pending (Gate 3)
            "taxonomy_version": TAXONOMY_VERSION, "run_at": run_at,
        })

    verdicts = pd.DataFrame(rows).sort_values(
        ["verdict", "rank_score"], ascending=[True, False])
    order = {"new": 0, "expand": 1, "adopt": 2, "already_covered": 3, "contrast": 4}
    verdicts["_o"] = verdicts["verdict"].map(order).fillna(9)
    verdicts = verdicts.sort_values(["_o", "rank_score"], ascending=[True, False]).drop(columns="_o")
    write_parquet(verdicts, VERDICTS_OUT)

    # the three lists
    for v, fn in (("new", "d6_new.csv"), ("expand", "d6_expand.csv"), ("adopt", "d6_adopt.csv")):
        verdicts[verdicts["verdict"].eq(v)].to_csv(D6_OUTPUT_DIR / fn, index=False)
    # comparison (all, incl. already_covered)
    verdicts.to_csv(COMPARISON, index=False)

    # per-candidate evidence tables (absorbs old n05 report-table role)
    for cat in base.index:
        f = facts[facts["candidate_category"].eq(cat)]
        f = f[f["is_profile_subtype"]] if f["is_profile_subtype"].any() else f
        cols = ["project_id", "subtype", "action_definition", "max_acres", "max_miles",
                "max_megawatts", "n_wells", "no_new_access_road", "within_existing_row",
                "previously_disturbed_land", "mitigation_dependence", "confidence",
                "citation_document_id", "citation_page"]
        f[[c for c in cols if c in f.columns]].sort_values("max_acres", ascending=False)\
            .to_csv(D6_OUTPUT_DIR / f"d6_candidate_evidence_{cat}.csv", index=False)

    print(f"[n07] verdicts -> {VERDICTS_OUT}")
    print(verdicts[["candidate_label", "verdict", "rank_score", "n_profile_fonsi",
                    "best_ce_structured_id", "adopt_targets", "expand_gaps"]].to_string(index=False))
    print(f"\n[n07] verdict counts: {verdicts['verdict'].value_counts().to_dict()}")
    print("[n07] three lists + comparison + per-candidate evidence written.")


if __name__ == "__main__":
    main()
