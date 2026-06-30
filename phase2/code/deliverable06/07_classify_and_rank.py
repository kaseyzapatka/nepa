"""D6 v2 — 07: classify candidates (new / expand / adopt) + rank + report tables.

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
bound parse — these are TEXT-SIMILAR candidate CEs pending coverage adjudication,
NOT verified coverage. The "expand"/"adopt" labels are candidate opportunities, not
confirmed findings (see report caveats). With the current candidate categories each
has a text-similar CE candidate, so expect adopt-style results, not NEW; surfacing
real NEW needs the broadened candidate generation + non-candidate clustering.
"""

import json
import os
import re

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, D6_REVIEW_DIR, ensure_d6_dirs, utc_now, write_parquet
from candidates import TAXONOMY_VERSION

BASE = D6_ANALYSIS_DIR / "candidate_base_rates.parquet"
FACTS = D6_ANALYSIS_DIR / "candidate_facts.parquet"
CE = D6_ANALYSIS_DIR / "candidate_ce_comparison.parquet"
MIT = D6_ANALYSIS_DIR / "candidate_mitigation_summary.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
VERDICTS_OUT = D6_ANALYSIS_DIR / "candidate_verdicts.parquet"
COMPARISON = D6_OUTPUT_DIR / "d6_comparison_table.csv"

# the corrected action_category candidate types, in report order (all actionable now, incl. wind)
CAND_ORDER = ["transmission_upgrade", "solar", "geothermal_exploration",
              "temporary_resource_assessment", "wind_onshore"]
MATCH_THRESHOLD = 0.40            # below this, the candidate has no real CE → NEW
EXPAND_METRICS = {"acres": "max_acres", "miles": "max_miles",
                  "kv": "max_kilovolts", "mw": "max_megawatts", "wells": "n_wells"}

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


def split_listish(v) -> list[str]:
    """Parse a list-like cell ('["Oregon", "Washington"]') into its individual items,
    so distinct-state / distinct-agency counts don't treat a multi-value string as one."""
    s = str(v).strip()
    try:
        x = json.loads(s) if s.startswith("[") else [s]
        return [str(i).strip() for i in (x if isinstance(x, list) else [x]) if str(i).strip()]
    except Exception:
        return [s] if s else []


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    facts = pd.read_parquet(FACTS)
    # "bounded" / CE-shaped = the Rule-B flag computed in 09 (LLM-bounded + transmission shape gate).
    facts["is_bounded"] = facts["is_ce_shaped"].astype(bool)
    ce = pd.read_parquet(CE) if CE.exists() else pd.DataFrame()
    mit = pd.read_parquet(MIT).set_index("candidate_category") if MIT.exists() else pd.DataFrame()

    # categories are now the corrected action_category values present in candidate_facts
    cats = [c for c in CAND_ORDER if c in set(facts["candidate_category"])]
    rows = []
    for cat in cats:
        cat_facts = facts[facts["candidate_category"].eq(cat)]
        prof = cat_facts[cat_facts["is_bounded"]]
        focus = prof if not prof.empty else cat_facts
        n_focus = len(focus)
        role = "profile"                                  # all corrected candidates are actionable
        label = str(cat_facts["candidate_label"].iloc[0])
        n_observed = int(cat_facts["project_id"].nunique())

        # our FONSI agencies/states from the CE-shaped subset (now carried in candidate_facts)
        our_tokens: set[str] = set()
        for a in focus["lead_agency_harmonized"].dropna():
            our_tokens |= our_agency_tokens(a)
        n_agencies = len({a for v in focus["lead_agency_harmonized"].dropna() for a in split_listish(v)})
        n_states = len({s for v in focus["project_state"].dropna() for s in split_listish(v)})

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
        # weighted contributions (stack to rank_score) — exposed for the classification figure
        c_novelty = round(0.30 * novelty, 4)
        c_volume = round(0.20 * volume, 4)
        c_diversity = round(0.15 * diversity, 4)
        c_limits = round(0.15 * has_limits, 4)
        c_mitigation = round(0.10 * (1 - case_specific_penalty), 4)
        c_role = round(0.10 * (1 if role == "profile" else 0), 4)
        rank_score = round(c_novelty + c_volume + c_diversity + c_limits + c_mitigation + c_role, 4)

        rows.append({
            "candidate_category": cat, "candidate_label": label,
            "role": role, "verdict": verdict, "rank_score": rank_score,
            "rank_novelty": c_novelty, "rank_volume": c_volume, "rank_diversity": c_diversity,
            "rank_limits": c_limits, "rank_mitigation": c_mitigation, "rank_role": c_role,
            "n_profile_fonsi": int(prof["project_id"].nunique()),  # bounded (rule + LLM-low-impact) projects
            "n_observed_fonsi": n_observed,
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

    # --- TOP-LEVEL: one slim, human-readable overview table ---
    def expand_note(js: str) -> str:
        gaps = json.loads(js) if js else []
        return "; ".join(f"{g['metric']}: {g['n_exceeding']} FONSIs exceed CE cap {g['ce_bound']} (up to {g['our_max']})"
                         for g in gaps)
    slim = verdicts.assign(expand_detail=verdicts["expand_gaps"].map(expand_note))[[
        "candidate_label", "verdict", "n_profile_fonsi", "best_ce_structured_id",
        "best_ce_agency", "adopt_targets", "expand_detail", "rank_score",
    ]].rename(columns={"candidate_label": "candidate", "n_profile_fonsi": "ce_shaped_fonsis",
                       "best_ce_structured_id": "existing_ce", "best_ce_agency": "existing_ce_agency"})
    slim.to_csv(COMPARISON, index=False)

    # --- REVIEW (drill-down, not client-facing): three lists + per-candidate evidence ---
    for v, fn in (("new", "d6_new.csv"), ("expand", "d6_expand.csv"), ("adopt", "d6_adopt.csv")):
        verdicts[verdicts["verdict"].eq(v)].to_csv(D6_REVIEW_DIR / fn, index=False)
    for cat in cats:
        f = facts[facts["candidate_category"].eq(cat)]
        f = f[f["is_bounded"]] if f["is_bounded"].any() else f
        cols = ["project_id", "subtype", "action_definition", "max_acres", "max_miles",
                "max_megawatts", "n_wells", "no_new_access_road", "within_existing_row",
                "previously_disturbed_land", "mitigation_dependence", "confidence",
                "citation_document_id", "citation_page"]
        f[[c for c in cols if c in f.columns]].sort_values("max_acres", ascending=False)\
            .to_csv(D6_REVIEW_DIR / f"d6_candidate_evidence_{cat}.csv", index=False)

    print(f"[07] verdicts -> {VERDICTS_OUT}")
    print(verdicts[["candidate_label", "verdict", "rank_score", "n_profile_fonsi",
                    "best_ce_structured_id", "adopt_targets", "expand_gaps"]].to_string(index=False))
    print(f"\n[07] verdict counts: {verdicts['verdict'].value_counts().to_dict()}")
    print("[07] three lists + comparison + per-candidate evidence written.")


if __name__ == "__main__":
    main()
