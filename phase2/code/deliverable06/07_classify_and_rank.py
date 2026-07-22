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
from ce_agency_crosswalk import is_covered

# G1 (refactor.md §5): client develop-shortlist recurrence gate — main >= 5 CE-shaped, exploratory 3-4.
SHORTLIST_R_MAIN = 5
SHORTLIST_R_EXPLORATORY = 3

# A3: fixed component weights of the opportunity rank_score (stack to 1.0). Exposed for the systematic
# weight-sensitivity analysis (rank_sensitivity()); mirrored by the c_* multipliers below.
RANK_WEIGHTS = {"novelty": 0.30, "volume": 0.20, "diversity": 0.15,
                "limits": 0.15, "mitigation": 0.10, "role": 0.10}
RANK_COMPONENTS = list(RANK_WEIGHTS)

# A1/#37: eCFR coverage gate. candidate_ce_coverage.parquet carries the reviewer's per-CE adjudication
# (covers/partially_covers/does_not_cover/unclear) over the top-5 CEs of each adopt/expand cell. The
# CELL-BEST coverage = the strongest verdict any of a cell's top-5 CEs earned (covers > partially >
# unclear > does_not_cover). Gate mapping (applied BEFORE G1 tiering so a flip flows through it):
#   does_not_cover   -> FLIP adopt/expand -> new (no CE covers it -> it is genuinely develop)
#   covers           -> keep verdict, verdict_confidence low -> "verified" (verified current-eCFR text)
#   partially_covers -> keep verdict, verdict_confidence -> "partial" (expand cells land here by design)
#   unclear          -> keep verdict, confidence stays low, needs_review=True
# Existence-guarded: a missing/unfilled coverage file leaves verdicts exactly as the deterministic pass
# produced them (loud warning). Only the 19 row-level "covers" (from verified eCFR-current text) promote
# a cell to "verified"; agency-doc / legacy-URL CEs are capped at partial/unclear/does_not_cover.
COVERAGE = D6_ANALYSIS_DIR / "candidate_ce_coverage.parquet"
COVERAGE_STRENGTH = {"covers": 4, "partially_covers": 3, "unclear": 2, "does_not_cover": 1}

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


def rank_sensitivity(verdicts: pd.DataFrame, n_draws: int = 2000, seed: int = 47) -> pd.DataFrame:
    """A3 — systematic weight-sensitivity of the opportunity ranking (replaces the informal 3-weighting
    table). Recovers the raw rank components (stored as weighted contributions rank_* = w * raw), then:
      (a) DIRICHLET sweep: n_draws weight vectors on the 6-simplex -> each cell's rank distribution;
      (b) ONE-AT-A-TIME: perturb each weight +/-50% (renormalized) -> each cell's rank swing.
    Reportable cells = non-contrast with >= 2 CE-shaped FONSIs (matches the report table)."""
    import numpy as np
    rc = verdicts[(verdicts["verdict"] != "contrast") & (verdicts["n_profile_fonsi"] >= 2)].copy()
    if rc.empty:
        return pd.DataFrame()
    # recover raw components: raw = weighted_contribution / original_weight
    raw = np.column_stack([
        rc[f"rank_{c}"].to_numpy(dtype=float) / RANK_WEIGHTS[c] for c in RANK_COMPONENTS
    ])  # (n_cells, 6)
    labels = rc["candidate_label"].to_numpy()
    w0 = np.array([RANK_WEIGHTS[c] for c in RANK_COMPONENTS])

    def ranks_for(weights):
        scores = raw @ np.asarray(weights)
        # rank 1 = highest score (ties -> min rank), like the report table
        order = (-scores).argsort()
        r = np.empty(len(scores), dtype=int)
        r[order] = np.arange(1, len(scores) + 1)
        # tie handling: equal scores share the min rank
        for s in np.unique(scores):
            idx = np.where(scores == s)[0]
            r[idx] = r[idx].min()
        return r

    point_rank = ranks_for(w0)
    rng = np.random.default_rng(seed)
    draws = rng.dirichlet(np.ones(len(RANK_COMPONENTS)), size=n_draws)
    rank_draws = np.array([ranks_for(w) for w in draws])          # (n_draws, n_cells)
    # OAT: each weight +/-50%, renormalized
    oat = {}
    for i, comp in enumerate(RANK_COMPONENTS):
        for sign, tag in ((1.5, "up"), (0.5, "dn")):
            w = w0.copy(); w[i] *= sign; w /= w.sum()
            oat[f"oat_{comp}_{tag}"] = ranks_for(w)

    out = pd.DataFrame({
        "candidate_label": labels,
        "point_rank": point_rank,
        "rank_median": np.median(rank_draws, axis=0).astype(int),
        "rank_p25": np.percentile(rank_draws, 25, axis=0).astype(int),
        "rank_p75": np.percentile(rank_draws, 75, axis=0).astype(int),
        "rank_best": rank_draws.min(axis=0),
        "rank_worst": rank_draws.max(axis=0),
        "pct_top3": (rank_draws <= 3).mean(axis=0).round(3),
    })
    for k, v in oat.items():
        out[k] = v
    return out.sort_values("point_rank").reset_index(drop=True)


def apply_coverage_gate(verdicts: pd.DataFrame) -> pd.DataFrame:
    """A1/#37 — gate adopt/expand verdicts on the eCFR coverage adjudication (see header). Adds
    cell_best_coverage / coverage_source / needs_review, updates verdict + verdict_confidence.
    Existence-guarded: missing or unfilled coverage file -> verdicts unchanged (loud warning)."""
    verdicts = verdicts.copy()
    verdicts["cell_best_coverage"] = ""
    verdicts["coverage_source"] = ""
    verdicts["needs_review"] = False
    if not COVERAGE.exists():
        print("[07][A1] WARNING: candidate_ce_coverage.parquet not found — coverage gate SKIPPED; "
              "adopt/expand verdicts remain deterministic (unverified). Run ce_ecfr_verify.py + adjudicate.")
        return verdicts
    cov = pd.read_parquet(COVERAGE)
    if "coverage_verdict" not in cov.columns or (cov["coverage_verdict"].fillna("") == "").all():
        print("[07][A1] WARNING: candidate_ce_coverage.parquet has no filled coverage_verdict — gate "
              "SKIPPED; verdicts remain unverified. Fill it via ce_ecfr_apply_verdicts.py.")
        return verdicts
    cov = cov[cov["coverage_verdict"].isin(COVERAGE_STRENGTH)].copy()
    cov["_ord"] = cov["coverage_verdict"].map(COVERAGE_STRENGTH)
    cell_best = (cov.sort_values(["candidate_category", "_ord", "retrieval_rank"],
                                 ascending=[True, False, True])
                    .groupby("candidate_category").first().reset_index())
    best_cov = dict(zip(cell_best["candidate_category"], cell_best["coverage_verdict"]))
    best_src = dict(zip(cell_best["candidate_category"], cell_best["source_type"]))

    n_flip = n_verified = n_partial = n_unclear = 0
    for i, r in verdicts.iterrows():
        cat = r["candidate_category"]
        cb = best_cov.get(cat)
        if cb is None:
            continue
        verdicts.at[i, "cell_best_coverage"] = cb
        verdicts.at[i, "coverage_source"] = best_src.get(cat, "")
        if cb == "does_not_cover":
            # flip adopt/expand -> new; make novelty consistent with a develop cell and re-derive rank_score
            if r["verdict"] in ("adopt", "expand"):
                old_c_nov = float(r["rank_novelty"])
                new_c_nov = round(RANK_WEIGHTS["novelty"] * 1.0, 4)   # develop novelty = 1.0
                verdicts.at[i, "verdict"] = "new"
                verdicts.at[i, "rank_novelty"] = new_c_nov
                verdicts.at[i, "rank_score"] = round(float(r["rank_score"]) - old_c_nov + new_c_nov, 4)
                verdicts.at[i, "verdict_confidence"] = "low"   # develop cell (evidence-strong flip, dev-standard confidence)
                n_flip += 1
        elif cb == "covers":
            verdicts.at[i, "verdict_confidence"] = "verified"
            n_verified += 1
        elif cb == "partially_covers":
            verdicts.at[i, "verdict_confidence"] = "partial"
            n_partial += 1
        elif cb == "unclear":
            verdicts.at[i, "needs_review"] = True
            n_unclear += 1
    print(f"[07][A1] coverage gate: {n_flip} flipped adopt/expand->new (does_not_cover), "
          f"{n_verified} verified (covers), {n_partial} partial, {n_unclear} needs_review (unclear)")
    return verdicts


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    facts = pd.read_parquet(FACTS)
    # "bounded" / CE-shaped = the Rule-B flag computed in 09 (LLM-bounded + transmission shape gate).
    facts["is_bounded"] = facts["is_ce_shaped"].astype(bool)
    ce = pd.read_parquet(CE) if CE.exists() else pd.DataFrame()
    mit = pd.read_parquet(MIT).set_index("candidate_category") if MIT.exists() else pd.DataFrame()

    # categories are now the tech_group x action grid cells present in candidate_facts (refactor)
    cats = sorted(set(facts["candidate_category"]))
    rows = []
    for cat in cats:
        cat_facts = facts[facts["candidate_category"].eq(cat)]
        prof = cat_facts[cat_facts["is_bounded"]]
        focus = prof if not prof.empty else cat_facts
        n_focus = len(focus)
        role = "profile"                                  # all corrected candidates are actionable
        label = str(cat_facts["candidate_label"].iloc[0])
        cell_tech = str(cat_facts["tech_group"].iloc[0]) if "tech_group" in cat_facts.columns else ""
        cell_action = str(cat_facts["action"].iloc[0]) if "action" in cat_facts.columns else ""
        cell_codif = bool(cat_facts["is_codifiable"].iloc[0]) if "is_codifiable" in cat_facts.columns else True
        n_observed = int(cat_facts["project_id"].nunique())

        # our FONSI agencies/states from the CE-shaped subset (now carried in candidate_facts)
        our_tokens: set[str] = set()
        for a in focus["lead_agency_harmonized"].dropna():
            our_tokens |= our_agency_tokens(a)
        n_agencies = len({a for v in focus["lead_agency_harmonized"].dropna() for a in split_listish(v)})
        n_states = len({s for v in focus["project_state"].dropna() for s in split_listish(v)})

        # best CE match
        best = {}
        ce_units_ranks18: set[str] = set()
        if not ce.empty:
            top = ce[ce["candidate_category"].eq(cat)].sort_values("retrieval_rank")
            if not top.empty:
                best = top.iloc[0].to_dict()
                # #38: agency tokens across ranks 1-8 (for the net-gap crosswalk annotation)
                for u in top.head(8)["agency_unit"].dropna():
                    ce_units_ranks18 |= ce_agency_tokens(u)
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
        # #38 (annotate-only): net targets after removing agencies already covered — by themselves,
        # their parent department, or a dept sibling — via ANY CE in ranks 1-8. verdict is NOT changed.
        adopt_targets_gross = adopt_targets
        adopt_targets_net = sorted(
            t for t in adopt_targets if not is_covered(t, ce_units_ranks18)
        ) if adopt_targets else []

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
        # weighted contributions (stack to rank_score) — exposed for the classification figure + A3
        c_novelty = round(RANK_WEIGHTS["novelty"] * novelty, 4)
        c_volume = round(RANK_WEIGHTS["volume"] * volume, 4)
        c_diversity = round(RANK_WEIGHTS["diversity"] * diversity, 4)
        c_limits = round(RANK_WEIGHTS["limits"] * has_limits, 4)
        c_mitigation = round(RANK_WEIGHTS["mitigation"] * (1 - case_specific_penalty), 4)
        c_role = round(RANK_WEIGHTS["role"] * (1 if role == "profile" else 0), 4)
        rank_score = round(c_novelty + c_volume + c_diversity + c_limits + c_mitigation + c_role, 4)

        rows.append({
            "candidate_category": cat, "candidate_label": label,
            "tech_group": cell_tech, "action": cell_action, "is_codifiable": cell_codif,
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
            "adopt_targets_gross": ", ".join(adopt_targets_gross),
            "adopt_targets_net": ", ".join(adopt_targets_net),
            "our_agencies": ", ".join(sorted(our_tokens)),
            "n_agencies": n_agencies, "n_states": n_states,
            "mitigated_share": round(mit_share, 3),
            "best_ce_description": str(best.get("ce_description", ""))[:200],
            "best_ce_url": best.get("canonical_source_url", ""),
            "verdict_confidence": "low",  # deterministic; LLM verification pending (Gate 3)
            "taxonomy_version": TAXONOMY_VERSION, "run_at": run_at,
        })

    verdicts = pd.DataFrame(rows)

    # --- A1/#37: eCFR coverage gate (may flip adopt/expand -> new; must precede G1 tiering) ---
    verdicts = apply_coverage_gate(verdicts)

    order = {"new": 0, "expand": 1, "adopt": 2, "already_covered": 3, "contrast": 4}
    verdicts["_o"] = verdicts["verdict"].map(order).fillna(9)
    verdicts = verdicts.sort_values(["_o", "rank_score"], ascending=[True, False]).drop(columns="_o")

    # --- G1: minimum-recurrence gate for the CLIENT develop shortlist (refactor.md §5) ---
    # develop (new) + codifiable cells get a tier by their CE-shaped count (n_profile_fonsi):
    #   main >= 5 · exploratory 3-4 · dropped < 3 (kept in verdicts for grid coloring, cut from d6_new.csv).
    def _shortlist_tier(r) -> str:
        if r["verdict"] != "new" or not bool(r["is_codifiable"]):
            return ""
        n = int(r["n_profile_fonsi"])
        if n >= SHORTLIST_R_MAIN:
            return "main"
        if n >= SHORTLIST_R_EXPLORATORY:
            return "exploratory"
        return "dropped"
    verdicts["shortlist_tier"] = verdicts.apply(_shortlist_tier, axis=1)
    write_parquet(verdicts, VERDICTS_OUT)

    # --- A3: systematic weight-sensitivity of the ranking (Dirichlet + one-at-a-time) ---
    sens = rank_sensitivity(verdicts)
    sens.to_csv(D6_OUTPUT_DIR / "rank_sensitivity.csv", index=False)
    print(f"[07][A3] rank sensitivity ({len(sens)} cells, 2000 Dirichlet draws) -> "
          f"{D6_OUTPUT_DIR / 'rank_sensitivity.csv'}")

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
    # d6_new.csv is the CLIENT develop shortlist -> exclude non-codifiable cells (manufacturing,
    # land/ROW authorization), then apply the G1 recurrence gate (drop < 3 CE-shaped).
    for v, fn in (("new", "d6_new.csv"), ("expand", "d6_expand.csv"), ("adopt", "d6_adopt.csv")):
        sub = verdicts[verdicts["verdict"].eq(v)]
        if v == "new":
            sub = sub[sub["is_codifiable"] == True]
            dropped = sub[sub["shortlist_tier"] == "dropped"]
            for _, d in dropped.sort_values("n_profile_fonsi").iterrows():
                print(f"[07][G1] DROP from develop shortlist (n_ce_shaped={int(d['n_profile_fonsi'])} "
                      f"< {SHORTLIST_R_EXPLORATORY}): {d['candidate_label']}")
            n_main = int((sub["shortlist_tier"] == "main").sum())
            n_expl = int((sub["shortlist_tier"] == "exploratory").sum())
            print(f"[07][G1] develop shortlist: {n_main} main + {n_expl} exploratory, "
                  f"{len(dropped)} dropped (was {len(sub)} codifiable-new cells)")
            sub = sub[sub["shortlist_tier"].isin(["main", "exploratory"])]
        sub.to_csv(D6_REVIEW_DIR / fn, index=False)
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
