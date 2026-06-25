"""D6 v2 — 05 (Track B): mitigation & boundary-condition analysis.

Hunts each candidate FONSI for *where the agency drew the significance line* —
because that line is a CE boundary. Two signal families:

1. **Mitigated FONSI** — no-significant-impact conditioned on committed mitigation.
   Dual signal (after D02 `deliverable02/significance_taxonomy.py`, cues copied
   here with attribution to keep D6 self-contained):
     - textual finding cue (BLM/DOE phrasing: "would be significant absent…",
       "with incorporation of … mitigation"), AND/OR
     - enforceable conditions from `fonsi_conditions.parquet`
       (role ∈ {mitigation_commitment, enforcement_or_permit_condition},
        obligation ∈ {required, committed}).
   Supersedes the coarse `mitigation_dependence` heuristic in 03.

2. **Boundary / conditional language** — the agency's counterfactual statements
   that outline the CE boundary ("would be significant if X exceeds…", "had the
   applicant not committed to Y, an EIS would be required", "no significant impact
   provided Z"). These hand us candidate CE bounding conditions directly.

The analytical move: recurring, consistent conditions across a class → a
codifiable CE design criterion (new/expand); idiosyncratic ones → disqualifier.

Outputs:
  - data/analysis/deliverable06/candidate_mitigation_boundary.parquet  (per project)
  - data/analysis/deliverable06/candidate_mitigation_summary.parquet   (per candidate; feeds 07)
  - output/deliverable06/candidate_mitigation_boundary_review.csv
"""

import os
import re

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import json

import duckdb
import pandas as pd

from common import D6_ANALYSIS_DIR, D6_REVIEW_DIR, ensure_d6_dirs, normalize_space, utc_now, write_parquet
from candidates import TAXONOMY_VERSION

PACKETS = D6_ANALYSIS_DIR / "candidate_evidence_packets.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
CONDITIONS = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
INVENTORY = D6_ANALYSIS_DIR / "fonsi_project_inventory.parquet"
FULL_PACKETS = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
OUT = D6_ANALYSIS_DIR / "candidate_mitigation_boundary.parquet"
SUMMARY_OUT = D6_ANALYSIS_DIR / "candidate_mitigation_summary.parquet"
CORPUS_STATS_OUT = D6_ANALYSIS_DIR / "corpus_mitigation_stats.parquet"
REVIEW = D6_REVIEW_DIR / "candidate_mitigation_boundary_review.csv"

# --- mitigated-FONSI textual cues (from D02 significance_taxonomy.py) ---
MITIGATED_CUES = re.compile(
    r"would be significant (?:absent|without|if|unless)|"
    r"with (?:the )?(?:incorporation|implementation|inclusion) of .{0,60}(?:mitigation|measure|condition)|"
    r"(?:mitigated|reduced) to (?:a level that is |)(?:below |less than )significan|"
    r"(?:reduce|minimize|lessen)[a-z]* .{0,50}(?:below|to less than|to a level) .{0,25}significan|"
    r"less[- ]than[- ]significant with mitigation", re.IGNORECASE)

# --- boundary / conditional language (NEW per feedback): the agency's own
#     "if X then significant / would require an EIS / provided Y → FONSI" lines ---
BOUNDARY_CUES = re.compile(
    r"would be significant (?:if|absent|without|unless)|"
    r"had .{0,40} not |"
    r"would (?:otherwise )?(?:require|trigger|warrant) an? (?:ea|eis|environmental (?:impact statement|assessment))|"
    r"(?:provided|so long as|as long as|conditioned (?:up)?on|only if) (?:that )?|"
    r"(?:if|where) .{0,60}(?:exceed|greater than|more than) .{0,50}(?:significant|require|eis)|"
    r"unless .{0,40}(?:mitigat|exceed)", re.IGNORECASE)

MIT_ROLES = ("mitigation_commitment", "enforcement_or_permit_condition")
MIT_OBLIG = ("required", "committed")


def extract_boundary_statements(text: str, limit: int = 4) -> list[str]:
    """Return sentence-ish windows around boundary-cue hits."""
    out, seen = [], set()
    for m in BOUNDARY_CUES.finditer(text or ""):
        start = text.rfind(".", 0, m.start()) + 1
        end = text.find(".", m.end())
        end = end if end != -1 else m.end() + 160
        snippet = normalize_space(text[start:end])[:280]
        key = snippet.lower()
        if snippet and key not in seen:
            seen.add(key)
            out.append(snippet)
        if len(out) >= limit:
            break
    return out


def corpus_mitigation_stats(run_at: str) -> pd.DataFrame:
    """Mitigated-FONSI tally across the FULL clean-energy EA-source FONSI corpus
    (all 452, not just the candidate subset), so the report can state how many of
    the corpus are mitigated FONSIs. Same dual signal as the per-candidate pass:
    textual finding/boundary cue OR >=1 enforceable committed condition."""
    inv = pd.read_parquet(INVENTORY)
    clean = set(inv.loc[inv["project_energy_type"] == "Clean", "project_id"].astype(str))
    pk = pd.read_parquet(FULL_PACKETS, columns=["project_id", "finding_text", "boundary_text"])
    pk["project_id"] = pk["project_id"].astype(str)
    pk = pk[pk["project_id"].isin(clean)].drop_duplicates("project_id")

    enf_ids: set[str] = set()
    if CONDITIONS.exists() and clean:
        ids = ",".join(f"'{p}'" for p in clean)
        enf = duckdb.connect().execute(
            f"""select distinct cast(project_id as varchar) pid from read_parquet('{CONDITIONS}')
                where cast(project_id as varchar) in ({ids})
                  and condition_role in {MIT_ROLES} and obligation_level in {MIT_OBLIG}"""
        ).df()
        enf_ids = set(enf["pid"])

    textual = (pk["finding_text"].fillna("") + " " + pk["boundary_text"].fillna("")) \
        .map(lambda t: bool(MITIGATED_CUES.search(t)))
    has_enf = pk["project_id"].isin(enf_ids)
    is_mit = textual | has_enf
    stats = pd.DataFrame([{
        "n_clean_fonsi": len(clean),
        "n_with_packet": len(pk),
        "n_mitigated_fonsi": int(is_mit.sum()),
        "mitigated_share": round(float(is_mit.mean()), 3) if len(pk) else 0.0,
        "n_textual_only": int((textual & ~has_enf).sum()),
        "n_enforceable_only": int((~textual & has_enf).sum()),
        "n_both_high_conf": int((textual & has_enf).sum()),
        "run_at": run_at,
    }])
    write_parquet(stats, CORPUS_STATS_OUT)
    return stats


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()
    packets = pd.read_parquet(PACKETS)
    packets["project_id"] = packets["project_id"].astype(str)
    corpus = pd.read_parquet(CORPUS)
    fonsi = corpus.loc[corpus["is_fonsi"]].copy()
    fonsi["project_id"] = fonsi["project_id"].astype(str)

    # enforceable conditions per project (reuse v1 fonsi_conditions)
    enf_by_project: dict[str, pd.DataFrame] = {}
    if CONDITIONS.exists():
        ids = ",".join(f"'{p}'" for p in packets["project_id"].unique())
        cond = duckdb.connect().execute(
            f"""select project_id, condition_role, obligation_level, resource_area, condition_text
                from read_parquet('{CONDITIONS}')
                where cast(project_id as varchar) in ({ids})
                  and condition_role in {MIT_ROLES} and obligation_level in {MIT_OBLIG}"""
        ).df()
        cond["project_id"] = cond["project_id"].astype(str)
        enf_by_project = {pid: g for pid, g in cond.groupby("project_id")}

    # per-project mitigation/boundary record (text shared across a project's categories)
    per_project: dict[str, dict] = {}
    for r in packets.itertuples(index=False):
        pid = r.project_id
        finding = " ".join(getattr(r, c, "") or "" for c in ("finding_text", "boundary_text"))
        textual = bool(MITIGATED_CUES.search(finding))
        enf = enf_by_project.get(pid)
        n_enf = 0 if enf is None else len(enf)
        areas = [] if enf is None else sorted({a for a in enf["resource_area"].dropna().astype(str)
                                               if a and a != "unknown"})
        is_mit = textual or n_enf > 0
        conf = "high" if (textual and n_enf > 0) else ("medium" if is_mit else "none")
        per_project[pid] = {
            "is_mitigated_fonsi": is_mit,
            "mitigation_confidence": conf,
            "mitigated_textual_cue": textual,
            "n_enforceable_conditions": n_enf,
            "mitigation_resource_areas": ", ".join(areas[:8]),
            "boundary_statements": json.dumps(extract_boundary_statements(finding)),
        }

    rows = []
    for fr in fonsi.itertuples(index=False):
        pid = fr.project_id
        if pid not in per_project:
            continue
        b = per_project[pid]
        rows.append({
            "project_id": pid, "candidate_category": fr.candidate_category,
            "subtype": fr.subtype, "is_profile_subtype": bool(fr.is_profile_subtype),
            **b, "run_at": run_at, "taxonomy_version": TAXONOMY_VERSION,
        })
    out = pd.DataFrame(rows)
    write_parquet(out, OUT)

    # per-candidate summary (profile subset) → feeds 07
    summary = []
    for cat, grp in out.groupby("candidate_category"):
        prof = grp[grp["is_profile_subtype"]]
        focus = prof if not prof.empty else grp
        n = len(focus)
        n_mit = int(focus["is_mitigated_fonsi"].sum())
        # recurring resource areas across mitigated projects in the class
        area_counts: dict[str, int] = {}
        for a in focus["mitigation_resource_areas"]:
            for x in [x.strip() for x in str(a).split(",") if x.strip()]:
                area_counts[x] = area_counts.get(x, 0) + 1
        top_areas = sorted(area_counts.items(), key=lambda kv: -kv[1])[:5]
        bstmts = [s for js in focus["boundary_statements"] for s in json.loads(js)]
        summary.append({
            "candidate_category": cat,
            "n_focus": n,
            "n_mitigated_fonsi": n_mit,
            "mitigated_share": round(n_mit / n, 3) if n else 0.0,
            "n_with_boundary_language": int((focus["boundary_statements"] != "[]").sum()),
            "top_mitigation_resource_areas": "; ".join(f"{k}({v})" for k, v in top_areas),
            "example_boundary_statements": json.dumps(bstmts[:5]),
            "run_at": run_at,
        })
    summ = pd.DataFrame(summary).sort_values("n_mitigated_fonsi", ascending=False)
    write_parquet(summ, SUMMARY_OUT)

    review_cols = ["project_id", "candidate_category", "subtype", "is_mitigated_fonsi",
                   "mitigation_confidence", "n_enforceable_conditions",
                   "mitigation_resource_areas", "boundary_statements"]
    out[review_cols].sort_values(["candidate_category", "project_id"]).to_csv(REVIEW, index=False)

    cstats = corpus_mitigation_stats(run_at)
    cs = cstats.iloc[0]
    print(f"[05] corpus-wide mitigated FONSIs: {int(cs.n_mitigated_fonsi)} of {int(cs.n_clean_fonsi)} "
          f"clean FONSIs ({cs.mitigated_share:.1%}) -> {CORPUS_STATS_OUT.name}")
    print(f"[05] candidate-scope mitigation/boundary rows={len(out)} -> {OUT}")
    print(f"[05] mitigated FONSIs (candidate scope): {int(out['is_mitigated_fonsi'].sum())} of {len(out)} "
          f"(textual+enforceable dual signal)")
    print(f"[05] projects with boundary language: {int((out['boundary_statements'] != '[]').sum())}")
    print(f"\n[05] per-candidate summary -> {SUMMARY_OUT}")
    print(summ[["candidate_category", "n_focus", "n_mitigated_fonsi", "mitigated_share",
                "n_with_boundary_language"]].to_string(index=False))


if __name__ == "__main__":
    main()
