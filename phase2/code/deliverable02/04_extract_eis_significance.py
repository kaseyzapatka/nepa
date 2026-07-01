"""D2 Phase 3b — EIS significance extraction (plan v2.11 §5, Tier 3).

GATED in the plan behind FONSI Gate 3 + the EIS retrieval-recall spike (6.1M-page substrate).
Built now for completeness; run on a --sample first. Substrate = shared document_sections
(no per-span IDs) → source_unit_id = document_section_id. Reuses the shared `extract_common`
assembly, so the determination schema is identical to the FONSI track. EIS mitigation (ROD
commitments) is out of scope for v1 → mitigation summary empty.

Run (key-free spike):  conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --dry-run --sample 500
Run (billable):        conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --model claude-haiku-4-5-20251001 --sample 500
"""
from __future__ import annotations

import argparse

import pandas as pd

import common as C
import extract_common as X
from candidate_gen import classify_determination, resource_guess, threshold_hits


def eis_candidates(sample: int) -> pd.DataFrame:
    """Impact/consequence sections in clean-EIS corpus projects that mention significance."""
    limit = f"LIMIT {sample}" if sample else ""
    sql = f"""
    WITH corpus_eis AS (
        SELECT project_id FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') WHERE process_type = 'EIS'
    )
    SELECT s.project_id, s.document_id, s.page_start, s.page_end, s.char_start, s.char_end,
           s.heading_title, s.section_topic_guess, s.section_text
    FROM read_parquet('{C.DOCUMENT_SECTIONS}') s
    JOIN corpus_eis USING (project_id)
    WHERE s.process_type = 'EIS'
      AND (lower(s.section_text) LIKE '%significant%'
           OR lower(s.heading_title) LIKE '%environmental consequence%'
           OR lower(s.section_topic_guess) LIKE '%impact%')
      AND s.section_words BETWEEN 20 AND 4000
    ORDER BY s.project_id, s.document_id, s.page_start
    {limit}
    """
    df = C.q(sql)
    if df.empty:
        return df
    df["source_substrate"] = "document_section"
    df["source_unit_id"] = [
        C.sha256_join(p, d, ps, pe, cs, ce, h) for p, d, ps, pe, cs, ce, h in zip(
            df.project_id, df.document_id, df.page_start, df.page_end,
            df.char_start, df.char_end, df.heading_title)]
    df["section_id"] = df["source_unit_id"]
    df["span_char_start"], df["span_char_end"] = df["char_start"], df["char_end"]
    df["source_span_sha256"] = None
    cls = df["section_text"].map(classify_determination)
    df["candidate_class_guess"] = cls.map(lambda x: x[0])
    df["determination_polarity_guess"] = cls.map(lambda x: x[1])
    df["matched_cue_group"] = cls.map(lambda x: x[2])
    res = df["section_text"].map(resource_guess)
    df["resource_area_guess"] = [
        (rg[0] if rg[0] != "unknown" else (stg or "unknown"))
        for rg, stg in zip(res, df["section_topic_guess"])]
    df["resource_subarea_guess"] = res.map(lambda x: x[1])
    df["evidence_text"] = df["section_text"].str.slice(0, 4000)
    df["evidence_text_sha256"] = df["section_text"].map(C.sha256_text)
    df = df.drop(columns=["section_text", "char_start", "char_end", "section_topic_guess"])
    # keep only real determination candidates or threshold-bearing sections (bound the set)
    keep = (df["candidate_class_guess"] != "not_a_determination") | \
           df["evidence_text"].map(lambda t: bool(threshold_hits(t)))
    return df[keep].reset_index(drop=True)


def eis_context() -> pd.DataFrame:
    return C.q(f"""
    SELECT c.project_id, c.doc_type AS doc_type, c.process_type, c.agency, c.agency_scope_status,
           c.agency_scope_rule, c.time_scope_status, c.analysis_scope,
           r.decision_date, r.decision_source_type, r.decision_confidence, r.decision_is_proxy,
           r.decision_period, r.applicability_period, r.fra_overlay, r.regime_assignment_status,
           coh.cohort_by_date
    FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') c
    LEFT JOIN read_parquet('{C.PROJECT_REGIME}') r USING (project_id)
    LEFT JOIN read_parquet('{C.PROJECT_COHORTS}') coh USING (project_id)
    WHERE c.process_type = 'EIS'
    """)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--model", default=X.DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=500, help="section cap (EIS is gated; spike first)")
    ap.add_argument("--out-suffix", default="_eis", help="write to *<suffix>.parquet to not clobber FONSI")
    args = ap.parse_args()
    print(f"D2 Phase 3b: EIS significance extraction — "
          f"{'DRY-RUN' if args.dry_run else f'LLM ({args.model})'}  (sample={args.sample or 'ALL'})")

    cand = eis_candidates(args.sample)
    if cand.empty:
        print("no EIS candidates for this sample."); return
    dets, thr = X.build_determinations(cand, None, eis_context(), args.dry_run, args.model)

    sfx = args.out_suffix
    det_path = C.D2_ANALYSIS_DIR / f"significance_determinations{sfx}.parquet"
    thr_path = C.D2_ANALYSIS_DIR / f"determination_thresholds{sfx}.parquet"
    cand_path = C.D2_ANALYSIS_DIR / f"significance_section_candidates{sfx}.parquet"
    C.write_parquet(cand, cand_path, "eis candidates")
    C.write_parquet(dets, det_path, "eis determinations")
    C.write_parquet(thr, thr_path, "eis threshold child")

    print(f"\neis candidates={len(cand):,}  determinations={len(dets):,}  threshold_rows={len(thr):,}")
    print("\ndetermination_class:\n" + dets["determination_class"].value_counts().to_string())
    print("\n[note] EIS is gated behind FONSI Gate 3 + a retrieval-recall spike; "
          "outputs use a _eis suffix so they never clobber the FONSI track.")


if __name__ == "__main__":
    main()
