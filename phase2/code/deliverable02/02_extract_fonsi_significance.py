"""D2 Phase 3 — FONSI significance extraction (plan v2.11 §5).

Pipeline: deterministic candidate generation (candidate_gen) -> frozen mitigation page-window
join (mitigation_signal_matches) -> determination records (+ threshold child) via the shared
`extract_common` assembly -> optional LLM adjudication on bounded windows.

Two run modes:
  --dry-run  (KEY-FREE): builds candidates, the mitigation join, and a deterministic
             determination table (extraction_method='regex') so the whole pipeline is runnable
             WITHOUT the billable API. This is what CI / the assistant runs.
  (real)     without --dry-run: adjudicates each window with the pinned Claude model
             (extraction_method='regex+llm'), temp 0. REQUIRES the Anthropic key
             (keychain 'nepa-anthropic' / ANTHROPIC_API_KEY) + budget approval. The USER runs it.

Run (key-free):  conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --dry-run
Run (billable):  conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse

import pandas as pd

import common as C
import extract_common as X
from candidate_gen import OBLIG_SQL, ROLES_SQL, generate_fonsi_candidates


def mitigation_signal_matches() -> pd.DataFrame:
    """cue finding-span x qualifying condition-row, same-section OR page-window +/-2 (frozen §3)."""
    sql = f"""
    WITH corpus_ea AS (
        SELECT project_id FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') WHERE process_type = 'EA'
    ),
    cue AS (
        SELECT project_id, document_id, section_id, evidence_span_id, page_start, page_end
        FROM read_parquet('{C.FONSI_SPANS}')
        JOIN corpus_ea USING (project_id)
        WHERE span_type = 'finding' AND manifest_role IN ('linked_ea','canonical_fonsi','supporting_fonsi')
    ),
    qc AS (
        SELECT project_id, document_id, section_id, page_number, resource_area,
               condition_role, obligation_level, source_span_sha256, condition_text
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL}) AND obligation_level IN ({OBLIG_SQL})
    )
    SELECT cue.evidence_span_id AS cue_evidence_span_id, cue.project_id, cue.document_id,
           cue.section_id AS cue_section_id, cue.page_start AS cue_page_start,
           cue.page_end AS cue_page_end,
           qc.section_id AS condition_section_id, qc.page_number AS condition_page_number,
           qc.resource_area, qc.condition_role, qc.obligation_level,
           qc.source_span_sha256, qc.condition_text,
           CASE WHEN cue.section_id = qc.section_id THEN 'same_section' ELSE 'windowed' END AS match_type
    FROM cue JOIN qc
      ON cue.project_id = qc.project_id AND cue.document_id = qc.document_id
     AND (cue.section_id = qc.section_id
          OR qc.page_number BETWEEN cue.page_start - 2 AND cue.page_end + 2)
    """
    m = C.q(sql)
    if m.empty:
        return m
    m["condition_row_id"] = [
        C.sha256_join(p, d, s, pg, ss, cr, ob, ra) for p, d, s, pg, ss, cr, ob, ra in zip(
            m.project_id, m.document_id, m.condition_section_id, m.condition_page_number,
            m.source_span_sha256, m.condition_role, m.obligation_level, m.resource_area)]
    m["condition_text_sha256"] = m["condition_text"].map(C.sha256_text)
    return m.drop(columns=["condition_text"])


def mitigation_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Per cue finding-span (=source_unit_id): matched count + role/obligation sets."""
    if matches.empty:
        return pd.DataFrame(columns=["source_unit_id", "matched_condition_row_count",
                                     "condition_role_set", "obligation_level_set",
                                     "mitigation_resource_areas", "mitigation_same_section"])
    g = matches.groupby("cue_evidence_span_id")
    return pd.DataFrame({
        "matched_condition_row_count": g["condition_text_sha256"].nunique(),
        "condition_role_set": g["condition_role"].apply(lambda s: ",".join(sorted(set(s)))),
        "obligation_level_set": g["obligation_level"].apply(lambda s: ",".join(sorted(set(s)))),
        "mitigation_resource_areas": g["resource_area"].apply(lambda s: ",".join(sorted(set(s)))),
        "mitigation_same_section": g["match_type"].apply(lambda s: "same_section" in set(s)),
    }).reset_index().rename(columns={"cue_evidence_span_id": "source_unit_id"})


def project_context() -> pd.DataFrame:
    return C.q(f"""
    SELECT c.project_id, c.doc_type AS doc_type, c.process_type, c.agency, c.agency_scope_status,
           c.agency_scope_rule, c.time_scope_status, c.analysis_scope,
           r.decision_date, r.decision_source_type, r.decision_confidence, r.decision_is_proxy,
           r.decision_period, r.applicability_period, r.fra_overlay, r.regime_assignment_status,
           coh.cohort_by_date
    FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') c
    LEFT JOIN read_parquet('{C.PROJECT_REGIME}') r USING (project_id)
    LEFT JOIN read_parquet('{C.PROJECT_COHORTS}') coh USING (project_id)
    WHERE c.process_type = 'EA'
    """)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="key-free deterministic pass")
    ap.add_argument("--model", default=X.DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=0, help="limit candidates (debug)")
    args = ap.parse_args()
    print(f"D2 Phase 3: FONSI significance extraction — "
          f"{'DRY-RUN (regex only, key-free)' if args.dry_run else f'LLM ({args.model})'}")

    cand = generate_fonsi_candidates()
    if args.sample:
        cand = cand.sort_values("evidence_text_sha256").head(args.sample)
    C.write_parquet(cand, C.SIGNIFICANCE_SECTION_CANDIDATES, "candidates")

    matches = mitigation_signal_matches()
    C.write_parquet(matches, C.MITIGATION_SIGNAL_MATCHES, "mitigation matches")

    dets, thr = X.build_determinations(cand, mitigation_summary(matches),
                                       project_context(), args.dry_run, args.model)
    C.write_parquet(dets, C.SIGNIFICANCE_DETERMINATIONS, "determinations")
    C.write_parquet(thr, C.DETERMINATION_THRESHOLDS, "threshold child")

    X.write_manifest({
        "significance_section_candidates": "data/analysis/deliverable02/significance_section_candidates.parquet",
        "mitigation_signal_matches": "data/analysis/deliverable02/mitigation_signal_matches.parquet",
        "significance_determinations": "data/analysis/deliverable02/significance_determinations.parquet",
        "determination_thresholds": "data/analysis/deliverable02/determination_thresholds.parquet",
    }, args.dry_run, args.model)

    print(f"\ncandidates={len(cand):,}  mitigation_matches={len(matches):,}  "
          f"determinations={len(dets):,}  threshold_rows={len(thr):,}")
    print(f"mitigation_flag=TRUE: {int(dets['mitigation_flag'].sum()):,}")
    print("\ndetermination_class:\n" + dets["determination_class"].value_counts().to_string())
    if args.dry_run:
        print("\n[dry-run] every row extraction_method='regex', needs_human_review=TRUE. "
              "Run WITHOUT --dry-run (billable) to adjudicate with the LLM.")


if __name__ == "__main__":
    main()
