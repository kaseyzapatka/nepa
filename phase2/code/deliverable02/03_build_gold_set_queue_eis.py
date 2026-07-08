"""D2 Phase 5 (EIS) — build the EIS gold-set labeling worksheet (deterministic; NO LLM, NO gold).

The EIS parallel of `03_build_gold_set_queue.py`. Same approach (stratified reading list of windows
for two independent reviewers to hand-label), but on the **EIS substrate** (`document_sections`,
process_type='EIS') and with **DISTINCT output files** so the EIS and FONSI gold sets never mix.

Grain matches the extractor: the window key is `evidence_span_id` = the section's `source_unit_id`
(document_section_id), which is exactly what `04_extract_eis_significance.py` writes as
`evidence_span_id`, so the labels join to the EIS determinations table 1:1.

Reviewers write a SEPARATE long CSV (one row per resource determination) per gold_labeling_eis.md;
merge with `gold_agreement.py --track eis`, validate with `05_validate_significance.py --track eis`.

Run:  conda run -n nepa python phase2/code/deliverable02/03_build_gold_set_queue_eis.py
Out:  phase2/output/deliverable02/significance_gold_queue_eis.csv   (windows to read)
      phase2/data/analysis/deliverable02/gold/significance_gold_queue_eis.parquet
"""
from __future__ import annotations

import pandas as pd

import common as C
from candidate_gen import classify_determination, resource_guess, threshold_hits

N_POS, N_NEG, FLOOR = 300, 100, 12
POOL = 12000  # deterministic md5-ordered sampling pool (bounds classification cost; ~EIS universe)


def _stratified(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """Proportional-with-floor stratified sample, ordered deterministically by evidence hash
    (identical logic to the FONSI 03)."""
    if df.empty:
        return df
    df = df.sort_values("evidence_text_sha256")
    picks, total = [], len(df)
    for _, sub in df.groupby(by, sort=True):
        quota = min(len(sub), max(FLOOR, round(n * len(sub) / total)))
        picks.append(sub.head(quota))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:
        out = out.sort_values("evidence_text_sha256").head(n)
    return out


def eis_gold_frame(pool: int = POOL) -> pd.DataFrame:
    """Classified EIS impact/consequence sections, NO keep filter (so `not_a_determination`
    negatives are retained). A deterministic md5-ordered pool bounds classification cost. The
    section query mirrors 04_extract_eis_significance.eis_candidates (kept in sync by hand)."""
    sql = f"""
    WITH corpus_eis AS (
        SELECT project_id, agency_scope_status
        FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') WHERE process_type = 'EIS'
    )
    SELECT s.project_id, s.document_id, s.page_start, s.page_end, s.char_start, s.char_end,
           s.heading_title, s.section_topic_guess, s.section_text, c.agency_scope_status
    FROM read_parquet('{C.DOCUMENT_SECTIONS}') s
    JOIN corpus_eis c USING (project_id)
    WHERE s.process_type = 'EIS'
      AND (lower(s.section_text) LIKE '%significant%'
           OR lower(s.heading_title) LIKE '%environmental consequence%'
           OR lower(s.section_topic_guess) LIKE '%impact%')
      AND s.section_words BETWEEN 20 AND 4000
    ORDER BY md5(concat_ws('|', s.project_id, s.document_id, CAST(s.page_start AS VARCHAR),
                           CAST(s.page_end AS VARCHAR), CAST(s.char_start AS VARCHAR),
                           CAST(s.char_end AS VARCHAR), s.heading_title))
    LIMIT {pool}
    """
    df = C.q(sql)
    if df.empty:
        return df
    df["source_substrate"] = "document_section"
    df["source_unit_id"] = [
        C.sha256_join(p, d, ps, pe, cs, ce, h) for p, d, ps, pe, cs, ce, h in zip(
            df.project_id, df.document_id, df.page_start, df.page_end,
            df.char_start, df.char_end, df.heading_title)]
    df["evidence_span_id"] = df["source_unit_id"]   # window key (== EIS determinations' evidence_span_id)
    df["section_id"] = df["source_unit_id"]
    cls = df["section_text"].map(classify_determination)
    df["candidate_class_guess"] = cls.map(lambda x: x[0])
    df["determination_polarity_guess"] = cls.map(lambda x: x[1])
    df["matched_cue_group"] = cls.map(lambda x: x[2])
    res = df["section_text"].map(resource_guess)
    df["resource_area_guess"] = [
        (rg[0] if rg[0] != "unknown" else (stg or "unknown"))
        for rg, stg in zip(res, df["section_topic_guess"])]
    df["resource_subarea_guess"] = res.map(lambda x: x[1])
    df["threshold_types_guess"] = df["section_text"].map(lambda t: ",".join(threshold_hits(t)))
    df["evidence_text"] = df["section_text"].str.slice(0, C.WINDOW_CHAR_CAP)
    df["evidence_text_sha256"] = df["section_text"].map(C.sha256_text)
    return df.drop(columns=["section_text", "char_start", "char_end", "section_topic_guess"])


def main() -> None:
    print("D2 Phase 5 (EIS): building EIS gold-set labeling queue ...")
    frame = eis_gold_frame()
    if frame.empty:
        print("no EIS candidate sections found — is the EIS corpus (significance_corpus / "
              "document_sections) built?"); return

    pos = frame[frame["candidate_class_guess"] != "not_a_determination"].copy()
    neg = frame[(frame["candidate_class_guess"] == "not_a_determination") &
                (frame["threshold_types_guess"] == "")].copy()

    pos_sample = _stratified(pos, "candidate_class_guess", N_POS)
    neg_sample = _stratified(neg, "agency_scope_status", N_NEG)
    queue = pd.concat([pos_sample, neg_sample], ignore_index=True)

    queue["gold_queue_run_at"] = C.utc_now()
    queue["schema_version"] = C.SCHEMA_VERSION

    # reading list only — labelers write a SEPARATE long CSV (one row per resource determination)
    order = ["project_id", "document_id", "section_id", "evidence_span_id", "source_substrate",
             "agency_scope_status", "page_start", "page_end", "heading_title",
             "candidate_class_guess", "determination_polarity_guess", "matched_cue_group",
             "resource_area_guess", "resource_subarea_guess", "threshold_types_guess",
             "evidence_text", "evidence_text_sha256", "gold_queue_run_at", "schema_version"]
    queue = queue[order]

    C.write_parquet(queue, C.GOLD_QUEUE_EIS, "eis gold queue")
    C.write_csv(queue, C.GOLD_QUEUE_EIS_CSV, "eis gold queue (analyst labels this)")

    print(f"\nqueued {len(queue):,} EIS candidates ({len(pos_sample)} positives + {len(neg_sample)} negatives)"
          f"  from pool of {len(frame):,} classified sections")
    print("\npositives by candidate_class_guess:")
    print(pos_sample["candidate_class_guess"].value_counts().to_string())
    if not neg_sample.empty:
        print("\nnegatives by agency_scope_status:")
        print(neg_sample["agency_scope_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
