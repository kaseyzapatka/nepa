"""D2 Phase 5 — build the gold-set labeling worksheet (deterministic; NO LLM, NO gold yet).

Emits a stratified queue of candidate determination WINDOWS for the analyst to HAND-LABEL
(plan §7). ~300 determination candidates + ~100 negatives, stratified by candidate class,
agency scope, threshold presence, and mitigation link. Sampling is deterministic (ordered by
evidence hash — reproducible, no seed).

MULTI-DETERMINATION GRAIN (2026-07-08): the extractor emits one row per
(window × resource_area × determination), so this worksheet is a *reading list* of windows, NOT
a fill-in-place answer sheet. Each labeler reads every window and writes a SEPARATE long CSV with
one row per resource-area determination the window concludes on (see gold_labeling.md). The two
labelers' long CSVs are merged by `gold_agreement.py` into `gold/significance_gold.parquet`.

Run:  conda run -n nepa python phase2/code/deliverable02/03_build_gold_set_queue.py
Out:  phase2/output/deliverable02/significance_gold_queue.csv   (windows to read)
      phase2/data/analysis/deliverable02/gold/significance_gold_queue.parquet
"""
from __future__ import annotations

import pandas as pd

import common as C
from candidate_gen import generate_fonsi_candidates

N_POS, N_NEG, FLOOR = 300, 100, 15


def _stratified(df: pd.DataFrame, by: str, n: int) -> pd.DataFrame:
    """Proportional-with-floor stratified sample, ordered deterministically by evidence hash."""
    if df.empty:
        return df
    df = df.sort_values("evidence_text_sha256")
    picks = []
    total = len(df)
    for _, sub in df.groupby(by, sort=True):
        quota = min(len(sub), max(FLOOR, round(n * len(sub) / total)))
        picks.append(sub.head(quota))
    out = pd.concat(picks, ignore_index=True)
    if len(out) > n:  # trim deterministically, keeping >=1 per stratum already guaranteed by floor
        out = out.sort_values("evidence_text_sha256").head(n)
    return out


def main() -> None:
    print("D2 Phase 5: building gold-set labeling queue ...")
    cand = generate_fonsi_candidates()
    scope = C.q(f"SELECT project_id, agency_scope_status FROM read_parquet('{C.SIGNIFICANCE_CORPUS}')")
    cand = cand.merge(scope, on="project_id", how="left")

    pos = cand[cand["candidate_class_guess"] != "not_a_determination"].copy()
    neg = cand[cand["candidate_class_guess"] == "not_a_determination"].copy()

    pos_sample = _stratified(pos, "candidate_class_guess", N_POS)
    # guarantee the mitigated-FONSI centerpiece is represented: >=50 mitigation-linked candidates
    N_MIT = 50
    in_sample = set(pos_sample["evidence_span_id"])
    have_mit = int(pos_sample["has_qual_cond_windowed"].sum())
    if have_mit < N_MIT:
        mit_pool = pos[pos["has_qual_cond_windowed"] & ~pos["evidence_span_id"].isin(in_sample)]
        mit_extra = mit_pool.sort_values("evidence_text_sha256").head(N_MIT - have_mit)
        pos_sample = pd.concat([pos_sample, mit_extra], ignore_index=True)

    neg_sample = _stratified(neg, "agency_scope_status", N_NEG)
    queue = pd.concat([pos_sample, neg_sample], ignore_index=True)

    queue["gold_queue_run_at"] = C.utc_now()
    queue["schema_version"] = C.SCHEMA_VERSION

    # reading list only — labelers write a SEPARATE long CSV (one row per resource determination),
    # so no fill-in-place gold_* columns here (that grain can't express multi-determination windows).
    order = ["project_id", "document_id", "manifest_role", "section_id", "evidence_span_id",
             "source_substrate", "agency_scope_status", "page_start", "page_end", "heading_title",
             "candidate_class_guess", "determination_polarity_guess", "matched_cue_group",
             "resource_area_guess", "resource_subarea_guess", "threshold_types_guess",
             "has_qual_cond_same_section", "has_qual_cond_windowed",
             "evidence_text", "evidence_text_sha256",
             "gold_queue_run_at", "schema_version"]
    queue = queue[order]

    C.write_parquet(queue, C.D2_GOLD_DIR / "significance_gold_queue.parquet", "gold queue")
    C.write_csv(queue, C.GOLD_QUEUE_CSV, "gold queue (analyst labels this)")

    print(f"\nqueued {len(queue):,} candidates ({len(pos_sample)} positives + {len(neg_sample)} negatives)")
    print("\npositives by candidate_class_guess:")
    print(pos_sample["candidate_class_guess"].value_counts().to_string())
    print("\nby agency_scope_status:")
    print(queue["agency_scope_status"].value_counts().to_string())


if __name__ == "__main__":
    main()
