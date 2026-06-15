"""
D4 EIS Recovery — Phase 0: baseline metrics, corrected source ceiling, validation fixtures.

Run BEFORE any behavior change (recover_eis.md §6 Phase 0). Reads the CURRENT production
parquets (the "before" snapshot) and writes:

  notes/deliverable04/phase0/
    baseline_metrics.csv          — one row per metric (candidate coverage, selected-date
                                     coverage, ROD/FEIS split, unknown-role reliance,
                                     extractable-text, source ceiling)
    source_ceiling.csv            — per-project endpoint/init-evidence/joint flags
    cohort_zero_candidate.txt      — fixtures for later phases
    cohort_init_only.txt
    cohort_dec_only.txt
    cohort_complete.txt
    cohort_ranker_blocked_init.txt (high-conf init blocked — Phase 4 labeling targets)

The GO/NO-GO number is the corrected joint source ceiling: how many of the 4,130 EIS
projects could have a complete timeline given the documents available. The ceiling
definition is validated by requiring >=95% of currently-complete projects to satisfy it.

Usage:  CONDA_DEFAULT_ENV=nepa python _phase0_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

HERE = Path(__file__).resolve().parent
PHASE2 = HERE.parent.parent
TLDIR = PHASE2 / "data" / "analysis" / "timeline"
PAGES = PHASE2 / "data" / "processed" / "eis" / "pages.parquet"
OUT = PHASE2 / "notes" / "deliverable04" / "phase0"
OUT.mkdir(parents=True, exist_ok=True)

PROJ = (TLDIR / "timeline_project_dates.parquet").as_posix()
CAND = (TLDIR / "timeline_candidates.parquet").as_posix()
IDX = (TLDIR / "timeline_document_index.parquet").as_posix()

INIT_ROLES = ("clear_initiation", "proxy_initiation")
DEC_ROLES = ("clear_decision", "proxy_decision", "final_eis")

con = duckdb.connect()
metrics: list[dict] = []


def add(metric: str, value, note: str = "") -> None:
    metrics.append({"metric": metric, "value": value, "note": note})
    print(f"  {metric:42s} {value:>8} {note}")


def dump_ids(name: str, df: pd.DataFrame) -> None:
    df["project_id"].to_csv(OUT / name, index=False, header=False)
    print(f"  wrote {name}: {len(df)}")


print("== Phase 0 baseline (current production snapshot) ==")

# ---------------------------------------------------------------------------
# 1. Extractable-text per EIS document (one-time stream over the pages file).
#    has_extractable_text(doc) = max trimmed page_text length > 100.
# ---------------------------------------------------------------------------
print("\n[1] Extractable-text per document (streaming pages.parquet)...")
doc_text = con.execute(
    f"""
    SELECT document_id, MAX(LENGTH(TRIM(page_text))) AS max_len
    FROM '{PAGES.as_posix()}'
    GROUP BY document_id
    """
).df()
con.register("doc_text", doc_text)

# ---------------------------------------------------------------------------
# 2. Per-project document-evidence flags from the index.
# ---------------------------------------------------------------------------
print("[2] Per-project document evidence (index + extractable text)...")
proj_doc = con.execute(
    f"""
    WITH idx AS (
        SELECT i.*, COALESCE(t.max_len, 0) AS doc_max_len
        FROM '{IDX}' i
        LEFT JOIN doc_text t USING (document_id)
        WHERE i.process_type = 'EIS'
    )
    SELECT
        project_id,
        MAX(CASE WHEN decision_doc_score >= 3.5 THEN 1 ELSE 0 END)               AS has_endpoint_doc,
        MAX(CASE WHEN doc_max_len > 100 THEN 1 ELSE 0 END)                       AS has_text_doc,
        MAX(CASE WHEN initiation_doc_score > 0 THEN 1 ELSE 0 END)                AS has_init_doc,
        MAX(CASE WHEN noi_publication_date IS NOT NULL
                   OR blm_initiation_date IS NOT NULL
                   OR doe_initiation_date IS NOT NULL THEN 1 ELSE 0 END)         AS has_register_init,
        MAX(CASE WHEN blm_decision_date IS NOT NULL
                   OR doe_decision_date IS NOT NULL THEN 1 ELSE 0 END)           AS has_register_decision
    FROM idx
    GROUP BY project_id
    """
).df()
con.register("proj_doc", proj_doc)

# ---------------------------------------------------------------------------
# 3. Candidate-coverage decomposition.
# ---------------------------------------------------------------------------
print("[3] Candidate coverage decomposition...")
cov = con.execute(
    f"""
    WITH proj AS (SELECT DISTINCT project_id FROM '{PROJ}' WHERE process_type='EIS'),
    init_c AS (SELECT DISTINCT project_id FROM '{CAND}'
               WHERE process_type='EIS' AND candidate_role IN {INIT_ROLES}),
    dec_c  AS (SELECT DISTINCT project_id FROM '{CAND}'
               WHERE process_type='EIS' AND candidate_role IN {DEC_ROLES}),
    any_c  AS (SELECT DISTINCT project_id FROM '{CAND}' WHERE process_type='EIS')
    SELECT
        p.project_id,
        CASE WHEN i.project_id IS NOT NULL THEN 1 ELSE 0 END AS has_init_cand,
        CASE WHEN d.project_id IS NOT NULL THEN 1 ELSE 0 END AS has_dec_cand,
        CASE WHEN a.project_id IS NOT NULL THEN 1 ELSE 0 END AS has_any_cand
    FROM proj p
    LEFT JOIN init_c i USING(project_id)
    LEFT JOIN dec_c  d USING(project_id)
    LEFT JOIN any_c  a USING(project_id)
    """
).df()
con.register("cov", cov)

total = len(cov)
both_types = int(((cov.has_init_cand == 1) & (cov.has_dec_cand == 1)).sum())
init_only = int(((cov.has_init_cand == 1) & (cov.has_dec_cand == 0)).sum())
dec_only = int(((cov.has_init_cand == 0) & (cov.has_dec_cand == 1)).sum())
other_only = int(((cov.has_init_cand == 0) & (cov.has_dec_cand == 0) & (cov.has_any_cand == 1)).sum())
zero_cand = int((cov.has_any_cand == 0).sum())

add("eis_total", total)
add("cand_both_types", both_types, f"{100*both_types/total:.1f}%")
add("cand_init_only", init_only, "have init cand, need decision cand")
add("cand_dec_only", dec_only, "have decision cand, need init cand")
add("cand_other_role_only", other_only, "only body_text/unknown etc.")
add("cand_zero", zero_cand, f"{100*zero_cand/total:.1f}%")

# ---------------------------------------------------------------------------
# 4. Selected-date coverage + ROD/FEIS split + unknown-role reliance.
# ---------------------------------------------------------------------------
print("[4] Selected-date coverage, ROD/FEIS split, unknown-role reliance...")
sel = con.execute(
    f"""
    SELECT
        COUNT(*) FILTER (WHERE initiation_date IS NOT NULL)                       AS has_init,
        COUNT(*) FILTER (WHERE decision_date IS NOT NULL)                         AS has_dec,
        COUNT(*) FILTER (WHERE initiation_date IS NOT NULL
                           AND decision_date IS NOT NULL)                          AS complete,
        COUNT(*) FILTER (WHERE decision_date IS NOT NULL AND has_rod)              AS dec_rod,
        COUNT(*) FILTER (WHERE decision_date IS NOT NULL
                           AND decision_is_feis_fallback)                          AS dec_feis,
        COUNT(*) FILTER (WHERE final_eis_date IS NOT NULL)                         AS final_eis_field
    FROM '{PROJ}' WHERE process_type='EIS'
    """
).df().iloc[0]
add("selected_has_init", int(sel.has_init), f"{100*sel.has_init/total:.1f}%")
add("selected_has_decision", int(sel.has_dec), f"{100*sel.has_dec/total:.1f}%")
add("selected_complete", int(sel.complete), f"{100*sel.complete/total:.1f}%  <-- headline")
add("decision_rod", int(sel.dec_rod), "ROD-sourced decisions")
add("decision_feis_fallback", int(sel.dec_feis), "FEIS-fallback decisions in decision_date")
add("final_eis_field_populated", int(sel.final_eis_field), "separate field (expect 0)")

unk = con.execute(
    f"""
    SELECT
        COUNT(DISTINCT project_id) FILTER (WHERE selected_for_decision AND candidate_role='unknown')   AS dec_unknown,
        COUNT(DISTINCT project_id) FILTER (WHERE selected_for_initiation AND candidate_role='unknown') AS init_unknown
    FROM '{CAND}' WHERE process_type='EIS'
    """
).df().iloc[0]
add("decision_winner_unknown_role", int(unk.dec_unknown), "Phase 4 regression-guard denominator")
add("initiation_winner_unknown_role", int(unk.init_unknown))

# ---------------------------------------------------------------------------
# 5. Extractable-text coverage (Phase 2 gate denominator).
# ---------------------------------------------------------------------------
print("[5] Extractable-text coverage...")
text_cov = int(proj_doc.has_text_doc.sum())
add("has_extractable_text", text_cov, f"{100*text_cov/total:.1f}%  Phase-2 gate denominator")

# ---------------------------------------------------------------------------
# 6. CORRECTED joint source ceiling (the GO/NO-GO).
#    has_init_evidence = extractable-text doc (narrative can yield NOI/scoping date)
#                        OR register init OR initiation_doc_score>0
#    has_decision_evidence = endpoint doc OR register decision
#    joint = both
# ---------------------------------------------------------------------------
print("[6] Corrected joint source ceiling...")
pd_ = proj_doc.copy()
pd_["has_init_evidence"] = (
    (pd_.has_text_doc == 1) | (pd_.has_register_init == 1) | (pd_.has_init_doc == 1)
).astype(int)
pd_["has_decision_evidence"] = (
    (pd_.has_endpoint_doc == 1) | (pd_.has_register_decision == 1)
).astype(int)
pd_["joint"] = ((pd_.has_init_evidence == 1) & (pd_.has_decision_evidence == 1)).astype(int)
pd_.to_csv(OUT / "source_ceiling.csv", index=False)

joint = int(pd_.joint.sum())
add("ceiling_has_decision_evidence", int(pd_.has_decision_evidence.sum()))
add("ceiling_has_init_evidence", int(pd_.has_init_evidence.sum()))
add("source_ceiling_joint", joint, f"{100*joint/total:.1f}%  <-- GO/NO-GO ceiling")

# Validate the ceiling definition against currently-complete projects.
complete_ids = con.execute(
    f"SELECT project_id FROM '{PROJ}' WHERE process_type='EIS' "
    f"AND initiation_date IS NOT NULL AND decision_date IS NOT NULL"
).df()
val = complete_ids.merge(pd_[["project_id", "joint"]], on="project_id", how="left")
pass_rate = val.joint.fillna(0).mean()
add("ceiling_validation_pass_rate", f"{100*pass_rate:.1f}%",
    "share of currently-complete satisfying ceiling (target >=95%)")
ceiling_valid = pass_rate >= 0.95

# ---------------------------------------------------------------------------
# 7. Phase-4 labeling target: high-confidence initiation blocked by ranker.
#    (have a valid init candidate with p_init_cal>0.5 but no selected initiation date)
# ---------------------------------------------------------------------------
print("[7] Phase-4 labeling cohort (high-conf init, no selected init)...")
ranker_blocked = con.execute(
    f"""
    WITH no_init AS (
        SELECT project_id FROM '{PROJ}'
        WHERE process_type='EIS' AND initiation_date IS NULL
    ),
    strong_init_cand AS (
        SELECT DISTINCT project_id FROM '{CAND}'
        WHERE process_type='EIS'
          AND candidate_role IN {INIT_ROLES}
          AND TRY_CAST(p_init_cal AS DOUBLE) > 0.5
    )
    SELECT n.project_id FROM no_init n JOIN strong_init_cand s USING(project_id)
    """
).df()
add("phase4_label_targets_init", len(ranker_blocked),
    "high-conf init, no selected init (label ~30)")

# ---------------------------------------------------------------------------
# 8. Write fixtures + metrics.
# ---------------------------------------------------------------------------
print("\n[8] Writing fixtures...")
dump_ids("cohort_complete.txt", complete_ids)
dump_ids("cohort_zero_candidate.txt", cov[cov.has_any_cand == 0])
dump_ids("cohort_init_only.txt", cov[(cov.has_init_cand == 1) & (cov.has_dec_cand == 0)])
dump_ids("cohort_dec_only.txt", cov[(cov.has_init_cand == 0) & (cov.has_dec_cand == 1)])
dump_ids("cohort_ranker_blocked_init.txt", ranker_blocked)

pd.DataFrame(metrics).to_csv(OUT / "baseline_metrics.csv", index=False)
print(f"\n  wrote baseline_metrics.csv ({len(metrics)} metrics)")

# ---------------------------------------------------------------------------
# 9. Verdict.
# ---------------------------------------------------------------------------
print("\n== GO/NO-GO ==")
print(f"  Corrected joint source ceiling: {joint} / {total} ({100*joint/total:.1f}%)")
print(f"  Ceiling definition valid (>=95% of complete pass): "
      f"{'YES' if ceiling_valid else 'NO — REVISIT DEFINITION'} ({100*pass_rate:.1f}%)")
if not ceiling_valid:
    print("  !! The ceiling definition excludes currently-complete projects; do not trust it.")
elif joint < 2891:
    print(f"  70% target (2,891) EXCEEDS local-document ceiling ({joint}). "
          f"70% complete requires Phase 5 (OCR/external register).")
    print(f"  Local-document-achievable complete-timeline ceiling = {joint} "
          f"({100*joint/total:.1f}%).")
else:
    print(f"  70% target (2,891) is within the local-document ceiling ({joint}). Reachable in principle.")
