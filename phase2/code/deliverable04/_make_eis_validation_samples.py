"""Generate the two Phase C acceptance-validation samples (read-only).

Outputs to phase2/output/deliverable04/eis_audit/:
  - eis_rod_promotion_sample.csv : EIS projects whose ROD/decision selection CHANGED under
    Phase C (newly selected or different date), stratified by selection tier + document type,
    plus any old confirmed-ROD selections that changed (possible regressions). Used to measure
    promotion PRECISION (gate: >= 90%) and "no loss of existing valid RODs".
  - eis_feis_sample.csv : EIS projects with a final_eis_date, stratified explicit vs proxy.
    Used to measure FEIS-date precision (gate: >= 90% at labeled granularity).
  - *_full.csv reference dumps (unsampled).

Each row carries the full candidate context window so a labeler can judge correctness from the
CSV. Blank gold_* columns are for the labeler to fill.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
OUT = ROOT / "phase2" / "output" / "deliverable04" / "eis_audit"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42

new = pd.read_parquet(TL / "timeline_project_dates.parquet")
old = pd.read_parquet(sorted(glob.glob(str(TL / "_backup_phaseBC/timeline_project_dates.*.parquet")))[-1])
cand = pd.read_parquet(TL / "timeline_candidates.parquet")
idx = pd.read_parquet(TL / "timeline_document_index.parquet")

eis_new = new[new.process_type == "EIS"].copy()
eis_old = old[old.process_type == "EIS"].copy()
doc_type = (idx[idx.process_type == "EIS"][["project_id", "document_id", "document_type_clean"]]
            .drop_duplicates(["project_id", "document_id"]))

# Map project_id -> old decision_date for the "changed vs old" comparison.
old_dec = eis_old.set_index("project_id")["decision_date"].to_dict()

# Full context for the SELECTED decision candidate. NOTE: do NOT use selected_for_decision —
# that flag is stale/multi-valued in the candidates parquet (it accumulates across 05 runs and is
# never reset). Match the EXACT selected date by document + page + parsed_date instead.
cand_eis = cand[cand.process_type == "EIS"].copy()
cand_eis["_date"] = cand_eis.parsed_date.astype(str).str[:10]
cand_eis["_page"] = cand_eis.page_number.astype(str)
sel_dec = (cand_eis[["project_id", "document_id", "_page", "_date", "candidate_id",
                     "candidate_role", "raw_date_text", "context_text", "model_context"]]
           .drop_duplicates(["project_id", "document_id", "_page", "_date"]))

# ---------- ROD promotion sample ----------
rod = eis_new[eis_new.decision_date.notna()].copy()
rod["old_decision_date"] = rod.project_id.map(lambda p: old_dec.get(p))
rod["rod_tier"] = rod.timeline_flags.str.extract(r"(eis_rod_\w+)")[0]
rod = rod.merge(doc_type, left_on=["project_id", "decision_document_id"],
                right_on=["project_id", "document_id"], how="left")
rod["_page"] = rod.decision_page_number.astype(str)
rod["_date"] = rod.decision_date.astype(str).str[:10]
rod = rod.merge(sel_dec[["project_id", "document_id", "_page", "_date",
                         "context_text", "model_context", "raw_date_text"]],
                left_on=["project_id", "decision_document_id", "_page", "_date"],
                right_on=["project_id", "document_id", "_page", "_date"],
                how="left", suffixes=("", "_cand"))

# "changed vs old": newly selected, or a different date than before.
rod["changed_vs_old"] = rod.old_decision_date.isna() | (
    rod.decision_date.astype(str) != rod.old_decision_date.astype(str))

# stratum for sampling: tier + doc-type-risk
def stratum(r):
    dt = str(r.document_type_clean or "").upper()
    if not r.changed_vs_old:
        return "unchanged_control"
    if dt == "FEIS":
        return "rod_lang_in_FEIS_doc"     # clear_decision promoted via ROD language inside a FEIS doc
    if dt not in ("ROD",):
        return "rod_lang_in_other_doc"
    return r.rod_tier or "rod_other"
rod["stratum"] = rod.apply(stratum, axis=1)

rod_cols = ["project_id", "stratum", "rod_tier", "document_type_clean",
            "decision_date", "decision_date_granularity", "decision_source_type",
            "decision_is_proxy", "old_decision_date", "decision_document_id",
            "decision_page_number", "raw_date_text", "context_text", "model_context"]
rod_out = rod[rod_cols].copy()
rod_out.to_csv(OUT / "eis_rod_promotions_full.csv", index=False)

# Stratified deterministic sample: prioritise the risky/changed strata.
TARGET = {"rod_lang_in_FEIS_doc": 10, "rod_lang_in_other_doc": 10, "eis_rod_body": 10,
          "eis_rod_proxy": 6, "eis_rod_clear": 6, "unchanged_control": 4}
parts = []
for s, n in TARGET.items():
    sub = rod_out[rod_out.stratum == s]
    if not sub.empty:
        parts.append(sub.sample(n=min(n, len(sub)), random_state=SEED))
rod_sample = pd.concat(parts, ignore_index=True) if parts else rod_out.head(0)
for c in ["gold_is_correct_rod", "gold_rod_date", "gold_rod_granularity", "gold_notes"]:
    rod_sample[c] = ""
rod_sample.to_csv(OUT / "eis_rod_promotion_sample.csv", index=False)

# ---------- FEIS sample ----------
feis = eis_new[eis_new.final_eis_date.notna()].copy()
feis = feis.merge(doc_type, left_on=["project_id", "final_eis_document_id"],
                  right_on=["project_id", "document_id"], how="left",
                  suffixes=("", "_feisdoc"))
feis["has_rod"] = feis.decision_date.notna()
feis_cols = ["project_id", "final_eis_date", "final_eis_date_granularity", "final_eis_is_proxy",
             "final_eis_source_type", "final_eis_confidence", "has_rod",
             "final_eis_document_id", "final_eis_page_number", "final_eis_evidence_text"]
feis_out = feis[feis_cols].copy()
feis_out.to_csv(OUT / "eis_feis_full.csv", index=False)

explicit = feis_out[~feis_out.final_eis_is_proxy].sample(
    n=min(12, (~feis_out.final_eis_is_proxy).sum()), random_state=SEED)
proxy = feis_out[feis_out.final_eis_is_proxy].sample(
    n=min(13, feis_out.final_eis_is_proxy.sum()), random_state=SEED)
feis_sample = pd.concat([explicit, proxy], ignore_index=True)
for c in ["gold_is_correct_feis", "gold_feis_date", "gold_feis_granularity", "gold_notes"]:
    feis_sample[c] = ""
feis_sample.to_csv(OUT / "eis_feis_sample.csv", index=False)

print(f"ROD promotions (full): {len(rod_out)} | sample: {len(rod_sample)}")
print("  ROD sample strata:", rod_sample.stratum.value_counts().to_dict())
print(f"FEIS (full): {len(feis_out)} | sample: {len(feis_sample)} "
      f"(explicit {len(explicit)}, proxy {len(proxy)})")
print(f"Wrote to {OUT}")
