"""Step 1 — partitioned recall diagnostic on the Codex-labeled validation samples (read-only).

For each labeled gold project, decide WHY the pick was right/wrong:
  correct   : selected date was right (gold_is_correct == yes)
  bucket1   : no real milestone date exists for THIS project -> correct answer is `missing`
              (we fabricated; e.g. cited other-project ROD, NOI, comment deadline; no ROD doc)
  bucket2   : the true date IS in the candidate pool -> we mis-picked (classifier/ranker/LLM fix)
  bucket3   : the true date is NOT in the candidate pool -> extraction/retrieval gap (03/02)
  unknown   : marked wrong but no gold date supplied AND project has the relevant doc type
              (can't tell bucket1 vs 2/3 without the true date)
  unsure    : labeler marked unsure
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
AUD = ROOT / "phase2" / "training" / "deliverable04" / "eis_validation"

cand = pd.read_parquet(TL / "timeline_candidates.parquet", columns=["project_id", "process_type", "parsed_date"])
idx = pd.read_parquet(TL / "timeline_document_index.parquet", columns=["project_id", "process_type", "document_type_clean"])
eis_cand = cand[cand.process_type == "EIS"]
pool_dates = eis_cand.groupby("project_id")["parsed_date"].apply(lambda s: set(s.dropna().astype(str).str[:10])).to_dict()
eis_idx = idx[idx.process_type == "EIS"]
has_rod_doc = set(eis_idx.loc[eis_idx.document_type_clean.astype(str).str.upper() == "ROD", "project_id"])
has_feis_doc = set(eis_idx.loc[eis_idx.document_type_clean.astype(str).str.upper() == "FEIS", "project_id"])


def in_pool(pid: str, gold: str) -> bool | None:
    """Is the gold date present among extracted candidates for the project?"""
    if not gold or gold == "nan":
        return None
    g = str(gold).strip()[:10]
    pool = pool_dates.get(pid, set())
    if len(g) == 10:
        return g in pool
    if len(g) == 7:  # month granularity
        return any(p[:7] == g for p in pool)
    if len(g) == 4:
        return any(p[:4] == g for p in pool)
    return None


def bucketize(rows, gold_col, correct_col, has_doc_set, label):
    out = []
    for _, r in rows.iterrows():
        pid = r.project_id
        corr = str(r[correct_col]).strip().lower()
        gold = str(r.get(gold_col) or "").strip()
        gold = "" if gold in ("nan", "none", "") else gold
        if corr == "yes":
            b = "correct"
        elif corr == "unsure":
            b = "unsure"
        else:  # no
            if gold:
                pres = in_pool(pid, gold)
                b = "bucket2_mispick_in_pool" if pres else "bucket3_extraction_gap"
            else:
                b = "bucket1_no_real_date" if pid not in has_doc_set else "unknown_has_doc_no_gold"
        out.append({"project_id": pid, "stratum": r.get("stratum", label),
                    "selected": r.get("decision_date" if label == "ROD" else "final_eis_date"),
                    "gold": gold, "correct": corr, "has_doc": pid in has_doc_set, "bucket": b})
    return pd.DataFrame(out)


print("=" * 70)
print("ROD (eis_rod_promotion_sample_labeled.csv)")
print("=" * 70)
rod = pd.read_csv(AUD / "eis_rod_promotion_sample_labeled.csv")
rb = bucketize(rod, "gold_rod_date", "gold_is_correct_rod", has_rod_doc, "ROD")
print("buckets:", rb.bucket.value_counts().to_dict())
print("\nbucket x stratum:")
print(pd.crosstab(rb.stratum, rb.bucket).to_string())
print("\nproject has a ROD document?  yes:", int(rb.has_doc.sum()), " no:", int((~rb.has_doc).sum()))

print("\n" + "=" * 70)
print("FEIS (eis_feis_sample_labeled.csv)")
print("=" * 70)
feis = pd.read_csv(AUD / "eis_feis_sample_labeled.csv")
feis["stratum"] = feis.final_eis_is_proxy.map({True: "proxy", False: "explicit"})
fb = bucketize(feis, "gold_feis_date", "gold_is_correct_feis", has_feis_doc, "FEIS")
print("buckets:", fb.bucket.value_counts().to_dict())
print("\nbucket x stratum:")
print(pd.crosstab(fb.stratum, fb.bucket).to_string())

rb.to_csv(AUD / "step1_rod_buckets.csv", index=False)
fb.to_csv(AUD / "step1_feis_buckets.csv", index=False)
print("\nWrote step1_rod_buckets.csv / step1_feis_buckets.csv")
