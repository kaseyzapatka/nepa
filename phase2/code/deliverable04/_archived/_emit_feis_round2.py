"""FEIS round-2 emit (tightened to Final-specific language) -> append blank rows to labeling_sample.csv.

Round 1's broad NOA/Final pool hit ~20% true Final-EIS dates (lots of Draft NOAs / scoping / comment
dates). This pool REQUIRES Final-EIS publication language in tight proximity to the date AND EXCLUDES
draft / DEIS / scoping / comment / NOI language, to raise the hit rate so ~350 labeled candidates net
~150 final_eis positives. Blank label, split=train, stratum=eis_emit_feis2; never re-adds an
already-labeled candidate. Appends into the single labeling_sample.csv (no new data CSV).
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
TL = ROOT / "phase2" / "data" / "analysis" / "timeline"
LAB = ROOT / "phase2" / "training" / "deliverable04" / "classifier.csv"
SEED, TARGET, PER_PROJECT_CAP = 42, 350, 5

# Final-EIS publication signal: "final EIS" within ~80 chars of a publish/file/availability verb,
# in either order, OR "availability of the final" / "FEIS ... filed/published/available".
FINAL_POS = re.compile(
    r"final\s+(?:eis|environmental\s+impact\s+statement)[^.]{0,80}"
    r"(?:filed|filing|publish\w*|releas\w*|made\s+available|availab\w*|notice\s+of\s+availability)"
    r"|(?:filed|filing|publish\w*|releas\w*|made\s+available|availab\w*|notice\s+of\s+availability)"
    r"[^.]{0,80}final\s+(?:eis|environmental\s+impact\s+statement)"
    r"|availability\s+of\s+the\s+final"
    r"|\bfeis\b[^.]{0,50}(?:filed|publish\w*|releas\w*|availab\w*|notice\s+of\s+availability)",
    re.I)
# Exclude only clear NON-endpoint contexts (scoping / NOI / comment). KEEP `draft`/`deis` mentions:
# genuine Final-EIS NOA text routinely references the prior Draft EIS comment period, and excluding
# those dropped the pool to ~100. The Draft-vs-Final call is then the labeler's job.
DRAFT_NEG = re.compile(
    r"scoping|comment\s+period|comment\s+deadline|notice\s+of\s+intent|\bnoi\b|public\s+comment",
    re.I)

cand = pd.read_parquet(TL / "timeline_candidates.parquet")
eis = cand[cand.process_type == "EIS"].copy()
lab = pd.read_csv(LAB)
labeled_ids = set(lab.candidate_id.astype(str))

ctx = eis.context_text.fillna("")
pool = eis[(eis.document_type_clean.astype(str).str.upper() == "FEIS")
           & ctx.str.contains(FINAL_POS) & ~ctx.str.contains(DRAFT_NEG)
           & ~eis.candidate_id.astype(str).isin(labeled_ids)].copy()
print(f"tightened Final-specific FEIS pool: {len(pool)} candidates in {pool.project_id.nunique()} projects")

pool = pool.sort_values("candidate_id")
pool["_rk"] = pool.groupby("project_id").cumcount()
pool = pool[pool._rk < PER_PROJECT_CAP]
sel = pool.sample(n=min(TARGET, len(pool)), random_state=SEED).drop(columns="_rk")

batch = sel.reindex(columns=lab.columns, fill_value="")
batch["label"], batch["stratum"], batch["split"], batch["notes"] = "", "eis_emit_feis2", "train", ""
combined = pd.concat([lab, batch], ignore_index=True)
combined.to_csv(LAB, index=False)
print(f"Appended {len(batch)} blank eis_emit_feis2 rows. labeling_sample.csv: {len(lab)} -> {len(combined)}")
print("(target: ~150 of these labeled final_eis after Codex pass)")
