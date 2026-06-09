"""Fold the Codex-labeled EIS worksheet into ranker.csv (project-level gold), then reserve part of
the new verified picks as frozen eval. No new dataset: updates ranker.csv + frozen_eval_ids.txt
in place (both backed up first). The worksheet is a transient vehicle and can be deleted after.

Reads:  training/deliverable04/_ranker_eis_labeling_batch.csv  (gold_pick / gold_type / gold_notes filled)
Writes: training/deliverable04/ranker.csv                      (decision_candidate_id, notes, split)
        training/deliverable04/frozen_eval_ids.txt             (grows toward ~50 protected ids)

Run AFTER Codex fills the worksheet:
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_fold_ranker_eis_batch.py
"""
from __future__ import annotations
import importlib.util
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("sel05", HERE / "05_select_dates.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
TRAINING = m.PHASE2 / "training" / "deliverable04"
WS = TRAINING / "_ranker_eis_labeling_batch.csv"
RANKER = TRAINING / "ranker.csv"
FROZEN = TRAINING / "frozen_eval_ids.txt"
BACKUPS = TRAINING / "_backups"
SEED, FROZEN_EVAL_TARGET = 42, 28   # ~28 eval leaves ~50 EIS positives for training (of ~78 total)
VALID_TYPES = {"rod", "feis", "none"}
NOTE = {"rod": "verified_rod", "feis": "verified_feis_fallback", "none": "verified_none"}

# pool candidate_ids for validation
pool_ids = set(pd.read_parquet(m.CANDIDATES_PATH, columns=["candidate_id"]).candidate_id.astype(str))

ws = pd.read_csv(WS, dtype=str, keep_default_na=False)
ws["_pick"] = ws.gold_pick.str.strip().str.lower().isin({"yes", "y", "true", "1", "x"})
ws["_type"] = ws.gold_type.str.strip().str.lower()

# ---- build one pick per project, with validation ----
picks = {}   # project_id -> (decision_candidate_id, type, note)
warn = []
for pid, grp in ws.groupby("project_id"):
    chosen = grp[grp._pick]
    none_rows = grp[grp._type.eq("none")]
    if len(chosen) >= 1:
        if len(chosen) > 1:
            warn.append(f"{pid}: {len(chosen)} gold_pick=yes rows -> taking the first")
        row = chosen.iloc[0]
        t = row._type if row._type in {"rod", "feis"} else "rod"
        cid = str(row.candidate_id)
        if cid not in pool_ids:
            warn.append(f"{pid}: picked candidate_id {cid} not in pool -> skipped"); continue
        picks[pid] = (cid, t, NOTE[t])
    elif len(none_rows):
        picks[pid] = ("none", "none", NOTE["none"])
    else:
        warn.append(f"{pid}: no gold_pick and no 'none' -> UNLABELED, skipped")

# ---- incorporate the focused FEIS-fallback re-label (overrides the ROD-first-suppressed 'none') ----
FOCUSED = WS.parent / "_ranker_feis_fallback_batch.csv"
if FOCUSED.exists():
    fw = pd.read_csv(FOCUSED, dtype=str, keep_default_na=False)
    fw["_pick"] = fw.gold_pick.str.strip().str.lower().isin({"yes", "y", "true", "1", "x"})
    n_over = 0
    for pid, grp in fw.groupby("project_id"):
        ch = grp[grp._pick]
        if len(ch):
            cid = str(ch.iloc[0].candidate_id)
            if cid in pool_ids:
                picks[pid] = (cid, "feis", NOTE["feis"]); n_over += 1
            else:
                warn.append(f"{pid}: focused FEIS pick {cid} not in pool -> skipped")
    print(f"focused FEIS-fallback: overrode {n_over} projects none -> feis")

print(f"worksheet projects: {ws.project_id.nunique()} | usable picks: {len(picks)}")
by_t = pd.Series([t for _, t, _ in picks.values()]).value_counts().to_dict()
print(f"  by type: {by_t}")
if warn:
    print("  WARNINGS:"); [print("   -", w) for w in warn[:20]]

# ---- fold into ranker.csv (update existing, append new) ----
BACKUPS.mkdir(parents=True, exist_ok=True)
g = pd.read_csv(RANKER, dtype=str, keep_default_na=False)
ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
(BACKUPS / f"ranker.pre_eisbatch_{ts}.csv").write_text(g.to_csv(index=False))
g = g.set_index("project_id")
updated = appended = 0
for pid, (cid, t, note) in picks.items():
    if pid in g.index:
        g.loc[pid, "decision_candidate_id"] = cid
        prev = str(g.loc[pid, "notes"]).strip()
        g.loc[pid, "notes"] = f"{prev};{note}".strip(";") if prev else note
        updated += 1
    else:
        g.loc[pid] = {c: "" for c in g.columns}
        g.loc[pid, ["process_type", "decision_candidate_id", "notes", "split"]] = ["EIS", cid, note, ""]
        appended += 1
g = g.reset_index()

# ---- reserve ~FROZEN_EVAL_TARGET protected ids (grow the existing registry; never un-reserve) ----
existing = {ln.strip() for ln in FROZEN.read_text().splitlines() if ln.strip()} if FROZEN.exists() else set()
# eligible to reserve: newly-verified projects with a REAL candidate (rod/feis, not none), not already frozen
rankable = [pid for pid, (cid, t, _) in picks.items() if t in {"rod", "feis"} and pid not in existing]
need = max(0, FROZEN_EVAL_TARGET - len(existing))
kind = {pid: t for pid, (cid, t, _) in picks.items()}
rdf = pd.DataFrame({"pid": rankable, "t": [kind[p] for p in rankable]})
reserve = []
if len(rdf) and need:
    for t, gg in rdf.groupby("t"):
        n = min(len(gg), round(need * len(gg) / len(rdf)))
        reserve += gg.sample(n=n, random_state=SEED).pid.tolist()
    reserve = reserve[:need]
frozen = sorted(existing | set(reserve))
(BACKUPS / f"frozen_eval_ids.pre_eisbatch_{ts}.txt").write_text("\n".join(sorted(existing)) + "\n")
FROZEN.write_text("\n".join(frozen) + "\n")

# ---- set frozen split: frozen -> test; blank (new) -> train; persist ----
g.loc[g.project_id.isin(frozen), "split"] = "test"
g.loc[g.split.str.strip().eq(""), "split"] = "train"
g.to_csv(RANKER, index=False)

# ---- summary + guardrail sanity ----
eisg = g[g.process_type == "EIS"]
ver = eisg[eisg.notes.str.contains("verified", na=False)]
print(f"\nranker.csv: updated {updated}, appended {appended} -> {len(g)} projects")
print(f"  EIS verified decision picks now: {len(ver)} (was 38)")
print(f"  split: {g.split.value_counts().to_dict()}")
print(f"frozen_eval_ids.txt: {len(existing)} -> {len(frozen)} (added {len(reserve)})")
leak = set(g[g.split == 'train'].project_id) & set(frozen)
print(f"GUARDRAIL sanity — train ids in frozen registry (must be 0): {len(leak)}")
print("\nNext: 05b --train  (will print 'guardrail OK'), then 05b --apply, then _gold_rank_check.py")
print("Then the worksheet can be deleted:", WS.name)
