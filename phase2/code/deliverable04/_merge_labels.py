#!/usr/bin/env python
"""Append-safe merge of agent labels into classifier.csv / ranker.csv (D4 night-2).

Reads the .labeled.csv files (candidate_id,label,evidence_span) produced from the agent results,
JOINS them back to the worksheets to recover full classifier columns, validates, and APPENDS to the
PRELABEL baselines. Guarantees: every original row is byte-identical, only brand-new candidate_ids
are added, split is blank (-> train), labels are valid. Aborts on any violation.

Run AFTER writing the *.labeled.csv files.  conda run -n nepa python ... _merge_labels.py
"""
import sys
import pandas as pd
from pathlib import Path

LAB = Path("/Users/Dora/git/consulting/nepa-night/phase2/labeling")
TRN = Path("/Users/Dora/git/consulting/nepa-night/phase2/training/deliverable04")
CLS = TRN / "classifier.csv"
RANK = TRN / "ranker.csv"
CLS_PRE = TRN / "classifier.PRELABEL.csv"
RANK_PRE = TRN / "ranker.PRELABEL.csv"
FROZEN = {ln.strip() for ln in (TRN / "frozen_eval_ids.txt").read_text().splitlines() if ln.strip()}
VALID = {"initiation", "decision", "neither", "final_eis"}
CAND_WS = ["init", "neither", "final_eis", "active"]

def die(msg):
    print(f"ABORT: {msg}"); sys.exit(1)

# ---- classifier ----
pre = pd.read_csv(CLS_PRE, dtype=str, keep_default_na=False)
pre_ids = set(pre["candidate_id"])
new_rows = []
for ws in CAND_WS:
    wpath, lpath = LAB / f"worksheet_{ws}.csv", LAB / f"worksheet_{ws}.labeled.csv"
    if not lpath.exists():
        print(f"  (skip {ws}: no labeled file)"); continue
    sheet = pd.read_csv(wpath, dtype=str, keep_default_na=False)
    sheet = sheet.drop(columns=["label", "evidence_span"], errors="ignore")  # drop empty placeholders
    lab = pd.read_csv(lpath, dtype=str, keep_default_na=False)
    lab["label"] = lab["label"].str.strip().str.lower()
    bad = set(lab["label"]) - VALID
    if bad: die(f"{ws}: invalid label values {bad}")
    m = sheet.merge(lab[["candidate_id", "label", "evidence_span"]], on="candidate_id", how="inner")
    if len(m) != len(lab):
        die(f"{ws}: {len(lab)} labels but {len(m)} matched worksheet candidate_ids")
    m = m[~m["candidate_id"].isin(pre_ids)]          # never overwrite existing
    m["notes"] = m["evidence_span"]; m["split"] = ""  # blank -> train
    new_rows.append(m)
    print(f"  {ws}: +{len(m)} new ({lab['label'].value_counts().to_dict()})")

if new_rows:
    add = pd.concat(new_rows, ignore_index=True)
    add = add.drop_duplicates("candidate_id")
    add = add[~add["candidate_id"].isin(pre_ids)]
    out_cols = list(pre.columns)
    for c in out_cols:
        if c not in add.columns:
            add[c] = ""
    add = add[out_cols]
    merged = pd.concat([pre, add], ignore_index=True)
    # APPEND-SAFETY: prefix must equal PRELABEL byte-for-byte
    if not merged.iloc[:len(pre)].reset_index(drop=True).equals(pre.reset_index(drop=True)):
        die("classifier prefix changed — original rows not preserved")
    if len(merged) != len(pre) + len(add):
        die("classifier row-count mismatch")
    merged.to_csv(CLS, index=False)
    print(f"classifier.csv: {len(pre)} -> {len(merged)} (+{len(add)}); prefix verified identical")
else:
    print("classifier: no new labels to merge")

# ---- ranker ----
rpath = LAB / "worksheet_ranker_answers.labeled.csv"
if rpath.exists():
    rpre = pd.read_csv(RANK_PRE, dtype=str, keep_default_na=False)
    rpre_ids = set(rpre["project_id"])
    ans = pd.read_csv(rpath, dtype=str, keep_default_na=False)
    wr = pd.read_csv(LAB / "worksheet_ranker.csv", dtype=str, keep_default_na=False)
    valid_by_proj = wr.groupby("project_id")["candidate_id"].apply(set).to_dict()
    rows = []
    for _, r in ans.iterrows():
        pid = r["project_id"]
        if pid in rpre_ids or pid in FROZEN:
            continue  # never relabel existing or frozen-eval projects
        init_id = r.get("true_initiation_candidate_id", "").strip()
        dec_id = r.get("true_decision_candidate_id", "").strip()
        init_id = "" if init_id.lower() in ("none", "nan", "") else init_id
        dec_id = "" if dec_id.lower() in ("none", "nan", "") else dec_id
        valid = valid_by_proj.get(pid, set())
        if init_id and init_id not in valid:
            print(f"  ranker {pid}: init id not in project candidates — dropping init"); init_id = ""
        if dec_id and dec_id not in valid:
            print(f"  ranker {pid}: dec id not in project candidates — dropping dec"); dec_id = ""
        if not init_id and not dec_id:
            continue  # no usable signal
        proc = wr[wr["project_id"] == pid]["process_type"].iloc[0] if (wr["project_id"] == pid).any() else ""
        rows.append({"project_id": pid, "process_type": proc,
                     "initiation_candidate_id": init_id, "decision_candidate_id": dec_id,
                     "notes": r.get("notes", ""), "split": ""})
    if rows:
        radd = pd.DataFrame(rows)
        for c in rpre.columns:
            if c not in radd.columns:
                radd[c] = ""
        radd = radd[list(rpre.columns)]
        rmerged = pd.concat([rpre, radd], ignore_index=True)
        if not rmerged.iloc[:len(rpre)].reset_index(drop=True).equals(rpre.reset_index(drop=True)):
            die("ranker prefix changed — original rows not preserved")
        rmerged.to_csv(RANK, index=False)
        print(f"ranker.csv: {len(rpre)} -> {len(rmerged)} (+{len(radd)}); prefix verified identical")
    else:
        print("ranker: no usable new picks to merge")
else:
    print("ranker: no labeled answers file")
print("MERGE COMPLETE.")
