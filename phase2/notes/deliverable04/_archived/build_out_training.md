# D4 — Build out the training set to ~1,000 labels per head (Codex run)

**Goal.** Grow `labeling_sample.csv` from today's **160 initiation / 165 decision** positives toward
**~1,000 per head**, so the SetFit classifier (and later DeBERTa) has enough signal — especially on
the **initiation head**, which is the weak one (its pool probabilities are near-zero, so it currently
contributes almost nothing to date selection in `05`).

**Why this is the bottleneck.** The decision head is at F1 ≈ 0.74 and roughly plateaued; the
initiation head is at F1 ≈ 0.65 and *starved*. The single highest-value thing we can do is harvest
many more **true initiation** examples (NOI/scoping/application-received dates), plus enough decision
and neither examples to keep the heads balanced.

This run follows the **existing active-learning workflow**: emit blank candidate rows → label them
→ the labels live **directly in `labeling_sample.csv`** (one file, the single source of truth). It does
**not** create a parallel dataset.

---

## Hard rules (read first — these protect the experiment)

1. **Never touch `split == "test"` rows.** The 154-row frozen test set is sacred. Every new row you
   add is `split = "train"`. Do not edit, relabel, or move any existing row.
2. **Only fill blank labels.** Guard on blank: if a `candidate_id` already has a non-empty `label`,
   skip it. Never overwrite an existing label.
3. **Dedup against everything already in the file** by `candidate_id` before appending.
4. **Label strictly by `labeling_rules.md`** — the label reflects what the `[[marked date]]` *is*,
   not what the regex `candidate_role` guessed (the role is often wrong; that's the point).
5. **Every label gets a `notes` value**: the rule that applies + a short direct quote (≤ 20 words)
   from the `model_context` that clinched it. Same format as the round-2 build.
6. **The work is resumable.** Apply labels in chunks; the guard-on-blank makes re-runs idempotent.

Files:
- Labels (sole source, frozen split): `phase2/output/deliverable04/labeling_sample.csv`
- Codebook (the labeling rules): `phase2/notes/deliverable04/labeling_rules.md`
- Scored candidate pool (read-only here): `phase2/data/analysis/timeline/timeline_candidates.parquet`

Environment: every command runs with the `nepa` env active (`CONDA_DEFAULT_ENV=nepa`).

---

## Step 1 — Emit a large, positive-rich batch of blank rows

Uncertainty sampling (the normal `--emit-batch`) surfaces mostly `neither`, which is too slow for a
build-out. Instead, harvest from the regex roles with the **highest true-positive base rate**:
`clear_initiation` for the init head, `clear_decision` / `proxy_decision` for the decision head. We
oversample `clear_initiation` because that head is the one we most need to feed.

Run this once (it appends blank rows to `labeling_sample.csv`):

```python
# phase2/code/deliverable04/_emit_buildout.py  (create + run once)
import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CAND = ROOT / "phase2/data/analysis/timeline/timeline_candidates.parquet"
LAB  = ROOT / "phase2/output/deliverable04/labeling_sample.csv"

# Per-role quota for the build-out. Tuned to yield ~800-900 NEW positives per head after labeling
# (clear_* roles are positive-rich but still need human/Codex confirmation; proxy_* and unknown
# add decision/neither coverage and hard cases). Adjust down if one night isn't enough — it's resumable.
QUOTAS = {
    "clear_initiation": 1200,   # init harvest (the starved head)
    "clear_decision":    900,   # decision harvest
    "proxy_decision":    400,   # decision + neither (proxy_decision is ~97% neither — teaches the boundary)
    "proxy_initiation":  400,   # init + neither
    "unknown":           300,   # hard cases
    "body_text":         300,   # hard cases / neither
}
SEED = 42

df  = pd.read_parquet(CAND)
lab = pd.read_csv(LAB)
labeled = set(lab["candidate_id"])

elig = df[~df["candidate_id"].isin(labeled)].copy()
# Only score-eligible candidates (skip register-authoritative / out-of-scope roles already handled).
parts = []
for role, n in QUOTAS.items():
    pool = elig[elig["candidate_role"] == role]
    if len(pool):
        parts.append(pool.sample(min(n, len(pool)), random_state=SEED))
batch = pd.concat(parts).drop_duplicates("candidate_id")

out = batch.reindex(columns=lab.columns, fill_value="")
out["label"]   = ""
out["notes"]   = ""
out["split"]   = "train"            # NEVER test
out["stratum"] = "buildout_2026_06"
pd.concat([lab, out], ignore_index=True).to_csv(LAB, index=False)
print(f"Appended {len(out)} blank rows (now {len(lab)+len(out)} total). "
      f"Blank to label: {len(out)}.")
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_emit_buildout.py
```

After this, `labeling_sample.csv` has ~3,500 new rows with blank `label`/`notes`, `split=train`,
`stratum=buildout_2026_06`.

---

## Step 2 — Label the blank rows (the bulk of the work)

Work through the blank rows (`label == ""`) and assign each `label` ∈ {`initiation`, `decision`,
`neither`} per **`labeling_rules.md`**. Read the `[[marked date]]` in `model_context` and ignore the
`candidate_role` guess.

Quick reference (full rules in `labeling_rules.md`):
- **initiation** — NOI published; application/ROW/permit filed or received; scoping started; FERC
  pre-filing approved; DOE *Initiator* signature; "posted to the NEPA Register / ePlanning".
- **decision** — NEPA Compliance Officer signature / operative "Date Determined"; Field-Manager
  authorization signature; "It is my decision…"; Decision Record date; FONSI/ROD; CX cover month
  (the CX *is* the determination); ROW grant issued.
- **neither** — specialist/reviewer signatures; SHPO/USFWS/tribal consultation; comment-period ends;
  EA/EIS/DEIS/FEIS **document cover months** (decision is the separate FONSI/ROD); citations; case
  numbers parsed as dates; permit term/expiration; surveys/meetings/inspections; historical refs.
- **Tie-breakers:** CX cover month → `decision`; EA/EIS cover month → `neither`; label only the
  `[[marked]]` date; an activity "conducted on" a date is `neither`.

**Label in chunks of ~200** and apply each chunk (Step 3) before moving on, so progress persists.

**This is an iterative loop, not a one-pass job.** The target is **~1,000 `train` positives for EACH
head** (initiation and decision). One emission (Step 1) will not get you there — `clear_initiation`
candidates are not all true initiations, so the *yield* of positives is lower than the number
labeled. So the loop is:

> **Repeat { Step 1 emit → Step 2 label → Step 3 apply → Step 4 count } until BOTH heads reach
> ~1,000 `train` positives.** Step 1 dedups against everything already labeled, so re-running it
> simply pulls a fresh batch. After each cycle, Step 4 tells you how far each head still has to go;
> if `initiation` is still short (it will lag — it's the starved head), re-emit and keep labeling,
> prioritizing `clear_initiation` rows.

Do not stop at one batch. Keep cycling until the counts are there (or you run out of night —
it's fully resumable, so a partial result is a fine stopping point).

---

## Step 3 — Apply labels directly into `labeling_sample.csv`

For each chunk, write a small apply script in the **same pattern as `apply_labels_al2.py`**: a
`LABELS` list of `{candidate_id, label, notes}` dicts, merged into `labeling_sample.csv` on
`candidate_id`, **guarding on blank** (only fills rows whose `label` is currently empty), preserving
column order, writing back in place.

```python
# phase2/output/deliverable04/apply_buildout_<chunk>.py
import pandas as pd
LABELS = [
    {"candidate_id": "....", "label": "initiation",
     "notes": "Initiation: scoping started, quote 'opened a 30-day scoping period'."},
    # ... one dict per candidate in this chunk ...
]
path = "phase2/output/deliverable04/labeling_sample.csv"
df = pd.read_csv(path, dtype=str, keep_default_na=False)
lab = pd.DataFrame(LABELS)
merged = df.merge(lab, on="candidate_id", how="left", suffixes=("", "_new"))
blank = merged["label"].astype(str).str.strip().eq("")
has_new = merged["label_new"].notna()
apply = blank & has_new
merged.loc[apply, "label"] = merged.loc[apply, "label_new"]
merged.loc[apply, "notes"] = merged.loc[apply, "notes_new"]
merged[df.columns].to_csv(path, index=False)
print(f"Applied {int(apply.sum())} labels to labeling_sample.csv")
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/output/deliverable04/apply_buildout_<chunk>.py
```

---

## Step 4 — Verify (run after each session and at the end)

```python
import pandas as pd
df = pd.read_csv("phase2/output/deliverable04/labeling_sample.csv")
df["label"] = df["label"].fillna("").str.strip().str.lower()
print("totals by split x label:")
print(df.pivot_table(index="label", columns="split", values="candidate_id",
                     aggfunc="count", fill_value=0))
# Frozen test must be unchanged: 18 initiation / 18 decision / 118 neither.
test = df[df["split"] == "test"]
print("\nfrozen test (must stay 18/18/118):", test["label"].value_counts().to_dict())
print("blank remaining:", int(df["label"].eq("").sum()))
```

**Success criteria for the labeling phase:**
- **The OLD frozen test is untouched DURING labeling** — still exactly 18 init / 18 dec / 118 neither.
  (It gets re-drawn in Step 5, deliberately, only after labeling is fully done.)
- `train` initiation positives reach **~1,000** (from 142); decision reaches **~1,000** (from 147).
- No `candidate_id` appears twice; no previously-labeled row changed.

---

## Step 5 — Re-freeze the test set (ONE TIME, after all labeling is complete)

**Why.** The current test set (18 init / 18 dec / 118 neither) was frozen when we had almost no
labels. At ~1,000/head it is far too small to measure per-head — let alone per-process — performance
reliably (a per-process slice today has ~5 positives). We deliberately **re-draw the split once**, now
that a real labeled corpus exists, then freeze it **permanently** and never touch it again.

**Why this is methodologically sound (not leakage).** Splitting *after* labeling is the normal order
of operations: the labels were assigned purely from each date's meaning, with **no model involved**,
so labeling is independent of the eventual split. No model has been trained on the new labels yet, and
we draw the test set **before** the next retrain, so the test stays genuinely held out. We are only
replacing a *premature* freeze — and we re-freeze exactly once.

**How — stratified by process × label, fixed seed, written once over ALL labeled rows:**

```python
# phase2/code/deliverable04/_refreeze_test.py  (create + run ONCE, after labeling is done)
import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("activate nepa")
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LAB = ROOT / "phase2/output/deliverable04/labeling_sample.csv"
TEST_FRACTION = 0.20   # ~200/head test if ~1,000/head labeled
FLOOR_PER_CELL = 15    # ensure each process x label cell is measurable
SEED = 42

df = pd.read_csv(LAB)
df["label"] = df["label"].fillna("").str.strip().str.lower()
labeled = df[df["label"].isin(["initiation", "decision", "neither"])].copy()

# Fresh stratified draw over ALL labeled rows — dissolves the old split entirely.
test_idx = []
for (_proc, _lab), grp in labeled.groupby(["process_type", "label"]):
    n = max(FLOOR_PER_CELL, round(len(grp) * TEST_FRACTION))
    n = min(n, len(grp) - 1) if len(grp) > 1 else 0     # never take a whole cell
    test_idx += grp.sample(n=n, random_state=SEED).index.tolist()

df["split"] = "train"
df.loc[df.index.isin(test_idx), "split"] = "test"
df.to_csv(LAB, index=False)

t = df[df["split"] == "test"]
print("NEW frozen test (test_v2):")
print(t.pivot_table(index="process_type", columns="label",
                    values="candidate_id", aggfunc="count", fill_value=0))
print("test total:", len(t), "| train total:", int((df['split']=='train').sum()))
```

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/_refreeze_test.py
```

**After this runs — the protocol resumes exactly as before, permanently:**
- This is now **`test_v2`**, frozen forever. Never re-draw again.
- New labels added in future rounds default to `split = "train"` (the test never grows or leaks).
- **The old baseline / round-1 / round-2 F1 numbers are superseded** — they were measured on the old
  18/18/118 test and are no longer comparable. The progression table restarts from the first model
  trained against `test_v2`. (That's the intended cost of the revamp.)

**One honest caveat to record:** because the build-out *oversampled* positive-rich regex roles, the
labeled set (and therefore this test set) is far more positive-heavy than the real candidate pool
(~10% positive). So **per-head F1 / recall on `test_v2` is now reliable**, but **precision-in-deployment
must come from the operating curve** (`04b --curve`, which applies the model to the *full* pool with
its true class balance) — do not read test-set precision as deployment precision.

---

## Step 6 — Retrain (next session, after the re-freeze)

```bash
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04_classify_candidates.py --train --backend setfit
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04_classify_candidates.py --eval
```

(`--train` auto-refreshes diagnostics 01–04; `02_metrics_by_round.csv` becomes the new progression,
measured against `test_v2`.)

---

## Notes / guardrails for Codex

- **Steps 1–4 are the loop for the night** (emit → label → apply → count, repeated until ~1,000/head).
  Prioritize `clear_initiation` rows — the init head is the bottleneck. Partial is fine (resumable).
- **Step 5 (re-freeze) is run exactly ONCE, only after ALL labeling is complete** — it is the only
  place the `split` column is ever rewritten. Do not run it mid-labeling. If you're unsure labeling is
  "done," stop before Step 5 and leave it for the human to run.
- **Until Step 5, never touch `split == "test"` rows.** New rows are `split = "train"`.
- Do **not** run `04 --train` (Step 6) mid-way; only after re-freeze, and it's slow — leave it to the
  human unless told otherwise.
- Do **not** edit `04_classify_candidates.py`, `05_select_dates.py`, or any pipeline script — this is
  a data-labeling task only. (`_emit_buildout.py` and `_refreeze_test.py` are the only new scripts.)
- If a `model_context` excerpt is too truncated to decide, label `neither` and note
  "context too truncated to determine" rather than guessing.
