# Timeline Classifier: Training Workflow

## The Core Loop

The pipeline uses **weak supervision → BERT training → manual correction → retrain** to
iteratively improve date classification without hand-labeling tens of thousands of examples.

---

## Step 1 — Regex Prep (`--regex-prep`)

Run once (or when the document corpus changes).

Scans all PDF pages with regex, extracts every date candidate plus a surrounding context
window. Saves raw candidates to:
- `data/analysis/regex_candidates_ce.parquet`
- `data/analysis/regex_candidates_ea.parquet`
- `data/analysis/regex_candidates_eis.parquet`

This is the foundation everything else builds on. The context window (currently 256 chars)
is what BERT reads to classify each date.

---

## Step 2 — Weak Supervision (`--bert-generate`)

For every date candidate in the regex cache, `auto_label_context()` in `extract_timeline.py`
applies hand-written pattern rules to assign a label:

| Label | Meaning |
|---|---|
| `decision` | Signature/approval/ROD/FONSI/date of determination |
| `initiation` | Application received / NOI / project start signal |
| `review` | Intermediate reviewer sign-off, not the final decision |
| `other` | Reference dates, historical context, unrelated |

Output: `data/analysis/bert_training_data.parquet`

**The quality ceiling here is the quality of the patterns.** If `auto_label_context()`
systematically mislabels a class (e.g., CE DOE Initiator dates labeled `decision` instead of
`initiation`), BERT will learn that mistake at scale.

Manual corrections in `data/analysis/manual_training_corrections.csv` override weak labels
for specific `(project_id, date)` pairs before training.

---

## Step 3 — BERT Training (`--bert-train --source CE|EA|EIS`)

Trains a DeBERTa classifier on the weakly-supervised labels. BERT generalizes beyond the
exact regex patterns — it learns contextual signals the rules can't express.

**Model config by source:**

| Source | Model | Epochs | LR | Reason |
|---|---|---|---|---|
| CE | deberta-v3-base | 3 | 5e-5 | Large dataset (~100k rows), base model is stable |
| EA | deberta-v3-small | 5 | 2e-5 | Small dataset (~11k rows), small model avoids collapse |
| EIS | deberta-v3-small | 5 | 1e-5 | Large dataset for small model (~45k rows), needs lower LR |

Trained models saved to `models/timeline_classifier_{ce,ea,eis}/`.

**Known instability:** `deberta-v3-small` is numerically sensitive. The default 5e-5 LR
causes gradient NaN (complete model collapse) right after warmup. EA is stable at 2e-5;
EIS needs 1e-5 because its larger dataset means `warmup_steps=200` covers only ~3% of
training, leaving the model at peak LR too early.

---

## Step 4 — Evaluate and Identify Failures

After training, check `models/timeline_classifier_{source}/evaluation_report.json` for
per-class F1. Then run on a sample to find systematic failures:

```bash
python code/extract/extract_timeline.py --bert-run --sample 20 --source CE \
    --output data/analysis/test20_bert_ce.parquet
```

Look for projects where `bert_timeline_status == 'missing_initiation'` — these are the
primary failure mode for CE.

---

## Step 5 — Improve Weak Supervision Patterns

Two levers:

### A. Fix `auto_label_context()` patterns (multiplier effect)
Add or fix patterns in `INITIATION_PATTERNS_STRONG`, `INITIATION_PATTERNS_MED`, or
source-specific lists (e.g., `INITIATION_PATTERNS_CE_ONLY`). A single new pattern can
correctly relabel hundreds of training examples the next time `--bert-generate` runs.

This is the highest-leverage intervention. Do this before adding manual corrections.

### B. Add manual corrections (`data/analysis/manual_training_corrections.csv`)
For specific mislabeled examples that patterns can't easily generalize:
1. Find candidates using `code/manual_training/01_find_ce_initiation_candidates.py`
2. Review `data/manual_training/review_ce_initiation_candidates.csv`, set `correct_type`
3. Run `python code/manual_training/02_apply_corrections.py` to merge into corrections file

Corrections target specific `(project_id, date)` pairs and override the weak label.

---

## Step 6 — Retrain (Second Pass)

```bash
python code/extract/extract_timeline.py --bert-generate
python code/extract/extract_timeline.py --bert-train --source CE   # or EA, EIS
```

The new `--bert-generate` incorporates both improved patterns and manual corrections.
BERT now learns from cleaner labels, improving recall on the previously missed class.

---

## Step 7 — Full Run + LLM Adjudication

```bash
python code/extract/extract_timeline.py --bert-run \
    --output data/analysis/projects_timeline_bert.parquet
```

Projects where BERT still can't find a complete timeline (both initiation and decision)
are passed to Claude for adjudication (`--llm-adjudicate`). This is feasible for EA (~500
projects) and EIS (~700 projects), but not CE (~19k projects) — CE relies on BERT + rules only.

---

## Current Status (as of April 2026)

| Source | Model F1 (macro) | Primary gap | Next action |
|---|---|---|---|
| CE | — (needs retrain) | Missing initiation (~70% of projects) | Review CSV → retrain |
| EA | 0.985 | — | Ready for `--bert-run` |
| EIS | collapsed (needs retrain) | Gradient instability fixed, awaiting rerun | Retrain tonight |

## Workflow Summary

```
--regex-prep          (once)
      ↓
--bert-generate       ← auto_label_context() patterns
      ↓                 manual_training_corrections.csv
--bert-train
      ↓
--bert-run --sample   ← check for failures
      ↓
fix patterns / add corrections
      ↓
--bert-generate → --bert-train   (repeat until F1 satisfactory)
      ↓
--bert-run (full)
      ↓
--llm-adjudicate      (EA + EIS edge cases only)
```
