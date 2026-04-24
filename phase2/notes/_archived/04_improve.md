# Step 4 — Improving the Classifier

After an initial BERT training pass using weak supervision, the models will have systematic
errors — classes they consistently get wrong because the auto-labeling patterns were imprecise.
This step is how you find and fix those errors before the final run.

---

## The two improvement levers

### 1. Fix the weak supervision patterns (highest leverage)

`auto_label_context()` in `extract_timeline.py` applies regex patterns to assign labels
(`decision`, `initiation`, `review`, `other`) to every date candidate. If a pattern is wrong
or missing, BERT learns that mistake at scale across thousands of examples.

Fixing or adding a pattern to `INITIATION_PATTERNS_CE_ONLY` (or the equivalent lists) can
correct hundreds of mislabeled training examples the next time `--bert-generate` runs. Always
do this before adding individual corrections.

### 2. Provide labeled examples (supervised correction)

BERT's performance is bounded by what it has seen. Giving it explicit correct-vs-wrong examples
for the cases it gets systematically wrong lets it learn patterns the regex rules can't express —
varied phrasing, form-field layouts, OCR artifacts, etc.

This is done by building a review CSV, filling in the correct labels, and merging them into
the master corrections file before retraining.

---

## Supervised sample workflow (`code/manual_supervision/`)

This is the primary improvement mechanism after the first training pass.

**Step 1 — Generate samples**
```bash
conda run -n nepa python code/manual_supervision/01_build_supervision_samples.py
```
Produces one CSV per source in `data/manual_supervision/`:
- `review_CE.csv`, `review_EA.csv`, `review_EIS.csv`
- ~200 rows per source: **50 per class** (decision / initiation / review / other)
- Each 50 splits as: **20 clear-correct examples** (high-confidence, diverse) +
  **30 likely-mislabeled** (conflicting signals, known failure modes)

The `why` column explains what signals fired and flags the suspected conflict so you know
why a row was included.

**Step 2 — Review and label**

Open the CSV. For each row:
- Read `context` to see the text surrounding the date
- Read `why` to understand what the model currently thinks and why it may be wrong
- Set `correct_label` to one of: `initiation` | `decision` | `review` | `other`
- Leave blank if you are unsure

Focus especially on the `likely_mislabeled` rows — these are the cases that will actually
move the model. The `correct` rows anchor what good examples look like.

**Step 3 — Apply corrections**
```bash
conda run -n nepa python code/manual_supervision/02_apply_supervision.py
```
Merges filled-in rows into `data/analysis/manual_training_corrections.csv`. Safe to re-run
— deduplicates automatically.

**Step 4 — Retrain**
```bash
conda run -n nepa python code/extract/extract_timeline.py --bert-generate
conda run -n nepa python code/extract/extract_timeline.py --bert-train --source CE
conda run -n nepa python code/extract/extract_timeline.py --bert-train --source EA
conda run -n nepa python code/extract/extract_timeline.py --bert-train --source EIS
```

---

## What the sampler finds

For each class the script finds two pools:

| Sample type | How it's selected |
|---|---|
| **correct** | Auto_label matches class + structural signals confirm it (section heading, dep_verb, sig_flag) + no conflicting cues — one per project, spread across years/doc_types |
| **likely_mislabeled** | Conflicting signals: e.g. labeled `decision` but has DOE Initiator text; labeled `review` but section is `fonsi`; labeled `other` but section is `references` with a decision verb |

Conflict detection uses the `section_label`, `dep_verb`, `sig_flag`, and `ner_decision_signal`
columns added during `--regex-prep` (Phase 1), combined with targeted regex checks on the
context text.

---

## Corrections file

All labeled examples — from both this workflow and the older
`code/manual_training/` scripts — land in the same file:

```
data/analysis/manual_training_corrections.csv
columns: project_id, date, correct_type, source_file
```

`--bert-generate` reads this file and overrides any weak-supervision label where
`(project_id, date)` matches. You can add to it at any time and retrain without
re-running `--regex-prep`.

---

## When to iterate

Run a sample after retraining to check for remaining gaps:
```bash
conda run -n nepa python code/extract/extract_timeline.py \
    --bert-run --sample 20 --source CE --output data/analysis/test_ce_retrain.parquet
```

If `bert_timeline_status == 'missing_initiation'` is still common, generate another round
of supervision samples focused on that class and repeat. One or two rounds is usually enough
before the model is ready for the full run and LLM adjudication.
