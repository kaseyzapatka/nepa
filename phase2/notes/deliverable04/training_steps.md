# D4 Timeline — Gold Label & BERT Training Steps

## Overview

The gold label system works at two levels:

- **Project-level** (`gold_initiation_date`, `gold_decision_date`, etc.) — were the pipeline's selected dates correct?
- **Candidate-level** (`gold_candidate_role`, `gold_error_category`) — is each extracted context window labeled with the right role? This is what trains BERT.

Three splits are defined in `timeline_gold_splits.parquet`:

| Split | Projects | Purpose |
|---|---:|---|
| `diagnostic_balanced_v2` | 153 | Understand error distribution before committing to full labeling |
| `train_enriched_v1` | ~1,000 | BERT fine-tuning training set |
| `test_representative_v1` | ~500 | Held-out evaluation |

The diagnostic split is 9 cells (CE/EA/EIS × Clean/Fossil/Other, ~17 each), stratified by `workflow_condition`:

| Condition | ~% | Meaning |
|---|---|---|
| `missing_failed` | 25% | Pipeline found no date — needs manual labeling |
| `ambiguous` | 25% | Multiple candidates, uncertain selection |
| `apparent_success` | 20% | Pipeline succeeded — verify it's correct |
| `retrieval_candidate_weak` | 15% | Poor candidate retrieval |
| `structured_edge` | 15% | Edge case (Tier A metadata, single-doc, etc.) |

The 100-project sample from script 00 is hard-required in the diagnostic split.

---

## Immutability rule

`01_build_gold_samples.py` has an explicit guard that **refuses to overwrite** existing splits
if the selected project IDs would change. This protects labeled data from being silently
invalidated by pipeline re-runs.

- **Default:** run without flags → safe, immutability enforced
- **`--overwrite-existing`:** deliberately replace splits (only once, before labeling starts)
- **Never** pass `--overwrite-existing` after you have imported any labels

---

## Phase A — Stabilize the pipeline

Run once after major pipeline changes (FEIS priority boost, new patterns, etc.).

**A1. Full pipeline re-run** (~3 hours, unattended):
```bash
cd /Users/Dora/git/consulting/nepa
conda run -n nepa python phase2/code/deliverable04/01_index.py
conda run -n nepa python phase2/code/deliverable04/02_retrieve.py --force
conda run -n nepa python phase2/code/deliverable04/03_extract_candidates.py --force
conda run -n nepa python phase2/code/deliverable04/05_select_dates.py --force
conda run -n nepa python phase2/code/deliverable04/07_validate.py --prepare-review
```

**A2. Rebuild gold splits from fresh pipeline output** (ONE-TIME, deliberate — do this once after the pipeline stabilizes, then never again with `--overwrite-existing`):
```bash
conda run -n nepa python phase2/code/deliverable04/validation/01_build_gold_samples.py --overwrite-existing
```

Outputs:
- `phase2/data/analysis/timeline/gold/timeline_gold_splits.parquet`
- `phase2/output/deliverable04/gold/splits/diagnostic_balanced_v2.csv`
- `phase2/output/deliverable04/gold/splits/diagnostic_balanced_v2_ids.txt`
- `phase2/output/deliverable04/gold/splits/diagnostic_balanced_v2_manifest.json`
- (same for train and test splits)

---

## Phase B — Prepare review packets (~30 min)

**B1. Pre-label with pipeline's best guess** (fills `gold_*` columns so you verify rather than label from scratch):
```bash
conda run -n nepa python phase2/code/deliverable04/validation/04_codex_prelabel_gold_packets.py \
    --split diagnostic_balanced_v2
```

**B2. Generate annotatable review CSVs** (50 projects per batch = 3 batches for diagnostic):
```bash
conda run -n nepa python phase2/code/deliverable04/validation/02_prepare_gold_review_packets.py \
    --split diagnostic_balanced_v2
```

Outputs (two files per batch):
- `phase2/output/deliverable04/gold/review_packets/diagnostic_balanced_v2_batch001_projects.csv`
- `phase2/output/deliverable04/gold/review_packets/diagnostic_balanced_v2_batch001_candidates.csv`
- (repeat for batch002, batch003)

---

## Phase C — Manual review (~4–8 hours for 153 projects)

Open each batch's two CSVs. Work through them project by project.

### Projects CSV — fill these columns:

| Column | What to enter |
|---|---|
| `gold_initiation_date` | Correct initiation date (YYYY-MM-DD) or blank if none |
| `gold_initiation_granularity` | `day`, `month`, or `year` |
| `gold_initiation_candidate_id` | ID of the candidate that should have been selected (from candidates CSV) |
| `gold_initiation_evidence_text` | Paste the supporting text |
| `gold_initiation_confidence` | `high`, `medium`, or `low` |
| `gold_initiation_missing_reason` | If no date: `no_document`, `image_pdf`, `not_in_nepatec`, `ambiguous_context` |
| `gold_decision_date` | (same pattern as initiation) |
| `gold_ambiguity_flag` | `TRUE` if genuinely uncertain |
| `gold_notes` | Any useful notes for future reviewers |
| `review_status` | `approved` (pipeline correct), `corrected` (you changed it), `needs_adjudication` |
| `reviewer` | Your name |

### Candidates CSV — fill these columns:

| Column | What to enter |
|---|---|
| `gold_candidate_role` | True role: `clear_decision`, `clear_initiation`, `proxy_decision`, `proxy_initiation`, `review`, `unknown`, `reject` |
| `gold_selected_for` | `decision`, `initiation`, `both`, or blank |
| `gold_error_category` | What went wrong: `false_positive`, `wrong_role`, `missed_date`, `correct` |
| `gold_candidate_notes` | Optional |

### Tips by workflow condition:

- **`apparent_success`**: Pre-labeled values are usually right. Scan and mark `approved`.
- **`ambiguous`**: Read all candidate evidence texts. Pick the most defensible date. Note your reasoning.
- **`missing_failed`**: The pipeline found nothing. Read the evidence and decide if a date exists but wasn't extracted, or is genuinely absent.
- **`retrieval_candidate_weak`** / **`structured_edge`**: Edge cases — use your judgment, flag anything unusual.

---

## Phase D — Import and validate labels (~15 min)

Run after completing each batch:
```bash
conda run -n nepa python phase2/code/deliverable04/validation/03_import_gold_labels.py \
    --split diagnostic_balanced_v2 --batch 1
# repeat for --batch 2, --batch 3
```

Outputs:
- `phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet`
- `phase2/data/analysis/timeline/gold/timeline_gold_candidates.parquet`
- `phase2/data/analysis/timeline/gold/timeline_gold_irr.parquet` (inter-rater reliability)
- `phase2/output/deliverable04/gold/reconciliation_queue.csv` (disagreements to resolve)

After all batches are imported, review the IRR report. Precision target: ≥95% for
decision dates (CE, EA, EIS), ≥90% for clear initiation (EA/EIS), ≥85% (CE).

---

## Phase E — Scale up with LLM pseudo-labeling (before BERT training)

153 diagnostic projects × ~7–8 candidates = ~1,100–1,200 labeled candidates. That's
enough for a pilot BERT run but not the full training set (target ~7,000).

**Efficient path:** use script 06 (Claude Haiku) to auto-label the training split candidates
via the API, then spot-check 15–20% manually. This produces ~7,000 labels overnight for <$10.

To pseudo-label:
```bash
conda run -n nepa python phase2/code/deliverable04/06_adjudicate_llm.py \
    --mode candidate_adjudication --process EA EIS
```

After spot-checking, import with `03_import_gold_labels.py`.

---

## Phase F — BERT fine-tuning (not yet built)

Once `timeline_gold_candidates.parquet` has ≥2,000 labeled examples:

1. Build a training-ready flat file from `gold_candidates` (context window + true role label)
2. Fine-tune `distilbert-base-uncased` or `bert-base-uncased` on the 8-class classification task
3. Export as ONNX, integrate into script 03 via `classifier_signal` in `_compute_candidate_score`
4. Re-run scripts 03–04 with classifier enabled

The architecture doc notes: **SetFit is NOT appropriate** for this task (it's a label quality
problem, not a data scarcity problem). Standard fine-tuned BERT on the full training set is the
correct approach.

---

## Key file locations

| File | Path |
|---|---|
| Gold split definitions | `phase2/data/analysis/timeline/gold/timeline_gold_splits.parquet` |
| Labeled project dates | `phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet` |
| Labeled candidates (BERT training data) | `phase2/data/analysis/timeline/gold/timeline_gold_candidates.parquet` |
| IRR report | `phase2/data/analysis/timeline/gold/timeline_gold_irr.parquet` |
| Split CSVs | `phase2/output/deliverable04/gold/splits/` |
| Review packets | `phase2/output/deliverable04/gold/review_packets/` |
| Build splits script | `phase2/code/deliverable04/validation/01_build_gold_samples.py` |
| Prepare packets script | `phase2/code/deliverable04/validation/02_prepare_gold_review_packets.py` |
| Import labels script | `phase2/code/deliverable04/validation/03_import_gold_labels.py` |
| Pre-label script | `phase2/code/deliverable04/validation/04_codex_prelabel_gold_packets.py` |
