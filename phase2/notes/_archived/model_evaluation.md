# Model Training: How to Interpret Output and Evaluate Results

## What Is Weak Supervision?

Normally, training a classifier requires humans to manually label thousands of examples
(e.g., "this date is an initiation date," "this date is a decision date"). That's slow
and expensive.

**Weak supervision** skips the manual labeling step by using heuristic rules — in this
case, regex patterns — to automatically generate labels. The labels aren't perfect, but
they're good enough to bootstrap a model.

For example: if the text around a date contains "Record of Decision" or "ROD signed,"
a regex assigns the label `decision`. If it contains "Notice of Intent" or "scoping
meeting," it gets `initiation`. The model then learns these patterns — and more
importantly, learns to generalize them to phrasing the regexes never explicitly covered.

**The tradeoff:** You get labeled data for free, but the model's ceiling is the quality
of your heuristics. Systematic regex mistakes become systematic model mistakes. That's
why the human review + manual corrections loop matters — it's how you inject ground
truth and push past the weak supervision ceiling.

---

## The Bootstrap Loop

This pipeline uses weak supervision — no manually labeled data is required to start.

```
Raw text
   ↓
Regex weak labels  ←──────────────────┐
   ↓                                  │
Train model on weak labels            │
   ↓                                  │
Run model on full dataset             │
   ↓                                  │
Human review → manual corrections     │
   ↓                                  │
Retrain (corrections override regexes)┘
   ↓
Good enough → use outputs
```

The regexes in `run_regex_prep()` auto-label candidate dates as `decision`, `initiation`,
`review`, or `other` based on surrounding context keywords. The model learns to replicate
this logic — and generalize it to cases the regexes miss.

**What the F1 score actually measures:** Agreement with the regex labels on held-out projects.
A 98% F1 means "the model agrees with the regexes 98% of the time" — not "98% correct on
real-world data." Real-world accuracy is validated by human review after `--bert-run`.

---

## Reading Training Log Output

Each log line looks like:
```
{'loss': 0.0159, 'grad_norm': 0.062, 'learning_rate': 1.65e-05, 'epoch': 3.63}
```

### `loss`
How wrong the model's predictions are. Lower = better. Range: 0 to ~2+.
- Starting loss ~1.3–1.4 is normal (random weights on 4 classes)
- Healthy training drops quickly: 1.3 → 0.9 → 0.4 → 0.2 in epoch 1
- By epoch 3+, values like 0.015–0.05 indicate the model has converged
- If loss stops dropping or rises, the model has peaked (see overfitting below)

### `grad_norm`
The magnitude of the weight update at each step. Think of it as "how hard the model
is correcting itself."
- Normal range: 0.01 – 5.0
- Occasional spikes to 10–20 are fine (model hit a hard batch and corrected)
- Consistently >50: unstable training, may need lower learning rate
- Trending toward 0 (like 0.02–0.06): model has settled, weights barely changing

### `learning_rate`
Shows where you are in the LR schedule:
- Starts low (~1e-5), ramps up during warmup (~200 steps), then decays
- Seeing it decrease (like epoch 3+) means you're in the decay phase — normal

### `epoch`
Fractional progress through training. `epoch: 3.63` = 63% through epoch 4 of 5.

---

## Reading Evaluation Output

Printed at the end of each epoch (when `eval_strategy="epoch"`):

```
{'eval_loss': 0.036, 'eval_accuracy': 0.983, 'eval_f1_macro': 0.974,
 'eval_f1_decision': 0.992, 'eval_f1_initiation': 0.984,
 'eval_f1_review': 0.962, 'eval_f1_other': 0.960, 'epoch': 2.0}
```

### `eval_f1_macro`
Average F1 across all 4 classes. **This is the primary metric.**
- >0.90: excellent
- 0.75–0.90: good
- 0.50–0.75: acceptable, worth retraining with corrections
- <0.50: class collapse or serious problem

### Per-class F1 (`eval_f1_initiation`, etc.)
F1 per label. Watch `eval_f1_initiation` most closely — it's the hardest class due to
imbalance (fewer examples, more ambiguous language).
- If any class F1 = 0.0: class collapse — model never predicts that class
- `eval_f1_initiation` < 0.60 after training: need more initiation examples or manual corrections

### `eval_loss`
Same as training loss but on the held-out test set. Should track training loss closely.
If eval_loss rises while training loss keeps dropping → overfitting.

### Overfitting signal
`load_best_model_at_end=True` protects against this automatically. If epoch 3 eval
is worse than epoch 2, the epoch 2 checkpoint is saved as the final model.

---

## Class Collapse

**What it is:** The model predicts only the majority class (e.g., always "decision"),
getting F1=0 for all other classes.

**Example of a collapsed model (EA deberta-v3-base, first training attempt):**
```json
{"decision": {"f1-score": 0.531, "recall": 1.0},
 "initiation": {"f1-score": 0.0},
 "review": {"f1-score": 0.0},
 "other": {"f1-score": 0.0}}
```

**Cause:** Model too large for the dataset size. deberta-v3-base (184M params) on 7.5K
rows couldn't learn minority classes — it just predicted the majority class every time.

**Fix:** Switch to a smaller model. deberta-v3-small (44M params) on the same data
produced eval_f1_macro=0.974.

---

## Weak Supervision Ceiling

The 98% F1 does NOT mean the model is 98% accurate on real project timelines.

It means:
- The model has learned to replicate the regex labeling rules
- It generalizes well to phrasing the regexes didn't cover
- It will make the same systematic mistakes the regexes make

**The ceiling:** If regexes mislabel certain doc types or phrasings, the model learns
those mistakes too. Real accuracy is only measurable by human review.

---

## Improving with Manual Corrections

After running `--bert-run` and reviewing outputs:

1. Create `data/analysis/manual_training_corrections.csv` with columns:
   ```
   project_id, date, label, context, dataset_source, doc_type
   ```
2. Add rows for projects where the model was wrong — correct date + correct label
3. Regenerate training data (corrections override regex labels for matching project_id+date):
   ```bash
   python code/extract/extract_timeline.py --bert-generate
   python code/extract/extract_timeline.py --bert-train --source EA
   ```

**Where to focus corrections:**
- Initiation dates first (hardest class, most impact)
- Projects where `bert_confidence = 'low'` or `timeline_status != 'complete'`
- 20–30 corrections per source is enough to see meaningful improvement

Most pipelines converge in 2–3 loops.

---

## Training Commands Reference

```bash
# Regenerate weak-supervision training data (all sources)
python code/extract/extract_timeline.py --bert-generate

# Train source-specific models (auto-selects model size and epochs by source)
python code/extract/extract_timeline.py --bert-train --source EA   # deberta-v3-small, 5 epochs
python code/extract/extract_timeline.py --bert-train --source EIS  # deberta-v3-small, 5 epochs
python code/extract/extract_timeline.py --bert-train --source CE   # deberta-v3-base, 3 epochs

# Run extraction with source-specific models
python code/extract/extract_timeline.py --bert-run --source EA --output projects_timeline_bert_ea.parquet
python code/extract/extract_timeline.py --bert-run --source EIS --output projects_timeline_bert_eis.parquet
python code/extract/extract_timeline.py --bert-run --source CE --output projects_timeline_bert_ce.parquet
```

Evaluation reports saved to `models/timeline_classifier_{source}/evaluation_report.json` after each training run.
