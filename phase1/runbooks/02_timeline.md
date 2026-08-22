# Timeline Extraction (Phase 1 — BERT)

**Purpose:** Extract initiation and decision dates for clean energy projects using a BERT classifier trained on weak supervision labels.
**Input:** `data/analysis/projects_combined.parquet` + page-level text.
**Output:** `data/analysis/projects_timeline_bert.parquet` (CE), `data/analysis/projects_timeline_bert_ea_llm.parquet` (EA)
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

```
data/
└── analysis/
    ├── regex_candidates_ce.parquet   ← regex prep output (CE)
    ├── regex_candidates_ea.parquet   ← regex prep output (EA)
    ├── regex_candidates_eis.parquet  ← regex prep output (EIS)
    ├── projects_timeline_bert.parquet         ← CE final output
    ├── projects_timeline_bert_ea_llm.parquet  ← EA final output
    └── projects_timeline_bert_eis_llm.parquet ← EIS final output
```

---

## Full rebuild (all steps)

### Step 1 — Regex prep

Extracts date candidates from documents and saves per-source cache files.

```bash
python code/extract/extract_timeline.py --regex-prep
```

Output: `data/analysis/regex_candidates_{ce,ea,eis}.parquet`

### Step 2 — Generate BERT training data

Builds the training set from the regex cache using weak supervision labels.

```bash
python code/extract/extract_timeline.py --bert-generate
```

Output: `data/analysis/bert_traindata.parquet`

### Step 3 — Train BERT model

Trains a single 4-class sequence classifier over the weak-supervision labels. `--bert-model` defaults to `distilbert-base-uncased`, and that is what was trained and committed: `phase1/models/timeline_classifier/config.json` reports `DistilBertForSequenceClassification`. There is one model directory on disk, not per-source classifiers.

```bash
# Train all sources
python code/extract/extract_timeline.py --bert-train

# Train a single source
python code/extract/extract_timeline.py --bert-train --source CE
```

Output: `models/timeline_classifier_ce/`, `models/timeline_classifier_ea/`, `models/timeline_classifier_eis/`

### Step 4 — Run BERT inference

```bash
# CE (full run)
python code/extract/extract_timeline.py --bert-run \
  --source CE --output projects_timeline_bert.parquet

# EA (full run)
python code/extract/extract_timeline.py --bert-run \
  --source EA --output projects_timeline_bert_ea.parquet
```

---

## Quick run (reuse cached regex — skip step 1)

Use when regex candidates are already cached and only BERT needs to rerun.

```bash
python code/extract/extract_timeline.py --bert-run \
  --use-regex-cache --output projects_timeline_bert.parquet
```

---

## Sample / test runs

```bash
# 20-project sample (fast sanity check)
python code/extract/extract_timeline.py --bert-run \
  --sample 20 --output test20_bert.parquet

# Hybrid LLM validation (edge cases only — not for production CE)
python code/extract/extract_timeline.py --llm-run --hybrid \
  --use-regex-cache --sample 20 \
  --model llama3.2:3b-instruct-q4_K_M --timeout 180 --workers 4 \
  --output test20_hybrid.parquet
```

---

## Notes

- CE runs at scale (~19k projects) — BERT + rules only; no per-project LLM calls.
- EA/EIS are smaller (~500/700 projects) — LLM adjudication is feasible for edge cases.
- Initiation coverage is the primary bottleneck: CE ~30%, EA ~62%, EIS ~48% at Phase 1 freeze.
- Phase 2 (`phase2/runbooks/02_timeline.md`) documents the improved pipeline with spaCy enrichment and richer BERT inputs.
- Status and known issues: [architecture/code/extract_timeline.md](../architecture/code/extract_timeline.md)
