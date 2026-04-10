# Technology-Specific Extraction

**Purpose:** Build technology-specific fields for transmission lines, geothermal projects, and pipelines.
**Input:** `data/analysis/projects_combined.parquet`
**Output:** `data/analysis/projects_{transmission,geothermal,pipeline}.parquet`
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

---

## Transmission

**Cost:** Step 1 free (~5–15 min). Step 2 ~$0.06 with 4 workers (~2 min).

**Step 1 — Rule-based extraction + page-level length recovery:**

```bash
python code/extract/extract_technology.py --run transmission
```

**Step 2 — LLM adjudication for ambiguous multi-candidate rows:**

Requires `ANTHROPIC_API_KEY`. Run step 1 first.

```bash
export ANTHROPIC_API_KEY='INSERT-KEY-HERE'

python code/extract/extract_technology.py --run llm --workers 4
```

---

## Geothermal

**Cost:** Free (BERT classifier, no API calls).

**Step 1 — Rule-based identification and regex phase classification:**

```bash
python code/extract/extract_technology.py --run geothermal
```

**Step 2 — Fine-tune DistilBERT classifier on regex-labeled rows (~5 min):**

Requires additional packages not in the base environment. Run once to install, then run the trainer.

```bash
conda run -n nepa pip install "accelerate>=0.26.0" transformers torch scikit-learn

python code/extract/extract_technology.py --geothermal-phase-train
```

**Step 3 — Apply classifier to rows where `phase == 'unknown'`:**

```bash
python code/extract/extract_technology.py --geothermal-phase-classify
```

---

## Pipelines

**Cost:** Step 1 free. Step 2 ~$0.45 with 4 workers (~2 min).

**Step 1 — Rule-based extraction + page-level length recovery:**

```bash
python code/extract/extract_technology.py --run pipeline
```

**Step 2 — LLM adjudication for ambiguous multi-candidate rows:**

Requires `ANTHROPIC_API_KEY`. Run step 1 first.

```bash
export ANTHROPIC_API_KEY='INSERT-KEY-HERE'

python code/extract/extract_technology.py --run pipeline llm --workers 4
```

---

## Notes

- Each technology type is independent; run only the ones needed.
- The geothermal DistilBERT packages (`transformers`, `torch`) are not in the standard conda env — install them only when running geothermal step 2.
