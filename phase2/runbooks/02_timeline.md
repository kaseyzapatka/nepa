# Timeline Extraction

**Purpose:** Extract initiation and decision dates for NEPA reviews using a BERT classifier with optional LLM adjudication.
**Input:** `data/analysis/projects_combined.parquet` + regex candidate cache.
**Output:** `data/analysis/timeline_{ce,ea,eis}.parquet`, `timeline_{ea,eis}_llm.parquet`, `timeline_targeted_llm.parquet`
**Cost:** LLM adjudication ~$0.60–$1.00 (Claude Haiku) for EA + EIS full runs. Targeted re-adjudication ~$0.44.
**Prerequisites:** Base dataset built ([runbook 01](01_base_dataset.md)).

> **Note on prior organization:** The targeted re-adjudication for programmatic/tiered projects was previously documented as a standalone "Deliverable 02" section in the README. It is step 7 in the full rebuild below because it operates on timeline outputs — it is a timeline operation, not a reviews operation. Reviews ([runbook 03](03_reviews.md)) must still run before step 7.

---

## File naming convention

| Stage | CE | EA | EIS |
|---|---|---|---|
| Regex cache | `timeline_regex_ce.parquet` | `timeline_regex_ea.parquet` | `timeline_regex_eis.parquet` |
| BERT training data | `bert_traindata.parquet` | ← same file | ← same file |
| Sample test (post-train) | `timeline_ce_sample{N}.parquet` | `timeline_ea_sample{N}.parquet` | `timeline_eis_sample{N}.parquet` |
| Full BERT run | `timeline_ce.parquet` | `timeline_ea.parquet` | `timeline_eis.parquet` |
| Post LLM adjudication | *(none)* | `timeline_ea_llm.parquet` | `timeline_eis_llm.parquet` |

Append `_v{N}` to sample files when iterating through multiple training rounds (e.g. `timeline_ce_sample20_v2.parquet`). Full run files are overwritten each time.

---

## Full rebuild (patterns or models changed)

Use when regex patterns or training logic have changed and classifiers need retraining from scratch.

```bash
# Step 1 — Rebuild regex candidate cache for all sources
python code/extract/extract_timeline.py --regex-prep --source CE
python code/extract/extract_timeline.py --regex-prep --source EA
python code/extract/extract_timeline.py --regex-prep --source EIS
# Output: data/analysis/timeline_regex_{ce,ea,eis}.parquet

# Step 2 — Rebuild BERT training data
python code/extract/extract_timeline.py --bert-generate
# Output: data/analysis/bert_traindata.parquet

# Step 3 — Retrain classifiers
python code/extract/extract_timeline.py --bert-train --source CE
python code/extract/extract_timeline.py --bert-train --source EA
python code/extract/extract_timeline.py --bert-train --source EIS

# Step 4 — Smoke test on 20 projects before committing to full run
python code/extract/extract_timeline.py --bert-run --sample 20 --source CE \
    --output data/analysis/timeline_ce_sample20.parquet

# Step 5 — Full BERT inference
python code/extract/extract_timeline.py --bert-run --source CE \
    --output data/analysis/timeline_ce.parquet
python code/extract/extract_timeline.py --bert-run --source EA \
    --output data/analysis/timeline_ea.parquet
python code/extract/extract_timeline.py --bert-run --source EIS \
    --output data/analysis/timeline_eis.parquet

# Step 6 — LLM adjudication for EA and EIS (not needed for CE at scale)
export ANTHROPIC_API_KEY='sk-ant-...'

python code/extract/extract_timeline.py --llm-adjudicate \
    --input data/analysis/timeline_ea.parquet \
    --provider claude \
    --output data/analysis/timeline_ea_llm.parquet

python code/extract/extract_timeline.py --llm-adjudicate \
    --input data/analysis/timeline_eis.parquet \
    --provider claude \
    --output data/analysis/timeline_eis_llm.parquet

# Step 7 — Targeted re-adjudication for programmatic/tiered projects
# Run after step 6 AND after reviews extraction (runbook 03) is complete.
# Auto-selects ~73 programmatic/tiered projects with missing initiation or decision dates.
python code/extract/extract_timeline.py \
    --llm-adjudicate \
    --input data/analysis/timeline_ea_llm.parquet,data/analysis/timeline_eis_llm.parquet \
    --nonstandard-incomplete \
    --max-candidates 125 \
    --context-chars 400 \
    --promote-rod-language \
    --year-window 15 \
    --provider claude \
    --output data/analysis/timeline_targeted_llm.parquet
```

**Step 7 flag notes:**
- `--nonstandard-incomplete` — filters to programmatic/tiered projects with missing dates; no manual ID file needed.
- `--max-candidates 125` — raises cap from 30 (EIS default) to 125 for large programmatic EISs.
- `--promote-rod-language` — promotes ROD/FONSI dates to Tier A even if BERT mislabeled them.
- `--year-window 15` — drops candidates >15 years before the latest found date, removing NEPA citation noise.
- Targeted output is ~73 rows. Full timeline files are not modified. Patched automatically by `00_setup.R`.

---

## Quick run (models already trained)

Use when models are current and only BERT inference + adjudication outputs need refreshing.

**CE only:**

```bash
python code/extract/extract_timeline.py --bert-run --source CE \
    --output data/analysis/timeline_ce.parquet
```

**EA:**

```bash
python code/extract/extract_timeline.py --bert-run --source EA \
    --output data/analysis/timeline_ea.parquet

python code/extract/extract_timeline.py --llm-adjudicate \
    --input data/analysis/timeline_ea.parquet \
    --provider claude \
    --output data/analysis/timeline_ea_llm.parquet
```

**EIS:**

```bash
python code/extract/extract_timeline.py --bert-run --source EIS \
    --output data/analysis/timeline_eis.parquet

python code/extract/extract_timeline.py --llm-adjudicate \
    --input data/analysis/timeline_eis.parquet \
    --provider claude \
    --output data/analysis/timeline_eis_llm.parquet
```

---

## Filtering and debugging

**Restrict to clean energy projects only** (add `--clean-energy` to any `--bert-run` command):

```bash
python code/extract/extract_timeline.py --bert-run --source EA --clean-energy \
    --output data/analysis/timeline_ea_clean.parquet
```

**Debug a single project** (searches CE, EA, and EIS sources automatically):

```bash
python code/extract/extract_timeline.py --project-id <UUID> --hybrid --use-regex-cache
```

---

## Manual training corrections (iterative improvement)

**On the first pass, skip manual corrections entirely** — run steps 1–5 above using weak supervision alone to get a working baseline. Then inspect the output, add corrections, and retrain. You do not need to get corrections right before starting.

After a full run, inspect the output for systematic errors (e.g. CE projects still missing initiation dates). Corrections are added to:

```
data/analysis/manual_training_corrections.csv
```

**Required columns:** `project_id`, `date`, `correct_type` (one of `decision`, `initiation`, `review`, `other`)

Once corrections are added, you only need to re-run steps 2–5 (not step 1 — the regex cache is unchanged):

```bash
# Regenerate training data with corrections applied
python code/extract/extract_timeline.py --bert-generate

# Retrain affected source(s)
python code/extract/extract_timeline.py --bert-train --source CE

# Smoke test
python code/extract/extract_timeline.py --bert-run --sample 20 --source CE \
    --output data/analysis/timeline_ce_sample20_v2.parquet

# Re-run full inference
python code/extract/extract_timeline.py --bert-run --source CE \
    --output data/analysis/timeline_ce.parquet
```

**Matching note:** Corrections are matched on `(project_id, date)`. If the same date appears in multiple contexts for a project, the correction applies to all of them. You can make the match more precise by also checking `context` in the CSV (not currently implemented — avoid duplicate dates per project when possible).

**Timing:** Add corrections after inspecting `--bert-run` output and identifying a cohort of systematic misses (e.g. 20–50 CE initiation false negatives). Corrections before the first run are not useful — you need the model's output to know what to fix. Plan on at least one correction cycle after the initial build.

---

## Notes

- `--source` accepts `CE`, `EA`, `EIS`, or comma-separated combinations. Defaults to `CE` if omitted.
- spaCy enrichment runs by default during `--regex-prep`. Use `--no-spacy` to skip.
- Initiation class imbalance is a known bottleneck; improve via pattern expansion + weighting + manual examples.
- Timeline selection prioritizes strong signature cues over latest-date heuristics.
- `dep_verb`, `sig_flag`, `ner_decision_signal` are the spaCy-enriched columns used by BERT.
