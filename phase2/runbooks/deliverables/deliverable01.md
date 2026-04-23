# D1 — NEPA Trigger Classification

**Purpose:** Classify the federal nexus that triggers NEPA review for each clean energy project.
**Input:** `data/analysis/projects_combined.parquet` + source-level docs/pages parquets (CE, EA, EIS).
**Output:** `data/analysis/nepa_trigger/projects_nepa_trigger.parquet` + validation batch CSV.
**Cost:** LLM tier (optional) ~$1–4 (Claude Haiku) depending on share of unresolved cases.
**Scope:** 20,725 clean energy projects (`project_energy_type = 'Clean'`).
**Scripts:**
- `phase2/code/deliverable01/01_extract_nepa_trigger.py` — extraction pipeline
- `phase2/code/deliverable01/02_analyze_triggers.R` — analysis and figures

---

## Classification scheme

Seven primary trigger classes, in priority order (used to break ties when multiple signals are present):

| Priority | Class | Typical federal nexus |
|---|---|---|
| 1 | `federal_action` | Agency itself proposes or implements the action |
| 2 | `federal_program` | Programmatic EIS, resource management plan, rulemaking |
| 3 | `federal_property_transaction` | Land exchange, conveyance, disposal |
| 4 | `federal_land` | Project on or crossing federal land; ROW grant |
| 5 | `federal_permit` | Agency permit, license, or authorization required |
| 6 | `federal_funding` | Federal grant, loan guarantee, cost share |
| 7 | `unknown` | Clear NEPA review but nexus not identifiable |

**Primary trigger:** the federal nexus most directly responsible for the agency's NEPA decision.
**Secondary triggers:** additional nexuses clearly present (`nepa_trigger_secondary`, stored as Arrow list<string>).
**`is_dual_nexus`:** computed flag for `federal_land` primary + `federal_permit` secondary (transmission line case).

---

## Five-tier extraction pipeline

Each tier runs only on projects not resolved by a prior tier.

| Tier | Method | Source | Notes |
|---|---|---|---|
| 1a | Agency metadata heuristics | `projects_combined.parquet` | BLM/USFS/USACE agency codes; verb signals disambiguate `federal_action` vs `federal_land` |
| 1b | Keyword patterns on title + description | `projects_combined.parquet` | ~30 patterns per class; returns first match |
| 2 | Document title keyword scan | `*_documents.parquet` | Scans all documents for a project; programmatic title patterns fire here |
| 3 | Purpose and Need section | `*_pages.parquet` | CE: full doc scan; EA/EIS: P&N header detection, first 5 paragraphs |
| 4 | Embedding cosine similarity | `*_pages.parquet` | `all-MiniLM-L6-v2`; class prototype vectors; pages 1–10 only |
| 5 | Claude Haiku LLM | `*_pages.parquet` | Only with `--use-llm`; purpose-and-need text + structured JSON prompt |

**Resolution rule:** the `resolved` dict accumulates results tier-by-tier. A project exits the pipeline when it receives a classification; remaining projects pass to the next tier. Every project receives exactly one row in the output (`is_unique` assertion enforced before write).

---

## Quick run (no LLM)

Runs tiers 1a–4 only. Appropriate for an initial build or after pattern changes.

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py
```

Estimated runtime: ~10–20 min on full 20,725-project scope (embedding tier is the bottleneck).

---

## Full run (with LLM tier)

Adds tier 5 (Claude Haiku) for projects unresolved after embedding similarity.

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --use-llm
```

---

## Sample / smoke test

Run on a random subset before committing to the full corpus:

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --sample 50
```

---

## EDA before full run (optional)

Prints class balance, agency counts, and document coverage stats without writing output:

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --eda
```

---

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--use-llm` | off | Enable tier 5 (Claude Haiku) for unresolved cases |
| `--sample N` | None | Random sample of N projects (for testing) |
| `--eda` | off | Print descriptive stats and exit without writing output |

`ANTHROPIC_API_KEY` must be set in the environment when `--use-llm` is used.

---

## Outputs

### `data/analysis/nepa_trigger/projects_nepa_trigger.parquet`

One row per project. Key columns:

| Column | Type | Description |
|---|---|---|
| `project_id` | string | Primary key |
| `nepa_trigger_primary` | string | Primary trigger class |
| `nepa_trigger_secondary` | list<string> | Additional triggers detected (may be empty list) |
| `nepa_trigger_multi` | list<string> | All detected classes (primary + secondary) |
| `nepa_trigger_confidence` | string | `high`, `medium`, or `low` |
| `nepa_trigger_evidence_text` | string | Short snippet used to classify |
| `nepa_trigger_evidence_source` | string | Where evidence came from (see values below) |
| `nepa_trigger_rule_id` | string | Rule that fired (format: `T{tier}_{slug}`) |
| `nepa_trigger_manual_review` | bool | Flag for human review |
| `nepa_trigger_notes` | string | Analyst notes |
| `is_dual_nexus` | bool | `federal_land` primary + `federal_permit` secondary |
| `nepa_trigger_extraction_run_at` | string | ISO-8601 UTC timestamp of pipeline run |
| `nepa_trigger_llm_run_at` | string | ISO-8601 UTC timestamp of LLM call (empty string if tier 5 not used) |

**`nepa_trigger_evidence_source` values:** `agency_metadata`, `title`, `description`, `doc_title`, `purpose_and_need`, `embedding`, `llm`.

### `data/analysis/nepa_trigger/validation_batches.csv`

One row per rule that generated any `manual_review = TRUE` cases. Sorted by batch size descending so the highest-volume broken rules surface first.

| Column | Description |
|---|---|
| `nepa_trigger_rule_id` | Rule identifier |
| `nepa_trigger_primary` | Trigger class the rule fires into |
| `batch_size` | Number of flagged cases for this rule |
| `sample_project_ids` | Up to 20 project IDs to inspect |

---

## Validation

### Automated assertions (run at write time)

The script enforces these before writing output; the run fails if any assertion fires:

1. **Uniqueness:** `len(final_df) == len(projects_df)` — exactly one row per project.
2. **Scope:** all `project_id` values are in the 20,725-project clean energy set.
3. **List column:** `nepa_trigger_secondary` is a Python list, not a JSON string.

### Batch-by-rule review (manual)

After a successful run, review the highest-volume flagged rules:

```bash
# Print top 10 rules by flagged volume
python -c "
import pandas as pd
df = pd.read_csv('data/analysis/nepa_trigger/validation_batches.csv')
print(df.head(10).to_string(index=False))
"
```

For each rule with precision below target: correct patterns in the tier functions, re-run on a sample, and check the batch again. Accept a rule batch if manual precision ≥ 0.85.

### Quick distribution check

```python
import pyarrow.parquet as pq, pandas as pd

df = pq.read_table("data/analysis/nepa_trigger/projects_nepa_trigger.parquet").to_pandas()
print(df["nepa_trigger_primary"].value_counts())
print(f"\nManual review rate: {df['nepa_trigger_manual_review'].mean():.1%}  (target: < 5%)")
print(f"Dual-nexus projects: {df['is_dual_nexus'].sum()} ({df['is_dual_nexus'].mean():.1%})")
print(f"Unknown rate: {(df['nepa_trigger_primary']=='unknown').mean():.1%}  (target: < 10%)")
```

**Targets:**
- Manual review rate < 5%
- `unknown` share < 10%
- `federal_land` or `federal_permit` should be plurality for clean energy (expect ~40–60% combined)

---

## Analysis and figures

After the trigger parquet is produced, run the R analysis script:

```bash
Rscript phase2/code/deliverable01/02_analyze_triggers.R
```

**Prerequisites:** `usmap` package must be installed in R:

```r
install.packages("usmap")
```

**Outputs** (written to `output/deliverable01/`):

| File | Description |
|---|---|
| `fig1_trigger_by_process.png` | 100% stacked bar — primary trigger × CE/EA/EIS |
| `fig2_agency_trigger_heatmap.png` | Viridis heatmap — top 18 agencies × trigger class |
| `fig3_trigger_combinations.png` | Top 10 primary + secondary combinations |
| `fig4_trigger_by_technology.png` | 100% stacked bar — trigger × energy technology |
| `fig5_state_choropleth.png` | State map — dominant trigger per state |
| `trigger_evidence_excerpts.csv` | 2 high-confidence quotable examples per class |
| `trigger_source_distribution.csv` | Evidence source × confidence breakdown |
| `trigger_rule_distribution.csv` | Top 25 rules by volume |

Fig 6 (trigger × review duration) is a placeholder — uncomment in the R script after `D4` timeline data (`data/analysis/projects_timeline_bert.parquet`) is available.

---

## Notes

- **`nepa_trigger_secondary` is a list-column in R.** `tidyr::unnest(df, nepa_trigger_secondary)` explodes it for frequency counts. `purrr::map2_chr()` handles it in Fig 3 (combination bar). Do not convert to a JSON string.
- **CE documents:** tiers 2–3 scan the full document (no page cap) because CEs are 1–3 pages. EA/EIS scan pages 1–10 only for performance.
- **Document routing** uses the `dataset_source` column (`CE`, `EA`, `EIS`) from `projects_combined.parquet` to select the correct `*_documents.parquet` and `*_pages.parquet` paths.
- **Programmatic patterns** are defined inline in the script (not imported from `extract_reviews.py`). `extract_reviews.py` is not a dependency of this pipeline.
- **`nepa_trigger_rule_id` format:** `T{tier}_{slug}` (e.g., `T1a_BLM_land`, `T1b_row_grant`, `T3_pan_sec404`). Use this to trace any classification back to the exact pattern that fired.
- **SetFit classifier** is documented in the plan but held in reserve. If the unknown rate after tiers 1–4 exceeds 15%, revisit SetFit as an additional tier between 4 and 5.
