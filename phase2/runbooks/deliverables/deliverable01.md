# D1 — NEPA Trigger Classification

**Purpose:** Classify the federal nexus that triggers NEPA review for each clean energy project.
**Input:** `data/analysis/projects_combined.parquet` + source-level docs/pages parquets (CE, EA, EIS).
**Output:** `data/analysis/nepa_trigger/projects_nepa_trigger.parquet`, `data/analysis/nepa_trigger/projects_funding_details.parquet`, and validation batch CSV.
**Cost:** LLM tier (optional) ~$1–4 (Claude Haiku) depending on share of unresolved cases; Tier 5 has a hard budget guardrail (default $10).
**Scope:** 20,725 clean energy projects (`project_energy_type = 'Clean'`).
**Scripts:**
- `phase2/code/deliverable01/01_extract_nepa_trigger.py` — extraction pipeline
- `phase2/code/deliverable01/02_create_figures.R` — analysis and figures

**Reference docs** (in `phase2/code/deliverable01/`):
- `_notes.md` — tactical notes, model selection rationale, threshold guidance
- `_legend.md` — rule ID format and common examples
- `_example_bank.md` — calibration examples (positives and hard negatives per class)
- `tier4_refactor_spec.md` — Tier 4 design spec
- `tier4_implementation_checklist.md` — implementation sequence

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

Each tier runs on projects not yet finalized by a prior tier. Tiers 1–3 produce results that are either **auto-accepted** (deterministic high-precision rules) or held as **provisional** (awaiting Tier 4 confirmation). Tier 4 processes all unfinalized projects.

| Tier | Method | Source | Notes |
|---|---|---|---|
| 1a | Agency metadata heuristics | `projects_combined.parquet` | BLM/USFS/NPS/FWS → `federal_land` (verb signals disambiguate action); FERC/FAA/FCC → `federal_permit`; DOE → provisional (funding vs action boundary) |
| 1b | Keyword patterns on title + description | `projects_combined.parquet` | ~30 patterns per class; returns first match; negation guard suppresses checklist false positives |
| 2 | Document title keyword scan | `*_documents.parquet` | Scans all documents for a project; programmatic title patterns fire here |
| 3 | Purpose and Need section | `*_pages.parquet` | CE: full doc scan with conservative pattern set (no sec404/arra/rmp); EA/EIS: P&N header detection, first 10 pages |
| 4 | Retrieval-first local NLI | `*_pages.parquet` | Chunk retrieval + `cross-encoder/nli-MiniLM2-L6-H768` zero-shot NLI; embedding fallback if NLI unavailable; DOE and CE-heavy rows route here |
| 5 | Claude Haiku LLM | Tier 4 evidence bundles | Only with `--use-llm`; receives top chunks + local NLI scores + provisional class; subject to budget guardrail |

**Routing policy:**
- `AUTO_ACCEPT_RULE_IDS` — deterministic high-precision rules that finalize immediately (e.g., `T1a_FERC_permit`, `T1b_special_use`, `T3_npdes`)
- `SEND_TO_TIER4_RULE_IDS` — ambiguous rules always routed to Tier 4 (e.g., `T1a_DOE_action`, `T3_sec404`, `T3_arra`)
- `AUDIT_FIRST_RULE_IDS` — rules held for review before promotion (e.g., `T3_rmp`)

---

## Recommended workflow for a new full run

### Step 1 — Calibrate hypotheses (first time, or after editing `HYPOTHESIS_TEMPLATES`)

Validates that the NLI hypothesis templates are well-calibrated before running at scale.
Takes ~2 minutes. Downloads the NLI model (~67MB) on first run.

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --calibrate
```

**Passing criteria:**
- Positive examples: correct class scores ≥ 0.75
- Hard negatives: all class scores ≤ 0.50

If any check fails, adjust `HYPOTHESIS_TEMPLATES` in the script and re-run calibration before proceeding. See `_example_bank.md` for guidance on hypothesis wording.

### Step 2 — Quick run (no LLM)

Runs tiers 1a–4. Appropriate for an initial build or after pattern changes.

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py
```

### Step 3 — Full run (with LLM tier)

Adds Tier 5 (Claude Haiku) for the small uncertain queue remaining after Tier 4. Tier 5 has a preflight guardrail: if the estimated spend exceeds the budget cap (default $10), it writes `tier5_queue.parquet` and stops unless `--force-tier5` is passed.

```bash
export ANTHROPIC_API_KEY='sk-ant-...'

conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --use-llm
```

### Step 4 — Refresh funding details only

Regenerates `projects_funding_details.parquet` from the existing trigger output without rerunning
the trigger classifier or rewriting `projects_nepa_trigger.parquet`.

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --funding-details-only
```

---

## Sample / smoke test

Run on a random subset before committing to the full corpus:

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --sample 50
```

---

## EDA before full run (optional)

Prints description coverage and process-type breakdown without writing output:

```bash
conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --eda
```

---

## CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--calibrate` | off | Validate NLI hypothesis templates against example bank; exit without extracting |
| `--eda` | off | Print descriptive stats and exit without writing output |
| `--use-llm` | off | Enable Tier 5 (Claude Haiku) for uncertain cases |
| `--force-tier5` | off | Override Tier 5 budget guardrail and send full queue |
| `--tier5-budget` | 10.0 | Hard stop budget in USD for Tier 5 spend |
| `--funding-details-only` | off | Regenerate `projects_funding_details.parquet` from the existing trigger output without rewriting `projects_nepa_trigger.parquet` |
| `--sample N` | None | Random sample of N projects (for testing) |

`ANTHROPIC_API_KEY` must be set in the environment when `--use-llm` is used.

---

## Outputs

### Primary output: `data/analysis/nepa_trigger/projects_nepa_trigger.parquet`

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
| `is_dual_nexus` | bool | `federal_land` primary + `federal_permit` secondary |
| `nepa_trigger_extraction_run_at` | string | ISO-8601 UTC timestamp of pipeline run |
| `nepa_trigger_llm_run_at` | string | ISO-8601 UTC timestamp of LLM call (empty string if Tier 5 not used) |

**`nepa_trigger_evidence_source` values:** `agency_metadata`, `title`, `description`, `doc_title`, `purpose_and_need`, `document_text`, `embedding`, `llm`.

### Funding details sidecar: `data/analysis/nepa_trigger/projects_funding_details.parquet`

One row per project where `nepa_trigger_primary == "federal_funding"`. This sidecar is intentionally
separate from the trigger output so funding mechanism and amount parsing can be refreshed without
rerunning the full trigger classifier.

Key columns:

| Column | Type | Description |
|---|---|---|
| `project_id` | string | Primary key, matching a funding-primary project |
| `federal_funding_type_primary` | string | Main mechanism: grant/award, formula grant, loan guarantee, federal loan, cooperative agreement, cost share, financial assistance, generic funding, or unknown |
| `federal_funding_type_multi` | list<string> | All funding mechanisms detected in project-specific funding context |
| `federal_funding_program_multi` | list<string> | Program/source labels such as `ARRA`, `EECBG`, `SEP`, `WAP`, `Title XVII`, `BIL`, `IRA`, `FOA` |
| `federal_funding_amount_usd` | double | Evidence-backed federal amount, if exactly one non-conflicting amount is found |
| `federal_funding_total_project_cost_usd` | double | Evidence-backed total project cost/value, if found |
| `federal_funding_recipient_cost_share_usd` | double | Evidence-backed recipient/non-federal cost share, if found |
| `federal_funding_share_pct` | double | Explicit funding percentage or computed federal amount / total project cost |
| `federal_funding_evidence_text` | string | Funding-specific evidence snippet |
| `federal_funding_evidence_source` | string | `trigger_evidence`, `project_metadata`, `doc_title`, or `document_text` |
| `federal_funding_confidence` | string | `high`, `medium`, or `low` |
| `federal_funding_manual_review` | bool | True for unknown mechanisms or conflicting amount candidates |
| `federal_funding_amount_candidates_json` | string | All parsed amount/percent candidates and conflict flags |
| `federal_funding_extraction_run_at` | string | ISO-8601 UTC timestamp of sidecar extraction |

Amount coverage is partial by design. Dollar amounts are only populated when the amount is tied to
project-specific federal funding language. Generic dollar amounts elsewhere in a NEPA document are
not treated as federal funding amounts.

### Validation batches: `data/analysis/nepa_trigger/validation_batches.csv`

Stratified sample of output rows for manual QA. Sampled by:
- rule_id (up to 20 rows per rule family)
- process_type (up to 20 rows per CE/EA/EIS)
- DOE agency (up to 20 rows)
- CE dataset (up to 20 rows)

Key columns: all output columns + `validation_batch` (batch label), `batch_kind` (`rule`/`process`/`agency`/`dataset`), `batch_size` (total rows in that group).

### Tier 4 diagnostic artifacts (written on every run)

| File | Description |
|---|---|
| `data/analysis/nepa_trigger/context_candidates.parquet` | All chunks retrieved and scored per project |
| `data/analysis/nepa_trigger/tier4_chunk_scores.parquet` | Per-chunk NLI scores for each candidate class |
| `data/analysis/nepa_trigger/tier4_doc_scores.parquet` | Aggregated doc-level scores and auto-resolve decisions |
| `data/analysis/nepa_trigger/tier5_queue.parquet` | Projects queued for Tier 5 (written before any LLM calls) |

---

## Validation

### Automated assertions (run at write time)

The script enforces these before writing output; the run fails if any assertion fires:

1. **Uniqueness:** `project_id` is unique — exactly one row per project.
2. **Scope:** all `project_id` values are in the 20,725-project clean energy set.
3. **List column:** `nepa_trigger_secondary` is a Python list, not a JSON string.

Funding sidecar assertions:

1. **Funding-only scope:** sidecar project IDs exactly match `nepa_trigger_primary == "federal_funding"`.
2. **No trigger overwrite:** `--funding-details-only` asserts that `projects_nepa_trigger.parquet` size and mtime are unchanged.
3. **Numeric validity:** amount fields are non-negative USD values; percentages are between 0 and 100.
4. **List columns:** funding mechanism and program multi-label fields are Python lists.
5. **Conflict handling:** conflicting amount candidates leave primary amount fields blank and set `federal_funding_manual_review = TRUE`.

### Batch-by-rule review (manual)

After a successful run, review the highest-volume rule batches:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/analysis/nepa_trigger/validation_batches.csv')
rule_batches = df[df['batch_kind'] == 'rule'].drop_duplicates('validation_batch')
print(rule_batches[['validation_batch','batch_size']].sort_values('batch_size', ascending=False).head(15).to_string(index=False))
"
```

Accept a rule batch if manual precision ≥ 0.85. For rules below that threshold, correct patterns in the tier functions, re-run on a sample, and re-check.

### Quick distribution check

```python
import pyarrow.parquet as pq, pandas as pd

df = pq.read_table("data/analysis/nepa_trigger/projects_nepa_trigger.parquet").to_pandas()
print(df["nepa_trigger_primary"].value_counts())
print(f"\nManual review rate: {df['nepa_trigger_manual_review'].mean():.1%}  (target: < 5%)")
print(f"Unknown rate: {(df['nepa_trigger_primary']=='unknown').mean():.1%}  (target: < 10%)")
print(f"Dual-nexus projects: {df['is_dual_nexus'].sum()} ({df['is_dual_nexus'].mean():.1%})")
print(f"\nTier 4 diagnostic check:")
t4 = pq.read_table("data/analysis/nepa_trigger/tier4_doc_scores.parquet").to_pandas()
print(f"  Tier 4 auto-resolved: {t4['auto_resolve'].sum()} / {len(t4)}")
print(f"  Mean top class score: {t4['top_class_score'].mean():.3f}")
```

**Targets:**
- Manual review rate < 5%
- `unknown` share < 10%
- `federal_land` or `federal_permit` should be plurality for clean energy (expect ~40–60% combined)

---

## Analysis and figures

After the trigger parquet is produced, run the R analysis script:

```bash
Rscript phase2/code/deliverable01/02_create_figures.R
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
| `fig8_funding_mechanism_counts.png` | Federal funding mechanism counts from `projects_funding_details.parquet` |
| `fig9_funding_program_counts.png` | Federal funding program/source label counts |
| `fig10_funding_amount_coverage.png` | Coverage of extracted funding amount fields |
| `federal_funding_detail_summary.csv` | Mechanism, program/source, and amount coverage summary |
| `trigger_evidence_excerpts.csv` | 2 high-confidence quotable examples per class |
| `trigger_source_distribution.csv` | Evidence source × confidence breakdown |
| `trigger_rule_distribution.csv` | Top 25 rules by volume |

Fig 6 (trigger × review duration) is a placeholder — uncomment in the R script after `D4` timeline data (`data/analysis/projects_timeline_bert.parquet`) is available.

---

## Notes

- **`nepa_trigger_secondary` is a list-column in R.** `tidyr::unnest(df, nepa_trigger_secondary)` explodes it for frequency counts. `purrr::map2_chr()` handles it in Fig 3 (combination bar). Do not convert to a JSON string.
- **CE documents:** Tier 3 scans the full document with a conservative pattern set that excludes `sec404`, `arra`, and `rmp` to avoid form checklist false positives. EA/EIS scan pages 1–10 only.
- **DOE corpus dominance:** ~79% of projects are DOE-led. DOE Tier 1a results are always provisional (`T1a_DOE_action`, `T1a_DOE_funding`) and routed to Tier 4 for confirmation. DOE routing quality is the single biggest driver of overall output quality.
- **NLI model:** Tier 4 uses `cross-encoder/nli-MiniLM2-L6-H768` (~67MB, zero-shot). Downloads automatically on first run. If unavailable, falls back to `all-MiniLM-L6-v2` cosine similarity. Run `--calibrate` to verify hypothesis templates before any full corpus run. See `_notes.md` ("Model selection rationale") for the full explanation of why NLI was chosen over fine-tuned BERT.
- **Hypothesis tuning:** `HYPOTHESIS_TEMPLATES` in the script is the primary lever for Tier 4 accuracy. The `--calibrate` flag scores all example bank entries and reports PASS/FAIL so you know immediately if a hypothesis change is an improvement.
- **Document routing** uses the `dataset_source` column (`CE`, `EA`, `EIS`) from `projects_combined.parquet` to select the correct `*_documents.parquet` and `*_pages.parquet` paths.
- **`nepa_trigger_rule_id` format:** `T{tier}_{slug}` (e.g., `T1a_BLM_land`, `T1b_row_grant`, `T3_sec404`, `T4_local_federal_funding`, `T5_llm`). Use this to trace any classification back to the exact rule that fired.
- **Negation guard:** a regex negation filter suppresses matches in contexts like "Section 404 permit is NOT required" and CE checkbox forms where the box is unchecked (`[ ]`). This prevents the most common false-positive patterns in CE boilerplate.
- **Funding details are sidecar-only:** grant/loan/amount extraction only runs for projects already classified as `federal_funding`. This avoids turning generic dollar amounts or unrelated grant terms in land/permit/program documents into funding facts.
- **Grant disambiguation:** funding detail extraction suppresses land-authorization uses of "grant" such as right-of-way grants, ROW grants, easement grants, and perpetual ROW grants.
- **Amount extraction coverage:** `federal_funding_amount_usd` is intentionally sparse. It requires an amount near project-specific federal funding language such as DOE Funding, Federal Cost Share, award/grant language, loan guarantee language, EECBG/SEP funding, or a similar cue.
