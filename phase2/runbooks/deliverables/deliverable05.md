# D5 — CE Spikes After Major Infrastructure Legislation

**Purpose:** Test whether the use of categorical exclusions (CEs) spikes after major infrastructure legislation (ARRA 2009, BIL/IIJA 2021, IRA 2022), whether the spiking actions are *associated with* the law (explicit citations in the documents), and *what types* of CEs were used (the categorical-exclusion category invoked).
**Scope:** All CEs (54,040 in the D4 timeline; ~96.4% placeable by a date), all energy types, all departments. EA/EIS are scanned for citations too, for the by-review-type contrast.
**Input:** `phase2/data/analysis/timeline/timeline_project_dates.parquet` (D4 dates), `phase1/data/analysis/projects_combined.parquet` (energy/department/agency/type), `phase2/data/processed/{ce,ea,eis}/{pages,documents}.parquet`.
**Output:** `phase2/data/analysis/deliverable05/{law_citations,ce_categories}.parquet`; figures + diagnostic CSVs in `phase2/output/deliverable05/{figures,diagnostics}/`.
**Cost:** $0 — no LLM/API. Pure regex over page text (~3–4 min full scan), metadata parse, and R analysis.
**Conda env:** `nepa` — Python via `/opt/anaconda3/envs/nepa/bin/python` (scripts 01–02 hard-require `CONDA_DEFAULT_ENV=nepa`). Script 03 is `Rscript`.

**Scripts** (in `phase2/code/deliverable05/`):
- `01_extract_law_citations.py` — scan CE/EA/EIS pages for ARRA/BIL/IRA citations (+ DOE funding-program signals)
- `02_build_ce_categories.py` — parse the document-level `ce_category` metadata into normalized CE codes
- `03_analyze_spikes.R` — join dates + citations + categories; produce all figures and diagnostic tables

> **Note:** there is no FRA / document-length analysis in D5 — that lives in D4 (`phase2/code/deliverable04/fra/`). D5 is spikes + attribution + CE-type only.

---

## Pipeline stages

| Stage | Script | Input → Output |
|---|---|---|
| Categories | `02_build_ce_categories.py` | `ce/documents.parquet` → `ce_categories.parquet` (project × normalized code) |
| Citations | `01_extract_law_citations.py` | `{ce,ea,eis}/{pages,documents}` + D4 timeline → `law_citations.parquet` (project × law) |
| Analysis | `03_analyze_spikes.R` | timeline + projects + 01 + 02 → figures + `d5_*.csv` |

Scripts 01 and 02 are independent; 03 depends on both **and** on the D4 timeline. Run order is 02 → 01 → 03 (02 is instant; 01 is the slow page scan).

---

## Workflow

### Step 1 — Build CE categories (fast, ~10 sec)

```bash
conda activate nepa
python phase2/code/deliverable05/02_build_ce_categories.py
```

Top `code_norm` should be real codes — `516 DM 11.9`, `B3.6`, `A9`, `B1.3`, `B5.1` — not raw strings. The script prints the top-15 codes and the schedule mix (DOE / DOI / EPAct).

### Step 2 — Smoke-test citation detection

```bash
python phase2/code/deliverable05/01_extract_law_citations.py --source ce --sample 400
```

The script prints, per law, the number of projects and 5 sample context snippets. **Eyeball the contexts**: ARRA snippets should reference the Recovery Act / ARRA (not "Resource Conservation and Recovery Act"), and any `IRA`/`BIL` acronym hits should sit near energy/infrastructure language. (A small ARRA sample may miss the 2010 DOE cluster — that's expected; the full run finds it.)

### Step 3 — Full citation scan (~3–4 min)

```bash
python phase2/code/deliverable05/01_extract_law_citations.py --source all
```

### Step 4 — Analysis, figures, and tables (~30 sec)

```bash
Rscript phase2/code/deliverable05/03_analyze_spikes.R
```

Outputs to `phase2/output/deliverable05/{figures,diagnostics}/`. Sanity checks: `fig_d5_ce_counts_by_year_doe_blm.png` should show the DOE 2010 spike with BLM flat; `d5_citation_rates.csv` should show ~59% ARRA-window CE citation; `d5_category_shift.csv` B5.1 ~49% window vs ~1% baseline.

---

## CLI reference

| Script | Flag | Notes |
|---|---|---|
| `01_extract_law_citations.py` | `--source {ce,ea,eis,all}` | Which corpus to scan (default `all`) |
| | `--sample N` | Limit to N random timeline projects per source (smoke test) |
| `02_build_ce_categories.py` | *(none)* | Reads `ce/documents.parquet`, writes `ce_categories.parquet` |
| `03_analyze_spikes.R` | *(none)* | Reads all inputs, writes figures + CSVs |

---

## Outputs

### `data/analysis/deliverable05/law_citations.parquet`

One row per (project, law). Key columns:

| Column | Notes |
|---|---|
| `project_id`, `process_type` | keys (`CE`/`EA`/`EIS`) |
| `law_name` | `ARRA`, `BIL`, `IRA`, or `DOE_funding` |
| `citation_count`, `n_docs_matched` | total matches / distinct documents matched |
| `first_match_type` | `full_name`, `acronym`, `program`, … |
| `first_context` | ±100-char snippet around the first match |
| `first_document_id`, `first_page_number` | provenance |
| `law_citations_extraction_run_at` | ISO-8601 UTC audit timestamp |

### `data/analysis/deliverable05/ce_categories.parquet`

One row per (project, normalized code). Key columns: `project_id`, `code_raw` (original string), `code_norm` (e.g. `B5.1`, `516 DM 11.9`, `EPAct §390`), `schedule` (`DOE (10 CFR 1021)` / `DOI (516 DM 11)` / `EPAct 2005 §390`), `code_description` (curated for high-frequency codes; falls back to the code), `ce_categories_extraction_run_at`.

### Figures (`output/deliverable05/figures/`)

| Figure | Track | Description |
|---|---|---|
| `fig_d5_ce_counts_by_year_all.png` | Spike | CEs by year, all (markers + N) |
| `fig_d5_ce_counts_by_year_byenergy.png` | Spike | CEs by year, by energy type |
| `fig_d5_counts_by_year_byprocess.png` | Spike | CE/EA/EIS by year (spike is CE-only) |
| `fig_d5_counts_by_year_byprocess_byenergy.png` | Spike | by process × energy |
| `fig_d5_ce_counts_by_year_bydept.png` | Spike | by lead department |
| `fig_d5_ce_counts_by_year_doe_blm.png` | Spike | **DOE vs BLM — headline finding** |
| `fig_d5_citations_by_year_byprocess.png` | Attribution | law-citing reviews by year (line) |
| `fig_d5_citation_rate_window_vs_baseline.png` | Attribution | citation rate, window vs baseline |
| `fig_d5_ce_category_shift_arra.png` | CE type | **B5.1 surge — ARRA category mix** |
| `fig_d5_technology_shift_arra.png` | CE type | ARRA-window technology mix |

### Diagnostic tables (`output/deliverable05/diagnostics/`)

`d5_spike_summary.csv`, `d5_citation_rates.csv`, `d5_category_shift.csv`, `d5_technology_shift.csv`, `d5_counts_by_year.csv`, `d5_counts_by_year_department.csv`, `d5_date_coverage_by_year.csv`. Each carries `scope`/`process` columns where applicable so the full energy × review-type cross is one tidy file per analysis.

---

## Reproduction steps

Run from the `nepa/` root in the `nepa` conda environment.

```bash
# 1. CE category metadata (fast, no page scan)
python phase2/code/deliverable05/02_build_ce_categories.py

# 2. Law-citation scan — all corpora (~3-4 min)
python phase2/code/deliverable05/01_extract_law_citations.py --source all

# 3. Analysis, figures, diagnostic tables
Rscript phase2/code/deliverable05/03_analyze_spikes.R
```

The report (`phase2/reports/deliverable05.qmd`) reads the figures + CSVs and recomputes inline values; render it with `quarto render phase2/reports/deliverable05.qmd`.

---

## Notes

- **Temporal anchor = D4 `decision_date`, with `initiation_date` fallback** for year placement → ~96.4% of CEs placeable (vs the 55% complete-timeline base D4 needs for *duration*). Safe for CEs because the median CE duration is ~20 days, so initiation and decision fall in the same year. A `date_basis` flag records which was used.
- **`project_id` is a STRUCT in the processed `documents.parquet` files** (`STRUCT("value" VARCHAR)`) — extract `project_id.value` in DuckDB. `document_id` is plain VARCHAR; the D4 timeline and `projects_combined` project_ids are plain VARCHAR.
- **ARRA short-name guard:** bare "Recovery Act" collides with the *Resource Conservation and Recovery Act* (RCRA, 1976) and other "…Recovery Act"s, so it is kept only with an affirming ARRA context (`reinvestment|stimulus|2009|111-5|ARRA`) and rejected near `conservation`. The `IRA`/`BIL` acronyms require energy/infrastructure context within ±200 chars.
- **Coverage-ramp confound:** raw CE counts mix real surges with NEPATEC's ingestion ramp (sparse pre-2009) and 2024–25 recency lag. The causal claim rests on **DOE-vs-BLM conditioning** (the ARRA spike is DOE-only; BLM, on the same dataset, is flat) and **citation evidence** (a Recovery Act citation cannot predate the law) — not on aggregate counts. ARRA has no usable pre-law baseline, so report no ARRA window/baseline ratio.
- **`ce_category` is a NEPATEC source field**, captured at ingestion by `extract_data.py` — the *code* is authoritative; the plain-English *descriptions* in script 02 are a hand-curated lookup (10 CFR 1021) covering only high-frequency codes.
- **Always activate `nepa`** — scripts 01–02 exit immediately if `CONDA_DEFAULT_ENV != nepa`.
