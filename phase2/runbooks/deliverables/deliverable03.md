# D3 — NEPA Review Patterns: Fossil Fuel vs. Decarbonization

**Purpose:** Compare how fossil fuel and clean energy projects move through NEPA — CE/EA/EIS rates by technology, CE citation patterns, visual impact analysis, geothermal vs. oil/gas comparison, and geographic distribution.
**Scope:** Clean energy (20,725) + Fossil fuel (10,783) = **31,508 projects**. "Other" projects are excluded at the Python build stage. Technology taxonomy: `phase2/notes/project_types.txt`.
**Scripts:**
- `phase2/code/deliverable03/01_build_nepa_reviews.py` — builds all analysis parquets
- `phase2/code/deliverable03/02_analyze_nepa_reviews.R` — produces all figures and tables

---

## Known data gaps

| Gap | Impact |
|-----|--------|
| `projects_nepa_trigger.parquet` covers clean energy only | `nepa_trigger_primary` is NULL for fossil projects; CE-by-trigger cross-tab is clean-energy only |
| `is_linear` not yet derived | Fig 3 (linear vs. non-linear) will be empty until a geometry field exists |
| `data/analysis/timeline.parquet` not yet built | Section 6 (timelines, Fig 17) silently skipped |
| "Other" projects excluded | Intentional — universe is Clean + Fossil only (31,508 projects) |

---

## Workflow

### Step 1 — Smoke test base table

```bash
conda run -n nepa python phase2/code/deliverable03/01_build_nepa_reviews.py --reviews --sample 500
```

Check that `process_type` is populated for both Clean and Fossil, and that `tech_group` shows real technology names (Wind, Solar, Geothermal…) rather than everything landing in "Other":

```python
import duckdb
conn = duckdb.connect("phase2")  # run from nepa/ root

conn.execute("""
    SELECT project_energy_type, process_type, count(*) as n
    FROM read_parquet('phase2/data/analysis/deliverable03/projects_nepa_reviews.parquet')
    GROUP BY 1, 2 ORDER BY 1, 2
""").fetchdf()

conn.execute("""
    SELECT tech_group, project_energy_type, count(*) as n
    FROM read_parquet('phase2/data/analysis/deliverable03/projects_nepa_reviews.parquet')
    GROUP BY 1, 2 ORDER BY 3 DESC
""").fetchdf()
```

### Step 2 — Smoke test CE citations

```bash
conda run -n nepa python phase2/code/deliverable03/01_build_nepa_reviews.py --ce --sample 100
```

Top `ce_code` values should be short normalized codes (`B1.3`, `A9`, `10 CFR 1021`), not raw JSON strings.

### Step 3 — Smoke test visual extraction (~2 min)

```bash
conda run -n nepa python phase2/code/deliverable03/01_build_nepa_reviews.py --visual --sample 20
```

Check the log for `schema key = project_id` or `document_id`. Inspect `visual_impacts_text` values — retrieved passages should be about visual resources, not boilerplate.

### Step 4 — Full run (~20–30 min)

```bash
conda run -n nepa python phase2/code/deliverable03/01_build_nepa_reviews.py
```

### Step 5 — Figures and tables

```bash
Rscript phase2/code/deliverable03/02_analyze_nepa_reviews.R
```

Outputs written to `phase2/output/deliverable03/`. Sections 2, 4, and 5 are skipped if the corresponding parquet is missing; Section 6 (timelines) is skipped if `data/analysis/timeline.parquet` doesn't exist.

---

## CLI reference — `01_build_nepa_reviews.py`

| Flag | Output parquet | Time | Notes |
|------|---------------|------|-------|
| `--reviews` | `projects_nepa_reviews.parquet` | ~10 sec | Base table; run this first |
| `--ce` | `ce_citations.parquet` | ~30 sec | One row per (project, CE citation) |
| `--visual` | `projects_visual_impacts.parquet` | ~20 min | Sentence-transformers embedding |
| `--geothermal` | `projects_geothermal_og.parquet` | ~5 sec | Requires `--reviews` to have run |
| *(no flags)* | all four | ~20–30 min | Runs modules in sequence |
| `--sample N` | — | — | Limit to N random projects; combine with any flag |

---

## Outputs

### `data/analysis/deliverable03/projects_nepa_reviews.parquet`

One row per project. Key columns:

| Column | Description |
|--------|-------------|
| `project_id` | Primary key |
| `project_energy_type` | `Clean`, `Fossil`, or `Other` |
| `energy_group` | `Decarbonization`, `Fossil Fuel`, or `Other` |
| `tech_group` | `Wind`, `Solar`, `Transmission`, `Geothermal`, `Natural Gas`, `Oil & Gas`, `Other Clean`, `Other Fossil`, `Other` |
| `process_type` | `CE`, `EA`, or `EIS` |
| `is_linear` | Boolean — NULL until geometry field is derived |
| `nepa_trigger_primary` | Trigger class — NULL for fossil projects |
| `lead_agency_harmonized` | JSON array; use `parse_json_first()` in R |
| `project_state`, `project_county` | JSON arrays; explode with `explode_column()` in R |
| `energy_group` | Pre-computed in Python — use this in R, do not re-derive |

### `data/analysis/deliverable03/ce_citations.parquet`

One row per citation. Key columns: `project_id`, `ce_raw` (original string), `ce_code` (normalized), `ce_description`.

### `data/analysis/deliverable03/projects_visual_impacts.parquet`

One row per project (deduplicated across EA/EIS sources; best similarity kept). Key columns: `project_id`, `source`, `visual_impacts_max_similarity`, `visual_impacts_text` (list of top chunks), `visual_section_found`, `visual_mention_count`.

### `data/analysis/deliverable03/projects_geothermal_og.parquet`

Subset of `projects_nepa_reviews.parquet` filtered to BLM projects where `tech_group` ∈ {Geothermal, Oil & Gas, Natural Gas}.

---

## Figure list (`02_analyze_nepa_reviews.R`)

| Figure | Section | Description |
|--------|---------|-------------|
| `fig1_review_rates_by_energy.png` | Review Rates | CE/EA/EIS: Clean vs. Fossil |
| `fig2_review_rates_by_tech.png` | Review Rates | CE/EA/EIS by technology (sorted by CE share) |
| `fig3_review_rates_linear.png` | Review Rates | Linear vs. Non-linear × energy group |
| `fig4_top_ce_codes.png` | CE Citations | Top 15 CE codes overall |
| `fig5_ce_by_energy.png` | CE Citations | Top CE codes by energy type |
| `fig6_ce_by_agency.png` | CE Citations | CE heatmap by agency |
| `fig7_state_decarb.png` | Geography | State choropleth — Decarbonization |
| `fig8_state_fossil.png` | Geography | State choropleth — Fossil Fuel |
| `fig9_county_decarb.png` | Geography | County choropleth — Decarbonization |
| `fig10_county_fossil.png` | Geography | County choropleth — Fossil Fuel |
| `fig11_state_process_facet.png` | Geography | State × process type × energy group |
| `fig12_visual_prevalence_by_tech.png` | Visual Impacts | Prevalence by technology and process type |
| `fig13_visual_similarity_dist.png` | Visual Impacts | Similarity score distribution (boxplot) |
| `fig14_visual_section_detection.png` | Visual Impacts | Section detection rate by technology |
| `fig15_geo_og_rates.png` | Geothermal/OG | CE/EA/EIS rates (BLM only) |
| `fig16_geo_og_states.png` | Geothermal/OG | Geographic overlap — top states |
| `fig17_duration_by_energy_process.png` | Timelines | Duration by period × process × energy *(conditional)* |

---

## Reproduction steps

Run from the `nepa/` root in the `nepa` conda environment.

```bash
# 1. Build all analysis parquets
conda run -n nepa python phase2/code/deliverable03/01_build_nepa_reviews.py

# 2. Produce all figures and tables
Rscript phase2/code/deliverable03/02_analyze_nepa_reviews.R
```

Outputs: `phase2/data/analysis/deliverable03/` (parquets) and `phase2/output/deliverable03/` (figures + CSVs).
