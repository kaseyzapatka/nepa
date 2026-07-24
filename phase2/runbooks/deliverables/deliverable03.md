# D3 — NEPA Review Patterns: Fossil Fuel vs. Decarbonization

**Purpose:** Compare how fossil fuel and clean energy projects move through NEPA — CE/EA/EIS rates by technology, CE citation patterns, visual impact analysis, geothermal vs. oil/gas comparison, and geographic distribution.
**Scope:** Clean energy (20,725) + Fossil fuel (10,783) = **31,508 projects**. "Other" projects are excluded at the Python build stage. Technology taxonomy: `phase2/notes/project_types.txt`.
**Scripts:**
- `phase2/code/deliverable03/01_identify_visual_impact_candidates.py` — converts the shared `document_sections.parquet` layer into D03's visual-impact section candidates (fast; no page-reading or embeddings)
- `phase2/code/deliverable03/02_build_nepa_reviews.py` — builds all analysis parquets (base review table, CE citations, visual impacts, geothermal/OG subset)
- `phase2/code/deliverable03/03_inventory_visual_sections.py` — standalone provenance script; inventories visual/aesthetic section-heading variants and documents how the heading regexes in `01`/`02` were derived
- `phase2/code/deliverable03/04_create_figures.R` — produces all figures and tables

---

## Known data gaps

| Gap | Impact |
|-----|--------|
| `projects_nepa_trigger.parquet` (D1) covers clean energy only | `nepa_trigger_primary` is NULL for fossil projects; CE-by-trigger cross-tab is clean-energy only |
| `data/analysis/timeline.parquet` not yet built | Section 6 (timelines, fig20) silently skipped |
| "Other" projects excluded | Intentional — universe is Clean + Fossil only (31,508 projects) |

`is_linear` is no longer a gap: it is derived from the NEPATEC project-type taxonomy (transmission/pipeline/corridor labels = linear) when `projects_reviews.parquet` isn't present, or taken directly from that file when it is.

---

## Workflow

### Step 1 — Smoke test base table

```bash
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --reviews --sample 500
```

Check that `process_type` is populated for both Clean and Fossil, and that `tech_group` shows real technology names (Wind, Solar, Geothermal…) rather than everything landing in "Other Clean"/"Other Fossil":

```python
import duckdb
conn = duckdb.connect()  # run from nepa/ root

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
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --ce --sample 100
```

Top `ce_code` values should be short normalized codes (`B1.3`, `A9`, `10 CFR 1021`), not raw JSON strings.

### Step 3 — Build the section-layer pipeline (preferred; ~few minutes)

The section-layer path replaces the old sentence-transformers-only visual extractor as the primary route to `visual_sections.parquet`, `visual_framing.parquet`, `visual_topics.parquet`, etc. It requires `projects_nepa_reviews.parquet` (Step 1) and the shared `document_sections.parquet` layer to already exist.

```bash
conda run -n nepa python phase2/code/deliverable03/01_identify_visual_impact_candidates.py
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --section-layer
```

`01` writes `visual_impact_sections_from_document_sections.parquet` and `projects_visual_text_from_document_sections.parquet`. `02 --section-layer` adapts those into `visual_sections.parquet` / `projects_visual_text.parquet`, then runs framing, topic modeling (NMF), VRM-element extraction, examples, and QA sampling.

### Step 4 — Smoke test legacy visual extraction (optional, ~2 min for a sample)

```bash
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --visual --sample 20
```

`--visual` runs the older sentence-transformers similarity search (`projects_visual_impacts.parquet`), preserved for fig12–14 calibration and as an input to `03`'s coverage check. Check the log for `schema key = project_id` or `document_id`. Inspect `visual_impacts_text` values — retrieved passages should be about visual resources, not boilerplate.

### Step 5 — Full run (no flags, ~20–30 min)

```bash
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py
```

Runs `--reviews`, `--ce`, `--visual` (legacy + new section-extraction stages: framing, topics, examples, QA), and `--geothermal` in sequence. This does **not** run the `--section-layer` path — run Step 3 separately for the preferred (faster) visual pipeline.

### Step 6 — Visual section heading inventory (optional, provenance)

```bash
conda run -n nepa python phase2/code/deliverable03/03_inventory_visual_sections.py
```

Requires `projects_nepa_reviews.parquet`; uses `projects_visual_impacts.parquet` if present to enrich coverage stats. Documents the visual/aesthetic heading-name variants found in EA/EIS text and the calibration behind the heading regexes reused by `01` and `02`. Not part of the report's render closure — run when auditing or extending the heading patterns.

### Step 7 — Figures and tables

```bash
Rscript phase2/code/deliverable03/04_create_figures.R
```

Outputs written to `phase2/output/deliverable03/`. Section 4 (Visual Impacts) figures are each independently guarded — a missing parquet only skips that figure. Section 6 (timelines) is skipped entirely if `data/analysis/timeline.parquet` doesn't exist.

---

## CLI reference — `02_build_nepa_reviews.py`

| Flag | Output parquet | Time | Notes |
|------|---------------|------|-------|
| `--reviews` | `projects_nepa_reviews.parquet` | ~10 sec | Base table; run this first |
| `--ce` | `ce_citations.parquet` | ~30 sec | One row per (project, CE citation) |
| `--visual` | `projects_visual_impacts.parquet` + `visual_sections.parquet`/`visual_framing.parquet`/`visual_topics.parquet`/`visual_examples.parquet`/`visual_qa_sample.parquet` | ~20 min | Legacy sentence-transformers extractor, then the new section-extraction stages (framing, topics, examples, QA) |
| `--section-layer` | `visual_sections.parquet`, `projects_visual_text.parquet`, `visual_framing.parquet`, `visual_topics.parquet`, `vrm_elements.parquet`, `visual_examples.parquet`, `visual_qa_sample.parquet` | ~few min | Preferred pipeline. Adapts `01_identify_visual_impact_candidates.py` output — no page I/O, no embeddings re-run |
| `--geothermal` | `projects_geothermal_og.parquet` | ~5 sec | Requires `--reviews` to have run |
| *(no flags)* | reviews + ce + visual (legacy + new stages) + geothermal | ~20–30 min | Runs modules in sequence; does not run `--section-layer` |
| `--sample N` | — | — | Limit to N random projects; combine with any flag |

---

## Outputs

### `data/analysis/deliverable03/projects_nepa_reviews.parquet`

One row per project. Key columns:

| Column | Description |
|--------|-------------|
| `project_id` | Primary key |
| `project_energy_type` | `Clean`, `Fossil`, or `Other` |
| `energy_group` | `Decarbonization` or `Fossil Fuel` (pre-computed in Python — use this in R, do not re-derive) |
| `tech_group` | `Geothermal`, `Wind`, `Solar`, `Transmission`, `Hydropower`, `Biomass`, `Energy Storage`, `CCS`, `Nuclear`, `Utilities`, `Other Renewable`, `Other Conventional`, `Other Clean` (Clean); `Land-based Oil & Gas`, `Offshore Oil & Gas`, `Coal`, `Pipeline`, `Rural Energy`, `Other Fossil` (Fossil) |
| `process_type` | `CE`, `EA`, or `EIS` |
| `is_linear` | Boolean — from `projects_reviews.parquet` if present, else derived from `project_type` (transmission/pipeline/corridor = linear) |
| `nepa_trigger_primary` | Trigger class from D1 — NULL for fossil projects |
| `lead_agency_harmonized` | JSON array; use `parse_json_first()` in R |
| `project_state`, `project_county` | JSON arrays; explode with `explode_column()` in R |

### `data/analysis/deliverable03/ce_citations.parquet`

One row per citation. Key columns: `project_id`, `ce_raw` (original string), `ce_code` (normalized), `ce_description`.

### `data/analysis/deliverable03/projects_visual_impacts.parquet`

Legacy sentence-transformers output. One row per project (deduplicated across EA/EIS sources; best similarity kept). Key columns: `project_id`, `source`, `visual_impacts_max_similarity`, `visual_impacts_text` (list of top chunks), `visual_section_found`, `visual_mention_count`.

### `data/analysis/deliverable03/visual_sections.parquet` / `projects_visual_text.parquet`

Section-layer pipeline output (from `--section-layer`, built on `01`'s heading-anchored section detection). One row per detected visual/aesthetic section (`visual_sections.parquet`) or per project with concatenated visual text (`projects_visual_text.parquet`). Feeds `visual_framing.parquet`, `visual_topics.parquet`, `vrm_elements.parquet`, `visual_examples.parquet`.

### `data/analysis/deliverable03/projects_geothermal_og.parquet`

Subset of `projects_nepa_reviews.parquet`: Clean geothermal projects (any agency) plus all oil & gas projects (`tech_group` in `Land-based Oil & Gas`, `Offshore Oil & Gas`; any agency). Not restricted to BLM — BLM is simply the dominant lead agency for onshore oil & gas and geothermal, so most rows happen to be BLM, but the filter itself is agency-agnostic.

---

## Figure list (`04_create_figures.R`)

| Figure | Section | Description |
|--------|---------|-------------|
| `fig1_review_rates_by_energy.png` | Review Rates | CE/EA/EIS: Clean vs. Fossil |
| `fig1b_within_agency.png` | Review Rates | Within-agency (BLM/DOE) review type comparison |
| `fig2_review_rates_by_tech.png` | Review Rates | CE/EA/EIS by technology (sorted by CE share) |
| `fig3_review_rates_linear.png` | Review Rates | Linear vs. non-linear × energy group |
| `fig4_top_ce_codes.png` | CE Citations | Top 15 CE codes overall |
| `fig5_ce_by_energy.png` | CE Citations | Top CE codes by energy type |
| `fig6_ce_by_agency.png` | CE Citations | CE heatmap by agency |
| `fig7_state_decarb.png` | Geography | State choropleth — Decarbonization |
| `fig8_state_fossil.png` | Geography | State choropleth — Fossil Fuel |
| `fig9_county_decarb.png` | Geography | County choropleth — Decarbonization |
| `fig10_county_fossil.png` | Geography | County choropleth — Fossil Fuel |
| `fig11a_state_process_decarb.png` | Geography | State × process type facet — Decarbonization |
| `fig11b_state_process_fossil.png` | Geography | State × process type facet — Fossil Fuel |
| `fig12_visual_project_counts.png` | Visual Impacts | Visual-analysis universe: project counts by tech_group × energy_group |
| `fig13_wordcloud_grid.png` | Visual Impacts | Word cloud 2×2 grid: EA/EIS × Decarb/Fossil (TF-IDF top terms) |
| `fig14_topic_prevalence.png` | Visual Impacts | Topic prevalence by group (top NMF topics) |
| `fig14b_topic_terms.png` | Visual Impacts | Topic term weights (companion to fig14) |
| `fig14d_nmf_elbow.png` | Visual Impacts | NMF elbow / k-selection validation |
| `fig18_visual_framing.png` | Visual Impacts | Framing comparison (CEQ-axis ratios by energy × process) |
| `fig19_visual_section_length.png` | Visual Impacts | Section length boxplots by tech_group × process_type |
| `fig19a_section_length_energy.png` | Visual Impacts | Section length boxplot collapsed to energy category |
| `fig21_vrm_elements.png` | Visual Impacts | BLM VRM element-level contrast rating distribution |
| `fig15_geo_og_rates.png` | Geothermal/OG | CE/EA/EIS rates: Geothermal vs. Oil & Gas |
| `fig16_geo_og_states.png` | Geothermal/OG | Geothermal vs. Oil & Gas share by state (100% stacked bar) |
| `fig17_geo_og_state_map.png` | Geothermal/OG | State choropleth — geothermal share (diverging) |
| `fig20_duration_by_energy_process.png` | Timelines | Duration by period × process × energy *(conditional on `timeline.parquet`)* |

---

## Reproduction steps

Run from the `nepa/` root in the `nepa` conda environment.

```bash
# 1. Build base review + CE tables
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --reviews --ce --geothermal

# 2. Build the section-layer visual pipeline (preferred)
conda run -n nepa python phase2/code/deliverable03/01_identify_visual_impact_candidates.py
conda run -n nepa python phase2/code/deliverable03/02_build_nepa_reviews.py --section-layer

# 3. Produce all figures and tables
Rscript phase2/code/deliverable03/04_create_figures.R
```

Outputs: `phase2/data/analysis/deliverable03/` (parquets) and `phase2/output/deliverable03/` (figures + CSVs).
