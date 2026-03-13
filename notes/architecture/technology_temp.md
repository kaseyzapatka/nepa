# Technology Universe Identification: Transmission, Geothermal, Pipeline

Reviewed 2026-03-12. Based on `code/extract/extract_technology.py` and
`code/deliverable06/01_transmission.R`, `02_geothermal.R`, `03_pipelines.R`.

---

## Transmission

### How identified
Three-tier strict gate — all four conditions must hold:

1. **Type tag**: `project_type` contains `"Electricity Transmission"` →
   `project_has_transmission_type_tag`
2. **Build text**: title+description matches `TRANSMISSION_BUILD_RE` (phrases like
   "new transmission line", "construct X kV line", "double-circuit line",
   ROW+transmission co-occurrence) → `project_has_transmission_build_text`
3. **Length threshold**: extracted length ≥ 1 mile (after LLM adjudication if triggered)
4. **NOT** flagged as maintenance (`project_is_transmission_maintenance`) — checked
   against **title only** to avoid excluding projects with incidental maintenance language
   in the description

`project_is_transmission` (alias of `project_is_transmission_strict`) = all four conditions.

A `project_is_transmission_broad` pre-filter (any mention of "transmission") gates the
expensive length extraction; non-broad rows skip candidate extraction entirely.

R analysis (`01_transmission.R`) filters to `project_is_transmission == TRUE` + clean
energy (`project_energy_type == "Clean"` via `prepare_deliverable6_data()`).

### What could be missed
- Projects where the transmission length appears only in document body text, not
  title/description. Page-level recovery (`--page-length-recovery` flag) recovers some.
  There is a known population of ~1,268 projects with tag+build text but no extractable
  length; many are ROW renewals that genuinely have no construction length even in pages.
- Projects using non-standard phrasing (e.g., "kV line" without "transmission") that
  doesn't hit `TRANSMISSION_BUILD_RE`.

### Extra filters verdict
**Keep all of them.** Without build-text filter and ≥ 1-mile threshold you'd sweep in
thousands of false positives (solar interconnects, substation-only permits, utility
coordination). The type tag alone is insufficient. The maintenance exclusion is necessary.

---

## Geothermal

### How identified
Keyword match on **`project_type` field only** (analogous to transmission's type-tag gate):

```
\b(geothermal|enhanced geothermal|egs)\b
```

Applied to `project_type` only (not full_text). `project_is_geothermal = TRUE` whenever
this matches. NEPATEC does carry geothermal-specific tags within `project_type`, making
a type-tag gate reliable. Earlier documentation incorrectly stated no such tag existed.

R analysis (`02_geothermal.R`) filters to `project_is_geothermal == TRUE` + clean energy.

Phase classification (`project_geothermal_phase`) is derived separately via
`_classify_geothermal_phase()` using regex patterns applied to **`full_text`** (title +
description + type + NOI title + aggregated document titles). This step is unchanged —
only the identification flag uses the narrower type_text field.
Roughly half of geothermal projects receive `phase = "unknown"` (geothermal keyword
present but no phase signal detected). Phase classification does not scan document pages.

### What could be missed
Projects where the geothermal project_type tag is missing but the project clearly involves
geothermal development (e.g., tagged only as "Renewable Energy Production"). This is the
trade-off of moving to a type-tag gate — tighter precision, potentially lower recall.

### Extra filters verdict
Nothing to remove — identification is a single type-tag match. No additional filters
(build text, length threshold, maintenance exclusion) exist for geothermal.

### Phase classification — two-stage pipeline

**Stage 1: Regex (always runs).** `_classify_geothermal_phase()` applies
`GEOTHERMAL_PHASE_PATTERNS` to `full_text`. Returns one of:
`exploration | drilling | plant | operations | multi_phase | unknown | none`.
Rows with no pattern match receive `unknown`; rows with no geothermal keyword at all
receive `none`.

**Stage 2: ML classifier (optional, run separately).** A fine-tuned DistilBERT
(`distilbert-base-uncased`) re-classifies the `unknown` rows. Trained on the regex-labeled
rows (exploration / drilling / plant / operations / multi_phase) using title + project_type +
first 100 words of description as input text. Labels: same five canonical phases
(`multi_phase` treated as a valid target class).

Run order:
```
python code/extract/extract_technology.py --run geothermal          # Stage 1
python code/extract/extract_technology.py --geothermal-phase-train  # train on labeled rows
python code/extract/extract_technology.py --geothermal-phase-classify  # update unknowns
```

Model saved to `data/models/geothermal_phase_classifier/`. Three columns added/updated
by the classify step:
- `project_geothermal_phase` — updated from `"unknown"` to predicted label
- `project_geothermal_phase_ml_confidence` — softmax score for the predicted label
- `project_geothermal_phase_ml_classified` — `True` for ML-predicted rows (audit flag)

---

## Pipeline

### How identified
Keyword match — **intentionally ignores the NEPATEC "Pipelines" tag**, which is
classified as a *fossil fuel* tag in the clean energy classification framework:

```
\bpipelines?\b|\bflowlines?\b|\bgathering lines?\b
```

Applied to `full_text` (same fields as above). Sub-types detected by keyword
co-occurrence (pipeline flag + additional keywords):

| Flag | Pattern |
|------|---------|
| `project_is_carbon_pipeline` | `\b(carbon\|co2\|carbon dioxide\|ccs\|carbon capture\|carbon sequestration)\b` |
| `project_is_hydrogen_pipeline` | `\bhydrogen\b` |
| `project_is_natural_gas_pipeline` | `\bnatural gas\b\|\bgas pipeline\b\|\bgas gathering\b\|\bgas line\b` |

In R, `add_pipeline_group()` adds two further categories from `project_type_txt`:
- **Oil/petroleum** (oil & gas / petroleum type tag)
- **Water/irrigation** (water resources / irrigation type tag)

Priority cascade: carbon > hydrogen > natural gas > oil/petroleum > water/irrigation > other.

### Clean energy filter — intentionally NOT applied
The pipeline analysis does NOT apply the clean energy filter:
- `analysis_all` (counts, geography, lengths): reads `projects_combined_path` directly,
  filters to `project_is_pipeline == TRUE` only — all energy types included.
- `analysis_timeline` (duration): `prepare_deliverable6_data(clean_only = FALSE)` — no
  clean energy filter.

This is intentional: the deliverable asks to compare carbon/hydrogen pipelines against
natural gas, which is fossil fuel. Applying the clean energy filter would remove the
natural gas baseline.

**Known documentation inconsistency**: `notes/architecture/deliverable06.md` (section 2)
and a comment in `03_pipelines.R` both claim the pipeline analysis is restricted to
"decarbonization technology" projects. The code is correct; those comments are wrong.

### What could be missed
Carbon/hydrogen pipelines described as "CO₂ injection" infrastructure or "hydrogen fuel
cell supply" without the word "pipeline"/"flowline" in title/description. Likely a small
gap since "pipeline" is nearly universal in NEPA documents for pipeline projects.

Sample sizes are very small (carbon ≈ 8, hydrogen ≈ 4). This is believed to reflect
genuine scarcity in the database, not a coverage gap.

### Extra filters verdict
**Keep the broadening** (flowlines, gathering lines — added beyond original `\bpipelines?\b`).
These are common synonyms in CCS, gas gathering, and hydrogen conveyance projects.
**Keep `clean_only = FALSE`** for pipeline analyses — removing it would eliminate the
natural gas baseline comparison.

---

## Summary Comparison

| | Transmission | Geothermal | Pipeline |
|---|---|---|---|
| Uses NEPATEC type tag | Yes (required) | **Yes** (project_type field) | No (tag = fossil fuel) |
| Text keyword filter | Strict build-text regex | Tag keyword only | Simple keyword (broadened) |
| Length threshold | Yes (≥ 1 mile) | No | No |
| Maintenance exclusion | Yes (title-only) | No | No |
| Clean energy filter | Yes | Yes | **No** |
| Page-level length recovery | Yes | No | No |
