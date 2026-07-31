# D2 — Determinations of Significance Across Resource Areas

**Purpose:** Characterize, per resource area, how agencies (BLM + the DOE agency family for FONSIs; all agencies for EIS) make the NEPA significance determination — CEQ context/intensity factors, resource thresholds, and mitigation dependence. Two tracks: **FONSI** (below-the-line findings) and **EIS** (above-the-line, significant findings).
**Input:** `data/analysis/projects_combined.parquet`, D6 FONSI finding-section spans (`deliverable06/fonsi_*`), and processed EA/EIS pages/sections. A hand-labeled gold set is required for validation (`05`).
**Output:** `data/analysis/deliverable02/significance_determinations.parquet` (FONSI) and `significance_determinations_eis.parquet` (EIS), each with a `determination_thresholds*.parquet` child table, plus validation metrics and figures.
**Cost:** Billable LLM (Anthropic Batch API). Measured: **FONSI ≈ $15** (Sonnet 5, 3,478 windows) · **EIS ≈ $110–115** (Sonnet 5, 21,854 windows). Dry-run and Stage-0 are $0. **The user launches all billable passes.**
**Scope:** FONSI headline base = 193 clean-energy EA→FONSI projects (from 452 corpus / 427 BLM+DOE). EIS = 506 analyzed projects (all agencies).
**Conda env:** `nepa` — run all Python inside the `nepa` conda environment.

**Scripts** (in `phase2/code/deliverable02/`):
- `00_resolve_framework_regime.py` → `01_build_d2_inventory.py` — deterministic corpus + regime + cohorts (chained by `_run.py`)
- `03_build_gold_set_queue.py` / `03_build_gold_set_queue_eis.py` — stratified labeling worksheets
- `02_extract_fonsi_significance.py` — FONSI extraction (dry-run / sync / batch)
- `04_extract_eis_significance.py` — EIS extraction (gated; `_eis`-suffixed outputs)
- `05_validate_significance.py` — Gate-3 validation vs hand-labeled gold (`--track fonsi|eis`)
- `06_create_figures.R` — headline tables + figures (takes **no flags**; EIS block auto-runs if the `_eis` parquet exists)

**Reference docs** (in `phase2/code/deliverable02/`): `gold_labeling.md`, `gold_labeling_eis.md`, `gold_adjudicate.md`, and the architecture doc `phase2/architecture/deliverables/deliverable02.md`.

---

## Design in one screen

- **Multi-determination extraction (v3 prompt, schema `d2_v2_11`).** Each LLM call returns a **list** of determinations — one per resource area the window concludes on — realizing the `document × resource_area × determination` grain. Window cap = 16,000 chars (FONSI) / 24,000 (EIS) so whole Environmental-Consequences chapters are read in full.
- **Thresholds in a child table.** The determination record carries only `primary_threshold_*`; every cited threshold is one row in `determination_thresholds.parquet`.
- **`agency_scope_status`** ∈ {`primary_blm_doe_family`, `context_other_agency`, `manual_scope_review`} is the headline-denominator gate. FONSI stays BLM+DOE; EIS analysis covers all agencies (descriptive, never uses the decision date).
- **Two-period regime:** `decision_period` (descriptive) + `applicability_period` (legal-method); `framework_regime` is a pinned alias of `decision_period`.
- **FONSI first, EIS later.** The FONSI track (~$15) is run, validated, and analyzed **before** the ~9×-larger EIS track (~$110). `04` writes `_eis`-suffixed outputs so the tracks never clobber each other.

---

## Stage 0 — deterministic foundation ($0, key-free; safe to re-run any time)

```bash
conda run -n nepa python phase2/code/deliverable02/_run.py                     # 00 regime → 01 corpus + cohorts
conda run -n nepa python phase2/code/deliverable02/03_build_gold_set_queue.py  # FONSI labeling worksheet
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --dry-run   # candidate build, no key/spend
```

`_run.py` chains only `00_resolve_framework_regime.py` → `01_build_d2_inventory.py`. The `--dry-run` on `02` builds the candidate windows and runs the deterministic assembly with no API client constructed.

---

## Stage 1 — FONSI LLM pass (💰 billable; ONE Keychain password via `--batch-run`)

```bash
# optional ~$1 sync spike first (sanity-check the prompt on 30 windows):
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --sample 30 --model claude-sonnet-5

# full pass — Batch API (50% price), submit + poll + fetch in one process:
conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --batch-run --model claude-sonnet-5
```

Batch modes: `--batch-run` (one password: submit → poll → fetch → build), or split `--batch-submit` / `--batch-fetch [--wait]` (one password each). Batches are auto-chunked to stay under the API's 100,000-request / 256 MB caps. `temperature=0` is sent only on Haiku — Sonnet 5 / Opus 4.8 reject sampling parameters.

Outputs: `significance_determinations.parquet`, `determination_thresholds.parquet`, `significance_section_candidates.parquet`, `mitigation_signal_matches.parquet`, `significance_run_manifest.parquet`.

---

## Stage 2 — validate + FONSI-only analysis ($0)

```bash
conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py   # Gate 3 vs hand-labeled gold (default --track fonsi)
Rscript phase2/code/deliverable02/06_create_figures.R                           # FONSI-only tables + figures
quarto render phase2/reports/deliverable02.qmd
```

Decide from these outputs whether the EIS pass is worth running. `06` takes **no flags** (it errors on any); its EIS block runs automatically only when `significance_determinations_eis.parquet` exists.

---

## Stage 3 — EIS LLM pass + combined analysis (💰 billable; gated on FONSI Gate 3)

```bash
# retrieval check, free:
conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --dry-run --sample 800
# full pass (Batch API, all sections):
conda run -n nepa python phase2/code/deliverable02/04_extract_eis_significance.py --batch-run --sample 0 --model claude-sonnet-5
# validate the EIS track:
conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py --track eis
# figures — EIS block auto-runs; NO flags:
Rscript phase2/code/deliverable02/06_create_figures.R
quarto render phase2/reports/deliverable02.qmd
```

`03_build_gold_set_queue_eis.py` builds the EIS labeling worksheet (run before the EIS gold is labeled). `--sample 0` on `04` = ALL sections; a positive `--sample` caps sections for a spike. `--out-suffix` (default `_eis`) keeps EIS outputs from clobbering FONSI.

---

## CLI reference

### `02_extract_fonsi_significance.py`

| Flag | Default | Notes |
|---|---|---|
| `--dry-run` | off | Key-free deterministic candidate pass (no spend) |
| `--model` | `X.DEFAULT_MODEL` | e.g. `claude-sonnet-5`, `claude-haiku-4-5` |
| `--sample N` | 0 | Limit candidates (debug / spike) |
| `--batch-run` | off | Batch API: submit → poll → fetch → build (one password) |
| `--batch-submit` / `--batch-fetch` | off | Split submit / fetch (one password each) |
| `--wait` | off | With `--batch-fetch`: poll until the batch ends |
| `--rejoin-mitigation` | off | Recompute the mitigation page-window join without re-calling the LLM |
| `--matching-rule` / `--mitigation-dep-rule` | defaults | Adjudication/mitigation-flag rule variants |

### `04_extract_eis_significance.py`

Same batch flags as `02`, plus: `--sample N` (default 500; `0` = ALL), `--gold-sample N`, `--out-suffix` (default `_eis`), `--use-cached-candidates`.

### `05_validate_significance.py`

| Flag | Default | Notes |
|---|---|---|
| `--track {fonsi,eis}` | `fonsi` | Which track's gold to validate against |

---

## Outputs

All analysis parquets are under `data/analysis/deliverable02/`.

| File | Track | Description |
|---|---|---|
| `significance_determinations.parquet` | FONSI | One row per `document × resource_area × determination`; `determination_class`, `determination_scope`, `primary_threshold_*`, `mitigation_dependent`, provenance |
| `determination_thresholds.parquet` | FONSI | Child table — one row per cited threshold |
| `significance_determinations_eis.parquet` | EIS | Same grain, `_eis` fields (`alternative_name`, `significance_factor`, `impact_type`) |
| `determination_thresholds_eis.parquet` | EIS | EIS threshold child table |
| `mitigation_signal_matches.parquet` | FONSI | Frozen cue-span × condition-row page-window join (mitigation flag) |
| `validation_metrics.parquet` / `validation_metrics_eis.parquet` | both | Gate-3 metrics (overall + held-out) |
| `significance_run_manifest.parquet` | both | Input/output paths, row counts, content hashes, model, prompt/schema versions |

`gold/` holds the adjudicated gold sets; `significance_gold_queue.csv` (in `output/deliverable02/`) is the hand-labeled worksheet `05` adopts automatically.

Figures + tables: `06_create_figures.R` → `output/deliverable02/analysis/` (`fig_fonsi_funnel.png`, `fig_eis_funnel.png`, `fig_fonsi_vs_eis.png`, `fig_eis_above_line.png`, `eis_*.csv`, `fonsi_*.csv`, …).

---

## Run results (confirmed from the current parquets)

- **FONSI:** 7,250 determinations from 3,478 windows (Sonnet 5, ~$15). Headline base = 193 projects / 258 documents; 1,990 primary-scope analytic determinations. ~58% of FONSIs are mitigated.
- **EIS:** 59,357 raw determinations from 21,854 windows (21,852 succeeded; ~$110–115). Analytic set = 13,240 determinations across 506 projects / 1,082 documents; 2,198 above-the-line.
- **Gate 3 (FONSI, held-out):** candidate-is-determination F1 0.978 · resource-detection 0.886 · class macro-F1 0.808 · mitigation-dependent 0.622 (0.584 held-out) · threshold accuracy 0.664.
- **Gate 3 (EIS, held-out):** window 0.835 · resource 0.679 · class macro-F1 0.686 · mitigation 0.704 · threshold 0.616. Recall is the soft spot (window recall 0.77) — EIS rates are reported as a well-grounded floor.

---

## Audit & validation

- Every output carries `schema_version` (`d2_v2_11`) + `*_run_at`; determinations carry `significance_extraction_run_at` (all rows) and `significance_llm_run_at` (LLM-success rows).
- Actual spend is auditable after any run via `significance_run_manifest.parquet` + `batch_manifest_*.json` (request counts) and the per-response `usage` fields.

---

## Notes

- **FONSI depends on D6.** The FONSI finding-section spans (`source_unit_id = evidence_span_id`) come from D6's span extraction; the 427→193 headline gap is a coverage limit of that upstream extraction, not a sampling choice. Rebuild D6's `fonsi_evidence_spans.parquet` before a FONSI re-run if the corpus changed.
- **Prompt caching does not help** — the shared ~300-token prefix is below the minimum cacheable size; per-window text dominates every request.
- **`06` is flag-free by design.** A former `--with-eis` flag folded EIS rows into the FONSI headline gate and polluted every FONSI table (193 analyzed projects became 325) — it was removed. Explore combined tracks in an ad-hoc session, not via `06`.
- **Mitigation flag (2026-07-22):** `mitigation_dependent` was tightened to the T5 rule (real same-resource overlap + ≥2 committed conditions); `--rejoin-mitigation` recomputes it without re-calling the LLM.
