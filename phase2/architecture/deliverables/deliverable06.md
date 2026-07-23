# D6: Patterns in FONSIs — Architecture (v3, tech_group x action grid)

> **Status (2026-07, as-built).** The pipeline is LLM-backed end to end. A one-pass
> comprehensive enrichment (`03_enrich_llm.py`, claude-sonnet-4-6, schema
> `d6_enrich_schema_v5`, 39-field extraction + a cheap classify-only re-ask) reads
> all 451/452 clean-energy EA→FONSI projects once; `10_action_label.py` derives a
> controlled action VERB per FONSI from the cached enrichment; `09_wire_enrichment.py`
> forms `tech_group__action` grid CELLS (52 of them) and overwrites the deterministic
> `candidate_facts` / `candidate_mitigation_summary` from `03`/`05` with the
> LLM-backed versions; `07_classify_and_rank.py` now classifies and ranks all 52
> cells (not the old 5 candidate categories). This file documents that as-built
> state. Older text below describing "5 candidates" / a gated `--use-llm` pass is
> superseded by this section and by the Pipeline Wiring section — kept only where
> it still describes the surviving deterministic Track A/B/C logic.

**Goal:** Identify a small, defensible shortlist of recurring clean-energy action
categories in prior EAs/FONSIs that may warrant CATF, policy, and legal review
for new or expanded categorical exclusions (CEs). For each candidate: a crisp
action definition, evidence it recurs with no significant impact, the recurring
bounding limits, whether findings depend on case-specific mitigation, whether an
existing CE already covers it, and traceable citations.

**Self-contained:** Partially. Reads `projects_combined.parquet` and the D3
`projects_nepa_reviews.parquet` / `ce_citations.parquet` for base-rate
denominators and CE-use evidence, and D4's `timeline_project_dates.parquet` for
authoritative decision dates (post-FRA tabulation). All FONSI inventory and
packet inputs are pre-built parquets under `data/analysis/deliverable06/`. D6
also writes one output D2 depends on: `retag_condition_resources.py` rebuilds
`fonsi_conditions.parquet`'s `resource_area` tagging, which D2's mitigation join
consumes.

**Status:** v3 grid pipeline built and running (LLM enrichment + deterministic
Tracks A/B/C + verdicts + figures + report all in). Two categorization schemes
now coexist on disk (see "Two categorization schemes" below); the grid scheme is
authoritative for verdicts/ranking/report. Forward plan and item numbering
(`#38`, `#39`, `#40`, `#44`, `#47`, `A1`-`A3`, `G1`) trace to
`phase2/code/deliverable06/feedback.md`.

---

## Plain-language summary (for clients)

We start from clean-energy projects whose environmental review **ended in a
Finding of No Significant Impact (FONSI)** — i.e., the agency did a full
Environmental Assessment (EA) and concluded the project would not significantly
harm the environment. A FONSI is strong evidence that an action *category* may be
a good candidate for a categorical exclusion (a class of actions agencies may
approve without a full EA/EIS).

The pipeline resolves each candidate category to **one of three actionable
verdicts**:

1. **NEW** — a recurring, CE-shaped FONSI class that no existing CE covers. A
   genuine legislative opportunity.
2. **EXPAND** — an existing CE covers the action, but our FONSIs go beyond its
   stated bounds and still get a FONSI — so the cap should widen.
3. **ADOPT** — an existing CE covers the action at agency X, but the agencies
   doing these FONSIs don't have it — so other agencies should adopt it.

*(Candidates that fall inside an existing CE's bounds and agency coverage are
dropped as already-covered.)*

The output is a *starting point* for discussion, not a legal determination. CATF
reviews it and tells us which categories to pursue in depth.

---

## What the corpus is (and is not)

**Deep-extraction corpus = clean-energy, EA-source FONSI projects.** These are
projects where NEPATEC's `document_type_clean = "FONSI"` (excluding RODs),
restricted to EA-source records, with one canonical FONSI chosen per project.
A FONSI is the *decision* that concludes an EA, so mining "FONSI projects" means
mining the **EA analysis plus its FONSI**.

The text we analyze is drawn from three document roles per project: **canonical
FONSI + linked EA(s) + supporting FONSI**. So we are reading the EAs, not just
the one-page finding.

**EIS and CE projects are context, not deep-mined.**

- An **EIS** means the agency found *significant* impact — the opposite of a CE
  candidate. We do not deep-extract EIS text; instead we **count** EIS projects
  in each candidate's base-rate denominator to show how often the action
  escalates to a full EIS (a survivorship-bias guard).
- A **CE** means the action was *already* categorically excluded. We count CE
  projects in the denominator too (evidence the action is sometimes already
  CE'd), but their document text is sparse.

So: **EA/FONSI = evidence; EIS + CE = base-rate context.** This is the right
shape for CE development — we look at what got a FONSI, and we measure how often
the same action category goes the other routes.

---

## Two categorization schemes (why they coexist)

The pipeline was built around a hand-picked, 5-category taxonomy
(`candidates.py`: transmission_upgrade, solar, geothermal_exploration,
temporary_resource_assessment, wind_onshore) and later refactored to a **fully
enumerated `tech_group__action` grid** (52 observed cells, every clean FONSI in
exactly one cell — no "other" black hole for the candidate corpus itself). Both
schemes are still on disk because different scripts run at different points in
the chain and some were never migrated:

| Scheme | `candidate_category` values | Built by | Consumed by |
|---|---|---|---|
| Legacy 5-category | `transmission_upgrade`, `solar`, `geothermal_exploration`, `temporary_resource_assessment`, `wind_onshore` | `01_select_candidate_corpus.py` (rule-based membership over `candidates.py`) | `candidate_corpus.parquet`, `candidate_base_rates.parquet`, `candidate_descriptive.parquet` (via `04`), `candidate_mitigation_boundary.parquet` (via `05`) |
| Grid (as-built, authoritative) | `f"{tech_group}__{action}"`, e.g. `Transmission__upgrade`, `Solar__new_build` (52 observed cells) | `09_wire_enrichment.py`, joining `fonsi_enrichment.parquet` (tech_group) x `fonsi_action_labels.parquet` (action verb from `10_action_label.py`) | `candidate_facts.parquet`, `candidate_mitigation_summary.parquet` (via `09`), `candidate_ce_comparison.parquet` (via `04`'s grid-aware CE-comparison block), `candidate_verdicts.parquet` (via `07`), all `d6_*.csv` report tables |

`07_classify_and_rank.py` reads `candidate_facts["candidate_category"]` — i.e.
the **grid** — so every verdict, rank, and report table is grid-cell-keyed.
`candidate_base_rates` / `candidate_descriptive` / `candidate_mitigation_boundary`
remain legacy-5-keyed and are largely superseded for the report by their
grid-cell equivalents (`corpus_mitigation_stats.parquet` from `09` replaces the
legacy corpus-wide mitigation rollup).

---

## Script quick-reference

All pipeline scripts live under `phase2/code/deliverable06/`. v1 scripts are
archived to `_archived_v1/` and are not orchestrated.

### Chained by `_run.py` (as-built order)

`01 → 02 → 03 → 04 → 05 → 06 → 09 → 07 → 11 → 12 → 13 → 14`, then `08` (R).

| Order | Script | Scheme | Plain-language role | What it does |
|---|---|---|---|---|
| 1 | `01_select_candidate_corpus.py` | legacy-5 | **pick** | Assigns projects to the 5 legacy candidate categories (tech_group + keyword rules); resource-assessment prevalence/de-overlap screen; storage-deployment scan. |
| 2 | `02_assemble_candidate_evidence.py` | legacy-5 | **gather** | Pulls each candidate FONSI's typed text from `fonsi_project_packets` + span-level provenance; falls back to `fonsi_document_sections` for missing projects. |
| 3 | `03_extract_candidate_facts.py` | legacy-5 | **extract (deterministic)** | Deterministic acres/miles/wells/siting-booleans/extraordinary-circumstances/spaCy+MiniLM action definition. Output is later fully overwritten by `09`; kept for the `05` mitigation join and as an audit trail. Its own `--use-llm` (narrow facts prompt) is superseded by `03_enrich_llm.py` and is not invoked by `_run.py`. |
| 4 | `04_base_rates_and_ce.py` | both | **contextualize + CE match** | Legacy-5 base rates/descriptive breakdowns (unchanged). CE comparison block was refactored to build the **grid cells** directly from `fonsi_enrichment.parquet` x `fonsi_action_labels.parquet` and rank CEs per cell (per-member median cosine, robust to long-query dilution) — this output (`candidate_ce_comparison.parquet`) is grid-keyed, not legacy-5-keyed. |
| 5 | `05_mitigation_and_boundary.py` | legacy-5 | **mitigated-FONSI + boundary (deterministic)** | Dual-signal mitigated-FONSI flag (D02 cues + `fonsi_conditions.parquet`) + boundary-language extraction, legacy-5-keyed. Superseded for the report by `09`'s LLM-backed `candidate_mitigation_summary`, kept as an independent deterministic cross-check. |
| 6 | `06_ce_landscape.py` | n/a (CE-only) | **CE landscape** | Embeds `ce.json`, cross-agency near-duplicate CEs, per-agency counts, numeric-bounds distribution. Unchanged. |
| 7 | `09_wire_enrichment.py` | grid | **wire LLM enrichment in** | Builds the 52 `tech_group__action` cells from `fonsi_enrichment.parquet` + `fonsi_action_labels.parquet`; overwrites `candidate_facts.parquet` and `candidate_mitigation_summary.parquet` with LLM-backed values (sizes, siting booleans, mitigation, verified action citation); writes `corpus_mitigation_stats.parquet`. **Aborts `_run.py` if `fonsi_enrichment.parquet` is missing** (see Pipeline Wiring). |
| 8 | `07_classify_and_rank.py` | grid | **verdict + rank** | Classifies each of the 52 grid cells into new/expand/adopt/already_covered; multi-factor `rank_score`; G1 recurrence gate (`shortlist_tier`); #38 annotate-only agency-crosswalk columns; writes `candidate_verdicts.parquet`, `d6_comparison_table.csv`, the three drill-down lists, per-cell evidence, and `rank_sensitivity.csv` (A3). |
| 9 | `11_expand_analysis.py` | grid | **#39 generalized expand** | For every grid cell with a matched CE and a stated numeric bound, compares the CE-shaped FONSIs' size distribution to the bound and suggests a raised cap (p90). Not gated by `07`'s expand verdict — reports on ALL bounded matches, not just the ones `07` already flagged. |
| 10 | `12_other_action_themes.py` | grid | **#40 within-cell theme mining** | Clusters the 92 `action=='other'` FONSIs on local sentence embeddings (KMeans + TF-IDF cluster labels) to surface sub-themes the 10-verb vocabulary missed. Supplementary only — asserts `candidate_verdicts.parquet` is byte-identical before/after. |
| 11 | `13_postfra_refresh.py` | grid | **A2 post-FRA tabulation** | Per grid cell, how many CE-shaped FONSIs were decided after the FRA cut date (2023-06-03) vs before vs undated, using D4's authoritative `decision_date` (merged in `09`). |
| 12 | `14_threshold_retrieval.py` | n/a (spans) | **#44 threshold retrieval** | Deterministic regex retrieval of significance-threshold phrases ("would be significant if", "not to exceed", etc.) over finding/condition/resource spans (span_type=='boundary' is nearly empty). |
| 13 | `08_create_figures.R` | grid + legacy | **figures** | Report figures (~16 PNGs) with `theme_catf`. Reads `candidate_verdicts`, `candidate_mitigation_summary`, `candidate_facts`, `candidate_corpus`, `ce_landscape_ces`, and `fonsi_enrichment` parquets. |

### Standalone-BILLABLE prerequisites (NOT in `_run.py`; run once, cached — re-runs are $0; user-launched)

| Script | Role |
|---|---|
| `03_enrich_llm.py` | **The** enrichment pass: one structured (tool-use) call per project extracts all 39 `ENRICHMENT_FIELDS` (schema `d6_enrich_schema_v5`) → `fonsi_enrichment.parquet`. Two cached stages (`--stage extract\|classify\|both`): EXTRACT (expensive, ~5,000 in / 1,700 out tok/call) then CLASSIFY (cheap re-ask of only `action_category`, ~1,340 in / 140 out tok/call, overwrites `action_category` and preserves the extraction value as `action_category_pass1`). `_run.py` **aborts** the chain if this output is missing. |
| `10_action_label.py` | Reuses the cached enrichment summary (no document re-read) to assign one controlled action VERB per FONSI from an 11-value vocabulary; `is_codifiable` is derived deterministically from the verb (not the LLM) → `fonsi_action_labels.parquet`. |
| `retag_condition_resources.py` | D6 `#47`: rebuilds `fonsi_conditions.parquet`'s `resource_area` tagging in place — Tier-1 heading-inheritance (**disabled by default since 2026-07-22**, commit 82d47e9: gold-validated precision 0.20; opt back in via `--use-tier1`), then Tier-2 scoped Haiku multi-label pass (deduped by condition-text hash) on `mitigation_commitment` rows only. Fixes a D2-facing quality gap (D6 verdicts/mitigation-share never read this field); `~$4.23` full run. |

### $0, network-cached scaffold (run once before `07`, safe to re-run)

| Script | Role |
|---|---|
| `ce_ecfr_verify.py` | D6 `A1`/`#37`: for the adopt/expand grid cells (24), fetches the canonical eCFR text of each cell's top-5 matched CEs ($0, cached to `data/raw/deliverable06/ecfr/`) and writes an empty-verdict `candidate_ce_coverage.parquet` + a human worksheet for a reviewer (or an optional billable `--llm --dry-run`-projected pass) to adjudicate `covers/partially_covers/does_not_cover/unclear`. |

### Other standalone tools

| Script | Role |
|---|---|
| `_run.py` | Orchestrator: runs `01→09→07→11→12→13→14` Python then `08` via Rscript. Aborts before `09` if `fonsi_enrichment.parquet` is absent. |
| `qa_deliverable06.py` | 25-assertion QA gate over the grid invariants (see QA section below). Run after the chain. |
| `candidates.py` | Legacy-5 rulebook: category membership rules, subtype definitions, CE-story text, storage-scan config. |
| `common.py` | Shared paths/IO helpers, including `D6_ANALYSIS_DIR` / `D6_OUTPUT_DIR` / `D6_REVIEW_DIR` / `D6_RAW_DIR` constants. |
| `ce_source.py` | Loads `notes/deliverable06/ce.json` (CE Explorer v2.0.0). |
| `bounds.py` | Parses numeric bounds (acres/miles/kV/MW/wells) from CE description text. |
| `embeddings.py` | `all-MiniLM-L6-v2` embedding wrapper (used in `04`, `06`, `12`). |
| `enrich_lib.py` | Shared enrichment machinery: span-based evidence-packet builder (typed, per-section budgeted, `[S#]`-tagged), stratified pilot sampler, tool-use call wrappers for enrichment/classification/action-labeling, quote verification against source spans, Keychain key loader. Imported by `03_enrich_llm.py`, `10_action_label.py`, `benchmark_models.py`, `retag_condition_resources.py`. |
| `prompts.py` | Single source of truth for all LLM prompts + schemas: `ENRICHMENT_FIELDS` (enrichment), `build_classification_prompt`/`classification_tool_schema` (Stage 2 re-classify), `build_action_label_prompt`/`ACTION_VERBS`/`is_codifiable_for` (Stage 3 action labeling). |
| `ce_agency_crosswalk.py` | D6 `#38`: parent-department ↔ sub-agency lookup (`DEPT_MEMBERS`), used ANNOTATE-ONLY by `07` to compute net vs gross adopt targets. |
| `mitigation_conditions.py` (in `code/extract/`, imported by `retag_condition_resources.py`) | Heading-inheritance classifier reused for Tier-1 of `#47`. |
| `extract_ce_catalog.py` | Renders `ce.json` to `notes/deliverable06/_ce_catalog_extracted.md`; run once after CE source updates. |
| `benchmark_models.py` | Model-selection tool: runs the production enrichment prompt through Haiku/Sonnet/Opus on a sample; calls the paid API; run once before committing to a model. |

---

## Enrichment Schema (`fonsi_enrichment.parquet`)

`fonsi_enrichment.parquet` is the single comprehensive LLM read of every clean
FONSI (452 rows: 451 successfully enriched + 1 skipped for no evidence) that
everything downstream (`09`, `10`, `04`'s CE-comparison block, `11`, `12`, `13`)
is keyed off. **63 columns total**, confirmed via
`DESCRIBE SELECT * FROM 'fonsi_enrichment.parquet'`:

- `project_id` (1 column)
- **39 substantive LLM-extracted fields**, from `ENRICHMENT_FIELDS` in
  `prompts.py` (single source of truth for both the prompt and the output
  columns — schema `d6_enrich_schema_v5`, prompt `d6_enrich_prompt_v5`). Code
  comments in `03_enrich_llm.py` still say "37-field schema" — that count is
  stale; `len(prompts.ENRICHMENT_FIELDS)` is 39.
- `evidence_cited` (1 column) — computed, not a raw LLM field: each verbatim
  quote from `evidence` / `significance_thresholds` / `referenced_ce_citations`
  / `ce_development_language`, resolved to its source span and verified
  (character-match, unicode-punctuation-folded) against the actual excerpt text.
- 2 audit timestamps, 6 metadata-passthrough columns, 5 computed
  confidence/QA columns, 9 Stage-2 classification columns (18 columns total).

### The 39 substantive LLM fields (grouped by consuming analysis)

**Action identity, scale, siting (Analysis 1):**

| Field | Type | Purpose |
|---|---|---|
| `action_summary` | string | 1-2 plain-English sentences describing the federal action. |
| `purpose_and_need` | string | 1 sentence on why the action is needed. |
| `action_category` | string | One of the 5 legacy candidate types + `other` (overwritten by Stage 2 classify). |
| `action_category_other` | string\|null | Short label if `action_category == 'other'`. |
| `action_label_freeform` | string | Normalized free-form action label, independent of category — used for `12`'s clustering. |
| `potential_ce_theme` | string\|null | If the action looks like a plausible NEW CE theme, a short name for it. |
| `why_not_current_candidate` | string\|null | If `other`, one phrase on what kind of action it is. |
| `is_bounded_low_impact` | boolean\|null | True if a small, routine, low-impact version of the action — the core "CE-shaped" gate (Rule B). |
| `bounded_rationale` | string | One sentence justifying `is_bounded_low_impact`. |
| `key_activities` | array[string] | Discrete physical activities (JSON-encoded). |
| `disturbance_acres` | number\|null | Ground-disturbance acres (never study/planning-area acreage). |
| `line_miles` | number\|null | Length of the transmission/distribution line itself. |
| `access_road_miles` | number\|null | Access-road miles, kept separate from `line_miles`. |
| `capacity_mw` | number\|null | Generation/storage capacity. |
| `voltage_kv` | number\|null | Transmission voltage. |
| `well_count` | integer\|null | Wells/borings/boreholes (incl. spelled-out counts) — operative bound for geothermal/resource-assessment. |
| `within_existing_row` | boolean\|null | Within an existing right-of-way / developed corridor. |
| `new_access_road` | boolean\|null | NEW (not merely improved) access roads built. |
| `previously_disturbed_land` | boolean\|null | Sited on previously disturbed/developed land. |
| `is_temporary` | boolean\|null | Temporary disturbance (survey/testing) vs. a permanent facility. |
| `land_ownership` | string\|null | One of `BLM`, `federal_other`, `private`, `mixed`, `other`. |

**Significance & mitigation (Analysis 2):**

| Field | Type | Purpose |
|---|---|---|
| `is_mitigated_fonsi` | boolean\|null | True if the FONSI depends on committed mitigation rather than inherent low impact. |
| `mitigation_dependence` | string | `none` / `design_feature_only` / `case_specific_dependent` / `permit_or_consultation_condition` / `monitoring_only` / `unclear`. |
| `mitigation_summary` | string | Short summary of committed mitigation measures. |
| `mitigation_resource_areas` | array[string] (JSON) | Resource areas the mitigation addresses (11-value enum + `other`). |
| `key_impacts` | array[string] (JSON) | Main environmental impacts, each tagged to a resource area. |
| `residual_impacts` | string\|null | Impacts remaining after mitigation. |
| `significance_thresholds` | array[object] (JSON) | Explicit threshold/counterfactual statements only, each with `statement`, `span_ref`, `metric`, `value`, `unit`, `is_project_fact`. |
| `extraordinary_circumstances` | string\|null | Extraordinary circumstances noted that could preclude a CE. |
| `decision_basis` | string | `inherently_low_impact` / `mitigated_to_below_significant` / `small_scale` / `other`. |
| `significance_factors` | array[string] (JSON) | Which intensity factors (context, controversy, cumulative_effects, etc.) the EA leaned on. |

**Direct-text / citation fields (verified downstream, not trusted blindly):**

| Field | Type | Purpose |
|---|---|---|
| `evidence` | array[object] (JSON) | Verbatim quotes backing `action`/`finding`/`size`/`mitigation` claims, each with `claim`, `span_ref`, `quote`. |
| `referenced_ce_citations` | array[object] (JSON) | Existing CEs/NEPA authorities the document itself cites. |
| `ce_development_language` | string\|null | Verbatim language signaling the action is routine/minor (CE-precedent phrasing). |
| `ce_development_span_ref` | string\|null | `[S#]` tag `ce_development_language` was copied from. |

**Context extras:**

| Field | Type | Purpose |
|---|---|---|
| `cooperating_agencies` | array[string] (JSON) | Named cooperating/consulting agencies. |
| `is_tiered` | boolean\|null | True if the EA tiers from a programmatic EIS/EA. |
| `tiers_from` | string\|null | The programmatic document it tiers from. |
| `extraction_confidence` | string | Model's self-rated confidence: `high`/`medium`/`low`. |

### Audit, metadata, and classification columns (24 columns)

| Column | Type | Description |
|---|---|---|
| `enrichment_extraction_run_at` | object | ISO-8601 UTC, all rows (this run). |
| `enrichment_llm_run_at` | object | ISO-8601 UTC, only on LLM success (the original call), else `""`. |
| `project_title`, `project_type`, `tech_group`, `lead_agency_harmonized`, `project_state`, `canonical_fonsi_document_id` | object | Metadata passthrough (self-contained analysis). |
| `n_quotes`, `n_verified_quotes` | int64 | Count of cited quotes and how many verified against source spans. |
| `verified_quote_rate` | float64 | `n_verified_quotes / n_quotes`. |
| `field_fill_rate` | float64 | Share of the 39 fields that are non-empty. |
| `confidence_score` | float64 | `0.6 * verified_quote_rate + 0.4 * field_fill_rate` — computed, not the model's self-rating. |
| `action_category_pass1` | object | The Stage-1 (extraction) `action_category`, preserved once before Stage 2 overwrites `action_category`. |
| `classification_parse_ok` | bool | Stage-2 call succeeded. |
| `classification_cache_hit` | bool | Stage-2 result served from cache. |
| `classification_error` | object | Stage-2 failure reason, if any. |
| `classification_confidence`, `classification_rationale` | object | Stage-2 model outputs. |
| `classification_prompt_version`, `classification_run_at`, `classification_config_sha` | object | Stage-2 audit trail (empty/unstamped if the call failed or was skipped — a partial classify run can never masquerade as complete). |

---

## Data flow

```mermaid
flowchart TD
    R[projects_nepa_reviews.parquet] --> N1[01 pick legacy-5]
    I[fonsi_project_inventory.parquet] --> N1
    N1 --> C[candidate_corpus.parquet]

    C --> N2[02 gather]
    P[fonsi_project_packets.parquet] --> N2
    N2 --> EV[candidate_evidence_packets.parquet]

    EV --> N3[03 extract deterministic]
    N3 --> F0[candidate_facts.parquet v1]

    LLM([03_enrich_llm.py<br/>STANDALONE BILLABLE<br/>39-field enrichment]) --> ENR[fonsi_enrichment.parquet<br/>451/452 enriched]
    ENR --> N10[10_action_label.py<br/>STANDALONE BILLABLE]
    N10 --> LAB[fonsi_action_labels.parquet<br/>action verb + is_codifiable]

    C --> N4[04 base rates + CE match]
    ENR --> N4
    LAB --> N4
    CE[ce.json] --> N4
    N4 --> BR[candidate_base_rates.parquet<br/>legacy-5]
    N4 --> XC[candidate_ce_comparison.parquet<br/>GRID, 52 cells x top-8]

    EV --> N5[05 mitigation + boundary]
    N5 --> MB[candidate_mitigation_boundary.parquet<br/>legacy-5]

    CE --> N6[06 CE landscape]
    N6 --> CL[ce_landscape_ces.parquet]

    ENR --> N9[09 wire enrichment]
    LAB --> N9
    F0 --> N9
    TL[D4 timeline_project_dates.parquet] --> N9
    N9 --> F[candidate_facts.parquet<br/>OVERWRITTEN, GRID, 451 rows]
    N9 --> MS[candidate_mitigation_summary.parquet<br/>OVERWRITTEN, GRID]
    N9 --> CS[corpus_mitigation_stats.parquet]

    F --> N7[07 classify + rank]
    XC --> N7
    MS --> N7
    N7 --> VD[candidate_verdicts.parquet<br/>52 cells]
    N7 --► CT[d6_comparison_table.csv]
    N7 --► DL[d6_new/expand/adopt.csv]
    N7 --► RS[rank_sensitivity.csv A3]

    F --> N11[11_expand_analysis.py #39]
    XC --> N11
    N11 --► EA[expand_analysis.csv]

    F --> N12[12_other_action_themes.py #40]
    ENR --> N12
    VD -.assert unchanged.-> N12
    N12 --> OAT[other_action_themes.parquet]

    F --> N13[13_postfra_refresh.py A2]
    N13 --► PFR[postfra_recurrence.csv]

    SP[fonsi_evidence_spans.parquet] --> N14[14_threshold_retrieval.py #44]
    N14 --► TC[threshold_candidates.csv]

    VD --> N8[08 R figures]
    MS --> N8
    F --> N8
    ENR --> N8
    N8 --► FIG[figures/fig_d6_*.png]

    VD --> ECFR[ce_ecfr_verify.py A1/#37<br/>$0, network-cached]
    XC --> ECFR
    ECFR --> COV[candidate_ce_coverage.parquet]

    COND[fonsi_conditions.parquet] --> RETAG[retag_condition_resources.py #47<br/>STANDALONE BILLABLE]
    RETAG --> COND

    FIG --> QMD[phase2/reports/deliverable06.qmd]
    VD --> QMD
```

---

## Inputs (all read-only)

| File | Role |
|---|---|
| `deliverable06/fonsi_project_inventory.parquet` | Clean EA-source FONSI projects (canonical selection). |
| `deliverable03/projects_nepa_reviews.parquet` | Full clean universe + `process_type` for base-rate denominators. |
| `projects_combined.parquet` | Used in `01` to enrich the universe with `project_title` / `project_description` for subtype matching. |
| `deliverable06/fonsi_project_packets.parquet` | Per-project typed text (action/finding/resource/condition/boundary), drawn from EA+FONSI documents. |
| `deliverable06/fonsi_evidence_spans.parquet` | Span-level provenance: `section_id`, `evidence_span_id`, `source_span_sha256`, page. Also the source for `14`'s threshold retrieval and `03_enrich_llm.py`'s balanced, per-section-budgeted evidence packets. |
| `deliverable06/fonsi_conditions.parquet` | Condition roles/obligations — reused for the mitigation signal in `03`/`05`, and rebuilt in place by `retag_condition_resources.py` (`#47`). |
| `deliverable06/fonsi_document_sections.parquet` | Fallback text source in `02`; also `03_enrich_llm.py`'s last-resort section-fallback packet for zero-span projects. |
| `notes/deliverable06/ce.json` | Canonical existing-CE source — CE Explorer export (v2.0.0, 2025-08-07), loaded via `ce_source.py`. |
| `deliverable03/ce_citations.parquet` | Project-level CE-use evidence (D3); used in `04` and `06`. |
| `deliverable06/fonsi_enrichment.parquet` | **Prerequisite, not produced by the chain.** The 39-field LLM enrichment; `09`/`10`/`04`/`11`/`12`/`13` all read it. `_run.py` aborts if absent. |
| `deliverable06/fonsi_action_labels.parquet` | **Prerequisite, not produced by the chain.** The action-verb labels from `10_action_label.py`; `09`/`04` read it. |
| `deliverable/timeline/timeline_project_dates.parquet` | D4's authoritative `decision_date`, merged in `09` onto CE-shaped grid-cell rows for the post-FRA tabulation (`13`). |

---

## Primary Outputs

All analysis parquets are written under `phase2/data/analysis/deliverable06/`.

| File | Scheme | Description |
|---|---|---|
| `fonsi_enrichment.parquet` | n/a | The 39-field + audit LLM enrichment of every clean FONSI. 452 rows (451 enriched, 1 skipped). See Enrichment Schema above. |
| `fonsi_action_labels.parquet` | n/a | Per-FONSI action verb (11-value vocabulary) + `is_codifiable` + Stage-3 audit columns. |
| `candidate_corpus.parquet` | legacy-5 | One row per (project, candidate_category) over the full clean universe + observed FONSIs. 13,145 rows. |
| `candidate_evidence_packets.parquet` | legacy-5 | Per-project typed text + span provenance for candidate FONSIs. |
| `candidate_facts.parquet` | **grid** (overwritten by `09`) | One row per enriched clean FONSI: `tech_group__action` cell, LLM-backed sizes/booleans/mitigation/citation, `is_ce_shaped`, `is_codifiable`. **451 rows.** |
| `candidate_base_rates.parquet` | legacy-5 | Three base-rate counts per legacy candidate. 5 rows. |
| `candidate_ce_comparison.parquet` | **grid** | Per-member-median-cosine ranked CE matches, top 8 per grid cell (all `manual_verification_status = pending`). 52 cells x up to 8. |
| `candidate_descriptive.parquet` | legacy-5 | Long-form per-candidate breakdowns by agency, state, and decision year. |
| `candidate_mitigation_boundary.parquet` | legacy-5 | Per (project, legacy candidate): deterministic mitigated-FONSI flag + boundary-language snippets. |
| `candidate_mitigation_summary.parquet` | **grid** (overwritten by `09`) | Per-grid-cell rollup of mitigation share + boundary language, LLM-backed. |
| `corpus_mitigation_stats.parquet` | n/a (corpus-wide) | Corpus-wide mitigated-FONSI share from `is_mitigated_fonsi` across all 451 enriched FONSIs (1 row). |
| `candidate_verdicts.parquet` | **grid** | One row per grid cell: verdict (new/expand/adopt/already_covered), rank_score (+ per-component breakdown), best CE match, adopt targets (gross + `#38` net), expand gaps, `shortlist_tier` (G1). **52 rows.** |
| `ce_landscape_ces.parquet` | n/a (CE-only) | Per-CE landscape: cross-agency near-duplicate links, bound parsing, agency unit. 2,105 rows. |
| `other_action_themes.parquet` | grid (supplementary) | `#40`: cluster id + label for the 92 `action=='other'` FONSIs. Terminal — asserted not to change `candidate_verdicts`. |
| `candidate_ce_coverage.parquet` | grid | `A1`/`#37`: empty-verdict eCFR-adjudication scaffold for the 24 adopt/expand cells' top-5 CE matches, ready for reviewer or LLM fill-in. |

Figures are written to `phase2/output/deliverable06/figures/` (~16 PNGs + matching `.rds`), covering: outcomes waffle, CE-match, sizes, classification, adoption gap, states map, timeline, mitigated overall/share, mitigation-roles wordcloud, and the CE-landscape set (agency counts, numeric-limit distribution, bounds lollipop, elbow/t-SNE cluster scatter, CE-split, coverage grid). See `output/deliverable06/README.md` for the current list.

The top-level client QA table and drill-down review files are in `phase2/output/deliverable06/`.

| File | Description |
|---|---|
| `d6_comparison_table.csv` | Single at-a-glance overview: candidate, verdict, CE-shaped FONSIs, existing CE, adopt targets, expand detail, rank score. |
| `expand_analysis.csv` | `#39`: generalized expand analysis — every (grid cell, metric) pair with a matched CE bound, the FONSI size distribution vs. that bound, and a suggested p90 cap. |
| `postfra_recurrence.csv` | `A2`: per grid cell, CE-shaped FONSI counts pre/post FRA cut date (2023-06-03) + undated, with a NEPATEC-coverage-lag caveat. |
| `threshold_candidates.csv` | `#44`: regex-matched significance-threshold phrases with snippets, by span type. |
| `rank_sensitivity.csv` | `A3`: Dirichlet (2,000 draws) + one-at-a-time weight-sensitivity of the opportunity ranking, for reportable cells (non-contrast, ≥2 CE-shaped FONSIs). |
| `ce_verification_worksheet.csv` / `notes/deliverable06/ce_ecfr_verification.md` | `A1`/`#37`: human-readable eCFR-adjudication worksheet from `ce_ecfr_verify.py`. |
| `review/d6_new.csv` | Client develop shortlist (verdict `new`, codifiable, G1-gated to `main`/`exploratory` tiers). |
| `review/d6_expand.csv` | Full verdict rows for EXPAND cells. |
| `review/d6_adopt.csv` | Full verdict rows for ADOPT cells. |
| `review/d6_candidate_evidence_<tech>__<action>.csv` | Per-grid-cell project-level evidence with citations (one file per cell). |
| `review/candidate_*_review.csv` | Per-pipeline-step QA (legacy-5 membership, extraction, mitigation, CE match, descriptive, storage scan). |
| `review/ce_landscape_summary.csv` | Existing-CE counts per agency unit. |

The client-facing deliverable is the rendered report at `docs/phase2/reports/deliverable06.html`
(source: `phase2/reports/deliverable06.qmd`).

---

## Module Architecture

### `03_enrich_llm.py` + `enrich_lib.py` — the enrichment pass (standalone, billable)

One structured tool-use call per clean FONSI extracts all 39 `ENRICHMENT_FIELDS`
from a **balanced, span-tagged evidence packet**: per-section budgets (3 action,
3 finding, 3 condition, 2 boundary, 2 resource excerpts, `[S#]`-tagged with
page/document/heading), plus up to 3 size-figure spans not already shown (fixes
packet-coverage misses on acres/miles/MW/kV). Falls back to the typed packet
text, then to broad `document_sections.parquet`, for the rare project with no
usable spans — no paid row is metadata-only.

**Two cached stages**, each independently resumable:
- **EXTRACT** — the expensive pass (~5,000 in / 1,700 out tok/call), cached on
  `(prompt_version, schema_version, model, packet_text)`.
- **CLASSIFY** — a cheap re-ask of only `action_category` (~1,340 in / 140 out
  tok/call) from the cached summary, with real category definitions + an enum
  tool schema; overwrites `action_category`, preserves the extraction value as
  `action_category_pass1`. Cached on `(classify_version, model, tool_schema,
  summary_prompt)`. Failed/skipped rows keep `action_category_pass1` and are
  NOT stamped with a version — a partially classified output can never
  masquerade as fully classified.

Every quote in `evidence` / `significance_thresholds` / `referenced_ce_citations`
/ `ce_development_language` is resolved to its source span and verified by exact
match (after folding curly quotes/dashes/ligatures/nbsp/ellipsis to ASCII) — this
is the `evidence_cited` / `n_verified_quotes` / `verified_quote_rate` machinery.
`confidence_score` is a computed `0.6*verified_quote_rate + 0.4*field_fill_rate`,
independent of the model's own `extraction_confidence` self-rating.

### `10_action_label.py` — action-verb labeling (standalone, billable)

Reuses the cached enrichment summary (no document re-read) to assign one
controlled verb (`new_build`, `upgrade`, `maintenance`, `decommissioning`,
`exploration`, `assessment`, `research_or_demonstration`, `manufacturing`,
`interconnection`, `land_or_row_authorization`, `other`) per FONSI, scoped
within its `tech_group` so `09` can form `tech_group__action` cells.
`is_codifiable` is derived deterministically from the verb — `manufacturing`
(a factory) and `land_or_row_authorization` (administrative) are not physical
actions a CE can codify; every other verb (including `other`, treated as
"physical action unknown, keep in the grid") is codifiable. Failed/skipped
calls fall back to `action='other', is_codifiable=True`, unstamped.

### `04_base_rates_and_ce.py` — base rates (legacy) + CE comparison (grid)

**Base rates (legacy-5, unchanged):** three explicit counts per legacy
candidate — full clean candidate universe by process_type (CE/EA/EIS),
candidate EA projects, observed EA-source FONSI projects.

**CE comparison (refactored to the grid):** builds `tech_group__action` cells
directly from `fonsi_enrichment.parquet` x `fonsi_action_labels.parquet` (not
from `candidate_corpus`), restricted to `is_bounded_low_impact` (falling back
to the full cell if none are bounded). For each cell, scores every CE in
`ce.json` by the **median cosine similarity over the cell's member action
summaries** (not a single pooled query) — this avoids long-query dilution and
means the top CE is the one the cell's members most *consistently* match, not
just the closest on average. Top 8 CEs retained per cell, all
`manual_verification_status = pending`; numeric bounds parsed via `bounds.py`.

### `05_mitigation_and_boundary.py` — Track B (legacy-5, deterministic)

Unchanged dual-signal mitigated-FONSI flag (textual cue in `finding_text`/
`boundary_text` + enforceable `fonsi_conditions.parquet` rows) and
boundary/conditional-language extraction, keyed on the legacy 5 candidates.
Superseded for the report by `09`'s LLM-backed `candidate_mitigation_summary`,
but kept as an independent deterministic cross-check.

### `06_ce_landscape.py` — Track C (unchanged)

Embeds all 2,105 CEs in `ce.json`, finds each CE's nearest CE in a different
agency unit (cosine ≥ 0.85 → cross-agency near-duplicate), reports per-agency
counts and numeric-bound distributions.

### `09_wire_enrichment.py` — wire the LLM enrichment into the grid

The pivot script: joins `fonsi_enrichment.parquet` to `fonsi_action_labels.parquet`
on `project_id`, forms `candidate_category = f"{tech_group}__{action}"`, and
computes **Rule B ("CE-shaped")**: `is_bounded_low_impact` (the LLM read) AND —
for Transmission cells only — a shape gate requiring `action == "upgrade"`,
`within_existing_row == True`, and `new_access_road != True`; every other
tech_group's cells gate on the LLM bounded judgment alone. Overwrites
`candidate_facts.parquet` (451 rows, one per enriched FONSI — no drop-to-other)
and `candidate_mitigation_summary.parquet` (grid-keyed rollups of
`is_mitigated_fonsi`, top resource areas, and boundary statements pulled from
`significance_thresholds`). Merges D4's authoritative `decision_date` for the
post-FRA tabulation. Also writes `corpus_mitigation_stats.parquet` (corpus-wide
mitigated share, split by `case_specific_dependent` vs. `design_feature_only`/
`none` dependence).

### `07_classify_and_rank.py` — verdict + rank (grid, as-built)

Operates on `candidate_facts["candidate_category"]` — the 52 grid cells, not
the legacy 5. For each cell: `is_bounded = is_ce_shaped` (from `09`); the
"focus" set is the CE-shaped subset (falls back to the full cell if none are
CE-shaped). Verdict logic (priority order):

| Verdict | Condition |
|---|---|
| `new` | best CE match score < 0.40 (`MATCH_THRESHOLD`) — no real CE match |
| `expand` | matched CE + FONSI numeric values exceed the CE's parsed bound (≥2 projects or ≥10% of the focus set, whichever is larger) |
| `adopt` | matched CE + our FONSI agencies not covered by the CE's agency unit |
| `already_covered` | matched, within bounds, same agency |

**A1/#37 — eCFR coverage gate (`apply_coverage_gate()`).** After the deterministic verdict pass and
**before** G1 tiering, 07 reads `candidate_ce_coverage.parquet` (the reviewer's per-CE eCFR adjudication,
built by `ce_ecfr_verify.py` + `ce_ecfr_apply_verdicts.py`) and gates each adopt/expand cell on its
**cell-best** coverage (strongest of the top-5: covers > partially_covers > unclear > does_not_cover):
`does_not_cover` **flips** the cell to `new` (rank_score re-derived to develop novelty); `covers` sets
`verdict_confidence="verified"`; `partially_covers` → `"partial"`; `unclear` sets `needs_review=True`.
Existence-guarded (missing/unfilled coverage → deterministic verdicts unchanged, loud warning). Adds
`cell_best_coverage`, `coverage_source`, `needs_review`. As-built outcome: covers 10 / partial 12 /
does_not_cover 1 (Hydropower__new_build flip) / unclear 1 → adopt 22→21, new 16→17.

**G1 — client develop-shortlist recurrence gate.** Every `new` + codifiable
cell gets a `shortlist_tier`: `main` (≥5 CE-shaped FONSIs, `SHORTLIST_R_MAIN`),
`exploratory` (3-4, `SHORTLIST_R_EXPLORATORY`), or `dropped` (<3, kept in
`candidate_verdicts.parquet` for grid coloring but excluded from
`d6_new.csv`). `d6_new.csv` (the client-facing shortlist) shows only
`main`+`exploratory` tiers; drops are logged to stdout with their CE-shaped
count.

**#38 — annotate-only agency crosswalk.** For each cell's adopt gap, computes
`adopt_targets` (raw agencies not covered by the matched CE's own agency unit)
and two annotation columns via `ce_agency_crosswalk.is_covered()`:
`adopt_targets_gross` (= `adopt_targets`, unchanged) and `adopt_targets_net`
(= `adopt_targets_gross` minus any agency whose parent department or a
department sibling already holds an equivalent CE, checked across the matched
CE's top-8 retrieved ranks, not just rank 1). **This never changes `verdict`**
— it is an annotation the eCFR pass (`A1`/`#37`) is the authoritative source
for reclassifying adopt → already_covered.

**A3 — systematic rank-sensitivity.** `rank_sensitivity()` recovers each
cell's raw (unweighted) rank components from the stored weighted contributions
(`rank_novelty`, `rank_volume`, ..., stored as `weight * raw`), then runs (a) a
2,000-draw Dirichlet sweep over the 6-component simplex to get each reportable
cell's rank distribution (median/p25/p75/best/worst/`pct_top3`), and (b) a
one-at-a-time ±50% perturbation of each weight. Reportable cells = non-contrast
with ≥2 CE-shaped FONSIs. Replaces an earlier informal 3-weighting table with
a systematic sensitivity result.

`rank_score` remains the same 6-factor weighted sum as before (novelty 0.30,
volume 0.20, diversity 0.15, limit-availability 0.15, `1 - mitigated_share`
0.10, profile-role 0.10), now computed over the CE-shaped focus set of each
grid cell rather than the legacy candidate's profile subtype.

### `11_expand_analysis.py` — `#39` generalized expand

Generalizes `07`'s transmission-specific expand test to every grid cell with a
matched CE (rank-1 in `candidate_ce_comparison`) and a stated numeric bound:
for each (cell, metric) pair, compares the CE-shaped FONSIs' full size
distribution against the CE's bound and reports `n_exceeding`,
`pct_exceeding`, and a `suggested_cap` (the 90th percentile of observed
values). This is a superset of `07`'s expand verdict — it surfaces near-miss
and marginal-exceedance cells `07` doesn't flag as `expand`.

### `12_other_action_themes.py` — `#40` within-cell 'other' clustering

The 11-verb vocabulary can't resolve every FONSI; 92 land in `action=='other'`.
This script embeds each one's `action_label_freeform` + `potential_ce_theme` +
truncated `action_summary` with `all-MiniLM-L6-v2`, picks k by silhouette score
(KMeans, k in [3, 8)), and labels each cluster with its top TF-IDF n-grams —
purely to surface sub-themes for human review. **Explicitly supplementary:**
it asserts `candidate_verdicts.parquet`'s SHA-256 is unchanged before and after
the run, and exits early (code 0) if `sentence-transformers` is unavailable.

### `13_postfra_refresh.py` — `A2` post-FRA tabulation

The corpus-answerable slice of "does CE-adoption guidance need refreshing given
the Fiscal Responsibility Act": per grid cell, counts CE-shaped FONSIs with
`decision_date` (D4's authoritative date, merged in `09`) after vs. before vs.
missing relative to the FRA cut date (2023-06-03, matching D4/D5). Explicitly
flags that current CE-adoption usage / agency implementation guidance needs
EXTERNAL sources not in NEPATEC 2.0 — not attempted here — and that NEPATEC's
2024-2025 ingestion lag means a low post-cut count is not evidence of low
current activity.

### `14_threshold_retrieval.py` — `#44` threshold-language retrieval

Deterministic regex retrieval (no LLM) over `finding`/`condition`/`resource`
evidence spans (`span_type == 'boundary'` is nearly empty, ~18 rows) for six
threshold-phrase patterns ("would be significant if", "would require an EIS",
"not to exceed", "no new access road", "within existing right-of-way",
"extraordinary circumstance"). DuckDB pushes the row selection down via an
`ILIKE`-friendly WHERE clause before the regex match/snippet extraction runs in
pandas.

### `ce_agency_crosswalk.py` — `#38` department crosswalk

A small, hand-curated `DEPT_MEMBERS` dict (DOI → BLM/BOR/NPS/USFWS/BIA/BOEM;
DOE → PMA/NNSA/WAPA/BPA/SWPA/SEPA; USDA → USFS; DOD → USACE) — the same real
federal-bureau token set already in `07`'s `OUR_AGENCY_ALIASES`, none invented.
`is_covered(token, ce_agency_tokens)` returns True if the token, its
department, or a department sibling is among the CE-holding tokens.

### `ce_ecfr_verify.py` — `A1`/`#37` eCFR verification scaffold

For the 24 adopt/expand cells, pulls the top-5 retrieved CEs
(`candidate_ce_comparison`) and fetches each one's canonical eCFR text via the
eCFR renderer API (`$0`, cached to `data/raw/deliverable06/ecfr/`, 0.4s
politeness delay). Classifies each source URL as `ecfr_current` (fetchable),
`ecfr_legacy` (cgi-bin node, manual fetch), or `agency_doc` (a CE that lives in
an agency's own NEPA-procedures document, not the eCFR — itself a finding).
Writes `candidate_ce_coverage.parquet` with an **empty** `coverage_verdict` for
a human reviewer (or an optional billable `--llm` pass, `--dry-run` for exact
projected cost) to fill in `covers`/`partially_covers`/`does_not_cover`/
`unclear`. `07` is documented to prefer this adjudication when present and fall
back to the rank-1 text-similarity match otherwise — this is the "one
remaining verification step" the report calls out: every adopt/expand verdict
is currently a text-similarity match, not confirmed eCFR coverage.

### `retag_condition_resources.py` — `#47` condition resource re-tag (standalone, billable)

Rebuilds `fonsi_conditions.parquet`'s `resource_area` column in place — this
is a D2-facing fix (D2's mitigation join reads this field for resource-level
F1; D6's own verdicts/mitigation-share never read it). Two tiers:
**Tier-1** (free; **disabled by default since 2026-07-22**, commit 82d47e9): rows the
keyword-dict tagger left `unknown` inherit their resource area from the section HEADING via
`mitigation_conditions.classify_resource_area_with_heading` (reused from
`code/extract/`). Gold validation measured Tier-1 precision at 0.20, so the current shipped
`fonsi_conditions.parquet` (rebuilt 2026-07-22) carries no Tier-1 labels; re-enable
explicitly with `--use-tier1`. The combined new-tag pipeline scores F1 0.831 vs the
keyword-baseline 0.397 (see `notes/deliverable06/retag_validation_score.md`).
**Tier-2** (billable Haiku, deduped): only
`mitigation_commitment` rows (the ones feeding D2's join) get a scoped
multi-label pass over the shared-12 resource vocabulary, deduped by
condition-text SHA-256 (~11,246 unique calls covering ~14,072 rows; cached, so
re-runs are $0). The output vocabulary is strictly the shared 12 areas +
`unknown` — `vegetation` (an enrichment-only value) is aliased to `biological`
so it never leaks in and breaks D2's `RESOURCE_CROSSWALK` lookup. `~$4.23` for
a full Tier-2 run.

### `08_create_figures.R` — report figures

Reads `candidate_verdicts`, `candidate_mitigation_summary`, `candidate_facts`,
`candidate_corpus`, `ce_landscape_ces`, and `fonsi_enrichment` parquets.
Produces ~16 PNGs with `theme_catf` (CATF navy/blue palette) at 300 dpi.

---

## QA

`qa_deliverable06.py` asserts 25 invariants (20 `check()` call sites; one
site — schema-column presence — loops over 6 required columns, each counted
separately) after the chain (`10_action_label` → `04` → `05` → `06` → `09` →
`07` → `08`). All 25 currently PASS. Key invariants:

- **Grid integrity**: `candidate_category == f"{tech_group}__{action}"` for
  every `candidate_facts` row; every enriched FONSI (451) lands in a cell
  (no drop-to-other); `is_codifiable` matches the deterministic verb rule
  exactly (0 violations either direction).
- **Coverage matches the report denominators**: enrichment 451/451,
  Stage-2 `classification_parse_ok` 451/451, every enriched FONSI has an
  action label.
- **G1**: no cell with `n_profile_fonsi < 3` in `d6_new.csv`; tiers are a
  subset of `{main, exploratory}`; every `main`-tier cell has ≥5 CE-shaped
  FONSIs.
- **`#38`**: `adopt_targets_net` is a subset of `adopt_targets_gross` for
  every cell; the adopt verdict *count* is unchanged (`== 22`, the pre-eCFR-gate
  baseline at which this check runs; the later `A1`/`#37` gate moves it to 21) —
  confirms the crosswalk is annotate-only, not verdict-altering.
- **`#40`**: `other_action_themes.parquet` covers exactly the 92
  `action=='other'` projects — no more, no fewer.
- **`#47`**: `fonsi_conditions.resource_area` stays within the shared-12 +
  `unknown` enum (no stray `vegetation` or other value that would break D2's
  crosswalk join).
- **Quote verification**: ≥90% verified-quote rate on the CE-shaped subset
  (currently 96.7%).
- **Client shortlist hygiene**: zero non-codifiable cells in `d6_new.csv`.

Run: `CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable06/qa_deliverable06.py`

---

## Run Results

<!-- d6-run-results: pull this section into the D6 report -->

Confirmed from the current pipeline outputs (QA gate: 25/25 PASS).

**Enrichment** (`fonsi_enrichment.parquet`): 452 clean FONSIs, 451 successfully
enriched (1 skipped for no evidence). Stage-2 classification parsed cleanly on
all 451 (`classification_parse_ok == 451`). Quote verification on the CE-shaped
subset: 96.7%.

**Grid cells** (`candidate_facts.parquet`): 451 rows (one per enriched FONSI),
spanning 52 distinct `tech_group__action` cells. 215 of 451 rows are
`is_ce_shaped` (Rule B: LLM-bounded +, for Transmission, the upgrade/within-ROW/
no-new-road shape gate); 236 are not.

**Corpus-wide mitigation** (`corpus_mitigation_stats.parquet`): 310 of 451
enriched FONSIs (68.7%) are `is_mitigated_fonsi`. Within the mitigated set, 309
are `case_specific_dependent` and 1 is `design_feature_only`/`none`.

**Verdicts** (`candidate_verdicts.parquet`): 52 rows.

| Verdict | Count |
|---|---:|
| adopt | 21 |
| new | 17 |
| already_covered | 12 |
| expand | 2 |

*(Final post-eCFR-gate tally. The deterministic pre-gate baseline is adopt 22 / new 16; the
`A1`/`#37` coverage gate flips one cell, `Hydropower__new_build`, from `adopt` to `new` —
see the gate description above.)*

**G1 shortlist tiers** (among the 17 final `new` cells — G1 tiering runs once, on the
post-`A1`/`#37`-gate verdict set): 6 `main` (≥5 CE-shaped), 3 `exploratory` (3-4 CE-shaped),
6 `dropped` (<3, excluded from `d6_new.csv`; includes the flipped `Hydropower__new_build`
cell at n=1); 37 cells have no tier (not verdict `new`, or not codifiable). `d6_new.csv`
(the client-facing shortlist) has 9 data rows (6 main + 3 exploratory) — unaffected by the
gate flip, which landed in `dropped`.

**`#38` crosswalk**: adopt verdict count unchanged at 22 (the pre-eCFR-gate baseline at
which `#38` runs; final adopt = 21 after the gate) after computing
`adopt_targets_net`/`adopt_targets_gross` — confirms annotate-only behavior.

**`#39` expand analysis** (`expand_analysis.csv`): 37 (grid cell, metric) rows.

**`A2` post-FRA** (`postfra_recurrence.csv`): 44 grid-cell rows + 1 TOTAL row.

**`#40` other-action themes** (`other_action_themes.parquet`): 92
`action=='other'` FONSIs clustered into k=7 groups (silhouette-selected).

**`#44` threshold retrieval** (`threshold_candidates.csv`): 846 matched
phrase-instances.

**`A3` rank sensitivity** (`rank_sensitivity.csv`): 32 reportable cells
(non-contrast, ≥2 CE-shaped FONSIs), 2,000 Dirichlet draws each.

**`A1`/`#37` eCFR scaffold** (`candidate_ce_coverage.parquet`): 120 rows across
24 adopt/expand cells (top-5 CE matches each), `coverage_verdict` empty pending
reviewer/LLM adjudication.

**CE landscape** (`ce_landscape_ces.parquet`): 2,105 CEs across 78 agency units
(unchanged from the legacy run).

---

## Output Schema

### `fonsi_enrichment.parquet`

See the [Enrichment Schema](#enrichment-schema-fonsi_enrichmentparquet) section
above for the full 63-column breakdown.

### `fonsi_action_labels.parquet`

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID |
| `action` | object | One of the 11 `ACTION_VERBS` |
| `is_codifiable` | bool | Deterministic from `action` (False only for `manufacturing`, `land_or_row_authorization`) |
| `action_confidence` | object | Model's self-rated confidence |
| `actionlabel_parse_ok` | bool | Call succeeded |
| `actionlabel_cache_hit` | bool | Served from cache |
| `actionlabel_error` | object | Failure reason, if any |
| `actionlabel_prompt_version` | object | `d6_actionlabel_v1`, or `""` if unstamped (failed/skipped) |
| `actionlabel_run_at` | object | ISO-8601 UTC (this run) |
| `actionlabel_llm_run_at` | object | ISO-8601 UTC on success, else `""` |

### `candidate_facts.parquet` (grid, as overwritten by `09`)

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID |
| `candidate_category` | object | `f"{tech_group}__{action}"` grid cell id |
| `candidate_label` | object | `f"{tech_group} — {action}"` |
| `tech_group` | object | NEPATEC tech group |
| `action` | object | Action verb from `10` |
| `is_codifiable` | bool | Deterministic from `action` |
| `subtype` | object | = `action` (kept for legacy column compatibility) |
| `is_profile_subtype` | bool | Always True (every enriched FONSI is in scope now) |
| `is_bounded_low_impact` | bool | LLM read, carried from enrichment |
| `is_ce_shaped` | bool | Rule B: the CE-shaped gate consumed by `07`/`11`/`13` |
| `candidate_role` | object | Always `profile` |
| `action_category`, `action_category_pass1`, `classification_confidence` | object | Legacy 5-category reference fields carried from enrichment |
| `project_title`, `project_type`, `lead_agency_harmonized`, `project_state` | object | Metadata passthrough |
| `is_fonsi` | bool | Always True |
| `action_definition` | object | Truncated `action_summary` (400 chars) |
| `max_acres`, `max_acres_any` | float64 | `disturbance_acres` |
| `acres_basis` | object | `llm_disturbance` or `none` |
| `max_miles` | float64 | `line_miles` |
| `max_megawatts` | float64 | `capacity_mw` |
| `max_kilovolts` | float64 | `voltage_kv` |
| `n_wells` | float64 | `well_count` |
| `within_existing_row` | bool | From enrichment |
| `no_new_access_road` | bool | `new_access_road is False` |
| `previously_disturbed_land`, `is_temporary` | bool | From enrichment |
| `has_sensitive_resource` | bool | True if `extraordinary_circumstances` non-empty |
| `extraordinary_circumstances` | object | Truncated (200 chars) |
| `mitigation_dependence`, `mitigation_summary` | object | From enrichment (summary truncated 500 chars) |
| `mitigation_resource_areas` | object | Comma-joined from the JSON array |
| `finding_rationale`, `quoted_span` | object | Verified action-quote snippet (truncated 300 / 900 chars) |
| `citation_document_id`, `citation_document_role`, `citation_evidence_span_id`, `citation_page`, `citation_verified`, `citation_claim` | mixed | Provenance of the verified action citation |
| `extraction_method` | object | `llm_enrichment` |
| `confidence` | object | `extraction_confidence` from enrichment, or `medium` |
| `llm_provider`, `llm_model`, `schema_version`, `taxonomy_version` | object | Audit |
| `candidate_extraction_run_at` | object | From enrichment's `enrichment_extraction_run_at` |
| `candidate_llm_run_at` | object | From enrichment's `enrichment_llm_run_at` |
| `decision_date` | datetime | Merged from D4 `timeline_project_dates.parquet` (null if not found) |

### `candidate_verdicts.parquet` (grid, `07`)

| Column | Type | Description |
|---|---|---|
| `candidate_category` | object | Grid cell id |
| `candidate_label`, `tech_group`, `action` | object | Cell identity |
| `is_codifiable` | bool | From `candidate_facts` |
| `role` | object | Always `profile` |
| `verdict` | object | `new` / `expand` / `adopt` / `already_covered` |
| `rank_score` | float64 | Multi-factor score (0-1) |
| `rank_novelty`, `rank_volume`, `rank_diversity`, `rank_limits`, `rank_mitigation`, `rank_role` | float64 | Weighted contributions (stack to `rank_score`); raw components recoverable by dividing by `RANK_WEIGHTS` |
| `n_profile_fonsi` | int64 | CE-shaped (bounded) FONSI count |
| `n_observed_fonsi` | int64 | Total FONSIs in the cell |
| `best_ce_structured_id`, `best_ce_agency`, `best_ce_match_score`, `best_ce_description`, `best_ce_url` | mixed | Top CE match |
| `expand_gaps` | object | JSON list of `{metric, ce_bound, our_max, n_exceeding}` |
| `adopt_targets` | object | Comma-separated agency tokens missing the CE (same as `adopt_targets_gross`) |
| `adopt_targets_gross` | object | `#38`: raw adopt gap |
| `adopt_targets_net` | object | `#38`: gap after removing dept/sibling-covered agencies (annotation only) |
| `our_agencies` | object | Comma-separated agency tokens in the cell's CE-shaped subset |
| `n_agencies`, `n_states` | int64 | Distinct agency/state counts (CE-shaped subset) |
| `mitigated_share` | float64 | From `candidate_mitigation_summary` |
| `verdict_confidence` | object | `low` (deterministic match + bound parse; eCFR verification pending) |
| `shortlist_tier` | object | G1: `main` / `exploratory` / `dropped` / `""` (only set for codifiable `new` cells) |
| `taxonomy_version`, `run_at` | object | Audit |

### `other_action_themes.parquet`

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID (one of the 92 `action=='other'` projects) |
| `tech_group` | object | NEPATEC tech group |
| `cluster_id` | int64 | KMeans cluster (k chosen by silhouette, 3-7) |
| `cluster_label` | object | Top-5 TF-IDF n-grams for the cluster |
| `other_action_extraction_run_at` | object | ISO-8601 UTC |
| `other_action_llm_run_at` | object | Always `""` (no LLM call in this script) |

### `corpus_mitigation_stats.parquet`

| Column | Type | Description |
|---|---|---|
| `n_clean_fonsi`, `n_with_packet` | int64 | 451 |
| `n_mitigated_fonsi` | int64 | Count with `is_mitigated_fonsi == True` |
| `mitigated_share` | float64 | `n_mitigated_fonsi / n_clean_fonsi` |
| `n_case_specific_dependent` | int64 | Mitigated FONSIs where `mitigation_dependence == case_specific_dependent` |
| `n_design_or_none` | int64 | Mitigated FONSIs where dependence is `design_feature_only` or `none` |
| `run_at` | object | ISO-8601 UTC |

### `ce_landscape_ces.parquet`

Unchanged — see prior schema (2,105 CEs, cross-agency near-duplicate links,
parsed numeric bounds, agency unit).

---

## Key Design Decisions

- **Narrow-first, then exhaustive within scope.** The legacy taxonomy picked
  candidates up front; the grid refactor keeps the "deep-extract, small N,
  verify" spirit but makes the categorization exhaustive (every enriched
  FONSI lands in exactly one grid cell) rather than hand-picked, so nothing is
  silently dropped before the report.
- **One enrichment pass, many consumers.** `03_enrich_llm.py` reads each FONSI
  ONCE and populates every field both Analysis 1 (action/scale/siting) and
  Analysis 2 (significance/mitigation) need — deliberately avoiding N separate
  narrow LLM passes.
- **Deterministic verb → deterministic codifiability.** `is_codifiable` is a
  pure function of the action verb (not an LLM judgment call), so the client
  shortlist's "is this even the kind of thing a CE can codify" gate is fully
  auditable.
- **G1 recurrence gate protects the client shortlist, not the analysis.**
  Low-recurrence `new` cells stay in `candidate_verdicts.parquet` (visible in
  the report's full grid) but are excluded from `d6_new.csv` — the client
  never sees a "develop this CE" recommendation backed by <3 CE-shaped FONSIs.
- **`#38` crosswalk is annotate-only by design.** Whether an agency gap is
  "real" ultimately depends on eCFR text, not a department org chart; the
  crosswalk narrows the adopt list for review without pre-empting the
  authoritative `A1`/`#37` eCFR adjudication.
- **Reuse, don't rebuild.** The corpus inventory and EA+FONSI section
  extraction are read-only inputs from v1/v2. The existing-CE source is the
  committed `ce.json` (CE Explorer) — no live fetch, no parquet snapshot.
  `retag_condition_resources.py` (`#47`) reuses `code/extract/
  mitigation_conditions.py`'s heading classifier for its free Tier-1 (now
  disabled by default — see Known Issues).
- **Provenance throughout.** Every enriched fact carries a verified quote +
  span/document/page reference; CE matches are ranking aids left pending
  eCFR verification; audit timestamps (`*_extraction_run_at` always,
  `*_llm_run_at` only on success, else `""`) are consistent across every
  script in the pipeline, including the newest ones (`11`-`14`).

---

## Model Selection & Cost

**Enrichment (`03_enrich_llm.py`, the big one):** claude-sonnet-4-6, ~5,000 in
/ 1,700 out tok per call, 452 FONSIs → the production run this doc reflects.
Stage 2 (classify) is ~1,340 in / 140 out tok per call, ~10x cheaper.

**`#47` re-tag (`retag_condition_resources.py`):** claude-haiku-4-5, scoped to
`mitigation_commitment` rows only, deduped by condition-text hash (~11,246
unique calls of ~14,072 rows), ~$4.23 for a full run.

**`A1`/`#37` eCFR adjudication (`ce_ecfr_verify.py --llm`):** optional, not yet
run — `--llm --dry-run` prints the exact projected cost from the actual fetched
CE-description token counts before any spend.

Pricing reference (claude-api skill table; verify before relying on it):

| Model | Input $/1M | Output $/1M |
|---|---:|---:|
| `claude-haiku-4-5` | $1 | $5 |
| `claude-sonnet-4-6` | $3 | $15 |
| `claude-opus-4-8` | $5 | $25 |

`benchmark_models.py` runs the production enrichment prompt through Haiku/
Sonnet/Opus on a sample before committing to a model for a full run — this is
how `claude-sonnet-4-6` was selected as the enrichment default.

---

## Known Issues and Cautions

- **`#47` Tier-1 heading inheritance is disabled by default (2026-07-22, commit 82d47e9).**
  Gold validation against the retag answer key measured Tier-1 precision at 0.20 — heading
  inheritance was mis-assigning resource areas at scale — so `retag_condition_resources.py`
  now skips it unless `--use-tier1` is passed, and the shipped `fonsi_conditions.parquet`
  contains no Tier-1 labels. Combined new-tag F1 is 0.831 vs the keyword-baseline 0.397
  (`notes/deliverable06/retag_validation_score.md`).
- **Two categorization schemes coexist on disk.** `candidate_corpus.parquet`,
  `candidate_base_rates.parquet`, `candidate_descriptive.parquet`, and
  `candidate_mitigation_boundary.parquet` are still legacy-5-keyed;
  everything downstream of `09` (facts, mitigation summary, CE comparison,
  verdicts, all report tables) is grid-keyed. Don't join a legacy-5-keyed
  parquet to a grid-keyed one on `candidate_category` — the values are from
  different vocabularies.
- **Verdicts are text-similarity matches pending eCFR verification.** Every
  `adopt`/`expand`/`already_covered` verdict rests on the CE with the highest
  median-cosine match to the cell's action summaries — a *ranking aid*, not
  confirmed legal coverage. `ce_ecfr_verify.py` builds the adjudication
  scaffold (`candidate_ce_coverage.parquet`) but `coverage_verdict` is
  currently empty (not yet reviewed or LLM-adjudicated); `07` still falls back
  to the rank-1 text-similarity match.
- **`#38`'s crosswalk is a heuristic on agency ORG STRUCTURE, not CE
  applicability.** A department sibling holding an equivalent CE doesn't
  guarantee the specific CE at issue extends legally to the "covered" agency;
  `adopt_targets_net` is an annotation to narrow review, not a verdict change
  (enforced by QA check `#38`).
- **`#39`'s suggested caps are descriptive, not prescriptive.** The p90-based
  `suggested_cap` in `expand_analysis.csv` is a starting point for a policy
  conversation about where a raised CE bound might sit, not a recommendation
  that the cap should literally be that value.
- **Post-FRA counts understate current activity.** `postfra_recurrence.csv`'s
  low post-2023-06-03 counts partly reflect NEPATEC 2.0's 2024-2025 ingestion
  lag, not necessarily low actual post-FRA CE-development activity — flagged
  explicitly in the script's `CAVEAT` string and carried into the CSV.
- **`#40`'s clustering is exploratory, not authoritative.** k is chosen by
  silhouette score over a small candidate range (3-7) on only 92 documents;
  cluster labels are automated TF-IDF n-grams, not human-reviewed category
  names.
- **`retag_condition_resources.py` (`#47`) changes D2's inputs, not D6's
  outputs.** No D6 headline number (mitigated-FONSI share, `mitigation_dependence`
  distribution, verdicts) reads `fonsi_conditions.resource_area`; this script
  exists purely to improve D2's resource-level join quality.
- **`ce_crosswalk.parquet` and `ce_explorer_snapshot.parquet`** in the analysis
  folder are v1 artifacts kept for provenance; not read by any current script.
  The canonical CE source is `ce.json` via `ce_source.py`.

---

## Methodological Notes

**Why narrow-first, then a fully enumerated grid?** The original hand-picked
5-category taxonomy risked missing recurring CE-shaped classes outside the
analyst's initial guesses. The refactor keeps the deep-extraction discipline
(one careful enrichment call per FONSI, verified quotes) but replaces the
hand-picked category list with an exhaustive `tech_group x action` grid, so
nothing is silently excluded before the verdict stage — the 16 pre-eCFR-gate `new`-verdict
cells the current run surfaces (17 final after the gate; vs. 0 under the legacy 5-category
run) is direct evidence this mattered.

**Why a two-stage (extract-then-classify) enrichment instead of one call?**
The extraction call is expensive (large evidence packet in, many fields out)
and its `action_category` field is a coarse first pass without real category
definitions in front of the model. Re-asking `action_category` alone, cheaply,
with an enum tool schema and explicit definitions, measurably improves
classification accuracy without re-paying for the whole extraction — and lets
a classifier fix (bumping `CLASSIFICATION_PROMPT_VERSION`) be applied for
~$1.40 instead of a full ~$3+ re-extraction.

**Why deterministic action-verb labeling in a separate step (`10`) rather than
inside the enrichment schema?** The action verb needs to be a controlled,
enum-constrained vocabulary that can be evolved independently of the (larger,
more expensive) enrichment schema, and `is_codifiable` needs to be a pure
function of the verb — not a judgment the LLM can second-guess — so the
client-facing "can this even become a CE" gate stays fully auditable.

**Why the median-cosine-over-members CE match instead of a single pooled
query per cell?** A single long query string (concatenating many members'
descriptions) dilutes toward the corpus's semantic center. Scoring each CE
against every cell member individually and taking the median means the winning
CE is the one members most *consistently* resemble — closer to how a human
reviewer would judge "does this CE fit the pattern" than a single blended
embedding.

**Why G1's recurrence floor (3 CE-shaped minimum) instead of showing every
`new` cell?** A CE recommendation backed by 1-2 FONSIs is not defensible
recurrence evidence — it could be an idiosyncratic project, not a pattern.
Kept dropped cells in `candidate_verdicts.parquet` (not deleted) so the full
grid is still auditable, but excluded from the client-facing `d6_new.csv` so
the develop shortlist only shows cells with real recurrence.

**Why is `#38`'s crosswalk annotate-only rather than verdict-changing?**
Department structure is a *plausible* proxy for CE applicability (agencies
under the same department often share NEPA implementing procedures) but it is
not dispositive — only the eCFR text of the specific CE determines whether it
legally extends to a sibling agency. Changing verdicts on an org-chart
heuristic would risk false "already_covered" reclassifications; the QA gate
(`#38` check, `n_adopt == 22` at the pre-eCFR-gate stage where it runs; the
`A1`/`#37` gate is the only step that later moves a verdict, 22→21) enforces
that this script never moves a verdict.

**Why `all-MiniLM-L6-v2` throughout (CE ranking, `12`'s theme clustering)?**
Fast, local, no API key, and its cosine geometry is well-suited to
phrase/sentence-level similarity at the scale needed here (52 grid cells x 8 CE
ranks, or 92 FONSIs for `12`) — reserving the LLM budget for the extraction and
classification calls where a general-purpose model's judgment is actually
needed.

---

## Analysis 3 — CE cluster topics (k-means, k = 8, deterministic)

The existing-CE t-SNE scatter groups the 2,105 federal CEs into 8 k-means families. The clustering
is **deterministic** (`06_ce_landscape.py`: stable sort of the CE load + cached embeddings, so
`cluster_km` is stable across runs). The silhouette is **low and flat (~0.035 at every k)**, so
**k = 8 is a readability choice, not a natural optimum** — several families are genuinely mixed.
Cluster labels come from distinctive n-gram phrases; the human-facing **Topic** labels are curated
in `08_create_figures.R` (`CE_TOPICS`, keyed on `cluster_km`) and shown on the scatter + the report family
table. Revisit `CE_TOPICS` if the clustering ever changes.

| cluster_km | Keywords (what's in it) | Topic | Coherence |
|---|---|---|---|
| 0 | leases, easements, licenses, permits, real property | Property leases, licenses, and permits | clean |
| 1 | geological / geophysical surveys, site assessments, data collection | Geological surveys and site assessments | clean |
| 4 | procurement, supportive / health / housing services, personnel | Goods, services, and personnel procurement | clean |
| 5 | rules, safety standards, product certification, labeling | Rules, standards, and guidance | clean |
| 2 | routine facility maintenance, groundskeeping, dredging | Routine maintenance and minor ground work | mixed |
| 3 | disposal of property / fixtures / structures, hazardous materials | Hazmat and disposal | mixed |
| 6 | monitoring equipment, rights-of-way, resident relocations | Monitoring and rights-of-way | mixed |
| 7 | FAA airport layout plans, equipment installation, surveillance | Airport layout plans and monitoring equipment | mixed |

---

## Reproduction

```bash
# 1. Prerequisite (STANDALONE, BILLABLE, run once — cached, re-runs are $0):
conda run -n nepa python phase2/code/deliverable06/03_enrich_llm.py --dry-run   # cost preview
conda run -n nepa python phase2/code/deliverable06/03_enrich_llm.py --workers 4
conda run -n nepa python phase2/code/deliverable06/10_action_label.py --dry-run
conda run -n nepa python phase2/code/deliverable06/10_action_label.py --workers 4

# 2. Optional $0 scaffold (safe to re-run):
conda run -n nepa python phase2/code/deliverable06/ce_ecfr_verify.py

# 3. Optional D2-facing fix (STANDALONE, BILLABLE, ~$4.23):
conda run -n nepa python phase2/code/deliverable06/retag_condition_resources.py --dry-run
conda run -n nepa python phase2/code/deliverable06/retag_condition_resources.py --run --workers 4

# 4. The chain (01 -> 09 -> 07 -> 11 -> 12 -> 13 -> 14, then 08 R figures):
conda run -n nepa python phase2/code/deliverable06/_run.py

# 5. QA gate:
conda run -n nepa python phase2/code/deliverable06/qa_deliverable06.py

# render the CE catalog .md (after ce.json updates)
conda run -n nepa python phase2/code/deliverable06/extract_ce_catalog.py

# render the report
quarto render phase2/reports/deliverable06.qmd
```
