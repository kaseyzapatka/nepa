# D6: Patterns in FONSIs — Architecture (v2, Narrow-First)

> ⚠️ **PARTIALLY SUPERSEDED (2026-06).** Sections describing the LLM pass as gated / not
> wired, and the run-results section reporting an older verdict mix (incl. a stale `expand`
> verdict), no longer match the shipped pipeline: the **one-pass LLM enrichment
> (`03_enrich_llm.py`, claude-sonnet-4-6, schema d6_enrich_schema_v5) is now wired in via
> `09_wire_enrichment.py`**, and current `candidate_verdicts.parquet` has **4 adopt / 1
> contrast**. Do not use this file as implementation truth until it is rebuilt; the
> authoritative description is the report (`phase2/reports/deliverable06.qmd`) and the
> review log in `phase2/code/deliverable06/feedback.md`.

**Goal:** Identify a small, defensible shortlist of recurring clean-energy action
categories in prior EAs/FONSIs that may warrant CATF, policy, and legal review
for new or expanded categorical exclusions (CEs). For each candidate: a crisp
action definition, evidence it recurs with no significant impact, the recurring
bounding limits, whether findings depend on case-specific mitigation, whether an
existing CE already covers it, and traceable citations.

**Self-contained:** Partially. Reads `projects_combined.parquet` and the D3
`projects_nepa_reviews.parquet` / `ce_citations.parquet` for base-rate
denominators and CE-use evidence. All FONSI inventory and packet inputs are
pre-built parquets under `data/analysis/deliverable06/`.

**Status:** v2 narrow-first pipeline built and running (deterministic pass, 01–08
chain). Tracks B/C + verdicts + figures + report all in. The LLM extraction pass
is wired but gated (Gate 3; requires `--use-llm`). Forward plan:
`phase2/plans/deliverable06_updates.md`.

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
dropped as already-covered; the wind-onshore candidate serves as a contrast
baseline rather than an actionable recommendation.)*

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

## Script quick-reference

All pipeline scripts live under `phase2/code/deliverable06/`. v1 scripts are
archived to `_archived_v1/` and are not orchestrated.

### Numbered pipeline (run by `_run.py` in order 01→08)

| Script | Track | Plain-language role | What it does |
|---|---|---|---|
| `01_select_candidate_corpus.py` | A | **pick** | Assigns projects to candidate categories (tech_group + keyword rules), splits subtypes, runs the resource-assessment prevalence + de-overlap screen and the storage-deployment scan. |
| `02_assemble_candidate_evidence.py` | A | **gather** | Pulls each candidate FONSI's typed text from `fonsi_project_packets` + span-level provenance. Falls back to `fonsi_document_sections` for any missing projects. |
| `03_extract_candidate_facts.py` | A | **extract** | Deterministic: context-aware acres/miles, wells (including spelled-out numbers), siting booleans, extraordinary-circumstances mention scan, spaCy+all-MiniLM action definitions. Gated LLM pass (Gate 3) via `--use-llm`; shared prompt imported from `prompts.py`. |
| `04_base_rates_and_ce.py` | A | **contextualize** | Three base-rate counts per candidate; agency/geography descriptive breakdowns; lexical+embedding CE-Explorer crosswalk (ranking aid only, all pending verification); existing-CE numeric bound parsing via `bounds.py`. |
| `05_mitigation_and_boundary.py` | B | **mitigated-FONSI + boundary** | Dual-signal mitigated-FONSI flag (D02 cues + enforceable conditions from `fonsi_conditions.parquet`). Extracts boundary/conditional language ("would be significant if…") directly from agency text. |
| `06_ce_landscape.py` | C | **CE landscape** | Embeds `ce.json`, identifies cross-agency near-duplicate CEs (cosine ≥ 0.85), per-agency counts, numeric-bounds distribution, best-effort usage from D3 `ce_citations`. |
| `07_classify_and_rank.py` | — | **verdict + rank** | Applies verdict logic (contrast→new→expand→adopt→already_covered); computes multi-factor `rank_score`; writes the slim top-level `d6_comparison_table.csv` plus the three drill-down opportunity lists and per-candidate evidence to `review/`. |
| `08_create_figures.R` | — | **figures** | Report figures with `theme_catf` / `ggsave` to `output/deliverable06/figures/`. Reads `candidate_verdicts`, `candidate_mitigation_summary`, `candidate_base_rates` parquets + `ce_landscape_summary.csv`. |

### Standalone tools (NOT in `_run.py` chain)

| Script | Role |
|---|---|
| `_run.py` | Orchestrator: runs `01`→`07` Python then `08` via Rscript. Flags `--use-llm` (Gate 3), `--model`, `--skip-figures`. |
| `candidates.py` | Versioned candidate rulebook: category membership rules, subtype definitions, CE-story text, storage-scan config. |
| `common.py` | Shared paths/IO helpers including `D6_REVIEW_DIR` constant. |
| `ce_source.py` | Loads `notes/deliverable06/ce.json` (CE Explorer v2.0.0). |
| `bounds.py` | Parses numeric bounds (acres/miles/kV/MW/wells) from CE description text. |
| `embeddings.py` | `all-MiniLM-L6-v2` embedding wrapper (used in 03, 04, 06). |
| `prompts.py` | Shared LLM prompt for `03 --use-llm`; a standalone module (not a numbered script) so it can be imported without triggering the pipeline. |
| `extract_ce_catalog.py` | Renders `ce.json` to `notes/deliverable06/_ce_catalog_extracted.md`; run once after CE source updates. |
| `benchmark_models.py` | Model-selection tool: runs the production prompt through Haiku/Sonnet/Opus on a sample; calls the paid API; run once before `--use-llm`. |

---

## Data flow

```mermaid
flowchart TD
    R[projects_nepa_reviews.parquet<br/>clean universe + process_type] --> N1[01 pick]
    I[fonsi_project_inventory.parquet<br/>EA-source FONSIs] --> N1
    CFG[candidates.py rulebook] --> N1
    N1 --> C[candidate_corpus.parquet]
    N1 --► MR[review/candidate_membership_review.csv]
    N1 --► SR[review/candidate_storage_scan_review.csv]

    C --> N2[02 gather]
    P[fonsi_project_packets.parquet<br/>typed EA+FONSI text] --> N2
    S[fonsi_evidence_spans.parquet<br/>span ids + page + hash] --> N2
    N2 --> EV[candidate_evidence_packets.parquet]

    EV --> N3[03 extract]
    COND[fonsi_conditions.parquet<br/>condition roles/obligations] --> N3
    LLM([gated LLM pass — Gate 3<br/>--use-llm]) -.-> N3
    N3 --> F[candidate_facts.parquet]
    N3 --► ER[review/candidate_extraction_review.csv]

    C --> N4[04 contextualize]
    CE[ce.json via ce_source.py<br/>2,105 CEs / 78 agency units] --> N4
    N4 --> BR[candidate_base_rates.parquet]
    N4 --> XC[candidate_ce_comparison.parquet]
    N4 --> DESC[candidate_descriptive.parquet]
    N4 --► CR[review/candidate_ce_comparison_review.csv]

    EV --> N5[05 Track B: mitigated-FONSI + boundary]
    COND --> N5
    N5 --> MB[candidate_mitigation_boundary.parquet]
    N5 --> MS[candidate_mitigation_summary.parquet]
    N5 --► MBR[review/candidate_mitigation_boundary_review.csv]

    CE --> N6[06 Track C: CE landscape]
    N6 --> CL[ce_landscape_ces.parquet]
    N6 --► CLS[review/ce_landscape_summary.csv]

    F --> N7[07 classify + rank]
    BR --> N7
    XC --> N7
    MS --> N7
    C --> N7
    N7 --> VD[candidate_verdicts.parquet]
    N7 --► CT[d6_comparison_table.csv<br/>slim QA table — 8 cols]
    N7 --► DL[review/d6_new.csv + d6_expand.csv + d6_adopt.csv]
    N7 --► PE[review/d6_candidate_evidence_&lt;category&gt;.csv]

    VD --> N8[08 R figures]
    MS --> N8
    BR --> N8
    CLS --> N8
    N8 --► FIG[figures/fig_d6_*.png]

    FIG --> QMD[phase2/reports/deliverable06.qmd]
    VD --> QMD
    MS --> QMD
```

---

## Inputs (all read-only)

| File | Role |
|---|---|
| `deliverable06/fonsi_project_inventory.parquet` | Clean EA-source FONSI projects (canonical selection). |
| `deliverable03/projects_nepa_reviews.parquet` | Full clean universe + `process_type` for base-rate denominators. |
| `projects_combined.parquet` | Used in `01` to enrich the universe with `project_title` / `project_description` for subtype matching. |
| `deliverable06/fonsi_project_packets.parquet` | Per-project typed text (action/finding/resource/condition/boundary), drawn from EA+FONSI documents. |
| `deliverable06/fonsi_evidence_spans.parquet` | Span-level provenance: `section_id`, `evidence_span_id`, `source_span_sha256`, page. |
| `deliverable06/fonsi_conditions.parquet` | Condition roles/obligations — reused for the mitigation signal in `03` and `05`. |
| `deliverable06/fonsi_document_sections.parquet` | Fallback text source in `02` for candidate projects missing from the packets. |
| `notes/deliverable06/ce.json` | **Canonical existing-CE source** — CE Explorer export (v2.0.0, 2025-08-07), loaded via `ce_source.py`. Replaces the v1 parquet snapshot; rendered to `notes/deliverable06/_ce_catalog_extracted.md` by `extract_ce_catalog.py`. |
| `deliverable03/ce_citations.parquet` | Project-level CE-use evidence (D3); used in `04` and `06` for citation counts. |

---

## Primary Outputs

All analysis parquets are written under `phase2/data/analysis/deliverable06/`.

| File | Description |
|---|---|
| `candidate_corpus.parquet` | One row per (project, candidate_category) over the full clean universe + observed FONSIs; includes subtype, process_type, is_fonsi, is_profile_subtype. 13,145 rows. |
| `candidate_evidence_packets.parquet` | Per-project typed text + span provenance for candidate FONSIs. |
| `candidate_facts.parquet` | Per (project, candidate_category): action definition, context-aware numeric limits, siting booleans, extraordinary-circumstances scan, mitigation signal, citation. 295 rows. |
| `candidate_base_rates.parquet` | Three base-rate counts per candidate (CE/EA/EIS universe, EA projects, observed FONSIs). 5 rows. |
| `candidate_ce_comparison.parquet` | Lexical+embedding ranked CE matches, top 8 per candidate (all `manual_verification_status = pending`). 40 rows. |
| `candidate_descriptive.parquet` | Long-form per-candidate breakdowns by agency, state, and decision year. 162 rows. |
| `candidate_mitigation_boundary.parquet` | Per (project, candidate_category): mitigated-FONSI flag (dual signal), confidence, enforceable conditions count, boundary-language snippets. 295 rows. |
| `candidate_mitigation_summary.parquet` | Per-candidate rollup of mitigation share + boundary language for `07` and `08`. 5 rows. |
| `candidate_verdicts.parquet` | One row per candidate: verdict (new/expand/adopt/already_covered/contrast), rank_score, best CE match, adopt targets, expand gaps. 5 rows. |
| `ce_landscape_ces.parquet` | Per-CE landscape: cross-agency near-duplicate links (cosine), bound parsing, agency unit. 2,105 rows. |

Figures are written to `phase2/output/deliverable06/figures/`.

| File | Description |
|---|---|
| `fig_d6_verdicts.png` | Candidate categories by CE verdict, colored by new/expand/adopt/contrast. |
| `fig_d6_evidence_volume.png` | CE-shaped (profile-subtype) FONSI count per candidate. |
| `fig_d6_mitigated_share.png` | Mitigated-FONSI share per candidate (Track B). |
| `fig_d6_ce_per_agency.png` | Existing-CE count per agency, top 15 (Track C). |

The top-level client QA table and drill-down review files are in `phase2/output/deliverable06/`.

| File | Description |
|---|---|
| `d6_comparison_table.csv` | Single at-a-glance overview: candidate, verdict, CE-shaped FONSIs, existing CE, adopt targets, expand detail, rank score. |
| `review/d6_new.csv` | Full verdict rows for NEW candidates. |
| `review/d6_expand.csv` | Full verdict rows for EXPAND candidates. |
| `review/d6_adopt.csv` | Full verdict rows for ADOPT candidates. |
| `review/d6_candidate_evidence_<category>.csv` | Per-candidate project-level evidence with citations (one file per candidate). |
| `review/candidate_membership_review.csv` | Gate 2 membership QA packet (observed FONSI only). |
| `review/candidate_storage_scan_review.csv` | Non-manufacturing storage-deployment hits (Gate 2 evidence). |
| `review/candidate_extraction_review.csv` | Per-row extraction QA (key fact fields). |
| `review/candidate_mitigation_boundary_review.csv` | Per-row mitigation/boundary QA. |
| `review/candidate_ce_comparison_review.csv` | Ranked CE matches for manual verification. |
| `review/candidate_descriptive_review.csv` | Agency/state/year breakdowns. |
| `review/ce_landscape_summary.csv` | Existing-CE counts per agency unit. |

The client-facing deliverable is the rendered report at `docs/phase2/reports/deliverable06.html`
(source: `phase2/reports/deliverable06.qmd`).

---

## Module Architecture

### `01_select_candidate_corpus.py` — corpus membership

Applies `candidates.py` membership rules over two frames: the full clean-energy
project universe (`projects_nepa_reviews`) and the observed FONSI projects
(`fonsi_project_inventory`). Builds `candidate_corpus` as a union of (universe +
observed FONSIs), flagging each row `is_fonsi` and `is_profile_subtype`.

Key design decisions: transmission and solar are split into labeled subtypes by
keyword match against a per-project text blob (title + description + project_type).
Temporary resource assessment (#4) is a cross-tech, keyword-driven candidate and
is de-overlapped against geothermal_exploration to avoid double-counting. The
storage-deployment scan is Gate 2 evidence only — it writes a review CSV and does
not affect corpus membership.

### `02_assemble_candidate_evidence.py` — text + provenance

For each candidate FONSI project, retrieves typed text columns
(`action_text`, `finding_text`, `resource_text`, `condition_text`, `boundary_text`,
`analysis_text`) from `fonsi_project_packets` and attaches span-level provenance
from `fonsi_evidence_spans`, priority-ordered by span type (action first, then
finding/boundary/condition/resource). For the rare project missing from the packets,
falls back to `fonsi_document_sections` and mints a stable SHA-256 provenance hash.

### `03_extract_candidate_facts.py` — deterministic extraction + gated LLM

**Deterministic pass:**

- *Numeric limits*: regex for acres, miles, MW, kV, wells — applied with a ±70-char
  context window; for acres and miles, prefers disturbance-footprint context (grading,
  clearing, ROW) over planning-area mentions. Spelled-out well counts ("twelve wells")
  recovered via a word-to-integer map.
- *Siting booleans*: `within_existing_row`, `no_new_access_road`,
  `previously_disturbed_land` — each a regex over the assembled text.
- *Extraordinary circumstances*: mention scan for the rare CE-gating categories
  (critical habitat, ESA-listed species, wilderness, Wild and Scenic rivers,
  100-year floodplain, National Register of Historic Places, prime farmland,
  sole-source aquifer). Generic resource areas excluded because they appear in
  nearly every EA and are not discriminating.
- *Action definition*: spaCy blank sentencizer splits `action_text`; candidate
  sentences must pass length + action-verb + anti-boilerplate filters. `all-MiniLM-L6-v2`
  ranks sentences against five action-description query templates; top match becomes
  `action_definition`.
- *Mitigation signal (preliminary)*: reused from `fonsi_conditions.parquet`
  condition roles/obligations (three-way: `case_specific_dependent`,
  `design_feature_only`, `uncertain`).

**Gated LLM pass (Gate 3):** `--use-llm` enables the Anthropic call (default
`claude-sonnet-4-6`). On success, overwrites `action_definition`,
`mitigation_dependence`, `mitigation_summary` with LLM output; sets
`candidate_llm_run_at`. No-op without `ANTHROPIC_API_KEY` or the `anthropic` SDK.
Results cached to `candidate_facts_llm_cache.json` to avoid re-calls.

Numeric caps are category-specific (e.g., solar limited to 50,000 acres / 5,000 MW;
geothermal_exploration to 300 wells) to guard against implausible extractions.

### `04_base_rates_and_ce.py` — base rates + CE comparison

**Base rates:** Three explicit counts per candidate — never one ambiguous "share":
1. full clean candidate universe by process_type (CE / EA / EIS);
2. candidate EA projects;
3. observed EA-source FONSI projects.

Also computes agency/state/decision-year descriptive breakdowns for the report
(`candidate_descriptive.parquet`).

**CE comparison:** Loads `ce.json` via `ce_source.py`. For each candidate, builds
a query string and scores every CE by a blended retrieval score
(0.65 × embedding cosine + 0.35 × token overlap when embeddings are available,
lexical-only otherwise). Top 8 CEs per candidate are retained. Also parses numeric
bounds from each matched CE via `bounds.py`. All matches remain
`manual_verification_status = pending`.

### `05_mitigation_and_boundary.py` — Track B

Supersedes the coarse `mitigation_dependence` heuristic from `03`. Applies a
**dual signal**:

1. *Textual cue*: regex over `finding_text` + `boundary_text` for BLM/DOE phrasing
   ("would be significant absent…", "with incorporation of … mitigation").
2. *Enforceable conditions*: rows from `fonsi_conditions.parquet` with
   `condition_role ∈ {mitigation_commitment, enforcement_or_permit_condition}` and
   `obligation_level ∈ {required, committed}`.

Confidence is `high` (both signals), `medium` (one signal), or `none`. Also
extracts **boundary/conditional language** — the agency's own counterfactual
statements about where the significance line sits ("would be significant if X
exceeds…"). These snippets feed `07` and `08`, and are the raw material for
codifiable CE design criteria.

Rolls up to `candidate_mitigation_summary` (per-candidate profile-subset
statistics) which is the `05` output consumed by `07` and `08`.

### `06_ce_landscape.py` — Track C

An independent analysis *of the existing CE body* rather than of FONSI projects.
Embeds all 2,105 CEs in `ce.json` with `all-MiniLM-L6-v2`, then for each CE
finds its nearest CE in a *different* agency unit (cosine ≥ 0.85 threshold →
cross-agency near-duplicate). Reports per-agency CE counts and numeric-bound
distributions (how often CEs state explicit limits and their ranges).

The 317 cross-agency near-duplicates surface harmonization / adoption-consolidation
candidates — context for the ADOPT verdict.

### `07_classify_and_rank.py` — verdict + tables

Integrates all tracks. Verdict logic (evaluated in priority order):

| Verdict | Condition |
|---|---|
| `contrast` | `candidate_role == "contrast"` (wind_onshore) |
| `new` | best CE match score < 0.40 (no real CE match) |
| `expand` | matched CE + FONSI numeric values systematically exceed CE's parsed bound (≥2 projects or ≥10% of focus set) |
| `adopt` | matched CE + our FONSI agencies not covered by the CE's agency unit |
| `already_covered` | matched, within bounds, same agency |

Multi-factor `rank_score` (0–1): 30% novelty (verdict tier), 20% evidence volume,
15% geographic diversity, 15% numeric-limit availability, 10% (1 − mitigated
share), 10% profile-role bonus.

Outputs: `candidate_verdicts.parquet` (machine-to-machine), `d6_comparison_table.csv`
(slim, 8 columns), three drill-down lists (`d6_new/expand/adopt.csv`), and one
`d6_candidate_evidence_<category>.csv` per candidate.

### `08_create_figures.R` — report figures

Standard house pattern (matches D4 `08_create_figures.R`, D5 `03_create_figures.R`).
Reads `candidate_verdicts`, `candidate_mitigation_summary`, `candidate_base_rates`
parquets and `review/ce_landscape_summary.csv`. Produces four PNG figures with
`theme_catf` (CATF navy/blue palette). `ggsave` at 300 dpi to `figures/`.

---

## Run Results

<!-- d6-run-results: pull this section into the D6 report -->

Run completed: 2026-06-23 (UTC 23:15).

**Candidate corpus** (`candidate_corpus.parquet`): 13,145 rows (one per
project × candidate_category, spanning both the full clean universe and observed
FONSIs).

**Universe and FONSI counts by candidate:**

| Candidate | Universe projects | CE | EA | EIS | Observed FONSIs | Profile (CE-shaped) FONSIs |
|---|---:|---:|---:|---:|---:|---:|
| transmission_upgrade | 6,830 | 6,400 | 186 | 244 | 149 | 37 |
| wind_onshore (contrast) | 867 | 693 | 74 | 100 | 62 | 0 |
| solar | 2,245 | 2,062 | 83 | 100 | 61 | 8 |
| geothermal_exploration | 873 | 819 | 24 | 30 | 21 | 7 |
| temporary_resource_assessment | 2,330 | 2,317 | 6 | 7 | 2 | 2 |

**Candidate facts** (`candidate_facts.parquet`): 295 rows (project × category).

**Verdicts** (`candidate_verdicts.parquet`): 5 rows.

| Candidate label | Verdict | Profile FONSIs | Best CE match | Rank score |
|---|---|---:|---|---:|
| Transmission upgrades within existing ROW | expand | 37 | TVA---1-16 | 0.7098 |
| Solar (CE-shaped subset) | adopt | 8 | DOE-1--5-87 | 0.4735 |
| Temporary resource assessment / site investigation | adopt | 2 | DOE-1--3-43 | 0.4370 |
| Geothermal exploration | adopt | 7 | DOE-1--3-43 | 0.4213 |
| Wind, onshore (contrast) | contrast | 0 | DOE-1--5-89 | 0.5314 |

Verdict counts: adopt 3, expand 1, contrast 1. NEW count = 0 — all current
candidates already map to an existing CE; surfacing net-new requires the
broadened candidate generation described in the forward plan.

**Mitigation** (`candidate_mitigation_summary.parquet`):

| Candidate | Profile FONSIs | Mitigated | Mitigated share | With boundary language |
|---|---:|---:|---:|---:|
| transmission_upgrade | 37 | 32 | 0.865 | 7 |
| geothermal_exploration | 7 | 6 | 0.857 | 3 |
| solar | 8 | 6 | 0.750 | 3 |
| temporary_resource_assessment | 2 | 1 | 0.500 | 1 |
| wind_onshore | 62 | 29 | 0.468 | 13 |

Total mitigated-FONSI flags (project × category level): 178 of 295.

**CE landscape** (`ce_landscape_ces.parquet`): 2,105 CEs across 78 agency units.
Cross-agency near-duplicates (cosine ≥ 0.85): 317.

---

## Output Schema

### `candidate_corpus.parquet`

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID |
| `candidate_category` | object | One of: transmission_upgrade, geothermal_exploration, solar, temporary_resource_assessment, wind_onshore |
| `candidate_label` | object | Human-readable label |
| `candidate_role` | object | profile, contrast |
| `ce_story` | object | CE narrative for the candidate (from `candidates.py`) |
| `subtype` | object | Fine-grained action subtype |
| `is_profile_subtype` | bool | True if this subtype is CE-shaped (the extraction focus) |
| `process_type` | object | CE, EA, or EIS |
| `tech_group` | object | NEPATEC tech group |
| `is_fonsi` | bool | True if project is in the observed EA-source FONSI set |
| `project_title` | object | Project title (FONSI projects only) |
| `canonical_fonsi_document_id` | object | Canonical FONSI document ID (FONSI projects only) |
| `lead_agency_harmonized` | object | Harmonized agency name (FONSI projects only) |
| `project_state` | object | State (FONSI projects only) |
| `taxonomy_version` | object | Version tag from `candidates.py` |
| `corpus_run_at` | object | ISO-8601 UTC timestamp |
| `input_hashes` | object | JSON of SHA-256 hashes of input files |

### `candidate_facts.parquet`

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID |
| `candidate_category` | object | Candidate category |
| `candidate_label` | object | Human-readable label |
| `subtype` | object | Action subtype |
| `is_profile_subtype` | bool | CE-shaped subtype flag |
| `candidate_role` | object | profile or contrast |
| `action_definition` | object | Best action sentence (spaCy+MiniLM, or LLM if enabled) |
| `max_acres` | float64 | Disturbance-footprint acres (preferred) or any in-range acres |
| `max_acres_any` | float64 | Any in-range acres mention (fallback) |
| `acres_basis` | object | disturbance, area_or_unspecified, or none |
| `max_miles` | float64 | Disturbance-context miles (preferred) or any in-range miles |
| `max_megawatts` | float64 | Largest MW value within cap |
| `max_kilovolts` | float64 | Largest kV value within cap |
| `n_wells` | float64 | Largest well count (including spelled-out numbers) |
| `duration` | object | Regex-extracted duration string |
| `within_existing_row` | bool | Siting constraint flag |
| `no_new_access_road` | bool | No-new-road constraint flag |
| `previously_disturbed_land` | bool | Brownfield/previously-disturbed flag |
| `has_sensitive_resource` | bool | Any extraordinary-circumstance mention found |
| `extraordinary_circumstances` | object | Comma-separated CE-gating terms found |
| `mitigation_dependence` | object | case_specific_dependent, design_feature_only, uncertain, or none |
| `mitigation_summary` | object | Sample condition text |
| `mitigation_resource_areas` | object | Comma-separated resource areas from conditions |
| `finding_rationale` | object | Quoted snippet from action/finding text |
| `citation_document_id` | object | First span's document ID |
| `citation_section_id` | object | First span's section ID |
| `citation_evidence_span_id` | object | First span's evidence ID |
| `citation_page` | float64 | First span's page number |
| `quoted_span` | object | Short quoted snippet |
| `extraction_method` | object | deterministic_regex+conditions or deterministic+llm |
| `confidence` | object | low (deterministic) or medium (with LLM) |
| `llm_provider` | object | anthropic or "" |
| `llm_model` | object | Model name or "" |
| `prompt_version` | object | Prompt version string |
| `schema_version` | object | d6_facts_v1 |
| `taxonomy_version` | object | Taxonomy version |
| `candidate_extraction_run_at` | object | ISO-8601 UTC (all rows) |
| `candidate_llm_run_at` | object | ISO-8601 UTC (only on LLM success, else "") |

### `candidate_verdicts.parquet`

| Column | Type | Description |
|---|---|---|
| `candidate_category` | object | Candidate category |
| `candidate_label` | object | Human-readable label |
| `role` | object | profile or contrast |
| `verdict` | object | new, expand, adopt, already_covered, or contrast |
| `rank_score` | float64 | Multi-factor score (0–1) |
| `n_profile_fonsi` | int64 | CE-shaped FONSI project count |
| `n_observed_fonsi` | int64 | Total observed FONSI project count |
| `best_ce_structured_id` | object | Top CE match structured ID |
| `best_ce_agency` | object | Top CE match agency name |
| `best_ce_match_score` | float64 | Blended retrieval score |
| `expand_gaps` | object | JSON list of metric/bound/exceedance dicts |
| `adopt_targets` | object | Comma-separated agency tokens missing the CE |
| `our_agencies` | object | Comma-separated agency tokens in our FONSI set |
| `n_agencies` | int64 | Distinct FONSI agency count (profile subset) |
| `n_states` | int64 | Distinct FONSI state count (profile subset) |
| `mitigated_share` | float64 | Share of profile FONSIs flagged as mitigated |
| `best_ce_description` | object | Truncated CE description (200 chars) |
| `best_ce_url` | object | Canonical CE source URL |
| `verdict_confidence` | object | low (deterministic; LLM verification pending) |
| `taxonomy_version` | object | Taxonomy version |
| `run_at` | object | ISO-8601 UTC |

### `candidate_mitigation_summary.parquet`

| Column | Type | Description |
|---|---|---|
| `candidate_category` | object | Candidate category |
| `n_focus` | int64 | Profile-subset FONSI count (or full if no profile) |
| `n_mitigated_fonsi` | int64 | Count with is_mitigated_fonsi = True |
| `mitigated_share` | float64 | n_mitigated_fonsi / n_focus |
| `n_with_boundary_language` | int64 | Count with non-empty boundary statements |
| `top_mitigation_resource_areas` | object | Top recurring resource areas (with counts) |
| `example_boundary_statements` | object | JSON list of up to 5 boundary snippets |
| `run_at` | object | ISO-8601 UTC |

### `candidate_mitigation_boundary.parquet`

| Column | Type | Description |
|---|---|---|
| `project_id` | object | UUID |
| `candidate_category` | object | Candidate category |
| `subtype` | object | Action subtype |
| `is_profile_subtype` | bool | Profile flag |
| `is_mitigated_fonsi` | bool | True if textual cue OR enforceable conditions present |
| `mitigation_confidence` | object | high, medium, or none |
| `mitigated_textual_cue` | bool | Textual cue found |
| `n_enforceable_conditions` | int64 | Count of enforceable condition rows |
| `mitigation_resource_areas` | object | Resource areas from conditions |
| `boundary_statements` | object | JSON list of extracted boundary-language snippets |
| `run_at` | object | ISO-8601 UTC |
| `taxonomy_version` | object | Taxonomy version |

### `candidate_base_rates.parquet`

| Column | Type | Description |
|---|---|---|
| `candidate_category` | object | Candidate category |
| `candidate_label` | object | Human-readable label |
| `candidate_role` | object | profile or contrast |
| `ce_story` | object | CE narrative |
| `n_universe_projects` | int64 | Distinct projects in the full clean candidate universe |
| `n_ce_universe` | int64 | Universe projects with process_type = CE |
| `n_ea_universe` | int64 | Universe projects with process_type = EA |
| `n_eis_universe` | int64 | Universe projects with process_type = EIS |
| `n_observed_fonsi_projects` | int64 | Distinct observed EA-source FONSI projects |
| `n_profile_fonsi_projects` | int64 | Observed FONSIs in the profile (CE-shaped) subtype |
| `n_projects_with_ce_citation` | int64 | Candidate projects with a D3 CE citation |
| `base_rate_caveat` | object | Interpretation warning |
| `taxonomy_version` | object | Taxonomy version |
| `analysis_version` | object | d6_stage_a_v2 |
| `analysis_run_at` | object | ISO-8601 UTC |

### `ce_landscape_ces.parquet`

| Column | Type | Description |
|---|---|---|
| `ce_id` | int64 | CE Explorer integer ID |
| `structured_id` | object | Human-readable structured ID (e.g., DOE-1--5-87) |
| `agency_unit` | object | Agency unit code |
| `agency_name` | object | Full agency name |
| `origin` | object | CE origin (regulation / statute / other) |
| `canonical_source_url` | object | Source URL |
| `context` | object | CE context text |
| `additional_context` | object | Additional context |
| `extraordinary_circumstances` | object | Extraordinary circumstances text |
| `ce_description` | object | CE description text |
| `source_url` | object | Source document URL |
| `source_version` | object | CE Explorer version |
| `source_version_date` | object | CE Explorer version date |
| `bound_acres` | float64 | Parsed acreage bound |
| `bound_miles` | float64 | Parsed miles bound |
| `bound_kv` | float64 | Parsed kV bound |
| `bound_mw` | float64 | Parsed MW bound |
| `bound_wells` | object | Parsed well-count bound |
| `states_any_bound` | bool | True if any numeric bound was parsed |
| `nearest_xagency_ce` | object | Structured ID of nearest CE in a different agency |
| `nearest_xagency_cosine` | float64 | Cosine similarity to nearest cross-agency CE |
| `nearest_xagency_unit` | object | Agency unit of nearest cross-agency CE |
| `xagency_near_duplicate` | bool | True if cosine ≥ 0.85 |
| `landscape_run_at` | object | ISO-8601 UTC |

---

## Key Design Decisions

- **Narrow-first.** Choose candidates up front and deep-extract only those
  (20–150 FONSIs each — small enough to read and to LLM affordably). Trust comes
  from small N + verification, not a big pipeline.
- **Three explicit verdict types.** NEW / EXPAND / ADOPT map directly to the
  three NEPA policy levers; the verdict logic is deterministic and inspectable in
  `07`. With the current five candidates, all already map to an existing CE, so
  the corpus yields EXPAND/ADOPT. NEW requires broadened candidate generation.
- **Reuse, don't rebuild.** The corpus inventory and EA+FONSI section extraction
  are read-only inputs from v1. The existing-CE source is the committed `ce.json`
  (CE Explorer), loaded via `ce_source.py` — no live fetch, no parquet snapshot.
- **Three explicit base-rate counts**, never one ambiguous "share": universe by
  process type, candidate EA projects, observed EA-source FONSI projects.
- **Deterministic first, LLM gated.** The deterministic pass is reproducible and
  runs without API keys; the LLM pass (`--use-llm`) refines action definitions,
  limit selection, and the mitigation determination once benchmarked.
- **Track B as an upgrade to the mitigation signal.** `05_mitigation_and_boundary`
  supersedes the coarse `mitigation_dependence` from `03` by applying the D02
  dual-signal (textual cue + enforceable conditions). The `03` field is kept for
  the deterministic review but the `05` summary feeds `07` and `08`.
- **Provenance throughout.** Every fact carries document/section/span IDs + a
  hash; CE matches are ranking aids left pending manual verification; audit
  timestamps (`*_extraction_run_at` always, `*_llm_run_at` only on success,
  else `""`) match the rest of the pipeline.
- **`D6_REVIEW_DIR` in `common.py`.** All drill-down/QA CSVs go to
  `output/deliverable06/review/` (not top-level output) to keep the client-facing
  folder clean. Only `d6_comparison_table.csv` and `figures/` sit at the top.

---

## Model Selection & Cost (Gate 3)

The LLM extraction pass (`03 --use-llm`) handles the subtle fields (action
definition, mitigation-dependence, extraordinary circumstances). The deterministic
pass already handles easy fields (acres, miles, MW, siting booleans), so model
choice is about quality on the *hard* fields, not cost — the whole job is a
rounding error.

Per call ≈ 1,650 input + ~400 output tokens. Pricing per 1M tokens (claude-api
reference, cached 2026-05-26 — verify before relying on it):

| Model | Input | Output | ~per call | All candidates (295) | Full corpus (452) |
|---|---:|---:|---:|---:|---:|
| `claude-haiku-4-5` | $1 | $5 | ~$0.004 | ~$1.05 | ~$1.65 |
| `claude-sonnet-4-6` (default) | $3 | $15 | ~$0.011 | ~$3.20 | ~$4.95 |
| `claude-opus-4-8` | $5 | $25 | ~$0.018 | ~$5.35 | ~$8.25 |

The **Batch API halves** all of these (async, fine for this non-interactive job).

**Default: Sonnet 4.6** (the workhorse), escalate to **Opus 4.8** where the
benchmark shows nuance gaps; Haiku only if the benchmark confirms it suffices.
Don't guess — run `benchmark_models.py` on a labeled sample to pick the **lowest
model that clears the accuracy threshold** (default 0.90).

---

## Known Issues and Cautions

- **No NEW verdicts from the current five candidates.** All five already map to a
  CE in `ce.json`. Surfacing genuine NEW opportunities requires the broadened
  candidate generation (Track A, forward plan). The current pipeline correctly
  identifies EXPAND and ADOPT levers.
- **Verdicts are deterministic first-pass only** (`verdict_confidence = "low"`)
  until the LLM verification pass (Gate 3) firms up action definitions and
  mitigation determinations.
- **Mitigation share is high** (86% for transmission upgrades, 86% for
  geothermal) because the profile subset skews toward complex/larger projects.
  Recurring, consistent mitigations can become codifiable CE design criteria;
  idiosyncratic ones are disqualifiers. Interpret the mitigated-FONSI flag as an
  input to that analysis, not as a disqualification.
- **`extraordinary_circumstances` is a mention scan, not a determination.**
  It flags whether the EA text mentions a CE-gating resource category; it does
  not determine whether the resource is present and impacted. That requires the
  LLM pass.
- **CE comparison is lexical + embedding ranked, unverified.** Every match is
  `manual_verification_status = pending`; CE scores are ranking aids only and
  never decide coverage.
- **NEPATEC "Clean + Transmission" tagging is noisy.** The
  `off_scope_misclassified` subtype in `candidate_corpus` surfaces non-clean
  projects (nuclear demo, gas plant, mining, an appliance-efficiency standard)
  that leaked into the clean-transmission universe. These are flagged in the
  membership review CSV; they are not in the profile subset.
- **Temporary resource assessment corpus is thin** (2 observed FONSIs). The
  category is large in the universe (2,317 CE projects) but nearly all are
  already CE'd; the "already_covered" drop is correct.
- **`ce_crosswalk.parquet` and `ce_explorer_snapshot.parquet`** in the analysis
  folder are v1 artifacts that remain for provenance; they are not read by any
  numbered v2 script. The canonical CE source is `ce.json` via `ce_source.py`.

---

## Methodological Notes

**Why narrow-first?** The v1 opportunity-scan approach (mine all 452 FONSIs for
any recurring pattern) produced a large but undefensible set. Narrow-first — pick
candidate action categories up front, extract deeply, verify — trades breadth for
defensibility. A small, verified shortlist is more useful to CATF than a long
unverified one.

**Why `ce.json` instead of a live fetch or parquet snapshot?** The CE Explorer
API is not stable; a live fetch would make the pipeline non-reproducible. A
committed JSON export (v2.0.0, 2025-08-07) is version-controlled, loadable with
stdlib `json`, and updated on a controlled schedule. The old CEQ xlsx was
removed because it duplicated the same data less cleanly.

**Why three base-rate counts?** A single "FONSI rate" is ambiguous — it depends
on denominator choice (all clean projects? only EA projects? only candidate
projects?). Three counts make the survivorship bias visible: seeing that
transmission upgrades are 86% CE-use in the universe tells us the EA/FONSI
subset is a self-selected group of projects the agency chose to study rather than
exclude directly.

**Why `all-MiniLM-L6-v2` for action definitions and CE ranking?** It is fast,
local, requires no API key, and produces embeddings suitable for cosine
comparisons at the phrase/sentence level. For the action-definition task, the
alternative (full LLM for all 295 rows) is 60× more expensive and is properly
reserved for the hard calls (Gate 3). For CE ranking, embedding similarity
catches synonyms that lexical overlap misses, while the 35% lexical weight
prevents pure semantic drift toward unrelated CEs.

**Why a dual signal for mitigated-FONSI (Track B)?** The textual-cue approach
alone misses enforceable conditions that are stated in the body of the EA without
using the specific D02 trigger phrases. The conditions approach alone misses
projects where the FONSI document uses the trigger language but the conditions are
categorized differently. Neither alone is sufficient; the union of both signals
(with a confidence hierarchy) gives the most complete picture while keeping the
flag inspectable.

**Why `verdict_confidence = "low"` throughout?** The deterministic numeric
parsing is best-effort (context-aware, but regex-based). The CE bound parsing is
also regex-based. Until the LLM pass verifies the key judgment calls (was the
action really within ROW? is the mitigation codifiable?), all verdicts should be
treated as directional.

---

## Reproduction

```bash
# deterministic pass (runs 01→07 Python + 08 R)
conda run -n nepa python phase2/code/deliverable06/_run.py

# with the gated LLM pass (Gate 3; needs ANTHROPIC_API_KEY)
conda run -n nepa python phase2/code/deliverable06/_run.py --use-llm

# skip figures (if Rscript unavailable)
conda run -n nepa python phase2/code/deliverable06/_run.py --skip-figures

# model selection before LLM run (calls paid API, run once)
conda run -n nepa python phase2/code/deliverable06/benchmark_models.py

# render the CE catalog .md (after ce.json updates)
conda run -n nepa python phase2/code/deliverable06/extract_ce_catalog.py

# render the report
quarto render phase2/reports/deliverable06.qmd
```

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
