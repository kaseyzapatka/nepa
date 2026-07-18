# D2 data dictionary — significance determinations

Deliverable 2 extracts NEPA **significance determinations** — the conclusions agencies reach
about whether a project's impacts are significant — from clean-energy EA/FONSI and EIS
documents. The pipeline (`phase2/code/deliverable02/`) resolves the governing regulatory
framework per project (script 00), builds a three-tier project corpus (01), generates
deterministic candidate windows and adjudicates them with an LLM on two parallel tracks —
FONSI (02) and EIS (04) — and validates both tracks against hand-labeled gold sets (03/03-eis,
`gold_agreement.py`, 05). Every parquet emitted into
`phase2/data/analysis/deliverable02/` (including `gold/`) is documented below. Row counts are
as of the current committed run (schema_version `d2_v2_11`, prompt_version `d2_v3`,
adjudication model `claude-sonnet-5`). Derived analysis tables (the CSVs the report reads) live
in `phase2/output/deliverable02/analysis/`, built by `06_create_figures.R`.

Conventions used throughout:

- All `*_run_at` columns are ISO-8601 UTC timestamps set when the emitting script ran.
- All `schema_version` columns carry the frozen plan version (`d2_v2_11`).
- `evidence_text_sha256` / `sha256_*` ids are deterministic SHA-256 hashes (see
  `common.py: sha256_text`, `sha256_join`) so ids are stable across reruns.
- Controlled vocabularies come from `significance_taxonomy.py` and are listed in full at first
  use; later tables reference them by name.

## Controlled vocabularies (significance_taxonomy.py)

- **determination_class** — `no_significant_impact`, `less_than_significant`,
  `less_than_significant_with_mitigation`, `significant_adverse`, `significant_unavoidable`,
  `eis_required`, `not_a_determination`, `ambiguous`
- **determination_scope** — `project_overall`, `resource_specific`, `alternative_specific`,
  `threshold_specific`, `programmatic_or_tiered`, `procedural`
- **determination_polarity** — `no_adverse`, `adverse_not_significant`, `adverse_significant`,
  `mixed`, `unknown`
- **shared_resource_area** (the shared D6 twelve + sentinels) — `air_quality`, `water`,
  `biological`, `cultural`, `visual`, `noise`, `soils_geology`, `socioeconomic`,
  `transportation`, `land_use`, `climate_ghg`, `public_health`, `unknown` (resource-specific
  finding whose resource genuinely could not be placed), plus code-assigned `project_wide`
  (project-level conclusions, scope = `project_overall`; never an LLM answer)
- **d2_resource_area** (subarea crosswalk under the shared area) — `water_quality`, `wetlands`,
  `floodplains`, `socioeconomics`, `environmental_justice`, `public_health`,
  `hazardous_materials`, `biological_special_status`, `cultural_historic`, `ghg_climate`,
  `air_quality`, `visual`, `noise`, `soils_geology`, `transportation`, `land_use`, `unknown`
- **threshold_type** — `NAAQS`, `PSD`, `ESA_take`, `ESA_jeopardy`, `NHPA_adverse_effect`,
  `wetland_floodplain`, `noise_threshold`, `visual_vrm`, `other_quantitative`, `none`, `unknown`
- **threshold_status** — `exceeds`, `does_not_exceed`, `may_exceed`, `mitigated_below`,
  `not_evaluated`, `unknown`
- **mitigation roles / obligations that count as enforceable** — `condition_role` in
  {`mitigation_commitment`, `enforcement_or_permit_condition`} AND `obligation_level` in
  {`required`, `committed`}

---

## 1. Project-level scaffolding

### project_regime.parquet

Producing script: `00_resolve_framework_regime.py`. Grain: **one row per project** in the
regime universe (clean EA-source FONSI projects ∪ clean EA/EIS projects). Rows: **1,326**.

| column | type | description |
|---|---|---|
| project_id | VARCHAR | NEPATEC project UUID. |
| process_type | VARCHAR | `EA` or `EIS` (from the corpus scope query). |
| decision_date | VARCHAR | ISO date from the D4 timeline (`timeline_project_dates.parquet`); `""` when missing. |
| decision_source_type | VARCHAR | D4 provenance of the decision date (which extraction source supplied it). |
| decision_confidence | VARCHAR | D4 confidence normalized to `high` / `medium` / `low` / `missing`. |
| decision_is_proxy | BOOLEAN | TRUE when the D4 decision date is a proxy (e.g. publication date), not a signed decision. |
| decision_period | VARCHAR | Descriptive "decided during" CEQ-framework period from decision_date: `pre_2020_ceq`, `ceq_2020_rule`, `ceq_2022_phase1`, `ceq_2024_phase2`, `ceq_2025_interim_removal`, `ceq_2026_final_removal`, `unknown`. |
| applicability_period | VARCHAR | Legal-method estimate of the applicable framework, computed from the initiation date (falls back to decision_date when initiation is missing); same vocabulary as decision_period. |
| fra_overlay | BOOLEAN | TRUE when decision_date >= 2023-06-03 (FRA enactment), i.e. the FRA statutory overlay applies. |
| regime_source_date | VARCHAR | The date the regime assignment was derived from (= decision_date; `""` when none). |
| regime_source_date_role | VARCHAR | `decision` when a date was used, `none` otherwise. |
| regime_source_date_is_proxy | BOOLEAN | Copy of decision_is_proxy for the source date. |
| regime_source_date_confidence | VARCHAR | Copy of the normalized decision_confidence for the source date. |
| regime_assignment_status | VARCHAR | Priority-resolved assignment quality: `not_applicable`, `missing_date`, `boundary_review` (within ±90 days of a CEQ cut or the FRA date), `low_confidence_review`, `assigned_proxy`, `assigned_medium_confidence`, `assigned_high`. Keeps uncertain rows out of headline splits. |
| regime_notes | VARCHAR | Free-text provenance, e.g. `raw_conf=...; applicability_from=initiation|decision`. |
| regime_run_at | VARCHAR | Run timestamp. |
| schema_version | VARCHAR | `d2_v2_11`. |

### significance_corpus.parquet

Producing script: `01_build_d2_inventory.py`. Grain: **one row per corpus project** (three
tiers: `mitigated_fonsi`, `straight_fonsi`, `eis_significant`). Rows: **1,205**.

| column | type | description |
|---|---|---|
| project_id | VARCHAR | NEPATEC project UUID. |
| process_type | VARCHAR | `EA` (FONSI tiers) or `EIS`. |
| doc_type | VARCHAR | `FONSI` or `FEIS` — the document class the tier's findings come from. |
| corpus_tier | VARCHAR | `mitigated_fonsi` (recall screen: mitigated-finding cue AND >=1 enforceable condition anywhere in the project), `straight_fonsi` (the complement within the clean EA-source FONSIs), `eis_significant` (clean EIS projects). |
| fonsi_subtype | VARCHAR | For mitigated: `mitigated_dual_signal`; for straight: `design_feature_or_partial` (conditions exist but no cue) or `no_mitigation_signal`; `""` for EIS. |
| mitigated_cue_hit | BOOLEAN | A finding span matched a `MITIGATED_SCREEN_CUES` regex (Gate-1 recall screen; always FALSE for EIS). |
| n_enforceable_conditions | BIGINT | Count of D6 condition rows with enforceable role+obligation for this project (0 for EIS). |
| mitigated_strict_same_section | BOOLEAN | Cue finding and a qualifying condition in the SAME section (the strict variant of the dual signal). |
| mitigated_windowed_pm2 | BOOLEAN | Cue finding and a qualifying condition within ±2 pages (the windowed variant). |
| agency | VARCHAR | Coarse display label: `BLM`, `DOE-family`, `other`. |
| agency_scope_status | VARCHAR | Headline-denominator gate: `primary_blm_doe_family`, `context_other_agency`, `manual_scope_review` (blank/unparseable lead agency). |
| agency_scope_rule | VARCHAR | Constant `blm_plus_doe_family` — the rule version that produced agency_scope_status. |
| off_mission_flag | BOOLEAN | OR of the five Phase-1 source exclusion flags (nuclear-tech-only, nuclear waste, military, broadband-only, utilities). ADVISORY ONLY — flagged rows stay in the broad Clean universe. |
| project_energy_type_strict | VARCHAR | Phase-1 strict energy classification, kept for a strict-clean sensitivity cut. |
| time_scope_status | VARCHAR | `in_scope_dated`, `pre_ARRA_dated`, `boundary_review` (within ±90 days of ARRA 2009-02-17), `missing_decision_date`. |
| analysis_scope | VARCHAR | `primary` when time_scope_status = `in_scope_dated`, else `context_or_validation`. |
| decision_date | VARCHAR | D4 decision date (ISO; may be null/empty). |
| decision_confidence | VARCHAR | D4 confidence as stored in the timeline table (not normalized here). |
| decision_is_proxy | BOOLEAN | D4 proxy flag. |
| lead_agency_harmonized | VARCHAR | Harmonized lead-agency string from the Phase-1/D6 inventory. |
| tech_group | VARCHAR | Technology group (from the FONSI inventory; for EIS enriched from `document_sections`). |
| project_state | VARCHAR | Project state. |
| project_title | VARCHAR | Project title (NEPATEC). |
| project_description | VARCHAR | Full project description (NEPATEC; the review CSV truncates, this parquet does not). |
| corpus_run_at | VARCHAR | Run timestamp. |
| schema_version | VARCHAR | `d2_v2_11`. |

### project_cohorts.parquet

Producing script: `01_build_d2_inventory.py` (`build_cohorts`). Grain: **one row per corpus
project**. Rows: **1,205**.

| column | type | description |
|---|---|---|
| project_id | VARCHAR | NEPATEC project UUID. |
| process_type | VARCHAR | `EA` or `EIS`. |
| agency_scope_status | VARCHAR | Copied from significance_corpus (see above). |
| agency_scope_rule | VARCHAR | Constant `blm_plus_doe_family`. |
| cohort_by_date | VARCHAR | Frozen decision-date bins: `pre_ARRA` (< 2009-02-17), `arra_to_bil` [ARRA, 2021-11-15), `bil_to_ira` [BIL, 2022-08-16), `ira_to_fra` [IRA, 2023-06-03), `post_fra` [FRA, present], `missing_decision_date`. |
| law_cited_arra | BOOLEAN | Project cites ARRA in the D5 law-citation table (`deliverable05/law_citations.parquet`). |
| law_cited_bil | BOOLEAN | Project cites BIL (same source). |
| law_cited_ira | BOOLEAN | Project cites IRA (same source). |
| law_cited_doe_funding | BOOLEAN | Project cites DOE funding authority (same source). |
| cohort_run_at | VARCHAR | Run timestamp. |
| schema_version | VARCHAR | `d2_v2_11`. |

---

## 2. FONSI track (script 02 + candidate_gen.py + extract_common.py)

### significance_section_candidates.parquet

Producing script: `02_extract_fonsi_significance.py` via
`candidate_gen.generate_fonsi_candidates()`. Grain: **one row per D6 FONSI finding span**
(candidate window) in the clean EA corpus. Rows: **3,478**.

| column | type | description |
|---|---|---|
| project_id | VARCHAR | NEPATEC project UUID. |
| document_id | VARCHAR | Source document id (D6 `fonsi_evidence_spans`). |
| manifest_role | VARCHAR | Document's role in the D6 FONSI manifest: `canonical_fonsi`, `supporting_fonsi`, `linked_ea`. |
| section_id | VARCHAR | D6 section id containing the span. |
| evidence_span_id | VARCHAR | D6 finding-span id — the window key used throughout D2 (joins to determinations and gold). |
| heading_title | VARCHAR | Section heading the span sits under. |
| page_start / page_end | BIGINT | Page range of the span. |
| source_span_sha256 | VARCHAR | D6 hash of the source span (span-identity provenance). |
| has_qual_cond_same_section | BOOLEAN | An enforceable D6 condition exists in the same section. |
| has_qual_cond_windowed | BOOLEAN | An enforceable D6 condition exists within ±2 pages. |
| source_substrate | VARCHAR | Constant `d6_evidence_span` — which text substrate the window came from. |
| source_unit_id | VARCHAR | Normalized unit id (= evidence_span_id for this substrate). |
| span_char_start / span_char_end | INTEGER | Always null — D6 spans carry no char offsets (plan §4). |
| candidate_class_guess | VARCHAR | Ordered-rule regex guess of the determination_class (heuristic only; the LLM assigns the final class). Vocabulary: determination_class. |
| determination_polarity_guess | VARCHAR | Regex polarity guess. Vocabulary: determination_polarity. |
| matched_cue_group | VARCHAR | Which `DETERMINATION_CUES` group fired: `document_outcome`, `explicit_less_than_significant`, `explicit_mitigated_lts`, `explicit_significant_adverse`, `none`. |
| resource_area_guess | VARCHAR | Keyword guess of the shared resource area (shared_resource_area vocabulary, no `project_wide`). |
| resource_subarea_guess | VARCHAR | Keyword guess of the d2_resource_area subarea. |
| threshold_types_guess | VARCHAR | Comma-joined `THRESHOLD_CUES` hits in the span (threshold_type values). |
| evidence_text | VARCHAR | Span text, sliced to the 16,000-char FONSI window cap. |
| evidence_text_sha256 | VARCHAR | Hash of the FULL span text (dedup / join key across reruns). |

### batch_candidates_fonsi.parquet

Producing script: `extract_common.submit_batch()` (invoked by 02 in batch mode). Grain: **one
row per submitted candidate window** — a frozen snapshot of the candidate frame at
submission so `--batch-fetch` can rebuild determinations later. Rows: **3,478**.
Schema: identical to `significance_section_candidates.parquet` plus:

| column | type | description |
|---|---|---|
| batch_custom_id | VARCHAR | Message-Batches request id, `fonsi-NNNNNN` in submission order; joins batch results back to windows. |

Companion (non-parquet): `batch_manifest_fonsi.json` — batch ids, request counts, model,
prompt/schema versions, submission timestamp.

### mitigation_signal_matches.parquet

Producing script: `02_extract_fonsi_significance.py` (`mitigation_signal_matches`). Grain:
**one row per (cue finding span × qualifying D6 condition row) pair** matched same-section or
within ±2 pages (the frozen dual-signal join, plan §3). Rows: **2,332**.

| column | type | description |
|---|---|---|
| cue_evidence_span_id | VARCHAR | Finding-span id on the cue side (= source_unit_id of the determination window). |
| project_id | VARCHAR | NEPATEC project UUID. |
| document_id | VARCHAR | Document holding both sides of the match. |
| cue_section_id | VARCHAR | Section of the finding span. |
| cue_page_start / cue_page_end | BIGINT | Page range of the finding span. |
| condition_section_id | VARCHAR | Section of the matched condition row. |
| condition_page_number | BIGINT | Page of the matched condition row. |
| resource_area | VARCHAR | D6 resource area of the condition (shared_resource_area vocabulary). |
| condition_role | VARCHAR | `mitigation_commitment` or `enforcement_or_permit_condition` (only enforceable roles enter the join). |
| obligation_level | VARCHAR | `required` or `committed` (only enforceable obligations enter the join). |
| source_span_sha256 | VARCHAR | D6 hash of the condition's source span. |
| match_type | VARCHAR | `same_section` or `windowed` (±2 pages). |
| condition_row_id | VARCHAR | Deterministic hash id of the condition row (project, document, section, page, span hash, role, obligation, resource). |
| condition_text_sha256 | VARCHAR | Hash of the condition text (used to count DISTINCT conditions per window). |

### significance_determinations.parquet

Producing script: `02_extract_fonsi_significance.py` via
`extract_common.build_determinations()`. Grain: **one row per (candidate window ×
resource area × determination)** — the LLM returns a list of determinations per window, so a
window explodes into multiple rows; byte-identical duplicates are deduplicated on
determination_instance_id. Rows: **7,250**. This is the FONSI-track analysis table.

| column | type | description |
|---|---|---|
| determination_instance_id | VARCHAR | Deterministic SHA-256 over (project, document, substrate, unit, resource, subarea, class, scope, threshold type+status, alternative_name, rationale hash) — the primary key. |
| source_substrate | VARCHAR | `d6_evidence_span` (FONSI track). |
| source_unit_id | VARCHAR | The window id (= D6 evidence_span_id). |
| project_id | VARCHAR | NEPATEC project UUID. |
| document_id | VARCHAR | Source document id. |
| process_type | VARCHAR | `EA`. |
| document_type_clean | VARCHAR | `FONSI` (from the corpus doc_type). |
| agency | VARCHAR | Coarse label from significance_corpus (`BLM` / `DOE-family` / `other`). |
| agency_scope_status | VARCHAR | See significance_corpus. |
| agency_scope_rule | VARCHAR | `blm_plus_doe_family`. |
| decision_date | VARCHAR | D4 decision date (project context join). |
| cohort_by_date | VARCHAR | See project_cohorts. |
| decision_source_type | VARCHAR | D4 provenance of the decision date (via project_regime). |
| decision_confidence | VARCHAR | Normalized D4 confidence (via project_regime). |
| decision_is_proxy | BOOLEAN | D4 proxy flag. |
| time_scope_status | VARCHAR | See significance_corpus. |
| analysis_scope | VARCHAR | `primary` / `context_or_validation`. |
| decision_period | VARCHAR | See project_regime. |
| applicability_period | VARCHAR | See project_regime. |
| fra_overlay | BOOLEAN | See project_regime. |
| regime_assignment_status | VARCHAR | See project_regime. |
| framework_regime | VARCHAR | Descriptive alias of decision_period, materialized once here for the report. |
| shared_resource_area | VARCHAR | The resource this determination concludes on (shared_resource_area vocabulary incl. `project_wide` for project_overall rows and `unknown`). Scope-authoritative: project-level conclusions are always `project_wide`. |
| d2_resource_area | VARCHAR | Keyword subarea, kept only when valid under the (authoritative) shared area, else `unknown` (d2_resource_area vocabulary). |
| resource_area_source | VARCHAR | `llm` (LLM assigned the resource) or `keyword` (regex-guess row). |
| determination_class | VARCHAR | The significance conclusion (determination_class vocabulary). LLM answer snapped onto the vocabulary; regex guess on dry-run/batch-missing rows. |
| determination_polarity | VARCHAR | Direction of the impact conclusion (determination_polarity vocabulary). |
| determination_scope | VARCHAR | What the conclusion covers (determination_scope vocabulary). |
| alternative_name | VARCHAR | `""` on the FONSI track (EIS-only field, kept in the shared schema for id stability). |
| rationale_text | VARCHAR | LLM's 1–2 sentence grounding for this resource's conclusion; `""` on regex rows. |
| primary_threshold_type | VARCHAR | The regulatory threshold the conclusion leans on (threshold_type vocabulary); LLM answer is authoritative, first regex cue hit is the fallback, `none` when not threshold-anchored. |
| primary_threshold_status | VARCHAR | Status relative to that threshold (threshold_status vocabulary; `none` when primary_threshold_type = `none`). |
| mitigation_flag | BOOLEAN | Raw WINDOW-level D6 dual-signal match (any enforceable condition matched this window). Over-attributes across multi-resource windows — use only for the DOCUMENT-level mitigated-FONSI rate. |
| mitigation_resource_matched | BOOLEAN | The window-level match covers THIS row's resource (or the row is project-level) — the precise per-resource attachment. |
| mitigation_dependent | BOOLEAN | Per-resource reporting field: mitigation_resource_matched OR determination_class = `less_than_significant_with_mitigation`. |
| mitigation_enforceability | VARCHAR | `permit_condition` when mitigation_resource_matched, else `none`. |
| matched_condition_row_count | BIGINT | Distinct D6 conditions matched to this window (0 when none). |
| condition_role_set | VARCHAR | Comma-joined set of condition roles matched to the window. |
| obligation_level_set | VARCHAR | Comma-joined set of obligation levels matched to the window. |
| mitigation_resource_areas | VARCHAR | Comma-joined set of resource areas the matched conditions cover. |
| section_id | VARCHAR | Section of the source window. |
| evidence_span_id | VARCHAR | = source_unit_id; the window key that joins to candidates and gold. |
| evidence_text | VARCHAR | The window text as read (capped at 16k chars on this track). |
| evidence_text_sha256 | VARCHAR | Hash of the full source text. |
| source_span_sha256 | VARCHAR | D6 source-span hash (provenance). |
| hash_semantics | VARCHAR | What source_span_sha256 hashes (= source_substrate). |
| page_start / page_end | BIGINT | Page range of the window. |
| span_char_start / span_char_end | INTEGER | Char offsets; null on this track (D6 spans carry none). |
| quoted_span | VARCHAR | First 300 chars of evidence_text (display convenience). |
| extraction_method | VARCHAR | `regex+llm` (LLM-adjudicated) or `regex` (dry-run / batch result missing). |
| confidence | DOUBLE | Fixed method confidence: 0.9 LLM, 0.5 regex (not a calibrated probability). |
| needs_human_review | BOOLEAN | TRUE for dry-run rows, missing batch results, model abstentions, `ambiguous`/`not_a_determination` classes, or off-vocab/unknown resources. |
| review_reason | VARCHAR | Why: `dry_run_regex_only`, `batch_result_missing`, `model_abstained`, `non_determination_or_ambiguous`, `resource_off_vocab`, `resource_unknown`, `""`. |
| llm_provider | VARCHAR | `anthropic` on LLM rows, `""` on regex rows. |
| llm_model | VARCHAR | Pinned model id on LLM rows (this run: `claude-sonnet-5`), `""` otherwise. |
| prompt_version | VARCHAR | `d2_v3` (multi-determination prompt). |
| input_hash | VARCHAR | SHA-256 over (project, evidence hash, prompt version, schema version, model) — reproducibility key for the LLM call. |
| response_hash | VARCHAR | SHA-256 of the raw LLM response text. |
| schema_version | VARCHAR | `d2_v2_11`. |
| significance_extraction_run_at | VARCHAR | Set on ALL rows when the build ran. |
| significance_llm_run_at | VARCHAR | Set per-row only when the LLM call succeeded; `""` on regex rows. |

### determination_thresholds.parquet

Producing script: `02_extract_fonsi_significance.py` via `extract_common` (child of the
determinations table). Grain: **one row per (determination × cited regulatory threshold)**;
emitted only for real determinations (never `not_a_determination` / `ambiguous`).
Rows: **3,052**.

| column | type | description |
|---|---|---|
| determination_instance_id | VARCHAR | Foreign key to significance_determinations. |
| project_id | VARCHAR | NEPATEC project UUID. |
| threshold_type | VARCHAR | The cited threshold (threshold_type vocabulary; union of regex cue hits and the LLM's primary). |
| threshold_status | VARCHAR | The parent's primary_threshold_status when this row IS the primary threshold, else `unknown`. |
| threshold_verbatim | VARCHAR | Reserved for a verbatim quote; currently always `""`. |
| threshold_evidence_sha256 | VARCHAR | Hash of the window text the threshold was found in. |
| threshold_specific_flag | BOOLEAN | TRUE when the parent determination_scope = `threshold_specific`. |
| schema_version | VARCHAR | `d2_v2_11`. |
| significance_extraction_run_at | VARCHAR | Run timestamp. |

---

## 3. EIS track (script 04, `_eis`-suffixed files)

The EIS track reuses the shared `extract_common` assembly, so the determination schema is the
FONSI schema plus two EIS-only columns. The substrate differs: candidate windows are
impact/consequence sections from the shared `document_sections.parquet` (no per-span D6 ids),
so `source_substrate` = `document_section` and `source_unit_id` = a deterministic hash of
(project, document, page range, char range, heading) that serves as section_id,
evidence_span_id, and the gold join key. EIS windows are read to a larger 24,000-char cap.

### significance_section_candidates_eis.parquet

Producing script: `04_extract_eis_significance.py` (`eis_candidates`). Grain: **one row per
candidate EIS section** — sections in clean-EIS corpus projects mentioning significance (or
with an environmental-consequences heading / impact topic), 20–4,000 words, kept only when the
regex class guess is a determination or a threshold cue fires, deduplicated on
(project_id, evidence_text_sha256) so a Draft+Final EIS repeating identical text counts once.
Rows: **21,854**.

| column | type | description |
|---|---|---|
| project_id | VARCHAR | NEPATEC project UUID. |
| document_id | VARCHAR | Source document id. |
| page_start / page_end | BIGINT | Page range of the section. |
| heading_title | VARCHAR | Section heading. |
| source_substrate | VARCHAR | Constant `document_section`. |
| source_unit_id | VARCHAR | Deterministic section hash id (see above); doubles as section_id / evidence_span_id downstream. |
| section_id | VARCHAR | = source_unit_id. |
| span_char_start / span_char_end | BIGINT | Section char offsets within the document (from document_sections). |
| source_span_sha256 | INTEGER | Always null on this substrate (typed INTEGER by parquet inference from all-null); the D6 span hash has no EIS equivalent. |
| candidate_class_guess | VARCHAR | Regex class guess (determination_class vocabulary). |
| determination_polarity_guess | VARCHAR | Regex polarity guess (determination_polarity vocabulary). |
| matched_cue_group | VARCHAR | Which cue group fired (see FONSI candidates). |
| resource_area_guess | VARCHAR | Keyword resource guess, falling back to the section's topic guess when unknown. |
| resource_subarea_guess | VARCHAR | Keyword subarea guess. |
| evidence_text | VARCHAR | Section text sliced to the 24,000-char EIS cap. |
| evidence_text_sha256 | VARCHAR | Hash of the FULL section text (dedup key). |
| batch_custom_id | VARCHAR | `eis-NNNNNN` batch request id (this file is written from the batch snapshot after fetch, so it carries the id). |

Note: unlike the FONSI candidates, this file has no `manifest_role`,
`has_qual_cond_same_section` / `has_qual_cond_windowed`, or `threshold_types_guess` columns —
those are D6-substrate concepts (EIS mitigation is out of scope for v1).

### batch_candidates_eis.parquet

Producing script: `extract_common.submit_batch()` (invoked by 04 in batch mode). Grain: **one
row per submitted EIS candidate window** — the frozen submission snapshot. Rows: **21,854**.
Schema identical to `significance_section_candidates_eis.parquet` (same content; the snapshot
is the authoritative copy the fetch path rebuilds from). Companion:
`batch_manifest_eis.json`.

### significance_determinations_eis.parquet

Producing script: `04_extract_eis_significance.py` via `extract_common.build_determinations(track="eis")`.
Grain: **one row per (candidate section × resource area × determination)**. Rows: **59,357**.
Schema: all columns of `significance_determinations.parquet` (same meanings; note
`process_type` = `EIS`, `document_type_clean` = `FEIS`, `source_substrate` / `hash_semantics`
= `document_section`, `source_span_sha256` all-null INTEGER, `span_char_start` / `span_char_end`
BIGINT and populated, and `alternative_name` populated verbatim from the text when the
conclusion is tied to a named alternative). The D6 mitigation join is empty on this track (EIS
mitigation / ROD commitments are out of scope for v1), so `mitigation_flag`,
`mitigation_resource_matched`, and `matched_condition_row_count` are always FALSE/0 and the
`*_set` columns `""` — but `mitigation_dependent` is still TRUE when the LLM assigns
`less_than_significant_with_mitigation`. Two EIS-only columns are appended:

| column | type | description |
|---|---|---|
| significance_factor | VARCHAR | PRIMARY driver of the conclusion, per the prompt vocabulary: `magnitude`, `duration`, `geographic_extent`, `cumulative`, `controversy`, `uncertainty`, `protected_resource`, `regulatory_threshold`, `mitigable`, `none`; `""` on regex rows. Stored as returned (not vocabulary-snapped). |
| impact_type | VARCHAR | `direct`, `indirect`, `cumulative`, `unspecified`; `""` on regex rows. Stored as returned — occasional off-vocabulary answers (e.g. `mixed`, `regional`) appear verbatim. |

### determination_thresholds_eis.parquet

Producing script: `04_extract_eis_significance.py` via `extract_common`. Grain: **one row per
(EIS determination × cited threshold)**. Rows: **27,212**. Schema identical to
`determination_thresholds.parquet`.

---

## 4. Gold sets and validation (`gold/` + validation metrics)

The gold apparatus is a fully parallel pair of tracks — FONSI (unsuffixed) and EIS (`_eis`) —
with distinct files so the two never mix. Flow per track: 03 builds a stratified reading list
of windows; two independent labelers (Claude, Codex) each write a long CSV with one row per
(window × resource) determination per `gold_labeling.md` / `gold_labeling_eis.md`;
`gold_agreement.py` aligns the two, auto-accepts core-field agreement, routes conflicts to an
adjudication CSV, and `--finalize` merges everything into the final gold parquet;
`05_validate_significance.py` scores the determinations table against it.

### gold/significance_gold_queue.parquet

Producing script: `03_build_gold_set_queue.py`. Grain: **one row per queued FONSI candidate
window** (stratified reading list: ~300 positives by candidate class with a >=50
mitigation-linked floor, ~100 negatives by agency scope; deterministic hash-ordered sampling,
no RNG). Rows: **400**. Columns are the FONSI candidate columns (see
significance_section_candidates.parquet) in worksheet order, plus:

| column | type | description |
|---|---|---|
| agency_scope_status | VARCHAR | Joined from significance_corpus (negative-stratification key). |
| gold_queue_run_at | VARCHAR | Run timestamp. |
| schema_version | VARCHAR | `d2_v2_11`. |

### gold/significance_gold_queue_eis.parquet

Producing script: `03_build_gold_set_queue_eis.py`. Grain: **one row per queued EIS candidate
window** (same design on the EIS substrate; negatives additionally require no threshold cue;
drawn from a deterministic 12,000-section pool). Rows: **400**. Columns are the EIS candidate
columns plus `evidence_span_id` (= source_unit_id, the key that joins 1:1 to the EIS
determinations), `threshold_types_guess`, `agency_scope_status`, `gold_queue_run_at`,
`schema_version`. No `manifest_role` / `has_qual_cond_*` (D6-substrate concepts).

### gold/labels_claude.csv, gold/labels_codex.csv, gold/labels_eis_claude.csv, gold/labels_eis_codex.csv

Produced by: the two labelers by hand, per `gold_labeling.md` (FONSI) /
`gold_labeling_eis.md` (EIS). Grain: **one row per (evidence_span_id × gold_resource_area)
determination the labeler found in a queued window** (long form; junk windows get a single row
with gold_resource_area = `none`). Columns: `evidence_span_id`, `gold_resource_area`,
`gold_is_determination`, `gold_determination_class`, `gold_determination_scope`,
`gold_primary_threshold_type`, `gold_primary_threshold_status`, `gold_mitigation_link`,
`gold_evidence_span_ok`, `gold_needs_human_review`, `gold_notes`, `labeler`,
`labeler_confidence`. Values use the controlled vocabularies above (case/space tolerant;
normalized on merge).

### gold/gold_agreed.parquet (390 rows), gold/gold_agreed_eis.parquet (188 rows)

Producing script: `gold_agreement.py` (merge step, no flags / `--track eis`). Grain: **one row
per (evidence_span_id × gold_resource_area) key on which BOTH labelers agree on the core
fields** (is_determination, class, mitigation_link); Claude's full record is kept.
Intermediate — superseded by the final gold below.

| column | type | description |
|---|---|---|
| evidence_span_id | VARCHAR | Window key (normalized). |
| gold_resource_area | VARCHAR | Labeled resource (shared_resource_area vocabulary, plus `none` marking a junk window). |
| gold_is_determination | VARCHAR | `TRUE`/`FALSE` — does this row assert a real determination. |
| gold_determination_class | VARCHAR | Labeled class (determination_class vocabulary). |
| gold_determination_scope | VARCHAR | Labeled scope (determination_scope vocabulary). |
| gold_primary_threshold_type | VARCHAR | Labeled threshold type (threshold_type vocabulary; blank/`none` when not threshold-anchored). |
| gold_primary_threshold_status | VARCHAR | Labeled threshold status (threshold_status vocabulary). |
| gold_mitigation_link | VARCHAR | Truthy when the conclusion depends on committed mitigation. |
| gold_evidence_span_ok | VARCHAR | Labeler's check that the window text was readable/complete. |
| gold_needs_human_review | VARCHAR | Labeler flagged the row for further review. |
| gold_notes | VARCHAR | Free-text labeler notes. |
| labeler_confidence | VARCHAR | `high` / `medium` / `low` (labeler self-report). |
| project_id | VARCHAR | Joined from the queue for context. |
| gold_source | VARCHAR | Constant `both_agree` in this file. |

### gold/significance_gold.parquet (932 rows), gold/significance_gold_eis.parquet (547 rows)

Producing script: `gold_agreement.py --finalize` (/ `--track eis --finalize`). Grain: **one row
per (evidence_span_id × gold_resource_area)** — the final adjudicated gold: agreed rows plus
analyst-adjudicated disagreements (from `output/deliverable02/gold_disagreements*.csv`).
Columns: all gold_agreed columns (with `gold_source` ∈ {`both_agree`, `human_adjudicated`} and
`labeler_confidence` = `adjudicated` on adjudicated rows) plus:

| column | type | description |
|---|---|---|
| holdout | BOOLEAN | Deterministic ~30% holdout BY WINDOW (hash of evidence_span_id; whole window in or out, stable across reruns). |
| double_coded | BOOLEAN | Always TRUE — every key was independently coded by both labelers. |
| gold_run_at | VARCHAR | Finalize timestamp. |
| schema_version | VARCHAR | `d2_v2_11`. |

### validation_metrics.parquet (FONSI), validation_metrics_eis.parquet (EIS)

Producing script: `05_validate_significance.py` (/ `--track eis`). Grain: **one row per
(metric × scope)**. Rows: **10** each (5 metrics × {overall, holdout}).

| column | type | description |
|---|---|---|
| metric | VARCHAR | `candidate_is_determination` (window-level: does the window hold >=1 real determination), `resource_determination_detection` (did the pipeline recover the right SET of (window × resource) determinations), `determination_class_macro_f1` (macro-F1 over classes with support >= 10, on matched pairs), `mitigation_dependent_f1` (on matched pairs), `primary_threshold_type_accuracy` (descriptive accuracy on matched pairs, reported in the precision column). |
| scope | VARCHAR | `overall` (all gold windows) or `holdout` (the ~30% window holdout only). |
| grain | VARCHAR | `window`, `window×resource`, or null for matched-pair metrics. |
| precision / recall / f1 | DOUBLE | Metric values (null where not applicable; accuracy lands in precision). |
| tp / fp / fn | DOUBLE | Confusion counts for the P/R/F1 metrics. |
| support | DOUBLE | Matched-pair count for the class/threshold metrics. |

The mismatched pairs behind these numbers are written to
`output/deliverable02/validation_disagreements*.csv` (`issue` ∈ {`missed_by_pipeline`,
`spurious_pipeline_determination`}).

---

## 5. Provenance: significance_run_manifest.parquet

Producing script: `extract_common.write_manifest()` (called by 02; the manifest covers the
FONSI-track write set of the most recent run). Grain: **one row per emitted artifact**.
Rows: **4** (candidates, mitigation matches, determinations, threshold child).

| column | type | description |
|---|---|---|
| artifact | VARCHAR | Logical artifact name (e.g. `significance_determinations`). |
| path | VARCHAR | Phase2-relative path of the parquet. |
| n_bytes | BIGINT | File size at manifest time. |
| sha256 | VARCHAR | Content hash of the file at manifest time (byte-level reproducibility check). |
| mode | VARCHAR | `llm` (adjudicated run) or `dry_run` (key-free regex pass). |
| model | VARCHAR | Pinned adjudication model (this run: `claude-sonnet-5`); `""` in dry-run mode. |
| prompt_version | VARCHAR | `d2_v3`. |
| schema_version | VARCHAR | `d2_v2_11`. |
| run_at | VARCHAR | Manifest timestamp. |

The EIS batch run's request-level provenance lives in `batch_manifest_eis.json` /
`batch_manifest_fonsi.json` (batch ids, request counts, model, submission time); the
determination rows themselves carry per-row provenance (`input_hash`, `response_hash`,
`llm_model`, `prompt_version`, `significance_extraction_run_at`, `significance_llm_run_at`).

## Derived tables

The report and figures read CSV tables derived from these parquets by
`06_create_figures.R`; they live in `phase2/output/deliverable02/analysis/` and are
regenerable from the parquets documented here. Review/adjudication worksheets
(`corpus_membership_review.csv`, `significance_gold_queue*.csv`, `gold_disagreements*.csv`,
`validation_disagreements*.csv`) live in `phase2/output/deliverable02/`.
