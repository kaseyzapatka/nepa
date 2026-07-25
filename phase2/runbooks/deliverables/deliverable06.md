# D6 — Patterns in FONSIs (CE-candidate shortlist)

**Purpose:** Identify a small, defensible shortlist of recurring clean-energy action categories in prior EAs/FONSIs that may warrant new or expanded categorical exclusions (CEs). For each `tech_group__action` grid cell: a crisp action definition, evidence it recurs with no significant impact, recurring bounding limits, mitigation dependence, whether an existing CE already covers it, and traceable citations. Resolves each cell to **NEW / EXPAND / ADOPT / already_covered**.
**Input:** `deliverable06/fonsi_*` pre-built inventory/packets/spans, D3 `projects_nepa_reviews.parquet` + `ce_citations.parquet` (base rates / CE use), D4 `timeline/timeline_project_dates.parquet` (decision dates for the post-FRA tabulation), `notes/deliverable06/ce.json` (CE Explorer catalog).
**Output:** `data/analysis/deliverable06/candidate_verdicts.parquet` (52 grid cells), `candidate_facts.parquet`, `candidate_ce_comparison.parquet`, `fonsi_enrichment.parquet`, plus ~16 figures and the client review CSVs.
**Cost:** Billable LLM (standalone, cached — re-runs are $0): the 39-field enrichment (`03_enrich_llm.py`, Sonnet, one call per 451 FONSIs) and the condition re-tag (`retag_condition_resources.py`, ~$4.23 Haiku). **The user launches all billable passes.** The `_run.py` chain itself is $0.
**Scope:** 452 clean-energy EA-source FONSI projects (451 enriched, 1 skipped for no evidence).
**Conda env:** `nepa`.

**Two categorization schemes coexist on disk** (see architecture doc): the legacy 5-category taxonomy (`candidates.py`) and the authoritative **`tech_group__action` grid** (52 observed cells). `07`/verdicts/report are all grid-keyed.

---

## Prerequisites — standalone billable passes (run ONCE, cached; user-launched)

These are **not** in `_run.py`. `_run.py` aborts before `09` if `fonsi_enrichment.parquet` is missing.

```bash
# 1. The enrichment pass (💰 Sonnet; one structured call per clean FONSI → 39 fields).
#    Preview cost first (no key/Keychain, no spend):
conda run -n nepa python phase2/code/deliverable06/03_enrich_llm.py --dry-run
conda run -n nepa python phase2/code/deliverable06/03_enrich_llm.py --workers 4
#    → fonsi_enrichment.parquet (452 rows: 451 enriched + 1 skipped, 63 cols)

# 2. The action-verb labeler (💰 Sonnet; reuses the cached enrichment summary, no doc re-read):
conda run -n nepa python phase2/code/deliverable06/10_action_label.py --dry-run
conda run -n nepa python phase2/code/deliverable06/10_action_label.py --workers 4
#    → fonsi_action_labels.parquet (action verb + is_codifiable)
```

`03_enrich_llm.py` has two cached stages (`--stage extract|classify|both`, default `both`): EXTRACT (expensive) then CLASSIFY (a cheap re-ask of only `action_category`). Both are cached on prompt/schema/model so re-runs are $0. Default model `claude-sonnet-4-6`. Run `benchmark_models.py` once before committing to a model.

---

## The chain (`_run.py`, $0, deterministic)

```bash
conda run -n nepa python phase2/code/deliverable06/_run.py
```

Runs, in order: `01 → 02 → 03 → 04 → 05 → 06 → 09 → 07 → 11 → 12 → 13 → 14`, then `08_create_figures.R` (skipped with a notice if `Rscript` is absent). Key stages:

| Order | Script | Role |
|---|---|---|
| `01` | `01_select_candidate_corpus.py` | Assign projects to the 5 legacy candidate categories (deterministic) |
| `02` | `02_assemble_candidate_evidence.py` | Gather each candidate FONSI's typed text + span provenance |
| `03` | `03_extract_candidate_facts.py` | Deterministic facts (later overwritten by `09`; kept for `05` + audit) |
| `04` | `04_base_rates_and_ce.py` | Legacy base rates **+ grid CE-comparison** (median-cosine ranked CEs per cell) |
| `05` | `05_mitigation_and_boundary.py` | Deterministic mitigated-FONSI + boundary cross-check (legacy-5) |
| `06` | `06_ce_landscape.py` | Embed 2,105 CEs; cross-agency near-duplicates; k-means (k=8) CE families |
| `09` | `09_wire_enrichment.py` | **Pivot:** build 52 `tech_group__action` cells; overwrite facts + mitigation with LLM-backed values; merge D4 decision dates. Aborts if enrichment missing |
| `07` | `07_classify_and_rank.py` | Verdict (new/expand/adopt/already_covered) + rank + G1 shortlist tiers + eCFR coverage gate |
| `11` | `11_expand_analysis.py` | #39 generalized expand: size vs CE-bound distributions |
| `12` | `12_other_action_themes.py` | #40 cluster the 92 `action=='other'` FONSIs (terminal; asserts verdicts unchanged) |
| `13` | `13_postfra_refresh.py` | A2 post-FRA tabulation (D4 `decision_date`, FRA cut 2023-06-03) |
| `14` | `14_threshold_retrieval.py` | #44 regex threshold-phrase retrieval over spans |
| `08` | `08_create_figures.R` | ~16 report figures (`theme_catf`) |

> **Footgun:** `_run.py --use-llm` runs the **OLD** narrow-facts pass inside `03_extract_candidate_facts.py`, **NOT** the 39-field enrichment. The real enrichment is the standalone `03_enrich_llm.py` above. Do not pass `--use-llm` unless you specifically want the legacy narrow pass.

---

## Optional standalone tools

```bash
# $0 eCFR coverage scaffold (run once before 07; caches eCFR text for the 24 adopt/expand cells):
conda run -n nepa python phase2/code/deliverable06/ce_ecfr_verify.py
#   → candidate_ce_coverage.parquet + ce_verification_worksheet.csv (empty verdicts for a reviewer to fill)
#   optional cost projection of an LLM adjudication: add  --llm --dry-run

# 💰 D2-facing condition re-tag (~$4.23 Haiku; rebuilds fonsi_conditions.resource_area in place):
conda run -n nepa python phase2/code/deliverable06/retag_condition_resources.py --dry-run     # Tier-1 preview + exact cost, no key
conda run -n nepa python phase2/code/deliverable06/retag_condition_resources.py --run --workers 4
#   Tier-1 heading-inheritance is DISABLED by default (gold precision 0.20); opt in with --use-tier1

# QA gate — 25 grid invariants (run after the chain):
conda run -n nepa python phase2/code/deliverable06/qa_deliverable06.py
```

---

## CLI reference

| Script | Flags |
|---|---|
| `03_enrich_llm.py` | `--model` (`claude-sonnet-4-6`), `--stage {both,extract,classify}`, `--pilot`, `--sample N`, `--workers`, `--dry-run` |
| `10_action_label.py` | `--model`, `--workers`, `--sample N`, `--dry-run` |
| `retag_condition_resources.py` | `--run`, `--dry-run`, `--model`, `--workers`, `--use-tier1` |
| `ce_ecfr_verify.py` | `--llm` (with `--dry-run`), `--dry-run`, `--model` |
| `_run.py` | `--use-llm` (legacy narrow facts pass — see footgun above) |

`--dry-run` on every billable script projects cost with **no key/Keychain access and no spend**.

---

## Primary outputs (`data/analysis/deliverable06/`)

| File | Description |
|---|---|
| `fonsi_enrichment.parquet` | 39-field + audit LLM enrichment of every clean FONSI. 452 rows (451 enriched), 63 cols |
| `fonsi_action_labels.parquet` | Per-FONSI action verb (11-value vocab) + `is_codifiable` |
| `candidate_facts.parquet` | One row per enriched FONSI (451), grid cell + LLM-backed sizes/booleans/mitigation, `is_ce_shaped` |
| `candidate_ce_comparison.parquet` | Top-8 median-cosine CE matches per grid cell (52 cells) |
| `candidate_verdicts.parquet` | One row per grid cell (52): verdict, `rank_score`, best CE, adopt targets, `shortlist_tier` |
| `corpus_mitigation_stats.parquet` | Corpus-wide mitigated-FONSI share (1 row) |
| `ce_landscape_ces.parquet` | Per-CE landscape (2,105 CEs): near-duplicates, parsed bounds, k-means family |
| `other_action_themes.parquet` | #40 cluster id + label for the 92 `action=='other'` FONSIs |
| `candidate_ce_coverage.parquet` | A1/#37 eCFR-adjudication scaffold (24 adopt/expand cells × top-5 CEs) |

Client-facing (in `output/deliverable06/`): `d6_comparison_table.csv`, `expand_analysis.csv`, `postfra_recurrence.csv`, `threshold_candidates.csv`, `rank_sensitivity.csv`, and `review/d6_new.csv` / `d6_expand.csv` / `d6_adopt.csv` + per-cell evidence CSVs. Figures in `output/deliverable06/figures/`.

---

## Run results (confirmed from the current parquets; QA gate 25/25 PASS)

- **Enrichment:** 452 clean FONSIs, 451 enriched (1 skipped); Stage-2 classification parsed on all 451; CE-shaped quote verification 96.7%.
- **Grid:** 451 facts rows across 52 `tech_group__action` cells; 215 of 451 are `is_ce_shaped`.
- **Corpus mitigation:** 310/451 (68.7%) `is_mitigated_fonsi` (309 case-specific-dependent, 1 design/none).
- **Verdicts (52 cells):** adopt 21 · new 17 · already_covered 12 · expand 2 (final post-eCFR-gate; the deterministic pre-gate baseline is adopt 22 / new 16, and the coverage gate flips `Hydropower__new_build` adopt→new).
- **G1 shortlist:** `d6_new.csv` = 9 rows (6 `main` + 3 `exploratory`).
- **CE landscape:** 2,105 CEs across 78 agency units; the t-SNE scatter groups them into **8 k-means families** (deterministic; silhouette ~0.035 flat, so k=8 is a readability choice — NOT BERTopic).

---

## Notes

- **Provenance throughout.** Every enriched fact carries a verified verbatim quote + span/document/page reference (`n_verified_quotes` / `verified_quote_rate`); `confidence_score = 0.6·verified_quote_rate + 0.4·field_fill_rate`, independent of the model's self-rating.
- **`is_codifiable` is deterministic** from the action verb (False only for `manufacturing` and `land_or_row_authorization`), so the client shortlist's "can a CE even codify this" gate is fully auditable.
- **CE matches are ranking aids, not legal findings.** Every adopt/expand verdict is a text-similarity match pending eCFR adjudication (`ce_ecfr_verify.py` → reviewer fill-in). The `#38` department crosswalk is annotate-only and never changes a verdict.
- **FRA cut date: 2023-06-03** (matches D4/D5). The post-FRA tabulation flags NEPATEC's 2024–2025 ingestion lag — a low post-cut count is not evidence of low current activity.
- **D6 writes one output D2 consumes:** `retag_condition_resources.py` rebuilds `fonsi_conditions.parquet`'s `resource_area`, which D2's mitigation join reads. D6's own verdicts never read that field.
