# Deliverable 02 Status (Programmatic & Tiered Reviews)

**Date**: 2026-02-21  
**Status**: Core extraction and analysis complete; targeted validation and decision work remains.

## Scope

Deliverable 02 covers clean-energy **EA/EIS** projects only and answers:
1. How many programmatic/tiered reviews are there?
2. Are tiered reviews completed faster?

## What Has Been Completed

1. Extraction pipeline is implemented and operational in `code/extract/extract_reviews.py`.
2. Scope is fixed to clean-energy EA/EIS in the extractor.
3. DuckDB-based page loading is integrated for both EA and EIS for faster runs.
4. `generic` and `tier 1` stand-in language is enabled by default in classification.
5. LLM fallback is available but now **opt-in** (`--use-llm`), with default runs using regex/title/metadata only.
6. Test output safety is in place (`--test` writes to `data/analysis/projects_reviews_test.parquet`).
7. Deliverable analysis scripts are in place:
   - `code/deliverable02/00_setup.R`
   - `code/deliverable02/01_reviews.R`
8. Deliverable report is in place at `reports/deliverable02.qmd`.
9. QA scripts were created for review validation in `code/exploratory/reviews/`:
   - `01_review_qc_overview.R`
   - `02_tier_term_logic_checks.R`
   - `03_tier_linkage_checks.R`

## Current Data Snapshot

Using `data/analysis/projects_reviews.parquet`:

- Total projects: **1,326**
- Standard: **1,165** (87.9%)
- Programmatic: **128** (9.7%)
- Tiered: **33** (2.5%)

By process:
- EA: 537 standard, 10 programmatic, 26 tiered
- EIS: 628 standard, 118 programmatic, 7 tiered

Duration summary from `output/deliverable2/tables/02_duration_summary.csv`:
- EA median days: Standard 421, Programmatic 311, Tiered 734
- EIS median days: Standard 1087, Programmatic 974, Tiered 768 (n=2; unstable)

## QA Results So Far

From `output/exploratory/reviews/`:

1. Structural QA checks currently pass (`01_issue_summary.csv` has zero flagged rows).
2. Tier-term logic checks currently show:
   - Tier 1 mentions not classified as programmatic: 0
   - Tier 2 mentions not classified as tiered: 0
3. Tier linkage checks identified 2 weak `tiers_from` strings for manual review:
   - `80f749d925f8a9b0301b808ee659b478`
   - `47e8e1d91193ef400612c85890aabcaf`

## What Is Left To Do

1. Regex validation (priority):
   - Manually review a sample of non-standard classifications for precision.
   - Manually review a sampled set of standard records to estimate false negatives.
   - Resolve the 2 weak `tiers_from` extractions and decide whether to normalize/fix these patterns.

2. LLM decision (priority):
   - Run side-by-side comparison on a labeled validation sample with and without `--use-llm`.
   - Decide LLM policy based on measured gain vs runtime/variability.
   - Recommended default remains no-LLM until benchmark evidence justifies routine LLM usage.

3. Follow-up analysis questions from `reports/deliverable02.qmd`:
   - Confirm whether captured programmatic/tiered records are correctly classified via manual validation sheet.
   - Confirm "Programmatic Collaborations" are grants/agreements, not PEIS/PEA reviews.
   - Revisit duration findings after finalized timeline coverage is merged.
   - Investigate whether longer tiered EA durations reflect complexity confounding vs tiering effect.

4. Report refinement:
   - Update narrative once validation outcomes are complete.
   - Add explicit note about small-sample limits for tiered EIS and programmatic EA inference.

## Suggested Next Sequence

1. Execute regex/manual validation pass and adjudicate weak tier-parent links.
2. Run LLM benchmark on the same adjudicated sample and make a clear go/no-go decision.
3. Recompute final Deliverable 02 outputs and refresh `reports/deliverable02.qmd`.
4. Record final decisions and validation stats back into this status file.

## Key Files

- Extraction: `code/extract/extract_reviews.py`
- Setup: `code/deliverable02/00_setup.R`
- Analysis: `code/deliverable02/01_reviews.R`
- Report: `reports/deliverable02.qmd`
- QA scripts: `code/exploratory/reviews/01_review_qc_overview.R`, `code/exploratory/reviews/02_tier_term_logic_checks.R`, `code/exploratory/reviews/03_tier_linkage_checks.R`
- QA outputs: `output/exploratory/reviews/`
