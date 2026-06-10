# Investigation brief: why are 1,376 Phase-1 clean reviews missing from Phase 2 (D4)?

Paste everything below to Claude.

---

You are investigating a coverage regression in the Deliverable 4 (D4) timeline pipeline.
Phase 1 produced timeline dates for 20,725 clean-energy reviews. The current Phase 2 timeline
output (`phase2/data/analysis/timeline/timeline_project_dates.parquet`) covers only 19,349 of
them. **1,376 clean-energy reviews that Phase 1 had are absent from Phase 2.** I need you to
determine, with evidence, *why* each of these projects produced no Phase 2 timeline — and group
the reasons into a small number of root causes with counts.

## The target list
- `phase2/notes/deliverable04/missing.csv` — the 1,376 missing projects, columns `project_id`,
  `process_type` (Phase-1 process type: 1,360 CE + 16 EA, 0 EIS).
- **Always use the full project_id UUID** in any output — never truncate it.

## What I already know (start here, don't re-derive)
- All 1,376 ARE present in `phase2/data/analysis/timeline/timeline_document_index.parquet`
  (the Phase 2 scanned/ingested set), so this is **not** an ingestion gap.
- All 1,376 have **zero rows** in `phase2/data/analysis/timeline/timeline_candidates.parquet`.
  → The failure is at the **candidate-generation / scan stage (script 03_)**, not adjudication (06_).
- Document profile of the missing cohort (from the document index): short documents (median
  **2 pages** vs 4 in the kept cohort), 96% marked `is_main_document`, document_type_category
  overwhelmingly **`decision`** (CE determinations), lead agency dominated by **DOE (1,285)**
  and **BLM (73)**. This smells like the known CE "template-specific form layout" gap.

## Your job — confirm the mechanism and quantify it
Work the candidate-generation path for these specific projects and answer:

1. **Where exactly do they drop out?** Read the D4 pipeline scripts in
   `phase2/code/deliverable04/` (especially `03_*` candidate extraction and anything that
   filters the document index before scanning). Identify the precise filter / code path that
   excludes these projects or yields no candidates. Quote the relevant lines with file:line.
2. **Page-text reality check.** For a sample (say 30–40) of the missing project_ids, pull the
   actual page text these documents point to (via the document index `file_id`/`document_id` →
   the NEPATEC pages source the pipeline uses). Is there extractable date text on the page at
   all? Distinguish:
   - (a) **No usable text** — scanned form / OCR failure / image-only → candidate stage has
     nothing to match.
   - (b) **Text present but no regex/section match** — date sits in a form field or layout the
     `03_` patterns don't capture.
   - (c) **Excluded upstream** — caught by an exclusion list (e.g.
     `phase2/notes/agencies_to_be_excluded.txt`, military filter) or a page/length threshold.
   - (d) **Process-type or universe mismatch** — reclassified or filtered between phases.
3. **Quantify.** Give a table: root-cause category × count (covering all 1,376, CE vs EA split),
   plus the DOE/BLM/USDA breakdown within each cause.
4. **Cross-check Phase 1.** Phase 1 *did* get dates for these via
   `phase1/data/analysis/projects_timeline_bert.parquet` (CE) and
   `projects_timeline_bert_ea_llm.parquet` (EA). What did Phase 1 do differently that recovered
   them? (e.g. different page selection, a form-layout-aware extractor, LLM pass.) Be concrete.
5. **Recommendation.** For each root cause, state whether it is recoverable in Phase 2 by:
   (i) loosening a `03_` filter/threshold, (ii) adding a form-layout/date pattern, (iii) an LLM
   fallback on this cohort, or (iv) genuinely unrecoverable (no text). Estimate how many of the
   1,376 each fix would recover.

## Deliverable
A short written report (root-cause table + the file:line where they drop out + recommended
fixes ranked by projects recovered). Save it to
`phase2/notes/deliverable04/missing_investigation_findings.md`. Use DuckDB or pandas for the
parquet work; never `pd.read_parquet()` a full pages file — query/filter to the target ids.
