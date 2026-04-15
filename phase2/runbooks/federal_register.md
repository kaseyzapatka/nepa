# Federal Register NOI Refresh

**Purpose:** Refresh Phase 2 Federal Register Notice of Intent (NOI) data and link high-confidence notices to NEPA projects.

**Default behavior:** `extract_data.py --mode analysis` does not call the Federal Register API. It merges `data/analysis/federal_register/noi_federal_register.parquet` if that file already exists, with a read-only fallback to the old top-level `data/analysis/noi_federal_register.parquet` artifact until the new subdirectory output is generated.

## Refresh Standalone

Run this when you want to query the Federal Register API and rebuild the Phase 2 NOI artifacts:

```bash
conda run -n nepa python phase2/code/extract/federal_register.py \
  --projects-path phase2/data/analysis/projects_combined.parquet \
  --output phase2/data/analysis/federal_register/noi_federal_register.parquet \
  --corpus-output phase2/data/analysis/federal_register/fr_noi_documents.parquet \
  --candidates-output phase2/data/analysis/federal_register/project_noi_candidates.parquet \
  --review-output phase2/data/analysis/federal_register/project_noi_manual_review.csv \
  --fetch-report-output phase2/data/analysis/federal_register/fr_noi_fetch_report.csv \
  --cache-path phase2/data/analysis/federal_register/fr_noi_cache.json \
  --all-projects
```

The corpus fetch uses date-windowed API calls because broad Federal Register searches can hit result/page caps and silently underrepresent recent years. The default windowing starts year-by-year and recursively splits capped windows to quarters, then months.

## Refresh During Analysis

Use this only when you want the analysis build to refresh API data before merging:

```bash
conda run -n nepa python phase2/code/extract/extract_data.py \
  --mode analysis \
  --refresh-federal-register
```

During refresh, the script prints progress for both phases:

- `[FR API]`: Federal Register query/date-window/page progress. Broad queries are split into year windows, and capped windows are split into quarters, then months if needed.
- `[FR match]`: offline project/document matching progress after the API corpus is saved.

## Outputs

- `data/analysis/federal_register/fr_noi_documents.parquet`: one row per Federal Register document, deduped by document number.
- `data/analysis/federal_register/fr_noi_fetch_report.csv`: one row per API query/date window, including count, total pages, cached pages, capped/split flags, and documents added.
- `data/analysis/federal_register/project_noi_candidates.parquet`: scored project/document candidate links.
- `data/analysis/federal_register/project_noi_manual_review.csv`: medium-confidence and ambiguous matches for review.
- `data/analysis/federal_register/noi_federal_register.parquet`: one row per NEPA project; only high-confidence accepted matches populate `noi_publication_date`.

## Merge Key

The downstream merge remains project-level by `project_id`. The Federal Register `document_number` is stored as provenance and matching evidence, but it is not a direct key in the NEPATEC project data.

## Coverage Expectations

The refresh includes CE, EA, and EIS projects across all energy categories. Most accepted NOI dates are expected to be EIS and some EA records; CE coverage may remain low because CE projects usually do not follow the same NOI publication workflow.
