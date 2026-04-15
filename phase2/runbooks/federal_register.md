# Federal Register NOI Refresh

**Purpose:** Refresh Phase 2 Federal Register Notice of Intent (NOI) data and link high-confidence notices to NEPA projects using direct NEPATEC document evidence.

**Default behavior:** `extract_data.py --mode analysis` does not call the Federal Register API or scan NEPATEC pages. It merges `data/analysis/federal_register/noi_federal_register.parquet` if that file already exists, with a read-only fallback to the old top-level `data/analysis/noi_federal_register.parquet` artifact until the new subdirectory output is generated.

## How Matching Works

Matching proceeds in two phases:

1. **Direct evidence (primary):** The workflow scans EA/EIS/CE NEPATEC page text for `[FR Doc. ...]` brackets and `federalregister.gov` URLs using DuckDB. Extracted document numbers are normalized (en-dash → ASCII hyphen) and joined directly to the API corpus by `document_number`. A match is auto-accepted only if the joined API record's title or agency also aligns with the project.

2. **Title matching (review only):** Projects without direct NEPATEC doc number evidence may still have title-based candidates, but these are routed to `project_noi_manual_review.csv` only — they never populate `noi_publication_date`. This prevents prior-iteration contamination (where title matching might silently return an NOI date from an earlier review cycle).

**Key rule:** `noi_publication_date` is only ever populated from a direct FR doc number join. Title matching is a review signal, not a date source.

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
  --evidence-output phase2/data/analysis/federal_register/nepatec_fr_evidence.parquet \
  --cache-path phase2/data/analysis/federal_register/fr_noi_cache.json \
  --all-projects
```

The corpus fetch uses date-windowed API calls (year → quarter → month for capped windows). Corpus queries now include scoping notices in addition to NOIs:

- `"Notice of Intent"`, `"Intent To Prepare"`, `"Notice To Prepare"`, `"Notice of Preparation"`
- `"Notice of Scoping"`, `"Notice of Public Scoping"`, `"Scoping for Environmental Impact"`

The NEPATEC evidence scan uses DuckDB and runs in ~30–60 seconds. Results are cached to `nepatec_fr_evidence.parquet` and reused on subsequent refreshes unless `--rescan-nepatec-evidence` is passed.

## Force Rescan of NEPATEC Evidence

The NEPATEC page scan is cached independently of the API corpus. To force a fresh scan (e.g., after NEPATEC processed data is rebuilt):

```bash
conda run -n nepa python phase2/code/extract/federal_register.py \
  ... \
  --rescan-nepatec-evidence
```

## Refresh During Analysis

Re-queries the FR API, rescans NEPATEC pages (~30–60s), and re-runs matching:

```bash
conda run -n nepa python phase2/code/extract/extract_data.py \
  --mode analysis \
  --refresh-federal-register
```

The NEPATEC page scan always runs when `--refresh-federal-register` is set. To skip the rescan and reuse a cached evidence file, use the standalone script directly with no `--rescan-nepatec-evidence` flag (see Refresh Standalone above).

During refresh, the script prints progress for all three phases:

- `[FR API]`: Federal Register query/date-window/page progress. Capped windows split to quarters, then months.
- `[fr-evidence]`: NEPATEC page scan progress per process type (EA, EIS, CE).
- `[FR match]`: offline project/document matching progress.

## Outputs

- `data/analysis/federal_register/fr_noi_documents.parquet`: one row per Federal Register document, deduped by document number.
- `data/analysis/federal_register/fr_noi_fetch_report.csv`: one row per API query/date window, including capped/split flags and documents added.
- `data/analysis/federal_register/nepatec_fr_evidence.parquet`: one row per FR Doc. number found per NEPATEC page; cached across API refreshes.
- `data/analysis/federal_register/project_noi_candidates.parquet`: scored project/document candidate links.
- `data/analysis/federal_register/project_noi_manual_review.csv`: CE matches, weak-corroboration direct matches, and title-only candidates for human review. `noi_publication_date` is never populated from these rows automatically.
- `data/analysis/federal_register/noi_federal_register.parquet`: one row per NEPA project; only high-confidence EA/EIS accepted matches populate `noi_publication_date`.

## Evidence Table (`nepatec_fr_evidence.parquet`)

Key columns:

| Column | Description |
|---|---|
| `project_id`, `process_type` | Project identity |
| `document_id`, `file_name`, `main_document` | Source document |
| `page_number` | Page where evidence was found |
| `evidence_type` | `fr_doc_noi` (passed proximity filter), `fr_doc_non_noi` (failed), `fr_url` |
| `fr_document_number` | Normalized to ASCII hyphen — the join key |
| `fr_document_number_raw` | Original extracted string (may contain en-dash) |
| `nearby_noi_phrase` | NOI-like phrase within 500 chars (if any) |
| `fr_date_text_parsed` | ISO date from prose ("Published in the Federal Register on...") |

Only `fr_doc_noi` and `fr_url` evidence types are used as join keys. `fr_doc_non_noi` rows are retained for diagnostics but excluded from accept/review decisions.

## Acceptance Rules

| Condition | Outcome |
|---|---|
| EA/EIS: NEPATEC doc number joins API + NOI proximity passed + corroboration | Auto-accept (`noi_publication_date` populated) |
| EA/EIS: doc number joins but corroboration weak | Manual review |
| CE: any NEPATEC doc number evidence | Manual review (no auto-accept) |
| Title-only match (no NEPATEC doc number) | Manual review only (never populates `noi_publication_date`) |
| No candidates | Unmatched |

## Provenance Fields in `noi_federal_register.parquet`

| Field | Description |
|---|---|
| `noi_date_evidence_type` | `nepatec_fr_doc_number` for direct-evidence accepted matches |
| `noi_nepatec_evidence_document_id` | NEPATEC document where FR doc number was found |
| `noi_nepatec_evidence_file_name` | File name of that document |
| `noi_nepatec_evidence_page_number` | Page number of the FR doc bracket |

## Merge Key

The downstream merge remains project-level by `project_id`. The Federal Register `document_number` is stored as provenance, but is not a direct key in the NEPATEC project data.

## Coverage Expectations

EIS and EA projects with Federal Register Notice PDFs in NEPATEC are the primary beneficiaries of the direct evidence path. CE projects are scanned but all CE matches go to manual review; no CE auto-accept until coverage patterns are established. The scoping notice queries may extend coverage to projects where the formal NOI was preceded by a separate scoping notice record.
