# Federal Register NOI/NOA Refresh

**Purpose:** Refresh Phase 2 Federal Register data and link high-confidence notices to NEPA projects:
- **NOI** (Notice of Intent): project initiation date → `noi_publication_date`
- **NOA** (Notice of Availability): project end date (FEIS/FONSI) → `noa_availability_date`

**Default behavior:** `extract_data.py --mode analysis` does not call the Federal Register API or scan NEPATEC pages. It merges `data/analysis/federal_register/noi_federal_register.parquet` if that file already exists.

## How Matching Works

Matching is entirely document-number-driven in two steps:

1. **NEPATEC page scan:** DuckDB scans all EA/EIS/CE pages for `FR Doc.` references and `federalregister.gov` URLs. Doc numbers are proximity-filtered (500-char window) with priority:
   - `fr_doc_noi` — NOI-like phrase nearby ("notice of intent", "notice of preparation", etc.)
   - `fr_doc_noa` — NOA-like phrase nearby ("final environmental impact statement", "fonsi", etc.)
   - `fr_doc_non_noi` — no recognized phrase (excluded from matching)

2. **Direct fetch + title match:** All unique valid doc numbers (`fr_doc_noi`, `fr_url`, `fr_doc_noa`) are fetched in a single combined pass (`GET /api/v1/documents/{doc_num}.json`). Fetched records are then split by title type:
   - NOI corpus: records with NOI-type titles (Notice of Intent / Notice of Preparation / Notice of Scoping)
   - NOA corpus: records with NOA-type titles (Final EIS, FONSI, Final EA availability)

   A match is **auto-accepted** only when direct doc number evidence + ≥N title tokens + correct title type + process alignment all agree.

**Key rules:**
- `noi_publication_date` is only populated from `fr_doc_noi`/`fr_url` evidence with an NOI-type FR title.
- `noa_availability_date` is only populated from `fr_doc_noa` evidence with an NOA-type FR title and matching process type (EIS→FEIS, EA→FONSI/Final EA).
- Agency/state/sponsor alone is not sufficient — title overlap is always required.
- CE projects: all evidence goes to manual review, never auto-accepted.
- Projects with no NEPATEC doc number evidence get no FR coverage.

## Refresh During Analysis (standard command)

Re-scans NEPATEC pages and direct-fetches all FR records:

```bash
conda run -n nepa python phase2/code/extract/extract_data.py \
  --mode analysis \
  --refresh-federal-register
```

The NEPATEC page scan always runs when `--refresh-federal-register` is set (~30–60s). To skip the rescan and reuse a cached evidence file, use the standalone script directly without `--rescan-nepatec-evidence` (see below).

## Refresh Standalone

Run this directly for more control over paths and options:

```bash
conda run -n nepa python phase2/code/extract/federal_register.py \
  --projects-path phase2/data/analysis/projects_combined.parquet \
  --output phase2/data/analysis/federal_register/noi_federal_register.parquet \
  --corpus-output phase2/data/analysis/federal_register/noi_documents.parquet \
  --candidates-output phase2/data/analysis/federal_register/project_noi_candidates.parquet \
  --evidence-output phase2/data/analysis/federal_register/nepatec_fr_evidence.parquet \
  --cache-path phase2/data/analysis/federal_register/fr_noi_cache.json \
  --all-projects
```

Add `--rescan-nepatec-evidence` to force a fresh NEPATEC page scan (required after adding `fr_doc_noa` evidence support, or after NEPATEC processed data is rebuilt).

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--projects-path` | `data/analysis/projects_combined.parquet` | Input projects |
| `--output` | `federal_register/noi_federal_register.parquet` | Project-level NOI+NOA output |
| `--corpus-output` | `federal_register/noi_documents.parquet` | Directly-fetched NOI FR records |
| `--candidates-output` | `federal_register/project_noi_candidates.parquet` | All scored NOI candidates |
| `--cache-path` | `federal_register/fr_noi_cache.json` | FR API response cache |
| `--evidence-output` | `federal_register/nepatec_fr_evidence.parquet` | NEPATEC page scan evidence |
| `--all-projects` | false | Include all projects |
| `--sample` | None | Random sample size |
| `--process-types` | `""` | Comma-separated (e.g. `EIS,EA`) |
| `--energy-types` | `""` | Comma-separated |
| `--throttle-seconds` | 0.25 | Delay between FR API calls |
| `--max-candidates-per-project` | 10 | Max candidates per project |
| `--quiet-progress` | false | Suppress progress logs |
| `--report-n` | 10 | Examples to print at end |
| `--rescan-nepatec-evidence` | false | Force fresh NEPATEC page scan |

NOA corpus (`noa_documents.parquet`) and candidates (`project_noa_candidates.parquet`) are written automatically with default paths — no separate CLI arguments needed.

## Progress Output

- `[fr-evidence]`: NEPATEC DuckDB scan progress per process type (EA, EIS, CE).
- `[FR direct]`: Direct API fetch progress — shows fetched/not_found/network_calls counts.
- `[FR match]`: Offline NOI matching progress.
- `[FR noa-match]`: Offline NOA matching progress.

## Outputs

### NOI (initiation)
- `noi_documents.parquet`: FR records for NOI doc numbers fetched from API.
- `project_noi_candidates.parquet`: All scored NOI candidate links.

### NOA (availability / end-of-process)
- `noa_documents.parquet`: FR records for NOA doc numbers fetched from API (FEIS/FONSI).
- `project_noa_candidates.parquet`: All scored NOA candidate links.

### Shared
- `nepatec_fr_evidence.parquet`: One row per FR Doc. number found per NEPATEC page; cached across refreshes. `evidence_type` values: `fr_doc_noi`, `fr_doc_noa`, `fr_doc_non_noi`, `fr_url`.
- `manual_review_ambiguous_candidates.csv`: Candidate rows for projects with multiple competing high-confidence NOI candidates.
- `manual_review_accepted_low_title_overlap.csv`: Accepted NOI rows with `noi_title_overlap_count <= 1`, for spot-checking.
- `noi_federal_register.parquet`: One row per NEPA project; `noi_publication_date` and `noa_availability_date` populated where auto-accepted.

`fr_noi_fetch_report.csv` was part of the legacy keyword/windowed fetch path and is removed during refresh.

## Acceptance Rules

### NOI (initiation date)

| Condition | Outcome |
|---|---|
| EA/EIS: `fr_doc_noi` evidence + ≥N title tokens + NOI-type FR title + no process conflict | Auto-accept (`noi_publication_date` populated) |
| EA/EIS: `fr_url` evidence + ≥N title tokens + NOI-type FR title + no process conflict | Auto-accept |
| EA/EIS: doc number joins but FR title is not NOI-type | Manual review |
| EA/EIS: doc number joins but title overlap < N | Manual review |
| EA/EIS: doc number joins but process conflict | Manual review |
| CE: any NEPATEC doc number evidence | Manual review (no auto-accept) |
| No NEPATEC doc number evidence | Unmatched |

### NOA (end-of-process date)

| Condition | Outcome |
|---|---|
| EIS: `fr_doc_noa` evidence + ≥N title tokens + FEIS-type FR title + EIS project | Auto-accept (`noa_availability_date` populated) |
| EA: `fr_doc_noa` evidence + ≥N title tokens + FONSI/Final EA title + EA project | Auto-accept |
| Any: `fr_doc_noa` evidence + NOA title + title overlap < N | Manual review |
| Any: `fr_doc_noa` evidence + process mismatch (EA project + FEIS title) | Manual review |
| CE: any `fr_doc_noa` evidence | Manual review |
| No `fr_doc_noa` evidence | Unmatched |

N = required title token overlap, scaled by title length: 1→1, 2→2, 3→2, 4+→3.

## Evidence Table (`nepatec_fr_evidence.parquet`)

| Column | Description |
|---|---|
| `project_id`, `process_type` | Project identity |
| `document_id`, `file_name`, `main_document` | Source document |
| `page_number` | Page where evidence was found |
| `evidence_type` | `fr_doc_noi` (NOI phrase nearby), `fr_doc_noa` (NOA phrase nearby), `fr_doc_non_noi` (no recognized phrase), `fr_url` |
| `fr_document_number` | Normalized to ASCII hyphen — the join key |
| `fr_document_number_raw` | Original extracted string (may contain en-dash) |
| `nearby_noi_phrase` | Nearest recognized phrase within 500 chars (if any) |
| `fr_date_text_parsed` | ISO date from prose ("Published in the Federal Register on...") |

## Provenance Fields in `noi_federal_register.parquet`

### NOI fields
| Field | Description |
|---|---|
| `noi_date_evidence_type` | `nepatec_fr_doc_number` for auto-accepted direct-evidence matches |
| `noi_nepatec_evidence_document_id` | NEPATEC document where FR doc number was found |
| `noi_nepatec_evidence_file_name` | File name of that document |
| `noi_nepatec_evidence_page_number` | Page number of the FR doc bracket |

### NOA fields
| Field | Description |
|---|---|
| `noa_availability_date` | FR publication date of FEIS/FONSI notice (end-of-process signal) |
| `noa_document_number` | FR document number of the NOA record |
| `noa_match_status` | `accepted`, `review_required`, or `unmatched` |
| `noa_match_reason` | Specific reason code |
| `noa_date_evidence_type` | `nepatec_fr_doc_noa` for auto-accepted matches |
| `noa_nepatec_evidence_document_id` | NEPATEC document where NOA doc number was found |
| `noa_nepatec_evidence_file_name` | File name of that document |
| `noa_nepatec_evidence_page_number` | Page number of the FR doc bracket |
