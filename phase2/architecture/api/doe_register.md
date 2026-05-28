# DOE NEPA Register: Architecture

## Overview

Fetches decision (FONSI/ROD) and initiation (NOI) dates for DOE EA and EIS projects from
public energy.gov pages. Outputs a project-level date table that feeds directly into the
D4 timeline pipeline as high-confidence Tier A metadata packets.

```
NEPATEC pages (DuckDB)
        │
        ▼
01_scan_doe_doc_numbers.py  ──► doe_case_evidence.parquet
        │                         (project_id ↔ DOE/EA-NNNN or DOE/EIS-NNNN)
        │
        ├──────────────────────────────────────────────────────────┐
        ▼                                                          ▼
02_fetch_doe_register.py                           03_fetch_project_pages.py
(energy.gov listing pages)                         (individual project pages)
  /nepa/listings/eis-0001-...                       /nepa/listings/ea-NNNN-documents-...
  ROD listing: 36 pages, 80 records                 Per doc: FONSI, ROD, NOI dates
  FONSI listing: 85 pages, 233 records              Covers ~85% of doc numbers not
        │                                           found in listings
        ▼                                                          │
doe_rod_lookup.parquet (80)                                        │
doe_fonsi_lookup.parquet (233)                                     │
doe_register_records.parquet (313 initial)  ◄──────────────────────┘
        │                                    (merged/updated by 03)
        │                661 total after merge: rod=131, fonsi=449, noi=51
        ▼
04_build_doe_dates.py
  - Exact join: doc_number_norm
  - Fallback join: base_number (strips SA-NN supplement suffixes)
  - Priority: FONSI > ROD for decision; NOI for initiation
        │
        ▼
doe_eplanning_dates.parquet
  (one row per project_id, 516 rows)
        │
        ▼
D4 pipeline (01_build_timeline_index.py → 02_retrieve_timeline_contexts.py)
  source_tier="metadata", confidence=5.0
  retrieval_reason="doe_register_decision" | "doe_register_initiation"
```

---

## Data Sources

### energy.gov Listing Pages
- ROD: `https://www.energy.gov/nepa/listings/records-decision` (paginated, 36 pages)
- FONSI: `https://www.energy.gov/nepa/listings/findings-no-significant-impact` (85 pages)
- Structure: `{date}\n{DOC/NUMBER}: {Title}\n{description block}`
- Coverage: 80 unique EIS/ROD records + 233 unique EA/FONSI records = 313 total

### energy.gov Individual Project Pages
- URL: `https://www.energy.gov/nepa/listings/{ea|eis}-{NNNN}-documents-available-download`
- Contains a full document list for each EA or EIS, with dates preceding each doc entry
- Key source that raised coverage from 23.6% to 78.7% of accepted projects
- Also provides NOI/scoping dates for EIS projects (otherwise unavailable)

### NEPATEC (Document Text)
- Used only in 01 to extract doc numbers embedded in PDF text
- Regex: `\bDOE/(EA-\d{4}|EIS-\d{4}(?:[-‐](?:S\d+|SA[-\s]?\d+))?)\b`
- Not used for date extraction

---

## Identifier System

DOE EA/EIS documents carry a department-wide identifier embedded in the document header:
- EA: `DOE/EA-NNNN` (4-digit, e.g. `DOE/EA-1658`)
- EIS: `DOE/EIS-NNNN` (4-digit, e.g. `DOE/EIS-0464`)
- Supplement suffixes: `DOE/EIS-0391-SA-05` — stripped to base number for fallback matching
- CX (Categorical Exclusion): Uses portal-internal `CX-NNNNNN` identifiers not embedded
  in PDF text. No public crosswalk. Hanford site uses `DOE/CX-NNNNN,RREGION` (RL-721 form),
  which appears in ~2.1% of CE NEPATEC pages but cannot join to the DOE portal.

---

## Acceptance Gate (01)

Evidence rows are classified as `accept`, `review`, or `skip`:

| Status | Condition |
|--------|-----------|
| `skip` | Non-EA/EIS process type or no DOE doc pattern found |
| `accept` | Single unique doc number found, OR dominant doc (≥2× second + ≥3 pages) |
| `review` | Multiple doc numbers found without a clear dominant (sent to manual review) |

Acceptance rates (all DOE NEPATEC, not just clean energy):
- EA: 344/916 = 37.6%
- EIS: 172/1,648 = 10.4%

---

## Coverage Against Clean Energy Universe

After running all three scripts, for DOE clean energy projects in projects_combined:

| Process | Universe | Matched | Decision | Initiation |
|---------|----------|---------|----------|-----------|
| EA      | 338      | 150 (44.4%) | 142 (42.0%) | 3 (0.9%) |
| EIS     | 241      | 53 (22.0%)  | 26 (10.8%)  | 17 (7.1%) |
| CE      | 16,140   | —       | —        | — (no portal access) |

Key limitation: DOE initiation (NOI) coverage for EA is very low (0.9%) because energy.gov
project pages rarely list NOI dates for EA documents (which typically don't require an NOI).

---

## D4 Integration

Both decision and initiation dates are injected as Tier A metadata packets in script 02:

```python
# In 02_retrieve_timeline_contexts.py
{
    "project_id": ...,
    "source_tier": "metadata",
    "retrieval_reason": "doe_register_decision",  # or "doe_register_initiation"
    "date_str": doe_decision_date,  # ISO-8601
    "confidence": 5.0,
    "labels": ["doe_register_tier_a"],
}
```

Prelabeled in script 03 as `clear_decision` or `clear_initiation` with confidence 5.0,
overriding any BERT/LLM output for these projects.

---

## Output Schema

`doe_eplanning_dates.parquet` — one row per accepted project:

| Column | Type | Notes |
|--------|------|-------|
| project_id | str | NEPATEC project identifier |
| process_type | str | EA or EIS |
| doe_doc_number | str | e.g. DOE/EA-1658 |
| doe_match_status | str | `found` or `accepted_not_found` |
| doe_decision_date | str | ISO-8601 date (FONSI or ROD) |
| doe_decision_date_type | str | `fonsi` or `rod` |
| doe_initiation_date | str | ISO-8601 date (NOI) |
| doe_initiation_date_type | str | `noi` |
| doe_decision_tier_a_eligible | bool | True if found + decision_date not null |
| doe_initiation_tier_a_eligible | bool | True if found + initiation_date not null |
| doe_fonsi_date_raw | str | Raw FONSI date before priority selection |
| doe_rod_date_raw | str | Raw ROD date |
| doe_noi_date_raw | str | Raw NOI date |
| built_at | str | ISO-8601 UTC build timestamp |

---

## Known Limitations

1. **CX (CE) dates**: Not available. energy.gov portal does not expose CE records publicly
   without authentication. EPA CDX also requires CDX account login.
2. **DOE EA initiation**: NOI is rarely published for EA documents; initiation dates must
   come from other signals (e.g., Federal Register notices via D4 script 01).
3. **EIS coverage gap**: Only 22% of clean energy EIS projects matched, primarily because
   EIS documents often cross-reference many doc numbers, triggering multi-doc ambiguity and
   falling into `review` status.
4. **EPA CDX (EIS Notice database)**: Requires CDX authentication. Not publicly accessible.
   NOI dates for EIS are sourced from individual energy.gov project pages instead.
