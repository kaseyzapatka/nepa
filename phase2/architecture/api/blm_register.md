# BLM National NEPA Register Integration

**Purpose:** Extract authoritative decision and initiation dates for BLM NEPA projects from the BLM National NEPA Register (eplanning.blm.gov), bypassing text extraction entirely for matched projects.

**Inputs:** NEPATEC page parquets (`phase2/data/processed/{ce,ea,eis}/`), `projects_combined.parquet`

**Outputs:** `phase2/data/analysis/blm_register/blm_eplanning_dates.parquet` — one row per BLM project with Tier A dates for the D4 timeline pipeline.

**Scripts:** `phase2/code/deliverable04/09a_scan_blm_case_numbers.py`, `09b_fetch_blm_register.py`, `09c_build_blm_dates.py`

**Why this matters:** BLM is 41% of the EA/EIS corpus (2,977 of 7,213 projects). The BLM NEPA Register contains authoritative FONSI and ROD dates that are more reliable than text-extracted dates. For matched projects these become `clear_decision` Tier A candidates at confidence 5.0, the highest in the D4 pipeline.

---

## Data Flow

```
NEPATEC pages (CE/EA/EIS)
        │
        │  DuckDB LIKE '%DOI-BLM-%' pre-filter
        │  + Python regex on matching pages
        ▼
nepatec_case_evidence.parquet
  (project_id, case_number, process_type, acceptance, ...)
        │
        │  POST /searchresults/?search_bar={case_number}  → project D365 ID
        │  GET  /Project-Home/?id={d365_id}               → date fields
        ▼
blm_register_cache.json          ← disk cache, cumulative
blm_register_records.parquet
  (case_number, start_date, fonsi_date, rod_date, noi_date, ...)
        │
        │  Join on case_number
        │  Pick best decision/initiation date per project
        ▼
blm_eplanning_dates.parquet
  (project_id, blm_decision_date, blm_initiation_date, blm_match_status, ...)
        │
        │  Merge into D4 index (01_build_timeline_index.py)
        ▼
timeline_document_index.parquet
  adds: blm_decision_tier_a_eligible, blm_initiation_tier_a_eligible
        │
        │  Build Tier A packets (02_retrieve_timeline_contexts.py)
        ▼
timeline_context_packets.parquet
  retrieval_reason: blm_register_decision | blm_register_initiation
        │
        │  Label in _prelabel_role() (03_extract_timeline_candidates.py)
        ▼
timeline_candidates.parquet
  candidate_role: clear_decision | clear_initiation  @ confidence 5.0
```

---

## Case Number Format

BLM tracks all NEPA actions with a structured case number embedded in every formal document:

```
DOI-BLM-{ST}-{OFFICE}-{YYYY}-{SEQ}-{TYPE}

ST      — 2-letter state code (AK, AZ, CA, CO, ID, MT, NM, NV, OR, UT, WY, ...)
OFFICE  — field office code (C060, P020, B010, ...)
YYYY    — 4-digit year the action was initiated
SEQ     — sequence number, zero-padded (0001, 0029, ...)
TYPE    — action type abbreviation:
          EA  = Environmental Assessment
          EIS = Environmental Impact Statement
          CX  = Categorical Exclusion (BLM's suffix for CE)
          RMP = Resource Management Plan
          DNA = Determination of NEPA Adequacy
          DR  = Decision Record
```

**OCR hazard:** The letter O and digit 0 are frequently confused in office codes. `DOI-BLM-CA-CO60-...` in NEPATEC text is almost always `DOI-BLM-CA-C060-...` in the register. The scan script normalizes `O→0` adjacent to digits in the office code segment before using the case number as a join key.

---

## BLM National NEPA Register API

The register runs on Microsoft Dynamics 365 Portal at `eplanning.blm.gov`. Despite being a D365 portal (which typically requires authentication for data access), search and project detail pages are publicly accessible via a session cookie + CSRF token pattern.

### Step A: Search (JSON)
```
POST https://eplanning.blm.gov/searchresults/
Content-Type: application/x-www-form-urlencoded
__RequestVerificationToken: {token}

search_bar={case_number}&download=false&get_total_count=false&filter_total_count=0
```

Returns JSON: `{"data": [{"nepanumber": "...", "projectid": "{d365_guid}", "nepastatus": "...", "type": "EA", ...}]}`

### Step B: Project page (HTML)
```
GET https://eplanning.blm.gov/Project-Home/?id={d365_guid}
```

Returns HTML with date fields in two structural patterns:

**Pattern 1 — Direct label (EA/CE):**
```
FONSI Date
01/22/2020
```

**Pattern 2 — Milestone + Actual Date (EIS):**
```
Record of Decision Publication
Actual Date
12/12/2019
```

**Date fields by process type:**

| Field | EA | EIS | CX |
|---|---|---|---|
| `start_date` | ✓ | ✓ | ✓ |
| `fonsi_date` | ✓ | — | — |
| `rod_date` | — | ✓ | — |
| `noi_date` | — | ✓ | — |
| `end_date` | sometimes | sometimes | sometimes |

### Session management
The CSRF token is fetched from `/_layout/tokenhtml` before each batch. Refresh every ~50 requests. Rate limit: 1.5 seconds between case numbers (2 requests per case number — one search, one project page).

---

## Acceptance Gate

After extracting case numbers from NEPATEC pages, each (project_id, case_number) pair is evaluated:

| Condition | `acceptance` |
|---|---|
| case_type matches project process_type + single case number | `accept` |
| case_type matches + multiple case numbers for project | `review` |
| case_type mismatch + main_document evidence | `review` |
| case_type mismatch + appendix evidence | `skip` |

**CE/CX special rule:** BLM uses `CX` (and sometimes `DNA`, `DR`, `SCX`, `DN`) as the case-type suffix for categorical exclusions. Projects with `process_type = CE` accept case numbers with `case_type` in `{CE, CX, DNA, DR, SCX, DN}`.

---

## Coverage Results (as of initial build)

| Process | BLM projects | Accepted matches | Decision dates | Initiation dates |
|---|---|---|---|---|
| EA | 2,217 | 1,253 | 1,157 (52%) | 1,068 (48%) |
| EIS | 760 | 55 | 52 (7%) | 52 (7%) |
| CE | 23,039 | 14,770 | 0 (0%) | ~12,400 est. (54%) |

**EA decision date coverage as % of all EA projects** (including non-BLM): 37.5%

**Why EA hits and EIS lags:** EIS documents frequently cross-reference EA case numbers (tiered or scoping EAs), reducing clean process_match hits. EIS case numbers appear less consistently in NEPATEC headers than EA case numbers.

**Why CE has no decision dates:** The register tracks CX project initiation (start date) but not the CX determination/signing date. BLM treats CX actions as internal administrative steps with no formal publication milestone, unlike FONSI (EA) or ROD (EIS).

---

## D4 Pipeline Integration

The three Tier A flags added to `timeline_document_index.parquet`:

```python
blm_case_number               — matched case number
blm_match_status              — accepted | review | not_in_register | unmatched
blm_decision_date             — ISO date string or null
blm_decision_date_type        — fonsi | rod | decision | end_date_proxy
blm_decision_tier_a_eligible  — bool: accepted + decision_date present
blm_initiation_date           — ISO date string or null
blm_initiation_tier_a_eligible— bool: accepted + initiation_date present
```

In `_prelabel_role()` (script 03), BLM Tier A packets are labeled at the highest confidence tier:
```python
if source_tier == "metadata" and "blm_register_decision" in retrieval_reason:
    return "clear_decision", 5.0, ["blm_register_tier_a"], []
if source_tier == "metadata" and "blm_register_initiation" in retrieval_reason:
    return "clear_initiation", 5.0, ["blm_register_tier_a"], []
```

---

## Known Limitations

- **EIS scan coverage is low (11% of BLM EIS)** because EIS documents reference EA case numbers more than their own EIS case number. Many EIS projects have `multi_case_flag=True` and are routed to `review` rather than `accept`.
- **Older case numbers not_found:** Pre-2010 BLM NEPA actions were entered into ePlanning with non-standard office codes that don't match the current register. No fix available.
- **CA Coastal office code variants:** Multiple non-standard codes (`C060`, `CO60`, `060`, `C05000`) appear for the California Coastal district. OCR normalization catches `CO60→C060` but other variants may still miss.
- **CX determination dates missing:** Structural gap — the register was not designed to track CE determination dates. Text extraction from NEPATEC remains the only path for CE decision dates.
- **Cache staleness:** `blm_register_cache.json` is not date-versioned. Re-run `09b_fetch_blm_register.py --refetch --case-types EA EIS` periodically to refresh records for recently-completed projects.
