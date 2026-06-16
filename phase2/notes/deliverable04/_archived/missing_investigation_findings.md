# D4 Timeline — Coverage-Regression Investigation: Findings

**Question:** Phase 1 produced timeline dates for 20,725 clean-energy reviews; Phase 2's
`timeline_project_dates.parquet` covers 19,349. Why are **1,376** clean-energy reviews
(1,360 CE + 16 EA, 0 EIS) absent from Phase 2?

**Answer (one line):** The Phase 2 CE scan **reads too few characters per page** — the candidate
date sits below char 2,000 on dense ~7k-char CE form pages and is truncated off before extraction.
That single mechanism explains ~94% of the loss. The 16 EA cases are a separate, mostly
non-regression mix (image-only decision PDFs, genuinely date-less docs, and cases where Phase 1
had a *wrong* date that Phase 2 correctly declined).

The fix plan for CE is in **[`../missing_investigation_CEplan.md`](../missing_investigation_CEplan.md)**.
This document is the evidence + EA detail.

---

## 1. Where they drop out (the mechanism)

All 1,376 are present in `timeline_document_index.parquet` but have **zero rows** in
`timeline_candidates.parquet` → failure is at candidate generation
(`02_retrieve.py` → `03_extract_candidates.py`), **not** adjudication (`06_`).

The chain (CE):

1. `01_index.py` does not score a bare `document_type_clean == "CE"`, so DOE/BLM CE-determination
   docs get `decision_doc_score = 0` → `scan_priority = priority_3`. (1,212 missing CE docs match
   this exactly.)
2. `priority_3` docs are scanned **only** by `build_tier_d_packets`
   (`02_retrieve.py` ≈ line 798); `build_tier_b_packets` — which already reads CE pages at full
   length — is restricted to `priority_1`/`priority_2` (line ≈558) and never touches them. **1,030
   of the recoverable CE projects have *only* `priority_3` docs.**
3. `build_tier_d_packets` stores **`"context_text": _truncate(text, 2000)`** (line ≈863).
4. The CE determination signature date (NEPA Compliance Officer / Field Office Manager block) is at
   the **bottom** of a dense form page; median full page = **7,107 chars**, so the date is **beyond
   char 2,000** and is cut off.
5. `03_`'s `extract_candidates_from_packet` only sees the 2,000-char slice → no date → no candidate.

A secondary loss: `_should_reject_date` (`03_` lines ≈525–566) scans the **whole block** for
`EXCLUSION_KEYWORDS`, so the real signature date is killed when it shares a block with a CE-form
`"categorical exclusion expires"` / `"expiration date"`, a statute citation, or a URL.

## 2. Root-cause table (all 1,376; CE / EA split)

| Bucket | Cause | n | CE | EA | DOE | BLM/DOI | USDA | Recoverable |
|---|---|---|---|---|---|---|---|---|
| **B** | **Truncation** — date present in full page text but beyond char 2,000 | **1,034** | 1,029 | 5 | 1,015 | 13 | 4 | **Yes** (raise cap) |
| **C** | **All dates rejected** — signature date shares a block with an exclusion keyword | 261 | 261 | 0 | 239 | 22 | 0 | Mostly (~251) — window the exclusion check |
| **A** | **No packets** — all of the project's docs are `defer` (never scanned) | 46 | 42 | 4 | 12 | 31 | 3 | Partly |
| **D** | **No parseable date** anywhere in the scanned pages (incl. image-only) | 35 | 28 | 7 | 24 | 8 | 1 | No |

**Page-text reality check** (37 a/b/c/d categories from the prompt): of 1,321 CE missing-cohort
packet pages, **1,293 (98%) have a parseable date in the FULL page text**, and for **1,178 (89%)
the first date is beyond char 2,000** (category **b — text present, layout/position the patterns
don't reach because of truncation**). Only **28** pages have no date at all (category **a/d**).
Category **c** (excluded upstream) ≈ the 46 no-packet projects (all-`defer` docs). No evidence of
category **d** process/universe mismatch — process types are stable across phases.

## 3. What Phase 1 did differently (cross-check)

Phase 1 read the **whole page/document** and caught the bottom-of-form signature date. Its
`bert_decision_date_source` for this CE cohort is literally the CE signature block:
- `"NEPA Compliance Officer Signature: … Date: 3/31/10 FIELD OFFICE MANAGER DETERMINATION"`
- `"Approved by Jason Anderson, DOE-ID NEPA Compliance Officer, on 09/17/2021."`
- `"Digitally signed by PETER SIEBACH Date: 2022.09.22"`

Phase 1 had **1,209 decision + 254 initiation** dates for the CE cohort. Phase 2 truncates the page
before the extractor sees that text — that is the entire CE regression. **Recovery simulation**
(running the real `03_` extractor on the full page text) yields a candidate for **1,029** CE
projects that currently get none.

## 4. Recommendations — CE (detail in the CE plan)

| Fix | File:line | Recovers (CE) | Risk |
|---|---|---|---|
| **1. Raise CE `tier_d` truncation cap** (2,000 → full page, e.g. 30k) | `02_retrieve.py` `build_tier_d_packets` ≈863 | **~1,029** | Low; keep EA/EIS at 2,000 to leave them unchanged |
| **2. Window the exclusion-keyword check** (±60 chars around the date, CE-gated) | `03_extract_candidates.py` `_should_reject_date` ≈525–566 + call site ≈821 | **~251** | Medium; gate to CE, validate vs regression set |
| **3. Rescue `defer` CE main docs** (or floor decision score on `document_type_category=="decision"`) | `01_index.py` `_compute_scores` ≈198 | ~30–40 | Medium; changes scan volume |
| — | Residual (image-only / genuine expiration-only) | ~40 unrecoverable | — |

Cumulative: Fix 1 → ~76%, +Fix 2 → ~94%, +Fix 3 → ~97% of the 1,360 CE.

## 5. The 16 EA — separate story, mostly NOT a true regression

EA is heterogeneous. Three distinct problems, and in several cases **Phase 2's absence is more
correct than Phase 1's wrong date**:

- **Image-only decision PDFs (6):** the FONSI/ROD exists but its `page_text` is **empty**
  (un-OCR'd scan) — `avglen = 0`. The decision date is locked in an image, so neither
  `ea_decision_full_read` nor any text path can reach it. 3 of these are also among the **4
  no-packet EA projects** (their only priority doc is the image-only FONSI; their narrative doc is
  `defer`), so they fall through entirely.
- **Truncation (same as CE) on the EA narrative doc (~6):** scanned at `tier_d` 2,000 chars; the
  date is below the cut. Extending Fix 1 to EA `tier_d` makes these yield candidates.
- **Genuinely date-less (3):** `7b598`, `c5310` (draft-EA only), `db787` (FONSI has text but no
  parseable decision date) — yield **zero** candidates even on full text. Phase 1 also failed.

**Phase 1 date quality was poor for much of this cohort** — its "dates" are often noise that Phase 2
correctly rejects:

| project_id | Phase-1 decision date | Phase-1 source (what it actually grabbed) | Verdict |
|---|---|---|---|
| 0937f672…0458635 | 2009-09-07 | `[LIRR] Long Island Railroad. Official Timetables` | **Wrong** (timetable); image-only FONSI |
| 18fa1b28…99e986f | 2020-09-24 | `… Revision Date: 09/24/2020 …` | **Wrong** (revision-date stamp); image-only FONSI |
| 7eb0b766…2439239 | 2016-01-05 | `… final 4(d) rule dated January 5, 2016 …` | **Wrong** (reg citation); recoverable via Fix 1 |
| 0d717f52…239d1b2 | 2007-09-01 | `Final Programmatic EIS Record of Decision … Sept 2007` | Weak (programmatic ROD ref); image-only FONSI |
| cea21587…d30a7d494 | 1996-09-06 | `an EIS wi~ not be prepar~, and BPA is issuing this FONSI` | Garbled OCR; recoverable via Fix 1 |
| cbeef882…d45e33306 | 2009-11-14 | `USFWS … to BLM Rawlins Field Manager` | Letter/proxy; recoverable via Fix 1 |
| 6 others | None / init-only | — | Phase 1 had no decision date either |

All EA decision-doc **filename dates are year-only** (`YYYY-01-01`), which the Tier A filename path
deliberately skips (`02_retrieve.py` ≈510), so filenames cannot rescue them.

### EA recommendations (low priority — only 16 projects)
1. **Extend Fix 1 (raise `tier_d` cap) to EA** → ~6 EA projects gain candidates (mostly proxy dates
   from the narrative doc). Validate against existing EA coverage before shipping (EA coverage is
   currently good; a blanket EA `tier_d` raise adds candidates corpus-wide, so re-run the EA
   regression check).
2. **Image-only FONSIs/RODs (6):** genuinely unrecoverable from text. Options — accept as a known
   gap, or run targeted OCR on these 6 documents (out of scope for the timeline pipeline). Do **not**
   reinstate Phase-1-style proxy dates: several of those were demonstrably wrong.
3. **No-packet EA (4):** apply a last-resort "scan `defer` docs when a project has zero packets"
   rule (also helps bucket-A CE), but expect low yield given the decision docs are image-only.
4. **Accept ~3 as unrecoverable** (`7b598`, `c5310`, `db787`) — no date in text; Phase 1 had none.

**Bottom line for EA:** at most ~4–6 of the 16 are genuine, recoverable regressions (truncation);
the rest are image-only PDFs or cases where Phase 1's date was noise. The EA gap is small and
partly reflects Phase 2 being *stricter*, not worse.

## 6. Reproduction

Cohort list: `phase2/notes/deliverable04/missing.csv` (full UUIDs). For each missing project's
page-based packets, pull the **full** page text from `phase2/data/processed/{ce,ea}/pages.parquet`
(DuckDB, filtered to the cohort's `document_id`s — never `pd.read_parquet` the whole pages file) and
run `extract_candidates_from_packet` from `03_` on the full text. Bucketing: `recover` = ≥1 candidate
from full text; `rejected` = a date parses but all are rejected; `nodate` = no parseable date;
`no_packets` = absent from the packets parquet. Agency from `project_department` in the document index.
EA decision-doc text status from per-document `AVG(LENGTH(page_text))` in `ea/pages.parquet`
(`= 0` ⇒ image-only).
