# Timeline Validation: c87a153c-f0c6-bd71-17e1-7e01ea9816a5

**Project title:** Geotechnical Investigation for the Southline Transmission Line Project
**Source:** CE
**Validated:** 2026-03-28
**Method:** test_spacy.py (regex candidates + spaCy enrichment layer)
**Cluster year (median of all candidates):** 2018

---

## True Timeline

| Role | Date | Evidence |
|---|---|---|
| **INITIATION** | **2018-05-01** | "On May 1, 2018, Southline Transmission, LLC, submitted an application for a right-of-way grant (Serial Number AZA03568101) from the Bureau of Land Management (BLM)" |
| **DECISION** | **2018-07-16** | "Date: 7/16/18 Scott C. Cooke Field Manager" — authorizing official signature at end of CE document |
| **Duration** | **76 days** | ~2.5 months, plausible for a CE geotechnical investigation |

---

## All 15 Candidates

| # | Date | Match | Type | Label | Confidence | Historical | Position % | Context |
|---|---|---|---|---|---|---|---|---|
| 1 | 1992-09-01 | September 1992 | prose | historical | low | yes — 3 dates in snippet, 26yr before cluster | 45% | Plan (RMP), Final Environmental Impact Statement (August 1991), and partial Records of Decision approved September 1992 and July 1994. |
| 2 | 2015-10-01 | October 2015 | prose | final | medium | no | 57% | which was analyzed in the Final Environmental Impact Statement (FEIS) dated October 2015. Proper implementation of the PCEMs and controls |
| 3 | 2016-08-22 | August 22, 2016 | prose | decision | high | no | 2% | Title: Geotechnical Investigation for the Southline Transmission Line Project approved on August 22, 2016 |
| 4 | 2018-05-01 | May 1, 2018 | prose | initiation | high | no | 11% | On May 1, 2018, Southline Transmission, LLC, submitted an application for a right-of-way grant (Serial Number AZA03568101) from the Bureau of Land Management (BLM) for geotechnical investigation along |
| 5 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 51% | 215(a) Have significant impacts on public health or safety. 6/14/2018 The proponent would be provided a Health and Safety Plan which would |
| 6 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 60% | rivers; or other ecologically significant or critical areas. 6/14/2018 Impacts would be negligible to minor because the routes would follow |
| 7 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 70% | 215(b) Have significant impacts on migratory birds; or other ecologically significant or critical areas. 6/14/2018 Impacts on wildlife would be minor. |
| 8 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 81% | environmental effects or involve unique or unknown environmental risks. 6/14/2018 The proposed action to bore holes for geotechnical studies does not have |
| 9 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 85% | in principle about future actions with potentially significant environmental effects. 6/14/2018 This action does not set a precedent for future action nor does it |
| 10 | 2018-06-14 | 6/14/2018 | form | unknown | low | no | 87% | insignificant but cumulatively significant environmental effects. 6/14/2018 The proposed action to drill bore holes for geotechnical studies does not |
| 11 | 2018-06-14 | 6/14/18 | form | decision | medium | no | 47% | in accordance with the decision of this RMP. " Project Lead 6/14/18 Date III. RESOURCE PROGRAM CONSULTATION & COORDINATION |
| 12 | 2018-06-22 | 6/22/18 | form | unknown | low | no | 99% | [List using bullets points; if none, state "None" or "N/A"] NEPA Coordinator: Date: 6/22/18 Assistant Field Manager: Assistant Field Manager: |
| 13 | 2018-07-02 | 7/2/18 | form | decision | medium | no | 99% | Date: 6/22/18 Assistant Field Manager: Recommended Rebula hopez Date: 7/2/18 V. DECISION Authorized Official: |
| 14 | 2018-07-11 | 7/11/18 | form | unknown | low | no | 91% | Signature Date Yes No Yes No 1. NRHP/Cultural X 7/11/18 2. TES Species X Els 3. Floodplains/Wetlands |
| 15 | 2018-07-16 | 7/16/18 | form | decision | medium | no | 100% | Date: 7/2/18 V. DECISION Authorized Official: Satt lake Date: 7/16/18 Scott C. Cooke Field Manager 6 |

---

## Classification Notes

**Row 1 (1992-09-01):** Background reference to prior RMP/FEIS/ROD history. Correctly filtered as historical by both signals (multi-date snippet + temporal outlier). Not a candidate.

**Row 2 (2015-10-01):** FEIS publication date from a prior EIS that this CE tiers from. Label `final` is correct — it's a reference document date, not part of the current CE timeline. spaCy found verb `date` (not in VERB_LABEL_MAP), fell back to keyword which matched "FEIS" → `final`.

**Row 3 (2016-08-22):** Appears in the document title ("approved on August 22, 2016"). Labeled `decision` correctly. However this predates the 2018 application, so it likely refers to a prior approval of a related project or document — not the CE decision for this review. Should be treated with caution by the ranker.

**Row 4 (2018-05-01):** Application submission date. Correctly labeled `initiation` with `confidence=high`. The true initiation date. spaCy found verb `submit` → `initiation`.

**Rows 5–10 (2018-06-14, six instances):** Date appears in a NEPA criteria checklist (section 215 items), one row per criterion. The date is a stamped audit timestamp, not a timeline event. Correctly filtered as `unknown/low`. Noise — should be excluded from BERT input.

**Row 11 (2018-06-14):** Project Lead signature line. Keyword matched "decision" from "decision of this RMP" (a reference, not this CE's decision). Marginal — should rank below rows 13 and 15.

**Row 12 (2018-06-22):** NEPA Coordinator sign-off. `unknown` because "NEPA Coordinator" and "Date:" are not in the keyword list. Should arguably be `decision`. A gap in the keyword coverage — candidate for adding "NEPA Coordinator" to DECISION_PATTERNS_STRONG.

**Row 13 (2018-07-02):** Intermediate recommendation signature ("Recommended"). `decision/medium` is plausible but this is not the final authorizing signature.

**Row 14 (2018-07-11):** Consultation checklist checkbox date. `unknown/low` is correct — this is a review/section 106 consultation tick, not a timeline event.

**Row 15 (2018-07-16):** Field Manager Scott C. Cooke's final signature. The true CE decision date. Correctly labeled `decision/medium`. The correct answer.

---

## Pipeline Performance on This Project

| Metric | Result |
|---|---|
| True initiation found in candidates | Yes (row 4) |
| True initiation correctly labeled | Yes — `initiation/high` |
| True decision found in candidates | Yes (row 15) |
| True decision correctly labeled | Yes — `decision/medium` |
| Noise filtered (unknown/low) | 9 of 15 removed from clean list |
| Historical filtered | 1 of 15 |
| Clean candidates passed to BERT | 6 of 15 |
| Biggest remaining risk | BERT must rank 2018-07-16 above 2018-06-14 and 2018-07-02 among three `decision` candidates |
