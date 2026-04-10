# Timeline Validation: 11a192f3-a62f-e26b-1073-dd47eb446856

**Project title:** "Clean Start" Development of a National Liquid Propane (Autogas) Refueling Network
**Source:** CE (DOE/NETL grant)
**Validated:** 2026-03-28
**Method:** test_spacy.py (regex candidates + spaCy enrichment layer, tiered patterns)

---

## True Timeline

| Role | Date | Evidence |
|---|---|---|
| **INITIATION** | **2013-09-12** | "DOE Initiator Signature: Neil Kirschner" — the DOE initiator signed the NEPA determination package on this date |
| **DECISION** | **2013-09-16** | Digital signature: "DN: cn=John Ganz, o=NETL...NEPA Compliance Officer" — NCO concurrence 4 days after initiator |
| **Duration** | **4 days** | Typical for DOE CE: initiator packages the determination, NCO reviews and concurs quickly |

Note: The grant performance period (Aug 2010 – Sept 2013) is the project period, not the NEPA review window. The formal NEPA determination was completed entirely in September 2013.

---

## All 6 Candidates

| # | Date | Match | Type | Label | Confidence | Tier | Pattern | Context |
|---|---|---|---|---|---|---|---|---|
| 1 | 2010-08-01 | Aug. 2010 | prose | unknown | low | unknown | — | Recipient Name: Texas State Technical College Sub-recipient Name: CleanFUEL Holdings, Inc. FY/Performance Period: FY13/Aug. 2010 Sept. 2013 Project Location… |
| 2 | 2011-11-07 | 11/7/2011 | form | boilerplate | low | boilerplate | `previous editions obsolete` | NETL F 451.1-1/1 Revised: 11/7/2011 Reviewed: 11/7/2011 (Previous Editions Obsolete) U.S. DEPARTMENT OF ENERGY - NETL |
| 3 | 2011-11-07 | 11/7/2011 | form | boilerplate | low | boilerplate | `previous editions obsolete` | (duplicate of row 2) |
| 4 | 2013-09-01 | Sept. 2013 | prose | unknown | low | unknown | — | Sub-recipient Name: CleanFUEL Holdings, Inc. FY/Performance Period: FY13/Aug. 2010 Sept. 2013 Project Location… |
| 5 | 2013-09-12 | 09/12/2013 | form | **initiation** | **high** | **strong** | `doe initiator signature` | DOE Initiator Signature: Neil Kirschner Therefore, the proposed action may be categorically excluded from further NEPA review. DOE Initiator Signature: Neil Kirschner Date: 09/12/2013 |
| 6 | 2013-09-16 | 2013.09.16 | form | **decision** | **high** | **strong** | `nepa compliance officer` | DN: cn=John Ganz, o=NETL, ou=ECD, email=john.ganz@netl.doe.gov, c=US Date: 2013.09.16 13:51:57 -0400 NEPA Compliance Officer's Comment: |

---

## Classification Notes

**Rows 1 & 4 (Aug. 2010, Sept. 2013) — `unknown`:**
Performance period start/end dates from the DOE grant header form. The spaCy tokenizer failed to anchor the abbreviated month formats ("Aug.", "Sept.") as date tokens. Even if it had, the context is a structured grant header with no NEPA-relevant verb signal. These dates represent the project funding period, not the NEPA review — correctly excluded.

**Rows 2 & 3 (11/7/2011) — `boilerplate`:**
Form version/revision dates from the NETL form header ("NETL F 451.1-1/1 Revised: 11/7/2011, Reviewed: 11/7/2011, Previous Editions Obsolete"). Correctly identified as boilerplate and filtered. These are form management dates, not NEPA timeline dates.

**Row 5 (2013-09-12) — `initiation/high/strong`:**
`"DOE Initiator Signature"` matched `INITIATION_PATTERNS_STRONG` pattern `doe initiator signature`. This is a DOE-specific form field that identifies the person who initiated the NEPA review. The true initiation date. Pipeline correctly identified it.

**Row 6 (2013-09-16) — `decision/high/strong`:**
`"NEPA Compliance Officer"` matched `DECISION_PATTERNS_STRONG` pattern `nepa compliance officer`. The YYYY.MM.DD digital signature timestamp format also matches `\d{4}\.\d{2}\.\d{2}` in the same pattern list. John Ganz (NCO) concurred with the CE determination 4 days after the initiator. The true decision date. Pipeline correctly identified it.

---

## Pipeline Performance on This Project

| Metric | Result |
|---|---|
| True initiation found in candidates | Yes (row 5) |
| True initiation correctly labeled | Yes — `initiation/high/strong` |
| True decision found in candidates | Yes (row 6) |
| True decision correctly labeled | Yes — `decision/high/strong` |
| Boilerplate correctly filtered | Yes — rows 2 & 3 |
| Performance-period dates excluded | Yes — rows 1 & 4 (`unknown/low`) |
| Clean candidates passed to LLM | 2 of 6 — exactly the right two |
| LLM selection task | Trivial: 1 initiation candidate, 1 decision candidate, both `high/strong` |

---

## Contrast with Previous (broken) Version

Before adding the full tiered pattern sets, this project returned **0 clean candidates** because the simplified `DATE_CONTEXT_KEYWORDS` dict lacked DOE-specific vocabulary. The fix was importing `DECISION_PATTERNS_STRONG` and `INITIATION_PATTERNS_STRONG` verbatim from `extract_timeline.py`, which already contain `doe initiator signature` and `nepa compliance officer` as strong signals.
