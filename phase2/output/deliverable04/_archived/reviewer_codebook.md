# Timeline Review Codebook — D4 Human QC

Version 1.0 · Derived from inter-rater analysis of sample_100 review (R1 vs R2, May 2026)

---

## Why this codebook exists

A structured comparison of two independent reviewers on the same 100 projects found 35%
exact agreement, rising to 55% when a ≤3-month tolerance was applied. The gap was almost
entirely explained by three calibration failures — not genuine analytical disagreement:

| Root cause | Projects | Share of disagreements |
|---|---|---|
| Day-precision noise (same milestone, different day) | 5 | 8% |
| Evidence-threshold difference (one reviewer blank, other had a date) | 28 | 43% |
| Milestone-interpretation difference (both had dates, 1–12 mo gap) | 25 | 38% |
| Fundamental / structural disagreement (>12 months) | 7 | 11% |

The rules below eliminate causes 1–3. Cause 4 (genuine hard cases) still requires
adjudication, and a flag for those is built into the protocol.

---

## Part 1 — Evidence Sufficiency Tiers

**Never leave a field blank when Tier 1, 2, or 3 evidence exists.** Record the date and
note the tier in the source field. Only leave blank when no evidence above Tier 4 is present.

### Tier 1 — Record unconditionally
- Authorizing-official digital or wet signature with explicit date (Field Manager, Forest
  Supervisor, NEPA Compliance Officer, Regional Director, etc.)
- BLM NEPA Register project start or decision date
- DOE CX Register or NEPA Workflow determination date
- Federal Register Notice of Intent (NOI) publication date (for EIS initiation)
- Federal Register Notice of Availability (NOA) publication date (for FEIS/FONSI)

### Tier 2 — Record; note as "document date"
- Explicit date statement in document body (e.g. "A Preliminary EA was published on
  November 10, 2018"; "The ROD was signed on July 29, 2016")
- FONSI/Decision Notice/ROD document header date when the specific month-year is stated
  and no signature date is present
- 30-day post-FEIS waiting period expiry (computed: FEIS NOA date + 30 days)

### Tier 3 — Record; flag as "proxy"
- Document cover/header month only (e.g. "March 2020" on cover page — use YYYY-MM-01)
- CE/EA NEPA number year when no other date is available (use YYYY-07-01 as mid-year proxy)
- Filename date (YYYY-01-01 proxy pattern) — only use if no body-text date found

### Tier 4 — Do NOT use as the primary date; may cite in notes
- Pre-NEPA project activity (applicant proposals, feasibility studies, state permits)
- Section 106 (cultural resources) consultation dates
- Section 7 (ESA) consultation dates
- State/local environmental review decisions (CEQA, state permits)
- Dates from supporting documents that are not the NEPA decision document
- Court orders, litigation dates, legislative authorizations

---

## Part 2 — Milestone Priority Hierarchy

Apply these rules in order. Stop at the **first tier that has evidence** and record that date.
Do not skip to a later tier because an earlier-tier date seems "wrong" — document your
reasoning in the notes instead.

### Initiation priority (highest → lowest)

1. **Federal Register NOI publication date** — the gold standard for EIS initiation
2. **BLM / DOE agency NEPA Register start date** — authoritative for CEs and EAs
3. **Formal scoping notice or public scoping period start date** (FR or newspaper publication)
4. **Draft EA / scoping letter sent to agencies and public** — first formal public NEPA act
5. **Application or formal request received date**, if explicitly stated in the NEPA record
   as triggering the review (not applicant files)
6. **Tribal / agency consultation letters sent date**, only if labelled as NEPA scoping
7. **Draft document publication month** as Tier 3 proxy
8. **Document header / cover year-month** as Tier 3 proxy

**Do not use** as initiation:
- Date a private party submitted an application to a non-federal entity
- Date a study, biological survey, or cultural resources survey was conducted
- Date a state commission approved a project
- Date an MOU, MOA, or interagency agreement was signed

### Decision priority (highest → lowest)

1. **Authorizing official digital or wet signature date** (Field Manager, Supervisor, etc.)
2. **BLM / DOE agency NEPA Register decision date**
3. **DOE CX Register or NEPA Workflow determination date**
4. **Explicit ROD / FONSI / Decision Notice signing date** stated in document body
5. **Federal Register NOA for FONSI or ROD**
6. **30-day post-FEIS waiting period expiry** (FEIS NOA + 30 days; earliest possible ROD)
7. **FEIS / Final EA publication month** as Tier 2 proxy (note: this precedes the ROD)
8. **Document cover month** as Tier 3 proxy

**Do not use** as decision:
- Date a Section 106 MOA or SHPO agreement was signed
- Date a Section 7 biological opinion or concurrence letter was issued
- Date a state governor signed a related bill or EO
- DEIS publication date (this is never a decision)
- Date a court opinion was issued

---

## Part 3 — Day-Precision Convention

Exact date known (day stated in document): record `YYYY-MM-DD`  
Month only (document header, "March 2019"): record `YYYY-MM-01`  
Year only: record `YYYY-01-01` and note in source field  
Date range stated ("spring 2010"): use first day of estimated mid-month (`2010-04-01`)

This eliminates same-month, different-day disagreements. Both reviewers should produce
identical dates from the same document when this convention is applied.

---

## Part 4 — Multi-Agency and Parallel-Review Projects

When multiple agencies are involved, record the **lead agency's** NEPA decision, not a
cooperating agency's parallel action.

| Situation | What to record |
|---|---|
| BLM EA + FHWA FONSI (highway ROW) | BLM Field Manager signature date |
| DOE EA + WAPA FONSI | DOE/WAPA authorizing official signature |
| Corps + EPA joint EIS | Corps District Commander ROD |
| HUD Part 58 EA (state/local RE) | Responsible Entity's FONSI/public notice date |
| Forest Service EA + BLM CE (same corridor) | Each agency's separate decision; record lead's |

Note the cooperating-agency date in `review_decision_source` for reference.

---

## Part 5 — Process-Type-Specific Rules

### Categorical Exclusions (CEs)

- **Initiation**: BLM Register start date if available; otherwise the date the application
  or project proposal was formally received by the agency (stated in CE body).
  The CE determination date itself is NOT the initiation date.
- **Decision**: Authorizing official (Field Manager / authorized officer) signature date.
  Preparer and reviewer initials are NOT the decision — look for the Field Manager line.
  Note: BLM CE forms include the statement "does not constitute an appealable decision" —
  this refers to *appeal rights*, not whether it is a NEPA decision. It **is** the decision.
- If start and decision are the same day (common for simple CEs): record the same date
  for both fields.

### Environmental Assessments (EAs)

- **Initiation**: NOI (rare for EAs) > scoping letters > draft EA publication date.
  For NRCS/BOR watershed EAs, the scoping/consultation letter date is standard.
  For DOE/WAPA interconnection EAs, the formal request submission date is the trigger.
- **Decision**: FONSI signature date > Decision Record > Decision Notice.
  For Forest Service EAs subject to 36 CFR 218 objection review: the Decision Notice
  is signed **after** objection resolution; use the Decision Notice signing date, not the
  objection period start.

### Environmental Impact Statements (EISs)

- **Initiation**: FR NOI date is authoritative. Scoping period start ≈ NOI date.
  For RMPs/programmatic EISs, the NOI may predate the project-specific EA/EIS by years —
  use the NOI for **this specific action**, not a tiering parent document's NOI.
- **Decision**: ROD signing date > FR NOA for ROD. The FEIS publication date is NOT the
  decision; it opens the 30-day waiting period. If no ROD is signed yet, record blank and
  note "FEIS published [date]; ROD pending."

---

## Part 6 — When to Leave a Date Blank

Leave blank **only** when:
- All available evidence is Tier 4 (pre-NEPA activity, state permits, ESA dates, etc.)
- The project is in the wrong category (e.g., a CE has no identifiable initiation
  separate from the decision, and no Register start date exists — leave initiation blank
  but record the decision)
- The packet contains no document text at all

Do **not** leave blank because:
- The evidence is only Tier 2 or Tier 3 (record it and note the confidence)
- The date "seems early" or "seems late" relative to your expectations
- The automated suggestion seemed wrong (make your own determination)

---

## Part 7 — Calibration Set

The 12 examples below are drawn from the 35 projects where both reviewers agreed
exactly. Review these before starting to calibrate your thresholds.

### CE examples (clear)

**[SID 26] WYWY 106322307 Chipcore 2" Water Pipeline ROW**  
Init: 2023-10-03 — BLM Register start date (Tier 1)  
Dec:  2023-12-15 — Digital signature "DARCI NATION Date: 2023.12.15 14:05:38" (Tier 1)  
Note: The CE document header references "July 2024" — a later document re-issue. The
digital signature date overrides any header date.

**[SID 44] L18K Earthscope Seismic Station Assignment**  
Init: 2019-03-18 — BLM Register start date (Tier 1)  
Dec:  2019-04-16 — "Bonnie Million Anchorage Field Manager -4/16/2019 Date" (Tier 1)

**[SID 29] BH Buildings 802 and 812 Damage Inspection**  
Init: blank — no application or register start date in packet  
Dec:  2010-01-07 — "Approved by SPRPMO NEPA Compliance Officer 01/07/10 Determination Date" (Tier 1)

### EA examples (clear)

**[SID 38] Beale WAPA Interconnection**  
Init: 2016-03-08 — "On March 8, 2016, Beale AFB submitted an interconnection request to WAPA"
  (Tier 2 — formal request documented in NEPA record as the trigger)  
Dec:  2020-11-30 — "SONJA ANDERSON Digitally signed Date: 2020.11.30 12:14:03" (Tier 1)

**[SID 40] East Fork Irrigation District Infrastructure Modernization**  
Init: 2019-03-12 — Consultation letters sent to SHPO and tribes March 12, 2019 (Tier 2 —
  NRCS EA scoping; consultation letters are the formal NEPA start for NRCS EAs)  
Dec:  2020-11-03 — "SCOTT ARMENTROUT Digitally signed Date: 2020.11.03 13:37:34" (Tier 1)

**[SID 96] Sierra Scenic Byway Roadside Hazard Abatement**  
Init: 2021-10-28 — "Public Involvement Project scoping was initiated on October 28, 2021" (Tier 2)  
Dec:  2022-08-05 — "08/05/2022 Dean A." Decision Notice signing (Tier 1)

### EIS examples (clear)

**[SID 12] Durham-Orange Light Rail Transit Project**  
Init: 2012-04-03 — "A Notice of Intent (NOI) was published in the Federal Register on
  April 3, 2012" (Tier 1)  
Dec:  blank — project cancelled; no ROD issued

**[SID 21] Kachess Drought Relief Pumping Plant**  
Init: 2013-10-30 — "Reclamation published in the Federal Register a Notice of Intent (NOI)
  to prepare an EIS [on October 30, 2013]" (Tier 1)  
Dec:  blank — FEIS published; ROD not confirmed in packet

**[SID 99] Mariana Islands Training and Testing**  
Init: 2011-09-16 — "FR NOI publication date: 2011-09-16; scoping period began with the
  issuance of the Notice of Intent in the Federal Register on 16 September 2011" (Tier 1)  
Dec:  blank — FEIS published May 2015; no ROD date in packet

### Known hard cases (for calibration)

**[SID 43] South Powder River Basin Coal** ← data-mixing alert  
The packet contains ROD text for "NARO NORTH FEDERAL COAL LEASE APPLICATION WYW150210,"
a separate coal lease whose documents were mixed in. Discard dates from that ROD.
The correct initiation is the Sept 12, 2000 Federal Register notice of lease application
receipt. The correct decision anchor is the Dec 24, 2003 FEIS NOA.

**[SID 17] Sutter Co. CO2 Capture and Storage** ← likely data-mixing  
One reviewer recorded 2014-11-24 as the decision date. No candidate in the packet supports
this date. The CE Register clearly shows 2023-02-22 (digital signature). Reject the 2014
date; record 2023-02-22.

---

## Part 8 — Quick-Reference Checklist

Before submitting your review for each project:

- [ ] Did I check for a **digital/wet signature** before using a document header date?
- [ ] Did I check the **BLM / DOE Register** entry before assigning any proxy date?
- [ ] If I left a field blank, is there truly **no Tier 1–3 evidence** — or did I apply
      an evidence threshold that was too strict?
- [ ] If I have a month-only date, did I use **YYYY-MM-01** not a mid-month day?
- [ ] For multi-agency projects, did I record the **lead agency's** decision?
- [ ] Did I confirm I am reading **this project's** documents and not a related/tiering
      document from a different action mixed into the packet?
- [ ] Is my "initiation" truly a **NEPA milestone** (NOI, scoping, Draft EA) and not a
      pre-NEPA activity (proposal, survey, ESA consultation, Section 106)?
- [ ] Is my "decision" the **final NEPA act** (ROD, FONSI, CX determination) and not
      an interim step (DEIS, FEIS, scoping report, Section 106 MOA)?
