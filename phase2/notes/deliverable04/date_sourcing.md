---
title: "D4: Date Sourcing & Provenance"
---

*As of 2026-07-15 (pipeline run `timeline_run_at` 2026-07-14). This page documents what the two timeline dates mean, the codebook used to label them, and the provenance hierarchy every published date traces to.*

Companion pages: [Coverage & Limitations](coverage_limitations.html) and [Known Issues & Deferred Items](known_issues.html).

## What the dates mean

Every project carries up to two milestone dates:

- **Initiation** — the NEPA process started (an application received, a Notice of Intent, scoping).
- **Decision** — the agency decided (a signature, a Record of Decision, a FONSI, a CE determination).

The codebook below defines what counts as each. It was used to hand-label the classifier training sample and doubles as the human-review codebook; the label reflects what a date **represents** in its surrounding text, regardless of what the regex pre-label guessed.

### initiation — the NEPA process started

- NOI published in the Federal Register ("NOI was published … which initiated the scoping period")
- Application / SF-299 / ROW / permit **filed or received** ("BLM received an application", "filed a right-of-way application on")
- Scoping started: "opened a 30-day scoping period", "scoping letter was sent", "external scoping notices sent out", internal scoping conducted, "posted to the NEPA Register / eplanning website"
- FERC "approved entry to the pre-filing process"
- DOE **Initiator** signature (the program office initiating the CX) — distinct from the NEPA Compliance Officer
- "Date Determined" used as a recovered CE initiation — see the special rule below

### decision — the agency decided

- NEPA Compliance Officer signature / "Date Determined" (when it is the operative determination)
- Field Manager / Field Office Manager / authorizing-official **authorization signature**
- "It is my decision …", Decision Record date, FONSI cover month, "DOE issued a FONSI on", ROW grant **issued**
- BLM CX form "Date:" header and **CX/Decision-Record cover months** (the CX document *is* the determination)
- USACE permit decision / appeal-options notification

### neither — everything else

- Specialist / reviewer signatures (wildlife/cultural/realty/biology/NEPA-coordinator), checklist initials
- SHPO / USFWS / tribal **consultation** dates (Section 106, ESA): concurrences, responses, meetings, BA submissions
- Comment-period **ends**, protest/objection-period ends, draft-EA/EIS **release** (mid-process), Final EIS NOA
- **EA / EIS / DEIS / FEIS / PEIS document cover months** (the decision is the FONSI/ROD, not the analysis document)
- NEPA **case numbers** parsed as dates; bibliographic / Federal Register **citations**; map/figure/drawing dates
- Permit **term/expiration** dates, prior grants/leases/RODs (historical), survey/inspection/field-visit dates
- Statistic/inventory **snapshots**, court opinions, construction-period dates, applicant POD/Mine-Plan cover dates

### Tie-breakers / conventions

- **Document cover month asymmetry:** a CX cover month → `decision` (the CX is the determination); an EA/EIS cover month → `neither` (the decision is the separate FONSI/ROD).
- **Multiple dates in one window:** label only the marked date; ignore the others.
- **Activity vs milestone:** a survey/meeting/inspection conducted "on" a date is `neither`, even if it sits next to a milestone.

### Special rule — "Date Determined" CE initiation recovery

DOE CX forms often carry both a "Date Determined: \<d1\>" and a later NEPA Compliance Officer **signature** \<d2\>. When both exist (d1 < d2):

- decision = the signature date (d2)
- initiation = the "Date Determined" date (d1), as a **proxy** initiation — a recovered CE processing-start bracket, not a true NOI/application. Confirmed deterministically in `05_select_dates.py` (must precede the decision) — no classifier or LLM involvement.

**Counter-rule (guard):** when "Date Determined" is the **only** date on the form, it can only be the **decision** — a lone Date-Determined is never promoted to initiation.

**Register-conflict convention:** when a DOE CX register determination date coincides with the document "Date Determined" and a later signature exists, the recovery still applies — the later signature becomes the decision and the Date Determined (= register date) becomes the proxy initiation. This intentionally overrides the register-date-as-decision default for this specific pattern.

Caveat: this proxy measures internal CX processing time, not full NEPA review duration. It is flagged (`date_determined_initiation`, `initiation_is_proxy`) so downstream analysis can include or exclude it.

## Where each date comes from — the source hierarchy

Every published date traces to one of four provenance tiers, applied in this order:

1. **Authoritative agency registers (Tier A).** BLM ePlanning and the DOE NEPA/CX registers supply official project-start and decision dates via their metadata APIs (~40,000 register dates). These carry confidence 5.0, bypass the learned models, and are preferred over document text — including the **Variant B rule** that a register initiation is admitted regardless of ranking score. Where a document-text CE decision disagrees with the register by more than ~2 years, the register wins.
2. **Document text.** Regex extraction pulls candidate dates with surrounding context; a SetFit **classifier** scores each candidate's role (`p_initiation` / `p_decision`), a LightGBM **ranker** orders candidates within each project, and `05_select_dates.py` applies per-process selection rules (e.g., the EIS tiered decision: ROD-eligible candidates first, FEIS-document candidates only as fallback).
3. **LLM adjudication.** Projects still missing a slot that has candidates are sent to Claude Haiku, which picks among the top-ranked candidates (or, in recovery mode, reads document chunks). Every adjudication is cached in `timeline_api_adjudications.parquet`, so a regenerated dataset re-applies the stored decisions deterministically at zero cost. LLM-sourced dates are labeled `api_adjudication`.
4. **Final-EIS-publication proxy (EIS only).** When no Record of Decision exists anywhere for an EIS, the Final-EIS publication date stands in as the decision, flagged `decision_is_feis_fallback`.

Separately, a small set of **human-verified dates** is injected as ground truth after selection (`05c_inject_ground_truth.py`; source label `ground_truth_verified`).

### Why initiation selects more reliably than decision

Initiation is a single, distinctive event (one NOI / application / scoping date), so the models isolate it cleanly. A decision is a cluster of look-alike dates — a document's cover month vs. its signature date, plus citations to *other* RODs — so the pipeline must disambiguate the operative decision from several decoys. That is why decision selection needs more candidates plus the authoritative-source rules above.

### The FEIS-publication fallback, stated plainly

Only a minority of EIS projects have a ROD in the corpus (~18%). The pipeline searches **ROD-first** (register ROD → ROD-typed document → ROD signed/mentioned in narrative, including inside the FEIS), and **only when no ROD exists anywhere** falls back to the Final-EIS publication date as the decision — clearly flagged (`decision_is_feis_fallback`). Because of source-data limits, that publication date is often **month + year granularity** (a title-page "Final EIS — June 2015"), stored as the 15th of the month; durations using it are month-precision. Why this gap cannot be cheaply closed is documented in [Known Issues & Deferred Items](known_issues.html).

## Confidence, status, and flags

Key provenance columns in `timeline_project_dates.parquet`:

| Column | Meaning |
|---|---|
| `initiation_source_type` / `decision_source_type` | Provenance label — `metadata` (BLM/DOE register), `document_text`, `api_adjudication`, `ground_truth_verified`; decision labels also include `rod`, `fonsi`, `ce_determination`, `eis_feis_fallback`, etc. |
| `*_confidence` | Source confidence; 5.0 = authoritative register (Tier A) |
| `*_is_proxy` | The date stands in for a milestone the documents never recorded directly (e.g., CE inferred application, Date-Determined initiation) |
| `decision_is_feis_fallback` | EIS only — decision is the Final-EIS publication date because no ROD exists in the corpus |
| `*_granularity` | `day` / `month` / `year`; month-only dates are stored imputed to the 15th (±15-day uncertainty). Detect month-precision dates via `*_granularity = 'month'` — the row-level `midpoint_imputed` flag marks only a subset of imputations and undercounts them |
| `timeline_status` | `complete_clear`, `complete_with_proxy`, `missing_initiation`, `missing_decision`, `missing_both`, `invalid_order`, `manual_review` |
| `timeline_flags` | Pipe-delimited diagnostics (`api_adjudicated`, `month_decision`, `imputed_month_midpoint_*`, `same_day`, …) |
| `*_evidence_text` | The sentence around the selected date — every published date is auditable back to its text |

**The headline duration frame, exactly:** `complete_clear` plus `complete_with_proxy` rows with valid ordering, **excluding year-granularity endpoints** (a day cannot be responsibly imputed from a year alone; this drops ~1,100 rows, almost all CE initiations), with month-granularity endpoints imputed to the mid-month 15th. Under this frame the report's duration medians are computed on CE n = 27,278 · EA n = 1,730 · EIS n = 1,321 — slightly fewer rows than raw completeness counts, which is why duration n's do not equal the coverage table's complete counts. Proxy-flagged rows are included but reported in sensitivity views so proxy reliance is always visible.

## Classifier training labels — the frozen split

(Referenced by `04_classify_candidates.py`.) The classifier trains on the hand-labeled sample in `training/deliverable04/classifier.csv`, whose `label` column follows the codebook above. Its `split` column is **frozen**: the test set was assigned once (stratified by process × label) and never grows — new labels added later default to `train`. This keeps "F1 went up" comparable across labeling rounds and prevents leakage; the frozen test IDs are pinned in `training/deliverable04/frozen_eval_ids.txt`. Proxy / Date-Determined rows stay **in** training (teaching the classifier to correct over-eager regex roles is its core job), while the CE Date-Determined pairing itself remains owned deterministically by `05_select_dates.py`.

This candidate-level apparatus is distinct from the **project-level gold** used by `07_validate.py` (`timeline_gold_projects.parquet`), which validates final selected dates end-to-end.
