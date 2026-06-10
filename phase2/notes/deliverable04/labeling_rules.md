# D4 Candidate Labeling Rules

How the `label` column in `output/labeling_sample.csv` was assigned (initiation | decision | neither). These rules define the two-head classifier targets and double as a human-review codebook. First applied in the 2026-06-02 first-pass labeling of the 288-row sample.

The label reflects what the date **represents**, read from the anchored `model_context` (target date wrapped in `[[ ]]`), regardless of the regex `candidate_role` (which is often wrong — the whole point of labeling).

## initiation — the NEPA process started
- NOI published in the Federal Register ("NOI was published … which initiated the scoping period")
- Application / SF-299 / ROW / permit **filed or received** ("BLM received an application", "filed a right-of-way application on")
- Scoping started: "opened a 30-day scoping period", "scoping letter was sent", "external scoping notices sent out", internal scoping conducted, "posted to the NEPA Register / eplanning website"
- FERC "approved entry to the pre-filing process"
- DOE **Initiator** signature (the program office initiating the CX) — distinct from the NEPA Compliance Officer
- "Date Determined" used as a recovered CE initiation — see special rule below

## decision — the agency decided
- NEPA Compliance Officer signature / "Date Determined" (when it is the operative determination)
- Field Manager / Field Office Manager / authorizing-official **authorization signature**
- "It is my decision …", Decision Record date, FONSI cover month, "DOE issued a FONSI on", ROW grant **issued**
- BLM CX form "Date:" header and **CX/Decision-Record cover months** (the CX document *is* the determination)
- USACE permit decision / appeal-options notification

## neither — everything else
- Specialist / reviewer signatures (wildlife/cultural/realty/biology/NEPA-coordinator), checklist initials
- SHPO / USFWS / tribal **consultation** dates (Section 106, ESA): concurrences, responses, meetings, BA submissions
- Comment-period **ends**, protest/objection-period ends, draft-EA/EIS **release** (mid-process), Final EIS NOA
- **EA / EIS / DEIS / FEIS / PEIS document cover months** (the decision is the FONSI/ROD, not the analysis document)
- NEPA **case numbers** parsed as dates; bibliographic / Federal Register **citations**; map/figure/drawing dates
- Permit **term/expiration** dates, prior grants/leases/RODs (historical), survey/inspection/field-visit dates
- Statistic/inventory **snapshots**, court opinions, construction-period dates, applicant POD/Mine-Plan cover dates

## Tie-breakers / conventions
- **Document cover month asymmetry:** a CX cover month → `decision` (the CX is the determination); an EA/EIS cover month → `neither` (the decision is the separate FONSI/ROD).
- **Multiple dates in one window:** label only the `[[marked]]` date; ignore the others.
- **Activity vs milestone:** a survey/meeting/inspection conducted "on" a date is `neither`, even if it sits next to a milestone.

## Special rule — "Date Determined" CE initiation recovery
DOE CX forms often carry both a "Date Determined: <d1>" and a later NEPA Compliance Officer **signature** "<d2>". When **both** exist (d1 < d2):
- decision = the signature date (d2)
- initiation = the "Date Determined" date (d1), as a **proxy** initiation — a recovered CE processing-start bracket, not a true NOI/application. Confirmed deterministically in `05_select_dates.py` (precedes the decision) → no classifier / no LLM adjudication.

**Counter-rule (guard):** when "Date Determined" is the **only** date on the form, it can only be the **decision** — never promote a lone Date-Determined to initiation. (Implemented as: a separate, later non-Date-Determined decision must exist.)

**Register-conflict decision (2026-06-02):** when a DOE CX **register** determination date coincides with the document "Date Determined" and a later signature exists, the rule **recovers even with the register** — the later signature becomes the decision and the Date Determined (= register date) becomes the proxy initiation. (This intentionally overrides the register-date-as-decision default for this specific pattern.)

Implemented in `05_select_dates.py` (CE only): if a `date_determined`-flagged candidate exists and a later non-Date-Determined decision-type candidate exists, decision = latest such signature, initiation = the Date Determined (proxy). Tagged in `03_extract_candidates.py` via the `date_determined` positive cue.

Caveat: this proxy measures internal CX processing time, not full NEPA review duration; it is flagged `date_determined_initiation` + `initiation_is_proxy` so downstream analysis can include/exclude it. **Takes effect after a full `03`→`05` re-run** (03 adds the `date_determined` flag to the candidates parquet).

## Train/test split (`split` column) — FROZEN
`labeling_sample.csv` is the **single source of truth** for classifier labels (`04_classify_candidates.py --train`); the former candidate-level `gold/` apparatus is retired. The `split` column holds `train | test`:

- The **test** set is **frozen**: assigned once via a stratified (process × label) 20% draw, seed 42 (158→154 rows: 18 initiation, 18 decision, 118 neither; balanced across CE/EA/EIS). It is the same set `--train` validates on and `--eval` scores.
- **New labels added later default to `train`** (blank `split` → train). Never extend the test set — that keeps "F1 went up" comparable across label-expansion rounds and prevents leakage. Re-freeze the test set only deliberately (e.g., a `test_v2`), never incidentally.
- `--eval` also writes misclassified test rows to `output/deliverable04/classifier_eval_errors.csv` (3-class confusion + per-process + per-regex-role breakdowns print to stdout). Use that file to pick the next rows to label (active learning) and to catch label errors.

Keep proxy / Date-Determined rows **in** training: the regex `proxy_*` roles are ~73–97% truly `neither` (cover-month / case-number false positives), and teaching the classifier to correct them is its core job. The CE Date-Determined → proxy-initiation *pairing* remains owned deterministically by `05_select_dates.py`; the classifier does not decide it.

> Distinct from **project-level gold** (`07_validate.py`, `timeline_gold_projects.parquet`): that is end-to-end validation of final selected dates against hand-checked project dates, and is unaffected by the candidate-level gold removal.
