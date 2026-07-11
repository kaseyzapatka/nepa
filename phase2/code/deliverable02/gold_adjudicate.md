# D2 Gold-Set Adjudication Prompt — Reviewer 3 (independent tie-breaker)

You are an expert NEPA analyst acting as the **independent third reviewer** for Deliverable 2
(*Determinations of significance across resource areas*). Two labelers — **Claude (Reviewer 1)** and
**Codex (Reviewer 2)** — independently labeled a gold set; where they disagree, **you break the
tie** by reading the underlying evidence and choosing the better answer. Your output becomes the
final answer key that grades the extraction pipeline (Gate 3), so decide carefully.

## Your job, in one sentence
For every disagreement row, read the passage and **pick the answer the evidence supports** — either
Reviewer 1's value or Reviewer 2's value (do not invent a third value) — and, for a resource only one
reviewer coded, decide whether that determination genuinely **exists in the passage** (keep) or not
(drop).

## Files
- **Input / output (edit IN PLACE):** the disagreements CSV.
  - FONSI: `phase2/output/deliverable02/gold_disagreements.csv`
  - EIS:   `phase2/output/deliverable02/gold_disagreements_eis.csv`
  Read it with a real CSV parser (`pandas`) — `evidence_text` contains commas and newlines.
- Full labeling rubric (the exact vocab + the hard-pair rules): `gold_labeling.md` (FONSI) /
  `gold_labeling_eis.md` (EIS). Follow it; the key rules are summarized below.

Each row has, for every field `X`: a `claude_X` value, a `codex_X` value, and an empty `final_X` you
fill. It also has `evidence_span_id`, `gold_resource_area`, `disagreement_kind`, `heading_title`,
`page_start`/`page_end`, and `evidence_text`.

## Decide per `disagreement_kind`

**`matched_field_conflict`** — both reviewers coded this (window × resource) but differ on one or more
fields. For **every** `final_*` field, set it to whichever reviewer's value the passage supports (if
the two agree on a field, copy that shared value). Base the class call on the evidence, not on which
reviewer you trust. You MUST set `final_gold_is_determination = TRUE` here (both agreed it is one).

**`claude_only_resource`** — Reviewer 1 coded a determination for this resource; Reviewer 2 did not.
Read the passage: **does it actually reach a significance conclusion about this resource?**
- YES → keep it: copy Reviewer 1's values into every `final_*` field (`final_gold_is_determination
  = TRUE`, etc.).
- NO (the passage doesn't conclude on this resource, or only mentions it in passing/background) →
  **drop it: leave `final_gold_is_determination` BLANK** (a blank drops the row from the gold set).

**`codex_only_resource`** — same, but Reviewer 2 coded it. Keep (copy Reviewer 2's values) if the
passage supports a determination for this resource; otherwise leave `final_gold_is_determination`
blank to drop.

## The judgment calls that matter most (from the rubric)
Almost all class disagreement is on these boundaries — decide from the passage:
- **`no_significant_impact` vs `less_than_significant`:** NSI = the agency concludes *no* significant
  impact (often the formal finding / "would not significantly affect"); LTS = an impact exists but is
  below the significance line ("less than significant", "minor", "negligible").
- **`less_than_significant` vs `less_than_significant_with_mitigation`:** `_with_mitigation` ONLY when
  the below-the-line conclusion **depends on committed/required mitigation** ("with implementation of
  the measures…", "would be significant absent mitigation"). Impacts minor by inherent design, or
  with only incidental/voluntary BMPs, are plain `less_than_significant`.
- **`significant_adverse` vs `significant_unavoidable`** (EIS): `_unavoidable` when the significant
  impact **cannot be fully mitigated / is unavoidable**; else `significant_adverse`.
- **`gold_mitigation_link`** is TRUE iff *this resource's* conclusion depends on committed mitigation
  (it tracks the `_with_mitigation` class decision).

Use the **exact controlled-vocabulary strings** (copy them verbatim from the reviewer values).
Booleans as TRUE/FALSE.

## Save incrementally — checkpoint every 50 windows
Process rows grouped by `evidence_span_id` (read each passage once, decide all its disagreed rows).
This is a ~1,000-row job — do NOT hold everything to the end. After every ~50 windows, **write the
CSV to disk with all `final_*` filled so far** (preserve every other column and every row; only fill
`final_*`). If you restart, skip rows whose `final_*` you've already filled. Never create per-batch
files — edit the one CSV in place.

## When done
Report: how many rows you kept vs dropped; for kept rows, how often you sided with Reviewer 1 vs
Reviewer 2 vs (agreed); and any rows you found genuinely unresolvable (set `final_gold_needs_human_
review = TRUE` and explain in `final_gold_notes`, but still make your best call). Do not run
`--finalize` yourself — the calling process does that.
