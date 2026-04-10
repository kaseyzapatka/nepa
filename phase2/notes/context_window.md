# Context Window Improvement Ideas

## Problem

The regex stage extracts a fixed character window around each matched date and stores it as
`context`. This window often captures unrelated text from adjacent document elements —
especially in EIS/EA documents where page headers, document reference codes, and revision
stamps are dense and appear on every page.

Example of a noisy context:
```
date:    2017-06-01
match:   June 2017
context: FBP-ER-GEN-WD-RPT-0076 Revision 6 June 2017 A-10 PORTS/REAL PROPERTY CONVEYANCE FINAL REV 6/6/29/2017
```
`June 2017` is a document revision stamp, not a NEPA timeline event. The context window
bled into an adjacent date reference (`FINAL REV 6/6/29/2017`), which is unrelated.

---

## Primary Recommendation: Deduplication by (project_id, date, match)

**What:** Collapse repeated candidates with identical `(project_id, date, match)` tuples
before BERT runs. Page headers repeat on every page — this generates dozens of identical
or near-identical rows per document.

**Impact:** Likely the single biggest win for EIS. EIS has 472K candidates across 4,130
projects (avg ~114/project). A significant fraction are header repeats. Deduplication
could reduce the candidate set by 30–50% for EIS with no loss of signal.

**Implementation:** Add a deduplication step in `run_regex_prep()` before saving the cache,
or as a pre-filter in `run_bert_timeline_extraction()` before the BERT loop:

```python
# After building results_df in run_regex_prep():
results_df = results_df.drop_duplicates(subset=["project_id", "date", "match"])
```

Or more conservatively, keep the first occurrence (which will have the lowest `position`
value, i.e., closest to the document start):

```python
results_df = (
    results_df
    .sort_values("position")
    .drop_duplicates(subset=["project_id", "date", "match"], keep="first")
)
```

**Risk:** Low. Repeated candidates with the same date+match string carry no additional
signal for BERT classification.

---

## Secondary Recommendation: Add `context_clean` as a post-processing column

**What:** Keep `context` (raw window) unchanged. Add a `context_clean` field that applies
trimming rules — BERT uses `context_clean` for classification, `context` stays for
debugging and audit.

**Cleaning rules to apply (in order):**
1. Trim to paragraph boundary — find the nearest `\n\n` or `\n` before and after the
   matched date and use that as the window boundary
2. Strip document reference codes — remove tokens matching
   `[A-Z]{2,}-[A-Z0-9]+-[A-Z]+-\w+-\d{4,}` (federal doc reference patterns)
3. Collapse runs of whitespace

**Why separate from `context`:** Cleaning rules will need iteration. Keeping the raw
context means you can re-run context cleaning without re-running regex prep (expensive
for EIS at 5.94 GB pages.parquet).

**Implementation location:** Either in `run_regex_prep()` when building `results_df`,
or as a standalone post-processing function that enriches the cache parquet.

---

## Other Potential Suggestions

### Header/boilerplate flag (`is_header_candidate`)

Add a boolean flag on each candidate row when its context matches known government
document header patterns:

```python
HEADER_PATTERNS = [
    r'[A-Z]{2,}-[A-Z0-9]+-[A-Z]+-\w+-\w+-\d{4}',   # doc reference codes
    r'\brev(?:ision)?\s+\d+\b',                       # Revision N
    r'\brev\s+\d+/\d+/\d{2,4}\b',                    # REV N/date stamp
    r'\bpage\s+[A-Z]?-?\d+\b',                        # page N / page A-10
    r'\bappendix\s+[A-Z]\b',                           # Appendix A headers
]
```

Don't drop flagged candidates — pass the flag to BERT scoring as a penalty. BERT's
`_select_best_decision()` and `_select_best_initiation()` already have a boilerplate
penalty mechanism; this extends it.

**Caution:** Pattern list needs maintenance. Federal agency document formats vary widely
(BLM, USACE, USFS, BOR all have different reference code schemas).

---

### Paragraph-bounded context window (at extraction time)

Instead of a fixed character window, bound the context to the enclosing paragraph
(text between `\n\n` breaks) at extraction time.

**Pro:** Semantically cleaner unit; eliminates the cross-element bleed in the example above.

**Con:** NEPA headers often have no paragraph breaks between elements, so this example
would still fail. More importantly, useful date context sometimes spans paragraphs:
```
On June 15, 2018,\n\nthe agency issued a Finding of No Significant Impact...
```
Paragraph-bounding would drop "the agency issued a Finding of No Significant Impact"
which is exactly the signal BERT needs.

**Verdict:** Use as a trimming heuristic in `context_clean`, not as the primary window.

---

### Dual context width (narrow + wide)

Extract two windows per candidate: `context_narrow` (±40–60 chars, sentence-bounded)
for BERT classification, `context_wide` (current ±150 chars) for human review.

**Hypothesis:** The classifier only needs the phrase immediately adjacent to the date
(e.g., "was signed on", "Notice of Intent published", "application received"). The wider
window adds noise without signal.

**To validate:** Run BERT on a sample using narrow vs wide context and compare
`bert_decision_confidence` and coverage. If narrow context degrades coverage, the wide
window is carrying real signal and this approach should be abandoned.

---

### Cross-page deduplication with position weighting

A variant of primary deduplication: instead of keeping the first occurrence, keep the
occurrence with the highest NEPA-keyword density in context. If "Notice of Intent" appears
in one context and "Revision 6" in another for the same `(project_id, date, match)`,
keep the first.

This is more complex than simple deduplication but would actively select the most
informative context for repeated dates.
