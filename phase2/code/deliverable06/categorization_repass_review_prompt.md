# Review request: targeted LLM re-classification pass (D6)

You are an expert reviewer (ML/NLP + applied policy data). Critique the proposal below
**before** it is implemented. Be skeptical and concrete. The author is a data scientist who
wants a *quick, cheap, reproducible* fix — not a research project — but it must be defensible
in front of a government-affairs client (Clean Air Task Force). Read the context, then answer
the review questions at the end.

---

## Background

Deliverable 6 finds recurring clean-energy actions that have cleared NEPA with a FONSI
(Finding of No Significant Impact), so agencies could consider codifying them as Categorical
Exclusions (CEs — action classes pre-approved to skip detailed NEPA review). ~451 clean-energy
FONSIs were each read by an LLM (Claude) in a **first pass** that extracted: a 1–2 sentence
`action_summary`, committed mitigation, stated significance thresholds, supporting quotes
(97% verified against source text), plus two coarse judgments — `action_category` (a 6-way
bucket) and `is_bounded_low_impact` (boolean).

**The problem.** The candidate counts in "Analysis 1" depend on categorizing each FONSI into an
action type, and the two available categorizers are both noisy when used *alone as a hard
filter*:
- a **keyword rule** on the raw title (false positives: a land withdrawal tagged "solar"; a VHF
  two-way radio upgrade tagged "transmission");
- the **pass-1 LLM bucket** `action_category` (its own false positives at scale: a botanical
  "Experimental Garden Array" and DOE grants tagged "solar"); and `is_bounded_low_impact` is too
  loose for CE-shape (it passes a 210-mile reconductor and a brand-new 4.5-mile line).

Reading/extraction (Analysis 2) is solid and unaffected; the existing-CE landscape (Analysis 3)
is independent. Only Analysis 1's grouping is in question. The rule missed only 3 real candidates
(union of both signals = 295 vs rule = 292), so this is a *cleaning* problem, not a recall problem.

## Proposed fix

A **targeted second LLM pass** that asks a *precise* question instead of a coarse bucket, run on
the **cached `action_summary`** (not the source PDFs), with answers stored in a **committed cache**
so the result is deterministic and free to reproduce.

**Prompt (operates on cached text):**

```
TASK: Classify ONE federal NEPA action to decide whether it fits the SHAPE of a Categorical
Exclusion for a recurring clean-energy action type. You are given the structured summary already
extracted from the EA that ended in a FONSI. Classify by what the action PHYSICALLY IS, not by
keywords. Funding/grants, studies, land withdrawals, research installations (e.g. a botanical
"garden array"), efficiency retrofits, communications/IT, control systems are "other" even if a
clean-energy word appears. A NEW line on NEW right-of-way is NOT transmission_upgrade.

INPUT (cached pass 1): title, action_summary, significance_thresholds, mitigation_summary, pass1_category

RETURN JSON:
  refined_category: transmission_upgrade | solar | geothermal_exploration |
                    temporary_resource_assessment | wind_onshore | other
  modifies_existing_infrastructure: true | false | null
  linear_extent_miles: number | null
  creates_new_right_of_way: true | false | null
  confidence: high | medium | low
  rationale: <=1 sentence grounded in the input
```

**`is_ce_shaped` is derived in code** (not asked of the model), so the definition is auditable:

```
is_ce_shaped = refined_category in the 5 clean types
           AND (modifies_existing_infrastructure == true OR category in {geothermal_exploration,
                temporary_resource_assessment})
           AND creates_new_right_of_way != true
# linear_extent_miles then drives adopt (<= existing CE cap, e.g. 25 mi) vs expand (> cap)
```

**Reproducibility mechanism (a real re-run, not a hand-filled patch):**
- A single standalone script `10_refine_classification.py` with the prompt embedded, run via the
  Anthropic API on a **pinned model snapshot at temperature 0**.
- **Scope:** ALL 451 FONSIs are re-classified from their cached `action_summary` — the keyword rule
  is removed from categorization entirely; the candidate set emerges from the model's `refined_category`.
- Output `refined_classification.parquet` is **committed** as the canonical artifact (so the report is
  stable). Anyone with the API key deletes it and re-runs the one command to regenerate it; there are
  **no hand-edited values**.
- Cost: ~340k input + ~40k output tokens total → ~$1–2 (Sonnet) / under $1 (Haiku). Re-reading the
  source PDFs is deliberately avoided — extraction (the summaries) was already verified at 97% quote
  fidelity, so only the *classification* is redone.
- Honest caveat: LLM outputs are not bit-for-bit deterministic even at temp 0; the pinned snapshot +
  temp 0 + committed output is the mitigation, and is standard for LLM-in-the-loop pipelines.

## Review questions

1. **Is a targeted re-pass the right fix**, versus (a) the agreement/intersection of the two
   existing signals, or (b) just trusting pass-1 `action_category`? What would you do?
2. **Is cached `action_summary` (≈100 tokens) sufficient input**, or will it too often omit the
   facts that matter (modifies-existing, miles, new ROW) and force "null"? Should the pass also
   feed back more cached fields, or is a source-document re-read unavoidable for a defensible
   `linear_extent_miles` / `creates_new_right_of_way`?
3. **Is the prompt well-specified?** Are the 6 category definitions and the "physical action, not
   keywords" instruction enough to kill the observed failure modes (garden→other, grant→other,
   new-build→not-shaped, 210-mi→expand)? What edge cases or category gaps are missing?
4. **Is deriving `is_ce_shaped` in code (not from the model) the right call**, and is the rule above
   correct? Is the implicit ≤25-mile adopt/expand cap defensible, or does it need per-CE sourcing?
5. **Is the reproducibility design honest and sound?** The canonical `refined_classification.parquet`
   comes from a real API run on a pinned model snapshot at temp 0, committed to the repo and
   regenerable by re-running the script. Given LLM non-determinism, is "pinned snapshot + temp 0 +
   committed output" an adequate reproducibility guarantee for a client deliverable, or is more needed
   (e.g., self-consistency voting, logged token-level seeds, a frozen offline model)?
6. **Is re-classifying all 451 (rule removed entirely) the right scope**, or does dropping the keyword
   pre-filter introduce a recall/precision risk the coarse first-pass bucket was masking?
7. **Anything else** that would embarrass us in front of the client, or any cheaper/more robust
   approach we're not seeing.
```
