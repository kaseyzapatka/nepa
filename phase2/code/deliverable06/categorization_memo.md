# D6 categorization — problem & way forward (memo for a second opinion)

*Self-contained; assumes no prior context on this project.*

## What D6 is supposed to do

Find recurring clean-energy actions that have **repeatedly cleared NEPA with a FONSI**
(a "Finding of No Significant Impact" — the agency decided a full environmental impact
statement wasn't needed), so agencies could consider codifying those action types as
**Categorical Exclusions** (CEs — action classes pre-approved to skip detailed NEPA review).
Three analyses:

- **Analysis 1** — group ~450 clean-energy FONSIs into candidate action *types* (transmission
  upgrades, solar, geothermal exploration, temporary site assessments) and ask which already
  have, or could adopt, an existing CE.
- **Analysis 2** — *read* the FONSIs for **meaning**: what mitigation each commits to, what
  "significance thresholds" it states ("impacts would be significant if X").
- **Analysis 3** — map the existing federal CE catalog (independent of the FONSIs).

## The architecture we built

For all 451 clean-energy FONSIs, an LLM (Claude) reads each document and returns structured
fields:

- **extraction** — a plain-language action summary, the committed mitigation, the
  significance-threshold statements, and supporting **quotes**;
- **judgments** — is the finding mitigation-dependent? is the action inherently low-impact?
- **one classification** — `action_category`, a coarse 6-way bucket.

Then plain rules/code aggregate those fields into the three analyses.

## The problem (it lives only in Analysis 1)

Each FONSI gets a category **two ways**, and they disagree on ~30% of candidates:

1. a **keyword rule** on the raw project title ("solar" appears → bucket = solar);
2. the **LLM's `action_category`** from reading the whole document.

**Neither is clean enough to use *alone* as a hard filter:**

- The **keyword rule** has false positives — a BLM **land withdrawal** tagged "solar" (the word
  appeared nearby); a **VHF two-way radio** upgrade tagged "transmission" (it said "upgrade").
- The **LLM bucket** has its *own* false positives at scale — a botanical **"Experimental Garden
  Array"** tagged "solar"; **DOE grants** tagged "solar." And its separate "inherently
  low-impact" flag passes a **210-mile reconductor** and a brand-new **4.5-mile line** — both
  genuinely low-impact, but not the tight shape an existing CE can cover (those cap ~25 miles,
  modify-existing only).

So "go fully by the LLM bucket" would ship a *botanical garden* as a solar CE candidate;
"keep the keyword rule" keeps *radio upgrades* as transmission. **The trustworthy set is where
the two agree.**

## Is the LLM "failing"? — No, and this is the crux

The thing we actually built the architecture around — **the LLM reading each document and
extracting meaning** — works, and is verified:

- **97% of the LLM's supporting quotes matched the source text** (we check each quote against
  the document).
- **Analysis 2** (the mitigation language and the "significant if X" threshold findings) is the
  **strongest, most defensible part of the deliverable**, and it does **not** use the buckets at
  all — it's computed across all FONSIs.
- On the rule-vs-LLM **disagreements**, the LLM is the one that's right (radio ≠ transmission).

The weak link is narrow and specific: we asked the LLM **one coarse question** ("which of 6
buckets?") and **one general one** ("is this low-impact?"), then used those coarse answers as
**precise filters** for the candidate counts. Coarse answers make poor precise filters. **That's
a question-design choice, not a failure of the LLM's reading.**

## What's affected vs. what isn't

| Part | Uses the category buckets? | Status |
|---|---|---|
| **Analysis 1** — candidate counts, adopt/expand recommendations | Yes | **Affected** — this is the whole issue |
| **Analysis 2** — mitigation + significance-threshold findings | No (corpus-wide) | **Unaffected** — 97% quote-verified, the substantive contribution |
| **Analysis 3** — existing CE catalog | No (different data) | **Unaffected** |

Even inside Analysis 1, the **headline (transmission upgrades, ~24 projects) is robust**; only
**solar** (small, inherently ambiguous) and **"temporary assessments"** (n=2, both mislabeled → 0)
are shaky.

## Ways forward

1. **Agreement set — recommended now.** Use what the rule and the LLM agree on: let the LLM
   delete the rule's false positives (radio, land withdrawals) and the rule delete the LLM's
   (garden, grants, the 210-mile job). Result: **transmission 24, geothermal 7, solar 2 (thin),
   temporary 0.** Quick, **no new LLM cost**, defensible. Solar shown as thin/unstable; temporary
   drops out of the recommendations.

2. **Targeted LLM re-pass — the real fix, if we want broader *clean* coverage.** Stop asking a
   coarse bucket; ask the **precise** reading questions the LLM is good at: *"Is this a
   modify-existing transmission action within existing right-of-way? How many miles? Does it
   create new right-of-way?"* This plays to the model's strength (reading specifics) and would
   expand coverage without importing garbage. Cost: **one billable enrichment run.**

3. **Raw LLM bucket — rejected.** Demonstrably ships garbage (the garden as solar, the 210-mile
   job as "bounded").

## Recommendation

Do **(1) now** to make the current report defensible, and consider **(2)** as a follow-up if a
larger, cleaner candidate set is wanted. **The core LLM-reading investment (Analysis 2) stands on
its own regardless of how Analysis 1's grouping is resolved** — so the architecture did its main
job; what needs tightening is one coarse classifier we leaned on too hard.
