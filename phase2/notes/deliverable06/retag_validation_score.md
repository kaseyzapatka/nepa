# D6 #47 re-tag validation — score report

Sheet: `/Users/Dora/git/consulting/nepa/phase2/notes/deliverable06/retag_validation_sheet.csv` — 80/80 rows labeled.

_1 row(s) flagged in `notes` as truncated/garbled — still scored, but treat them as a floor on achievable precision._

## Overall — new tags vs old keyword tags

Micro-averaged over (row, resource_area) pairs; `unknown` excluded from both sides (it is scored separately below).

| metric | OLD (keyword) | NEW (tier1+haiku) | delta |
|---|---|---|---|
| precision | 0.812 | 0.758 | -0.054 |
| recall | 0.263 | 0.919 | +0.656 |
| f1 | 0.397 | 0.831 | +0.434 |
| exact_set_match | 0.4 | 0.7 | +0.300 |
| tp / fp / fn | 26 / 6 / 73 | 91 / 29 / 8 | |

## Any-overlap rate (the rule D2 adopts)

Share of rows with a real resource area where **at least one** predicted area is correct. This is what D2's impact<->mitigation join actually depends on.

- OLD: **0.464**
- NEW: **0.893**

## 'unknown' handling

- **OLD** — gold-unknown rows 26, predicted-unknown 48; unknown precision 0.5 / recall 0.923; over-tagged 2 (gold says none, pipeline invented one), under-tagged 24 (gold says something, pipeline said none)
- **NEW** — gold-unknown rows 26, predicted-unknown 26; unknown precision 0.923 / recall 0.923; over-tagged 2 (gold says none, pipeline invented one), under-tagged 2 (gold says something, pipeline said none)

## Per stratum (new tags)

| stratum | n | precision | recall | f1 | exact | any-overlap |
|---|---|---|---|---|---|---|
| `new_haiku` | 24 | 0.66 | 0.971 | 0.786 | 0.542 | 0.875 |
| `new_tier1` | 10 | nan | 0.0 | nan | 0.8 | 0.0 |
| `changed` | 18 | 0.784 | 0.889 | 0.833 | 0.444 | 0.944 |
| `unchanged` | 14 | 0.947 | 1.0 | 0.973 | 0.929 | 1.0 |
| `still_unknown` | 14 | nan | nan | nan | 1.0 | nan |

## Per stratum — did the re-tag help? (old vs new f1)

| stratum | n | old f1 | new f1 | delta |
|---|---|---|---|---|
| `new_haiku` | 24 | nan | 0.786 | +nan |
| `new_tier1` | 10 | nan | nan | +nan |
| `changed` | 18 | 0.444 | 0.833 | +0.389 |
| `unchanged` | 14 | 0.75 | 0.973 | +0.223 |
| `still_unknown` | 14 | nan | nan | +nan |

## Human holistic verdict (`is_correct`)

- `yes`: 48 (60%)
- `partial`: 18 (22%)
- `no`: 14 (18%)

| stratum | yes | partial | no |
|---|---|---|---|
| `new_haiku` | 13 | 8 | 3 |
| `new_tier1` | 2 | 0 | 8 |
| `changed` | 8 | 9 | 1 |
| `unchanged` | 11 | 1 | 2 |
| `still_unknown` | 14 | 0 | 0 |

## How to read this

- **precision** is the number that gates D2's claim upgrade. High precision means a predicted resource area can be trusted when it fires.
- **any-overlap** is the closest single number to what D2's join needs.
- **`still_unknown` precision is not meaningful** — that stratum's predictions are all empty. Read its `unknown` agreement instead: high agreement means leaving them untagged was right; low agreement means there is recoverable signal we are still missing.
- A **negative delta on `unchanged`** would mean the re-tag broke rows that were already right — check that before shipping.

**This does not by itself move D2's `mitigation_dependent_f1` (0.612 overall / 0.623 holdout).** That metric is scored against D2's own gold set, which labels the impact side only. A strong score here is the precondition for upgrading D2's resource-level mitigation caveat to a finding — not the upgrade itself.
