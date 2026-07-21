# #47 sizing pilot — findings (LOCAL, $0, no LLM/API)

Reproduce: `conda run -n nepa python pilot_47_resource_tagging.py` (this dir).
Artifacts: `pilot47_summary.txt` (run log), `pilot47_examples.csv` (20 eyeball rows),
`pilot47_tier_tags.parquet` (per-condition baseline/tier1/tier2 tags).
Role split + mitigation-only gains from the supplementary DuckDB query (below).

## Headline numbers
- Baseline unknown: **50.8%** (35,994 / 70,802).
- Tier-1 (section-heading inheritance): resolves **1,726 of 35,994 unknowns = 4.8%** → unknown 48.4%.
  LOW yield because unknowns cluster under GENERIC headings (Finding of No Significant Impact 11.6k,
  Decision 6.1k, Environmental Consequences, Mitigation Measures) that carry no resource signal.
- Tier-2 (all-MiniLM-L6-v2 vs prototype sentences), of the 34,268 still-unknown:
  - @0.45: resolves 12,633 (36.9%) → combined unknown **30.6%**
  - @0.55: resolves 3,414 (10.0%) → combined unknown 43.6%

## Proxy precision (a) — enrichment cross-check (agreement w/ project LLM mitigation_resource_areas)
Universe: 15,972 conditions in 366 enriched projects.
- Overall agreement: baseline **0.588**, tier1 **0.594**, tier2 **0.539** (tier2 LOWERS agreement).
- NEWLY-RESOLVED agreement (the real signal):
  - tier1 heading: **0.722** (n=345) — HIGH precision.
  - tier2 embed @0.45: **0.346** (n=2,398) — ~status-quo precision (no better than the ~33% problem #47 names).

## Proxy precision (b) — my eyeball judgments (indicative-only, n=20)
- **Tier-1: 7/10 plausible (70%).** Wrong ones: "protect survey monuments" tagged biological (heading
  "Vegetation"); generic restoration tagged visual; BRAC facility demolition tagged biological (grab-bag
  heading). Failure mode = a resource-specific heading spanning a non-resource condition.
- **Tier-2: 4/10 plausible (40%).** Wrong ones dominated by generic FONSI/DECISION boilerplate force-fit
  to socioeconomic: "an EIS is not required", "will not conflict with decisions in the plan", "indemnify
  the United States against liability", "None of the impacts are considered significant". Also a
  wildlife-blasting-timing condition mis-tagged noise (should be biological).
- Both eyeball rates track the enrichment agreement (72% / ~35–40%) — the two proxies corroborate.

## Sharpening: the 51% headline overstates the FIXABLE problem (role split)
Unknown share by condition_role:
- uncertain 56.9% unk (35,778 rows — largest bucket, mostly procedural/boilerplate)
- **mitigation_commitment 36.4% unk (14,072)** ← the conditions #47 actually cares about
- monitoring_requirement 48.8%, baseline_design_feature 38.7%, best_management_practice 39.7%
- enforcement_or_permit_condition 67.1%, legal_or_procedural_boilerplate 62.2% ← SHOULD be unknown
  (not resource-specific; assigning them a resource area is what tier-2 gets wrong)

Restricted to **mitigation_commitment only** (the #47-relevant set):
- baseline unknown **36.4%** → tier1 34.5% → tier1+tier2@0.45 **17.2%**
  (but the tier2 slice inherits ~35% precision, so quality of that drop is poor)

## Bottom line
- **Tier-1 heading inheritance: adopt it.** Free, high precision (~72%), but only ~2pp of global unknown
  (4.8% of unknowns; ~1.9pp on mitigation_commitment). Small, clean, safe win.
- **Tier-2 embeddings as prototyped: do NOT ship.** High yield (37%) but ~35% precision — it *lowers*
  overall agreement and mostly force-fits procedural/decision boilerplate into socioeconomic/public_health.
  It does not solve #47. Salvage paths: add a "none/procedural" reject class + a per-area reject margin,
  restrict tagging to mitigation_commitment/BMP/design-feature roles (skip uncertain/boilerplate), and
  raise the threshold — but even @0.55 precision only nudges up while yield collapses to 10%.
- **A real fix needs the paid LLM residual pass** on the conditions that are genuinely resource-bearing
  yet unresolved. Scoping to mitigation-relevant roles shrinks the residual sharply:
  - Global residual after tier1+tier2@0.45: 21,635 rows → **Haiku ~$5.6 / Sonnet ~$21**.
  - If instead we skip tier-2 and send only the mitigation_commitment unknowns after tier-1 (≈4,857 rows):
    Haiku ~$1.2 / Sonnet ~$4.7 — cheaper AND higher-value than blanket embedding.
- **Reframe for the #47 plan item:** the ceiling is not "51% → near-0". Much of the 51% is legitimately
  non-resource. The tractable target is the ~36% unknown within mitigation_commitment; free tiers get it
  to ~34% at high precision (tier-1 only), and a scoped Haiku pass (~$1–5) is the honest way to the rest.
