# D2 #52 — resource-level mitigation claim: two prose options (user chooses)

**Status: Option (a) SHIPPED 2026-07-23.** The user chose Option (a) — upgrade the resource-level
mitigation claim to a finding, framed as any-overlap resource matching — paired with the #53 rule
tightening (T5) and a re-validation of `mitigation_dependent_f1` (now F1 0.622 / precision 0.53).
The shipped headline, computed live from the re-tightened join output, is **22.8%** (222 of 974
flagged significant / less-than-significant FONSI determinations paired with a same-resource
committed condition; ~50% of analyzed FONSIs carry at least one such pairing) — an "about a
quarter" finding, **not** the "most / each" phrasing the draft topline below sketched. It lives in
`phase2/reports/deliverable02.qmd` (section "Pairing each effect with a same-resource commitment")
with all four guardrails: (i) validation-provenance methods note, (ii) aggregate-only any-overlap
finding via inline R, (iii) descriptive per-resource splits + the sociocultural→socioeconomic
taxonomy note, (iv) `mitigation_dependent` shown only as a labeled screening metric. Both drafts are
kept below for the record. Numbers below are all measured, not estimated.

## The evidence, in one place

- **Condition→resource tagging is now validated** (D6 #47, 80-row opus-agent gold, after the Tier-1
  heading rule was disabled for measured 0.20 precision): overall F1 **0.83**, precision **0.76**,
  recall 0.92, **any-overlap 0.89**, exact-set 0.70 — up from the pure-keyword baseline (F1 0.40,
  any-overlap 0.46). The tags a mitigation commitment carries can now be trusted when they fire.
- **D2's determination-level metric did not rise from the re-tag.** `mitigation_dependent_f1`
  (any-overlap) went 0.570 → **0.566** (precision **0.41**, recall 0.90); primary 0.599 → 0.596.
  This is expected: the metric is **precision-bound by the matching rule** (an impact counts as
  mitigation-dependent if *any* committed condition shares its resource area), not by tag quality.
- **Tightening the rule helps** (#53, scored on the same D2 gold): requiring ≥2 matched conditions
  lifts F1 to **0.608** (precision 0.475); requiring a real resource overlap *and* ≥2 conditions
  lifts F1 to **0.622** (precision **0.53**). None reaches high precision — the ceiling is ~0.53.
- **Model-gold caveat (applies to both options):** D2's gold is a two-pass agent-labeled set on the
  impact/determination side; the condition-tag gold is a single opus-agent 80-row hand-label. Both are
  defensible validation sets, but neither is an independent human panel.

---

## Option (a) — UPGRADE to a finding, framed as any-overlap resource matching  ✅ SHIPPED 2026-07-23

> **Note (shipped):** the draft below opens "Most … pair each flagged effect," a placeholder written
> before the number was computed. The measured share is **22.8%** (any-overlap, analyzed set), so the
> shipped prose leads with "about a quarter," not "most/each." The `[X%]` placeholder resolved to
> 22.8%. Everything else in the draft — the any-overlap framing, the validated-tag provenance, the
> inclusive-rule caveat, the ~0.53 tightened precision — shipped as written.

> **Most decarbonization FONSIs pair each flagged effect with a same-resource mitigation commitment.**
> Of the flagged significant/less-than-significant determinations, **[X%]** are matched to at least one
> committed mitigation condition that addresses the *same resource area* (any-overlap matching). This
> pairing rests on condition→resource tags that are independently validated: on an 80-row gold set (71 distinct projects)
> the tags score **0.89 any-overlap accuracy** and 0.76 precision, a large gain over the prior
> keyword tags (0.46). *Caveat:* the determination-level match rule is deliberately inclusive
> (any-overlap), so it over-attributes — at the determination grain its precision against D2's gold is
> ~0.41, rising to ~0.53 under a stricter ≥2-condition rule. Read the share as **which resources
> recurringly attract mitigation**, not as a per-project legal claim that each finding is
> mitigation-dependent. Both gold sets are model-adjudicated, not human-panel.

*Use if:* the client wants the resource-level pattern surfaced as a finding and will accept the
explicit any-overlap + model-gold framing.

---

## Option (b) — KEEP the caveat, add the improved-validation note

> **Resource-level mitigation attribution is directional.** Whether a specific finding is
> *mitigation-dependent* — its no-significant-impact conclusion resting on a committed measure for the
> *same* resource — is reported as a pattern, not a per-project determination. The condition→resource
> tags underlying it were rebuilt and are now validated (80-row gold: 0.89 any-overlap, 0.76
> precision, up from 0.46), so the *tags* can be trusted; the remaining limit is the matching rule,
> which is inclusive by design and over-attributes at the determination level (precision ~0.41, ~0.53
> under a stricter ≥2-condition rule). Until that rule is tightened and re-validated, treat the
> resource-level mitigation share as indicative.

*Use if:* the client wants the conservative framing retained, now strengthened by the note that the
tag quality is no longer the bottleneck (the rule is).

---

## Recommendation for the choice (not shipped)

Option (b) is the honest default today: the tags are validated but the determination-level precision
(~0.41, ~0.53 tightened) does not support a hard per-finding claim. If a finding is wanted, Option (a)
with the any-overlap framing is defensible **only** if paired with a rule tightening (#53 T2 or T5) and
a re-validation of `mitigation_dependent_f1` afterward. Ship neither without that step.
