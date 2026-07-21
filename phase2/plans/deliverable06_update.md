# Deliverable 6 (FONSI Patterns) — remaining-work implementation plan (2026-07-20)

## Start here (read first — written for a fresh session with no prior context)

**What D6 is.** Deliverable 6 mines clean-energy EA→FONSI documents for recurring low-impact action
patterns, then reads each tech×action grid cell as a categorical-exclusion (CE) opportunity:
*adopt* an existing CE, *expand* a CE's bound, or *develop* a net-new CE.

**What is already DONE (do not redo):**
- The **LLM enrichment pass has run and is wired in** — `fonsi_enrichment.parquet` (452 FONSIs,
  63 columns), 97% quote-verified. `09_wire_enrichment.py` rebuilds `candidate_facts`,
  `candidate_mitigation_summary`, and `corpus_mitigation_stats` from it.
- The **v2 pipeline (01–10) is built and running**; the report `phase2/reports/deliverable06.qmd`
  (1,194 lines) renders off those outputs.
- **Item #45 is CLOSED** (the LLM `mitigation_dependence` enum already replaced the deterministic
  mitigation flag; the report already splits the metric — see the #45 section for the evidence).
- **D3 was rescoped and committed separately as `e305bce`. It is NOT part of D6 work** — do not pull
  D3 changes into this effort.

**What this document covers.** The 15 still-open D6 items: **Group A (A1–A4)** from the 2026-07-20
report review, and **Group B (#37–#47)** from `.claude/todo.json`. Every item below states its own
tier. It is the focused *execution* companion to `deliverable06_updates.md` (the durable D6 tracker)
and `deliverable06_refactor.md` (the built tech×action grid); it supersedes neither.

**Tiers used throughout:** **Today** = deterministic, $0, agent-executable now · **User-launched** =
agent builds/prices/stops, the *user* runs the billable call · **Deferred** = with written rationale.

**The three rules that constrain everything here:** (1) **no agent ever launches a billable API run** —
set up, state cost, stop (key in macOS Keychain `nepa-anthropic`; test with `--dry-run`/`py_compile`,
never trip the credential prompt); (2) **DuckDB for all large-parquet reads**, committed scripts only,
no ad-hoc `/tmp` analysis; (3) **render Quarto in the base env** (system quarto 1.3.433), never the
`nepa` env.

**Highest-value item if you only do one thing:** **#47** — a ~1-afternoon, ~\$3–5 fix that unlocks D2's
resource-level mitigation claim. It was previously mis-scoped as a 6–10 hour, \$60–230 defer; a local
pilot (evidence committed in `phase2/notes/deliverable06/`) corrected that.

---

**Scope.** Execution plan for the 15 open D6 items: Group A (A1–A4, from the 2026-07-20 report
review) and Group B (todo.json #37–#47). It enumerates the concrete file edits, data flow, cost,
and a same-day schedule.

**State assumed current (verified 2026-07-20):**
- LLM enrichment has run: `phase2/data/analysis/deliverable06/fonsi_enrichment.parquet` — 452 rows,
  63 columns (37 LLM-extracted fields + audit/metadata/classification). Audit timestamps present
  (`enrichment_extraction_run_at`, `enrichment_llm_run_at`). Quote-verification 97%.
- `09_wire_enrichment.py` overwrites `candidate_facts` / `candidate_mitigation_summary` /
  `corpus_mitigation_stats` with the LLM-backed versions; report consumes those.
- `_run.py` order: 01 → 02 → 03_extract_candidate_facts → 04 → 05 → 06 → **09** → 07 → 08.R.
  `03_enrich_llm.py` and `10_action_label.py` are standalone (billable), run manually before `_run.py`.
- Rendering: base env (system quarto 1.3.433 + Rscript), never the nepa env.

**Population reconciliation (used throughout):** the 452 clean EA→FONSIs =
**293 candidate-tech** (Transmission 149, Wind 62, Solar 61, Geothermal 21) +
**159 non-candidate** (Other Clean 67, Nuclear 33, Biomass 27, Energy Storage 21, Hydropower 7, CCS 4).
Action-label `other` cell = 314 FONSIs. `candidate_ce_comparison.parquet` already holds
**ranks 1–8** per candidate_category (52 categories × 8 = 416 rows) with `bound_{acres,miles,mw,kv,wells}`
and `canonical_source_url` per CE. Verdict split (52 cells): **adopt 22, new 16, already_covered 12,
expand 2**; retrieval_score median 0.396 (MATCH_THRESHOLD 0.40) → 36 cells have a CE ≥ 0.40, of which
**24 are adopt/expand** — the A1/#37 adjudication set. All 52 currently `verdict_confidence="low"`.

**Hard constraints baked in (all items):** no agent launches a billable API run — set up, state cost,
stop, user launches (Keychain `nepa-anthropic`; use `--dry-run`/`py_compile` for tests, never trip the
credential prompt). eCFR citations must be canonical eCFR URLs (CE Explorer is discovery-only). DuckDB
for all large-parquet reads; no ad-hoc `/tmp` analysis — committed scripts only. New extraction outputs
carry the two audit timestamps. Report prose: bold sparingly.

---

## Summary — effort & cost table

| Item | What | Files | Agent-hrs | Human-review | API cost | Tier |
|---|---|---|---|---|---|---|
| A4 | Fix stale header comment (qmd 17–20) | deliverable06.qmd | 0.1 | 2 min | $0 | **Today** |
| A3 | Systematic weight-sensitivity grid + report table; fix qmd-1128 contradiction | 07, 08.R, qmd | 2 | 15 min | $0 | **Today** |
| #38 | Adopt-broader agency crosswalk (parent dept, ranks 1–8) | 07, new `ce_agency_crosswalk.py` | 2.5 | 20 min | $0 | **Today** |
| #39 | Expand workflow: full size dist vs every bounded CE cap + figure | new `11_expand_analysis.py`, 08.R, qmd | 3 | 20 min | $0 | **Today** |
| #40 | Net-new CE discovery: cluster themes over 159 non-candidate FONSIs | new `12_net_new_themes.py`, 08.R, qmd | 3 | 30 min | $0 (local embeds) | **Today** |
| #41 | Provenance: input_hashes + canonical eCFR URLs in CE claims | 01, qmd | 1 | 10 min | $0 | **Today** |
| #42 | Architecture doc update (63-col schema + wiring) | architecture/deliverables/deliverable06.md | 1.5 | 10 min | $0 | **Today** (architect agent) |
| #45 | Confirm LLM mitigation enum wired; relabel report metric | qmd (+ verify 09/05) | 1 | 15 min | $0 | **Today** (mostly done) |
| #46 | JSON-Schema enums on `enrichment_tool_schema()` | prompts.py | 0.5 | 5 min | $0 (prophylactic) | **Today** |
| A2 | Post-FRA refresh: corpus-answerable tabulation + honest caveat | new `13_postfra_refresh.py`, qmd | 2.5 | 30 min | $0 corpus / external=human | **Today** (corpus part) |
| A1+#37 | eCFR verification + top-5 CE coverage adjudication (24 adopt/expand cells); gate 07 | new `ce_ecfr_verify.py`, 07, ce_verification.py, qmd | 3 setup | 2–4 hr review | $0 (human) **or** ~$0.5–$4 (LLM, user-launched) | **Start today; adjudication human-today or user-launched LLM** |
| #44 | Threshold/boundary retrieval pass | new `14_threshold_retrieval.py`; opt. re-enrich | 2 (deterministic) | 20 min | $0 tabulation / ~$20–25 re-enrich | **Deterministic today; re-enrich deferred** |
| #47 | fonsi_conditions resource_area re-tag (unlocks D2 F1) — Tier-1 rules + scoped Haiku pass over 14,072 commitments (11,246 unique) | `mitigation_conditions.py`, new tagging script; D2 join re-run | ~4 (1 afternoon) | 50–100 gold labels | ~$3–5 Haiku, **user-launched** | **Actionable** (step 3 user-launched) |
| #43 | 25–40 project manual gold set | new `gold/` + scorer | 2 setup | 8–12 hr human | $0 | **Defer (human bottleneck)** |

Totals for the "Today" tier: ~**19 agent-hours** of mechanical/deterministic work (parallelizable across
workers) + ~3 hr human review. Only A1/#37 (optional LLM path) and #44 (optional re-enrich) carry any
billable cost, both user-launched and both with a $0 fallback that also finishes today.

---

## Dependency graph

```
                         (enrichment.parquet — already current)
                                      │
      ┌───────────────┬───────────────┼───────────────┬───────────────┐
      │               │               │               │               │
   A4 (qmd)      #46 (prompts)   #42 (arch doc)   #41 (provenance)  A2-corpus (13_)
   independent    independent     independent      ──┐               independent
                                                     │
   #38 (crosswalk) ──► 07 verdicts ◄── A3 (weights) ─┤
   #39 (11_expand) ──► 08.R figs ◄───────────────────┤──► deliverable06.qmd ──► render (base env)
   #40 (12_netnew) ──► 08.R figs ◄───────────────────┤
   A1+#37 (ce_ecfr_verify → coverage_verdict) ──► 07 ─┘   (07 re-run picks up #38 + #37 gate)
   #45 (verify + relabel) ──► qmd
   #44 (14_threshold, standalone) ──► qmd note        (re-enrich path: 03_enrich_llm — billable, deferred)
   #47 (conditions retag) ── out-of-band: Tier-1 rules → deduped Haiku pass → D2 join re-run — ACTIONABLE
   #43 (gold set) ── standalone scorer — DEFERRED
```

**Convergence points:** (1) `07_classify_and_rank.py` is re-run once after #38 + A3 + #37 land (all
touch verdicts/ranks). (2) `08_create_figures.R` is re-run once after #39 + #40 + A3 add figures. (3)
`deliverable06.qmd` is edited by A1/A2/A3/A4/#39/#40/#41/#45, then rendered **once** at the end in the
base env. Sequence the report render last.

---

## Group A

### A4 — Stale header comment cleanup (LOW) — TODAY, trivial

**Objective.** qmd lines 17–20 still say action definitions/limits are "directional pending the
LLM-assisted extraction/verification pass"; that pass ran. Acceptance: the HTML comment reflects that
figures/tables read the LLM-verified enrichment (97% quote-verified), no "pending" language.

**Files.** `phase2/reports/deliverable06.qmd` (lines 17–20 only).
**Sequence.** One-line edit; folded into the final render.
**Effort.** 0.1 agent-hr. **Cost.** $0. **Risk.** None.

### A3 — Formal weight-sensitivity table (MED) — TODAY

**Objective.** Replace the informal 3-weighting table (`tbl-d6-rank-sensitivity`, qmd 335–357) with a
**systematic** grid/perturbation analysis of the 6 fixed rank weights
(`novelty .30, volume .20, diversity .15, limits .15, mitigation .10, role .10`, defined in
`07_classify_and_rank.py` lines 167–172), and resolve the contradiction: qmd line 1128 calls a formal
table "a recommended follow-up" while the informal one already exists.

Acceptance criteria (rendered report):
- A reproducible sensitivity artifact `output/deliverable06/rank_sensitivity.csv` written by 07, holding,
  per candidate cell, its rank under (a) a **Dirichlet/uniform perturbation** sweep (e.g. 2,000 random
  weight vectors on the 6-simplex) — report each cell's rank **distribution** (median, IQR, best, worst,
  % of draws in top-3); and (b) a **one-at-a-time** ±50% perturbation of each weight holding others fixed.
- Report table `tbl-d6-rank-sensitivity` rebuilt from that CSV (rank median + IQR band), caption states
  the top group is a band. qmd line 1128 edited to point at the now-formal table (drop "recommended
  follow-up").

**Files.**
- `07_classify_and_rank.py`: after building `verdicts`, add a `def rank_sensitivity(verdicts, n_draws=2000, seed=…)` that recomputes `rank_score` from the six exposed `rank_*` component columns under sampled weights (no re-derivation of components needed — they are already persisted), writes `D6_OUTPUT_DIR/rank_sensitivity.csv`. Deterministic seed.
- `08_create_figures.R` (optional): a small ridgeline/interval figure of per-cell rank distribution.
- `deliverable06.qmd`: rebuild `tbl-d6-rank-sensitivity` from the CSV; edit line 1128.

**Data flow.** verdicts.parquet (has `rank_novelty…rank_role`) → sensitivity CSV → report table.
**Effort.** 2 agent-hr. **Cost.** $0 (pure recompute). **Risk/decision.** Perturbation family — recommend
**Dirichlet(1,…,1) on the simplex** (weights stay normalized, interpretable) plus the OAT sweep for a
tornado read. Keep the original weights as the reported point estimate.

---

### A2 — Post-FRA refresh (HIGH) — corpus part TODAY; external part = human

**Objective.** The corpus is almost entirely pre-FRA (cut **June 3, 2023**). Deliver what the report
requires (qmd 617–624): (1) post-cut CE-adoption usage, (2) post-FRA EA/FONSI recurrence of the
candidate categories in-corpus, (3) agency implementation guidance since the FRA — while being honest
about 2024–2025 ingestion lag.

Acceptance criteria:
- A committed script tabulates, per candidate category, the count of FONSIs/EAs with a decision_date
  **after 2023-06-03** (join D6 candidates to D4 `decision_date`, already merged into `candidate_facts`
  by 09 — `n_dt` dated rows). Output `output/deliverable06/postfra_recurrence.csv`.
- Report gains a short subsection under "Timing & the FRA caveat": the post-cut recurrence counts, an
  explicit **ingestion-lag caveat** (NEPATEC 2.0 coverage of 2024–2025 is incomplete, so a low post-cut
  count is not evidence of low activity), and a clear split of *answerable-from-corpus* vs
  *needs-external-sources* (implementation guidance, live CE-adoption filings — flagged as human/web
  follow-up, not attempted by an agent).

**Files.** new `13_postfra_refresh.py` (DuckDB; reads `candidate_facts.parquet` with the merged
`decision_date`); `deliverable06.qmd` subsection. New CSV output carries `postfra_extraction_run_at`
(no LLM → `postfra_llm_run_at=""`).
**Effort.** 2.5 agent-hr (corpus part). **Cost.** $0. **Risk/decision.** Dates known for ~⅔ of bounded
set; report the dated denominator explicitly. The external-guidance item is **not** agent-automatable and
is not a billable-run candidate — mark it a human task in the report's next-steps, do not fabricate.

---

### A1 + #37 — eCFR verification of CE matches + top-5 coverage adjudication (HIGH/MED) — MERGED; scaffold+fetch TODAY, adjudication human-today or USER-LAUNCHED LLM

These are the same verification, at two grains: A1 = confirm each *adopt/expand* CE against canonical
eCFR text; #37 = do it over the **top-5** retrieved CEs (covers / partially covers / does not cover /
unclear) and gate 07 on the adjudicated verdict instead of raw top-1 (`ce.iloc[0]`).

**Objective / acceptance:**
- A committed script fetches **canonical eCFR text** per candidate CE (the `canonical_source_url` already
  in `candidate_ce_comparison.parquet` and `ce.json`) for ranks 1–5 of each candidate_category, and
  produces `candidate_ce_coverage.parquet` with, per (candidate_category, retrieval_rank):
  `coverage_verdict ∈ {covers, partially_covers, does_not_cover, unclear}`, `bound_confirmed`,
  `ecfr_text_sha256`, `canonical_source_url`, plus the two audit timestamps.
- `07_classify_and_rank.py` reads `candidate_ce_coverage.parquet`: verdicts gate on the **best
  adjudicated-covering** CE among ranks 1–5, not `iloc[0]`. `verdict_confidence` promotes from hardcoded
  `"low"` to the adjudication's confidence. "Adopt/expand" language stays "candidate, pending" only where
  no rank 1–5 CE is confirmed covering.
- `ce_verification.py` worksheet upgraded: replace the hardcoded `ASSESS` dict with the adjudicated
  `candidate_ce_coverage` rows (top-5, not top-1 `head(1)`); keep the human `reviewer_confirms_covers`
  column. Report `Caveats & next steps` (qmd 1097–1104) updated: "the one remaining verification step"
  becomes "completed via `candidate_ce_coverage`" (or "in progress" if adjudication is human-pending).

**Two adjudication paths (user chooses):**
1. **Human, $0, finishes today.** eCFR fetch is deterministic ($0). The worksheet already exists; a
   reviewer fills `coverage_verdict` for the **24 adopt/expand** categories (or the 36 with a CE ≥ 0.40),
   top-5 CE each. Focused on adopt/expand that is ~24 categories; if only the best CE per category is
   confirmed, ~24 reads; full top-5 is ~120 reads → ~2–4 hr human. Recommended for a same-day close if
   the reviewer scopes to adopt/expand best-CE first.
2. **LLM-assisted, user-launched, ~$0.5–$4.** A `--dry-run`-testable runner sends per (category, CE):
   eCFR text + candidate action profile → structured `coverage_verdict`. Units: top-5 × the 24
   adopt/expand categories = **~120 calls** (or ~180 for all 36 CE-≥0.40 cells; 260 for the full 52-cell
   sweep). Per call ≈ 4k input / 300 output tokens. **Cost: ~120 calls → Haiku (\$0.80/M in, \$4/M out)
   ≈ \$0.5; Sonnet (\$3/M in, \$15/M out) ≈ \$2.0. Full 260-call sweep → Haiku ≈ \$1.1; Sonnet ≈ \$4.3.**
   Billable, Keychain `nepa-anthropic`, **agent sets up and stops; user launches.**

**Files.** new `ce_ecfr_verify.py` (deterministic eCFR fetch via WebFetch/eCFR API → text cache under
`data/raw/deliverable06/ecfr/`, writes `candidate_ce_coverage.parquet`; optional `--llm` adjudication
runner behind a `--dry-run` guard); edit `07_classify_and_rank.py` (gate); edit `ce_verification.py`
(consume adjudication); edit `deliverable06.qmd` (1097–1104). New parquet carries
`ce_coverage_extraction_run_at` + `ce_coverage_llm_run_at`.
**Data flow.** ce_comparison (ranks 1–5 + URLs) → eCFR fetch → coverage.parquet → 07 gate + worksheet + report.
**Effort.** 3 agent-hr setup. **Cost.** $0 (eCFR fetch + human) or ~$0.9–$4 (LLM, user-launched).
**Risk/decisions.** (a) eCFR fetch must use the **canonical eCFR URL**, not CE Explorer (constraint). (b)
Some `ce.json` URLs may be section-level vs part-level — script must record the fetched citation exactly.
(c) **Recommend the human path today** (actionable set is ~10 categories) and ship the LLM runner as a
staged, priced, user-launchable option for the full 52-category sweep.

---

## Group B (remaining, not already covered by A1/#37)

### #38 — Adopt-broader rebuild + agency crosswalk (MED) — TODAY

**Objective.** 07's ADOPT test is a coarse token diff (`our_tokens - ce_units`, lines 140, 55–72). Add a
lightweight crosswalk so "agency X lacks this CE" checks (a) **parent department** and dept-wide vs
subagency scope, and (b) whether **any** of ranks 1–8 already gives the target agency an equivalent CE
before calling it a gap. (`candidate_ce_comparison` holds ranks 1–8 — `TOP_CE=8` at
`04_base_rates_and_ce.py` line 51; do **not** raise `TOP_CE`, that would force an unbudgeted retrieval
re-run.)

Acceptance: `candidate_verdicts.parquet` gains `adopt_targets_net` (target agencies after removing those
already covered by a rank 1–8 CE they or their parent department holds) and `adopt_targets_gross` (the
old value). Report adopt narrative uses the net figure; any shrinkage is stated.

**Files.** new `ce_agency_crosswalk.py` (a committed dept→subagency map: DOE→{PMA/WAPA/BPA/SWPA/SEPA,
NNSA, BOR-is-Interior…}, Interior→{BLM, BOR, NPS, USFWS, BIA, BOEM}, USDA→{USFS}, USACE→Army; derived
from `ce.json` `agency_name`/`agency_unit` tokens, no hand-invented agencies); edit
`07_classify_and_rank.py` ADOPT block to consult ranks 1–8 + the crosswalk.
**Data flow.** ce_comparison (ranks 1–8, agency_unit) + crosswalk → 07 ADOPT test → verdicts.
**Effort.** 2.5 agent-hr. **Cost.** $0. **Risk/decision.** Keep the crosswalk **data-derived** from
ce.json unit codes; do not invent agencies (project rule). Recommend surfacing both gross/net so the
report can show the crosswalk's effect transparently.

### #39 — Expand workflow: full size distribution vs every bounded CE cap (MED) — TODAY

**Objective.** Generalize the transmission-only expand case to **all** bounded CEs: compare the full
FONSI size distribution per metric against each bounded CE's stated cap → "raise-the-cap"
recommendations. Uses the LLM-cleaned sizes already in enrichment (`disturbance_acres`, `line_miles`,
`access_road_miles`, `capacity_mw`, `voltage_kv`, `well_count`) vs `bound_{acres,miles,mw,kv,wells}` in
`candidate_ce_comparison`.

Acceptance: `output/deliverable06/expand_analysis.csv` — per (candidate_category, metric) with a bounded
matched CE: n FONSIs, distribution (min/median/p90/max), CE cap, n and % exceeding, suggested raised cap
(e.g. p90 or max of the in-corpus distribution). Report "Expand" section extended beyond transmission
(currently only line-miles fires); a figure generalizing `fig-d6-sizes` to all metrics with a bound.

**Files.** new `11_expand_analysis.py` (DuckDB join enrichment sizes ↔ ce_comparison bounds); extend
`08_create_figures.R`; `deliverable06.qmd` Expand subsection. CSV carries the two audit timestamps.
**Effort.** 3 agent-hr. **Cost.** $0 (data present). **Risk/decision.** Most matched CEs are qualitative
(no numeric bound) — only transmission line-miles currently fires numerically; be explicit that
qualitative-bound CEs cannot fire a numeric expand and remain "verify against CE text." Recommend
reporting the raised-cap suggestion as p90 (robust to outliers) with max shown alongside.

### #40 — Net-new CE discovery via clustering (MED) — TODAY, unifies with report next-step

**Objective (unified).** The report next-step "cluster the 314 *other*-action FONSIs" and todo #40
"cluster over the ~159 non-candidate FONSIs" are one task. Cluster the LLM-extracted theme fields to
surface candidate net-new CE themes. The enrichment already extracted exactly the right fields:
`potential_ce_theme` (378 populated), `action_label_freeform` (451), `action_summary`,
`why_not_current_candidate` (232) — so **no new LLM call is needed**; clustering runs on local embeddings.

Acceptance: `data/analysis/deliverable06/net_new_themes.parquet` — cluster assignments over the target
set (recommend the **159 non-candidate** FONSIs as the primary lens, with the 314 action-`other` cells as
a documented sensitivity), each cluster with a c-TF-IDF label, size, representative themes, and a
nearest-existing-CE novelty score (reuse `06_ce_landscape.py` embedding + retrieval pattern). Report:
convert the two informal "further frontier" mentions (qmd 117, 1084, 1104, 1112) into a short "net-new
candidate themes" table.

**Files.** new `12_net_new_themes.py` (reuse `all-MiniLM-L6-v2` via `embeddings.py`, KMeans/HDBSCAN +
c-TF-IDF labels as in `06_ce_landscape.py`); extend `08_create_figures.R`; `deliverable06.qmd`.
Parquet carries the two audit timestamps (extraction; `net_new_llm_run_at=""`, clustering is local).
**Data flow.** enrichment (theme fields, non-candidate subset) → embeddings → clusters → novelty gate vs
ce.json → themes.parquet → report table.
**Effort.** 3 agent-hr. **Cost.** $0 (local embeddings; theme text already extracted). **Risk/decision.**
n=159 is small for HDBSCAN — recommend KMeans with k chosen by silhouette (as `06` already does with
`_k_selection`), and treat clusters as "candidate themes to review," not findings (D6 framing). Document
the 159-vs-314 choice; the 159 non-candidate set is the cleaner "no existing candidate CE" population.

### #41 — Provenance fixes (LOW) — TODAY

**Objective.** (a) Add `projects_combined.parquet` to `candidate_corpus`'s `input_hashes`; (b) ensure CE
claims cite **canonical eCFR URLs** (dovetails with A1). Acceptance: `candidate_corpus.parquet`
`input_hashes` includes the projects_combined hash; the report's CE mentions and the verification
worksheet carry the `canonical_source_url` (already in ce_comparison and the enrichment
`referenced_ce_citations`).

**Files.** `01_select_candidate_corpus.py`: 01 does not currently hash `projects_combined.parquet` (it
reads `D03_REVIEWS`, which is derived from it, plus `FONSI_INVENTORY`; `input_hashes([D03_REVIEWS,
FONSI_INVENTORY])` at line 180) — add the `projects_combined` path to the `input_hashes()` call list,
no dataframe load needed. `07_classify_and_rank.py`: **no edit needed** — `best_ce_url` already exists in
the slim table; #41 only surfaces it in `deliverable06.qmd` CE tables. **Effort.** 1 agent-hr. **Cost.**
$0. **Risk.** None; the URLs already exist, this is surfacing + one hash path.

### #42 — Architecture doc update (LOW) — TODAY (architect agent)

**Objective.** Update `phase2/architecture/deliverables/deliverable06.md` with the settled 63-column
enrichment schema (37 LLM fields + audit/metadata/classification), the `09`-wiring
(facts/mitigation/corpus_stats overwrite), the `_run.py` order, and the standalone billable steps
(`03_enrich_llm`, `10_action_label`). Acceptance: the doc's schema section lists all 37 LLM fields with
types; the pipeline section matches `_run.py`.

**Files.** `phase2/architecture/deliverables/deliverable06.md`. **Delegate to the `architect` agent**
(reads code + parquets, writes the doc). **Effort.** 1.5 agent-hr. **Cost.** $0. **Risk.** Do this **last**
among today's items so it captures the new scripts (11–14, crosswalk, ce_ecfr_verify).

### #45 — LLM mitigation enum replaces deterministic flag (MED) — TODAY, mostly done

**Status found:** `09_wire_enrichment.py` **already** (a) carries the LLM `mitigation_dependence` enum
per row (line 169), (b) uses the LLM `is_mitigated_fonsi` for the mitigated share (line 210), and (c)
splits corpus stats into `n_case_specific_dependent` vs `n_design_or_none` (lines 237–247). The report
already reads these (qmd 40, 90, 638–640, 690–699). `05_mitigation_and_boundary.py` stays deterministic
**by design** (pre-wiring fallback; 09 overwrites its outputs later in the same run).

**Remaining work (small):** (1) confirm the render reflects 09's outputs (it does when 09 has run — it
has). (2) Optional label alignment: todo #45 asks for "committed-condition-present vs likely-but-for-
dependent"; the report currently uses "case-specific-dependent vs design-feature-only/none." Decide
whether to relabel or keep. Acceptance: report metric explicitly names the two sub-shares with a
one-line definition. **Files.** `deliverable06.qmd` (wording only, if relabeling). **Effort.** 1
agent-hr. **Cost.** $0. **Recommendation:** keep the existing enum labels (they match the LLM schema and
the mermaid decision tree at qmd 669–699); add a single clarifying sentence rather than renaming, to
avoid schema drift. Close #45 as done-with-note.

### #46 — JSON-Schema enums on the enrichment tool schema (LOW) — TODAY (prophylactic)

**Status found:** `prompts.py` `classification_tool_schema()` and `action_label_tool_schema()` already
enum-constrain their categoricals; `enrichment_tool_schema()` does **not** — `mitigation_dependence`,
`action_category`, `land_ownership`, `decision_basis`, `extraction_confidence` are typed `string` with the
allowed values only in prose. Acceptance: those five fields get JSON-Schema `enum` arrays matching the
prose vocab.

**Files.** `phase2/code/deliverable06/prompts.py` (`enrichment_tool_schema()`). Test with `py_compile`
only — **do not** call the API (would trip the Keychain prompt). **Effort.** 0.5 agent-hr. **Cost.** $0.
**Risk/decision.** This only affects **future** enrichment re-runs (the completed run used the prose
schema). Land it now so any re-run (e.g. #44) validates hard; note in the diff that it does not change
current outputs.

### #44 — Boundary/significance threshold retrieval (MED) — deterministic part TODAY; re-enrich deferred

**Status found:** `significance_thresholds` is already populated for **265/452** FONSIs by the enrichment;
`span_type=='boundary'` is nearly empty (**18** of ~110k evidence spans vs 54,914 `resource` / 14,923
`finding` / 3,268 `condition`). So the LLM already captures thresholds for most FONSIs, but the *packet*
under-feeds boundary spans.

**Objective (two parts):**
1. **Deterministic tabulation (today, $0):** a committed retrieval pass that searches finding/condition/
   resource spans for threshold phrases ("would be significant if", "would require an EIS", "not to
   exceed", "no new access road", "within existing right-of-way", "extraordinary circumstances") and
   writes `output/deliverable06/threshold_candidates.csv` (project_id, span, matched phrase, span_type).
   Report gains a "significance thresholds" mini-section from this + the existing `significance_thresholds`
   field. Acceptance: threshold coverage reported; the 187 FONSIs missing an LLM threshold get a regex
   fallback count.
2. **Re-enrich (deferred, billable):** feed a "threshold candidates" block into the enrichment packet and
   re-run `03_enrich_llm.py` to lift `significance_thresholds` recall. **Cost of the full re-run:** 452
   projects × ~6k in / ~2k out ≈ Sonnet **~\$20–25**, Haiku **~\$6**. Billable, user-launched. **Defer** —
   the deterministic tabulation delivers most of the reporting value today without a paid re-run.

**Files.** new `14_threshold_retrieval.py` (DuckDB over `fonsi_evidence_spans.parquet`); `deliverable06.qmd`
mini-section; (deferred) `enrich_lib.build_evidence_packet` + `prompts.build_enrichment_prompt`. CSV
carries the two audit timestamps. **Effort.** 2 agent-hr (deterministic). **Recommendation:** ship part 1
today; stage part 2 as a priced, user-launchable re-run only if the reviewer finds the 187-gap material.

### #47 — fonsi_conditions resource_area attribution (MED) — ACTIONABLE (~1 afternoon + ~$3–5)

> **Revised 2026-07-20 from a local $0 sizing pilot.** An earlier draft of this plan called #47 a
> 6–10 hour, \$60–230 defer. That estimate was wrong: it priced a blanket 70k-row LLM pass. The
> pilot below shows the fixable universe is ~14k rows (11,246 unique), not 70k. Evidence committed at
> `phase2/notes/deliverable06/pilot_47_resource_tagging.py` (+ `pilot47_findings.md`,
> `pilot47_examples.csv`, `pilot47_summary.txt`) — reproduce with
> `conda run -n nepa python phase2/notes/deliverable06/pilot_47_resource_tagging.py`.

**Root cause (stated plainly).** `condition → resource_area` tags come from `classify_resource_area()`
in `phase2/code/extract/mitigation_conditions.py` — a pure **keyword-counting dictionary**, max-count-wins,
`"unknown"` when zero keywords match. **No LLM was ever applied to the 70,802 condition rows.** D6's
enrichment LLM produces `mitigation_resource_areas` at **project level only** (452 FONSIs) and never reads
these tags; D2's LLM tags resources on the **impact side only**. So D2's impact↔mitigation join is
LLM-quality on one side and keyword-quality on the other — **that asymmetry is the F1 ≈ 0.43 cap.**

**Key reframe — the 51% headline overstates the fixable problem.** Unknown share by `condition_role`
(verified from `fonsi_conditions.parquet`):

| condition_role | rows | unknown |
|---|---|---|
| uncertain | 35,778 | 56.9% |
| **mitigation_commitment** | **14,072** | **36.4% (5,117 rows)** |
| monitoring_requirement | 8,515 | 48.8% |
| baseline_design_feature | 4,464 | 38.7% |
| enforcement_or_permit_condition | 4,037 | 67.1% |
| best_management_practice | 2,345 | 39.7% |
| legal_or_procedural_boilerplate | 1,591 | 62.2% |

Enforcement / legal / boilerplate rows **should** be `unknown` — they have no resource area. The rows that
actually feed D2's join are the **14,072 mitigation commitments**.

**Pilot results (local, $0).**
- **Tier-1 section-heading inheritance:** resolves only **4.8%** of unknowns — they cluster under generic
  headings ("Finding of No Significant Impact" 11.6k, "Decision" 6.1k). Enrichment agreement **0.722** on
  newly-resolved; eyeball 7/10. **Verdict: adopt** (free, precise) but small (~2pp).
- **Tier-2 all-MiniLM-L6-v2 embeddings vs prototypes:** resolves 36.9% of the remainder @0.45 **but**
  agreement only **0.346** (eyeball 4/10) and it *lowers* overall agreement 0.588 → 0.539. Failure mode:
  FONSI boilerplate ("an EIS is not required", indemnification language) force-fit into
  socioeconomic/public_health. **Verdict: DO NOT SHIP as prototyped.**
- **Two failure modes, not one** — the pass must cover **all 14,072**, not just the blanks, because a wrong
  tag hurts D2's join as much as a missing one:
  - (a) **~5,117 commitments untagged** — operational language the dictionary cannot see. Real examples:
    "Avoid ground work within PACs between March 1 and August 31" (wildlife); "disturbed areas would be
    revegetated using native shrubs" (vegetation); "conservation measures developed in consultation with
    USFWS, NMFS and DFG" (biological).
  - (b) **~8,955 tagged commitments** carry single-label max-count tags that mis-attribute multi-resource
    conditions. Real example: "prevent degradation of adjacent water sources and fisheries habitat" tagged
    biological-only; it is water **+** biological.

**Revised fix (replaces the old defer).**
1. **Land Tier-1 heading rules** (free, ~72% precision). **Do not** ship the embedding tier.
2. **Dedupe first.** Exact-duplicate `condition_text` rows exist (verified — the "fisheries habitat"
   sentence repeats; the top duplicate recurs 190×). **Computed: the 14,072 commitments hold 11,246 unique
   normalized texts (79.9%) — a 20.1% cost saving**; unknown slice 5,117 rows → 4,152 unique, tagged slice
   8,955 → 7,094 unique. Tag unique sentences once, fan results back out by text hash.
3. **One scoped Haiku pass over all 14,072 (deduped → 11,246 calls), multi-label output.**
   ≈250 input / ≈25 output tokens each → ~2.8M in, ~0.28M out → Haiku (\$0.80/M in, \$4/M out) ≈ **\$3.4**
   (budget **\$3–5**). **USER-LAUNCHED ONLY** — key in macOS Keychain `nepa-anthropic`; no agent runs it.
   Agent builds the script, tests with `--dry-run`/`py_compile`, states cost, stops.
4. **Validation gate.** Hand-label **50–100** conditions as a gold set and score **before** any
   republication. The existing "33% precision" figure came from a **6-row** check — do not trust it as a
   baseline either. This is **distinct from #43's** larger enrichment-field gold set (different target,
   much smaller).
5. **Re-run deterministic joins (free).** D6 resource tables; D2's mitigation join at
   `02_extract_fonsi_significance.py:31-83` is plain SQL + groupby over **cached** LLM outputs — **no D2 or
   D6 LLM run is re-bought anywhere.** Cosmetic side effect to flag: `condition_row_id` hashes
   `resource_area`, so re-tagged rows get new IDs — ID churn in diffs, not a change in findings.

**Effort/cost.** ~**1 afternoon** + ~**\$3–5** (was 6–10 hrs + \$60–230). **Tier: actionable** (step 3 is
user-launched).
**Payoff.** Unlocks D2's resource-level mitigation claim ("X% of flagged significant impacts paired with a
same-resource commitment") from a caveat to a defensible finding. D6's own headline findings
(mitigated-FONSI share, `mitigation_dependence`, CE verdicts) **never depended on these tags and do not
move** — this is a D2-facing fix delivered from D6.

### #43 — 25–40 project manual gold set (LOW, optional) — DEFER (human bottleneck)

**Objective.** A human-labeled gold set of 25–40 FONSIs for the LLM-extracted fields (sizes, siting
booleans, mitigation_dependence, action_category, significance thresholds) to score the pilot/model
objectively. Acceptance: `phase2/data/analysis/deliverable06/gold/` with a labeling template + a committed
scorer that reports per-field precision/recall against `fonsi_enrichment.parquet`.

**Why defer:** the code (template + scorer) is ~2 agent-hours, but the **labeling itself is 8–12 human
hours** — the binding constraint, not automatable, and not needed to close today's report. **Cost.** $0
(no API). **Recommendation:** build the template + scorer scaffold today only if spare capacity; the
labeling is a separate human work session. This is the objective-evaluation backstop for #37/#44 model
choices — worth doing before any large paid re-enrichment, but not on the critical path today.

---

## Same-day execution schedule (2026-07-20)

**Wave 1 — parallel workers, deterministic, $0 (no cross-dependencies):**
- Worker α: A4 (header) + #46 (enum schema, `py_compile` test) + #41 (input_hashes + URL surfacing).
- Worker β: #39 `11_expand_analysis.py` + figure.
- Worker γ: #40 `12_net_new_themes.py` (local embeddings) + figure.
- Worker δ: A2 `13_postfra_refresh.py` + #44 part-1 `14_threshold_retrieval.py`.
- Orchestrator (self): #38 crosswalk + A3 sensitivity in 07 (these two touch 07 — do serially, myself).

**Wave 2 — converge on 07 and figures (after Wave 1 + #38/A3):**
- Run `ce_ecfr_verify.py` eCFR fetch (deterministic, $0) → `candidate_ce_coverage.parquet`; wire the 07
  gate (A1/#37). Decide adjudication path with the user (human-today vs LLM user-launched).
- Re-run `07_classify_and_rank.py` (picks up #38 + A3 sensitivity CSV + #37 gate).
- Re-run `08_create_figures.R` (picks up #39 + #40 figures).

**Wave 3 — report + docs (serial, last):**
- Edit `deliverable06.qmd` for A2/A3/A4/#39/#40/#41/#45 and the A1/#37 caveat.
- Render in the **base env** (system quarto 1.3.433), verify figures/tables.
- `architect` agent: #42 architecture doc (captures the new scripts).

**Tiering recap:**
- **Finishable today (agent, $0):** A2 (corpus), A3, A4, #38, #39, #40, #41, #42, #45, #46, #44-part1, and
  the deterministic scaffold + eCFR fetch of A1/#37. Also #47 **step 1** (Tier-1 heading rules) + **step 2**
  (dedupe, already computed: 11,246 unique).
- **Start today, needs a user-launched LLM pass (each has a $0 same-day fallback):** A1/#37 coverage
  adjudication (human path finishes today at $0; LLM path ~$0.5–$4 user-launched); **#47 step 3** — scoped
  Haiku pass over the 11,246 unique mitigation commitments, ~$3–5, then free deterministic D2/D6 join
  re-runs; #44 re-enrichment (~$20–25 Sonnet / ~$6 Haiku, user-launched) — optional, deterministic
  tabulation ships today.
- **Defer with rationale:** #43 (gold set — 8–12 human labeling hours, off critical path; note #47's own
  50–100-label validation gate is separate and much smaller), and **#44's second half** (feeding threshold
  candidates into the enrichment packet + full re-enrichment — the deterministic tabulation delivers most
  of the reporting value without buying a re-run).

**Billable-run guardrail:** every LLM path above (A1/#37 adjudication, #44 re-enrich, #47 LLM tagging) is
**set up and priced but not launched** by any agent — Keychain `nepa-anthropic`, tested via `--dry-run`/
`py_compile`, launched only by the user.
