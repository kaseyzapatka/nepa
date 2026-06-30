# Deliverable 6 feedback

## 1. Blocking

1. **Adopt verdicts are promoted from unverified CE retrieval matches.**  
   References: `phase2/code/deliverable06/04_base_rates_and_ce.py:184` sets every CE match to `manual_verification_status = "pending"`; `phase2/data/analysis/deliverable06/candidate_ce_comparison.parquet` currently has 40/40 matches pending; `phase2/code/deliverable06/07_classify_and_rank.py:110-146` uses the top retrieved CE to assign `adopt` without checking manual verification; the report then says the actions "already have" CEs and "would cover them" in `phase2/reports/deliverable06.qmd:325-369`.  
   Recommended fix: add a required CE-coverage adjudication field before `07` runs, e.g. `coverage_status in {verified_covers, verified_does_not_cover, uncertain}` plus reviewer notes and citation to the exact eCFR text. `07` should emit `adopt_candidate_pending_verification` or `unknown` unless the best CE has `verified_covers`. Reword all report claims to "candidate match pending verification" until that review exists.

2. **The expand/adopt decision treats "no numeric CE bound" as "no expand issue," which is not defensible for CEs.**  
   References: `phase2/code/deliverable06/07_classify_and_rank.py:119-146` only creates an `expand` gap when a parsed numeric bound exists; all current top CE matches have null `bound_acres`, `bound_miles`, `bound_kv`, `bound_mw`, and `bound_wells` in `candidate_ce_comparison.parquet`; the report states "there is nothing to exceed" in `phase2/reports/deliverable06.qmd:265-268`.  
   Recommended fix: distinguish `no_numeric_bound` from `within_scope`. For qualitative CEs, add a qualitative coverage test for operative terms such as "routine," "minor," "existing infrastructure," "small number," "commercially available," and "site characterization." If the qualitative scope is not verified, classify as `coverage_unknown` rather than `adopt`; do not say "no expand" solely because no number was parsed.

3. **The report's "bounded, low-impact" subset is not the same as the LLM-bounded subset.**  
   References: `phase2/code/deliverable06/09_wire_enrichment.py:112` carries `is_bounded_low_impact`, but `phase2/code/deliverable06/07_classify_and_rank.py:95-98`, `phase2/code/deliverable06/08_analyze.R:127-145`, and `phase2/code/deliverable06/08_analyze.R:230-257` use only `is_profile_subtype`. Current `candidate_facts.parquet` has 54 profile rows, but only 42 have `is_bounded_low_impact == True`; 12 are explicitly false, including 11 transmission-upgrade profile rows. The report calls all 53 distinct profile projects "bounded, low-impact" in `phase2/reports/deliverable06.qmd:157-180` and `phase2/reports/deliverable06.qmd:417-418`.  
   Recommended fix: either filter all actionable counts, rank inputs, figures, and examples to `is_profile_subtype & is_bounded_low_impact == TRUE`, or rename the current subset as "rule-profiled recurring subtype" and explicitly separate the 12 LLM-not-bounded cases. Recompute `candidate_verdicts`, `fig_d6_outcomes_waffle.png`, `fig_d6_keep_bounded.png`, `fig_d6_adoption_gap.png`, and the transmission count after this decision.

4. **The reproduction path does not generate the LLM-backed deliverable it reports.**  
   References: `_run.py` runs `09_wire_enrichment.py` only if `fonsi_enrichment.parquet` already exists (`phase2/code/deliverable06/_run.py:67-75`), while `08_analyze.R` and the report read `fonsi_enrichment.parquet` unconditionally (`phase2/code/deliverable06/08_analyze.R:421-423`; `phase2/reports/deliverable06.qmd:38-39`). The reproduction instructions only say to run `_run.py` and render Quarto (`phase2/reports/deliverable06.qmd:842-851`).  
   Recommended fix: make `03_enrich_llm.py` an explicit prerequisite in the reproduction section, including model, prompt/schema version, cache path, and expected row counts. `09_wire_enrichment.py` should fail loudly if enrichment is missing, stale, wrong schema, or not full-run. If deterministic mode is still supported, make `08_analyze.R` and the report avoid enrichment-only figures and claims.

5. **Mitigation examples are labeled "verbatim" even though the field is an LLM summary.**  
   References: the prompt defines `mitigation_summary` as a "short summary" in `phase2/code/deliverable06/prompts.py:105-106`; the report presents `mitigation_summary` as "verbatim from the cited record" in `phase2/reports/deliverable06.qmd:574-590`. The actual quote-verification machinery is separate in `phase2/code/deliverable06/enrich_lib.py:398-418`.  
   Recommended fix: either relabel the table as "LLM mitigation summary" or rebuild it from verified `evidence_cited` rows where `claim == "mitigation"` and `verified == TRUE`. Do not use the word "verbatim" for `mitigation_summary`.

6. **The headline "adopt" recommendation overstates current post-FRA relevance.**  
   References: decision dates are merged from D4 in `phase2/code/deliverable06/09_wire_enrichment.py:150-156`; the timeline figure is built from profile rows in `phase2/code/deliverable06/08_analyze.R:230-257`. Current `candidate_facts.parquet` has decision dates for only 36 of 54 profile rows, with 35 pre-FRA and 1 post-FRA; the report acknowledges the limitation but still says adoption is "exactly the efficiency the record points to" in `phase2/reports/deliverable06.qmd:389-405`.  
   Recommended fix: make the main finding explicitly historical: "pre-FRA EAs indicate likely adoption opportunities, pending post-FRA refresh." Add a required post-FRA check before client-facing recommendations: current CE adoption use, recent EA/FONSI recurrence, and any agency implementation guidance since June 3, 2023.

## 2. Should-fix

1. **The one skipped clean FONSI is not handled transparently.**  
   References: `phase2/code/deliverable06/09_wire_enrichment.py:88-102` drops rows with null `action_summary`; `phase2/output/deliverable06/review/fonsi_enrichment_coverage.csv` shows project `115c30ebb825bebb76c359fd95c535fe` failed with `llm_error = no_evidence`; `candidate_corpus.parquet` has 295 FONSI project-category rows while LLM-backed `candidate_facts.parquet` has 294.  
   Recommended fix: keep a placeholder row with `enrichment_status = no_evidence` and propagate it through facts/stats, or explicitly exclude it in a denominator table. The report should say "452 clean EA-source FONSIs; 451 enriched; one no-evidence geothermal development record excluded from enrichment-dependent analyses."

2. **The report mixes the 452 source denominator and the 451 enriched denominator.**  
   References: setup computes `n_clean` from `fonsi_project_inventory.parquet` in `phase2/reports/deliverable06.qmd:44-48`, while `corpus_mitigation_stats.parquet` has `n_clean_fonsi = 451`; the report says "310 of 452" while using a 451-based mitigation share in `phase2/reports/deliverable06.qmd:81-83` and `phase2/reports/deliverable06.qmd:546-550`.  
   Recommended fix: create separate variables `n_clean_fonsi_source = 452` and `n_enriched_fonsi = 451`; use the enriched denominator for LLM-derived mitigation shares and show the excluded-row note.

3. **Candidate assignment and LLM action categories disagree in material places.**  
   References: candidate categories come from regex/tech-group rules in `phase2/code/deliverable06/01_select_candidate_corpus.py:82-130`; `09_wire_enrichment.py:107-148` joins the enrichment onto those categories but does not use `action_category` for validation. In current `candidate_facts.parquet`, profile mismatches include 5 solar-profile rows whose LLM `action_category` is `other`, 3 transmission-profile rows whose LLM `action_category` is `other`, and both temporary-resource rows whose LLM categories are `solar` or `transmission_upgrade`.  
   Recommended fix: add a QA gate before `07`: for each profile row, require `candidate_category == action_category` or a documented override. Move mismatches to a review table and exclude unresolved mismatches from the bounded/actionable counts.

4. **The CE similarity threshold and random-baseline claim are not documented as computed outputs.**  
   References: query terms are hand-coded in `phase2/code/deliverable06/04_base_rates_and_ce.py:53-59`; retrieval blends embedding cosine and lexical overlap in `phase2/code/deliverable06/04_base_rates_and_ce.py:160-174`; the hard threshold is `MATCH_THRESHOLD = 0.40` in `phase2/code/deliverable06/07_classify_and_rank.py:47`; the report claims an unrelated-CE baseline of about 0.07 and rarely above 0.20 in `phase2/reports/deliverable06.qmd:229-239`.  
   Recommended fix: write a `candidate_ce_similarity_null.parquet` with random or permuted candidate-to-CE scores, quantiles, seed, and method. Use that artifact to justify the threshold, or remove the numeric baseline claim and call the scores uncalibrated retrieval ranks.

5. **No statistical uncertainty is shown for small-n comparisons.**  
   References: rank and mitigation shares use raw proportions in `phase2/code/deliverable06/07_classify_and_rank.py:148-166` and `phase2/code/deliverable06/09_wire_enrichment.py:162-190`; the report flags temporary resource assessment as n = 2 in `phase2/reports/deliverable06.qmd:362-364` but still displays point estimates throughout.  
   Recommended fix: add Wilson intervals or exact binomial intervals for mitigation shares and bounded counts, and add a "thin evidence" flag for any candidate with fewer than 10 bounded/enriched examples. Do not rank n = 2 candidates with the same visual confidence as n = 37 candidates.

6. **Rank scores are arbitrary and not sensitivity-tested.**  
   References: component weights are fixed in `phase2/code/deliverable06/07_classify_and_rank.py:152-166`; the report describes the score as "transparent" in `phase2/reports/deliverable06.qmd:302-315`.  
   Recommended fix: add a sensitivity table showing rank order under at least three plausible weight sets: volume-heavy, mitigation-risk-heavy, and verification-confidence-heavy. If rank order changes, report priority bands rather than a single precise score.

7. **The current CE catalog may be stale relative to the client question.**  
   References: `phase2/code/deliverable06/ce_source.py:6-13` documents the local CE Explorer export as the source; `ce_landscape_ces.parquet` uses CE Explorer source version date 2025-08-07; the report presents "existing federal CEs" in `phase2/reports/deliverable06.qmd:131-137` and `phase2/reports/deliverable06.qmd:680-688`.  
   Recommended fix: state the CE catalog snapshot date in the methodology and add a pre-publication check against current eCFR/agency CE text. If the deliverable is meant to be current as of publication, refresh `phase2/notes/deliverable06/ce.json` and re-run `04`, `06`, `07`, `08`, and the report.

8. **The "uncategorized/net-new pool" is described as if it were analyzed, but it is only a residual.**  
   References: `phase2/code/deliverable06/01_select_candidate_corpus.py:132-181` builds only matched candidate rows and does not create an `other` row; `phase2/code/deliverable06/08_analyze.R:72-80` computes "Not recurring" as `n_clean - n_candidate`; the report says every FONSI is sorted into five types or a catch-all in `phase2/reports/deliverable06.qmd:172-177` and says net-new CEs likely live there in `phase2/reports/deliverable06.qmd:813-821`.  
   Recommended fix: either build an explicit `other`/uncategorized table from the enrichment `action_label_freeform` field and cluster it, or reword the report to say "not classified by the current hand-picked candidate rules; not analyzed for net-new patterns in this deliverable."

9. **The architecture file is stale and conflicts with current outputs.**  
   References: `phase2/architecture/deliverables/deliverable06.md:15-18` describes the LLM pass as gated/not wired; `phase2/architecture/deliverables/deliverable06.md:396-406` reports an older run with one `expand` verdict. Current `candidate_verdicts.parquet` has 4 `adopt` and 1 `contrast`.  
   Recommended fix: update the architecture document after the feedback fixes, or clearly mark the run-results section as superseded. Until then, do not use it as implementation documentation.

10. **Output README is stale.**  
    References: `phase2/output/deliverable06/README.md:12-13` says there are 4 report figures, but `phase2/output/deliverable06/figures/` has 17 PNGs; `phase2/output/deliverable06/README.md:25-27` says outputs are deterministic and first-pass, while current facts/verdicts are LLM-wired.  
    Recommended fix: update README to list current client-facing files, explain which review CSVs are authoritative, and identify the LLM run and schema version.

11. **Quote verification is not carried into `candidate_facts.parquet`.**  
    References: `_action_citation()` in `phase2/code/deliverable06/09_wire_enrichment.py:66-81` prefers a verified action quote but falls back to any action/finding quote without storing `verified`; current enrichment has 84 unverified quote records across 2,847 evidence rows and 14 rows with no verified action quote.  
    Recommended fix: add `citation_verified`, `citation_claim`, and `citation_span_ref` to `candidate_facts.parquet`; report tables that claim "source-verified" should filter to `citation_verified == TRUE` or visibly flag unverified rows.

12. **`corpus_mitigation_stats.parquet` reuses old column names with new meanings.**  
    References: `phase2/code/deliverable06/09_wire_enrichment.py:194-209` writes `n_enforceable_only` as "case-specific dependent" and `n_both_high_conf` as "design-only," which no longer matches the meanings documented in `phase2/code/deliverable06/05_mitigation_and_boundary.py:118-126`.  
    Recommended fix: rename the LLM-backed columns to `n_case_specific_dependent` and `n_design_or_none`, or write a new stats schema version so downstream readers do not misinterpret them as dual-signal regex/condition counts.

13. **Several boundary examples are generic NEPA/CEQA definitions rather than project-specific CE bounds.**  
    References: `phase2/code/deliverable06/prompts.py:115-122` asks for explicit threshold/counterfactual statements; the report renders selected examples in `phase2/reports/deliverable06.qmd:638-678`; current `candidate_mitigation_summary.parquet` includes examples such as "NEPA defines significance..." and "A significant impact is defined by CEQA..." for solar.  
    Recommended fix: filter `significance_thresholds` to `is_project_fact == TRUE` or to statements with an actionable metric/condition before calling them CE bounds. Move generic definitions to a separate "not a bound" QA bucket.

14. **The CE landscape near-duplicate analysis is useful context but not evidence that the four D6 candidates are adoptable.**  
    References: `phase2/code/deliverable06/06_ce_landscape.py:64-81` builds similarity components for all CEs; the report uses this as precedent in `phase2/reports/deliverable06.qmd:780-811`.  
    Recommended fix: keep this as background, but do not let it substitute for per-candidate CE coverage verification. Add one sentence: "This shows adoption is common in the CE corpus; it does not verify that the shortlisted CE matches cover the D6 actions."

## 3. Nice-to-have

1. **Add automated QA assertions after each run.**  
   References: current row counts differ across `candidate_corpus.parquet` (295 FONSI project-category rows), `candidate_facts.parquet` (294), and `candidate_mitigation_boundary.parquet` (295).  
   Recommended fix: add a lightweight `qa_deliverable06.py` that asserts expected row counts, full candidate coverage or explicit exclusions, no `manual_verification_status == pending` in client-facing verdicts, and no "bounded" counts using rows where `is_bounded_low_impact == FALSE`.

2. **Move stale v1 artifacts out of the active analysis directory or label them in a manifest.**  
   References: v1 outputs such as `ce_crosswalk.parquet`, `ce_explorer_snapshot.parquet`, `fonsi_actions.parquet`, `fonsi_candidate_categories.parquet`, and `project_action_archetypes.parquet` remain in `phase2/data/analysis/deliverable06/`, while active v2 outputs were modified on June 24-26.  
   Recommended fix: create `phase2/data/analysis/deliverable06/_archived_v1/` for stale artifacts or add `artifact_manifest.csv` with `active`, `input_from_v1`, and `archived` statuses.

3. **Clean generated code-directory clutter.**  
   References: `phase2/code/deliverable06/.DS_Store` and `phase2/code/deliverable06/__pycache__/` are present in the code directory.  
   Recommended fix: remove them from the working tree if tracked, add ignore rules if needed, and keep the deliverable code folder limited to source files plus intentional archived code.

4. **Use registered tables or parameterized DuckDB joins instead of interpolated `IN (...)` strings.**  
   References: `phase2/code/deliverable06/02_assemble_candidate_evidence.py:70-82`, `phase2/code/deliverable06/03_extract_candidate_facts.py:312-317`, `phase2/code/deliverable06/05_mitigation_and_boundary.py:144-150`, and `phase2/code/deliverable06/enrich_lib.py:247-251` construct SQL from project IDs.  
   Recommended fix: register a DataFrame of IDs and join in DuckDB. This avoids empty-list failures, quoting assumptions, and very long query strings.

5. **Make embedding dependency failures explicit.**  
   References: `phase2/code/deliverable06/04_base_rates_and_ce.py:160-165` and `phase2/code/deliverable06/06_ce_landscape.py:99-133` silently fall back if embeddings are unavailable; the report relies on embedding-based CE similarity and t-SNE figures.  
   Recommended fix: for the client-facing run, fail if `sentence-transformers/all-MiniLM-L6-v2` is unavailable, or stamp outputs with `embedding_available = FALSE` and suppress similarity/cluster claims.

6. **Expose model/prompt run metadata in the report.**  
   References: `fonsi_enrichment.parquet` includes `prompt_version`, `schema_version`, and `enrichment_llm_run_at`, but the report only says "Claude Sonnet" in `phase2/reports/deliverable06.qmd:522-535`.  
   Recommended fix: add a small methodology note with model ID, prompt/schema version, run date, parse success (451/452), quote verification rate (2,763/2,847 = 97.0%), and skipped-row count.

7. **Improve figure captions where the plotted data are context rather than decision evidence.**  
   References: `fig_d6_sizes.png` is built from all CE catalog limits in `phase2/code/deliverable06/08_analyze.R:166-207`, not from the matched CEs only.  
   Recommended fix: caption it as "context from the full CE catalog" and keep the actual matched-CE expand test in a separate table showing each best match and whether it has numeric or qualitative limits.

## 4. Anticipated client questions

1. **How were the data filtered/scoped, and what is excluded? What is the denominator?**  
   Current answer: Source denominator is 452 clean EA-source FONSI projects from `fonsi_project_inventory.parquet`; this reconciles to D3 clean EA projects. Analysis 1 then narrows to 293 distinct projects in any current candidate category and 53 distinct profile projects, computed in `phase2/reports/deliverable06.qmd:44-48`. Enrichment succeeds for 451 of 452 FONSIs; one no-evidence geothermal development row is excluded from enrichment-dependent outputs.  
   Does the deliverable answer it? Partially. It gives 452 and the 53/452 scope, but it does not clearly separate source, candidate, profile, LLM-bounded, and enriched denominators.  
   Recommended answer/fix: add a denominator table: 452 clean EA-source FONSIs; 451 enriched; 293 in any candidate rule; 53 distinct profile projects; 42 profile project-category rows LLM-bounded true; one no-evidence exclusion. State that CE and EIS projects are counted only for base-rate context, not deeply extracted.

2. **What time period does this cover, and are comparisons across periods apples-to-apples?**  
   Current answer: Decision-date coverage for the profile subset comes from D4 via `phase2/code/deliverable06/09_wire_enrichment.py:150-156`; `fig_d6_timeline.png` shows profile rows by decision year. Current data: 36 of 54 profile rows are dated, 35 pre-FRA and 1 post-FRA.  
   Does the deliverable answer it? Partially. The report discusses FRA timing in `phase2/reports/deliverable06.qmd:389-405`, but the main recommendation still sounds current rather than historical.  
   Recommended answer/fix: state: "This is mostly a pre-FRA historical record; it supports candidate adoption review, not proof that agencies are still running these EAs post-FRA." Add a post-FRA refresh or a current agency CE-use check before client action.

3. **Are differences shown statistically meaningful, or could they be noise / small-n artifacts?**  
   Current answer: The deliverable mostly reports counts and shares, not inferential statistics. Temporary resource assessment has only 2 profile FONSIs; geothermal has 7; solar has 8; transmission has 37 (`candidate_verdicts.parquet`).  
   Does the deliverable answer it? Not enough. The report has a thin-evidence note in `phase2/reports/deliverable06.qmd:362-364`, but no uncertainty intervals or sensitivity.  
   Recommended answer/fix: say "These are descriptive recurrence counts, not statistically tested differences." Add confidence intervals to mitigation shares, and mark any candidate with n < 10 as "illustrative / low-confidence." Do not rank thin-n candidates as if their point estimates are stable.

4. **How are missing, null, or ambiguous records handled, and could that bias the results?**  
   Current answer: One clean FONSI has no enrichment evidence (`fonsi_enrichment_coverage.csv`); many LLM fields are null by design, including 85 null `is_mitigated_fonsi`, 21 null `is_bounded_low_impact`, and 14 rows with no verified action quote in `fonsi_enrichment.parquet`. `09_wire_enrichment.py:88-102` drops null `action_summary` rows from candidate facts.  
   Does the deliverable answer it? No. It reports 97% quote verification but does not fully discuss null fields or the no-evidence exclusion.  
   Recommended answer/fix: add a missingness table by key field and candidate category. Treat null booleans as unknown, not false, in all shares. Carry no-evidence rows forward with status flags.

5. **What are the key assumptions, and how sensitive are the conclusions to them?**  
   Current answer: Key assumptions are: hand-picked candidate regex rules (`candidates.py`), profile subtype equals CE-shaped (`01_select_candidate_corpus.py:166-169`), retrieval score >= 0.40 means a close CE (`07_classify_and_rank.py:47`), numeric-bound exceedance is the expand test (`07_classify_and_rank.py:119-130`), and unmatched agency tokens imply adopt targets (`07_classify_and_rank.py:132-146`).  
   Does the deliverable answer it? Partially, but not as an assumption/sensitivity section.  
   Recommended answer/fix: add an "Assumptions and sensitivity" section. Show how counts and ranks change if using `is_bounded_low_impact == TRUE`, if requiring verified CE coverage, and if moving the CE similarity threshold from 0.40 to 0.50.

6. **Can these numbers be reproduced, and do they reconcile with prior deliverables or known totals?**  
   Current answer: The 452 clean EA-source FONSI denominator reconciles to D3 clean EA projects. But the reproduction command in `phase2/reports/deliverable06.qmd:846-848` does not generate the LLM enrichment, and architecture/run docs conflict with current outputs (`phase2/architecture/deliverables/deliverable06.md:396-406`).  
   Does the deliverable answer it? Not reliably. A clean run without an existing `fonsi_enrichment.parquet` will not reproduce the reported LLM-backed figures and report claims.  
   Recommended answer/fix: publish a run manifest with input hashes, active scripts, model/prompt version, expected row counts, and exact commands including `03_enrich_llm.py`. Update architecture and README after rerun.

7. **What is the single high-level takeaway, and what should not be over-read from it?**  
   Current answer: The defensible takeaway is: "Historical clean-energy EA/FONSI records show recurring low-impact-looking action classes, especially transmission work, that may warrant CE adoption/harmonization review."  
   Does the deliverable answer it? It overstates. Current wording says "all four resolve to adopt" and "existing CE would cover them" before CE coverage is verified (`phase2/reports/deliverable06.qmd:323-369`).  
   Recommended answer/fix: use: "D6 identifies candidate adoption opportunities, not final legal coverage determinations." Do not over-read it as proof that TVA/DOE CEs already cover every listed FONSI, proof that no expand opportunities exist, or evidence of current post-FRA agency behavior.

8. **Why are there 53 bounded FONSIs in the report but 54 profile rows in facts?**  
   Current answer: The report counts distinct profile projects (`phase2/reports/deliverable06.qmd:47`), while `candidate_facts.parquet` is project x candidate_category and has 54 profile rows because one project appears in two profile categories.  
   Does the deliverable answer it? No.  
   Recommended answer/fix: add a grain note wherever counts are shown: "Project counts are distinct projects unless labeled project-category rows; one project can map to more than one candidate."

9. **Are the CE catalog comparisons current and official?**  
   Current answer: The pipeline uses a local CE Explorer JSON snapshot through `phase2/code/deliverable06/ce_source.py:1-13`, with each record carrying an eCFR source URL. CE Explorer is a discovery index, not itself the legal text.  
   Does the deliverable answer it? Partially. The report names CE Explorer, but not enough currentness/official-text caveat.  
   Recommended answer/fix: add: "CE matches were retrieved from CE Explorer snapshot version/date X and must be checked against current eCFR or agency CE procedures before use." Then perform that check for all four recommendations before client delivery.

10. **Does "mitigated FONSI" mean the action is unsuitable for a CE?**  
    Current answer: Not automatically. The report explains that recurring mitigation can be translated into design criteria, while case-specific mitigation is a warning sign (`phase2/reports/deliverable06.qmd:559-572`).  
    Does the deliverable answer it? Mostly, but the examples and shares need uncertainty and null handling.  
    Recommended answer/fix: say: "Mitigation dependence is a risk screen. It does not disqualify a category by itself, but a CE should encode recurring avoidance/minimization measures and exclude cases needing project-specific commitments." Use verified mitigation quotes, not LLM summaries, as examples.

## Resolution

Worked in priority order. Status per item (batch 1 — Blocking + adjacent Should-fix):

### Blocking
1. **Adopt from unverified matches — fixed (text).** Reframed the whole Main Finding: heading → "Candidate CE-adoption opportunities"; added a `callout-warning` ("candidate matches, not coverage determinations… pending eCFR verification"); "already has a CE / would cover" → "close text match / may cover (pending verification)"; bullets "adopt the X CE" → "candidate match to the X CE"; adopt-table caption + adoption-gap text softened. (Code-side `coverage_status` adjudication field not added — there is no human reviewer yet to populate `verified_covers`; the honest state is "pending," which the report now says everywhere. Flag if you want the column stubbed anyway.)
2. **"No numeric bound" ≠ "no expand" — fixed (text).** Step 5 reworded: none of the matched CEs states a number, so no *numeric* expand can fire — explicitly *not* the same as "no expand exists"; qualitative coverage "must be verified, not assumed," and points to the transmission CE #19 / expand case.
3. **Profile ≠ LLM-bounded (53 vs 42) — answered-in-text; needs-my-input on the harder option.** Added a `callout-note` in Steps 2-3: "bounded" is a *rule* label; the LLM agrees for 42 of 54 profile rows and flags 12 (11 transmission long-rebuilds) as not inherently bounded; counts use the rule-profiled set and *flag* (not drop) the 12; states that an `is_bounded_low_impact == TRUE` filter would tighten to 42 (transmission 26). **Decision needed:** keep the rule-profiled headline (current, non-destructive) vs. actually *filter* to 42/26 and recompute every figure/count. I chose the non-destructive default; tell me to switch if you want the filter.
4. **Reproduction doesn't generate the LLM output — fixed.** `_run.py` now **aborts loudly** at `09_wire_enrichment.py` if `fonsi_enrichment.parquet` is missing (was a silent deterministic fallback). Report Reproduction section now lists `03_enrich_llm.py` as the required first step (model, schema, ~451/452, output path) and notes the abort.
5. **Mitigation "verbatim" — fixed.** Table relabeled "LLM summary" (caption, column header, lead-in all say model summary, not verbatim); points to `evidence_cited` / the boundary table for the quote-verified text.
6. **Overstates post-FRA relevance — fixed (text).** Timing section reworded to "this makes the finding a historical one… pre-FRA EAs indicate likely adoption opportunities, pending a post-FRA refresh," and requires the post-FRA check (current CE use, recent recurrence, agency guidance since 2023-06-03) before client action. Removed "exactly the efficiency the record points to."

### Should-fix (done in this batch)
13. **Generic NEPA/CEQA definitions in boundary examples — fixed.** `first_threshold()` now drops statements matching a generic-definition regex (`defines significan`, `defined by/under/as`, `CEQA`, `context and intensity`, `40 CFR`, `1508`) so only project-specific thresholds surface.
14. **Near-duplicate landscape ≠ coverage proof — answered-in-text.** Added: "This shows adoption is common in the CE corpus; it does not verify that the four shortlisted CE matches actually cover the D6 actions — that remains the pending eCFR check."

### Should-fix (batch 2)
1 & 2. **Denominator mixing (452 vs 451) — fixed (text).** Added a **Scope, denominators & how-to-read** `callout-important` in Methodology with a table: 452 source / `r n_enriched`=451 enriched / 293 candidate / 53 projects (54 rows) / 42 LLM-bounded / 1 no-evidence excluded. New setup vars `n_enriched`, `n_excluded_noev`, `n_profile_rows`, `n_llm_bounded`.
7. **CE catalog stale — answered (text).** Snapshot date `2025-08-07` now stated in the scope callout with a "verify against current eCFR" caveat. (Refresh-and-rerun not done — that's a data refresh you'd trigger; flagged.)
8. **Residual pool described as analyzed — fixed (text).** Methodology Step 2 and "Where net-new CEs would come from" reworded: the residual is "not classified by the current rules / not analyzed in this deliverable," not an examined category.
9. **Architecture stale — fixed.** Added a "PARTIALLY SUPERSEDED" banner to `architecture/deliverables/deliverable06.md` (LLM now wired; verdicts 4 adopt/1 contrast).
10. **README stale — fixed.** Updated figure list (≈16, not 4), named the authoritative review CSVs, and stated the deliverable is LLM-backed (model/schema/verification) and CE matches are pending verification.

### Nice-to-have (batch 2)
6. **Model/prompt metadata — partly answered.** Model `claude-sonnet-4-6`, schema `d6_enrich_schema_v5`, 451/452, ~97% verification now in the scope callout + README. (A dedicated methods block can be added if you want it more prominent.)
7. **Sizes figure caption — fixed.** Recaptioned as "context from the **full** CE catalog (not the four matched CEs)."

### Anticipated client questions (batch 2)
Answered in the scope callout / narrative where a client looks: **Q1** denominators (table), **Q3** "descriptive counts, not tested" + thin-evidence n<10, **Q4** "null = unknown, not false" (full missingness table still to add), **Q7** one-line takeaway, **Q8** project-vs-row grain note, **Q9** CE snapshot + verify caveat, **Q10** "mitigation is a risk screen, not a disqualifier." **Q2/Q6** addressed in batch 1 (historical framing / reproduction). **Q5** (assumptions & sensitivity section) still to add.

### Should-fix (batch 3)
3. **Candidate vs LLM `action_category` mismatch — answered (text).** Verified **10 of 54**
   profile rows mismatch (5 solar→other, 3 transmission→other, 2 temporary); now flagged in
   the Steps 2-3 callout as a QA gate to resolve, not silently kept. (A hard code gate in
   `07` that *excludes* unresolved mismatches is deferred — that changes counts and overlaps
   the #3-blocking filter decision below.)
4. **Baseline/threshold not a computed output — fixed (text).** Reworded: scores are
   "uncalibrated retrieval ranks," the ≤0.20 grey band is "illustrative, not a formally
   computed null." Dropped the asserted 0.07.

### Anticipated client questions (batch 3)
- **Q5 — answered.** New "Assumptions & sensitivity" section: candidate rules, rule-vs-LLM
  bounded (53→42), 0.40→0.50 threshold, numeric-vs-qualitative expand, fixed rank weights
  (priority bands for n<10), and verified-coverage requirement.
- **Q4 — substantially answered** (null=unknown in scope callout; full missingness *table* by
  field×category deferred).
- **Q6 — substantially answered** (Reproduction section + scope callout give model/schema/
  counts/commands; input-hash manifest deferred).

All 10 client questions are now answered in the narrative / methods / captions.

### Deferred — code changes that re-run the pipeline / change schema / need your input
These remain open by design; each changes committed outputs (re-run of 09/07/08), alters a
schema, or needs a decision. Flagged rather than guessed:

- **#3-blocking decision (needs-your-input):** keep the rule-profiled headline (current,
  53/37, non-destructive + caveated) **or** actually filter to `is_bounded_low_impact==TRUE`
  (42/26) and recompute every figure/count. Default kept; tell me to switch.
- **Should-fix 5 (Wilson/exact intervals on shares):** regenerates the mitigated-share and
  rank figures — deferred (changes figures; want confirmation).
- **Should-fix 6 (rank weight-sensitivity table):** needs you to confirm the 3 weight sets
  (volume / mitigation-risk / verification-confidence) — described qualitatively in
  Assumptions for now.
- **Should-fix 11 (`citation_verified`/`citation_claim` into `candidate_facts`):** schema add
  in 09 + re-run; then filter "source-verified" tables — deferred.
- **Should-fix 12 (rename `corpus_mitigation_stats` columns):** rename in 09 + update the
  report setup ref + re-run — deferred (low risk, will batch with 11).
- **Nice 1 (`qa_deliverable06.py`), 2 (archive v1 data artifacts), 4 (parameterized DuckDB),
  5 (embedding-availability stamp):** engineering hygiene; 2/5 are quick, 1/4 are larger.
- **Client Q4 full missingness table, Q6 input-hash manifest:** new artifacts.

Recommend doing 11+12+5+2 together in one re-run batch, and getting your call on the
#3-blocking filter and the 5/6 statistics before those (they reshape figures).
