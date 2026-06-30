# Deliverable 6 — Analysis 4: Net-new CE discovery from the "other" residual

**Status:** plan for review (not yet built).
**Author:** Claude (for Codex review).
**Scope:** a new sub-analysis of D6 that examines the FONSIs the candidate pipeline set aside, to
surface **net-new** Categorical Exclusion (CE) candidates — recurring low-impact clean-energy
action types that have *no* existing CE.

---

## 1. The question

Of the clean-energy EA→FONSIs that are **not** one of the 5 existing-CE candidate types
(transmission upgrade, solar, geothermal exploration, temporary assessment, onshore wind),
**which recurring, low-impact action types have no existing CE** — i.e., are candidate **net-new
CEs** an agency could write from scratch?

## 2. Why this analysis is needed (net-new is empty *by design*, not by finding)

D6's three verdicts are **develop** (net-new: a recurring action with no existing CE), **expand**
(exceeds an existing CE's bound), and **adopt** (an existing CE at another agency covers it). The
5 candidate types were hand-picked *because they already recur and already have CEs* — they were
built to test adopt/expand. The LLM classifier sorts every FONSI into one of those 5 or **"other."**

Consequence: all 5 candidate types resolve to **adopt**, and **develop/net-new is empty — not
because no net-new actions exist, but because the design only examined types that already have
CEs.** A net-new CE is, by definition, a recurring action type with no existing CE; such types
*cannot* be among the 5, so they sit unexamined in the **"other"** bucket. This analysis examines it.

## 3. Inputs (what we already have — most of the groundwork is done)

Source: `data/analysis/deliverable06/fonsi_enrichment.parquet` (the corrected, v2-classified
enrichment of all 451 clean FONSIs). The "other" residual and the relevant pre-extracted fields:

| Quantity | Count | Notes |
|---|---|---|
| `action_category == "other"` (with a summary) | **314** | the residual the candidate pipeline set aside |
| of those, `potential_ce_theme` non-null | **266** | **pass-1 already named a plausible net-new CE theme per FONSI** |
| of those, `is_bounded_low_impact == TRUE` | **217** | the net-new candidate pool (low-impact only) |

Already-extracted per-FONSI fields we reuse (no new extraction needed for naming):
- `potential_ce_theme` — a short, LLM-named plausible net-new CE theme (e.g. "small-scale
  biomass/biogas energy facility at existing industrial site", "combined heat and power at
  existing industrial facility", "transmission interconnection tap line").
- `action_label_freeform` — a normalized action label.
- `why_not_current_candidate` — one phrase on why it's outside the 5 candidate types.
- `is_bounded_low_impact`, `action_summary`, `lead_agency_harmonized`, `project_state`.

**Themes already visible in the raw fields (pre-clustering):** biomass / biogas / biorefinery ·
combined heat & power at industrial sites · EV-battery manufacturing · landfill- & blast-furnace-gas
power · renewable-energy research labs · weed / vegetation management · small wind + storage demos ·
transmission access roads / interconnection taps. The task is to **aggregate the ~266 scattered
free-text themes into sized, recurring clusters**, then screen and rank them.

## 4. Method

### 4.1 Scope the pool
Filter the 314 "other" to **`is_bounded_low_impact == TRUE` (217)** — net-new CEs only make sense
for low-impact, repeatable actions; the 97 large/greenfield "other" are not CE-able. Keep the
bounded flag as a downstream ranking input. (Open decision 9.4: whether to include unbounded as
context-only.)

### 4.2 Embed
Embed each FONSI's `potential_ce_theme` (primary) concatenated with a trimmed `action_summary`
(context), using **`sentence-transformers/all-MiniLM-L6-v2`** via the existing `embeddings.embed()`
helper — the same model D6 already uses in `04`/`06` and D3. 384-d vectors. Deterministic.

### 4.3 Cluster  → **HDBSCAN** (recommended; see §5 for the decision)
Run HDBSCAN on the embeddings (cosine/`euclidean`-on-normalized) with `min_cluster_size` ≈ 3–5
(the recurrence floor) and `min_samples` tuned for stability. HDBSCAN:
- **auto-detects the number of clusters** (no fixed *k*), and
- **labels low-density points as noise (`-1`)** — exactly what net-new needs: keep the *dense,
  recurring* themes, discard the one-offs (a one-off action is not a CE candidate).

Output: a `cluster_id` per FONSI (with `-1` = one-off/noise), plus a cluster-membership table.

### 4.4 Name + screen each cluster (LLM — one cheap pass)
For **each cluster** (not each FONSI — ~20–40 calls total), send the cluster's member
`potential_ce_theme`s + representative `action_summary`s to Claude Sonnet (pinned, temp 0,
tool-use schema, cached — mirroring the `03 --stage classify` pattern). Ask it to return:
- `theme_name` — a concise recurring-action name,
- `coherent` (bool) — is the cluster one coherent action type,
- `net_new_verdict` — one of `plausible_net_new` / `belongs_to_existing_type` /
  `funding_not_codifiable` / `not_coherent`,
- `is_bounded_low_impact` (cluster-level judgment),
- `rationale` (one sentence).

This screens out clusters that are really (a) one of the 5 existing types (e.g. interconnection
taps → transmission), (b) **funding actions** (DOE loans/grants — a CE codifies a physical action,
not financing), or (c) incoherent mixes.

### 4.5 Rank
Score each `plausible_net_new`, coherent cluster by **recurrence × spread × boundedness**:
`score = f(n_fonsi) * g(n_agencies, n_states) * bounded_share`, with an explicit recurrence floor
(`n_fonsi >= R`, R ≈ 3–5; open decision 9.3). Produce a ranked shortlist of net-new CE candidate
themes.

### 4.6 Verify the top themes (eCFR)
For the top-ranked themes, confirm the "no existing CE" claim against the CE catalog / eCFR — the
same human step as the adopt worksheet (`ce_verification.py`); a theme with a close existing CE is
*adopt/expand*, not net-new.

## 5. Methodology decision — clustering algorithm (please review)

| Option | Auto-*k*? | Isolates one-offs? | New deps | Consistency w/ D6 | Verdict |
|---|---|---|---|---|---|
| **HDBSCAN** (+ our embeddings + LLM naming) | yes | **yes (noise = `-1`)** | `hdbscan` (1, small/stable) | embeddings already standard | **recommended — best task fit** |
| sklearn `AgglomerativeClustering` (distance-threshold, cosine) + filter small clusters | yes | via post-filter | none | matches `06` sklearn usage | strong no-install fallback (~90%) |
| sklearn `KMeans` (06's choice) | no (fixed *k*) | no | none | exact `06` precedent | rejected — forces *k*, absorbs one-offs |
| full **BERTopic** | yes | yes | `bertopic`+`umap`+`hdbscan` (3) | not actually used in D6 | rejected — wrapper we don't need; we already have embeddings + a better namer (LLM) than c-TF-IDF |
| **LLM-only grouping** (no embeddings) | n/a | n/a | none | fits the session's LLM-first direction | rejected for the *grouping* step — less systematic/reproducible than embedding clustering; use the LLM where it's strongest (naming + viability screen) |

**Recommendation:** embeddings (`all-MiniLM-L6-v2`) → **HDBSCAN** → LLM naming/screening. It is the
lean core of BERTopic (we skip UMAP/c-TF-IDF because n≈217 is small and the LLM names clusters far
better than keyword lists), and HDBSCAN's noise model is the single feature that best matches
"recurring vs one-off." Cost of "best" = one small install (`hdbscan`). If we want zero new deps,
the `AgglomerativeClustering` fallback stays fully consistent with `06`.

## 6. Implementation

New script **`code/deliverable06/10_net_new.py`** (08 is the R analysis, 09 wires enrichment; 10 is
free), reproducibility mirroring the classify pass:
- reads `fonsi_enrichment.parquet`; filters the bounded "other" pool.
- `embeddings.embed()` for vectors (deterministic, pinned model).
- HDBSCAN with fixed params + `random_state` where applicable (deterministic).
- LLM cluster naming/screening via a new `prompts.build_netnew_prompt()` + tool schema, called
  through an `enrich_lib.call_*`-style helper, **cached** on a cluster-content hash + a
  `NETNEW_PROMPT_VERSION` (so re-runs are free and the committed output is canonical).
- writes the outputs in §7.

Add a small `--dry-run` (cost preview, no key) like `03`.

## 7. Outputs / deliverables
- `data/analysis/deliverable06/net_new_themes.parquet` — one row per cluster: `cluster_id`,
  `theme_name`, `net_new_verdict`, `n_fonsi`, `n_agencies`, `n_states`, `bounded_share`,
  `rank_score`, `example_project_ids`.
- `output/deliverable06/review/net_new_membership.csv` — FONSI → cluster drill-down.
- A report **Analysis 4** section in `deliverable06.qmd` (the ranked shortlist + a theme-landscape
  figure: cluster sizes / recurrence), and a **linked page** (mirroring the CE verification
  worksheet) the client can be pointed to.
- QA-gate additions (`qa_deliverable06.py`): membership totals reconcile to the 217 pool; every
  ranked theme is `plausible_net_new` + coherent; no funding-only theme in the shortlist.

## 8. Validation / QA
- Spot-check each shortlisted cluster's member summaries (does the theme hold?).
- Confirm the "no existing CE" claim for the top themes against the CE catalog (re-use `06`'s
  cross-agency CE landscape + the eCFR check).
- Sanity bounds: total clustered + noise == 217; no cluster spans obviously unrelated actions.

## 9. Open decisions / risks (for the reviewer)
1. **Funding vs physical action.** Many "other" are DOE loans/grants. A CE codifies a *physical*
   action, not financing — exclude funding-only themes, or reframe to the underlying build? (Plan:
   the LLM screen tags `funding_not_codifiable`; exclude from the shortlist.)
2. **Transmission-adjacent themes** (access roads, interconnection taps) — fold into the existing
   transmission analysis instead of counting as net-new? (Plan: LLM screen tags
   `belongs_to_existing_type`.)
3. **Recurrence threshold R** — how many FONSIs make a theme "worth a CE"? Proposed R = 3–5; this
   sets `min_cluster_size` and the shortlist floor.
4. **Pool scope** — bounded-only (217), or all 314 with boundedness as a ranking factor?
5. **One FONSI → one theme** — `potential_ce_theme` is single-valued, so no multi-membership; fine.
6. **Cluster stability** — HDBSCAN params (`min_cluster_size`, `min_samples`, metric) need a short
   tuning pass; document the chosen values + a stability check (cluster count vs params).

## 10. Dependencies & tooling
- **Available in `nepa` env:** `sentence_transformers`, `sklearn`, `anthropic`. (`06` already uses
  sklearn + `all-MiniLM-L6-v2`.)
- **Not installed:** `bertopic`, `umap-learn`, `hdbscan`. The recommended path needs **`hdbscan`**
  (one small package); the fallback needs nothing.

## 11. Cost & effort
- Embedding + clustering: **free** (local, ~217 short texts).
- LLM cluster naming/screening: **~$0.50–$2** (one call per cluster, ~20–40 clusters, Sonnet,
  cached).
- Build effort: ~half a day (one script + a report section + a linked page).

## 12. Reproducibility
Same guarantees as the classify pass: pinned embedding model, deterministic clustering (fixed
params/seed), LLM naming on a **pinned Sonnet snapshot at temp 0 with a committed cache** keyed on
cluster content + `NETNEW_PROMPT_VERSION`. Re-running regenerates the committed outputs; no
hand-edited values. Data parquets reproduce via the script (the cache makes the LLM step free on
re-run).

## 13. Questions for the reviewer (Codex)
1. Is **HDBSCAN + LLM naming** the right call over the no-dep `AgglomerativeClustering` fallback,
   given n≈217 and the goal of isolating one-offs? Any reason to prefer the full BERTopic stack?
2. Is **bounded-only (217)** the right pool, or should net-new be judged on all 314?
3. Is the **funding-not-codifiable** exclusion correct, or should "federal financial assistance to
   build X" count toward an X CE?
4. Is screening/naming **per cluster** (cheap, ~30 calls) sufficient, or should the model also see
   each FONSI to avoid a bad cluster summary driving the verdict?
5. Recurrence floor **R** — what threshold makes a theme a credible CE candidate?
6. Anything that would make this analysis mislead the client (e.g., over-reading a small cluster
   as a "recurring" net-new opportunity).
