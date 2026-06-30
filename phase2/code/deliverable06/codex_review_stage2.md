# Code review request (Codex): D6 Stage-2 LLM classification

You are reviewing a specific, self-contained change. Be skeptical and concrete. Point to
exact files/lines. The author is a data scientist; the deliverable is client-facing
(Clean Air Task Force), so correctness and reproducibility matter more than cleverness.

## Commit under review

`268943a` — "[D6] add Stage-2 LLM classification (--stage classify) to fix action_category"

Files changed (all under `phase2/code/deliverable06/`):
- `prompts.py` — new `build_classification_prompt()`, `classification_tool_schema()`,
  `CLASSIFICATION_PROMPT_VERSION`, `ACTION_CATEGORIES`.
- `enrich_lib.py` — new `call_classification()` and `classify_preflight()` (mirror
  `call_enrichment` / `preflight`); added two imports from `prompts`.
- `03_enrich_llm.py` — refactored into `run_extraction()` + new `run_classification()`,
  driven by a new `--stage {both,extract,classify}` flag (default `both`).

## Why this change exists

D6 groups ~451 clean-energy EA→FONSIs into action types to find categorical-exclusion (CE)
candidates. Each FONSI is read once by Claude Sonnet in `03` (a 39-field structured
extraction), which also assigns `action_category`. That field's instruction was just six bare
labels with **no definitions and no enum constraint**, so keyword-similar actions were
mislabeled: a botanical "Experimental Garden **Array**" → `solar`; a BLM **land withdrawal** →
`solar`; a VHF two-way **radio** upgrade → `transmission_upgrade`.

Re-running the whole 39-field extraction to fix one field costs ~$19 **and** would regenerate
(and perturb) the already-finalized Analysis-2 fields (mitigation, significance thresholds,
verified quotes). So classification was split into a **separate, cheaply-cached stage** that
re-asks ONLY the category from the **already-extracted summary** (no document re-read), with
real definitions + an enum schema, and **overwrites** `action_category` (preserving the
extraction value as `action_category_pass1`).

## Design intent (verify it actually holds)

- `--stage both` (default): extraction (cache-aware) then classification → a from-scratch run
  is fully correct in one command.
- `--stage classify`: reuse the committed extraction output, re-classify only (~$1.49 for 452,
  Sonnet, temp 0). This is how the author fixes the classifier **now** without re-paying $19.
- Two independent caches: extraction keyed on `PROMPT_VERSION|SCHEMA_VERSION|model|packet`
  (`cache_key`), classification keyed on `CLASSIFICATION_PROMPT_VERSION|model|summary-prompt`
  (`classify_key`). Bumping the classify version forces a classify-only re-run; the extraction
  cache is untouched.
- Reproducibility: committed `fonsi_enrichment.parquet` is canonical; anyone re-runs the script
  (pinned model, temp 0) to regenerate. No hand-edited values.

## Explicitly OUT OF SCOPE for this review (do not flag as missing)

- The **downstream rewiring is not done yet.** `07/08/09` and the report still build Analysis 1
  from the keyword `candidate_category`; switching them to the corrected `action_category` +
  shape fields (`within_existing_row`, `line_miles`, `new_access_road`) is the *next* commit.
- The real classify pass has not been run yet (no API key in this environment); only
  compile + dry-run + a mock end-to-end test were done.

## Review questions

1. **Cache correctness.** Is `classify_key` sufficient to guarantee determinism and correct
   invalidation? The prompt text embeds the summary, so a changed summary changes the key — good
   — but the *definitions* live in `build_classification_prompt`, not in the key except via
   `CLASSIFICATION_PROMPT_VERSION`. If someone edits the prompt wording **without** bumping the
   version, stale cached answers are silently reused. Is that an acceptable footgun, or should
   the key hash the prompt template too?
2. **Overwrite + audit logic** (`run_classification`). `action_category_pass1` is set only if
   absent, then `action_category` is overwritten. Trace a *second* `--stage classify` run on an
   already-classified parquet: is `pass1` still the extraction value (not the
   previously-classified value)? Is the NaN-safe `n_changed` correct?
3. **Failure / skip handling.** No-summary rows are skipped (keep their category); a failed API
   call (`parsed is None`) keeps the pass-1 category. Is silently keeping pass-1 on failure the
   right call for a client deliverable, or should the run fail loudly above some error rate (the
   extraction stage enforces `MIN_SUCCESS_*`; classification does not)?
4. **Is classifying from the summary sound?** The summary is ~100 tokens and was written by the
   same model in pass 1. Does re-classifying from it (rather than the full evidence packet) risk
   propagating a bad summary, or systematically lose signal the packet had? The shape fields
   (`within_existing_row`, `line_miles`) still come from extraction — is that split coherent?
5. **Prompt quality.** Do the category definitions + the "physical action, not keywords" rules in
   `build_classification_prompt` actually kill the named failure modes (garden→other, grant→other,
   radio→other, new-build→other)? What categories or edge cases are under-specified (e.g.,
   battery storage, hydro, distribution vs transmission, hybrid solar+storage)?
6. **The vestigial extraction `action_category`.** Extraction still produces a coarse
   `action_category` that classification overwrites — so a from-scratch `both` run asks for the
   category twice (once wasted). Acceptable wart, or should `action_category` be removed from the
   extraction schema (which would force a one-time $19 re-extraction)?
7. **Refactor fidelity.** `run_extraction` is the old `main()` body moved into a function. Diff it
   against the prior version: is any behavior changed (cache use, checkpointing, the
   `MIN_SUCCESS` gate, the evidence/coverage CSVs, the `suffix`/`mode` handling)?
8. **Thread-safety / SDK use.** `call_classification` mirrors `call_enrichment` (shared client,
   `max_retries`, `tool_choice` forced). Any concurrency or error-path issue?
9. **Anything that would embarrass us with the client**, or a simpler/cheaper/more robust design.

## How to verify locally

```
conda run -n nepa python 03_enrich_llm.py --dry-run            # both-stage cost preview, no key
conda run -n nepa python 03_enrich_llm.py --stage classify --dry-run   # ~$1.49, no key
conda run -n nepa python -m py_compile prompts.py enrich_lib.py 03_enrich_llm.py
```
