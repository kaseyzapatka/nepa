# D03 Visual Impact Analysis — Methods Notes

## Topic Modeling: NMF vs. LDA

### Why NMF (Non-negative Matrix Factorization)?

We use NMF with TF-IDF vectorization rather than LDA for this corpus. The key reasons:

**TF-IDF normalization matters here.** Visual impact sections share a universal vocabulary — nearly every section uses words like *visible*, *quality*, *color*, *form*, *view*, *contrast*. These terms appear in 50–70% of all documents. TF-IDF down-weights these high-frequency, low-discrimination terms by their document frequency, so NMF's topics are built from terms that are *relatively distinctive* to a subset of documents rather than universal to all of them. LDA uses raw term counts (no normalization), so high-frequency terms like *impact*, *visual*, *quality* dominate every topic regardless of their discriminating power.

**NMF produces parts-based additive topics.** NMF factorizes the term-document matrix into non-negative parts, which tend to correspond to coherent sub-vocabularies. The result is topics that read more like a coherent theme (e.g., "shadow flicker / blade / turbine / frequency") rather than a probability distribution over all terms including noise. LDA's generative model works well for long, topically diverse documents (e.g., news corpora); for thematically focused corpora like visual impact sections, it tends to produce noisy topics with accidental co-occurrences (numbers, geographic names, stopword leakage).

**Empirical comparison result.** We ran both models on the same corpus (n ≈ 1,300 training documents after stopword filtering). NMF produced 4–5 interpretable topics aligned to visual impact type and project context. LDA produced topics dominated by number tokens ("14", "15", "000") and geographic clusters — artifacts of raw count representation in a corpus where documents vary widely in length.

### Stopword Strategy

Standard English stopwords are augmented with three domain-specific layers:

1. **Visual boilerplate** (`visual`, `scenic`, `aesthetics`, `viewshed`, `view`, `views`, `scenery`, `landscape`): These appear in nearly every visual section and define the topic space rather than discriminate within it.
2. **BLM administrative jargon** (`blm`, `vrm`, `kop`, `management`, `plan`, `class`, `lands`): BLM Resource Management Plans (large programmatic EIS documents) otherwise dominate the corpus and produce spurious "BLM RMP" topics that are not about visual impact types.
3. **NEPA process boilerplate** (`impact`, `impacts`, `effect`, `effects`, `significant`, `mitigation`, `analysis`, `alternative`): Universal across all NEPA documents; not discriminating between visual impact types.
4. **Roman numerals** (`ii`, `iii`, `iv`, `vi`, `vii`, `viii`, `ix`, `xi`, `xii`): BLM VRM class designations ("Class II VRM", "Class III areas") cause these to become dominant NMF terms unless stopped.

Project-type vocabulary (`turbine`, `blade`, `pipeline`, `pad`, `well`) is intentionally *not* stopped — these terms are the primary discriminating signal that separates wind/solar shadow flicker topics from O&G corridor topics. The resulting topics partially reflect project type, which is expected and interpretable.

---

## Topic Modeling: The Full Pipeline (Code Walkthrough)

All code is in `build_topics()` in `phase2/code/deliverable03/02_build_nepa_reviews.py`.

### Step 1 — Text input (lines ~1497–1500)

Uses `visual_analysis_text` — the project's visual section text after two cleaning passes:

1. `_clean_for_analysis()`: strips TOC dot-leader lines, VRM acreage tables, OCR garbage lines, dash-table rows, URL lines, project ID codes (`ABC-123`), and page cross-references.
2. `_extract_impact_sentences()`: keeps only sentences containing specific visual impact phrases (shadow flicker, glare, viewshed, contrast rating, VRM class, night sky, etc.). Projects with fewer than 3 matching sentences fall back to their full cleaned text.

**This is the biggest lever** — too strict a sentence filter and NMF lacks signal; too loose and project-description boilerplate dominates.

### Step 2 — TF-IDF vectorizer (lines ~1489–1495)

```
ngram_range=(1, 3)   # unigrams + bigrams + trigrams
min_df=5             # term must appear in ≥5 projects (drops rare project-specific terms)
max_df=0.55          # term must appear in ≤55% of projects (drops near-universal terms)
max_features=10,000  # vocabulary cap — DO NOT raise above 10k (see note below)
stop_words=nepa_stop # NEPA_DOMAIN_STOPWORDS + sklearn English stops
```

**Critical: keep `max_features=10,000`.** Raising to 15,000 adds low-discrimination terms that cause NMF to produce only 2 meaningful topics (the other 2 components learn near-identical patterns and get zero project assignments at argmax). Empirically tested: trigrams + 10k features restores all 4 topics; trigrams + 15k features collapses to 2.

**`scenic` is in `NEPA_DOMAIN_STOPWORDS`.** The bigram `scenic quality` appears in 42.6% of documents — frequent enough that, when `scenic` is a free token, NMF anchors 3 of its 4 components to this one boilerplate phrase and collapses the topic structure. `visual` and `visible` are intentionally *not* stopped: their bigrams (`visual contrast` 36%, `visual character` 36%) are below max_df=0.55 and provide useful signal. `landscape` is also free but its unigram (66%) is dropped by max_df; its bigram `landscape character` (30%) survives and is discriminating.

**`max_df=0.55` keeps `contrast` (45.1%) in vocabulary.** At max_df=0.4, `contrast` was dropped, collapsing the VRM contrast rating topic. At 0.55, it survives and anchors Topics 0 and 3.

Every run writes `phase2/output/deliverable03/nmf_vocab_diagnostic.csv` showing which terms were kept, dropped as rare (`doc_freq < 5`), or dropped as universal (`pct_docs > 55%`). Inspect this file to verify the filters are behaving as intended.

### Step 3 — NMF factorization (lines ~1505, 1520)

```python
n_components = min(4, max(2, len(train_texts) // 10))
# → with ~1,310 training docs this gives 4 topics
NMF(n_components=4, random_state=42, max_iter=400, alpha_W=0.001)
```

Trains on heading-anchored projects (~1,310), then transforms the ~281 fallback-only projects. `alpha_W=0.001` applies light L1 regularization on document-topic weights, encouraging sparse topic assignments.

### Step 4 — Topic label building (lines ~1538–1562)

Pulls the top 12 terms per topic by NMF component weight. Labels prioritize terms from `_IMPACT_LABEL_VOCAB` (shadow flicker, glare, contrast, viewshed, night sky, etc.) before filling with remaining top terms. This ensures labels surface impact-type vocabulary (e.g., "shadow / flicker") rather than generic project terms (e.g., "wind / turbine / area").

### Fine-tuning levers

| Lever | Where | Effect |
|---|---|---|
| `_VISUAL_IMPACT_SENT_RE` | line ~673 | Controls which sentences enter `visual_analysis_text`. Too strict → NMF starved; too loose → boilerplate dominates. |
| `_is_noisy_line()` | line ~750 | Filters VRM tables, OCR garbage, dash tables, URLs before sentence filtering. |
| `NEPA_DOMAIN_STOPWORDS` | line ~620 | Words removed before NMF. Add terms that dominate topics without being informative. |
| `n_components` | line ~1505 | Number of topics. Currently capped at 4; lower = broader topics, fewer redundant clusters. |
| `min_df` | line ~1492 | Minimum document frequency to enter vocabulary. `min_df=5` excludes rare project-specific terms. |
| `max_df` | line ~1493 | Maximum document frequency proportion. `max_df=0.7` excludes near-universal terms. |
| `_IMPACT_LABEL_VOCAB` | line ~1538 | Only affects topic label strings, not assignments. Add terms to surface impact vocab in labels. |

---

## The Current Topics (4-topic run)

Current 4-topic structure (as of May 2026 run; ngram=(1,3), max_df=0.55, max_features=10k, scenic stopped):

| Topic | n (projects) | Top discriminating terms | Interpretation |
|-------|-------------|--------------------------|----------------|
| 0 | 432 (299 decarb, 133 fossil) | contrast, visual contrast, rating, objectives, glare, **solar**, sensitivity, moderate, contrast rating | **VRM Contrast Rating & Solar Glare** — BLM VRM rating methodology applied to solar and transmission projects; glare from panels; VRI sensitivity; moderate contrast outcomes |
| 1 | 69 (66 decarb, 3 fossil) | **shadow, flicker, shadow flicker**, turbine, wind, turbines, hours, receptor, year, wind turbine | **Wind Turbine Shadow Flicker** — specialized analysis of rotating-blade shadow cast on nearby receptors; annual shadow hours vs. regulatory thresholds |
| 2 | 804 (448 decarb, 356 fossil) | transmission, light, visual character, river, line, lighting, industrial, **structures, plant**, glare, terminal, station, byway, natural, integrity | **Industrial & Infrastructure Corridors** — O&G, pipeline, and transmission projects; structures and industrial facilities; river corridor crossings; scenic byways; artificial lighting impacts |
| 3 | 286 (103 decarb, 183 fossil) | objectives, contrast, **managed, integrity, vri**, landscape character, classes, dominate, sensitivity | **BLM VRM Objectives & Landscape Management** — formal VRM compliance framework for O&G/pipeline EIS; managed VRM classes; Visual Resource Inventory; dominance; scenic integrity objectives |

**Term-weight profile notes** (from fig14b):
- Topic 1 has a very sharp elbow: shadow/flicker/shadow flicker weights (1.2) are 3× the next term. Textbook coherent topic.
- Topic 0: smooth decline; "contrast" at 0.80 is the anchor but solar/glare secondary terms distinguish it from Topic 3.
- Topic 3: objectives and contrast nearly tied at 0.38/0.37; distinguished from Topic 0 by managed/vri/classes vocabulary.
- Topic 2: completely **flat profile** — all 10 terms between 0.20–0.24. This is a residual catch-all for infrastructure/corridor projects, not a poorly-separated topic. These project types genuinely share visual-impact prose and NMF cannot further discriminate them.

**Why does "contrast" appear in Topics 0 AND 3?** NMF components are additive basis vectors, not exclusive clusters. "Contrast" appears in ~45% of all documents (the most common visual-impact term after boilerplate). Both components have non-zero loadings for it — the assignments are made by argmax, so a document goes to whichever component has the highest total weight. The interpretive distinction is in the secondary vocabulary, not the primary anchor term.

**Why is the NMF auto-label for Topic 2 "glare / transmission / light" when glare has higher weight in Topic 0?** Label artifact: the auto-labeler prioritizes impact-vocabulary terms (`_IMPACT_LABEL_VOCAB`), picking "glare" as the first such term in Topic 2's term list. The actual weight of glare in Topic 2 (0.20) is lower than in Topic 0 (0.38). Use the interpretive labels in all figures, not the auto-generated labels.

**Should we use 3 topics instead of 4?** 3 topics (merge 0+3 into "VRM contrast") would simplify but lose the solar/glare vs. O&G compliance distinction. 5+ topics would produce more flat-profile residual bins without meaningful new themes. 4 is the right number given this corpus.

---

## Vocab Diagnostic File

`phase2/output/deliverable03/nmf_vocab_diagnostic.csv` is written on every run. Columns:

| Column | Meaning |
|--------|---------|
| `term` | The ngram (unigram or bigram) |
| `doc_freq` | Number of training documents it appears in |
| `pct_docs` | Percentage of training documents |
| `status` | `kept` / `dropped_rare` (< 5 docs) / `dropped_universal` (> 70% of docs) / `dropped_max_features` |

To inspect what's being dropped in R:
```r
vocab <- read_csv("phase2/output/deliverable03/nmf_vocab_diagnostic.csv")
vocab |> filter(status == "dropped_universal") |> arrange(desc(pct_docs)) |> head(30)
vocab |> filter(status == "dropped_rare") |> arrange(desc(doc_freq)) |> head(30)
vocab |> filter(status == "kept") |> arrange(desc(pct_docs)) |> head(30)
```

---

## Framing vs. Sentiment Analysis

### What the framing figure (fig18) measures

`build_framing()` applies three domain-specific CEQ §1508.27-anchored lexicons to each project's visual section text:

- **Adversity ratio**: count of adverse/negative phrases ÷ count of all directional phrases. Negative phrases: *adverse*, *detrimental*, *degrade*, *harm*, *damage*. Positive: *beneficial*, *enhance*, *improve*. Neutral: *no effect*, *negligible*.
- **Significance ratio**: count of high-significance phrases ÷ total significance phrases.
- **Mitigation ratio**: count of strong/specific mitigation language ÷ total mitigation phrases.

This is domain-specific sentiment, not general sentiment. It is **negation-aware** — *no significant adverse impact* does not count as "adverse."

### Why general sentiment tools are wrong for this corpus

VADER, TextBlob, and similar tools would classify nearly all NEPA text as "negative" because they treat *adverse*, *impact*, *affect*, *damage*, *significant* as negative words — but in NEPA these are neutral technical terms. A finding of "no significant adverse visual impact" would score as strongly negative with general sentiment tools, which is the opposite of the actual meaning.

### What would improve on the current framing figure

1. **Extended adversity lexicon**: Add more domain-specific pairs to the existing lexicon (e.g., *incompatible*, *intrusive*, *dominant*, *overwhelm* → adverse; *compatible*, *subordinate*, *blend*, *screened* → beneficial).
2. **Intensity scoring**: Weight terms by severity (*severely degrade* > *slightly alter*).
3. **Zero-shot NLI**: Frame the question as a hypothesis — "This text concludes that the project has a significant adverse visual impact" — and score each project's text against it using a cross-encoder NLI model (e.g., `cross-encoder/nli-deberta-v3-base`). More flexible than a fixed lexicon and handles novel phrasing. Medium effort.
4. **LLM-based structured scoring**: Ask Claude to read each visual section and return structured scores (impact direction, severity, mitigation quality). Most accurate but expensive at scale (~1,600 projects).

The current CEQ lexicon approach is appropriate for the scale and interpretability requirements of this project. Option 3 (zero-shot NLI) is the most practical upgrade path.

---

## VRM Compliance Flag (Option A)

Two new regex patterns (`VRM_MEETS_RE`, `VRM_EXCEEDS_RE`) are applied per sentence in `_count_framing_axes()` and exposed in `build_framing()` output:

- `vrm_compliance_flag` — True if any sentence asserts the project *meets* VRM objectives and no sentence asserts it *exceeds* them.
- `vrm_noncompliant_flag` — True if any sentence asserts the project *exceeds* VRM class objectives.
- `vrm_meets` / `vrm_exceeds` — raw counts per project.

**Compliance terms**: *consistent with...objective/class/VRM*, *meets...objective/class/VRM*, *within...VRM...class*, *no adverse contrast*, *would comply*, *in compliance with*.

**Non-compliance terms**: *exceed...objective/class/VRM*, *would not meet...objective/class/VRM*, *inconsistent with...objective/class/VRM*, *above...VRM...class*, *strong contrast...exceed*.

After NMF topic assignment, `build_topics()` writes `vrm_topic_compliance_diagnostic.csv` cross-tabbing VRM-citing projects by topic × compliance flag. Inspect this file to determine whether compliant and non-compliant VRM findings cluster into separate NMF topics. If they do not, keep the contrast topic unified (Option D) and rely on the element-level analysis (fig21) for differentiation.

---

## VRM Element-Level Contrast Rating Extraction (fig21)

`build_vrm_elements()` extracts per-element BLM VRM contrast ratings from visual section text. Elements: **form**, **line**, **color**, **texture**, **scale**, **vividness**. Ratings normalized to: **None**, **Weak**, **Moderate**, **Strong**.

### Four extraction patterns (in priority order):
1. **Table/list**: `Form: Strong` — element then rating, separated by `:`, `—`, or whitespace.
2. **Contrast-of**: `contrast of form: strong` — explicit "contrast of/in element" phrasing.
3. **Rating-then-contrast**: `strong contrast in form` — rating word precedes "contrast in/of element".
4. **Rating-element-contrast**: `strong form contrast` — rating word precedes element word before "contrast".

When multiple ratings are found for the same element (e.g., from multiple sections), the **strongest** is kept.

### Outputs:
- `vrm_elements.parquet` — long format: one row per (project_id × element × rating).
- `vrm_elements_summary.csv` — element × energy_group × rating with n_projects and %.
- `fig21_vrm_elements.png` — 100% stacked bar, faceted by Decarbonization vs Fossil Fuel.

### Interpretation guidance:
- Coverage will be partial (~20–40% of projects): only EIS documents with explicit VRM methodology tables use the element-level vocabulary. EA documents and non-BLM projects rarely do.
- Low coverage does not invalidate the finding — it identifies which project types use the formal VRM rating framework.
- Compare decarb vs fossil distributions to see whether wind/solar projects get different element-level contrast ratings than O&G corridor projects.

---

## Note on fig14b (Term Weight Chart)

The faceted lollipop chart (`fig14b_topic_terms.png`) shows NMF component weights for the top 10 terms per topic. NMF weights are not directly interpretable as probabilities or frequencies — they are dimensionless coordinates in the factorized space. The chart is most useful for:

- Confirming that topic labels reflect the highest-weight terms (internal validation)
- Showing relative term importance *within* a topic (term A is 3× more central than term B)
- Identifying whether a topic has a sharp "elbow" (one term dominates) or a flatter profile (several terms equally contribute)

It is NOT useful for comparing term weights *across* topics (scales differ between facets), and should not be presented to a non-technical audience without explanation.

---

## Noise-line calibration corpus (`visual_impact_to_remove.txt`)

`phase2/notes/deliverable03/visual_impact_to_remove.txt` is a hand-collected corpus of **example lines to exclude** from extracted visual-impact section text — table spillage (VRM/VRI class-acreage rows, emissions-comparison tables), OCR garbage from scanned PDFs, URL-only lines, and lines dominated by numbers. It is a human-curated reference, **not read by any code at runtime**: `02_build_nepa_reviews.py`'s `_is_noisy_line()` filter (regexes `_VRM_CLASS_RE`, `_DASH_CELL_RE`, `_URL_RE`, `_OCR_JUNK_RE`, `_NUMERIC_TOKEN_RE`, etc.) was **generalized from** these examples and the file is cited in two comments there (near lines ~789 and ~807) as the calibration source. The text is verbatim from public federal NEPA documents.

Do not add a header or reformat the file: although nothing parses it today, its value is as a faithful sample of the raw noise the filters must catch. If the noise filters are ever retuned, extend this corpus with new failing examples first, then adjust the regexes to cover them.

---

## Dormant timeline section (`04_create_figures.R` Section 6)

`04_create_figures.R` carries a **Section 6** that is gated off at runtime: it activates only when
`phase2/data/analysis/timeline.parquet` exists, and that file is intentionally absent — timeline
analysis is delivered by **Deliverable 4**, not D3. The section is kept only as a stub. This note
transplants the two latent-bug records that were previously held only in the (now-deleted)
`plans/deliverable03_update.md`, so they are not lost if the section is ever revived:

1. **Wrong FRA cut date.** Section 6 hardcodes `as.Date("2023-08-16")` (the CEQ-rule effective
   date). If revived, change it to **`2023-06-03`** (FRA enactment) to match D4's `FRA_CUT_DATE` and
   the Phase 1 D5 convention. Using 2023-08-16 would misclassify pre/post-FRA reviews in the ~2.5
   month gap.
2. **Register-anchored duration artifact.** Section 6 re-derives `duration_days` from raw
   start/decision dates **without `initiation_source_type` awareness**. It would therefore reproduce
   the fossil-EA ~40-day artifact: land-based oil & gas EAs whose initiation is a Federal Register
   notice show a register-anchored median of ~40 days, versus a document-anchored ~154 days — a known
   D4 finding. Any revived D3 duration analysis must branch on `initiation_source_type` (as D4 does)
   before reporting durations, or it will understate fossil-EA timelines by roughly 4×.
