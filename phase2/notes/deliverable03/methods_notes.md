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
ngram_range=(1, 2)   # unigrams + bigrams
min_df=5             # term must appear in ≥5 projects (drops rare project-specific terms)
max_df=0.7           # term must appear in ≤70% of projects (drops near-universal terms)
max_features=10,000  # vocabulary cap
stop_words=nepa_stop # NEPA_DOMAIN_STOPWORDS + sklearn English stops
```

Every run writes `phase2/output/deliverable03/nmf_vocab_diagnostic.csv` showing which terms were kept, dropped as rare (`doc_freq < 5`), or dropped as universal (`pct_docs > 70%`). Inspect this file to verify the filters are behaving as intended.

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

After capping at 4 topics and adding Roman numeral stopwords, the expected topic structure is:

| Topic | Expected vocabulary | Interpretation |
|-------|---------------------|----------------|
| 0 | shadow, flicker, turbine, blade, frequency | Wind turbine shadow flicker analysis |
| 1 | glare, glint, anti-reflective, panel, coating, night sky | Solar glare / light pollution |
| 2 | contrast, rating, color, form, texture, vividness | BLM VRM contrast rating methodology |
| 3 | pipeline, corridor, pad, well, reclamation, disturbance | O&G / pipeline corridor impacts |

Topic 2 (contrast/color/form) reflects BLM's Visual Resource Management framework, which uses a standardized vocabulary of *form*, *line*, *color*, *texture*, *vividness* to score how much a project contrasts with surrounding landscape character. This methodology appears across multiple project types on BLM-managed land and is expected to be the broadest topic.

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

## Note on fig14b (Term Weight Chart)

The faceted lollipop chart (`fig14b_topic_terms.png`) shows NMF component weights for the top 10 terms per topic. NMF weights are not directly interpretable as probabilities or frequencies — they are dimensionless coordinates in the factorized space. The chart is most useful for:

- Confirming that topic labels reflect the highest-weight terms (internal validation)
- Showing relative term importance *within* a topic (term A is 3× more central than term B)
- Identifying whether a topic has a sharp "elbow" (one term dominates) or a flatter profile (several terms equally contribute)

It is NOT useful for comparing term weights *across* topics (scales differ between facets), and should not be presented to a non-technical audience without explanation.
