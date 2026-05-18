# D03 Visual Impact Analysis — Methods Notes

## Topic Modeling: NMF vs. LDA

### Why NMF (Non-negative Matrix Factorization)?

We use NMF with TF-IDF vectorization rather than LDA for this corpus. The key reasons:

**TF-IDF normalization matters here.** Visual impact sections share a universal vocabulary — nearly every section uses words like *visible*, *quality*, *color*, *form*, *view*, *contrast*. These terms appear in 50–70% of all documents. TF-IDF down-weights these high-frequency, low-discrimination terms by their document frequency, so NMF's topics are built from terms that are *relatively distinctive* to a subset of documents rather than universal to all of them. LDA uses raw term counts (no normalization), so high-frequency terms like *impact*, *visual*, *quality* dominate every topic regardless of their discriminating power.

**NMF produces parts-based additive topics.** NMF factorizes the term-document matrix into non-negative parts, which tend to correspond to coherent sub-vocabularies. The result is topics that read more like a coherent theme (e.g., "shadow flicker / blade / turbine / frequency") rather than a probability distribution over all terms including noise. LDA's generative model works well for long, topically diverse documents (e.g., news corpora); for thematically focused corpora like visual impact sections, it tends to produce noisy topics with accidental co-occurrences (numbers, geographic names, stopword leakage).

**Empirical comparison result.** We ran both models on the same corpus (n ≈ 1,800 documents after stopword filtering). NMF produced 5 interpretable topics aligned to visual impact type and project context. LDA (12 components, online learning) produced topics dominated by number tokens ("14", "15", "000") and geographic clusters — artifacts of raw count representation in a corpus where documents vary widely in length.

### Stopword Strategy

Standard English stopwords are augmented with three domain-specific layers:

1. **Visual boilerplate** (`visual`, `scenic`, `aesthetics`, `viewshed`, `view`, `views`, `scenery`, `landscape`): These appear in nearly every visual section and define the topic space rather than discriminate within it.
2. **BLM administrative jargon** (`blm`, `vrm`, `kop`, `management`, `plan`, `class`, `lands`): BLM Resource Management Plans (large programmatic EIS documents) otherwise dominate the corpus and produce spurious "BLM RMP" topics that are not about visual impact types.
3. **NEPA process boilerplate** (`impact`, `impacts`, `effect`, `effects`, `significant`, `mitigation`, `analysis`, `alternative`): Universal across all NEPA documents; not discriminating between visual impact types.

Project-type vocabulary (`turbine`, `blade`, `pipeline`, `pad`, `well`) is intentionally *not* stopped — these terms are the primary discriminating signal that separates wind/solar shadow flicker topics from O&G corridor topics. The resulting topics partially reflect project type, which is expected and interpretable.

---

## Topic Modeling: How It Works (High-Level Summary)

1. **Input**: Each project's concatenated visual section text (heading-anchored sections or high-density fallback sections, concatenated in page order). One document per project, ~1,800 documents total.

2. **Vectorization (TF-IDF)**: The text is converted to a term-document matrix. Each cell contains the TF-IDF weight of a term in a document — high if the term is frequent in this document but rare across all documents. The vocabulary is filtered to 2–3 word n-grams with `min_df=10` (term must appear in at least 10 documents) and `max_df=0.85` (term must appear in at most 85% of documents).

3. **NMF Factorization**: The term-document matrix is factorized into two non-negative matrices: a *topic-term* matrix (what words define each topic) and a *document-topic* matrix (how much each document belongs to each topic). We fit `n_components=5` topics with `max_iter=400`.

4. **Topic assignment**: Each document is assigned its highest-weight topic (argmax of its row in the document-topic matrix). The topic label is formed from the top 3 terms, with visual impact terminology (flicker, glare, contrast, silhouette) surfaced ahead of project-type terms when both appear.

5. **Output**: Each project gets a `topic_id` (0–4) and `topic_label`. The `visual_topic_terms_detail.csv` shows the top 15 terms and their NMF weights for each topic, enabling inspection of what the model learned.

### The Five Topics (Current Run)

| Topic | Label | Core Vocabulary | Interpretation |
|-------|-------|-----------------|----------------|
| 0 | shadow flicker / turbine / blade | flicker, blade, turbine, frequency, shadow | Wind shadow flicker analysis |
| 1 | pipeline / corridor / right-of-way | pipeline, corridor, right-of-way, pad, reclamation | O&G and transmission corridors |
| 2 | contrast / color / form | contrast, color, form, texture, vividness | BLM VRM contrast rating methodology |
| 3 | glare / reflection / anti-reflective | glare, anti-reflective, reflection, panel, coating | Solar PV glare/glint analysis |
| 4 | night sky / dark sky / light | night sky, dark sky, flare, light pollution, skyglow | Artificial light / night sky impacts |

Topic 2 (contrast/color/form) reflects BLM's Visual Resource Management (VRM) contrast rating methodology, which uses a standardized vocabulary of *form*, *line*, *color*, *texture*, *vividness* to score how much a project contrasts with the surrounding landscape character. This methodology appears across multiple project types on BLM-managed land.

---

## Note on fig14b (Term Weight Chart)

The faceted lollipop chart (`fig14b_topic_terms.png`) shows NMF component weights for the top 10 terms per topic. NMF weights are not directly interpretable as probabilities or frequencies — they are dimensionless coordinates in the factorized space. The chart is most useful for:

- Confirming that topic labels reflect the highest-weight terms (internal validation)
- Showing relative term importance *within* a topic (term A is 3× more central than term B)
- Identifying whether a topic has a sharp "elbow" (one term dominates) or a flatter profile (several terms equally contribute)

It is NOT useful for comparing term weights *across* topics (scales differ between facets), and should not be presented to a non-technical audience without explanation. For the report, it is best included as a technical appendix figure or replaced with a simple term-list table.
