# Implementation Spec: spaCy Dependency Parsing for Date Context Classification

**Goal:** Replace the keyword-window approach in `get_date_context()` with spaCy dependency parsing to produce richer, syntactically grounded features for date type classification. This targets the CE initiation gap and EA/EIS indirect-reference misclassification without replacing BERT or the regex extraction layer.

---

## Background

Read these files before starting:
- `code/extract/extract_timeline.py` — current pipeline; focus on `get_date_context()` (line ~431), `DATE_CONTEXT_KEYWORDS`, and `extract_dates_from_text()`
- `notes/architecture/timeline_refactor.md` — architectural context and three-tier strategy

The current `get_date_context()` extracts a 100-character substring window around a date match and checks for keyword presence. This fails for indirect references like:
- `"Application received: 01/15/2020"` — `received` is not in `DATE_CONTEXT_KEYWORDS['start']`
- `"The FONSI was issued in August 2018"` — `issued` is not in any keyword list
- `"RoW application submitted January 2019"` — returns `unknown`

The fix: walk the dependency tree from the date token up to its head verb, then use the verb lemma and subject noun as classification signals.

---

## What to implement

### 1. Add spaCy dependency context extraction

Add a new function `get_date_context_spacy(sentence: str, date_char_start: int, date_char_end: int) -> dict` to `extract_timeline.py`.

The function should:
1. Load `en_core_web_sm` model (lazy-load it once as a module-level variable)
2. Parse the sentence containing the date using `nlp(sentence)`
3. Identify the token(s) overlapping the date span by character offset
4. Walk up the dependency tree from the date token until reaching a VERB or AUX token (or the root)
5. Collect the head verb's lemma and any nsubj/nsubjpass children
6. Return a dict: `{"head_verb": str, "subject": str|None, "dep_path": str}`

```python
# Lazy-load pattern to avoid loading spaCy on import
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP
```

### 2. Add a verb-to-label rule map

Add a module-level dict `VERB_LABEL_MAP` that maps head verb lemmas to date context labels. Use the same label names as `DATE_CONTEXT_KEYWORDS`:

```python
VERB_LABEL_MAP = {
    # Decision
    "sign":     "decision",
    "approve":  "decision",
    "issue":    "decision",
    "authorize": "decision",
    "determine": "decision",
    "adopt":    "decision",
    "execute":  "decision",
    # Initiation / submission
    "receive":  "submission",
    "submit":   "submission",
    "file":     "submission",
    # Notice
    "publish":  "notice",
    "notice":   "notice",
    "post":     "notice",
    # Draft / final
    "release":  "draft",   # default; refine with subject if needed
    "circulate": "draft",
    # Scoping
    "conduct":  "scoping",  # "scoping meeting was conducted"
    "hold":     "scoping",  # "scoping meeting was held"
    # Comment
    "close":    "comment",  # "comment period closes"
    "end":      "comment",
    "open":     "comment",
}
```

Refine `receive`/`submit` disambiguation: if subject contains `"application"` or `"request"`, map to `"initiation"` instead of `"submission"`.

### 3. Update `get_date_context()` to use spaCy as primary signal

Modify `get_date_context()` to call the new spaCy function and try the verb map first, falling back to the existing keyword window if spaCy returns no confident result:

```python
def get_date_context(text, match_start, match_end, window=100):
    # --- New: spaCy dependency path ---
    sentence = _extract_sentence(text, match_start, match_end)
    if sentence:
        spacy_result = get_date_context_spacy(sentence, ...)
        verb = spacy_result.get("head_verb")
        subj = spacy_result.get("subject", "")
        if verb in VERB_LABEL_MAP:
            label = VERB_LABEL_MAP[verb]
            # Refine: receive/submit + application subject → initiation
            if label == "submission" and subj and any(w in subj for w in ("application", "request", "row")):
                label = "initiation"
            return label

    # --- Fallback: existing keyword window ---
    start = max(0, match_start - window)
    end = min(len(text), match_end + window)
    context = text[start:end].lower()
    for context_type, keywords in DATE_CONTEXT_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in context:
                return context_type

    return 'unknown'
```

Add a helper `_extract_sentence(text, match_start, match_end)` that returns the sentence containing the match. Use simple sentence boundary detection: find the nearest preceding and following newline or sentence-ending punctuation within a 500-char window.

### 4. Add `--use-spacy` flag to CLI

Add an optional `--use-spacy` boolean flag to the argument parser. When set, `get_date_context()` uses the spaCy path. Default to `False` for backwards compatibility. Document it in the `--help` text.

When `--use-spacy` is not set, the pipeline behaves exactly as before.

### 5. Batch processing with `nlp.pipe()`

When running `--regex-prep` with `--use-spacy`, collect all sentences across all projects into a list and process them with `nlp.pipe(texts, batch_size=256)` rather than calling `nlp(text)` per date. This avoids loading the model overhead per document.

---

## What NOT to change

- Do not modify the regex date extraction patterns (`DATE_PATTERNS`, `parse_date_match()`, `extract_dates_from_text()`)
- Do not modify the BERT training or inference code
- Do not modify the LLM adjudication path
- Do not change the output schema of `regex_candidates.parquet`
- `get_date_context()` must continue to return the same label strings (`'start'`, `'decision'`, `'unknown'`, etc.) — downstream BERT training labels are unchanged

---

## Dependency

```
pip install spacy
python -m spacy download en_core_web_sm
```

Add to `requirements.txt` or conda env:
```
spacy>=3.7
```

---

## Testing

After implementing, run:

```bash
python code/extract/extract_timeline.py --regex-prep
python code/extract/extract_timeline.py --bert-run --sample 20 --use-spacy --output test20_spacy.parquet
```

Then compare `test20_spacy.parquet` against `test20_bert.parquet`:
- Check `bert_initiation_label` distribution: expect fewer `unknown` labels for CE projects
- Spot-check 5 CE projects where initiation was previously `unknown` — confirm spaCy assigned a label and it matches the document

Also verify backwards compatibility:
```bash
python code/extract/extract_timeline.py --bert-run --sample 20 --output test20_nospacy.parquet
```
Output should be identical to pre-change behavior.
