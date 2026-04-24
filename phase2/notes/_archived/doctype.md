An idea to use doc_type, which tags the document type from where a date comes.

```
Good news: doc_type is already wired in — partially. Here's the current state:
Decision selection (_select_best_decision(), line ~1727): already uses it


_doc_type_boost = 3 if doc_type in {'ROD', 'FONSI'} else 0
score = decision_strength + boilerplate_penalty + (2 × confidence) + doc_type_boost
Initiation selection (_select_best_initiation(), lines 1764–1799): ignores it entirely


score = _init_score + confidence   # no doc_type weighting at all
That's the gap. For initiation, the logic should be the inverse — a date from a DEA or DEIS is much more likely to be an initiation signal (draft publication ≈ start of review), while a ROD/FONSI date found in the initiation candidates is probably noise. Right now BERT has to figure that out from context alone.

The natural fix is to add a parallel _doc_type_init_boost to _select_best_initiation():


INITIATION_DOC_TYPES  = {'DEA', 'DEIS'}          # strong initiation signal
DECISION_DOC_TYPES_NEG = {'ROD', 'FONSI', 'FEIS'} # penalise in initiation

doc_type_init_boost = (
     2 if doc_type in INITIATION_DOC_TYPES   else
    -2 if doc_type in DECISION_DOC_TYPES_NEG else
     0
)
score = _init_score + confidence + doc_type_init_boost

```