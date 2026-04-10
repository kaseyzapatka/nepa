# DeBERTa Model Comparison for Timeline Classification

## Model Overview

| Model | Params | Avg GLUE | Best for |
|---|---|---|---|
| DeBERTa-v3-xsmall | 22M | ~79–80 | Very small datasets (<2K rows), fastest inference |
| DistilBERT | 66M | ~77 | Legacy baseline; knowledge-distilled from BERT |
| DeBERTa-v3-small | 44M | ~82 | Small-to-medium datasets (2K–15K rows) |
| DeBERTa-v3-base | 184M | ~87 | Large datasets (>15K rows); highest accuracy |

## This Pipeline

| Source | Model | Epochs | Train rows (approx) | Rationale |
|---|---|---|---|---|
| EA | deberta-v3-small | 5 | ~7,500 | Base collapsed (class imbalance + too many params) |
| EIS | deberta-v3-small | 5 | ~2,500 | Same reason; also small dataset |
| CE | deberta-v3-base | 3 | ~60,000+ | Large enough to support full base model |

## Why DeBERTa beats DistilBERT at fewer parameters (deberta-v3-small)

DistilBERT (66M) was distilled from BERT-base — it retains BERT's architecture but compresses it.
DeBERTa-v3 uses disentangled attention (separate position and content embeddings) + ELECTRA-style
pre-training, which produces substantially better representations even at fewer parameters.

- deberta-v3-small (44M): +5 GLUE points over DistilBERT despite 33% fewer parameters
- deberta-v3-xsmall (22M): +2–3 GLUE points over DistilBERT despite 67% fewer parameters

## Class collapse risk by model × dataset size

Class collapse (model predicts only majority class) occurs when the model has too many parameters
relative to training data diversity. Observed in this project: deberta-v3-base trained on EA data
(7.5K rows) collapsed entirely to "decision" (F1=0 for initiation/review/other).

Rule of thumb:
- deberta-v3-base needs ~20K+ balanced rows to learn minority classes reliably
- deberta-v3-small is safe at 5K–15K rows
- deberta-v3-xsmall could work at 1K–5K rows but hasn't been tested here

## Notes on xsmall

`microsoft/deberta-v3-xsmall` is available on HuggingFace. If EIS coverage remains poor after
retraining with deberta-v3-small, xsmall is worth trying — it has less capacity to overfit the
majority class. Tradeoff: lower ceiling accuracy on well-represented classes.

Pip/dependencies: all DeBERTa-v3 variants require `sentencepiece` (same as small/base).
