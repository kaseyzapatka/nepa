# D4 Timeline Classifier — Training & Workflow

_Last updated: 2026-06-04. Phase 1 (active-learning loop) complete. Phase 2 (calibration + operating curve) is next._

---

The classifying happens in `04_classify_candidates.py`, which is a **two-head SetFit scorer** (`p_initiation`,
`p_decision`) that sits between candidate extraction (`03_extract_candidates.py`) and date selection (`05_select_dates.py`). 

Its job is
classify dates as either `initiation`, `decision`, or `neither` and attach a measure of confidence. Per project, dates are selected by confidence level and fed to Claude in `06_adjudicate_llm.py`, who adjudicates between high confidence decisions and decide the final--most likely--initiation and decision dates


## HIGH-LEVEL ROADMAP

**Phase 1 — Active-learning loop on SetFit ✅ COMPLETE.**
Emit uncertain batch → label (rules in `labeling_rules.md`) → retrain → compare on frozen
test. Two rounds completed; decision head plateaued at F1=0.737 and init gains are diminishing
(+0.059 R1, +0.034 R2). AL loop declared done. See results table in "Current state" below.

**Phase 2 — Calibrate + build the operating curve.**
SetFit head outputs are uncalibrated (`p` caps ~0.735), so `0.5`/`0.7` thresholds are meaningless. Calibration teaches the model what its confidence scores really mean, so instead of guessing whether a score like 0.70 is trustworthy, you know from past data roughly how often predictions with that score are actually correct and can decide when to trust the model versus send the case for further review.
So we want to fit Platt/isotonic calibration on the frozen test, then produce a curve: for each threshold τ →
(auto-resolved %, precision at τ, # routed to LLM, est. API cost). This curve is both the
**confidence story** for the deliverable and the knob for routing thresholds.

**Phase 3 — Wire confidence into selection + routing.**
- `05_select_dates.py`: derive a per-project **timeline confidence** from the selected
  candidates' calibrated probs; add `low_confidence` / `conflicting` statuses.
- `06_adjudicate_llm.py`: routing already consumes classifier confidence (queue trigger +
  top-k packets — see below). Re-tune `ROUTE_CONF_THRESHOLD` against calibrated probs and
  revisit `ROUTED_TOPK` (currently 3 — risky at F1 ~0.55; the true date can rank below 3).

**Phase 4 — Run LLM adjudication + measure lift.**
Run `06 --mode candidate_adjudication` on the routed queue; measure coverage gain and $ cost.
Use `--dry-run --sample N` first to size the queue/cost.

**Phase 5 — SetFit → DeBERTa (deferred).**
Switch criteria: ~1,000–1,500 labels/head **and** SetFit frozen-test F1 plateaued. We're at
init=142 / decision=147 train positives — not close. The `TransformerBackend` stub in `04`
already matches the train/predict/save/load contract, so `05`/`06`/CLI don't change when we swap.
Optional side-by-side once ~300+/head.

**Cleanup (deferred — project to-do #22).**
Remove the candidate-level gold scripts (`labeling/01–05`) and data dirs
(`data/analysis/timeline/gold/`, `output/deliverable04/gold/`). KEEP project-level gold used by
`07_validate.py` (`timeline_gold_projects.parquet`) — that's separate end-to-end validation.


## Current state (2026-06-04)

**Single source of truth for labels:** `phase2/output/deliverable04/labeling_sample.csv`
(1,168 rows). The `split` column is **frozen**: 1,014 `train` / 154 `test`. New labels added later
default to `train` so the test set never grows or leaks.

| | full file | train | frozen test |
|---|---:|---:|---:|
| initiation | 160 | 142 | 18 |
| decision | 165 | 147 | 18 |
| neither | 843 | 725 | 118 |

### AL progression — frozen-test F1 (154 rows, same set every round)

| Model version | Train rows | Init F1 | Init P / R | Decision F1 | Decision P / R | Notes |
|---|---:|---:|---|---:|---|---|
| `20260603T222207Z` | 614 | 0.556 | .556 / .556 | 0.647 | .688 / .611 | Baseline (pre-AL) |
| `20260604T032402Z` | 814 | 0.615 | .571 / .667 | 0.737 | .700 / .778 | Round 1: +200 (52 dec, 17 init, 131 neither); uncertainty sampling |
| `20260604T060644Z` | 1,014 | **0.649** | .632 / .667 | **0.737** | .700 / .778 | Round 2: +200 (60 clear_init floor, 140 uncertain); **AL loop closed** |

**Round 2 error analysis (21/154 misclassified):**
- Init FP: 9 → 7 (clear_initiation seeding reduced neither→initiation noise)
- Decision: exactly flat — head has hit SetFit ceiling at 147 train positives
- 6 init false negatives inspected: 2 learnable (NEPA Register, application→decision), 1 structural (exempt `review` role, never scored in production), 3 hard/ambiguous — not worth another AL round
- Pool rescored with `20260604T060644Z` ✅ (285,747 eligible candidates)

**Uncalibrated probabilities remain the top gap:** the round 2 model reaches `max(p_init)=0.893` /
`max(p_dec)=0.927`, but the distribution is strongly bimodal — ~90% of candidates sit near 0,
then a sharp jump to 0.83+ at p95. The 0.3–0.7 middle band is nearly empty, so `06`'s
`ROUTE_CONF_THRESHOLD=0.70` is effectively a step function, not a precision knob.
Phase 2 calibration maps the raw scores to honest probabilities so the threshold means something.

### Pool scoring sanity check (model `20260604T060644Z`)

Run to verify scoring at any time:

```python
import pandas as pd, numpy as np
df = pd.read_parquet('phase2/data/analysis/timeline/timeline_candidates.parquet')
s = df[df['classifier_model_version'].eq('20260604T060644Z')]
p_i = pd.to_numeric(s['p_initiation'], errors='coerce').dropna()
p_d = pd.to_numeric(s['p_decision'],   errors='coerce').dropna()

print(f'Scored: {len(s):,} / {len(df):,}')
print('classifier_label:', s['classifier_label'].value_counts().to_dict())

for name, p in [('p_initiation', p_i), ('p_decision', p_d)]:
    print(f'\n{name} percentiles:')
    for q in [50, 75, 90, 95, 99, 100]:
        print(f'  p{q}: {np.percentile(p, q):.4f}')

for t in [0.50, 0.60, 0.70, 0.80, 0.90]:
    print(f'p_init >= {t}: {(p_i>=t).sum():,}   p_dec >= {t}: {(p_d>=t).sum():,}')
```

**Results (2026-06-04):**

| | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| `p_initiation` | 0.008 | 0.011 | 0.199 | 0.834 | 0.886 | **0.893** |
| `p_decision` | 0.008 | 0.011 | 0.140 | 0.911 | 0.924 | **0.927** |

| `classifier_label` | count |
|---|---:|
| neither | 236,692 |
| decision | 25,503 |
| initiation | 23,552 |

| threshold τ | `p_init ≥ τ` | `p_dec ≥ τ` |
|---|---:|---:|
| 0.50 | 23,552 | 25,503 |
| 0.60 | 22,027 | 24,710 |
| 0.70 | 19,964 | 23,701 |
| 0.80 | 16,556 | 22,155 |
| 0.90 | 0 | 17,023 |

The bimodal gap (p75=0.011 → p95=0.83) is the calibration problem in concrete form: scores are
either near-0 or near-0.85+, with almost nothing in between. Platt scaling will map the 0.85+
cluster to honest probabilities and make the operating curve meaningful.

Backup of pre-round-1 labels: `labeling_sample.pre_al1.csv`.

---

## IMMEDIATE NEXT STEPS — PHASE 2 (calibration)

All commands from repo root with the env active: `conda activate nepa`.

---

### Build `phase2/code/deliverable04/04b_calibrate.py`

Self-contained script (imports only stdlib + numpy/pandas/sklearn + 04 internals).
Three CLI modes: `--fit`, `--curve`, `--apply`.

#### Environment guard (same as `04`)

```python
import os
if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")
```

#### Imports and path constants

```python
import argparse, importlib.util, json, pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT   = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
```

#### Loading `04` internals via importlib

`04_classify_candidates.py` starts with a digit so it can't be imported with `import`.
Use `importlib.util` instead:

```python
def _load_04():
    spec = importlib.util.spec_from_file_location(
        "classify_candidates",
        Path(__file__).parent / "04_classify_candidates.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_04 = _load_04()
load_model           = _04.load_model
build_input_text     = _04.build_input_text
MODEL_DIR            = _04.MODEL_DIR
LABELING_SAMPLE_PATH = _04.LABELING_SAMPLE_PATH
CANDIDATES_PATH      = _04.CANDIDATES_PATH
OUTPUT_DIR           = _04.OUTPUT_DIR
TEST_SPLIT_VALUE     = _04.TEST_SPLIT_VALUE
LABEL_ORDER          = _04.LABEL_ORDER   # ["initiation", "decision"]
```

#### Calibrator paths

```python
CAL_INIT_PATH = MODEL_DIR / "calibrator_init.pkl"
CAL_DEC_PATH  = MODEL_DIR / "calibrator_dec.pkl"
CURVE_PATH    = OUTPUT_DIR / "calibration_curve.csv"

# Claude Haiku 4.5 pricing (input tokens only — context is the cost driver)
HAIKU_COST_PER_TOKEN = 0.80 / 1_000_000   # $0.80 per 1M input tokens
AVG_CONTEXT_TOKENS   = 500                 # rough per-candidate estimate
```

---

#### `--fit` mode — fit Platt calibrators on the 154-row frozen test

**Why Platt (LogisticRegression on 1 feature), not isotonic:** isotonic regression needs
many more points to avoid overfitting. With only 18 positives per head, Platt (2-parameter
sigmoid) is the right choice.

Steps:

1. Load `labeling_sample.csv`, keep rows where `split == test` → 154 rows.
2. Call `load_model(MODEL_DIR)` to get the current SetFit model.
3. Score the 154 rows:
   ```python
   texts  = [build_input_text(r) for _, r in df.iterrows()]
   y_prob = model.predict_proba(texts)   # shape (154, 2); col 0=init, col 1=dec
   ```
4. Build true binary labels per head:
   ```python
   y_true_init = (df["label"].str.strip() == "initiation").astype(int).to_numpy()
   y_true_dec  = (df["label"].str.strip() == "decision").astype(int).to_numpy()
   ```
5. For each head, fit a `LogisticRegression(C=1.0, solver='lbfgs')` with the raw prob as the
   single feature (`X = y_prob[:, i].reshape(-1, 1)`, `y = y_true_*`):
   ```python
   cal_init = LogisticRegression(C=1.0, solver='lbfgs').fit(y_prob[:,0].reshape(-1,1), y_true_init)
   cal_dec  = LogisticRegression(C=1.0, solver='lbfgs').fit(y_prob[:,1].reshape(-1,1), y_true_dec)
   ```
6. Save both calibrators:
   ```python
   pickle.dump(cal_init, open(CAL_INIT_PATH, "wb"))
   pickle.dump(cal_dec,  open(CAL_DEC_PATH,  "wb"))
   ```
7. Print a calibration-quality table: split the 154 test rows into 5 equal bins by raw
   `p_initiation` score; for each bin print `[raw_lo, raw_hi] | mean_raw | mean_calibrated | actual_positive_rate`.
   Do the same for `p_decision`. This shows whether calibration improved reliability.

---

#### `--curve` mode — operating curve (requires `--fit` first)

Steps:

1. Load both calibrators from pkl files.
2. Load `timeline_candidates.parquet`; keep only rows where
   `classifier_model_version == "20260604T060644Z"` (285,747 rows).
3. Apply calibrators to pool probabilities:
   ```python
   p_i_raw = pd.to_numeric(pool["p_initiation"], errors="coerce").fillna(0).values
   p_d_raw = pd.to_numeric(pool["p_decision"],   errors="coerce").fillna(0).values
   p_i_cal = cal_init.predict_proba(p_i_raw.reshape(-1,1))[:,1]
   p_d_cal = cal_dec.predict_proba( p_d_raw.reshape(-1,1))[:,1]
   p_max_cal = np.maximum(p_i_cal, p_d_cal)
   ```
4. Also score the frozen test rows with calibrated probs (re-use `y_prob` from fit or
   re-score) to compute per-threshold precision.
5. For `tau` in `np.concatenate([np.arange(0.10, 0.50, 0.10), np.arange(0.50, 0.96, 0.05)])`:
   - `n_auto   = int((p_max_cal >= tau).sum())`
   - `n_routed = int((p_max_cal <  tau).sum())`
   - `auto_pct = n_auto / len(pool) * 100`
   - `est_cost = n_routed * AVG_CONTEXT_TOKENS * HAIKU_COST_PER_TOKEN`
   - Precision on frozen test at this τ: among test rows where `max(p_cal) >= tau`, what
     fraction have the correct label? Compute separately for init head, dec head, combined.
     (`combined` = correct if the argmax head prediction matches the true label for any
     positive; skip neither rows from the denominator.)
6. Collect into a DataFrame and write to `CURVE_PATH` with columns:
   `tau, n_auto_resolved, auto_resolved_pct, n_routed, est_cost_usd,
    precision_init, precision_dec, precision_combined`.
7. Print a formatted table to stdout. Also print a one-line recommendation:
   the lowest τ where `precision_combined >= 0.85`.

---

#### `--apply` mode — write calibrated scores back to pool parquet

Steps:

1. Load both calibrators.
2. Load `timeline_candidates.parquet`.
3. For rows where `classifier_model_version == "20260604T060644Z"`:
   - Compute `p_init_cal` and `p_dec_cal` via calibrators (same as `--curve` step 3).
4. Write `p_init_cal` and `p_dec_cal` columns back to the parquet (add columns if absent).
5. Print: `"Applied calibrated scores to N candidates."`.

---

#### CLI wiring

```python
def main():
    parser = argparse.ArgumentParser(description="Calibrate D4 classifier + build operating curve.")
    parser.add_argument("--fit",   action="store_true", help="Fit Platt calibrators on frozen test.")
    parser.add_argument("--curve", action="store_true", help="Build operating curve (requires --fit first).")
    parser.add_argument("--apply", action="store_true", help="Write p_init_cal/p_dec_cal back to candidates parquet.")
    args = parser.parse_args()
    if args.fit:   run_fit()
    if args.curve: run_curve()
    if args.apply: run_apply()
    if not (args.fit or args.curve or args.apply):
        parser.print_help()

if __name__ == "__main__":
    main()
```

---

#### Run order

```bash
# from repo root, env active
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --fit
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --curve
# optional — only needed if 05/06 should consume calibrated probs directly:
CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --apply
```

---

### After the curve is built — re-tune routing thresholds (Phase 3 entry)

Read `output/deliverable04/calibration_curve.csv`. Pick the lowest τ where
`precision_combined >= 0.85` and `n_routed` is affordable (target ≤ 5,000 projects for the
LLM queue). That τ becomes the new `ROUTE_CONF_THRESHOLD` in `06_adjudicate_llm.py`.
Also bump `ROUTED_TOPK` from 3 → 5 at the same time (one-line change in `06`).

---


## Key commands

```bash
conda activate nepa            # required by every script

# Train / evaluate (sole source = labeling_sample.csv; frozen split)
python phase2/code/deliverable04/04_classify_candidates.py --train --backend setfit
python phase2/code/deliverable04/04_classify_candidates.py --eval

# Score the full candidate pool (writes p_initiation/p_decision/classifier_* back)
python phase2/code/deliverable04/04_classify_candidates.py [--process CE EA EIS]

# Active-learning: append the N most-uncertain UNLABELED candidates (split=train, blank label)
python phase2/code/deliverable04/04_classify_candidates.py --emit-batch 200
#   -> then label the blank rows (see labeling_rules.md), then --train

# LLM adjudication of the routed queue (after scoring)
python phase2/code/deliverable04/06_adjudicate_llm.py --mode candidate_adjudication --dry-run --sample 10
```

`--emit-batch` config (constants in `04`): `AL_CLEAR_INIT_N=60` (floor for `clear_initiation` role,
any uncertainty — replaces the old `AL_CE_INIT_FRACTION`), band `[0.35, 0.65]` for the remaining
uncertainty slice. `06` routing constants: `ROUTE_CONF_THRESHOLD=0.70`,
`COMPETE_DECISION_PROB=0.50`, `COMPETE_DECISION_MIN_N=2`, `ROUTED_TOPK=3`.

## Key files

| What | Path |
|---|---|
| Classifier (train/eval/score/emit-batch) | `phase2/code/deliverable04/04_classify_candidates.py` |
| Labels (sole source, frozen split) | `phase2/output/deliverable04/labeling_sample.csv` |
| Labeling codebook + split protocol | `phase2/notes/deliverable04/labeling_rules.md` |
| Model + meta (frozen-test metrics) | `phase2/data/analysis/timeline/models/candidate_classifier/` |
| Eval errors (active-learning fuel) | `phase2/output/deliverable04/classifier_eval_errors.csv` |
| Scored candidate pool | `phase2/data/analysis/timeline/timeline_candidates.parquet` |
| Date selection | `phase2/code/deliverable04/05_select_dates.py` |
| LLM adjudication / routing | `phase2/code/deliverable04/06_adjudicate_llm.py` |
| Project-level validation (KEEP) | `phase2/code/deliverable04/07_validate.py` |

---

## Decisions locked in (2026-06-03)

- **Gold removed from training.** `04` reads only `labeling_sample.csv`; the candidate-level gold
  fallback and gold-parquet eval were deleted. (Project-level gold for `07` is unaffected.)
- **Frozen test set.** `split` column, stratified by process×label, seed 42, assigned once. New
  labels default to `train`. `--train` validates on it; `--eval` scores it.
- **Proxy / Date-Determined rows stay in training** — regex `proxy_decision` is ~97% truly
  `neither`, `proxy_initiation` ~73% `neither`; correcting that noise is the classifier's core
  job. The CE Date-Determined → proxy-initiation *pairing* stays owned deterministically by `05`.
- **`06` routing consumes classifier confidence** (low best-confidence or competing decisions →
  route; top-k packets ranked by classifier score; `p_init`/`p_dec` surfaced to Claude).
- **Hot paths vectorized** (`compute_eligible_mask`, `build_input_texts`) — no row-wise
  apply/iterrows on the 414k pool. Future runs: launch unbuffered + progress bar.

## Open caveats

- **Probabilities uncalibrated** (bimodal: ~90% near 0, then jump to 0.85+ at p95). Phase 2 is
  the fix — do not trust any threshold until Platt calibration is fitted and the operating curve
  is built. Raw `max(p_init)=0.893`, `max(p_dec)=0.927` as of model `20260604T060644Z`.
- **`ROUTED_TOPK=3`** likely too low at current F1 — raise to 5 once calibrated thresholds
  are set (one-line knob in `06`).
- **Init recall ceiling**: 6 persistent FNs on the frozen test; 3 are hard/ambiguous cases,
  1 is an exempt role that never scores in production. Accept this as the SetFit floor.
- Two distinct "gold" concepts: candidate-level (retired) vs project-level `07_validate` (kept).
