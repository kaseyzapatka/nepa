"""
Classify timeline date candidates with a trained two-head model (D4).

This is the learned scorer that sits between candidate extraction (03) and date
selection (05). Regex prelabels in 03 are noisy in the ambiguous middle band;
this script reads the context + date and produces two independent probabilities
per candidate:

    p_initiation = P(this date is an initiation date)
    p_decision   = P(this date is a decision date)

Selection (05) and the LLM adjudicator (06) consume these scores to rank
candidates and pick the final initiation/decision pair.

----------------------------------------------------------------------------
Design (see also: phase2/architecture/deliverables/deliverable04.md)
----------------------------------------------------------------------------
Two-head, not 7-way. Binary heads are far more reliable than a single multiclass
role classifier, and their outputs ARE the ranking we need. A candidate can be
high-decision / low-initiation independently. Implemented as ONE shared-encoder
model with a multi-label ("one-vs-rest") head — the encoder is shared, the two
heads are independent.

One model, not three. CE/EA/EIS share an encoder; the process type is injected
as a leading token ("[CE] ...") so the model can specialise without splitting the
(scarce) labels three ways. Revisit per-process models only if error analysis
shows cross-process interference AND labels become plentiful.

Eligible pool (what the model reads). Only the ambiguous middle:
    - candidate_role in {clear_initiation, clear_decision, proxy_initiation,
      proxy_decision, body_text, unknown}  AND  role_confidence_score < 5.0
Exempt (never scored — regex is authoritative or the role is out of scope):
    - role_confidence_score == 5.0  (register Tier A + strong text cues)
    - candidate_role in {review, historical, reject}

----------------------------------------------------------------------------
SetFit now, BERT later — how and when to switch
----------------------------------------------------------------------------
The model lives behind a backend abstraction (TimelineClassifier). Today the
default backend is SetFit (sentence-transformer encoder + contrastive
fine-tune + light head): it reaches usable accuracy with ~50-100 labels/class,
which is the regime we are in (no human gold yet). Same encoder family as D3.

Switch to a fully fine-tuned encoder (DeBERTa-v3 / RoBERTa sequence classifier
with two heads) when ALL of these hold:
    1. >= ~1,000-1,500 labels per head (init / decision) after active learning,
    2. SetFit eval F1 has plateaued across the last two label expansions, and
    3. per-process error analysis shows the shared model is underfitting a
       structural pattern more labels would fix.
At that point implement TransformerBackend (stub below) — the train/predict/
save/load contract is identical, so 05/06 and this CLI do not change.

----------------------------------------------------------------------------
Modes
----------------------------------------------------------------------------
    --train     fit the model on timeline_gold_candidate_training.parquet,
                save to the model dir with version metadata.
    --eval      score the held-out test split, report per-head P/R/F1.
    (default)   score eligible candidates in the run dir's candidates parquet,
                write p_initiation / p_decision / classifier_* columns back.
                If no trained model exists, pass through with a warning (writes
                neutral scores) so the orchestrated pipeline never breaks.

Usage:
    python 04_classify_candidates.py --train [--backend setfit]
    python 04_classify_candidates.py --eval
    python 04_classify_candidates.py [--process CE EA EIS] [--sample-ids path] [--append]
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
GOLD_DIR = TIMELINE_DIR / "gold"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"

CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
# Primary training source: the simple human-labeled sample emitted by 03 (label column:
# initiation | decision | neither). Falls back to the formal gold training table.
LABELING_SAMPLE_PATH = OUTPUT_DIR / "labeling_sample.csv"
TRAINING_PATH = GOLD_DIR / "timeline_gold_candidate_training.parquet"
GOLD_CANDIDATES_PATH = GOLD_DIR / "timeline_gold_candidates.parquet"  # has 'split'
MODEL_DIR = TIMELINE_DIR / "models" / "candidate_classifier"

DEFAULT_BACKEND = "setfit"
DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SPLIT = "test_representative_v1"

# Two heads, fixed order: column 0 = initiation, column 1 = decision.
LABEL_ORDER = ["initiation", "decision"]

INITIATION_ROLES = {"clear_initiation", "proxy_initiation"}
DECISION_ROLES = {"clear_decision", "proxy_decision"}
# historical is ELIGIBLE (not exempt): an audit (2026-06-02) showed regex "historical" is
# only ~16% reliably historical — it sweeps in real Field-Manager/NEPA-officer signature
# dates. Scoring it lets the classifier rescue those misfiles (genuine historicals score low
# on both heads and stay out; misfiled signatures score high P_decision and get recovered).
ELIGIBLE_ROLES = INITIATION_ROLES | DECISION_ROLES | {"body_text", "unknown", "historical"}
EXEMPT_ROLES = {"review", "reject"}

# Decision threshold for the discrete classifier_label (ranking uses raw probs).
LABEL_THRESHOLD = 0.5

# Fraction of the labeled sample held out by --train for self-contained validation
# (no separate gold test set required).
HOLDOUT_FRACTION = 0.2


# ---------------------------------------------------------------------------
# Input construction + label derivation
# ---------------------------------------------------------------------------
def build_input_text(row: pd.Series) -> str:
    """
    Prepend a process token so one shared model can specialise per process, and use the
    anchored `model_context` (target date wrapped in [[ ]]) so the model knows WHICH date
    it is scoring. Falls back to context_text if model_context is absent (older parquets).
    """
    proc = str(row.get("process_type") or "NA").strip().upper()
    heading = str(row.get("heading_title") or "").strip()
    context = str(row.get("model_context") or row.get("context_text") or row.get("context_cleaned") or "").strip()
    head = f"[{proc}]"
    if heading:
        return f"{head} {heading} :: {context}"
    return f"{head} {context}"


def roles_to_labels(roles: pd.Series) -> np.ndarray:
    """Multi-hot [is_initiation, is_decision] from a gold role column."""
    r = roles.fillna("").astype(str).str.strip()
    init = r.isin(INITIATION_ROLES).astype(int).to_numpy()
    dec = r.isin(DECISION_ROLES).astype(int).to_numpy()
    return np.stack([init, dec], axis=1)


def is_eligible(role: object, conf_score: object) -> bool:
    """Eligible = ambiguous middle band the model should adjudicate."""
    role = str(role or "").strip()
    if role in EXEMPT_ROLES:
        return False
    if role not in ELIGIBLE_ROLES:
        return False
    try:
        if float(conf_score) >= 5.0:
            return False  # regex authoritative (register Tier A / strong cue)
    except (TypeError, ValueError):
        pass
    return True


# ---------------------------------------------------------------------------
# Backend abstraction — SetFit today, fine-tuned transformer later
# ---------------------------------------------------------------------------
class TimelineClassifier:
    """Contract every backend must satisfy. 05/06 depend only on this."""

    def train(self, texts: list[str], labels: np.ndarray) -> None: ...
    def predict_proba(self, texts: list[str]) -> np.ndarray:  # (N, 2)
        ...
    def save(self, path: Path) -> None: ...
    @classmethod
    def load(cls, path: Path) -> "TimelineClassifier": ...


class SetFitBackend(TimelineClassifier):
    """Sentence-transformer + contrastive fine-tune + multi-label head."""

    def __init__(self, model=None, base_model: str = DEFAULT_BASE_MODEL):
        self._model = model
        self.base_model = base_model

    def train(self, texts: list[str], labels: np.ndarray) -> None:
        try:
            from setfit import SetFitModel, Trainer, TrainingArguments
            from datasets import Dataset
        except ImportError as e:
            raise SystemExit(
                "SetFit not installed in the 'nepa' env. Install with:\n"
                "    pip install setfit datasets\n"
                f"(import error: {e})"
            )
        self._model = SetFitModel.from_pretrained(
            self.base_model, multi_target_strategy="one-vs-rest"
        )
        train_ds = Dataset.from_dict(
            {"text": list(texts), "label": [list(map(int, row)) for row in labels]}
        )
        args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20)
        trainer = Trainer(model=self._model, args=args, train_dataset=train_ds)
        trainer.train()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SetFit model not loaded.")
        probs = self._model.predict_proba(list(texts))
        return _to_two_probs(probs)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(path))

    @classmethod
    def load(cls, path: Path) -> "SetFitBackend":
        from setfit import SetFitModel
        return cls(model=SetFitModel.from_pretrained(str(path)))


class TransformerBackend(TimelineClassifier):
    """
    Fine-tuned encoder (DeBERTa-v3 / RoBERTa) with a 2-logit multi-label head.

    Not implemented yet — activate per the "SetFit now, BERT later" criteria in
    the module docstring. Implementation sketch:
        - AutoModelForSequenceClassification(num_labels=2,
              problem_type="multi_label_classification")
        - BCEWithLogits loss; sigmoid at inference -> (N, 2) probabilities
        - same save/load via save_pretrained; same input_text() construction
    The CLI, 05, and 06 do not change when this replaces SetFitBackend.
    """

    def train(self, texts, labels):
        raise NotImplementedError(
            "TransformerBackend is a stub. Use --backend setfit until the "
            "label count / plateau criteria in the module docstring are met."
        )

    def predict_proba(self, texts):
        raise NotImplementedError("TransformerBackend is a stub.")

    def save(self, path):
        raise NotImplementedError

    @classmethod
    def load(cls, path):
        raise NotImplementedError


def _to_two_probs(probs) -> np.ndarray:
    """Normalise a backend's probability output to a clean (N, 2) array."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.shape[1] == 2:
        return arr
    # one-vs-rest may emit per-label (N, n_labels) already; otherwise pad/trim
    if arr.shape[1] == 1:
        return np.hstack([arr, arr])
    return arr[:, :2]


def make_backend(name: str) -> TimelineClassifier:
    if name == "setfit":
        return SetFitBackend()
    if name == "transformer":
        return TransformerBackend()
    raise SystemExit(f"Unknown backend: {name!r} (use 'setfit' or 'transformer').")


def load_model(model_dir: Path) -> tuple[TimelineClassifier | None, dict]:
    meta_path = model_dir / "classifier_meta.json"
    if not meta_path.exists():
        return None, {}
    meta = json.loads(meta_path.read_text())
    backend = meta.get("backend", DEFAULT_BACKEND)
    if backend == "setfit":
        return SetFitBackend.load(model_dir), meta
    if backend == "transformer":
        return TransformerBackend.load(model_dir), meta
    return None, meta


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _labels_from_label_col(series: pd.Series) -> np.ndarray:
    """Map the simple `label` column (initiation|decision|neither) to two-head targets."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    valid = {"initiation", "decision", "neither"}
    bad = sorted(set(s) - valid)
    if bad:
        print(f"WARNING: ignoring unrecognized label values {bad} "
              "(use exactly: initiation | decision | neither).")
    init = s.eq("initiation").astype(int).to_numpy()
    dec = s.eq("decision").astype(int).to_numpy()
    return np.stack([init, dec], axis=1)


def run_train(backend_name: str, model_dir: Path) -> None:
    # Primary path: the simple human-labeled sample from 03 (label column).
    if LABELING_SAMPLE_PATH.exists():
        df = pd.read_csv(LABELING_SAMPLE_PATH)
        if "label" not in df.columns:
            raise SystemExit(f"{LABELING_SAMPLE_PATH} has no 'label' column.")
        df = df[df["label"].fillna("").astype(str).str.strip().ne("")]
        if df.empty:
            raise SystemExit(
                f"{LABELING_SAMPLE_PATH} has no filled labels yet.\n"
                "Fill the 'label' column with initiation | decision | neither, then re-run."
            )
        labels = _labels_from_label_col(df["label"])
        src = LABELING_SAMPLE_PATH.name
    elif TRAINING_PATH.exists():
        df = pd.read_parquet(TRAINING_PATH)
        df = df[df["gold_candidate_role"].fillna("").astype(str).str.strip().ne("")]
        if df.empty:
            raise SystemExit("Training table has no labeled rows (gold_candidate_role empty).")
        labels = roles_to_labels(df["gold_candidate_role"])
        src = TRAINING_PATH.name
    else:
        raise SystemExit(
            f"No labels found. Fill {LABELING_SAMPLE_PATH} (label column: "
            "initiation | decision | neither), or build the formal gold training table."
        )

    texts = [build_input_text(r) for _, r in df.iterrows()]
    n = len(texts)
    n_neither = int((labels.sum(axis=1) == 0).sum())
    print(f"Loaded {n} labeled rows from {src} "
          f"(init={int(labels[:,0].sum())}, decision={int(labels[:,1].sum())}, neither={n_neither}).")

    # Stratified hold-out so --train also validates without a separate gold test set.
    cls = labels[:, 0] * 1 + labels[:, 1] * 2  # 0=neither, 1=initiation, 2=decision
    try:
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        tr_idx, te_idx = train_test_split(
            idx, test_size=HOLDOUT_FRACTION, random_state=42, stratify=cls
        )
    except Exception as e:
        print(f"WARNING: stratified split failed ({e}); training on all rows, no hold-out eval.")
        tr_idx, te_idx = np.arange(n), np.array([], dtype=int)

    print(f"Training {backend_name} on {len(tr_idx)} rows; holding out {len(te_idx)} for validation.")
    backend = make_backend(backend_name)
    backend.train([texts[i] for i in tr_idx], labels[tr_idx])

    metrics: dict = {}
    if len(te_idx):
        te_prob = backend.predict_proba([texts[i] for i in te_idx])
        te_true = labels[te_idx]
        te_pred = (te_prob >= LABEL_THRESHOLD).astype(int)
        print(f"\nHold-out validation ({len(te_idx)} rows):")
        for i, head in enumerate(LABEL_ORDER):
            tp = int(((te_pred[:, i] == 1) & (te_true[:, i] == 1)).sum())
            fp = int(((te_pred[:, i] == 1) & (te_true[:, i] == 0)).sum())
            fn = int(((te_pred[:, i] == 0) & (te_true[:, i] == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            print(f"  {head:11s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")
            metrics[head] = {"precision": round(prec, 3), "recall": round(rec, 3),
                             "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    backend.save(model_dir)
    meta = {
        "backend": backend_name,
        "base_model": getattr(backend, "base_model", None),
        "label_order": LABEL_ORDER,
        "n_labeled": n,
        "n_train": int(len(tr_idx)),
        "n_holdout": int(len(te_idx)),
        "n_init_pos": int(labels[:, 0].sum()),
        "n_decision_pos": int(labels[:, 1].sum()),
        "holdout_metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    (model_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + meta to {model_dir} (version {meta['model_version']}).")


def run_eval(model_dir: Path) -> None:
    model, meta = load_model(model_dir)
    if model is None:
        raise SystemExit(f"No trained model at {model_dir}. Run --train first.")
    if not GOLD_CANDIDATES_PATH.exists():
        raise SystemExit(f"No gold candidates table at {GOLD_CANDIDATES_PATH}.")
    df = pd.read_parquet(GOLD_CANDIDATES_PATH)
    df = df[(df.get("split") == TEST_SPLIT) &
            df["gold_candidate_role"].fillna("").astype(str).str.strip().ne("")]
    if df.empty:
        raise SystemExit(f"No labeled rows in test split {TEST_SPLIT}.")

    texts = [build_input_text(r) for _, r in df.iterrows()]
    y_true = roles_to_labels(df["gold_candidate_role"])
    y_prob = model.predict_proba(texts)
    y_pred = (y_prob >= LABEL_THRESHOLD).astype(int)

    print(f"Eval on {len(texts)} test candidates (model {meta.get('model_version')}):")
    for i, head in enumerate(LABEL_ORDER):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {head:11s}  P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}  "
              f"(tp={tp} fp={fp} fn={fn})")


def run_score(args: argparse.Namespace, model_dir: Path) -> None:
    # Run-dir resolution mirrors scripts 03 and 05.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.sample_ids:
        run_dir = TIMELINE_DIR / "sample_runs" / Path(args.sample_ids).stem
    else:
        run_dir = TIMELINE_DIR
    candidates_path = run_dir / "timeline_candidates.parquet"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Candidates not found: {candidates_path}\nRun 03_extract_candidates.py first."
        )

    df = pd.read_parquet(candidates_path)
    if args.process:
        df = df[df["process_type"].isin(args.process)]
    if args.sample_ids:
        with open(args.sample_ids) as f:
            ids = {ln.strip() for ln in f if ln.strip()}
        df = df[df["project_id"].isin(ids)]

    run_at = datetime.now(timezone.utc).isoformat()

    # Ensure columns exist.
    for col, default in [
        ("p_initiation", 0.0), ("p_decision", 0.0),
        ("classifier_label", ""), ("classifier_score", 0.0),
        ("classifier_backend", ""), ("classifier_model_version", ""),
        ("classifier_run_at", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    eligible_mask = df.apply(
        lambda r: is_eligible(r.get("candidate_role"), r.get("role_confidence_score")), axis=1
    )
    n_elig = int(eligible_mask.sum())
    print(f"Scoring {n_elig}/{len(df)} eligible candidates from {candidates_path}.")

    model, meta = load_model(model_dir)
    if model is None or n_elig == 0:
        if model is None:
            print("WARNING: no trained model found — passing through with neutral "
                  "scores. Train with --train once real gold labels exist.")
        df.loc[eligible_mask, "classifier_label"] = "unscored"
        df["classifier_run_at"] = run_at
        _write_back(df, run_dir, candidates_path, args.append)
        return

    sub = df[eligible_mask]
    texts = [build_input_text(r) for _, r in sub.iterrows()]
    probs = model.predict_proba(texts)
    p_init, p_dec = probs[:, 0], probs[:, 1]

    df.loc[eligible_mask, "p_initiation"] = p_init
    df.loc[eligible_mask, "p_decision"] = p_dec
    df.loc[eligible_mask, "classifier_score"] = np.maximum(p_init, p_dec)
    labels = np.where(
        (p_init < LABEL_THRESHOLD) & (p_dec < LABEL_THRESHOLD), "neither",
        np.where(p_init >= p_dec, "initiation", "decision"),
    )
    df.loc[eligible_mask, "classifier_label"] = labels
    df.loc[eligible_mask, "classifier_backend"] = meta.get("backend", "")
    df.loc[eligible_mask, "classifier_model_version"] = meta.get("model_version", "")
    df["classifier_run_at"] = run_at

    _write_back(df, run_dir, candidates_path, args.append)
    print(f"Wrote classifier scores for {n_elig} candidates "
          f"(model {meta.get('model_version')}).")


def _write_back(df: pd.DataFrame, run_dir: Path, candidates_path: Path, append: bool) -> None:
    """Merge scored rows back into the candidates parquet by candidate_id."""
    if append and candidates_path.exists():
        full = pd.read_parquet(candidates_path)
        scored = df.set_index("candidate_id")
        score_cols = [
            "p_initiation", "p_decision", "classifier_label", "classifier_score",
            "classifier_backend", "classifier_model_version", "classifier_run_at",
        ]
        for col in score_cols:
            if col not in full.columns:
                full[col] = 0.0 if col.startswith("p_") or col == "classifier_score" else ""
        full = full.set_index("candidate_id")
        full.update(scored[score_cols])
        full.reset_index().to_parquet(candidates_path, index=False)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(candidates_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify timeline date candidates (D4).")
    parser.add_argument("--process", nargs="+", default=["CE", "EA", "EIS"],
                        choices=["CE", "EA", "EIS"])
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--append", action="store_true",
                        help="Merge scores into existing candidates parquet by candidate_id.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-dir", help="Override run directory (read/write candidates here).")
    parser.add_argument("--train", action="store_true", help="Train the model from gold labels.")
    parser.add_argument("--eval", action="store_true", help="Evaluate on the held-out test split.")
    parser.add_argument("--backend", default=DEFAULT_BACKEND, choices=["setfit", "transformer"])
    parser.add_argument("--model-dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    if args.train:
        run_train(args.backend, model_dir)
        return
    if args.eval:
        run_eval(model_dir)
        return
    run_score(args, model_dir)


if __name__ == "__main__":
    main()
