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
    --train     fit the model on the `split == "train"` rows of labeling_sample.csv,
                save to the model dir with version metadata, and report metrics on
                the frozen `split == "test"` rows.
    --eval      score the frozen `split == "test"` rows of labeling_sample.csv,
                report per-head P/R/F1 (same set --train validates on).
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
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"

CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
# Single source of truth for labels: the human-labeled sample (03 emits it; humans fill the
# `label` column: initiation | decision | neither). The `split` column (train|test) is FROZEN —
# assigned once, stratified by process x label (see labeling_rules.md). New labels added later
# default to `train`, so the test set never grows or leaks. The former gold/ candidate-level
# training+test apparatus is retired (it held regex echoes, not human gold). NOTE: this is
# distinct from the project-level gold used by 07_validate.py for end-to-end validation.
LABELING_SAMPLE_PATH = OUTPUT_DIR / "labeling_sample.csv"
MODEL_DIR = TIMELINE_DIR / "models" / "candidate_classifier"
# SetFit writes a training checkpoint every save_steps (~260MB each). Pin it to a fixed, gitignored
# path under models/ (covered by .gitignore `*models/`) so checkpoints never scatter into the CWD
# (`./checkpoints/`) at whatever directory training is launched from, and never enter git.
CHECKPOINT_DIR = TIMELINE_DIR / "models" / "_setfit_checkpoints"
EVAL_ERRORS_PATH = OUTPUT_DIR / "classifier_eval_errors.csv"  # misclassified test rows (--eval)

DEFAULT_BACKEND = "setfit"
DEFAULT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEST_SPLIT_VALUE = "test"  # value in the labeling_sample.csv `split` column

# Three heads, fixed order: 0 = initiation, 1 = decision (ROD for EIS), 2 = final_eis (EIS Final-EIS
# publication / Notice of Availability). Independent binary one-vs-rest heads — the "binary heads,
# not 7-way multiclass" design, extended by one head. Existing initiation/decision/neither rows are
# automatically final_eis-NEGATIVES, so adding the head needs NO relabeling of existing data.
LABEL_ORDER = ["initiation", "decision", "final_eis"]

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


def compute_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """Vectorized is_eligible over a whole frame — equivalent to is_eligible row-by-row but
    O(1) passes instead of a 414k-row Python apply. Use this on large pools (scoring, batch
    selection); reserve scalar is_eligible for single-row checks."""
    role = df["candidate_role"].fillna("").astype(str).str.strip()
    conf = pd.to_numeric(df["role_confidence_score"], errors="coerce").fillna(0.0)  # unparseable -> eligible
    return role.isin(ELIGIBLE_ROLES) & (conf < 5.0)


def _col_or_na(frame: pd.DataFrame, name: str) -> pd.Series:
    """A stripped string column with ''/missing as <NA> (so fillna chains skip empties)."""
    if name not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    s = frame[name].astype("string").str.strip()
    return s.mask(s.eq(""), pd.NA)


def build_input_texts(frame: pd.DataFrame) -> list[str]:
    """Vectorized equivalent of build_input_text over a frame (fast for large pools)."""
    proc = frame["process_type"].fillna("NA").astype(str).str.strip().str.upper()
    head = _col_or_na(frame, "heading_title").fillna("")
    ctx = (_col_or_na(frame, "model_context")
           .fillna(_col_or_na(frame, "context_text"))
           .fillna(_col_or_na(frame, "context_cleaned"))
           .fillna(""))
    with_head = "[" + proc + "] " + head + " :: " + ctx
    no_head = "[" + proc + "] " + ctx
    return with_head.where(head.ne(""), no_head).tolist()


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
        # num_iterations=12: embedding_loss plateaued by ~iter 4 in the 20-iter run (see eis_audit
        # progress notes); 12 keeps a safety margin while ~halving CPU wall-clock on the Intel box.
        # output_dir pins checkpoints to a fixed gitignored path (not the launch-CWD's ./checkpoints).
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=12,
                                 output_dir=str(CHECKPOINT_DIR))
        trainer = Trainer(model=self._model, args=args, train_dataset=train_ds)
        trainer.train()

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("SetFit model not loaded.")
        probs = self._model.predict_proba(list(texts))
        return _to_label_probs(probs)

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


def _to_label_probs(probs) -> np.ndarray:
    """Normalise a backend's probability output to (N, k) where k = number of heads the loaded
    model actually has. Does NOT pad to len(LABEL_ORDER): a legacy 2-head model returns (N, 2) and
    run_score guards the optional 3rd (final_eis) column accordingly — so a 3-head LABEL_ORDER stays
    backward-compatible with a model trained before the head was added."""
    arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


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
    """Map the `label` column (initiation|decision|final_eis|neither) to multi-head targets,
    one column per LABEL_ORDER head. `neither` is all-zeros."""
    s = series.fillna("").astype(str).str.strip().str.lower()
    valid = set(LABEL_ORDER) | {"neither"}
    bad = sorted(set(s) - valid)
    if bad:
        print(f"WARNING: ignoring unrecognized label values {bad} "
              f"(use exactly: {' | '.join(LABEL_ORDER)} | neither).")
    cols = [s.eq(name).astype(int).to_numpy() for name in LABEL_ORDER]
    return np.stack(cols, axis=1)


def _load_labeled_sample() -> pd.DataFrame:
    """Load the human-labeled sample (sole label source); require `label` + `split`."""
    if not LABELING_SAMPLE_PATH.exists():
        raise SystemExit(
            f"No labels found: {LABELING_SAMPLE_PATH} is missing.\n"
            "Run 03_extract_candidates.py to emit it, fill the `label` column "
            "(initiation | decision | neither), then re-run."
        )
    df = pd.read_csv(LABELING_SAMPLE_PATH)
    if "label" not in df.columns:
        raise SystemExit(f"{LABELING_SAMPLE_PATH} has no 'label' column.")
    df = df[df["label"].fillna("").astype(str).str.strip().ne("")].copy()
    if df.empty:
        raise SystemExit(
            f"{LABELING_SAMPLE_PATH} has no filled labels yet.\n"
            "Fill the `label` column with initiation | decision | neither, then re-run."
        )
    if "split" not in df.columns:
        raise SystemExit(
            f"{LABELING_SAMPLE_PATH} has no `split` column. Assign a FROZEN train/test split "
            "first (stratified by process x label; see labeling_rules.md). New rows added "
            "later should default to `train` so the test set never grows or leaks."
        )
    # Rows added after the freeze (blank split) default to train — never to the frozen test set.
    df["split"] = df["split"].fillna("train").astype(str).str.strip().replace("", "train")
    return df


def _head_metrics(y_true: np.ndarray, y_prob: np.ndarray, title: str) -> dict:
    """Print per-head P/R/F1 at LABEL_THRESHOLD and return them as a dict."""
    y_pred = (y_prob >= LABEL_THRESHOLD).astype(int)
    print(f"\n{title} ({len(y_true)} rows):")
    metrics: dict = {}
    for i, head in enumerate(LABEL_ORDER):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"  {head:11s} P={prec:.3f} R={rec:.3f} F1={f1:.3f} (tp={tp} fp={fp} fn={fn})")
        metrics[head] = {"precision": round(prec, 3), "recall": round(rec, 3),
                         "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}
    return metrics


def _f1_per_head(y_true: np.ndarray, y_prob: np.ndarray) -> list[float]:
    """Per-head F1 at LABEL_THRESHOLD, no printing (for sliced breakdowns)."""
    y_pred = (y_prob >= LABEL_THRESHOLD).astype(int)
    out = []
    for i in range(y_true.shape[1]):
        tp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((y_pred[:, i] == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((y_pred[:, i] == 0) & (y_true[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return out


def _predicted_label(p_init: float, p_dec: float) -> str:
    """Single-label view, identical to the rule run_score / 05 use for classifier_label."""
    if p_init < LABEL_THRESHOLD and p_dec < LABEL_THRESHOLD:
        return "neither"
    return "initiation" if p_init >= p_dec else "decision"


def _error_report(df: pd.DataFrame, y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Deep frozen-test diagnostics: 3-class confusion, per-process and per-regex-role
    breakdowns, and a CSV of misclassified rows to drive the next labeling round."""
    df = df.reset_index(drop=True)
    p_init, p_dec = y_prob[:, 0], y_prob[:, 1]
    pred = pd.Series([_predicted_label(a, b) for a, b in zip(p_init, p_dec)], name="pred")
    true = df["label"].reset_index(drop=True).rename("true")

    print("\n3-class confusion (rows=true, cols=pred):")
    print(pd.crosstab(true, pred, dropna=False).to_string())

    # CE-initiation is the known weak spot — confirm per process quantitatively.
    print("\nPer-process head F1 (n_pos in parens):")
    for proc in sorted(df["process_type"].dropna().unique()):
        m = (df["process_type"] == proc).to_numpy()
        f1 = _f1_per_head(y_true[m], y_prob[m])
        print(f"  {proc:4s} init={f1[0]:.3f} (n={int(y_true[m,0].sum())})  "
              f"decision={f1[1]:.3f} (n={int(y_true[m,1].sum())})")

    # Did the model correct the noisy regex roles? (proxy_decision is ~97% truly neither.)
    print("\nPredicted label by regex candidate_role:")
    print(pd.crosstab(df["candidate_role"], pred).to_string())

    mis = df.copy()
    mis["pred"], mis["p_initiation"], mis["p_decision"] = pred, p_init.round(3), p_dec.round(3)
    mis = mis[mis["label"] != mis["pred"]]
    cols = [c for c in ["candidate_id", "process_type", "candidate_role", "label", "pred",
                        "p_initiation", "p_decision", "raw_date_text", "model_context", "notes"]
            if c in mis.columns]
    mis[cols].to_csv(EVAL_ERRORS_PATH, index=False)
    print(f"\n{len(mis)}/{len(df)} misclassified -> {EVAL_ERRORS_PATH}")


def run_train(backend_name: str, model_dir: Path) -> None:
    df = _load_labeled_sample()
    is_test = df["split"].eq(TEST_SPLIT_VALUE)
    tr_df, te_df = df[~is_test], df[is_test]
    n_tr, n_te = len(tr_df), len(te_df)

    tr_texts = [build_input_text(r) for _, r in tr_df.iterrows()]
    tr_labels = _labels_from_label_col(tr_df["label"])
    n_neither = int((tr_labels.sum(axis=1) == 0).sum())
    print(f"Loaded {len(df)} labeled rows from {LABELING_SAMPLE_PATH.name} "
          f"(train={n_tr}, frozen test={n_te}).")
    print(f"Training {backend_name} on {n_tr} train rows "
          f"(init={int(tr_labels[:,0].sum())}, decision={int(tr_labels[:,1].sum())}, "
          f"neither={n_neither}).")
    backend = make_backend(backend_name)
    backend.train(tr_texts, tr_labels)

    metrics: dict = {}
    if n_te:
        te_texts = [build_input_text(r) for _, r in te_df.iterrows()]
        te_true = _labels_from_label_col(te_df["label"])
        te_prob = backend.predict_proba(te_texts)
        metrics = _head_metrics(te_true, te_prob, "Frozen-test validation")

    backend.save(model_dir)
    meta = {
        "backend": backend_name,
        "base_model": getattr(backend, "base_model", None),
        "label_order": LABEL_ORDER,
        "n_labeled": int(len(df)),
        "n_train": n_tr,
        "n_test": n_te,
        "n_init_pos_train": int(tr_labels[:, 0].sum()),
        "n_decision_pos_train": int(tr_labels[:, 1].sum()),
        "test_metrics": metrics,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
    }
    (model_dir / "classifier_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved model + meta to {model_dir} (version {meta['model_version']}).")

    # Refresh diagnostics owned by the train step (01 inventory, 02 progression, and the
    # frozen-test 03/04 when a test split exists). Best-effort: never fail a train on a
    # diagnostics hiccup.
    try:
        import _diagnostics as diag
        diag.write_label_inventory(df)
        diag.update_metrics_by_round(meta)
        if n_te:
            diag.write_confusion(te_df, te_prob)
            diag.write_per_process(te_df, te_prob, te_true)
        print(f"  diagnostics updated -> {diag.DIAG_DIR}")
    except Exception as e:
        print(f"  (diagnostics skipped: {e})")


def run_eval(model_dir: Path) -> None:
    model, meta = load_model(model_dir)
    if model is None:
        raise SystemExit(f"No trained model at {model_dir}. Run --train first.")
    df = _load_labeled_sample()
    df = df[df["split"].eq(TEST_SPLIT_VALUE)]
    if df.empty:
        raise SystemExit(
            f"No rows with split == '{TEST_SPLIT_VALUE}' in {LABELING_SAMPLE_PATH.name}."
        )
    texts = [build_input_text(r) for _, r in df.iterrows()]
    y_true = _labels_from_label_col(df["label"])
    y_prob = model.predict_proba(texts)
    _head_metrics(y_true, y_prob, f"Frozen-test eval (model {meta.get('model_version')})")
    _error_report(df, y_true, y_prob)

    # Refresh the frozen-test confusion + per-process diagnostics (03, 04).
    try:
        import _diagnostics as diag
        diag.write_confusion(df, y_prob)
        diag.write_per_process(df, y_prob, y_true)
        print(f"  diagnostics updated -> {diag.DIAG_DIR}")
    except Exception as e:
        print(f"  (diagnostics skipped: {e})")


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
        ("p_initiation", 0.0), ("p_decision", 0.0), ("p_final_eis", 0.0),
        ("classifier_label", ""), ("classifier_score", 0.0),
        ("classifier_backend", ""), ("classifier_model_version", ""),
        ("classifier_run_at", ""),
    ]:
        if col not in df.columns:
            df[col] = default

    eligible_mask = compute_eligible_mask(df)
    n_elig = int(eligible_mask.sum())
    print(f"Scoring {n_elig}/{len(df)} eligible candidates from {candidates_path}.")

    model, meta = load_model(model_dir)
    if model is None or n_elig == 0:
        if model is None:
            print("WARNING: no trained model found — passing through with neutral "
                  "scores. Run --train (fits on labeling_sample.csv) to enable scoring.")
        df.loc[eligible_mask, "classifier_label"] = "unscored"
        df["classifier_run_at"] = run_at
        _write_back(df, run_dir, candidates_path, args.append)
        return

    sub = df[eligible_mask]
    texts = build_input_texts(sub)
    probs = model.predict_proba(texts)
    p_init, p_dec = probs[:, 0], probs[:, 1]
    # final_eis is the 3rd head; guard for legacy 2-head models so scoring stays backward-compatible.
    p_feis = probs[:, 2] if probs.shape[1] > 2 else np.zeros_like(p_init)

    # Document-type gate: a final-EIS publication date definitionally comes from a final EIS
    # document, so zero p_final_eis for non-FEIS-typed candidates. On the frozen test this lifts
    # final_eis precision 0.50->0.74 (drops 18 non-FEIS false positives) at a 0.977 recall ceiling
    # (only 1/44 true positives live outside an FEIS doc). Every downstream consumer (argmax label,
    # ranker, 04b --apply, 05 selection, 06 routing) inherits the gate from this single rule.
    is_feis = (sub["document_type_clean"].astype(str).str.upper().str.strip() == "FEIS").to_numpy()
    n_gated = int((~is_feis & (p_feis > 0)).sum())
    p_feis = np.where(is_feis, p_feis, 0.0)
    print(f"Doc-type gate: p_final_eis zeroed on {int((~is_feis).sum())} non-FEIS candidates "
          f"({n_gated} had p_final_eis>0).")

    df.loc[eligible_mask, "p_initiation"] = p_init
    df.loc[eligible_mask, "p_decision"] = p_dec
    df.loc[eligible_mask, "p_final_eis"] = p_feis
    stack = np.vstack([p_init, p_dec, p_feis]).T
    df.loc[eligible_mask, "classifier_score"] = stack.max(axis=1)
    # 3-way argmax with a `neither` floor (all heads below threshold -> neither).
    head_names = np.array(LABEL_ORDER)
    labels = np.where(stack.max(axis=1) < LABEL_THRESHOLD, "neither", head_names[stack.argmax(axis=1)])
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
            "p_initiation", "p_decision", "p_final_eis", "classifier_label", "classifier_score",
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


# ---------------------------------------------------------------------------
# Active-learning batch selection (--emit-batch)
# ---------------------------------------------------------------------------
# Uncertainty band on the top head probability (a candidate the model can't place
# between positive and `neither`).
AL_UNCERTAINTY_LO, AL_UNCERTAINTY_HI = 0.35, 0.65
# Round 2: reserve a fixed floor for clear_initiation candidates (any process, any
# uncertainty score). Eval showed 12/21 clear_initiation test rows predicted `neither` —
# the model is under-confident on the signal the regex already called high-confidence.
# These are NOT band-limited: we want them even when the model thinks it knows the answer.
AL_CLEAR_INIT_N = 60


def run_emit_batch(n: int, model_dir: Path) -> None:
    """Append the N most-uncertain UNLABELED candidates to labeling_sample.csv for the
    next labeling round. Deterministic: ranked by |max(p_init,p_dec) - 0.5| with
    candidate_id as the tiebreak (no RNG), so the same scored pool reproduces the same batch."""
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"No candidates parquet at {CANDIDATES_PATH}. Run 03 then default scoring.")
    df = pd.read_parquet(CANDIDATES_PATH)
    if "p_initiation" not in df.columns or "p_decision" not in df.columns:
        raise SystemExit("Candidates are unscored (no p_initiation/p_decision). "
                         "Run `04_classify_candidates.py` (default scoring) first.")
    if not LABELING_SAMPLE_PATH.exists():
        raise SystemExit(f"{LABELING_SAMPLE_PATH} is missing — cannot append a batch.")

    pool = df[compute_eligible_mask(df)].copy()
    labeled_ids = set(pd.read_csv(LABELING_SAMPLE_PATH, usecols=["candidate_id"])["candidate_id"])
    pool = pool[~pool["candidate_id"].isin(labeled_ids)]
    if pool.empty:
        raise SystemExit("No unlabeled eligible candidates available to emit.")

    pool["_p_init"] = pd.to_numeric(pool["p_initiation"], errors="coerce").fillna(0.0)
    pool["_p_dec"] = pd.to_numeric(pool["p_decision"], errors="coerce").fillna(0.0)
    pool["_maxp"] = pool[["_p_init", "_p_dec"]].max(axis=1)
    pool["_uncertainty"] = (pool["_maxp"] - 0.5).abs()  # smaller = more uncertain
    pool = pool.sort_values(["_uncertainty", "candidate_id"]).reset_index(drop=True)

    # (1) clear_initiation floor (not band-limited): fix the model's under-confidence on
    # candidates the regex already flagged as clear initiation signals.
    n_ci = min(AL_CLEAR_INIT_N, n)
    ci_sel = pool[pool["candidate_role"].eq("clear_initiation")].head(n_ci)
    # (2) Uncertainty slice from the rest, limited to the informative band.
    taken = set(ci_sel["candidate_id"])
    band = pool["_maxp"].between(AL_UNCERTAINTY_LO, AL_UNCERTAINTY_HI)
    rest = pool[band & ~pool["candidate_id"].isin(taken)].head(n - len(ci_sel))
    sel = pd.concat([ci_sel.assign(stratum="al2_clear_init"),
                     rest.assign(stratum="al2_uncertain")]).head(n)

    existing = pd.read_csv(LABELING_SAMPLE_PATH)
    batch = sel.reindex(columns=existing.columns, fill_value="")
    batch["label"] = ""
    batch["notes"] = ""
    batch["split"] = "train"          # never the frozen test set
    batch["stratum"] = sel["stratum"].values
    pd.concat([existing, batch], ignore_index=True).to_csv(LABELING_SAMPLE_PATH, index=False)

    meta_path = model_dir / "classifier_meta.json"
    mv = json.loads(meta_path.read_text()).get("model_version", "unknown") if meta_path.exists() else "unknown"
    ids_path = OUTPUT_DIR / f"al_batch_{mv}_ids.txt"
    ids_path.write_text(
        f"# active-learning batch | model_version={mv} | n={len(sel)} "
        f"(clear_init={len(ci_sel)}, uncertain={len(sel) - len(ci_sel)}) | "
        f"rank=|max(p)-0.5| asc, tiebreak candidate_id | band=[{AL_UNCERTAINTY_LO},{AL_UNCERTAINTY_HI}]\n"
        + "\n".join(sel["candidate_id"].astype(str)) + "\n"
    )
    print(f"Emitted {len(sel)} unlabeled candidates -> {LABELING_SAMPLE_PATH.name} "
          f"(clear_init={len(ci_sel)}, uncertain={len(sel) - len(ci_sel)}; selector model {mv}).")
    print(f"  uncertainty |max(p)-0.5|: min={sel['_uncertainty'].min():.3f} "
          f"median={sel['_uncertainty'].median():.3f} max={sel['_uncertainty'].max():.3f}")
    print(f"  batch ids -> {ids_path}\n  Fill the `label` column for these rows, then run --train.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify timeline date candidates (D4).")
    parser.add_argument("--process", nargs="+", default=["CE", "EA", "EIS"],
                        choices=["CE", "EA", "EIS"])
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--append", action="store_true",
                        help="Merge scores into existing candidates parquet by candidate_id.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-dir", help="Override run directory (read/write candidates here).")
    parser.add_argument("--train", action="store_true",
                        help="Train on labeling_sample.csv (split==train); validate on frozen test.")
    parser.add_argument("--eval", action="store_true", help="Evaluate on the held-out test split.")
    parser.add_argument("--emit-batch", type=int, metavar="N", default=None,
                        help="Active learning: append the N most-uncertain UNLABELED candidates "
                             "to labeling_sample.csv (split=train, blank label) for the next round.")
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
    if args.emit_batch is not None:
        run_emit_batch(args.emit_batch, model_dir)
        return
    run_score(args, model_dir)


if __name__ == "__main__":
    main()
