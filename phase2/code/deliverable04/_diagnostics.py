"""
Shared D4 diagnostics writers — called by the workflow scripts at the step that produces each
diagnostic, so phase2/output/deliverable04/diagnostics/ always reflects the latest run. There is
no standalone "run all diagnostics" script: each output is refreshed by its natural owner.

Numbered in the order the D4 workflow produces them:

    01_label_inventory.csv          04 --train   labels by process x label x split
    02_metrics_by_round.csv         04 --train   frozen-test P/R/F1 per AL round (progression)
    03_frozen_test_confusion.csv    04 --train / --eval   3-class confusion (current model)
    04_frozen_test_per_process.csv  04 --train / --eval   per-process per-head F1
    05_calibration_reliability.csv  04b --fit    raw vs calibrated reliability bins
    06_operating_curve_candidate.csv  04b --curve  candidate-level auto/route + precision
    07_operating_curve_project.csv    04b --curve  project-level auto/route + cost
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DIAG_DIR = ROOT / "phase2" / "output" / "deliverable04" / "diagnostics"
LABEL_THRESHOLD = 0.5

# Frozen-test metrics per AL round. Seed of the recorded progression; the current model is
# merged in automatically from its meta on every --train, so this grows without manual edits.
METRICS_HISTORY = [
    {"round": "baseline", "model_version": "20260603T222207Z", "train_rows": 614,
     "init_f1": 0.556, "init_p": 0.556, "init_r": 0.556,
     "dec_f1": 0.647, "dec_p": 0.688, "dec_r": 0.611, "notes": "pre-active-learning"},
    {"round": "al_round_1", "model_version": "20260604T032402Z", "train_rows": 814,
     "init_f1": 0.615, "init_p": 0.571, "init_r": 0.667,
     "dec_f1": 0.737, "dec_p": 0.700, "dec_r": 0.778, "notes": "+200 uncertainty sample"},
    {"round": "al_round_2", "model_version": "20260604T060644Z", "train_rows": 1014,
     "init_f1": 0.649, "init_p": 0.632, "init_r": 0.667,
     "dec_f1": 0.737, "dec_p": 0.700, "dec_r": 0.778, "notes": "+200 clear_init floor + uncertain"},
]


def _ensure() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


# --- 01 ---------------------------------------------------------------------
def write_label_inventory(labeled_df: pd.DataFrame) -> None:
    _ensure()
    df = labeled_df.copy()
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    df["split"] = df["split"].fillna("train").astype(str).str.strip().replace("", "train")
    df = df[df["label"].ne("")]
    pivot = (df.pivot_table(index=["process_type", "label"], columns="split",
                            values="candidate_id", aggfunc="count", fill_value=0)
             .reset_index())
    pivot.to_csv(DIAG_DIR / "01_label_inventory.csv", index=False)


# --- 02 ---------------------------------------------------------------------
def update_metrics_by_round(meta: dict) -> None:
    _ensure()
    hist = {r["model_version"]: dict(r) for r in METRICS_HISTORY}
    path = DIAG_DIR / "02_metrics_by_round.csv"
    if path.exists():  # preserve any previously-recorded rounds
        for r in pd.read_csv(path).to_dict("records"):
            hist.setdefault(r["model_version"], r)
    mv, tm = meta.get("model_version"), meta.get("test_metrics", {})
    if mv and tm:
        hist[mv] = {
            "round": hist.get(mv, {}).get("round", "current"),
            "model_version": mv, "train_rows": meta.get("n_train"),
            "init_f1": tm.get("initiation", {}).get("f1"),
            "init_p": tm.get("initiation", {}).get("precision"),
            "init_r": tm.get("initiation", {}).get("recall"),
            "dec_f1": tm.get("decision", {}).get("f1"),
            "dec_p": tm.get("decision", {}).get("precision"),
            "dec_r": tm.get("decision", {}).get("recall"),
            "notes": hist.get(mv, {}).get("notes", "from classifier_meta.json"),
        }
    pd.DataFrame(list(hist.values())).to_csv(path, index=False)


# --- 03 / 04 ----------------------------------------------------------------
def _predicted_label(p_init: float, p_dec: float) -> str:
    if p_init < LABEL_THRESHOLD and p_dec < LABEL_THRESHOLD:
        return "neither"
    return "initiation" if p_init >= p_dec else "decision"


def write_confusion(test_df: pd.DataFrame, y_prob: np.ndarray) -> None:
    _ensure()
    pred = [_predicted_label(a, b) for a, b in zip(y_prob[:, 0], y_prob[:, 1])]
    true = test_df["label"].astype(str).str.strip().str.lower()
    pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="pred"), dropna=False) \
        .to_csv(DIAG_DIR / "03_frozen_test_confusion.csv")


def _head_f1(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    out = []
    for i in range(2):
        pred = (y_prob[:, i] >= LABEL_THRESHOLD).astype(int)
        tp = int(((pred == 1) & (y_true[:, i] == 1)).sum())
        fp = int(((pred == 1) & (y_true[:, i] == 0)).sum())
        fn = int(((pred == 0) & (y_true[:, i] == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return out[0], out[1]


def write_per_process(test_df: pd.DataFrame, y_prob: np.ndarray, y_true: np.ndarray) -> None:
    _ensure()
    rows = []
    for proc in sorted(test_df["process_type"].dropna().unique()):
        m = (test_df["process_type"] == proc).to_numpy()
        i_f1, d_f1 = _head_f1(y_true[m], y_prob[m])
        rows.append({"process_type": proc,
                     "n_init_pos": int(y_true[m, 0].sum()), "init_f1": round(i_f1, 3),
                     "n_dec_pos": int(y_true[m, 1].sum()), "dec_f1": round(d_f1, 3)})
    pd.DataFrame(rows).to_csv(DIAG_DIR / "04_frozen_test_per_process.csv", index=False)


# --- 05 ---------------------------------------------------------------------
def write_calibration_reliability(y_prob: np.ndarray, y_true: np.ndarray,
                                  p_i_cal: np.ndarray, p_d_cal: np.ndarray) -> None:
    _ensure()
    rows = []
    for head, raw, cal, yt in [("initiation", y_prob[:, 0], p_i_cal, y_true[:, 0]),
                               ("decision", y_prob[:, 1], p_d_cal, y_true[:, 1])]:
        order = np.argsort(raw)
        for b, idx in enumerate(np.array_split(order, 5)):
            if len(idx) == 0:
                continue
            rows.append({"head": head, "bin": b + 1,
                         "raw_lo": round(float(raw[idx].min()), 4),
                         "raw_hi": round(float(raw[idx].max()), 4),
                         "mean_raw": round(float(raw[idx].mean()), 4),
                         "mean_calibrated": round(float(cal[idx].mean()), 4),
                         "actual_positive_rate": round(float(yt[idx].mean()), 4),
                         "n": int(len(idx))})
    pd.DataFrame(rows).to_csv(DIAG_DIR / "05_calibration_reliability.csv", index=False)


# --- 06 / 07 ----------------------------------------------------------------
def write_operating_curves(cand_curve: pd.DataFrame, proj_curve: pd.DataFrame) -> None:
    _ensure()
    cand_curve.to_csv(DIAG_DIR / "06_operating_curve_candidate.csv", index=False)
    proj_curve.to_csv(DIAG_DIR / "07_operating_curve_project.csv", index=False)
