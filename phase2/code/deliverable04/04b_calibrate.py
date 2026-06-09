"""
Calibrate D4 candidate-classifier probabilities and build an operating curve.

The learned scorer in 04_classify_candidates.py emits two raw SetFit head
probabilities:

    p_initiation = P(candidate is an initiation date)
    p_decision   = P(candidate is a decision date)

Those probabilities are useful for ranking but not calibrated enough to use as
thresholds directly. This script fits one Platt calibrator per head on the
frozen test split, then reports the tradeoff between automatic resolution and
LLM routing at candidate-confidence thresholds.

Usage:
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --fit
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --curve
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/04b_calibrate.py --apply
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import importlib.util
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"


def _load_04():
    spec = importlib.util.spec_from_file_location(
        "classify_candidates",
        Path(__file__).parent / "04_classify_candidates.py",
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not import 04_classify_candidates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_04 = _load_04()
load_model = _04.load_model
build_input_text = _04.build_input_text
MODEL_DIR = _04.MODEL_DIR
LABELING_SAMPLE_PATH = _04.LABELING_SAMPLE_PATH
CANDIDATES_PATH = _04.CANDIDATES_PATH
OUTPUT_DIR = _04.OUTPUT_DIR
TEST_SPLIT_VALUE = _04.TEST_SPLIT_VALUE
LABEL_ORDER = _04.LABEL_ORDER

CAL_INIT_PATH = MODEL_DIR / "calibrator_init.pkl"
CAL_DEC_PATH = MODEL_DIR / "calibrator_dec.pkl"
CAL_FEIS_PATH = MODEL_DIR / "calibrator_feis.pkl"  # 3rd head (EIS-only final_eis fallback)


def _calibrate_one(raw, cal) -> np.ndarray:
    """Platt-calibrate a single head's raw probabilities -> calibrated P(positive)."""
    return cal.predict_proba(np.asarray(raw, dtype=float).reshape(-1, 1))[:, 1]
# Operating-curve CSVs are written into diagnostics/ via _diagnostics.write_operating_curves.

# Claude Haiku 4.5 pricing (input tokens only; context is the cost driver).
HAIKU_COST_PER_TOKEN = 0.80 / 1_000_000
# The LLM is called ONCE PER ROUTED PROJECT (06 packs the top-k candidate packets into a
# single prompt), not once per candidate. Estimate ~1,500 input tokens per project call:
# system prompt (~300) + project metadata (~150) + ROUTED_TOPK candidate packets (~1,000).
AVG_PROJECT_PROMPT_TOKENS = 1500

# Predicted-positive labels: only these candidates ever enter the LLM pipeline. Candidates
# predicted `neither` by 04 are DROPPED here — never auto-resolved, never routed, never sent.
POSITIVE_LABELS = ["initiation", "decision"]

DEFAULT_POOL_MODEL_VERSION = "20260604T060644Z"


def _load_labeling_sample() -> pd.DataFrame:
    if not LABELING_SAMPLE_PATH.exists():
        raise SystemExit(f"Missing labeling sample: {LABELING_SAMPLE_PATH}")
    df = pd.read_csv(LABELING_SAMPLE_PATH)
    required = {"label", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{LABELING_SAMPLE_PATH} is missing columns: {missing}")
    df["split"] = df["split"].fillna("train").astype(str).str.strip().replace("", "train")
    df["label"] = df["label"].fillna("").astype(str).str.strip().str.lower()
    df = df[df["label"].ne("")].copy()
    bad = sorted(set(df["label"]) - {"initiation", "decision", "neither", "final_eis"})
    if bad:
        raise SystemExit(
            "Unrecognized label values in labeling_sample.csv: "
            + ", ".join(repr(x) for x in bad)
        )
    return df


def _load_test_rows() -> pd.DataFrame:
    df = _load_labeling_sample()
    test = df[df["split"].eq(TEST_SPLIT_VALUE)].copy()
    if test.empty:
        raise SystemExit(
            f"No rows with split == {TEST_SPLIT_VALUE!r} in {LABELING_SAMPLE_PATH.name}."
        )
    return test


def _score_test_rows() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict]:
    model, meta = load_model(MODEL_DIR)
    if model is None:
        raise SystemExit(f"No trained classifier found at {MODEL_DIR}. Run 04 --train first.")
    df = _load_test_rows()
    texts = [build_input_text(r) for _, r in df.iterrows()]
    y_prob = np.asarray(model.predict_proba(texts), dtype=float)
    if y_prob.ndim != 2 or y_prob.shape[1] < 2:
        raise SystemExit(f"Expected model.predict_proba to return shape (N, k>=2); got {y_prob.shape}.")
    # Ensure a 3rd (final_eis) column; a legacy 2-head model -> zeros (head never fires).
    if y_prob.shape[1] < 3:
        y_prob = np.hstack([y_prob, np.zeros((y_prob.shape[0], 3 - y_prob.shape[1]))])
    y_prob = y_prob[:, :3]
    y_true = np.stack(
        [
            df["label"].eq("initiation").astype(int).to_numpy(),
            df["label"].eq("decision").astype(int).to_numpy(),
            df["label"].eq("final_eis").astype(int).to_numpy(),
        ],
        axis=1,
    )
    return df.reset_index(drop=True), y_prob, y_true, meta


def _load_calibrators() -> tuple[LogisticRegression, LogisticRegression, LogisticRegression]:
    missing = [p for p in [CAL_INIT_PATH, CAL_DEC_PATH, CAL_FEIS_PATH] if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing calibrator file(s): "
            + ", ".join(str(p) for p in missing)
            + "\nRun --fit first."
        )
    with open(CAL_INIT_PATH, "rb") as f:
        cal_init = pickle.load(f)
    with open(CAL_DEC_PATH, "rb") as f:
        cal_dec = pickle.load(f)
    with open(CAL_FEIS_PATH, "rb") as f:
        cal_feis = pickle.load(f)
    return cal_init, cal_dec, cal_feis


def _calibrated_probs(
    p_init_raw: np.ndarray,
    p_dec_raw: np.ndarray,
    cal_init: LogisticRegression,
    cal_dec: LogisticRegression,
) -> tuple[np.ndarray, np.ndarray]:
    p_init_raw = np.asarray(p_init_raw, dtype=float).reshape(-1, 1)
    p_dec_raw = np.asarray(p_dec_raw, dtype=float).reshape(-1, 1)
    p_i_cal = cal_init.predict_proba(p_init_raw)[:, 1]
    p_d_cal = cal_dec.predict_proba(p_dec_raw)[:, 1]
    return p_i_cal, p_d_cal


def _print_bin_table(
    head_name: str,
    raw_prob: np.ndarray,
    cal_prob: np.ndarray,
    y_true: np.ndarray,
) -> None:
    order = np.argsort(raw_prob)
    bins = np.array_split(order, 5)
    print(f"\n{head_name} calibration bins (frozen test, sorted by raw score):")
    print("  raw_lo  raw_hi  mean_raw  mean_cal  actual_positive_rate  n")
    for idx in bins:
        if len(idx) == 0:
            continue
        raw_bin = raw_prob[idx]
        cal_bin = cal_prob[idx]
        true_bin = y_true[idx]
        print(
            f"  {raw_bin.min():6.3f}  {raw_bin.max():6.3f}  "
            f"{raw_bin.mean():8.3f}  {cal_bin.mean():8.3f}  "
            f"{true_bin.mean():20.3f}  {len(idx):3d}"
        )


def _model_version(meta: dict | None = None) -> str:
    if meta and meta.get("model_version"):
        return str(meta["model_version"])
    meta_path = MODEL_DIR / "classifier_meta.json"
    if meta_path.exists():
        data = json.loads(meta_path.read_text())
        if data.get("model_version"):
            return str(data["model_version"])
    return DEFAULT_POOL_MODEL_VERSION


def _load_versioned_pool(model_version: str) -> pd.DataFrame:
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Missing candidate pool: {CANDIDATES_PATH}")
    pool = pd.read_parquet(CANDIDATES_PATH)
    if "classifier_model_version" not in pool.columns:
        raise SystemExit(f"{CANDIDATES_PATH} has no classifier_model_version column.")
    pool = pool[pool["classifier_model_version"].eq(model_version)].copy()
    if pool.empty:
        raise SystemExit(
            f"No candidates found with classifier_model_version == {model_version!r}."
        )
    return pool


def run_fit() -> None:
    df, y_prob, y_true, meta = _score_test_rows()
    print(
        f"Loaded {len(df)} frozen-test rows from {LABELING_SAMPLE_PATH.name} "
        f"(model {meta.get('model_version', 'unknown')})."
    )
    print(
        f"Test positives: initiation={int(y_true[:, 0].sum())}, "
        f"decision={int(y_true[:, 1].sum())}, final_eis={int(y_true[:, 2].sum())}."
    )

    cal_init = LogisticRegression(C=1.0, solver="lbfgs").fit(
        y_prob[:, 0].reshape(-1, 1),
        y_true[:, 0],
    )
    cal_dec = LogisticRegression(C=1.0, solver="lbfgs").fit(
        y_prob[:, 1].reshape(-1, 1),
        y_true[:, 1],
    )
    # 3rd head: class_weight='balanced' — final_eis is rare on the frozen test (~44 pos),
    # so an unweighted Platt fit collapses toward the negative prior.
    cal_feis = LogisticRegression(C=1.0, solver="lbfgs", class_weight="balanced").fit(
        y_prob[:, 2].reshape(-1, 1),
        y_true[:, 2],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(CAL_INIT_PATH, "wb") as f:
        pickle.dump(cal_init, f)
    with open(CAL_DEC_PATH, "wb") as f:
        pickle.dump(cal_dec, f)
    with open(CAL_FEIS_PATH, "wb") as f:
        pickle.dump(cal_feis, f)

    p_i_cal, p_d_cal = _calibrated_probs(y_prob[:, 0], y_prob[:, 1], cal_init, cal_dec)
    p_f_cal = _calibrate_one(y_prob[:, 2], cal_feis)
    _print_bin_table("p_initiation", y_prob[:, 0], p_i_cal, y_true[:, 0])
    _print_bin_table("p_decision", y_prob[:, 1], p_d_cal, y_true[:, 1])
    _print_bin_table("p_final_eis", y_prob[:, 2], p_f_cal, y_true[:, 2])

    # Diagnostic 05 — calibration reliability bins (refreshed by the fit step).
    try:
        import _diagnostics as diag
        diag.write_calibration_reliability(y_prob, y_true, p_i_cal, p_d_cal)
    except Exception as e:
        print(f"  (diagnostics skipped: {e})")

    print(
        f"\nNote: calibrators are fit on {len(df)} frozen-test rows "
        f"(initiation={int(y_true[:, 0].sum())}, decision={int(y_true[:, 1].sum())} positives). "
        "Platt scaling; precision estimates in the curve at very high τ (few test rows above "
        "threshold) are directional. test_v2 is positive-heavy, so curve precision is optimistic "
        "vs deployment — read deployment precision off the full-pool operating point."
    )
    print(f"\nSaved calibrators:")
    print(f"  {CAL_INIT_PATH}")
    print(f"  {CAL_DEC_PATH}")
    print(f"  {CAL_FEIS_PATH}")


def _precision_at_thresholds(
    y_prob_raw: np.ndarray,
    y_true: np.ndarray,
    labels: pd.Series,
    cal_init: LogisticRegression,
    cal_dec: LogisticRegression,
    thresholds: np.ndarray,
) -> dict[float, dict[str, float]]:
    p_i_cal, p_d_cal = _calibrated_probs(y_prob_raw[:, 0], y_prob_raw[:, 1], cal_init, cal_dec)
    p_max = np.maximum(p_i_cal, p_d_cal)
    # Predicted label for every row: whichever head is higher (used for combined precision).
    pred_label = np.where(p_i_cal >= p_d_cal, "initiation", "decision")
    true_label = labels.fillna("").astype(str).str.strip().str.lower().to_numpy()

    out: dict[float, dict[str, float]] = {}
    for tau in thresholds:
        init_auto = p_i_cal >= tau
        dec_auto  = p_d_cal >= tau
        # All auto-resolved rows (including true-neither rows that are false positives).
        # precision_combined = TP / (TP + FP): fraction of auto-resolved predictions that
        # match the true label (neither rows predicted as init/dec count as wrong).
        above_tau = p_max >= tau

        out[float(tau)] = {
            "precision_init": float(y_true[init_auto, 0].mean()) if init_auto.any() else np.nan,
            "precision_dec":  float(y_true[dec_auto,  1].mean()) if dec_auto.any()  else np.nan,
            "precision_combined": (
                float((pred_label[above_tau] == true_label[above_tau]).mean())
                if above_tau.any()
                else np.nan
            ),
        }
    return out


def _project_aggregates(pool: pd.DataFrame, p_i_cal: np.ndarray, p_d_cal: np.ndarray) -> pd.DataFrame:
    """Per-project best calibrated confidence for each head. The LLM is called once per
    project, so this is the unit that actually drives routing + cost. best_init / best_dec
    are the most-confident init/decision candidate the project has (max over its candidates)."""
    tmp = pd.DataFrame({
        "project_id": pool["project_id"].to_numpy(),
        "p_i_cal": p_i_cal,
        "p_d_cal": p_d_cal,
    })
    g = tmp.groupby("project_id")
    return pd.DataFrame({
        "best_init": g["p_i_cal"].max(),
        "best_dec": g["p_d_cal"].max(),
    })


def run_curve() -> None:
    cal_init, cal_dec, _cal_feis = _load_calibrators()  # routing curve uses init/decision only
    test_df, y_prob_test, y_true_test, meta = _score_test_rows()
    model_version = _model_version(meta)
    pool = _load_versioned_pool(model_version)

    p_i_raw = pd.to_numeric(pool["p_initiation"], errors="coerce").fillna(0.0).to_numpy()
    p_d_raw = pd.to_numeric(pool["p_decision"], errors="coerce").fillna(0.0).to_numpy()
    p_i_cal, p_d_cal = _calibrated_probs(p_i_raw, p_d_raw, cal_init, cal_dec)
    p_max_cal = np.maximum(p_i_cal, p_d_cal)

    # --- neither filter: only predicted-positive candidates ever enter the LLM pipeline.
    # Candidates 04 labeled `neither` are dropped here (not auto-resolved, not routed, not sent).
    label = pool["classifier_label"].fillna("").astype(str).str.strip().str.lower()
    is_pos = label.isin(POSITIVE_LABELS).to_numpy()
    n_pool = len(pool)
    n_neither = int((~is_pos).sum())
    n_pos = int(is_pos.sum())

    # --- per-project aggregates (over ALL candidates: "does this project have a confident
    # init/decision date available"). Project count is the real LLM-call denominator.
    proj = _project_aggregates(pool, p_i_cal, p_d_cal)
    n_proj = len(proj)
    best_init = proj["best_init"].to_numpy()
    best_dec = proj["best_dec"].to_numpy()
    # Projects that have at least one positive-predicted candidate (others go to the separate
    # document-recovery path, not candidate adjudication — reported as a constant, not routed here).
    proj_has_pos = pool.loc[is_pos, "project_id"].nunique()

    thresholds = np.concatenate([np.arange(0.10, 0.50, 0.10), np.arange(0.50, 0.96, 0.05)])
    thresholds = np.round(thresholds, 2)
    test_precision = _precision_at_thresholds(
        y_prob_test, y_true_test, test_df["label"], cal_init, cal_dec, thresholds,
    )

    cand_rows, proj_rows = [], []
    for tau in thresholds:
        prec = test_precision[float(tau)]

        # ----- candidate-level (quality view; positive-predicted candidates only) -----
        pos_auto = int(((p_max_cal >= tau) & is_pos).sum())
        pos_routed = int(((p_max_cal < tau) & is_pos).sum())
        cand_rows.append({
            "tau": float(tau),
            "n_pos_auto": pos_auto,
            "pos_auto_pct": pos_auto / n_pos * 100 if n_pos else np.nan,
            "n_pos_routed": pos_routed,
            "precision_init": prec["precision_init"],
            "precision_dec": prec["precision_dec"],
            "precision_combined": prec["precision_combined"],
        })

        # ----- project-level (operating + cost view; one LLM call per routed project) -----
        auto_both = int(((best_init >= tau) & (best_dec >= tau)).sum())   # full timeline auto
        auto_either = int(((best_init >= tau) | (best_dec >= tau)).sum()) # >=1 date auto
        n_routed_proj = n_proj - auto_both
        est_cost = n_routed_proj * AVG_PROJECT_PROMPT_TOKENS * HAIKU_COST_PER_TOKEN
        proj_rows.append({
            "tau": float(tau),
            "proj_auto_both": auto_both,
            "auto_both_pct": auto_both / n_proj * 100 if n_proj else np.nan,
            "proj_auto_either": auto_either,
            "auto_either_pct": auto_either / n_proj * 100 if n_proj else np.nan,
            "n_routed_projects": n_routed_proj,
            "est_cost_usd": est_cost,
            "precision_combined": prec["precision_combined"],
        })

    cand_curve = pd.DataFrame(cand_rows)
    proj_curve = pd.DataFrame(proj_rows)
    # Operating curves live in diagnostics/ (06, 07) — refreshed by this curve step.
    import _diagnostics as diag
    diag.write_operating_curves(cand_curve, proj_curve)

    # ---- header: the funnel that reframes the denominator ----
    print(f"\nPool: {n_pool:,} eligible candidates across {n_proj:,} projects "
          f"(classifier_model_version={model_version}).")
    print(f"  Predicted `neither` (DROPPED, never sent): {n_neither:,} "
          f"({n_neither / n_pool * 100:.1f}%)")
    print(f"  Predicted init/decision (enter pipeline):  {n_pos:,} "
          f"({n_pos / n_pool * 100:.1f}%), spanning {proj_has_pos:,} projects")
    print(f"  Per-project LLM cost estimate @ {AVG_PROJECT_PROMPT_TOKENS} tok/call, "
          f"${HAIKU_COST_PER_TOKEN * 1e6:.2f}/1M tok.")

    print("\n=== CANDIDATE-LEVEL (classifier quality; positive predictions only) ===")
    cdisp = cand_curve.copy()
    cdisp["pos_auto_pct"] = cdisp["pos_auto_pct"].map(lambda x: f"{x:6.2f}")
    for col in ["precision_init", "precision_dec", "precision_combined"]:
        cdisp[col] = cdisp[col].map(lambda x: "" if pd.isna(x) else f"{x:6.3f}")
    print(cdisp.to_string(index=False))

    print("\n=== PROJECT-LEVEL (operating point + cost; one LLM call per routed project) ===")
    pdisp = proj_curve.copy()
    pdisp["auto_both_pct"] = pdisp["auto_both_pct"].map(lambda x: f"{x:6.2f}")
    pdisp["auto_either_pct"] = pdisp["auto_either_pct"].map(lambda x: f"{x:6.2f}")
    pdisp["est_cost_usd"] = pdisp["est_cost_usd"].map(lambda x: f"{x:8.2f}")
    pdisp["precision_combined"] = pdisp["precision_combined"].map(
        lambda x: "" if pd.isna(x) else f"{x:6.3f}")
    print(pdisp.to_string(index=False))
    print(f"\nWrote {diag.DIAG_DIR}/06_operating_curve_candidate.csv")
    print(f"Wrote {diag.DIAG_DIR}/07_operating_curve_project.csv")

    # ---- recommendation: precision is candidate-level (no project gold); report the
    # project coverage it buys. ----
    eligible = proj_curve.merge(
        cand_curve[["tau", "precision_combined"]].rename(columns={"precision_combined": "_pc"}),
        on="tau",
    )
    eligible = eligible[eligible["_pc"].ge(0.85)]
    if eligible.empty:
        print("\nRecommendation: no threshold reaches candidate precision_combined >= 0.85 on "
              "the frozen test. The model is not yet confident enough for high-precision "
              "auto-resolution — see options in the report.")
    else:
        rec = eligible.sort_values("tau").iloc[0]
        print(f"\nRecommendation: lowest tau with candidate precision_combined >= 0.85 is "
              f"{rec['tau']:.2f} -> auto-resolves {rec['auto_both_pct']:.1f}% of projects "
              f"(both dates), routes {int(rec['n_routed_projects']):,} projects, "
              f"est_cost ${rec['est_cost_usd']:.2f}.")


def run_apply() -> None:
    cal_init, cal_dec, cal_feis = _load_calibrators()
    model_version = _model_version()
    if not CANDIDATES_PATH.exists():
        raise SystemExit(f"Missing candidate pool: {CANDIDATES_PATH}")
    df = pd.read_parquet(CANDIDATES_PATH)
    if "classifier_model_version" not in df.columns:
        raise SystemExit(f"{CANDIDATES_PATH} has no classifier_model_version column.")
    mask = df["classifier_model_version"].eq(model_version)
    n = int(mask.sum())
    if n == 0:
        raise SystemExit(
            f"No candidates found with classifier_model_version == {model_version!r}."
        )

    for col in ["p_init_cal", "p_dec_cal", "p_feis_cal"]:
        if col not in df.columns:
            df[col] = np.nan

    p_i_raw = pd.to_numeric(df.loc[mask, "p_initiation"], errors="coerce").fillna(0.0).to_numpy()
    p_d_raw = pd.to_numeric(df.loc[mask, "p_decision"], errors="coerce").fillna(0.0).to_numpy()
    p_i_cal, p_d_cal = _calibrated_probs(p_i_raw, p_d_raw, cal_init, cal_dec)
    df.loc[mask, "p_init_cal"] = p_i_cal
    df.loc[mask, "p_dec_cal"] = p_d_cal

    # 3rd head: only present once the pool is re-scored with the 3-head model (writes p_final_eis).
    if "p_final_eis" in df.columns:
        p_f_raw = pd.to_numeric(df.loc[mask, "p_final_eis"], errors="coerce").fillna(0.0).to_numpy()
        p_f_cal = _calibrate_one(p_f_raw, cal_feis)
        # Preserve the scoring-time doc-type gate: the balanced calibrator maps raw 0 -> ~0.24,
        # which would re-inflate p_feis_cal on gated non-FEIS candidates. Force gated rows back to 0.
        p_f_cal = np.where(p_f_raw > 0, p_f_cal, 0.0)
        df.loc[mask, "p_feis_cal"] = p_f_cal
        feis_note = "incl. p_feis_cal (doc-type gate preserved)"
    else:
        feis_note = "p_final_eis not in pool yet -> p_feis_cal left NaN (re-score with 3-head model first)"
    df.to_parquet(CANDIDATES_PATH, index=False)
    print(f"Applied calibrated scores to {n:,} candidates ({feis_note}).")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate D4 classifier + build operating curve."
    )
    parser.add_argument("--fit", action="store_true", help="Fit Platt calibrators on frozen test.")
    parser.add_argument("--curve", action="store_true", help="Build operating curve (requires --fit first).")
    parser.add_argument("--apply", action="store_true", help="Write p_init_cal/p_dec_cal back to candidates parquet.")
    args = parser.parse_args()
    if args.fit:
        run_fit()
    if args.curve:
        run_curve()
    if args.apply:
        run_apply()
    if not (args.fit or args.curve or args.apply):
        parser.print_help()


if __name__ == "__main__":
    main()
