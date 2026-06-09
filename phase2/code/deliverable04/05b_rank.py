"""
Learned selection ranker for D4 — LightGBM learning-to-rank (lambdarank).

WHERE THIS SITS
---------------
`05_select_dates.py` picks the final initiation / decision date per project using a HAND-WEIGHTED
sum of features (candidate_score_components). This script LEARNS that ranking instead: per project,
which candidate is THE initiation, and which is THE decision. It consumes the full feature set —
the classifier probabilities AND every structural signal (doc type, role, page position,
granularity, cross-candidate agreement, mention count, negative cues) — not just the classifier
score.

It needs PROJECT-LEVEL labels (the true candidate_id per project), produced by the
`project_gold_labeling.md` pass into `project_gold_sample.csv`. Candidate-level init/dec/neither
labels can't teach within-head selection; this can.

TWO MODELS, SAME FEATURES
-------------------------
One LGBMRanker per head (init, decision). Identical feature matrix; the relevance label differs
(1 for the gold candidate in that head, 0 for the project's other candidates). lambdarank learns to
rank the true candidate to the top of its project group.

MODES
-----
  --train   fit both rankers on the `split == "train"` gold projects; report eval on `split == "test"`.
  --eval    top-1 / MRR accuracy on the held-out gold projects (does the ranker's #1 == the gold pick?).
  --apply   score every candidate in the pool, write `learned_init_score` / `learned_decision_score`
            columns back to timeline_candidates.parquet.

HOW 05 CONSUMES IT (wire AFTER the ranker is trained + validated)
-----------------------------------------------------------------
In `05_select_dates.py`, after `ranking_score` is computed, prefer the learned score when present:
    if "learned_decision_score" in cands:  # decision pass
        decision_cands["ranking_score"] = decision_cands["learned_decision_score"].fillna(
            decision_cands["ranking_score"])
    if "learned_init_score" in cands:       # initiation pass
        initiation_cands["ranking_score"] = initiation_cands["learned_init_score"].fillna(
            initiation_cands["ranking_score"])
The disambiguation rules (earliest-init, day>month, CE-month-only) and chronology filter stay in 05.
Until the columns exist, 05 falls back to the heuristic — nothing breaks.

Usage:
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --train
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --eval
    CONDA_DEFAULT_ENV=nepa python phase2/code/deliverable04/05b_rank.py --apply
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


# ---------------------------------------------------------------------------
# Load 05 internals (digit-prefixed filename -> importlib)
# ---------------------------------------------------------------------------
def _load_05():
    spec = importlib.util.spec_from_file_location(
        "select_dates", Path(__file__).parent / "05_select_dates.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("Could not import 05_select_dates.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_05 = _load_05()
TIMELINE_DIR = _05.TIMELINE_DIR
OUTPUT_DIR = _05.OUTPUT_DIR
TRAINING_DIR = _05.PHASE2 / "training" / "deliverable04"   # label INPUTS
CANDIDATES_PATH = _05.CANDIDATES_PATH

GOLD_SAMPLE_PATH = TRAINING_DIR / "ranker.csv"   # was output/project_gold_sample.csv
# Guardrail registry: project_ids reserved for evaluation that must NEVER be trained on.
# A label is training XOR evaluation. run_train hard-fails if any train project is in here.
FROZEN_EVAL_IDS_PATH = TRAINING_DIR / "frozen_eval_ids.txt"


def _load_frozen_eval_ids() -> set[str]:
    if not FROZEN_EVAL_IDS_PATH.exists():
        return set()
    return {ln.strip() for ln in FROZEN_EVAL_IDS_PATH.read_text().splitlines() if ln.strip()}


def _assert_no_eval_leak(train_gold: pd.DataFrame) -> None:
    """Hard-fail if any training project is in the frozen-eval registry (train/eval contamination)."""
    frozen = _load_frozen_eval_ids()
    leak = set(train_gold["project_id"].astype(str)) & frozen
    if leak:
        raise SystemExit(
            f"[GUARDRAIL] {len(leak)} training project(s) are in the frozen-eval registry "
            f"({FROZEN_EVAL_IDS_PATH.name}) — that is train/eval contamination. Offending ids: "
            f"{sorted(leak)[:5]}{'…' if len(leak) > 5 else ''}. Mark them split=test in ranker.csv "
            f"or remove them from the registry."
        )
    if frozen:
        print(f"  guardrail OK: {len(frozen)} frozen-eval ids, none in the train split.")
RANKER_DIR = TIMELINE_DIR / "models" / "candidate_ranker"
RANKER_INIT_PATH = RANKER_DIR / "ranker_init.pkl"
RANKER_DEC_PATH = RANKER_DIR / "ranker_decision.pkl"
RANKER_META_PATH = RANKER_DIR / "ranker_meta.json"

TEST_FRACTION = 0.20
SEED = 42

# Feature columns. Numeric ones are precomputed in the candidates parquet (03/04/05); we add
# agreement_count + granularity_num. Categoricals are passed to LightGBM as native categories.
GRAN_NUM = {"day": 2.0, "month": 1.0, "year": 0.0}
NUM_FEATURES = [
    "p_init_cal", "p_dec_cal", "p_feis_cal", "p_final_eis", "p_initiation", "p_decision",
    "role_confidence_score", "source_strength", "role_cue_strength",
    "document_priority", "section_priority", "page_priority", "position_signal",
    "position_pct", "section_position_pct", "repeated_mention_signal",
    "negative_penalty", "date_mention_count", "agreement_count", "granularity_num",
]
CAT_FEATURES = ["candidate_role", "process_type", "date_granularity", "document_type_category"]
HEADS = [("initiation", "initiation_candidate_id", "learned_init_score", RANKER_INIT_PATH),
         ("decision", "decision_candidate_id", "learned_decision_score", RANKER_DEC_PATH)]


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for a set of candidate rows. Robust to missing columns (filled with 0/NA).
    Numerics coerced + NaN-filled; categoricals cast to pandas 'category' for native LightGBM."""
    f = pd.DataFrame(index=df.index)
    # calibrated probs fall back to raw when --apply hasn't been run
    pi = pd.to_numeric(df.get("p_initiation"), errors="coerce")
    pd_ = pd.to_numeric(df.get("p_decision"), errors="coerce")
    f["p_init_cal"] = pd.to_numeric(df.get("p_init_cal"), errors="coerce").fillna(pi)
    f["p_dec_cal"] = pd.to_numeric(df.get("p_dec_cal"), errors="coerce").fillna(pd_)
    # final_eis head (EIS FEIS-as-decision fallback): p_feis_cal is the calibrated score, gated to
    # FEIS docs; fall back to raw p_final_eis when --apply (04b) hasn't run. 0 for non-EIS/non-FEIS.
    pf = pd.to_numeric(df.get("p_final_eis"), errors="coerce")
    f["p_feis_cal"] = pd.to_numeric(df.get("p_feis_cal"), errors="coerce").fillna(pf)
    f["p_final_eis"] = pf
    f["p_initiation"] = pi
    f["p_decision"] = pd_
    for col in ["role_confidence_score", "source_strength", "role_cue_strength",
                "document_priority", "section_priority", "page_priority", "position_signal",
                "position_pct", "section_position_pct", "repeated_mention_signal",
                "negative_penalty", "date_mention_count"]:
        f[col] = pd.to_numeric(df.get(col), errors="coerce")
    # Cross-candidate agreement: how many candidates resolve to the same date — computed
    # PER PROJECT so it's identical at train / eval / apply time (a global value_counts over the
    # whole pool would give wildly different magnitudes and corrupt --apply scores).
    pdates = pd.to_datetime(df.get("parsed_date"), errors="coerce")
    if "project_id" in df.columns:
        f["agreement_count"] = (
            pdates.groupby(df["project_id"]).transform(lambda s: s.map(s.value_counts()))
            .fillna(1).astype(float)
        )
    else:
        f["agreement_count"] = pdates.map(pdates.value_counts()).fillna(1).astype(float)
    f["granularity_num"] = df.get("date_granularity").map(GRAN_NUM).astype(float)
    f = f[NUM_FEATURES].fillna(0.0)
    for c in CAT_FEATURES:
        f[c] = df.get(c, pd.Series(index=df.index, dtype="object")).astype("category")
    return f


# ---------------------------------------------------------------------------
# Gold loading + split
# ---------------------------------------------------------------------------
def _load_gold() -> pd.DataFrame:
    if not GOLD_SAMPLE_PATH.exists():
        raise SystemExit(
            f"No project gold at {GOLD_SAMPLE_PATH}.\n"
            "Run the project_gold_labeling.md pass first (emit -> label -> apply)."
        )
    g = pd.read_csv(GOLD_SAMPLE_PATH, dtype=str, keep_default_na=False)
    filled = g["initiation_candidate_id"].str.strip().ne("") | g["decision_candidate_id"].str.strip().ne("")
    g = g[filled].copy()
    if g.empty:
        raise SystemExit(f"{GOLD_SAMPLE_PATH} has no filled picks yet.")
    # Assign a deterministic stratified train/test split (by process) when blank.
    if "split" not in g.columns or g["split"].str.strip().eq("").all():
        g["split"] = "train"
        for _proc, grp in g.groupby("process_type"):
            n_test = max(1, round(len(grp) * TEST_FRACTION))
            test_ids = grp.sample(n=min(n_test, len(grp) - 1) if len(grp) > 1 else 0,
                                  random_state=SEED).index
            g.loc[test_ids, "split"] = "test"
    return g


def _group_frame(cand: pd.DataFrame, gold: pd.DataFrame, idcol: str) -> tuple[pd.DataFrame, list[int], np.ndarray]:
    """Build the per-project ranking frame for one head: candidates of each gold project, sorted so
    project groups are contiguous, with relevance=1 on the gold candidate. Projects whose gold pick
    is 'none'/missing (no positive) are dropped — lambdarank needs a relevant item per group."""
    gmap = {r.project_id: getattr(r, idcol) for r in gold.itertuples()
            if str(getattr(r, idcol)).strip() not in ("", "none")}
    sub = cand[cand["project_id"].isin(gmap)].copy()
    sub = sub.sort_values("project_id")
    rel = sub.apply(lambda r: 1 if gmap.get(r["project_id"]) == r["candidate_id"] else 0, axis=1)
    # keep only groups that actually contain their gold candidate
    has_pos = sub.assign(_r=rel).groupby("project_id")["_r"].transform("max").astype(bool)
    sub, rel = sub[has_pos], rel[has_pos]
    sizes = sub.groupby("project_id", sort=False).size().tolist()
    return sub, sizes, rel.to_numpy()


# ---------------------------------------------------------------------------
# Train / eval / apply
# ---------------------------------------------------------------------------
def _import_lgbm():
    try:
        from lightgbm import LGBMRanker
        return LGBMRanker
    except ImportError as e:
        raise SystemExit("LightGBM not installed in 'nepa'. Install:\n    pip install lightgbm\n"
                         f"(import error: {e})")


def _fit_one(LGBMRanker, X, y, groups):
    model = LGBMRanker(
        objective="lambdarank", n_estimators=300, learning_rate=0.05,
        num_leaves=15, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X, y, group=groups, categorical_feature=CAT_FEATURES)
    return model


def _topk_metrics(model, cand: pd.DataFrame, gold: pd.DataFrame, idcol: str) -> dict:
    """Top-1 accuracy + MRR on a set of gold projects (those with a real gold candidate)."""
    gmap = {r.project_id: getattr(r, idcol) for r in gold.itertuples()
            if str(getattr(r, idcol)).strip() not in ("", "none")}
    hits, rr, n = 0, 0.0, 0
    per_proc: dict[str, list[int]] = {}
    for pid, true_id in gmap.items():
        c = cand[cand["project_id"] == pid]
        if c.empty or true_id not in set(c["candidate_id"]):
            continue
        scores = model.predict(build_features(c))
        order = c["candidate_id"].to_numpy()[np.argsort(-scores)]
        rank = int(np.where(order == true_id)[0][0]) + 1
        hit = int(rank == 1)
        hits += hit
        rr += 1.0 / rank
        n += 1
        proc = c["process_type"].iloc[0]
        per_proc.setdefault(proc, []).append(hit)
    return {
        "n_projects": n,
        "top1_accuracy": round(hits / n, 3) if n else None,
        "mrr": round(rr / n, 3) if n else None,
        "per_process_top1": {p: round(np.mean(v), 3) for p, v in per_proc.items()},
    }


def run_train() -> None:
    LGBMRanker = _import_lgbm()
    gold = _load_gold()
    cand = pd.read_parquet(CANDIDATES_PATH)
    tr, te = gold[gold["split"] == "train"], gold[gold["split"] == "test"]
    _assert_no_eval_leak(tr)   # guardrail: training must never include a frozen-eval project
    print(f"Gold: {len(gold)} projects ({len(tr)} train / {len(te)} test).")

    RANKER_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"trained_at": datetime.now(timezone.utc).isoformat(),
            "n_gold_train": int(len(tr)), "n_gold_test": int(len(te)),
            "features_num": NUM_FEATURES, "features_cat": CAT_FEATURES, "heads": {}}

    for head, idcol, _scorecol, path in HEADS:
        sub, sizes, y = _group_frame(cand, tr, idcol)
        if not sizes:
            print(f"  [{head}] no usable gold groups — skipped.")
            continue
        model = _fit_one(LGBMRanker, build_features(sub), y, sizes)
        with open(path, "wb") as fh:
            pickle.dump(model, fh)
        m = _topk_metrics(model, cand, te, idcol) if len(te) else {}
        meta["heads"][head] = {"n_train_groups": len(sizes), "eval": m}
        print(f"  [{head}] trained on {len(sizes)} groups -> {path.name}")
        if m:
            print(f"           held-out top1={m['top1_accuracy']} mrr={m['mrr']} "
                  f"(n={m['n_projects']}) per-process={m['per_process_top1']}")
    RANKER_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved ranker meta -> {RANKER_META_PATH}")


def run_eval() -> None:
    gold = _load_gold()
    cand = pd.read_parquet(CANDIDATES_PATH)
    te = gold[gold["split"] == "test"]
    if te.empty:
        raise SystemExit("No test-split gold projects to evaluate.")
    for head, idcol, _scorecol, path in HEADS:
        if not path.exists():
            print(f"  [{head}] no trained ranker — run --train first.")
            continue
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        m = _topk_metrics(model, cand, te, idcol)
        print(f"[{head}] held-out top1={m['top1_accuracy']} mrr={m['mrr']} "
              f"(n={m['n_projects']}) per-process={m['per_process_top1']}")


def run_apply() -> None:
    cand = pd.read_parquet(CANDIDATES_PATH)
    n_written = 0
    for head, _idcol, scorecol, path in HEADS:
        if not path.exists():
            print(f"  [{head}] no trained ranker — skipping {scorecol}.")
            continue
        with open(path, "rb") as fh:
            model = pickle.load(fh)
        cand[scorecol] = model.predict(build_features(cand))
        n_written += 1
        print(f"  wrote {scorecol}")
    if n_written:
        cand.to_parquet(CANDIDATES_PATH, index=False)
        print(f"Applied learned ranker scores to {len(cand):,} candidates "
              f"({n_written} head(s)) -> {CANDIDATES_PATH.name}")
    else:
        print("No rankers found; nothing written. Run --train first.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Learned LightGBM selection ranker (D4).")
    ap.add_argument("--train", action="store_true", help="Fit both rankers on the gold train split.")
    ap.add_argument("--eval", action="store_true", help="Top-1 / MRR on the held-out gold split.")
    ap.add_argument("--apply", action="store_true", help="Write learned_*_score columns to the pool.")
    args = ap.parse_args()
    if args.train:
        run_train()
    if args.eval:
        run_eval()
    if args.apply:
        run_apply()
    if not (args.train or args.eval or args.apply):
        ap.print_help()


if __name__ == "__main__":
    main()
