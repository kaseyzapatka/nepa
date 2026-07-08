"""D2 Phase 5 — validate extraction against the hand-labeled gold set (plan v2.11 §7).

Multi-determination grain (2026-07-08): both the gold set and the extractor are keyed by
(evidence_span_id x resource_area), so validation is a set-matching problem per window. Gated on
`gold/significance_gold.parquet` EXISTING (built by gold_agreement.py --finalize from the two
labelers' long CSVs). Metrics:

  - candidate_is_determination (WINDOW grain): does the window hold >=1 real determination? P/R/F1
  - resource_determination_detection ((window x resource) grain): did we recover the right SET of
    resource determinations? P/R/F1 (this is the multi-determination completeness metric)
  - determination_class macro-F1 over COMMON classes, on MATCHED (window x resource) pairs
  - mitigation-dependence F1, on matched pairs
  - primary_threshold_type accuracy, on matched pairs (descriptive)

Reports overall AND on the >=30%-by-window holdout. Emits a disagreement review queue. Metrics
with inadequate support are reported descriptively, not as pass/fail (plan §7).

Run:  conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import common as C

MIN_SUPPORT = 10  # below this, report descriptively (no macro-F1)
NOT_DET = "not_a_determination"

# deterministic tie-break when a window/resource carries >1 predicted determination: keep the most
# consequential conclusion (mirrors the gold labeler's "keep the operative/final conclusion" rule
# in gold_labeling.md). Lower rank = kept.
_CLASS_RANK = {c: i for i, c in enumerate([
    "significant_unavoidable", "significant_adverse", "eis_required",
    "less_than_significant_with_mitigation", "less_than_significant",
    "no_significant_impact", "ambiguous", "not_a_determination"])}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("1", "true", "yes", "y", "t"))


def _norm(s: pd.Series) -> pd.Series:
    # collapse case + space/dash the same way extract_common._norm_vocab does, so a labeler's
    # "air quality" joins to the extractor's canonical "air_quality" (else a real match reads as FP/FN).
    return (s.astype(str).str.strip().str.lower()
            .str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False))


def macro_f1(gold: pd.Series, pred: pd.Series) -> tuple[float, pd.DataFrame]:
    labels = sorted(set(gold) | set(pred))
    rows, f1s = [], []
    for lab in labels:
        tp = int(((gold == lab) & (pred == lab)).sum())
        fp = int(((gold != lab) & (pred == lab)).sum())
        fn = int(((gold == lab) & (pred != lab)).sum())
        support = int((gold == lab).sum())
        m = prf(tp, fp, fn)
        rows.append({"label": lab, "support": support, **m})
        if support >= MIN_SUPPORT:
            f1s.append(m["f1"])
    macro = round(sum(f1s) / len(f1s), 3) if f1s else float("nan")
    return macro, pd.DataFrame(rows)


def _real_gold(gold: pd.DataFrame) -> pd.DataFrame:
    """Gold rows that assert a real determination (drops the junk `none` rows)."""
    g = gold[_truthy(gold["gold_is_determination"]) &
             (_norm(gold["gold_determination_class"]) != NOT_DET)].copy()
    g["resource"] = _norm(g["gold_resource_area"])
    g["gclass"] = _norm(g["gold_determination_class"])
    g["gmit"] = _truthy(g["gold_mitigation_link"])
    g["gthr"] = _norm(g["gold_primary_threshold_type"]).replace({"": "none", "nan": "none"})
    return g[["evidence_span_id", "resource", "gclass", "gmit", "gthr"]]


def _real_pred(det: pd.DataFrame, windows: set) -> pd.DataFrame:
    """Predicted real determinations restricted to the gold windows; one row per (window,resource).
    When a window/resource carries >1 row (differing by scope/threshold), keep the most
    consequential class (deterministic; mirrors the gold labeler's operative-conclusion rule)."""
    p = det[det["evidence_span_id"].isin(windows) &
            (_norm(det["determination_class"]) != NOT_DET)].copy()
    p["resource"] = _norm(p["shared_resource_area"])
    p["pclass"] = _norm(p["determination_class"])
    p["pmit"] = p["mitigation_dependent"].fillna(False).astype(bool)
    p["pthr"] = _norm(p["primary_threshold_type"]).replace({"": "none", "nan": "none"})
    p["_rank"] = p["pclass"].map(_CLASS_RANK).fillna(len(_CLASS_RANK))
    p = (p.sort_values(["evidence_span_id", "resource", "_rank", "pclass"])
          .drop_duplicates(["evidence_span_id", "resource"], keep="first"))
    return p[["evidence_span_id", "resource", "pclass", "pmit", "pthr"]]


def evaluate(gold: pd.DataFrame, det: pd.DataFrame, tag: str) -> tuple[list, pd.DataFrame]:
    windows = set(gold["evidence_span_id"])
    gr = _real_gold(gold)
    pr = _real_pred(det, windows)
    out = []

    # (1) window-level detection: does the window hold any real determination?
    gpos, ppos = set(gr["evidence_span_id"]), set(pr["evidence_span_id"])
    out.append({"metric": "candidate_is_determination", "scope": tag, "grain": "window",
                **prf(len(gpos & ppos), len(ppos - gpos), len(gpos - ppos))})

    # (2) resource-determination detection: did we recover the right SET of (window,resource)?
    merged = gr.merge(pr, on=["evidence_span_id", "resource"], how="outer", indicator=True)
    tp = int((merged["_merge"] == "both").sum())
    fn = int((merged["_merge"] == "left_only").sum())     # gold has it, pred missed
    fp = int((merged["_merge"] == "right_only").sum())    # pred asserted, gold has none
    out.append({"metric": "resource_determination_detection", "scope": tag,
                "grain": "window×resource", **prf(tp, fp, fn)})

    matched = merged[merged["_merge"] == "both"]
    if len(matched) >= MIN_SUPPORT:
        # (3) determination class on matched pairs
        mf1, _ = macro_f1(matched["gclass"].astype(str), matched["pclass"].astype(str))
        out.append({"metric": "determination_class_macro_f1", "scope": tag,
                    "f1": mf1, "support": len(matched)})
        # (4) mitigation dependence on matched pairs
        gm, pm = matched["gmit"].astype(bool), matched["pmit"].astype(bool)
        out.append({"metric": "mitigation_dependent_f1", "scope": tag,
                    **prf(int((gm & pm).sum()), int((~gm & pm).sum()), int((gm & ~pm).sum()))})
        # (5) primary threshold type accuracy on matched pairs (descriptive)
        tacc = float((matched["gthr"].astype(str) == matched["pthr"].astype(str)).mean())
        out.append({"metric": "primary_threshold_type_accuracy", "scope": tag,
                    "precision": round(tacc, 3), "support": len(matched)})
    else:
        out.append({"metric": "determination_class_macro_f1", "scope": tag,
                    "note": f"matched pairs {len(matched)} < {MIN_SUPPORT} — descriptive only"})
    return out, merged


def main(track: str = "fonsi") -> None:
    # fonsi = the original (unchanged) paths; eis = the parallel EIS gold + EIS determinations
    gold_path = C.GOLD_EIS if track == "eis" else C.GOLD
    det_path = C.SIGNIFICANCE_DETERMINATIONS_EIS if track == "eis" else C.SIGNIFICANCE_DETERMINATIONS
    sfx = "_eis" if track == "eis" else ""
    prompt = "gold_labeling_eis.md" if track == "eis" else "gold_labeling.md"
    if not gold_path.exists():
        print(f"[gold not found] {gold_path.relative_to(C.PHASE2)}")
        print(f"Build it: both labelers write the labels CSVs per {prompt}, then run "
              f"gold_agreement.py --track {track} and gold_agreement.py --track {track} --finalize.")
        sys.exit(0)
    if not det_path.exists():
        print(f"[determinations not found] {det_path.name} — run "
              f"{'04' if track == 'eis' else '02'} first."); sys.exit(0)

    gold = C.q(f"SELECT * FROM read_parquet('{gold_path}')")
    det = C.q(f"""SELECT evidence_span_id, shared_resource_area, determination_class,
                         mitigation_dependent, primary_threshold_type
                  FROM read_parquet('{det_path}')""")
    for col in ("gold_resource_area", "gold_determination_class", "gold_is_determination"):
        if col not in gold.columns:
            print(f"[gold schema] missing '{col}' — is this the multi-determination gold "
                  "(gold_agreement.py --finalize)?"); sys.exit(0)

    n_win = gold["evidence_span_id"].nunique()
    print(f"gold: {len(gold):,} rows across {n_win:,} windows  |  "
          f"determinations table: {len(det):,} rows")

    metrics, merged = evaluate(gold, det, "overall")
    if "holdout" in gold.columns and _truthy(gold["holdout"]).any():
        hgold = gold[_truthy(gold["holdout"])]
        metrics += evaluate(hgold, det, "holdout")[0]

    mdf = pd.DataFrame(metrics)
    C.write_parquet(mdf, C.D2_ANALYSIS_DIR / f"validation_metrics{sfx}.parquet", "metrics")

    # review queue: every mismatched (window,resource) pair (missed or spurious)
    disagree = merged[merged["_merge"] != "both"].copy()
    disagree["issue"] = disagree["_merge"].map({"left_only": "missed_by_pipeline",
                                                "right_only": "spurious_pipeline_determination"})
    C.write_csv(disagree.drop(columns=["_merge"]),
                C.D2_OUTPUT_DIR / f"validation_disagreements{sfx}.csv", "review queue")
    print("\n" + mdf.to_string(index=False))
    print(f"\nmismatched (window×resource) pairs: {len(disagree):,} "
          f"(missed {int((disagree['issue'] == 'missed_by_pipeline').sum()):,}, "
          f"spurious {int((disagree['issue'] == 'spurious_pipeline_determination').sum()):,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["fonsi", "eis"], default="fonsi",
                    help="which gold set / determinations to validate (default fonsi)")
    args = ap.parse_args()
    print(f"[track={args.track}]")
    main(args.track)
