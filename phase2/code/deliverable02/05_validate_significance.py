"""D2 Phase 5 — validate extraction against the hand-labeled gold set (plan v2.11 §7).

Gated on the gold set EXISTING. The analyst labels `output/deliverable02/significance_gold_queue.csv`
(the `gold_*` columns) and saves it as `gold/significance_gold.parquet`; per-threshold labels go in
`gold/significance_gold_thresholds.parquet`. This script then computes the tiered metrics:

  - binary candidate is_determination: precision / recall / F1 (needs the negative class)
  - determination_class macro-F1 over COMMON classes (min support)
  - shared_resource_area accuracy / F1 over common areas
  - mitigation-link (mitigation_flag) F1
  - threshold CHILD-TABLE precision / recall / F1 + per-threshold status accuracy
  - section/candidate coverage note

Reports overall AND on the >=30% holdout. Emits a disagreement review queue. Metrics with
inadequate support are reported descriptively, not as pass/fail (plan §7).

Run:  conda run -n nepa python phase2/code/deliverable02/05_validate_significance.py
"""
from __future__ import annotations

import sys

import pandas as pd

import common as C

MIN_SUPPORT = 10  # below this, report descriptively (no macro-F1)


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("1", "true", "yes", "y", "t"))


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


def main() -> None:
    if not C.GOLD.exists():
        print(f"[gold not found] {C.GOLD.relative_to(C.PHASE2)}")
        print("Label output/deliverable02/significance_gold_queue.csv (the gold_* columns) and "
              "save it as gold/significance_gold.parquet. Then re-run. See HANDOFF.md.")
        sys.exit(0)
    if not C.SIGNIFICANCE_DETERMINATIONS.exists():
        print("[determinations not found] run 02 first."); sys.exit(0)

    gold = C.q(f"SELECT * FROM read_parquet('{C.GOLD}')")
    det = C.q(f"""SELECT evidence_span_id, determination_class, shared_resource_area,
                         mitigation_flag, primary_threshold_type
                  FROM read_parquet('{C.SIGNIFICANCE_DETERMINATIONS}')""")
    j = gold.merge(det, on="evidence_span_id", how="left", suffixes=("_gold", "_pred"))
    j["_pred_is_det"] = j["determination_class"].fillna("not_a_determination") != "not_a_determination"
    j["_gold_is_det"] = _truthy(j["gold_is_determination"])
    print(f"gold rows={len(gold):,}  joined to a determination={int(j['determination_class'].notna().sum()):,}")

    def report(sub: pd.DataFrame, tag: str) -> list[dict]:
        out = []
        tp = int((sub["_pred_is_det"] & sub["_gold_is_det"]).sum())
        fp = int((sub["_pred_is_det"] & ~sub["_gold_is_det"]).sum())
        fn = int((~sub["_pred_is_det"] & sub["_gold_is_det"]).sum())
        cand = prf(tp, fp, fn)
        out.append({"metric": "candidate_is_determination", "scope": tag, **cand})
        det_rows = sub[sub["_gold_is_det"] & sub["determination_class"].notna()]
        if len(det_rows) >= MIN_SUPPORT:
            mf1, _ = macro_f1(det_rows["gold_determination_class"].astype(str),
                              det_rows["determination_class"].astype(str))
            out.append({"metric": "determination_class_macro_f1", "scope": tag, "f1": mf1,
                        "support": len(det_rows)})
            racc = float((det_rows["gold_resource_area"].astype(str) ==
                          det_rows["shared_resource_area"].astype(str)).mean())
            out.append({"metric": "resource_area_accuracy", "scope": tag,
                        "precision": round(racc, 3), "support": len(det_rows)})
            gmit, pmit = _truthy(det_rows["gold_mitigation_link"]), det_rows["mitigation_flag"].fillna(False)
            out.append({"metric": "mitigation_flag_f1", "scope": tag,
                        **prf(int((gmit & pmit).sum()), int((~gmit & pmit).sum()), int((gmit & ~pmit).sum()))})
        else:
            out.append({"metric": "determination_class_macro_f1", "scope": tag,
                        "note": f"support {len(det_rows)} < {MIN_SUPPORT} — descriptive only"})
        return out

    metrics = report(j, "overall")
    if "holdout" in j.columns and _truthy(j["holdout"]).any():
        metrics += report(j[_truthy(j["holdout"])], "holdout")

    # threshold child metrics (if the gold companion exists)
    if C.GOLD_THRESHOLDS.exists() and C.DETERMINATION_THRESHOLDS.exists():
        gt = C.q(f"SELECT determination_instance_id, threshold_type FROM read_parquet('{C.GOLD_THRESHOLDS}')")
        pt = C.q(f"SELECT determination_instance_id, threshold_type FROM read_parquet('{C.DETERMINATION_THRESHOLDS}')")
        gset = set(map(tuple, gt.values)); pset = set(map(tuple, pt.values))
        tp = len(gset & pset)
        metrics.append({"metric": "threshold_child_prf", "scope": "overall",
                        **prf(tp, len(pset - gset), len(gset - pset))})
    else:
        metrics.append({"metric": "threshold_child_prf", "scope": "overall",
                        "note": "no significance_gold_thresholds.parquet yet"})

    mdf = pd.DataFrame(metrics)
    C.write_parquet(mdf, C.D2_ANALYSIS_DIR / "validation_metrics.parquet", "metrics")
    disagree = j[j["_pred_is_det"] != j["_gold_is_det"]]
    C.write_csv(disagree, C.D2_OUTPUT_DIR / "validation_disagreements.csv", "review queue")
    print("\n" + mdf.to_string(index=False))
    print(f"\ndisagreements (pred vs gold is_determination): {len(disagree):,}")


if __name__ == "__main__":
    main()
