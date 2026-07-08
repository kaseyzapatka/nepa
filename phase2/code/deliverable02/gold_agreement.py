"""D2 gold-set dual-labeler merge (Claude + Codex -> adjudicated gold).

Flow (see gold_labeling.md):
  1. Both labelers write gold/labels_claude.csv and gold/labels_codex.csv.
  2. This script (no flags): per-field agreement report; rows agreeing on the CORE fields
     (is_determination, class, resource_area, mitigation_link) are auto-accepted; the rest go
     to output/deliverable02/gold_disagreements.csv with empty final_* columns for the analyst.
  3. Analyst fills final_* in that CSV, then: python gold_agreement.py --finalize
     -> gold/significance_gold.parquet (+ deterministic 30% holdout) for 05_validate.

Run:  conda run -n nepa python phase2/code/deliverable02/gold_agreement.py [--finalize]
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

import common as C

LABELS_CLAUDE = C.D2_GOLD_DIR / "labels_claude.csv"
LABELS_CODEX = C.D2_GOLD_DIR / "labels_codex.csv"
DISAGREE_CSV = C.D2_OUTPUT_DIR / "gold_disagreements.csv"
AGREED_PARQUET = C.D2_GOLD_DIR / "gold_agreed.parquet"

CORE = ["gold_is_determination", "gold_determination_class",
        "gold_resource_area", "gold_mitigation_link"]
ALL_GOLD = CORE + ["gold_determination_scope", "gold_primary_threshold_type",
                   "gold_primary_threshold_status", "gold_evidence_span_ok",
                   "gold_needs_human_review", "gold_notes"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})


def _load(path, labeler: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[missing] {path} — have {labeler} label per gold_labeling.md first.")
    df = pd.read_csv(path)
    if "evidence_span_id" not in df.columns:
        sys.exit(f"{path}: needs an evidence_span_id column.")
    return df.set_index("evidence_span_id")


def merge() -> None:
    cl, cx = _load(LABELS_CLAUDE, "Claude"), _load(LABELS_CODEX, "Codex")
    queue = pd.read_parquet(C.D2_GOLD_DIR / "significance_gold_queue.parquet") \
        .set_index("evidence_span_id")
    # the queue ships with EMPTY gold_*/bookkeeping placeholder columns — drop so labels join cleanly
    queue = queue.drop(columns=[c for c in queue.columns
                                if c.startswith("gold_") or c in ("double_coded", "holdout")])
    ids = queue.index
    missing_cl, missing_cx = ids.difference(cl.index), ids.difference(cx.index)
    if len(missing_cl) or len(missing_cx):
        print(f"[warn] unlabeled rows — claude: {len(missing_cl)}, codex: {len(missing_cx)}")

    both = ids.intersection(cl.index).intersection(cx.index)
    print(f"rows labeled by both: {len(both)}/{len(ids)}")
    print("\nper-field agreement:")
    agree_all = pd.Series(True, index=both)
    for f in CORE + ["gold_determination_scope", "gold_primary_threshold_type"]:
        a = (_norm(cl.loc[both, f]) == _norm(cx.loc[both, f]))
        core_tag = " [core]" if f in CORE else ""
        print(f"  {f:38s} {a.mean():6.1%}{core_tag}")
        if f in CORE:
            agree_all &= a

    agreed_ids, disagree_ids = both[agree_all], both[~agree_all]
    print(f"\nagree on ALL core fields: {len(agreed_ids)}  |  disagreements: {len(disagree_ids)}")

    # auto-accept agreed rows (Claude's values == Codex's on core; keep Claude's full record,
    # but flag scope/threshold rows where the two differ on non-core fields)
    agreed = queue.loc[agreed_ids].join(cl.loc[agreed_ids, ALL_GOLD + ["labeler_confidence"]])
    agreed["gold_source"] = "both_agree"
    C.write_parquet(agreed.reset_index(), AGREED_PARQUET, "agreed gold")

    dis = queue.loc[disagree_ids, ["project_id", "heading_title", "page_start", "page_end",
                                   "evidence_text"]].copy()
    for f in ALL_GOLD:
        dis[f"claude_{f}"] = cl.loc[disagree_ids, f] if f in cl.columns else ""
        dis[f"codex_{f}"] = cx.loc[disagree_ids, f] if f in cx.columns else ""
    for f in ALL_GOLD:
        dis[f"final_{f}"] = ""      # analyst fills these
    C.write_csv(dis.reset_index(), DISAGREE_CSV, "ADJUDICATE: fill final_* columns")
    print("\nNext: adjudicate the disagreements CSV, then re-run with --finalize.")


def finalize() -> None:
    agreed = pd.read_parquet(AGREED_PARQUET).set_index("evidence_span_id")
    dis = pd.read_csv(DISAGREE_CSV).set_index("evidence_span_id")
    done = dis[_norm(dis["final_gold_is_determination"]).ne("")]
    print(f"adjudicated: {len(done)}/{len(dis)} disagreement rows")
    if len(done) < len(dis):
        print("[warn] unadjudicated disagreement rows are EXCLUDED from the gold set.")
    queue = pd.read_parquet(C.D2_GOLD_DIR / "significance_gold_queue.parquet") \
        .set_index("evidence_span_id")
    queue = queue.drop(columns=[c for c in queue.columns
                                if c.startswith("gold_") or c in ("double_coded", "holdout")])
    adj = queue.loc[done.index].copy()
    for f in ALL_GOLD:
        adj[f] = done[f"final_{f}"]
    adj["gold_source"] = "human_adjudicated"

    gold = pd.concat([agreed.assign(), adj]).reset_index()
    # deterministic ~30% holdout by hash (stable across reruns; no RNG per repo convention)
    gold["holdout"] = gold["evidence_span_id"].map(
        lambda s: int(C.sha256_text(s)[:8], 16) % 10 < 3)
    gold["double_coded"] = True   # dual-labeler design: every row was independently coded twice
    gold["gold_run_at"] = C.utc_now()
    gold["schema_version"] = C.SCHEMA_VERSION
    C.write_parquet(gold, C.GOLD, "FINAL GOLD")
    print(f"holdout rows: {int(gold['holdout'].sum())}/{len(gold)}")
    print("Gold ready — run 05_validate_significance.py after the LLM pass.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    finalize() if args.finalize else merge()
