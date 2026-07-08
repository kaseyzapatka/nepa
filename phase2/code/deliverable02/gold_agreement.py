"""D2 gold-set dual-labeler merge (Claude + Codex -> adjudicated gold).

Multi-determination grain (2026-07-08): each labeler writes a LONG CSV with one row per
(evidence_span_id x gold_resource_area) determination (see gold_labeling.md). This script aligns
the two on that composite key, so a resource one labeler found and the other missed shows up as a
disagreement (not silently dropped).

Flow:
  1. Both labelers write gold/labels_claude.csv and gold/labels_codex.csv (long form).
  2. This script (no flags): aligns on (evidence_span_id, gold_resource_area); per-field agreement
     over MATCHED keys; rows agreeing on the CORE fields (is_determination, class, mitigation_link)
     are auto-accepted; keys present in only one labeler (resource-set disagreement) + matched rows
     differing on core go to output/deliverable02/gold_disagreements.csv with empty final_* columns.
  3. Analyst fills final_* in that CSV, then: python gold_agreement.py --finalize
     -> gold/significance_gold.parquet (+ deterministic 30% holdout BY WINDOW) for 05_validate.

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

KEY = ["evidence_span_id", "gold_resource_area"]
# core fields that must match for a matched row to auto-accept (resource_area is part of the key)
CORE = ["gold_is_determination", "gold_determination_class", "gold_mitigation_link"]
ALL_GOLD = ["gold_is_determination", "gold_determination_class", "gold_determination_scope",
            "gold_primary_threshold_type", "gold_primary_threshold_status", "gold_mitigation_link",
            "gold_evidence_span_ok", "gold_needs_human_review", "gold_notes"]


def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().replace({"nan": "", "none": ""})


def _key_series(df: pd.DataFrame) -> pd.Series:
    return _norm(df["evidence_span_id"]) + "||" + _norm(df["gold_resource_area"])


def _load(path, labeler: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[missing] {path} — have {labeler} label per gold_labeling.md first.")
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in KEY:
        if col not in df.columns:
            sys.exit(f"{path}: long-form gold CSV needs a '{col}' column (see gold_labeling.md).")
    df["_key"] = _key_series(df)
    dups = df["_key"][df["_key"].duplicated()].tolist()
    if dups:
        sys.exit(f"{path}: duplicate (evidence_span_id, gold_resource_area) keys, e.g. {dups[:3]} "
                 f"— each resource determination must be a single row.")
    return df.set_index("_key")


def merge() -> None:
    cl, cx = _load(LABELS_CLAUDE, "Claude"), _load(LABELS_CODEX, "Codex")
    # window-level context for the disagreement sheet (join by evidence_span_id)
    queue = pd.read_parquet(C.D2_GOLD_DIR / "significance_gold_queue.parquet")
    ctx = queue.set_index("evidence_span_id")[["project_id", "heading_title", "page_start",
                                               "page_end", "evidence_text"]]

    keys_cl, keys_cx = set(cl.index), set(cx.index)
    both = sorted(keys_cl & keys_cx)
    only_cl = sorted(keys_cl - keys_cx)
    only_cx = sorted(keys_cx - keys_cl)
    print(f"determination rows — claude: {len(cl):,}  codex: {len(cx):,}")
    print(f"matched keys: {len(both):,}  |  claude-only: {len(only_cl):,}  |  "
          f"codex-only: {len(only_cx):,}")
    print("(claude-only / codex-only = resource-set disagreements: a resource one labeler coded "
          "and the other did not)")

    bi = pd.Index(both)
    print("\nper-field agreement (matched keys):")
    agree_all = pd.Series(True, index=bi)
    for f in CORE + ["gold_determination_scope", "gold_primary_threshold_type"]:
        a = (_norm(cl.loc[bi, f]) == _norm(cx.loc[bi, f])) if f in cl.columns and f in cx.columns \
            else pd.Series(False, index=bi)
        print(f"  {f:38s} {a.mean():6.1%}{'  [core]' if f in CORE else ''}")
        if f in CORE:
            agree_all &= a

    agreed_keys = bi[agree_all]
    disagree_keys = bi[~agree_all]
    print(f"\nmatched & agree on ALL core fields: {len(agreed_keys)}  |  "
          f"matched but differ: {len(disagree_keys)}")

    # ---- auto-accept: matched rows agreeing on core; keep Claude's full record ----
    agreed = cl.loc[agreed_keys, KEY + ALL_GOLD + ["labeler_confidence"]].copy()
    agreed = agreed.join(ctx["project_id"], on="evidence_span_id")
    agreed["gold_source"] = "both_agree"
    C.write_parquet(agreed.reset_index(drop=True), AGREED_PARQUET, "agreed gold")

    # ---- disagreements: matched-but-differ + every single-labeler key ----
    def _row(key, side_cl, side_cx, kind):
        span, res = key.split("||", 1)
        rec = {"evidence_span_id": span, "gold_resource_area": res, "disagreement_kind": kind}
        for f in ALL_GOLD:
            rec[f"claude_{f}"] = side_cl.get(f, "") if side_cl is not None else ""
            rec[f"codex_{f}"] = side_cx.get(f, "") if side_cx is not None else ""
            rec[f"final_{f}"] = ""     # analyst fills these (blank final_gold_is_determination = drop)
        return rec

    recs = []
    for k in disagree_keys:
        recs.append(_row(k, cl.loc[k], cx.loc[k], "matched_field_conflict"))
    for k in only_cl:
        recs.append(_row(k, cl.loc[k], None, "claude_only_resource"))
    for k in only_cx:
        recs.append(_row(k, None, cx.loc[k], "codex_only_resource"))
    dis = pd.DataFrame(recs)
    if not dis.empty:
        dis = dis.merge(ctx.reset_index()[["evidence_span_id", "project_id", "heading_title",
                                           "page_start", "page_end", "evidence_text"]],
                        on="evidence_span_id", how="left")
    C.write_csv(dis, DISAGREE_CSV, "ADJUDICATE: fill final_* columns (blank = drop that row)")
    print("\nNext: adjudicate the disagreements CSV, then re-run with --finalize.")


def finalize() -> None:
    agreed = pd.read_parquet(AGREED_PARQUET)
    dis = pd.read_csv(DISAGREE_CSV, dtype=str).fillna("")
    kept = dis[_norm(dis["final_gold_is_determination"]).ne("")].copy()
    print(f"adjudicated (kept): {len(kept)}/{len(dis)} disagreement rows "
          f"(blank final_gold_is_determination = dropped)")

    adj_cols = {}
    for f in ALL_GOLD:
        adj_cols[f] = kept[f"final_{f}"]
    adj = pd.DataFrame({"evidence_span_id": kept["evidence_span_id"],
                        "gold_resource_area": kept["gold_resource_area"], **adj_cols})
    adj["labeler_confidence"] = "adjudicated"
    adj["gold_source"] = "human_adjudicated"

    gold = pd.concat([agreed, adj], ignore_index=True)
    gold = gold.drop_duplicates(subset=KEY, keep="first").reset_index(drop=True)
    # deterministic ~30% holdout BY WINDOW (whole window in or out; stable across reruns, no RNG)
    gold["holdout"] = gold["evidence_span_id"].map(
        lambda s: int(C.sha256_text(s)[:8], 16) % 10 < 3)
    gold["double_coded"] = True   # dual-labeler design: every key was independently coded twice
    gold["gold_run_at"] = C.utc_now()
    gold["schema_version"] = C.SCHEMA_VERSION
    C.write_parquet(gold, C.GOLD, "FINAL GOLD (one row per window x resource)")
    n_win = gold["evidence_span_id"].nunique()
    print(f"gold rows: {len(gold):,} across {n_win:,} windows  |  "
          f"holdout rows: {int(gold['holdout'].sum())} "
          f"({gold.loc[gold['holdout'], 'evidence_span_id'].nunique()} windows)")
    print("Gold ready — run 05_validate_significance.py after the LLM pass.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--finalize", action="store_true")
    args = ap.parse_args()
    finalize() if args.finalize else merge()
