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

--track {fonsi,eis} (default fonsi) switches ALL paths to the parallel EIS gold set
(labels_eis_*.csv, significance_gold_queue_eis.parquet, significance_gold_eis.parquet, ...) per
gold_labeling_eis.md — the FONSI default behavior is unchanged.

Run:  conda run -n nepa python phase2/code/deliverable02/gold_agreement.py [--track eis] [--finalize]
      [--stats-only] recompute agreement without writing; [--force] overwrite an adjudicated
      worksheet (timestamped backup written first)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone

import pandas as pd

import common as C


def _paths(track: str) -> dict:
    """Resolve the DISTINCT file set for a track; fonsi = the original (unchanged) paths."""
    if track == "eis":
        return {"labels_claude": C.D2_GOLD_DIR / "labels_eis_claude.csv",
                "labels_codex": C.D2_GOLD_DIR / "labels_eis_codex.csv",
                "queue": C.GOLD_QUEUE_EIS,
                "agreed": C.D2_GOLD_DIR / "gold_agreed_eis.parquet",
                "disagree": C.D2_OUTPUT_DIR / "gold_disagreements_eis.csv",
                "gold": C.GOLD_EIS, "prompt": "gold_labeling_eis.md"}
    return {"labels_claude": C.D2_GOLD_DIR / "labels_claude.csv",
            "labels_codex": C.D2_GOLD_DIR / "labels_codex.csv",
            "queue": C.GOLD_QUEUE,
            "agreed": C.D2_GOLD_DIR / "gold_agreed.parquet",
            "disagree": C.D2_OUTPUT_DIR / "gold_disagreements.csv",
            "gold": C.GOLD, "prompt": "gold_labeling.md"}


KEY = ["evidence_span_id", "gold_resource_area"]
# core fields that must match for a matched row to auto-accept (resource_area is part of the key)
CORE = ["gold_is_determination", "gold_determination_class", "gold_mitigation_link"]
ALL_GOLD = ["gold_is_determination", "gold_determination_class", "gold_determination_scope",
            "gold_primary_threshold_type", "gold_primary_threshold_status", "gold_mitigation_link",
            "gold_evidence_span_ok", "gold_needs_human_review", "gold_notes"]


def _norm_key(s: pd.Series) -> pd.Series:
    # key/identity normalization: collapse case + space/dash but PRESERVE the vocab token "none"
    # (the junk-row resource marker) so it stays self-documenting in the stored gold.
    return (s.astype(str).str.strip().str.lower()
            .str.replace(" ", "_", regex=False).str.replace("-", "_", regex=False))


def _norm(s: pd.Series) -> pd.Series:
    # field-VALUE normalization for agreement comparison: like _norm_key, but also folds the
    # empties nan/none -> "" (a "none" threshold and a blank mean the same when comparing fields).
    return _norm_key(s).replace({"nan": "", "none": ""})


def _key_series(df: pd.DataFrame) -> pd.Series:
    return _norm_key(df["evidence_span_id"]) + "||" + _norm_key(df["gold_resource_area"])


def _load(path, labeler: str) -> pd.DataFrame:
    if not path.exists():
        sys.exit(f"[missing] {path} — have {labeler} label per gold_labeling.md first.")
    df = pd.read_csv(path, dtype=str).fillna("")
    for col in KEY:
        if col not in df.columns:
            sys.exit(f"{path}: long-form gold CSV needs a '{col}' column (see gold_labeling.md).")
    # canonicalize the key columns IN PLACE so the auto-accept path (reads these columns raw) and
    # the disagreement/finalize path (derives them from the normalized _key) store identical values.
    df["evidence_span_id"] = _norm_key(df["evidence_span_id"])
    df["gold_resource_area"] = _norm_key(df["gold_resource_area"])   # preserves "none" for junk rows
    df["_key"] = _key_series(df)
    dups = df["_key"][df["_key"].duplicated()].tolist()
    if dups:
        sys.exit(f"{path}: duplicate (evidence_span_id, gold_resource_area) keys, e.g. {dups[:3]} "
                 f"— each resource determination must be a single row.")
    return df.set_index("_key")


def _guard_worksheet(p: dict, force: bool) -> None:
    """Refuse to overwrite an adjudicated worksheet unless --force (which snapshots it first).
    The disagreements CSV is the one artifact this script writes that the analyst then hand-fills
    (the final_* columns); a bare re-run of the merge step would silently erase that hand work
    — as happened 2026-07-15, recovered from Time Machine."""
    path = p["disagree"]
    if not path.exists():
        return
    try:
        old = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return
    fin = [c for c in old.columns if c.startswith("final_")]
    if not fin or not len(old):
        return
    n_filled = int(old[fin].apply(lambda s: s.str.strip().ne("")).any(axis=1).sum())
    if not n_filled:
        return
    if not force:
        sys.exit(f"[guard] {path.name} holds {n_filled} hand-adjudicated rows (non-empty final_* "
                 f"cells); re-running the merge would erase them. Use --stats-only to recompute "
                 f"agreement without writing, or --force to overwrite (timestamped backup kept).")
    bak = path.with_name(f"{path.stem}.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.bak.csv")
    shutil.copy2(path, bak)
    print(f"[guard] --force: adjudicated worksheet backed up to {bak.name}")


def merge(p: dict, stats_only: bool = False) -> None:
    cl, cx = _load(p["labels_claude"], "Claude"), _load(p["labels_codex"], "Codex")
    # window-level context for the disagreement sheet (join by evidence_span_id)
    queue = pd.read_parquet(p["queue"])
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

    if stats_only:
        print("\n[stats-only] agreement recomputed; nothing written.")
        return

    # ---- auto-accept: matched rows agreeing on core; keep Claude's full record ----
    agreed = cl.loc[agreed_keys, KEY + ALL_GOLD + ["labeler_confidence"]].copy()
    agreed = agreed.join(ctx["project_id"], on="evidence_span_id")
    agreed["gold_source"] = "both_agree"
    C.write_parquet(agreed.reset_index(drop=True), p["agreed"], "agreed gold")

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
    else:  # zero disagreements: still write a header-bearing CSV so --finalize reads it cleanly
        cols = (["evidence_span_id", "gold_resource_area", "disagreement_kind"]
                + [f"{pfx}_{f}" for f in ALL_GOLD for pfx in ("claude", "codex", "final")]
                + ["project_id", "heading_title", "page_start", "page_end", "evidence_text"])
        dis = pd.DataFrame(columns=cols)
    C.write_csv(dis, p["disagree"], "ADJUDICATE: fill final_* columns (blank = drop that row)")
    print("\nNext: adjudicate the disagreements CSV, then re-run with --finalize.")


def finalize(p: dict) -> None:
    agreed = pd.read_parquet(p["agreed"])
    try:
        dis = pd.read_csv(p["disagree"], dtype=str).fillna("")
    except pd.errors.EmptyDataError:      # truly empty file (no header) -> no disagreements
        dis = pd.DataFrame(columns=["evidence_span_id", "gold_resource_area"]
                           + [f"final_{f}" for f in ALL_GOLD])
    kept = (dis[_norm(dis["final_gold_is_determination"]).ne("")].copy()
            if "final_gold_is_determination" in dis.columns and len(dis) else dis.iloc[0:0])
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
    C.write_parquet(gold, p["gold"], "FINAL GOLD (one row per window x resource)")
    n_win = gold["evidence_span_id"].nunique()
    print(f"gold rows: {len(gold):,} across {n_win:,} windows  |  "
          f"holdout rows: {int(gold['holdout'].sum())} "
          f"({gold.loc[gold['holdout'], 'evidence_span_id'].nunique()} windows)")
    print("Gold ready — run 05_validate_significance.py after the LLM pass.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", choices=["fonsi", "eis"], default="fonsi",
                    help="which gold set (default fonsi; eis = the parallel EIS gold, distinct files)")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute + print agreement stats without writing any file (safe for "
                         "verification once the worksheet is adjudicated)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an adjudicated disagreements worksheet (a timestamped .bak "
                         "copy is written first)")
    args = ap.parse_args()
    p = _paths(args.track)
    print(f"[track={args.track}]  gold -> {p['gold'].name}")
    if args.finalize:
        finalize(p)
    else:
        if not args.stats_only:
            _guard_worksheet(p, args.force)
        merge(p, stats_only=args.stats_only)
