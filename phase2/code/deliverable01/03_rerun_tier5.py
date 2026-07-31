"""Standalone Tier 5 replay for D1 — re-adjudicate unknowns WITHOUT re-running tiers 0-4.

Two modes:

1. Live (default) — reads the persisted Tier 5 queue (tier5_queue.parquet) plus the
   canonical trigger parquet, re-runs the Claude Haiku adjudication for queue rows whose
   current classification is still `unknown` (or the full queue with --all-queue), splices
   the results, and recomputes the derived multi-label columns. Billable.

2. Record replay (--from-record PATH) — reads the committed Tier 5 adjudication record
   (tier5_adjudication_record.csv, the frozen RAW LLM verdicts) and rebuilds the published
   Tier 5 classifications deterministically, WITHOUT any API call. This is how a reviewer
   reproduces the exact published D1 output: the raw verdicts are re-materialized and the
   same ingest-time hierarchy reconciliation (_reconcile_to_hierarchy) is re-applied, so
   primary == primary_hierarchy for every replayed row.

The heavy lifting (prompt construction, API call, result validation, hierarchy
reconciliation) is imported from 01_extract_nepa_trigger.py — this script contains no
duplicated classification logic.

Run (live; billable; requires ANTHROPIC_API_KEY):
  conda run -n nepa python phase2/code/deliverable01/03_rerun_tier5.py
Run (record replay; deterministic, NO API):
  conda run -n nepa python phase2/code/deliverable01/03_rerun_tier5.py --from-record \
      phase2/code/deliverable01/tier5_adjudication_record.csv
Options:
  --dry-run     load everything, report what WOULD change, make no writes (works in both modes)
  --from-record PATH   deterministic replay from the committed record (no API calls)
  --all-queue   (live only) re-adjudicate every queue row, not just current unknowns
  --limit N     (live only) cap the number of projects sent (testing)

After a live run, regenerate the funding sidecar:
  conda run -n nepa python phase2/code/deliverable01/01_extract_nepa_trigger.py --funding-details-only
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

# 01_extract_nepa_trigger.py starts with a digit — import via importlib.
_spec = importlib.util.spec_from_file_location(
    "extract_nepa_trigger", HERE / "01_extract_nepa_trigger.py"
)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["extract_nepa_trigger"] = _mod
_spec.loader.exec_module(_mod)  # runs the module's conda-env guard too

QUEUE_PATH = _mod.TIER5_QUEUE_PATH
CANON_PATH = _mod.PROJECTS_NEPA_TRIGGER_PATH


def _as_list(x) -> list:
    """Normalize list-like cells: parquet round-trips list<string> columns as numpy
    arrays, which fail `isinstance(x, list)` checks. None/NaN → []."""
    if x is None or isinstance(x, float):
        return []
    if isinstance(x, list):
        return x
    return list(x)


def _recompute_derived(final: pd.DataFrame) -> pd.DataFrame:
    """Mirror the derived-column assembly from the main pipeline (idempotent)."""
    final = final.copy()
    for col in ("nepa_trigger_secondary", "nepa_trigger_multi"):
        final[col] = final[col].apply(_as_list)

    final["is_dual_nexus"] = (
        (final["nepa_trigger_primary"] == "federal_land")
        & final["nepa_trigger_secondary"].apply(lambda x: "federal_permit" in x)
    )

    hierarchy = _mod.TRIGGER_HIERARCHY

    def _sorted_multi(classes):
        ranked = {c: hierarchy.index(c) if c in hierarchy else 99 for c in classes}
        return sorted(classes, key=ranked.__getitem__)

    final["nepa_trigger_count"] = final["nepa_trigger_multi"].apply(len)
    final["nepa_trigger_combo"] = final["nepa_trigger_multi"].apply(
        lambda x: "|".join(_sorted_multi(x)) if len(x) else ""
    )
    final["nepa_trigger_primary_hierarchy"] = final["nepa_trigger_multi"].apply(
        lambda x: _mod._hierarchy_primary(x) if len(x) else "unknown"
    )
    return final


def _splice_and_write(canon: pd.DataFrame, results: list[dict], source_label: str,
                      dry_run: bool) -> None:
    """Splice Tier 5 results onto the canonical parquet, recompute derived columns, and
    (unless dry-run) write a timestamped pre-replay backup followed by the new parquet.

    Only the project_ids present in `results` may change; row order is preserved."""
    if not results:
        raise SystemExit("No results to splice — canonical parquet left untouched.")

    res_df = pd.DataFrame(results)
    res_df["nepa_trigger_extraction_run_at"] = canon["nepa_trigger_extraction_run_at"].iloc[0]
    replay_at = datetime.now(timezone.utc).isoformat()

    keep_cols = [c for c in canon.columns if c in res_df.columns]
    order = canon["project_id"].tolist()
    changed_ids = set(res_df["project_id"])
    merged = pd.concat(
        [canon[~canon["project_id"].isin(changed_ids)], res_df[keep_cols]],
        ignore_index=True,
    )
    merged = _recompute_derived(merged)
    merged = merged[list(canon.columns)]
    # Preserve the canonical row order so the only diffs vs the backup are the changed rows.
    merged = merged.set_index("project_id").loc[order].reset_index()

    assert len(merged) == len(canon), "row count changed during splice"
    assert merged["project_id"].is_unique, "duplicate project_ids after splice"

    canon_pri = canon.set_index("project_id")["nepa_trigger_primary"]
    merged_pri = merged.set_index("project_id")["nepa_trigger_primary"]
    changed = int((merged_pri != canon_pri).sum())
    # Every primary change must be confined to the replayed project_ids.
    changed_pids = set(merged_pri.index[(merged_pri != canon_pri).values])
    assert changed_pids <= changed_ids, "primary changed outside the replayed project_ids"
    # After reconciliation, primary must equal the hierarchy-resolved primary everywhere.
    mismatch = int((merged["nepa_trigger_primary"] != merged["nepa_trigger_primary_hierarchy"]).sum())

    n_unknown = int((merged["nepa_trigger_primary"] == "unknown").sum())
    n_dual = int(merged["is_dual_nexus"].sum())
    print(f"[{source_label}] {changed} primary classifications changed vs current parquet; "
          f"primary!=primary_hierarchy: {mismatch}; unknown: {n_unknown:,}; "
          f"is_dual_nexus: {n_dual:,}; rows: {len(merged):,}")

    if dry_run:
        print("dry-run: no files written.")
        return

    backup = CANON_PATH.with_name(
        CANON_PATH.stem + f"_pre_replay_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
    )
    canon.to_parquet(backup, index=False)
    merged.to_parquet(CANON_PATH, index=False)
    print(f"replay {replay_at}: wrote {CANON_PATH.name}; backup: {backup.name}")
    print("Now regenerate the sidecar: conda run -n nepa python "
          "phase2/code/deliverable01/01_extract_nepa_trigger.py --funding-details-only")


def replay_from_record(record_path: Path, dry_run: bool) -> None:
    """Deterministically rebuild the published Tier 5 classifications from the committed
    raw-verdict record — no API calls. Re-applies the same hierarchy reconciliation as the
    live ingest path so the output matches the published parquet exactly."""
    if not record_path.exists():
        raise SystemExit(f"No adjudication record at {record_path}.")
    canon = pd.read_parquet(CANON_PATH)
    rec = pd.read_csv(record_path, dtype=str, keep_default_na=False)
    print(f"record: {len(rec):,} rows | canonical: {len(canon):,} rows")

    results: list[dict] = []
    for _, r in rec.iterrows():
        confidence = r["confidence"] or "medium"
        raw_primary = r["llm_primary"]
        raw_secondary = [c for c in r["llm_secondary"].split("|") if c]
        primary, secondary = _mod._reconcile_to_hierarchy(raw_primary, raw_secondary)
        res = _mod.make_result(
            project_id=r["project_id"],
            primary=primary,
            confidence=confidence,
            evidence_text=r["evidence_text"],
            evidence_source="llm",
            rule_id=r["rule_id"],
            secondary=secondary,
            manual_review=(confidence == "low"),
            route_policy="llm",
            route_reason=(
                f"replay-from-record | llm_raw_primary={raw_primary} "
                f"llm_raw_secondary={r['llm_secondary']}"
            ),
        )
        # Preserve the original per-row LLM timestamp verbatim (deterministic replay).
        res["nepa_trigger_llm_run_at"] = r["llm_run_at"]
        results.append(res)

    print(f"rebuilt {len(results):,} Tier 5 results from record (no API calls)")
    _splice_and_write(canon, results, source_label="from-record", dry_run=dry_run)


def replay_live(all_queue: bool, limit: int | None, dry_run: bool) -> None:
    if not QUEUE_PATH.exists():
        raise SystemExit(f"No Tier 5 queue at {QUEUE_PATH} — run the full pipeline once first.")
    queue = pd.read_parquet(QUEUE_PATH)
    canon = pd.read_parquet(CANON_PATH)
    print(f"queue: {len(queue):,} rows | canonical: {len(canon):,} rows")

    unknown_ids = set(canon.loc[canon["nepa_trigger_primary"] == "unknown", "project_id"])
    send = queue if all_queue else queue[queue["project_id"].isin(unknown_ids)]
    if limit:
        send = send.head(limit)
    est = len(send) * _mod.ESTIMATED_TIER5_COST_PER_PROJECT
    print(f"to adjudicate: {len(send):,} projects (canonical unknowns: {len(unknown_ids):,}) "
          f"| est. spend ~${est:.2f}")

    if dry_run:
        print("dry-run: no API calls made.")
        return
    if send.empty:
        print("nothing to adjudicate.")
        return

    projects_cols = send[["project_id", "project_title", "lead_agency_harmonized",
                          "dataset_source"]].copy()
    results = _mod.tier5_llm(send, projects_cols)
    print(f"tier5_llm returned {len(results)} results")
    _splice_and_write(canon, results, source_label="live", dry_run=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone Tier 5 replay (see module docstring)")
    ap.add_argument("--dry-run", action="store_true", help="No writes/API; report what WOULD change")
    ap.add_argument("--from-record", type=str, default=None,
                    help="Deterministic replay from the committed adjudication record CSV (no API)")
    ap.add_argument("--all-queue", action="store_true", help="(live) Replay the full queue, not just unknowns")
    ap.add_argument("--limit", type=int, default=None, help="(live) Cap number of projects sent")
    args = ap.parse_args()

    if args.from_record:
        replay_from_record(Path(args.from_record), dry_run=args.dry_run)
    else:
        replay_live(all_queue=args.all_queue, limit=args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
