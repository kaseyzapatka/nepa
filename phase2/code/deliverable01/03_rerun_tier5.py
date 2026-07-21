"""Standalone Tier 5 replay for D1 — re-adjudicate unknowns WITHOUT re-running tiers 0-4.

Reads the persisted Tier 5 queue (tier5_queue.parquet, written by every run of
01_extract_nepa_trigger.py since 2026-07-20) plus the canonical trigger parquet,
re-runs the Claude Haiku adjudication for queue rows whose current classification
is still `unknown` (or the full queue with --all-queue), splices the results, and
recomputes the derived multi-label columns.

The heavy lifting (prompt construction, API call, result validation, hierarchy
resolution) is imported from 01_extract_nepa_trigger.py — this script contains no
duplicated classification logic.

Run (billable; requires ANTHROPIC_API_KEY):
  conda run -n nepa python phase2/code/deliverable01/03_rerun_tier5.py
Options:
  --dry-run     load everything, report what WOULD be sent, make no API calls (key-free)
  --all-queue   re-adjudicate every queue row, not just current unknowns
  --limit N     cap the number of projects sent (testing)

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


def _recompute_derived(final: pd.DataFrame) -> pd.DataFrame:
    """Mirror the derived-column assembly from the main pipeline (idempotent)."""
    final = final.copy()
    final["is_dual_nexus"] = (
        (final["nepa_trigger_primary"] == "federal_land")
        & final["nepa_trigger_secondary"].apply(
            lambda x: "federal_permit" in x if isinstance(x, list) else False
        )
    )

    hierarchy = _mod.TRIGGER_HIERARCHY

    def _sorted_multi(classes):
        ranked = {c: hierarchy.index(c) if c in hierarchy else 99 for c in classes}
        return sorted(classes, key=ranked.__getitem__)

    final["nepa_trigger_count"] = final["nepa_trigger_multi"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )
    final["nepa_trigger_combo"] = final["nepa_trigger_multi"].apply(
        lambda x: "|".join(_sorted_multi(x)) if isinstance(x, list) and len(x) else ""
    )
    final["nepa_trigger_primary_hierarchy"] = final["nepa_trigger_multi"].apply(
        lambda x: _mod._hierarchy_primary(x) if isinstance(x, list) else "unknown"
    )
    return final


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone Tier 5 replay (see module docstring)")
    ap.add_argument("--dry-run", action="store_true", help="No API calls; report the send-set")
    ap.add_argument("--all-queue", action="store_true", help="Replay the full queue, not just unknowns")
    ap.add_argument("--limit", type=int, default=None, help="Cap number of projects sent")
    args = ap.parse_args()

    if not QUEUE_PATH.exists():
        raise SystemExit(f"No Tier 5 queue at {QUEUE_PATH} — run the full pipeline once first.")
    queue = pd.read_parquet(QUEUE_PATH)
    canon = pd.read_parquet(CANON_PATH)
    print(f"queue: {len(queue):,} rows | canonical: {len(canon):,} rows")

    unknown_ids = set(canon.loc[canon["nepa_trigger_primary"] == "unknown", "project_id"])
    send = queue if args.all_queue else queue[queue["project_id"].isin(unknown_ids)]
    if args.limit:
        send = send.head(args.limit)
    est = len(send) * _mod.ESTIMATED_TIER5_COST_PER_PROJECT
    print(f"to adjudicate: {len(send):,} projects (canonical unknowns: {len(unknown_ids):,}) "
          f"| est. spend ~${est:.2f}")

    if args.dry_run:
        print("dry-run: no API calls made.")
        return
    if send.empty:
        print("nothing to adjudicate.")
        return

    projects_cols = send[["project_id", "project_title", "lead_agency_harmonized",
                          "dataset_source"]].copy()
    results = _mod.tier5_llm(send, projects_cols)
    print(f"tier5_llm returned {len(results)} results")
    if not results:
        raise SystemExit("No results returned — canonical parquet left untouched.")

    res_df = pd.DataFrame(results)
    res_df["nepa_trigger_extraction_run_at"] = canon["nepa_trigger_extraction_run_at"].iloc[0]
    # Audit trail: replayed successes carry a fresh nepa_trigger_llm_run_at (set inside
    # tier5_llm); the full pre-replay state is preserved in the timestamped backup written
    # below. The canonical schema is deliberately unchanged.
    replay_at = datetime.now(timezone.utc).isoformat()

    keep_cols = [c for c in canon.columns if c in res_df.columns]
    merged = pd.concat(
        [canon[~canon["project_id"].isin(set(res_df["project_id"]))],
         res_df[keep_cols]],
        ignore_index=True,
    )
    merged = _recompute_derived(merged)
    merged = merged[list(canon.columns)]

    assert len(merged) == len(canon), "row count changed during splice"
    assert merged["project_id"].is_unique, "duplicate project_ids after splice"
    changed = (merged.set_index("project_id")["nepa_trigger_primary"]
               != canon.set_index("project_id")["nepa_trigger_primary"]).sum()

    backup = CANON_PATH.with_name(
        CANON_PATH.stem + f"_pre_replay_{datetime.now().strftime('%Y%m%d_%H%M')}.parquet"
    )
    canon.to_parquet(backup, index=False)
    merged.to_parquet(CANON_PATH, index=False)
    n_unknown = (merged["nepa_trigger_primary"] == "unknown").sum()
    print(f"replay {replay_at}: {changed} primary classifications changed; "
          f"unknown now {n_unknown:,}. Backup: {backup.name}")
    print("Now regenerate the sidecar: conda run -n nepa python "
          "phase2/code/deliverable01/01_extract_nepa_trigger.py --funding-details-only")


if __name__ == "__main__":
    main()
