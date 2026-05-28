"""
Prepare gold-standard timeline review packets.

Creates one project-level CSV and one candidate-level CSV per review batch.
The project CSV is for final initiation/decision labels. The candidate CSV is
for candidate-role labels used in training/error analysis.

Outputs:
    phase2/output/deliverable04/gold/review_packets/<split>_batchNNN_projects.csv
    phase2/output/deliverable04/gold/review_packets/<split>_batchNNN_candidates.csv

Usage:
    python 11_prepare_gold_review_packets.py --split diagnostic_balanced_v2
    python 11_prepare_gold_review_packets.py --split train_enriched_v1 --batch 1
    python 11_prepare_gold_review_packets.py --all
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
GOLD_DIR = TIMELINE_DIR / "gold"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"
PACKET_DIR = OUTPUT_DIR / "gold" / "review_packets"

SPLITS_PATH = GOLD_DIR / "timeline_gold_splits.parquet"
PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"

DEFAULT_BATCH_SIZE = 50
DEFAULT_CAPS = {"CE": 10, "EA": 14, "EIS": 16}
ROLE_PRIORITY = {
    "clear_decision": 0,
    "clear_initiation": 1,
    "proxy_decision": 2,
    "proxy_initiation": 3,
    "review": 4,
    "unknown": 5,
    "historical": 6,
    "reject": 7,
}

GOLD_PROJECT_COLUMNS = [
    "gold_initiation_date",
    "gold_initiation_granularity",
    "gold_initiation_type",
    "gold_initiation_source_type",
    "gold_initiation_candidate_id",
    "gold_initiation_document_id",
    "gold_initiation_page_number",
    "gold_initiation_evidence_text",
    "gold_initiation_confidence",
    "gold_initiation_missing_reason",
    "gold_decision_date",
    "gold_decision_granularity",
    "gold_decision_type",
    "gold_decision_source_type",
    "gold_decision_candidate_id",
    "gold_decision_document_id",
    "gold_decision_page_number",
    "gold_decision_evidence_text",
    "gold_decision_confidence",
    "gold_decision_missing_reason",
    "gold_ambiguity_flag",
    "gold_notes",
    "reviewer",
    "review_status",
    "reconciler",
    "adjudication_reason",
    "adjudicated_at",
]

GOLD_CANDIDATE_COLUMNS = [
    "gold_candidate_role",
    "gold_selected_for",
    "gold_error_category",
    "gold_candidate_notes",
]


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _truncate(text: object, n: int = 500) -> str:
    s = "" if text is None or pd.isna(text) else " ".join(str(text).split())
    return s if len(s) <= n else s[: n - 3].rstrip() + "..."


def _candidate_sort_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_role_order"] = out["candidate_role"].map(lambda x: ROLE_PRIORITY.get(str(x), 99))
    out["_score"] = pd.to_numeric(out.get("ranking_score", 0), errors="coerce").fillna(0)
    out["_selected"] = (
        out.get("selected_for_initiation", False).fillna(False).astype(bool)
        | out.get("selected_for_decision", False).fillna(False).astype(bool)
    ).astype(int)
    return out.sort_values(
        ["_selected", "_role_order", "_score", "parsed_date"],
        ascending=[False, True, False, True],
    )


def _candidate_display_caps(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return pd.DataFrame()
    counts = candidates_df.groupby(["process_type", "project_id"]).size().reset_index(name="n_candidates")
    rows = []
    for process_type, group in counts.groupby("process_type"):
        stats = group["n_candidates"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99])
        row = {"process_type": process_type}
        row.update({k: stats.get(k) for k in stats.index})
        rows.append(row)
    return pd.DataFrame(rows)


def _top_candidate_text(cands: pd.DataFrame, role_filter: list[str], n: int = 5) -> str:
    if cands.empty:
        return ""
    sub = cands[cands["candidate_role"].isin(role_filter)].copy()
    if sub.empty:
        return ""
    sub = sub.nlargest(n, "ranking_score")
    parts = []
    for _, row in sub.iterrows():
        parts.append(
            f"{row.get('candidate_id')} | {row.get('parsed_date')} "
            f"[{row.get('candidate_role')}|{row.get('role_confidence')}|"
            f"score={row.get('ranking_score', 0):.1f}] "
            f"doc={str(row.get('document_type_clean', ''))[:40]} "
            f"page={row.get('page_number')} | {_truncate(row.get('context_text'), 240)}"
        )
    return " ||| ".join(parts)


def _codex_suggestion_from_pipeline(row: pd.Series, role: str, hide: bool) -> dict:
    if hide:
        return {
            f"codex_{role}_date": "",
            f"codex_{role}_type": "",
            f"codex_{role}_candidate_id": "",
            f"codex_{role}_notes": "hidden_for_irr_blind_review",
        }
    date_val = row.get(f"{role}_date")
    is_proxy = bool(row.get(f"{role}_is_proxy", False))
    evidence = _truncate(row.get(f"{role}_evidence_text"), 240)
    return {
        f"codex_{role}_date": "" if pd.isna(date_val) else date_val,
        f"codex_{role}_type": "proxy" if is_proxy else ("clear" if pd.notna(date_val) else "missing"),
        f"codex_{role}_candidate_id": "",
        f"codex_{role}_notes": evidence,
    }


def _load_project_metadata(project_ids: set[str]) -> pd.DataFrame:
    cols = [
        "project_id", "project_title", "project_sector", "project_type",
        "project_sponsor", "project_location", "process_type",
        "project_energy_type", "lead_agency_harmonized", "project_department",
        "project_state", "project_county", "project_doc_count",
    ]
    projects = pd.read_parquet(PROJECTS_PATH)
    keep = [c for c in cols if c in projects.columns]
    return projects[projects["project_id"].isin(project_ids)][keep].copy()


def _build_project_packet(
    split_rows: pd.DataFrame,
    projects: pd.DataFrame,
    dates: pd.DataFrame,
    candidates: pd.DataFrame,
    batch_id: str,
) -> pd.DataFrame:
    packet = split_rows.merge(projects, on=["project_id", "process_type", "project_energy_type"], how="left", suffixes=("", "_project"))
    if not dates.empty:
        packet = packet.merge(dates, on=["project_id", "process_type"], how="left", suffixes=("", "_pipeline"))
        for col in ["timeline_status", "timeline_flags"]:
            pipeline_col = f"{col}_pipeline"
            if pipeline_col in packet.columns:
                packet[col] = packet[pipeline_col].where(packet[pipeline_col].notna(), packet[col])

    project_candidate_groups = {
        pid: grp.copy()
        for pid, grp in candidates.groupby("project_id", sort=False)
    } if not candidates.empty else {}

    if not candidates.empty:
        candidate_counts = candidates.groupby("project_id").agg(
            n_candidates_current=("candidate_id", "nunique"),
            n_relevant_candidates_current=(
                "candidate_role",
                lambda s: s.isin([
                    "clear_initiation", "proxy_initiation",
                    "clear_decision", "proxy_decision",
                ]).sum(),
            ),
            n_unknown_candidates_current=("candidate_role", lambda s: (s == "unknown").sum()),
            n_historical_candidates_current=("candidate_role", lambda s: (s == "historical").sum()),
        )
        packet["n_candidates"] = packet["project_id"].map(candidate_counts["n_candidates_current"]).fillna(0).astype(int)
        packet["n_relevant_candidates"] = (
            packet["project_id"].map(candidate_counts["n_relevant_candidates_current"]).fillna(0).astype(int)
        )
        packet["n_unknown_candidates"] = (
            packet["project_id"].map(candidate_counts["n_unknown_candidates_current"]).fillna(0).astype(int)
        )
        packet["n_historical_candidates"] = (
            packet["project_id"].map(candidate_counts["n_historical_candidates_current"]).fillna(0).astype(int)
        )

    packet["batch_id"] = batch_id
    packet["review_pass"] = packet["irr_required"].map(lambda x: "primary_blind" if bool(x) else "primary")
    packet["hide_codex_suggestions"] = packet["irr_required"].astype(bool)
    packet["top_initiation_candidates"] = packet["project_id"].map(
        lambda pid: _top_candidate_text(project_candidate_groups.get(pid, pd.DataFrame()), ["clear_initiation", "proxy_initiation"])
    )
    packet["top_decision_candidates"] = packet["project_id"].map(
        lambda pid: _top_candidate_text(project_candidate_groups.get(pid, pd.DataFrame()), ["clear_decision", "proxy_decision"])
    )

    codex_rows = []
    for _, row in packet.iterrows():
        hide = bool(row.get("hide_codex_suggestions"))
        d = {}
        d.update(_codex_suggestion_from_pipeline(row, "initiation", hide))
        d.update(_codex_suggestion_from_pipeline(row, "decision", hide))
        d["codex_missing_reason"] = ""
        d["codex_notes"] = "pipeline_prefill" if not hide else "codex_final_dates_hidden_for_irr"
        codex_rows.append(d)
    packet = pd.concat([packet.reset_index(drop=True), pd.DataFrame(codex_rows)], axis=1)

    for col in GOLD_PROJECT_COLUMNS:
        if col not in packet.columns:
            packet[col] = ""

    preferred = [
        "batch_id", "split", "prior_split", "review_pass", "irr_required",
        "hide_codex_suggestions", "project_id", "process_type", "project_energy_type",
        "sample_weight", "sample_stratum", "workflow_condition", "timeline_status",
        "timeline_flags", "project_title", "lead_agency_harmonized",
        "project_department", "project_state", "project_county", "project_doc_count",
        "doc_count_bin", "total_pages", "max_document_pages", "appendix_count",
        "n_candidates", "n_relevant_candidates", "n_unknown_candidates",
        "n_historical_candidates", "n_packets", "max_packet_score",
        "initiation_date", "initiation_date_granularity", "initiation_source_type",
        "initiation_confidence", "initiation_is_proxy", "initiation_evidence_text",
        "initiation_document_id", "initiation_page_number",
        "decision_date", "decision_date_granularity", "decision_source_type",
        "decision_confidence", "decision_is_proxy", "decision_evidence_text",
        "decision_document_id", "decision_page_number", "duration_days",
        "codex_initiation_date", "codex_initiation_type",
        "codex_initiation_candidate_id", "codex_initiation_notes",
        "codex_decision_date", "codex_decision_type",
        "codex_decision_candidate_id", "codex_decision_notes",
        "codex_missing_reason", "codex_notes",
        "top_initiation_candidates", "top_decision_candidates",
    ] + GOLD_PROJECT_COLUMNS
    preferred = [c for c in preferred if c in packet.columns]
    return packet[preferred]


def _candidate_cap_for(process_type: str, args: argparse.Namespace) -> int:
    if process_type == "CE":
        return args.cap_ce
    if process_type == "EA":
        return args.cap_ea
    if process_type == "EIS":
        return args.cap_eis
    return DEFAULT_CAPS.get(process_type, 12)


def _select_candidates_for_review(candidates: pd.DataFrame, split_rows: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    selected_parts = []
    split_meta = split_rows.set_index("project_id")
    for pid, group in candidates.groupby("project_id", sort=False):
        process_type = str(group["process_type"].iloc[0])
        cap = _candidate_cap_for(process_type, args)
        sorted_group = _candidate_sort_key(group)
        selected_parts.append(sorted_group.head(cap))
    out = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    if out.empty:
        return out

    out["split"] = out["project_id"].map(split_meta["split"])
    out["batch_id"] = split_rows["batch_id"].iloc[0]
    out["sample_weight"] = out["project_id"].map(split_meta["sample_weight"])
    out["irr_required"] = out["project_id"].map(split_meta["irr_required"])
    out["review_pass"] = out["irr_required"].map(lambda x: "primary_blind" if bool(x) else "primary")
    out["codex_candidate_role"] = out["candidate_role"]
    out["codex_candidate_notes"] = out.apply(
        lambda r: f"rule={r.get('rule_ids')} score={r.get('ranking_score', 0):.1f} "
                  f"pos={r.get('positive_cue_flags', '')} neg={r.get('negative_cue_flags', '')}",
        axis=1,
    )
    for col in GOLD_CANDIDATE_COLUMNS:
        out[col] = ""

    preferred = [
        "batch_id", "split", "review_pass", "irr_required", "project_id",
        "process_type", "candidate_id", "parsed_date", "date_granularity",
        "raw_date_text", "candidate_role", "role_confidence",
        "role_confidence_score", "ranking_score", "selected_for_initiation",
        "selected_for_decision", "is_proxy", "source_tier", "retrieval_tier",
        "candidate_source_type", "document_id", "page_number", "section_id",
        "context_packet_id", "document_title", "file_name",
        "document_type_clean", "document_type_category", "main_document",
        "heading_title", "parent_heading_title", "positive_cue_flags",
        "negative_cue_flags", "rule_ids", "date_mention_count",
        "codex_candidate_role", "codex_candidate_notes", "context_text",
    ] + GOLD_CANDIDATE_COLUMNS
    preferred = [c for c in preferred if c in out.columns]
    return out[preferred]


def _batch_rows(splits: pd.DataFrame, split_name: str, batch_size: int, batch: int | None) -> list[tuple[int, pd.DataFrame]]:
    sub = splits[splits["split"] == split_name].copy()
    if sub.empty:
        raise ValueError(f"No rows found for split {split_name!r}")
    sub = sub.sort_values(["process_type", "project_energy_type", "workflow_condition", "project_id"]).reset_index(drop=True)
    sub["batch_number"] = (sub.index // batch_size) + 1
    if batch is not None:
        sub = sub[sub["batch_number"] == batch]
        if sub.empty:
            raise ValueError(f"No rows for split={split_name} batch={batch}")
    return [(int(b), g.drop(columns=["batch_number"]).copy()) for b, g in sub.groupby("batch_number", sort=True)]


def prepare_packets(args: argparse.Namespace) -> None:
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"Gold splits not found: {SPLITS_PATH}\nRun 10_build_gold_samples.py first.")

    splits = pd.read_parquet(SPLITS_PATH)
    splits_to_run = sorted(splits["split"].unique()) if args.all else [args.split]
    if not args.all and not args.split:
        raise ValueError("Specify --split or --all.")

    project_ids = set(splits[splits["split"].isin(splits_to_run)]["project_id"])
    projects = _load_project_metadata(project_ids)
    dates = _read_optional_parquet(DATES_PATH)
    candidates = _read_optional_parquet(CANDIDATES_PATH)
    if not candidates.empty:
        candidates = candidates[candidates["project_id"].isin(project_ids)].copy()
        print("Current candidate count percentiles by process:")
        print(_candidate_display_caps(candidates).to_string(index=False))

    PACKET_DIR.mkdir(parents=True, exist_ok=True)
    for split_name in splits_to_run:
        for batch_number, rows in _batch_rows(splits, split_name, args.batch_size, args.batch):
            batch_id = f"{split_name}_batch{batch_number:03d}"
            rows = rows.copy()
            rows["batch_id"] = batch_id
            batch_ids = set(rows["project_id"])
            project_packet = _build_project_packet(
                rows,
                projects[projects["project_id"].isin(batch_ids)],
                dates[dates["project_id"].isin(batch_ids)] if not dates.empty else pd.DataFrame(),
                candidates[candidates["project_id"].isin(batch_ids)] if not candidates.empty else pd.DataFrame(),
                batch_id,
            )
            candidate_packet = _select_candidates_for_review(
                candidates[candidates["project_id"].isin(batch_ids)] if not candidates.empty else pd.DataFrame(),
                rows,
                args,
            )

            project_path = PACKET_DIR / f"{batch_id}_projects.csv"
            candidate_path = PACKET_DIR / f"{batch_id}_candidates.csv"
            project_packet.to_csv(project_path, index=False)
            candidate_packet.to_csv(candidate_path, index=False)
            print(f"Wrote: {project_path} ({len(project_packet)} projects)")
            print(f"Wrote: {candidate_path} ({len(candidate_packet)} candidates)")

            if args.batch is not None:
                # Explicit batch mode writes only that batch.
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare D4 gold review packets.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--split", choices=["diagnostic_balanced_v2", "train_enriched_v1", "test_representative_v1"])
    group.add_argument("--all", action="store_true", help="Prepare packets for all splits.")
    parser.add_argument("--batch", type=int, help="Only write one batch number for the selected split.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cap-ce", type=int, default=DEFAULT_CAPS["CE"])
    parser.add_argument("--cap-ea", type=int, default=DEFAULT_CAPS["EA"])
    parser.add_argument("--cap-eis", type=int, default=DEFAULT_CAPS["EIS"])
    args = parser.parse_args()

    prepare_packets(args)


if __name__ == "__main__":
    main()
