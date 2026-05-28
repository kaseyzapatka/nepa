"""
Validate timeline extraction against the 100-project gold sample (D4).

Modes:
  --prepare-review   Join pipeline outputs to sample, write annotatable review packet.
  --validate         Compare gold labels to pipeline dates, write validation reports.
                     (Default when timeline_sample100.csv has gold labels filled.)

Inputs:
    phase2/output/deliverable04/timeline_sample100.csv         (stable sample)
    phase2/data/analysis/timeline/timeline_project_dates.parquet
    phase2/data/analysis/timeline/timeline_candidates.parquet

Outputs:
    phase2/output/deliverable04/timeline_sample100_review_packet.csv   (prepare-review)
    phase2/data/analysis/timeline/timeline_validation_sample.parquet   (validate)
    phase2/output/deliverable04/timeline_sample100_validation_projects.csv
    phase2/output/deliverable04/timeline_sample100_validation_summary.csv
    phase2/output/deliverable04/timeline_sample100_rule_diagnostics.csv

Usage:
    python 05_validate_timeline_sample.py --prepare-review
    python 05_validate_timeline_sample.py --validate --reviewed-packet timeline_sample100_review_packet.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
GOLD_DIR = TIMELINE_DIR / "gold"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"

SAMPLE_PATH = OUTPUT_DIR / "timeline_sample100.csv"
DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
GOLD_PROJECTS_PATH = GOLD_DIR / "timeline_gold_projects.parquet"
VALIDATION_PARQUET = TIMELINE_DIR / "timeline_validation_sample.parquet"
REVIEW_PACKET_PATH = OUTPUT_DIR / "timeline_sample100_review_packet.csv"
VALIDATION_PROJECTS_PATH = OUTPUT_DIR / "timeline_sample100_validation_projects.csv"
VALIDATION_SUMMARY_PATH = OUTPUT_DIR / "timeline_sample100_validation_summary.csv"
RULE_DIAGNOSTICS_PATH = OUTPUT_DIR / "timeline_sample100_rule_diagnostics.csv"

# Acceptance thresholds (plan section 8)
THRESHOLDS = {
    "decision_precision": {"CE": 0.95, "EA": 0.95, "EIS": 0.95},
    "initiation_precision": {"CE": 0.85, "EA": 0.90, "EIS": 0.90},
    "invalid_order_rate": 0.02,
}


# ---------------------------------------------------------------------------
# Granularity-aware date matching (plan section 8)
# ---------------------------------------------------------------------------

def _dates_match(
    pipeline_date_str: str | None,
    gold_date_str: str | None,
    gold_granularity: str | None,
) -> bool | None:
    """
    Return True/False per granularity-aware matching rules.
    Return None if gold or pipeline date is null.
    """
    if not gold_date_str or not pipeline_date_str:
        return None
    try:
        p = pd.Timestamp(pipeline_date_str)
        g = pd.Timestamp(gold_date_str)
    except Exception:
        return False

    gran = str(gold_granularity or "day").strip().lower()
    if gran == "day":
        return p.date() == g.date()
    if gran == "month":
        return p.year == g.year and p.month == g.month
    if gran == "year":
        return p.year == g.year
    # unknown: try exact
    return p.date() == g.date()


def _granularity_mismatch(
    pipeline_granularity: str | None,
    gold_granularity: str | None,
) -> bool:
    """True when gold is day-level but pipeline selected only month/year."""
    gold_gran = str(gold_granularity or "").strip().lower()
    pipe_gran = str(pipeline_granularity or "").strip().lower()
    if gold_gran != "day":
        return False
    return pipe_gran in ("month", "year", "unknown")


def _normal_ci_95(successes: float, denominator: float) -> tuple[float | None, float | None]:
    """Return a rough 95% normal-approximation CI for a proportion."""
    if denominator is None or denominator <= 0:
        return None, None
    p = successes / denominator
    se = math.sqrt(max(0.0, p * (1 - p) / denominator))
    lo = max(0.0, p - 1.96 * se)
    hi = min(1.0, p + 1.96 * se)
    return round(lo, 4), round(hi, 4)


def _weighted_match_rate(df: pd.DataFrame, match_col: str) -> float | None:
    if df.empty or "sample_weight" not in df.columns:
        return None
    weights = pd.to_numeric(df["sample_weight"], errors="coerce").fillna(1.0)
    denom = weights.sum()
    if denom <= 0:
        return None
    successes = weights[df[match_col] == True].sum()
    return round(float(successes / denom), 4)


# ---------------------------------------------------------------------------
# Prepare review packet
# ---------------------------------------------------------------------------

def prepare_review(
    sample_df: pd.DataFrame,
    dates_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join sample to pipeline outputs and produce the annotatable review packet.
    """
    project_ids = set(sample_df["project_id"].tolist())

    pipe_cols = [
        "project_id", "process_type",
        "initiation_date", "initiation_date_granularity", "initiation_source_type",
        "initiation_confidence", "initiation_is_proxy", "initiation_evidence_text",
        "initiation_document_id", "initiation_page_number",
        "decision_date", "decision_date_granularity", "decision_source_type",
        "decision_confidence", "decision_is_proxy", "decision_evidence_text",
        "decision_document_id", "decision_page_number",
        "duration_days", "timeline_status", "timeline_flags",
    ]
    pipe_sub = dates_df[dates_df["project_id"].isin(project_ids)][
        [c for c in pipe_cols if c in dates_df.columns]
    ].copy()
    pipe_sub = pipe_sub.rename(columns={
        "initiation_date": "pipeline_initiation_date",
        "initiation_date_granularity": "pipeline_initiation_granularity",
        "initiation_source_type": "pipeline_initiation_source_type",
        "initiation_confidence": "pipeline_initiation_confidence",
        "initiation_is_proxy": "pipeline_initiation_is_proxy",
        "initiation_evidence_text": "pipeline_initiation_evidence",
        "decision_date": "pipeline_decision_date",
        "decision_date_granularity": "pipeline_decision_granularity",
        "decision_source_type": "pipeline_decision_source_type",
        "decision_confidence": "pipeline_decision_confidence",
        "decision_is_proxy": "pipeline_decision_is_proxy",
        "decision_evidence_text": "pipeline_decision_evidence",
        "timeline_status": "pipeline_timeline_status",
        "timeline_flags": "pipeline_timeline_flags",
    })

    # Top candidates per project
    def top_cands(project_id: str, role_filter: list[str], n: int = 5) -> str:
        sub = candidates_df[
            (candidates_df["project_id"] == project_id) &
            (candidates_df["candidate_role"].isin(role_filter))
        ]
        if sub.empty:
            return ""
        sub = sub.nlargest(n, "ranking_score")
        parts = []
        for _, row in sub.iterrows():
            parts.append(
                f"{row.get('parsed_date')} [{row.get('candidate_role')}|"
                f"{row.get('role_confidence')}|score={row.get('ranking_score', 0):.1f}] "
                f"doc={str(row.get('document_type_clean', ''))[:30]} "
                f"| {str(row.get('context_text', ''))[:150]}"
            )
        return " ||| ".join(parts)

    top_init_map = {
        pid: top_cands(pid, ["clear_initiation", "proxy_initiation"])
        for pid in project_ids
    }
    top_dec_map = {
        pid: top_cands(pid, ["clear_decision", "proxy_decision"])
        for pid in project_ids
    }

    packet = sample_df.merge(pipe_sub, on="project_id", how="left")
    packet["top_initiation_candidates"] = packet["project_id"].map(top_init_map)
    packet["top_decision_candidates"] = packet["project_id"].map(top_dec_map)

    return packet


# ---------------------------------------------------------------------------
# Validation metrics
# ---------------------------------------------------------------------------

def run_validation(reviewed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compare gold labels to pipeline dates.
    Returns (projects_df, summary_df, diagnostics_df).
    """
    run_at = datetime.now(timezone.utc).isoformat()
    df = reviewed_df.copy()

    pipeline_date_cols = {
        "initiation": "pipeline_initiation_date",
        "decision": "pipeline_decision_date",
    }
    gold_date_cols = {
        "initiation": "gold_initiation_date",
        "decision": "gold_decision_date",
    }
    gold_gran_cols = {
        "initiation": "gold_initiation_granularity",
        "decision": "gold_decision_granularity",
    }
    gold_type_cols = {
        "initiation": "gold_initiation_type",
        "decision": "gold_decision_type",
    }
    pipeline_gran_cols = {
        "initiation": "pipeline_initiation_granularity",
        "decision": "pipeline_decision_granularity",
    }

    # Compute per-row match flags
    for role in ["initiation", "decision"]:
        pipe_col = pipeline_date_cols[role]
        gold_col = gold_date_cols[role]
        gran_col = gold_gran_cols[role]
        pgran_col = pipeline_gran_cols[role]

        df[f"{role}_date_match"] = df.apply(
            lambda r, pc=pipe_col, gc=gold_col, gr=gran_col: _dates_match(
                r.get(pc), r.get(gc), r.get(gr)
            ),
            axis=1,
        )
        df[f"{role}_granularity_mismatch"] = df.apply(
            lambda r, gr=gran_col, pg=pgran_col: _granularity_mismatch(r.get(pg), r.get(gr)),
            axis=1,
        )

    df["validation_run_at"] = run_at

    # Project-level output
    projects_cols = [
        "sample_id", "project_id", "process_type", "project_energy_type",
        "sample_stratum", "split", "sample_weight",
        "gold_initiation_date", "gold_initiation_granularity", "gold_initiation_type",
        "gold_decision_date", "gold_decision_granularity", "gold_decision_type",
        "gold_initiation_source_type", "gold_initiation_candidate_id",
        "gold_initiation_missing_reason", "gold_decision_source_type",
        "gold_decision_candidate_id", "gold_decision_missing_reason",
        "gold_notes", "gold_reviewer", "reviewer",
        "pipeline_initiation_date", "pipeline_initiation_granularity",
        "pipeline_initiation_source_type", "pipeline_initiation_confidence",
        "pipeline_initiation_is_proxy",
        "pipeline_decision_date", "pipeline_decision_granularity",
        "pipeline_decision_source_type", "pipeline_decision_confidence",
        "pipeline_decision_is_proxy",
        "pipeline_timeline_status", "pipeline_timeline_flags",
        "initiation_date_match", "decision_date_match",
        "initiation_granularity_mismatch", "decision_granularity_mismatch",
        "validation_run_at",
    ]
    projects_cols = [c for c in projects_cols if c in df.columns]
    projects_df = df[projects_cols].copy()

    # Summary metrics by process type
    summary_rows = []
    for pt in ["CE", "EA", "EIS", "ALL"]:
        if pt == "ALL":
            sub = df
        else:
            sub = df[df["process_type"] == pt]
        if sub.empty:
            continue

        # Decision precision: among rows where gold_decision_type == clear AND pipeline date is non-null
        dec_gold_clear = sub[sub["gold_decision_type"].str.strip() == "clear"]
        dec_pred_nonnull = dec_gold_clear[dec_gold_clear["pipeline_decision_date"].notna()]
        dec_precision = (
            dec_pred_nonnull["decision_date_match"].sum() / len(dec_pred_nonnull)
            if len(dec_pred_nonnull) > 0 else float("nan")
        )
        dec_recall = (
            dec_gold_clear["decision_date_match"].sum() / len(dec_gold_clear)
            if len(dec_gold_clear) > 0 else float("nan")
        )
        dec_precision_successes = dec_pred_nonnull["decision_date_match"].sum()
        dec_recall_successes = dec_gold_clear["decision_date_match"].sum()
        dec_precision_ci = _normal_ci_95(dec_precision_successes, len(dec_pred_nonnull))
        dec_recall_ci = _normal_ci_95(dec_recall_successes, len(dec_gold_clear))

        # Initiation precision
        init_gold_clear = sub[sub["gold_initiation_type"].str.strip() == "clear"]
        init_pred_nonnull = init_gold_clear[init_gold_clear["pipeline_initiation_date"].notna()]
        init_precision = (
            init_pred_nonnull["initiation_date_match"].sum() / len(init_pred_nonnull)
            if len(init_pred_nonnull) > 0 else float("nan")
        )
        init_recall = (
            init_gold_clear["initiation_date_match"].sum() / len(init_gold_clear)
            if len(init_gold_clear) > 0 else float("nan")
        )
        init_precision_successes = init_pred_nonnull["initiation_date_match"].sum()
        init_recall_successes = init_gold_clear["initiation_date_match"].sum()
        init_precision_ci = _normal_ci_95(init_precision_successes, len(init_pred_nonnull))
        init_recall_ci = _normal_ci_95(init_recall_successes, len(init_gold_clear))

        invalid_order = (sub["pipeline_timeline_status"] == "invalid_order").sum() if "pipeline_timeline_status" in sub else 0
        invalid_order_rate = invalid_order / len(sub) if len(sub) > 0 else 0.0

        same_day_count = (
            sub["pipeline_timeline_flags"].str.contains("same_day", na=False).sum()
            if "pipeline_timeline_flags" in sub else 0
        )
        gt25y_count = (
            sub["pipeline_timeline_flags"].str.contains("duration_gt_25y", na=False).sum()
            if "pipeline_timeline_flags" in sub else 0
        )
        proxy_init_count = (
            sub["pipeline_initiation_is_proxy"].sum() if "pipeline_initiation_is_proxy" in sub else 0
        )
        proxy_dec_count = (
            sub["pipeline_decision_is_proxy"].sum() if "pipeline_decision_is_proxy" in sub else 0
        )

        # Coverage
        init_coverage = sub["pipeline_initiation_date"].notna().sum() / len(sub)
        dec_coverage = sub["pipeline_decision_date"].notna().sum() / len(sub)

        summary_rows.append({
            "process_type": pt,
            "n_projects": len(sub),
            "thin_subgroup_warning": len(sub) < 50,
            "n_gold_clear_initiation": len(init_gold_clear),
            "n_gold_clear_decision": len(dec_gold_clear),
            "n_decision_precision_denominator": len(dec_pred_nonnull),
            "n_initiation_precision_denominator": len(init_pred_nonnull),
            "initiation_precision": round(init_precision, 4) if pd.notna(init_precision) else None,
            "initiation_precision_ci95_low": init_precision_ci[0],
            "initiation_precision_ci95_high": init_precision_ci[1],
            "initiation_recall": round(init_recall, 4) if pd.notna(init_recall) else None,
            "initiation_recall_ci95_low": init_recall_ci[0],
            "initiation_recall_ci95_high": init_recall_ci[1],
            "decision_precision": round(dec_precision, 4) if pd.notna(dec_precision) else None,
            "decision_precision_ci95_low": dec_precision_ci[0],
            "decision_precision_ci95_high": dec_precision_ci[1],
            "decision_recall": round(dec_recall, 4) if pd.notna(dec_recall) else None,
            "decision_recall_ci95_low": dec_recall_ci[0],
            "decision_recall_ci95_high": dec_recall_ci[1],
            "weighted_initiation_precision": _weighted_match_rate(init_pred_nonnull, "initiation_date_match"),
            "weighted_initiation_recall": _weighted_match_rate(init_gold_clear, "initiation_date_match"),
            "weighted_decision_precision": _weighted_match_rate(dec_pred_nonnull, "decision_date_match"),
            "weighted_decision_recall": _weighted_match_rate(dec_gold_clear, "decision_date_match"),
            "initiation_coverage": round(init_coverage, 4),
            "decision_coverage": round(dec_coverage, 4),
            "invalid_order_count": int(invalid_order),
            "invalid_order_rate": round(invalid_order_rate, 4),
            "same_day_count": int(same_day_count),
            "duration_gt_25y_count": int(gt25y_count),
            "proxy_initiation_count": int(proxy_init_count),
            "proxy_decision_count": int(proxy_dec_count),
            "meets_decision_threshold": (
                dec_precision >= THRESHOLDS["decision_precision"].get(pt, 0.95)
                if pd.notna(dec_precision) else None
            ),
            "meets_initiation_threshold": (
                init_precision >= THRESHOLDS["initiation_precision"].get(pt, 0.90)
                if pd.notna(init_precision) else None
            ),
            "meets_invalid_order_threshold": invalid_order_rate <= THRESHOLDS["invalid_order_rate"],
            "validation_run_at": run_at,
        })

    summary_df = pd.DataFrame(summary_rows)

    # Rule diagnostics: false positives (match=False where pipeline has a date) and false negatives
    diag_rows = []
    for _, row in df.iterrows():
        for role in ["initiation", "decision"]:
            gold_type = str(row.get(f"gold_{role}_type", "")).strip()
            gold_date = row.get(f"gold_{role}_date")
            pipe_date = row.get(f"pipeline_{role}_date")
            match = row.get(f"{role}_date_match")

            if gold_type == "clear" and pipe_date is not None and match is False:
                diag_rows.append({
                    "project_id": row.get("project_id"),
                    "process_type": row.get("process_type"),
                    "role": role,
                    "error_type": "false_positive" if pipe_date else "false_negative",
                    "gold_date": gold_date,
                    "gold_granularity": row.get(f"gold_{role}_granularity"),
                    "pipeline_date": pipe_date,
                    "pipeline_source": row.get(f"pipeline_{role}_source_type"),
                    "pipeline_confidence": row.get(f"pipeline_{role}_confidence"),
                    "gold_notes": row.get("gold_notes"),
                })
            elif gold_type == "clear" and pipe_date is None:
                diag_rows.append({
                    "project_id": row.get("project_id"),
                    "process_type": row.get("process_type"),
                    "role": role,
                    "error_type": "false_negative",
                    "gold_date": gold_date,
                    "gold_granularity": row.get(f"gold_{role}_granularity"),
                    "pipeline_date": None,
                    "pipeline_source": None,
                    "pipeline_confidence": None,
                    "gold_notes": row.get("gold_notes"),
                })

    diag_df = pd.DataFrame(diag_rows) if diag_rows else pd.DataFrame()
    return projects_df, summary_df, diag_df


def prepare_gold_validation_frame(gold_df: pd.DataFrame, dates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join normalized gold project rows to pipeline selected dates.
    """
    pipe_cols = [
        "project_id", "process_type",
        "initiation_date", "initiation_date_granularity", "initiation_source_type",
        "initiation_confidence", "initiation_is_proxy", "initiation_evidence_text",
        "decision_date", "decision_date_granularity", "decision_source_type",
        "decision_confidence", "decision_is_proxy", "decision_evidence_text",
        "duration_days", "timeline_status", "timeline_flags",
    ]
    pipe_cols = [c for c in pipe_cols if c in dates_df.columns]
    pipe_sub = dates_df[pipe_cols].copy()
    pipe_sub = pipe_sub.rename(columns={
        "initiation_date": "pipeline_initiation_date",
        "initiation_date_granularity": "pipeline_initiation_granularity",
        "initiation_source_type": "pipeline_initiation_source_type",
        "initiation_confidence": "pipeline_initiation_confidence",
        "initiation_is_proxy": "pipeline_initiation_is_proxy",
        "decision_date": "pipeline_decision_date",
        "decision_date_granularity": "pipeline_decision_granularity",
        "decision_source_type": "pipeline_decision_source_type",
        "decision_confidence": "pipeline_decision_confidence",
        "decision_is_proxy": "pipeline_decision_is_proxy",
        "timeline_status": "pipeline_timeline_status",
        "timeline_flags": "pipeline_timeline_flags",
    })
    return gold_df.merge(pipe_sub, on=["project_id", "process_type"], how="left")


def validate_gold_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Validate allowed values in gold_* columns."""
    allowed_granularity = {"day", "month", "year", "unknown", ""}
    allowed_type = {"clear", "proxy", "missing", "reject", ""}
    errors = []
    for col, allowed in [
        ("gold_initiation_granularity", allowed_granularity),
        ("gold_decision_granularity", allowed_granularity),
        ("gold_initiation_type", allowed_type),
        ("gold_decision_type", allowed_type),
    ]:
        if col not in df.columns:
            continue
        bad = df[~df[col].fillna("").str.strip().isin(allowed)]
        if not bad.empty:
            errors.append(f"{col}: invalid values {bad[col].unique().tolist()}")
    if errors:
        print(f"WARNING: Gold label validation issues:\n" + "\n".join(errors))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate timeline sample.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--prepare-review", action="store_true", help="Generate annotatable review packet.")
    mode_group.add_argument("--validate", action="store_true", help="Compare pipeline dates to gold labels.")
    parser.add_argument("--reviewed-packet", help="Path to filled review packet CSV (for --validate mode).")
    parser.add_argument(
        "--gold-split",
        choices=["diagnostic_balanced_v2", "train_enriched_v1", "test_representative_v1"],
        help="Validate against normalized gold project labels for this split.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    if args.gold_split:
        if args.prepare_review:
            raise ValueError("--gold-split is only supported with --validate. Use 11_prepare_gold_review_packets.py for review packets.")
        if not GOLD_PROJECTS_PATH.exists():
            raise FileNotFoundError(f"Gold projects not found: {GOLD_PROJECTS_PATH}\nRun 12_import_gold_labels.py first.")
        if not DATES_PATH.exists():
            raise FileNotFoundError(f"Pipeline dates not found: {DATES_PATH}")

        gold_df = pd.read_parquet(GOLD_PROJECTS_PATH)
        gold_df = gold_df[gold_df["split"] == args.gold_split].copy()
        if gold_df.empty:
            raise ValueError(f"No gold project rows found for split {args.gold_split}")
        dates_df = pd.read_parquet(DATES_PATH)
        reviewed_df = prepare_gold_validation_frame(gold_df, dates_df)
        reviewed_df = validate_gold_columns(reviewed_df)
        projects_df, summary_df, diag_df = run_validation(reviewed_df)

        prefix = f"timeline_{args.gold_split}"
        validation_parquet = TIMELINE_DIR / f"{prefix}_validation.parquet"
        projects_path = OUTPUT_DIR / f"{prefix}_validation_projects.csv"
        summary_path = OUTPUT_DIR / f"{prefix}_validation_summary.csv"
        diagnostics_path = OUTPUT_DIR / f"{prefix}_rule_diagnostics.csv"

        projects_df.to_parquet(validation_parquet, index=False)
        projects_df.to_csv(projects_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        if not diag_df.empty:
            diag_df.to_csv(diagnostics_path, index=False)

        print(f"Wrote: {validation_parquet}")
        print(f"Wrote: {projects_path}")
        print(f"Wrote: {summary_path}")
        if not diag_df.empty:
            print(f"Wrote: {diagnostics_path}")

        print("\n=== VALIDATION SUMMARY ===")
        for _, row in summary_df.iterrows():
            pt = row["process_type"]
            print(
                f"  {pt:4s}: dec_prec={row['decision_precision']} "
                f"init_prec={row['initiation_precision']} "
                f"dec_recall={row['decision_recall']} "
                f"init_recall={row['initiation_recall']} "
                f"thin={row['thin_subgroup_warning']}"
            )
        return

    # Load stable sample
    if not SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Sample not found: {SAMPLE_PATH}\nRun 00_sample_timeline_projects.py first.")
    sample_df = pd.read_csv(SAMPLE_PATH)
    print(f"Sample: {len(sample_df)} projects")

    if args.prepare_review:
        if not DATES_PATH.exists():
            raise FileNotFoundError(f"Pipeline dates not found: {DATES_PATH}\nRun 04_select_timeline_dates.py first.")
        if not CANDIDATES_PATH.exists():
            raise FileNotFoundError(f"Candidates not found: {CANDIDATES_PATH}")

        dates_df = pd.read_parquet(DATES_PATH)
        candidates_df = pd.read_parquet(CANDIDATES_PATH)

        # Filter to sample projects
        sample_ids = set(sample_df["project_id"])
        dates_sub = dates_df[dates_df["project_id"].isin(sample_ids)]
        cands_sub = candidates_df[candidates_df["project_id"].isin(sample_ids)]

        print(f"Building review packet for {len(sample_ids)} projects...")
        packet = prepare_review(sample_df, dates_sub, cands_sub)
        packet.to_csv(REVIEW_PACKET_PATH, index=False)
        print(f"Wrote: {REVIEW_PACKET_PATH}")
        n_pipeline_init = packet["pipeline_initiation_date"].notna().sum()
        n_pipeline_dec = packet["pipeline_decision_date"].notna().sum()
        print(f"Pipeline coverage: initiation={n_pipeline_init}/{len(packet)}, decision={n_pipeline_dec}/{len(packet)}")

    elif args.validate:
        reviewed_path = args.reviewed_packet or str(REVIEW_PACKET_PATH)
        if not Path(reviewed_path).exists():
            raise FileNotFoundError(f"Reviewed packet not found: {reviewed_path}")

        print(f"Loading reviewed packet: {reviewed_path}")
        reviewed_df = pd.read_csv(reviewed_path)
        reviewed_df = validate_gold_columns(reviewed_df)

        # Check that gold columns have been filled
        filled = reviewed_df["gold_initiation_type"].notna() & (reviewed_df["gold_initiation_type"] != "")
        n_filled = filled.sum()
        print(f"Gold labels filled: {n_filled}/{len(reviewed_df)} rows have gold_initiation_type")
        if n_filled == 0:
            print("WARNING: No gold labels found. Fill gold_* columns in the review packet first.")

        projects_df, summary_df, diag_df = run_validation(reviewed_df)

        # Write Parquet validation_sample
        if not projects_df.empty:
            projects_df.to_parquet(VALIDATION_PARQUET, index=False)
            print(f"Wrote: {VALIDATION_PARQUET}")

        projects_df.to_csv(VALIDATION_PROJECTS_PATH, index=False)
        summary_df.to_csv(VALIDATION_SUMMARY_PATH, index=False)
        print(f"Wrote: {VALIDATION_PROJECTS_PATH}")
        print(f"Wrote: {VALIDATION_SUMMARY_PATH}")

        if not diag_df.empty:
            diag_df.to_csv(RULE_DIAGNOSTICS_PATH, index=False)
            print(f"Wrote: {RULE_DIAGNOSTICS_PATH} ({len(diag_df)} diagnostic rows)")

        # Print summary
        print("\n=== VALIDATION SUMMARY ===")
        for _, row in summary_df.iterrows():
            pt = row["process_type"]
            print(
                f"  {pt:4s}: dec_prec={row['decision_precision']:.0%} "
                f"init_prec={row['initiation_precision']:.0%}  "
                f"invalid_order={row['invalid_order_rate']:.1%}  "
                f"dec_cov={row['decision_coverage']:.0%}  init_cov={row['initiation_coverage']:.0%}"
            )
            fails = []
            if row.get("meets_decision_threshold") is False:
                fails.append(f"decision precision below {THRESHOLDS['decision_precision'].get(pt, 0.95):.0%}")
            if row.get("meets_initiation_threshold") is False:
                fails.append(f"initiation precision below {THRESHOLDS['initiation_precision'].get(pt, 0.90):.0%}")
            if row.get("meets_invalid_order_threshold") is False:
                fails.append(f"invalid order rate above {THRESHOLDS['invalid_order_rate']:.0%}")
            if fails:
                print(f"    BELOW THRESHOLD: {', '.join(fails)}")


if __name__ == "__main__":
    main()
