"""
Import reviewed gold-standard timeline labels.

Reads project-level and candidate-level review CSVs produced by
11_prepare_gold_review_packets.py, validates labels, and writes normalized
Parquet tables under phase2/data/analysis/timeline/gold/.

Outputs:
    timeline_gold_project_reviews.parquet
    timeline_gold_projects.parquet
    timeline_gold_candidate_reviews.parquet
    timeline_gold_candidates.parquet
    timeline_gold_candidate_training.parquet
    timeline_gold_irr.parquet
    timeline_gold_reconciliation_queue.csv

Usage:
    python 12_import_gold_labels.py --projects reviewed_projects.csv
    python 12_import_gold_labels.py --projects reviewed_projects.csv --candidates reviewed_candidates.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
TIMELINE_DIR = PHASE2 / "data" / "analysis" / "timeline"
GOLD_DIR = TIMELINE_DIR / "gold"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04" / "gold"

SPLITS_PATH = GOLD_DIR / "timeline_gold_splits.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
PROJECT_REVIEWS_PATH = GOLD_DIR / "timeline_gold_project_reviews.parquet"
PROJECTS_FINAL_PATH = GOLD_DIR / "timeline_gold_projects.parquet"
CANDIDATE_REVIEWS_PATH = GOLD_DIR / "timeline_gold_candidate_reviews.parquet"
CANDIDATES_FINAL_PATH = GOLD_DIR / "timeline_gold_candidates.parquet"
CANDIDATE_TRAINING_PATH = GOLD_DIR / "timeline_gold_candidate_training.parquet"
IRR_PATH = GOLD_DIR / "timeline_gold_irr.parquet"
IRR_SUMMARY_PATH = OUTPUT_DIR / "timeline_gold_irr_summary.csv"
RECONCILIATION_QUEUE_PATH = OUTPUT_DIR / "timeline_gold_reconciliation_queue.csv"

ALLOWED_GRANULARITY = {"day", "month", "year", "unknown", ""}
ALLOWED_PROJECT_TYPE = {"clear", "proxy", "missing", "reject", ""}
ALLOWED_CONFIDENCE = {"high", "medium", "low", ""}
ALLOWED_MISSING_REASON = {
    "no_evidence", "only_proxy", "retrieval_miss", "parser_miss",
    "ambiguous", "not_applicable", "",
}
ALLOWED_CANDIDATE_ROLE = {
    "clear_initiation", "proxy_initiation", "clear_decision", "proxy_decision",
    "review", "historical", "reject", "unknown", "",
}
ALLOWED_SELECTED_FOR = {"initiation", "decision", "none", "alternate_valid", ""}
ALLOWED_ERROR_CATEGORY = {
    "legal_citation", "historical_project", "prior_programmatic",
    "review_signoff", "specialist_review", "future_schedule", "map_revision",
    "form_boilerplate", "other_project", "wrong_role", "other", "",
}
FINAL_REVIEW_PASSES = {"primary", "primary_blind", "reconciled", "final"}
RECONCILED_REVIEW_PASSES = {"reconciled", "final"}
COMPARE_PROJECT_COLS = [
    "gold_initiation_date", "gold_initiation_granularity", "gold_initiation_type",
    "gold_initiation_candidate_id", "gold_initiation_missing_reason",
    "gold_decision_date", "gold_decision_granularity", "gold_decision_type",
    "gold_decision_candidate_id", "gold_decision_missing_reason",
]


def _clean_str(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_bool(value: object) -> bool:
    text = _clean_str(value).lower()
    return text in {"true", "1", "yes", "y"}


def _row_hash(parts: list[object]) -> str:
    raw = "|".join(_clean_str(x) for x in parts)
    return hashlib.sha1(raw.encode()).hexdigest()[:24]


def _validate_enum(df: pd.DataFrame, col: str, allowed: set[str], errors: list[str]) -> None:
    if col not in df.columns:
        return
    bad = df[~df[col].fillna("").astype(str).str.strip().isin(allowed)]
    if not bad.empty:
        errors.append(f"{col}: invalid values {bad[col].dropna().unique().tolist()[:20]}")


def _validate_date(value: object, label: str, errors: list[str], row_id: str) -> str:
    text = _clean_str(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        errors.append(f"{row_id}: could not parse {label}={text!r}")
        return text
    return parsed.date().isoformat()


def _validate_project_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = ["project_id", "split", "process_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Project review CSV missing required columns: {missing}")

    df = df.copy()
    errors: list[str] = []
    for col in [
        "gold_initiation_granularity", "gold_decision_granularity",
    ]:
        _validate_enum(df, col, ALLOWED_GRANULARITY, errors)
    for col in ["gold_initiation_type", "gold_decision_type"]:
        _validate_enum(df, col, ALLOWED_PROJECT_TYPE, errors)
    for col in ["gold_initiation_confidence", "gold_decision_confidence"]:
        _validate_enum(df, col, ALLOWED_CONFIDENCE, errors)
    for col in ["gold_initiation_missing_reason", "gold_decision_missing_reason"]:
        _validate_enum(df, col, ALLOWED_MISSING_REASON, errors)

    for idx, row in df.iterrows():
        row_id = f"row {idx} project {row.get('project_id')}"
        for role in ["initiation", "decision"]:
            type_col = f"gold_{role}_type"
            date_col = f"gold_{role}_date"
            gran_col = f"gold_{role}_granularity"
            evidence_col = f"gold_{role}_evidence_text"
            candidate_col = f"gold_{role}_candidate_id"
            missing_col = f"gold_{role}_missing_reason"
            label_type = _clean_str(row.get(type_col))
            date_text = _clean_str(row.get(date_col))

            normalized_date = _validate_date(row.get(date_col), date_col, errors, row_id)
            df.at[idx, date_col] = normalized_date

            if label_type in {"clear", "proxy"}:
                if not date_text:
                    errors.append(f"{row_id}: {type_col}={label_type} requires {date_col}")
                if not _clean_str(row.get(gran_col)):
                    errors.append(f"{row_id}: {type_col}={label_type} requires {gran_col}")
                if not (_clean_str(row.get(evidence_col)) or _clean_str(row.get(candidate_col))):
                    errors.append(
                        f"{row_id}: {type_col}={label_type} requires evidence text or candidate id"
                    )
            if label_type in {"missing", "reject"} and not _clean_str(row.get(missing_col)):
                errors.append(f"{row_id}: {type_col}={label_type} requires {missing_col}")

    if errors:
        raise ValueError("Project review validation failed:\n" + "\n".join(errors[:100]))

    run_at = datetime.now(timezone.utc).isoformat()
    df["imported_at"] = run_at
    if "reviewer" not in df.columns:
        df["reviewer"] = "reviewer"
    if "review_pass" not in df.columns:
        df["review_pass"] = "primary"
    df["review_id"] = df.apply(
        lambda r: _row_hash([r.get("split"), r.get("project_id"), r.get("reviewer"), r.get("review_pass")]),
        axis=1,
    )
    return df


def _validate_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    required = ["candidate_id", "project_id", "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Candidate review CSV missing required columns: {missing}")

    df = df.copy()
    errors: list[str] = []
    _validate_enum(df, "gold_candidate_role", ALLOWED_CANDIDATE_ROLE, errors)
    _validate_enum(df, "gold_selected_for", ALLOWED_SELECTED_FOR, errors)
    _validate_enum(df, "gold_error_category", ALLOWED_ERROR_CATEGORY, errors)

    if CANDIDATES_PATH.exists():
        known = set(pd.read_parquet(CANDIDATES_PATH, columns=["candidate_id"])["candidate_id"].astype(str))
        bad = df[~df["candidate_id"].astype(str).isin(known)]
        if not bad.empty:
            errors.append(f"Unknown candidate_id values: {bad['candidate_id'].head(20).tolist()}")

    if errors:
        raise ValueError("Candidate review validation failed:\n" + "\n".join(errors[:100]))

    run_at = datetime.now(timezone.utc).isoformat()
    df["imported_at"] = run_at
    if "reviewer" not in df.columns:
        df["reviewer"] = "reviewer"
    if "review_pass" not in df.columns:
        df["review_pass"] = "primary"
    df["candidate_review_id"] = df.apply(
        lambda r: _row_hash([r.get("split"), r.get("candidate_id"), r.get("reviewer"), r.get("review_pass")]),
        axis=1,
    )
    return df


def _append_dedup(path: Path, new_df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    combined = combined.drop_duplicates(key_cols, keep="last")
    combined.to_parquet(path, index=False)
    return combined


def _build_final_projects(project_reviews: pd.DataFrame) -> pd.DataFrame:
    if project_reviews.empty:
        return project_reviews
    df = project_reviews.copy()
    df["irr_required_bool"] = df.get("irr_required", False).map(_parse_bool) if "irr_required" in df.columns else False
    df["review_pass_clean"] = df.get("review_pass", "primary").fillna("primary").astype(str).str.strip()

    final_parts = []
    non_irr = df[~df["irr_required_bool"] & df["review_pass_clean"].isin(FINAL_REVIEW_PASSES)]
    if not non_irr.empty:
        final_parts.append(non_irr)

    irr_df = df[df["irr_required_bool"]].copy()
    for _, group in irr_df.groupby(["split", "project_id"], dropna=False):
        group = group.sort_values("imported_at")
        reconciled = group[group["review_pass_clean"].isin(RECONCILED_REVIEW_PASSES)]
        if not reconciled.empty:
            final_parts.append(reconciled.tail(1))
            continue

        comparison = group[~group["review_pass_clean"].isin(RECONCILED_REVIEW_PASSES)]
        if len(comparison) >= 2 and not _project_disagreement_fields(comparison):
            final_parts.append(comparison.tail(1))

    final = pd.concat(final_parts, ignore_index=True) if final_parts else pd.DataFrame()
    if final.empty:
        return final

    final = final.sort_values(["split", "project_id", "imported_at"])
    final = final.drop_duplicates(["split", "project_id"], keep="last")
    final = final.drop(columns=["irr_required_bool", "review_pass_clean"], errors="ignore")
    final.to_parquet(PROJECTS_FINAL_PATH, index=False)
    return final


def _build_final_candidates(candidate_reviews: pd.DataFrame) -> pd.DataFrame:
    if candidate_reviews.empty:
        return candidate_reviews
    df = candidate_reviews.copy()
    df["review_pass_clean"] = df.get("review_pass", "primary").fillna("primary").astype(str).str.strip()
    final = df[df["review_pass_clean"].isin(FINAL_REVIEW_PASSES)].copy()
    final = final.sort_values(["split", "candidate_id", "imported_at"])
    final = final.drop_duplicates(["split", "candidate_id"], keep="last")
    final = final.drop(columns=["review_pass_clean"], errors="ignore")
    final.to_parquet(CANDIDATES_FINAL_PATH, index=False)
    return final


def _build_candidate_training(final_candidates: pd.DataFrame) -> pd.DataFrame:
    if final_candidates.empty:
        return final_candidates
    train = final_candidates[
        (final_candidates["split"] != "test_representative_v1")
        & final_candidates["gold_candidate_role"].fillna("").astype(str).str.strip().ne("")
    ].copy()
    if not train.empty:
        train.to_parquet(CANDIDATE_TRAINING_PATH, index=False)
    return train


def _project_disagreement_fields(group: pd.DataFrame) -> list[str]:
    disagreement_fields = []
    for col in COMPARE_PROJECT_COLS:
        if col not in group.columns:
            continue
        vals = set(group[col].fillna("").astype(str).str.strip())
        if len(vals) > 1:
            disagreement_fields.append(col)
    return disagreement_fields


def _granularity_aware_equal(
    left_date: object,
    left_granularity: object,
    right_date: object,
    right_granularity: object,
) -> bool:
    left_text = _clean_str(left_date)
    right_text = _clean_str(right_date)
    if not left_text and not right_text:
        return True
    if not left_text or not right_text:
        return False
    left = pd.to_datetime(left_text, errors="coerce")
    right = pd.to_datetime(right_text, errors="coerce")
    if pd.isna(left) or pd.isna(right):
        return False

    granularities = {_clean_str(left_granularity).lower(), _clean_str(right_granularity).lower()}
    if "year" in granularities:
        return left.year == right.year
    if "month" in granularities:
        return left.year == right.year and left.month == right.month
    return left.date() == right.date()


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    expected = 0.0
    for label in labels:
        expected += (left.count(label) / n) * (right.count(label) / n)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return round((observed - expected) / (1 - expected), 4)


def _build_irr_summary(pair_rows: list[dict]) -> pd.DataFrame:
    if not pair_rows:
        return pd.DataFrame()
    rows = []
    for role in ["initiation", "decision"]:
        type_left = [_clean_str(r[f"{role}_type_left"]) for r in pair_rows]
        type_right = [_clean_str(r[f"{role}_type_right"]) for r in pair_rows]
        n = len(pair_rows)
        type_agree = sum(a == b for a, b in zip(type_left, type_right))

        clear_pairs = [
            r for r in pair_rows
            if _clean_str(r[f"{role}_type_left"]) == "clear"
            or _clean_str(r[f"{role}_type_right"]) == "clear"
        ]
        exact_date_agree = sum(bool(r[f"{role}_exact_date_agreement"]) for r in clear_pairs)
        gran_date_agree = sum(bool(r[f"{role}_granularity_aware_date_agreement"]) for r in clear_pairs)
        clear_n = len(clear_pairs)

        rows.append({
            "role": role,
            "n_projects": n,
            "type_percent_agreement": round(type_agree / n, 4) if n else None,
            "type_cohen_kappa": _cohen_kappa(type_left, type_right),
            "n_clear_date_pairs": clear_n,
            "exact_date_agreement": round(exact_date_agree / clear_n, 4) if clear_n else None,
            "granularity_aware_date_agreement": round(gran_date_agree / clear_n, 4) if clear_n else None,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        })
    return pd.DataFrame(rows)


def _build_irr(project_reviews: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if project_reviews.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = project_reviews.copy()
    if "irr_required" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df = df[df["irr_required"].map(_parse_bool)].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["review_pass_clean"] = df.get("review_pass", "primary").fillna("primary").astype(str).str.strip()

    rows = []
    pair_rows = []
    for (split, pid), group in df.groupby(["split", "project_id"], dropna=False):
        comparison = group[~group["review_pass_clean"].isin(RECONCILED_REVIEW_PASSES)].copy()
        comparison = comparison.sort_values(["reviewer", "review_pass_clean", "imported_at"])
        if len(comparison) < 2:
            continue
        reviewers = group[[
            "review_id", "reviewer", "review_pass", "imported_at"
        ] + [c for c in COMPARE_PROJECT_COLS if c in group.columns]].to_dict("records")
        disagreement_fields = _project_disagreement_fields(comparison)
        reconciled = group[group["review_pass_clean"].isin(RECONCILED_REVIEW_PASSES)].sort_values("imported_at")
        rec_row = reconciled.iloc[-1] if not reconciled.empty else pd.Series(dtype=object)

        left = comparison.iloc[0]
        right = comparison.iloc[1]
        pair = {
            "split": split,
            "project_id": pid,
        }
        for role in ["initiation", "decision"]:
            pair[f"{role}_type_left"] = left.get(f"gold_{role}_type")
            pair[f"{role}_type_right"] = right.get(f"gold_{role}_type")
            pair[f"{role}_exact_date_agreement"] = (
                _clean_str(left.get(f"gold_{role}_date"))
                == _clean_str(right.get(f"gold_{role}_date"))
            )
            pair[f"{role}_granularity_aware_date_agreement"] = _granularity_aware_equal(
                left.get(f"gold_{role}_date"),
                left.get(f"gold_{role}_granularity"),
                right.get(f"gold_{role}_date"),
                right.get(f"gold_{role}_granularity"),
            )
        pair_rows.append(pair)

        rows.append({
            "split": split,
            "project_id": pid,
            "n_reviews": len(comparison),
            "n_total_rows_including_reconciliation": len(group),
            "reviewer_payload_json": json.dumps(reviewers, default=str),
            "disagreement_fields": "|".join(disagreement_fields),
            "needs_reconciliation": bool(disagreement_fields) and reconciled.empty,
            "reconciled": not reconciled.empty,
            "final_review_id": _clean_str(rec_row.get("review_id")),
            "reconciler": _clean_str(rec_row.get("reconciler")) or _clean_str(rec_row.get("reviewer")),
            "adjudication_reason": _clean_str(rec_row.get("adjudication_reason")) or _clean_str(rec_row.get("gold_notes")),
            "adjudicated_at": _clean_str(rec_row.get("adjudicated_at")) or _clean_str(rec_row.get("imported_at")),
            "initiation_exact_date_agreement": pair["initiation_exact_date_agreement"],
            "initiation_granularity_aware_date_agreement": pair["initiation_granularity_aware_date_agreement"],
            "initiation_type_agreement": _clean_str(left.get("gold_initiation_type")) == _clean_str(right.get("gold_initiation_type")),
            "decision_exact_date_agreement": pair["decision_exact_date_agreement"],
            "decision_granularity_aware_date_agreement": pair["decision_granularity_aware_date_agreement"],
            "decision_type_agreement": _clean_str(left.get("gold_decision_type")) == _clean_str(right.get("gold_decision_type")),
            "irr_computed_at": datetime.now(timezone.utc).isoformat(),
        })

    irr = pd.DataFrame(rows)
    summary = _build_irr_summary(pair_rows)
    if not irr.empty:
        irr.to_parquet(IRR_PATH, index=False)
        queue = irr[irr["needs_reconciliation"]].copy()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        queue.to_csv(RECONCILIATION_QUEUE_PATH, index=False)
    if not summary.empty:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary.to_csv(IRR_SUMMARY_PATH, index=False)
    return irr, summary


def import_labels(project_csv: str, candidate_csv: str | None) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    projects = pd.read_csv(project_csv)
    projects = _validate_project_rows(projects)
    all_project_reviews = _append_dedup(PROJECT_REVIEWS_PATH, projects, ["review_id"])
    final_projects = _build_final_projects(all_project_reviews)
    irr, irr_summary = _build_irr(all_project_reviews)

    print(f"Wrote: {PROJECT_REVIEWS_PATH} ({len(all_project_reviews):,} review rows)")
    if not final_projects.empty:
        print(f"Wrote: {PROJECTS_FINAL_PATH} ({len(final_projects):,} final project rows)")
    if not irr.empty:
        print(f"Wrote: {IRR_PATH} ({len(irr):,} IRR rows)")
        if not irr_summary.empty:
            print(f"Wrote: {IRR_SUMMARY_PATH}")
        print(f"Wrote: {RECONCILIATION_QUEUE_PATH}")

    if candidate_csv:
        candidates = pd.read_csv(candidate_csv)
        candidates = _validate_candidate_rows(candidates)
        all_candidate_reviews = _append_dedup(CANDIDATE_REVIEWS_PATH, candidates, ["candidate_review_id"])
        final_candidates = _build_final_candidates(all_candidate_reviews)
        training = _build_candidate_training(final_candidates)
        print(f"Wrote: {CANDIDATE_REVIEWS_PATH} ({len(all_candidate_reviews):,} review rows)")
        if not final_candidates.empty:
            print(f"Wrote: {CANDIDATES_FINAL_PATH} ({len(final_candidates):,} final candidate rows)")
        if not training.empty:
            print(f"Wrote: {CANDIDATE_TRAINING_PATH} ({len(training):,} training rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import D4 gold-standard timeline labels.")
    parser.add_argument("--projects", required=True, help="Reviewed project-level CSV.")
    parser.add_argument("--candidates", help="Reviewed candidate-level CSV.")
    args = parser.parse_args()

    import_labels(args.projects, args.candidates)


if __name__ == "__main__":
    main()
