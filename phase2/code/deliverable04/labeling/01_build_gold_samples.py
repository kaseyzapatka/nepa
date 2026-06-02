"""
Build gold-standard sample splits for D4 timeline labeling.

Outputs:
    phase2/data/analysis/timeline/gold/timeline_gold_splits.parquet
    phase2/output/deliverable04/gold/splits/<split>.csv
    phase2/output/deliverable04/gold/splits/<split>_ids.txt

Usage:
    python 10_build_gold_samples.py
    python 10_build_gold_samples.py --dry-run
    python 10_build_gold_samples.py --overwrite-existing
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
GOLD_DIR = TIMELINE_DIR / "gold"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"
GOLD_OUTPUT_DIR = OUTPUT_DIR / "gold" / "splits"

PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"
DATES_PATH = TIMELINE_DIR / "timeline_project_dates.parquet"
CANDIDATES_PATH = TIMELINE_DIR / "timeline_candidates.parquet"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"
MANUAL_QUEUE_PATH = OUTPUT_DIR / "timeline_manual_review_queue.csv"
SAMPLE100_PATH = OUTPUT_DIR / "timeline_sample100.csv"
SPLITS_PATH = GOLD_DIR / "timeline_gold_splits.parquet"

SEED = 20260528
PROCESS_TYPES = ["CE", "EA", "EIS"]
ENERGY_TYPES = ["Clean", "Fossil", "Other"]

DIAGNOSTIC_TARGET = {
    ("CE", "Clean"): 17, ("CE", "Fossil"): 17, ("CE", "Other"): 16,
    ("EA", "Clean"): 17, ("EA", "Fossil"): 17, ("EA", "Other"): 16,
    ("EIS", "Clean"): 17, ("EIS", "Fossil"): 17, ("EIS", "Other"): 16,
}

TRAIN_TARGET = {
    ("CE", "Clean"): 213, ("CE", "Fossil"): 101, ("CE", "Other"): 286,
    ("EA", "Clean"): 37, ("EA", "Fossil"): 63, ("EA", "Other"): 100,
    ("EIS", "Clean"): 36, ("EIS", "Fossil"): 30, ("EIS", "Other"): 134,
}

TEST_TARGET = {
    ("CE", "Clean"): 157, ("CE", "Fossil"): 74, ("CE", "Other"): 211,
    ("EA", "Clean"): 5, ("EA", "Fossil"): 8, ("EA", "Other"): 12,
    ("EIS", "Clean"): 6, ("EIS", "Fossil"): 5, ("EIS", "Other"): 22,
}

TRAIN_BUCKET_TARGETS = {
    "missing_failed": 0.25,
    "ambiguous": 0.25,
    "apparent_success": 0.20,
    "retrieval_candidate_weak": 0.15,
    "structured_edge": 0.15,
}


def _file_hash(path: Path) -> str:
    if not path.exists():
        return "missing"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ids_hash(ids: list[str]) -> str:
    payload = "\n".join(sorted(str(x) for x in ids))
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): _file_hash(path)
        for path in [
            PROJECTS_PATH,
            INDEX_PATH,
            DATES_PATH,
            CANDIDATES_PATH,
            PACKETS_PATH,
            MANUAL_QUEUE_PATH,
            SAMPLE100_PATH,
        ]
    }


def normalize_energy(value: object) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return text if text in {"Clean", "Fossil", "Other"} else "Other"


def doc_count_bin(process_type: str, count: object) -> str:
    n = pd.to_numeric(count, errors="coerce")
    if pd.isna(n):
        return "unknown"
    if process_type == "CE":
        if n <= 1:
            return "1_doc"
        if n == 2:
            return "2_docs"
        return "3plus_docs"
    if process_type == "EA":
        if n <= 2:
            return "1_2_docs"
        if n <= 5:
            return "3_5_docs"
        if n <= 14:
            return "6_14_docs"
        return "15plus_docs"
    if process_type == "EIS":
        if n <= 2:
            return "1_2_docs"
        if n <= 10:
            return "3_10_docs"
        if n <= 55:
            return "11_55_docs"
        return "56plus_docs"
    return "unknown"


def _load_projects() -> pd.DataFrame:
    cols = [
        "project_id", "project_title", "process_type", "project_energy_type",
        "lead_agency_harmonized", "project_department", "project_state",
        "project_county", "project_doc_count", "project_has_decision_doc",
        "project_has_final_doc", "project_has_draft_doc", "project_has_appendix_doc",
        "noi_publication_date", "noi_match_status", "noi_match_confidence",
        "noa_availability_date", "noa_match_status",
    ]
    projects = pd.read_parquet(PROJECTS_PATH)
    projects = projects[[c for c in cols if c in projects.columns]].copy()

    projects = projects[projects["process_type"].isin(PROCESS_TYPES)].copy()
    projects["project_energy_type"] = projects["project_energy_type"].map(normalize_energy)
    projects["project_doc_count"] = pd.to_numeric(
        projects.get("project_doc_count"), errors="coerce"
    )
    projects["doc_count_bin"] = projects.apply(
        lambda r: doc_count_bin(r["process_type"], r["project_doc_count"]), axis=1
    )
    return projects


def _load_project_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["project_id"])
    df = pd.read_parquet(path)
    keep = [c for c in columns if c in df.columns]
    if "project_id" not in keep and "project_id" in df.columns:
        keep.insert(0, "project_id")
    return df[keep].copy()


def build_sampling_frame() -> pd.DataFrame:
    projects = _load_projects()

    if INDEX_PATH.exists():
        index = pd.read_parquet(INDEX_PATH)
        burden = (
            index.groupby("project_id", as_index=False)
            .agg(
                n_index_documents=("document_id", "nunique"),
                total_pages=("doc_page_count", "sum"),
                max_document_pages=("doc_page_count", "max"),
                appendix_count=("is_appendix_like", "sum"),
                n_decision_title_cues=("has_decision_title_cue", "sum"),
                n_initiation_title_cues=("has_initiation_title_cue", "sum"),
                n_high_priority_docs=("scan_priority", lambda s: s.isin(["priority_1", "priority_2"]).sum()),
                has_sections=("has_sections", "max"),
            )
        )
        projects = projects.merge(burden, on="project_id", how="left")
    else:
        for col in [
            "n_index_documents", "total_pages", "max_document_pages", "appendix_count",
            "n_decision_title_cues", "n_initiation_title_cues", "n_high_priority_docs",
            "has_sections",
        ]:
            projects[col] = 0

    dates = _load_project_columns(
        DATES_PATH,
        [
            "project_id", "timeline_status", "timeline_flags",
            "initiation_date", "decision_date", "initiation_confidence",
            "decision_confidence", "initiation_is_proxy", "decision_is_proxy",
        ],
    )
    if not dates.empty:
        projects = projects.merge(dates.drop_duplicates("project_id"), on="project_id", how="left")

    if CANDIDATES_PATH.exists():
        cands = pd.read_parquet(CANDIDATES_PATH)
        cand_roll = (
            cands.groupby("project_id", as_index=False)
            .agg(
                n_candidates=("candidate_id", "nunique"),
                n_unknown_candidates=("candidate_role", lambda s: (s == "unknown").sum()),
                n_historical_candidates=("candidate_role", lambda s: (s == "historical").sum()),
                n_relevant_candidates=("candidate_role", lambda s: s.isin([
                    "clear_initiation", "proxy_initiation", "clear_decision", "proxy_decision"
                ]).sum()),
                max_candidate_score=("ranking_score", "max"),
            )
        )
        projects = projects.merge(cand_roll, on="project_id", how="left")
    else:
        for col in [
            "n_candidates", "n_unknown_candidates", "n_historical_candidates",
            "n_relevant_candidates", "max_candidate_score",
        ]:
            projects[col] = 0

    if PACKETS_PATH.exists():
        packets = pd.read_parquet(PACKETS_PATH)
        packet_roll = (
            packets.groupby("project_id", as_index=False)
            .agg(
                n_packets=("context_packet_id", "nunique"),
                max_packet_score=("retrieval_score", "max"),
                has_metadata_packet=("source_tier", lambda s: (s == "metadata").any()),
            )
        )
        projects = projects.merge(packet_roll, on="project_id", how="left")
    else:
        for col in ["n_packets", "max_packet_score", "has_metadata_packet"]:
            projects[col] = 0

    if MANUAL_QUEUE_PATH.exists():
        queue = pd.read_csv(MANUAL_QUEUE_PATH, usecols=["project_id"])
        projects["in_manual_queue"] = projects["project_id"].isin(set(queue["project_id"]))
    else:
        projects["in_manual_queue"] = False

    fill_zero = [
        "n_index_documents", "total_pages", "max_document_pages", "appendix_count",
        "n_decision_title_cues", "n_initiation_title_cues", "n_high_priority_docs",
        "n_candidates", "n_unknown_candidates", "n_historical_candidates",
        "n_relevant_candidates", "max_candidate_score", "n_packets", "max_packet_score",
    ]
    for col in fill_zero:
        if col in projects.columns:
            projects[col] = pd.to_numeric(projects[col], errors="coerce").fillna(0)

    projects["timeline_status"] = projects.get("timeline_status", pd.Series(index=projects.index, dtype=object)).fillna("not_run")
    projects["timeline_flags"] = projects.get("timeline_flags", pd.Series(index=projects.index, dtype=object)).fillna("")
    projects["workflow_condition"] = projects.apply(_workflow_condition, axis=1)
    projects["sampling_frame_built_at"] = datetime.now(timezone.utc).isoformat()
    return projects


def _workflow_condition(row: pd.Series) -> str:
    status = str(row.get("timeline_status") or "not_run")
    flags = str(row.get("timeline_flags") or "")

    if status in {"missing_initiation", "missing_decision", "missing_both", "invalid_order", "manual_review"}:
        return "missing_failed"
    if any(flag in flags for flag in [
        "multiple_high_score_candidates", "low_confidence_selection",
        "proxy_only", "non_day_granularity",
    ]):
        return "ambiguous"
    if status in {"complete_clear", "complete_with_proxy"}:
        return "apparent_success"
    if bool(row.get("has_metadata_packet")) or pd.notna(row.get("noi_publication_date")):
        return "structured_edge"
    if (
        float(row.get("max_packet_score") or 0) >= 2
        or float(row.get("n_unknown_candidates") or 0) >= 5
        or float(row.get("n_historical_candidates") or 0) >= 5
        or float(row.get("total_pages") or 0) >= 1000
    ):
        return "retrieval_candidate_weak"
    if (
        float(row.get("n_decision_title_cues") or 0) > 0
        or float(row.get("n_initiation_title_cues") or 0) > 0
        or float(row.get("appendix_count") or 0) > 10
    ):
        return "structured_edge"
    return "not_run"


def _sample_n(df: pd.DataFrame, n: int, seed: int, weights: pd.Series | None = None) -> pd.DataFrame:
    if n <= 0:
        return df.iloc[0:0].copy()
    if len(df) < n:
        raise ValueError(f"Need {n} rows but only {len(df)} available.")
    if weights is not None:
        weights = weights.reindex(df.index).fillna(0)
        if weights.sum() <= 0:
            weights = None
    return df.sample(n=n, random_state=seed, weights=weights, replace=False)


def _weighted_by_burden(df: pd.DataFrame) -> pd.Series:
    doc_counts = df["doc_count_bin"].value_counts()
    agencies = df["lead_agency_harmonized"].fillna("missing").value_counts()
    weights = (
        df["doc_count_bin"].map(lambda x: 1 / max(1, doc_counts.get(x, 1)))
        * df["lead_agency_harmonized"].fillna("missing").map(lambda x: 1 / math.sqrt(max(1, agencies.get(x, 1))))
    )
    high_burden = pd.to_numeric(df["total_pages"], errors="coerce").fillna(0) >= 1000
    weights = weights * (1 + high_burden.astype(float))
    return weights


def _sample_enriched_cell(cell: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    selected_parts = []
    selected_ids: set[str] = set()
    targets = {k: int(math.floor(n * v)) for k, v in TRAIN_BUCKET_TARGETS.items()}
    remainder = n - sum(targets.values())
    # Put rounding remainder into ambiguous/missing buckets first.
    for key in ["ambiguous", "missing_failed", "retrieval_candidate_weak", "structured_edge", "apparent_success"]:
        if remainder <= 0:
            break
        targets[key] += 1
        remainder -= 1

    for i, (bucket, quota) in enumerate(targets.items()):
        available = cell[(cell["workflow_condition"] == bucket) & ~cell["project_id"].isin(selected_ids)]
        take = min(quota, len(available))
        if take > 0:
            part = _sample_n(available, take, seed + i, _weighted_by_burden(available))
            selected_parts.append(part)
            selected_ids.update(part["project_id"])

    selected = pd.concat(selected_parts, ignore_index=False) if selected_parts else cell.iloc[0:0]
    remaining_n = n - len(selected)
    if remaining_n > 0:
        available = cell[~cell["project_id"].isin(selected_ids)]
        fill = _sample_n(available, remaining_n, seed + 99, _weighted_by_burden(available))
        selected = pd.concat([selected, fill], ignore_index=False)

    return selected


def _load_existing_sample100() -> pd.DataFrame:
    if not SAMPLE100_PATH.exists():
        raise FileNotFoundError(f"Existing diagnostic v1 sample not found: {SAMPLE100_PATH}")
    sample = pd.read_csv(SAMPLE100_PATH)
    sample = sample[["project_id", "process_type", "project_energy_type"]].copy()
    sample["project_energy_type"] = sample["project_energy_type"].map(normalize_energy)
    return sample


def _build_split(
    frame: pd.DataFrame,
    split_name: str,
    targets: dict[tuple[str, str], int],
    excluded_ids: set[str],
    seed: int,
    enriched: bool,
    required_ids: set[str] | None = None,
) -> pd.DataFrame:
    required_ids = required_ids or set()
    parts = []
    for idx, ((process_type, energy_type), target_n) in enumerate(targets.items()):
        cell = frame[
            (frame["process_type"] == process_type)
            & (frame["project_energy_type"] == energy_type)
            & ~frame["project_id"].isin(excluded_ids)
        ].copy()

        required_cell = frame[
            frame["project_id"].isin(required_ids)
            & (frame["process_type"] == process_type)
            & (frame["project_energy_type"] == energy_type)
        ].copy()
        if len(required_cell) > target_n:
            raise ValueError(
                f"{split_name} has {len(required_cell)} required rows for "
                f"{process_type}/{energy_type}, target is {target_n}."
            )

        if not required_cell.empty:
            cell = cell[~cell["project_id"].isin(set(required_cell["project_id"]))]

        remaining = target_n - len(required_cell)
        if enriched:
            sampled = _sample_enriched_cell(cell, remaining, seed + idx * 1000)
        else:
            sampled = _sample_n(cell, remaining, seed + idx * 1000, None)

        combined = pd.concat([required_cell, sampled], ignore_index=False)
        parts.append(combined)

    split = pd.concat(parts, ignore_index=True)
    split["split"] = split_name
    return split


def _assign_sample_weights(split_df: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    weights = []
    universe_counts = frame.groupby(["process_type", "project_energy_type"]).size().to_dict()
    sample_counts = split_df.groupby(["process_type", "project_energy_type"]).size().to_dict()
    for _, row in split_df.iterrows():
        key = (row["process_type"], row["project_energy_type"])
        weights.append(universe_counts.get(key, 0) / max(1, sample_counts.get(key, 1)))
    return pd.Series(weights, index=split_df.index)


def _mark_irr(splits: pd.DataFrame, seed: int) -> pd.DataFrame:
    splits = splits.copy()
    splits["irr_required"] = False
    selected_ids: set[str] = set()

    for i, process_type in enumerate(PROCESS_TYPES):
        available = splits[(splits["process_type"] == process_type) & ~splits["project_id"].isin(selected_ids)]
        take = min(50, len(available))
        part = _sample_n(available, take, seed + i * 100, _weighted_by_burden(available))
        selected_ids.update(part["project_id"])

    high_amb = splits[
        splits["workflow_condition"].isin(["missing_failed", "ambiguous"])
        & ~splits["project_id"].isin(selected_ids)
    ]
    if not high_amb.empty:
        take = min(15, len(high_amb))
        part = _sample_n(high_amb, take, seed + 999, _weighted_by_burden(high_amb))
        selected_ids.update(part["project_id"])

    splits.loc[splits["project_id"].isin(selected_ids), "irr_required"] = True
    splits["irr_group"] = splits["irr_required"].map(lambda x: "double_review" if x else "")
    return splits


def _add_split_metadata(splits: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    source_hashes = _source_hashes()
    out = splits.copy()
    out["sample_seed"] = SEED
    out["sample_weight"] = 1.0
    rep_mask = out["split"] == "test_representative_v1"
    if rep_mask.any():
        out.loc[rep_mask, "sample_weight"] = _assign_sample_weights(out[rep_mask], frame)

    split_hashes = {}
    for split_name, group in out.groupby("split"):
        split_hashes[split_name] = _ids_hash(group["project_id"].astype(str).tolist())

    out["split_id_hash"] = out["split"].map(split_hashes)
    out["source_input_hashes"] = json.dumps(source_hashes, sort_keys=True)
    out["split_created_at"] = datetime.now(timezone.utc).isoformat()
    out["sample_stratum"] = (
        out["split"] + "/" + out["process_type"] + "/"
        + out["project_energy_type"] + "/" + out["doc_count_bin"].fillna("unknown")
    )
    return out


def _check_immutability(new_splits: pd.DataFrame, overwrite_existing: bool) -> None:
    if overwrite_existing or not SPLITS_PATH.exists():
        return
    existing = pd.read_parquet(SPLITS_PATH)
    if existing.empty:
        return
    for split_name in sorted(set(existing["split"]) & set(new_splits["split"])):
        old_ids = existing.loc[existing["split"] == split_name, "project_id"].astype(str).tolist()
        new_ids = new_splits.loc[new_splits["split"] == split_name, "project_id"].astype(str).tolist()
        if _ids_hash(old_ids) != _ids_hash(new_ids):
            raise SystemExit(
                f"Refusing to overwrite immutable split {split_name}: selected IDs differ. "
                "Use --overwrite-existing only when intentionally versioning/regenerating outputs."
            )


def build_gold_splits() -> pd.DataFrame:
    frame = build_sampling_frame()
    sample100 = _load_existing_sample100()
    sample100_ids = set(sample100["project_id"].astype(str))

    diag = _build_split(
        frame,
        "diagnostic_balanced_v2",
        DIAGNOSTIC_TARGET,
        excluded_ids=set(),
        seed=SEED,
        enriched=True,
        required_ids=sample100_ids,
    )
    diag["prior_split"] = diag["project_id"].map(lambda pid: "diagnostic_balanced_v1" if pid in sample100_ids else "")

    excluded = set(diag["project_id"].astype(str))
    train = _build_split(
        frame,
        "train_enriched_v1",
        TRAIN_TARGET,
        excluded_ids=excluded,
        seed=SEED + 10_000,
        enriched=True,
    )

    excluded.update(train["project_id"].astype(str))
    test = _build_split(
        frame,
        "test_representative_v1",
        TEST_TARGET,
        excluded_ids=excluded,
        seed=SEED + 20_000,
        enriched=False,
    )

    for part in [train, test]:
        part["prior_split"] = ""

    splits = pd.concat([diag, train, test], ignore_index=True)
    splits = _mark_irr(splits, SEED + 30_000)
    splits = _add_split_metadata(splits, frame)

    keep_cols = [
        "split", "prior_split", "project_id", "process_type", "project_energy_type",
        "sample_stratum", "sample_seed", "sample_weight", "irr_required", "irr_group",
        "workflow_condition", "timeline_status", "timeline_flags",
        "project_title", "lead_agency_harmonized", "project_department",
        "project_state", "project_county", "project_doc_count", "doc_count_bin",
        "total_pages", "max_document_pages", "appendix_count", "n_candidates",
        "n_relevant_candidates", "n_unknown_candidates", "n_historical_candidates",
        "n_packets", "max_packet_score", "has_metadata_packet", "noi_publication_date",
        "noi_match_status", "noi_match_confidence", "noa_availability_date",
        "noa_match_status", "split_id_hash", "source_input_hashes",
        "split_created_at", "sampling_frame_built_at",
    ]
    keep_cols = [c for c in keep_cols if c in splits.columns]
    return splits[keep_cols].sort_values(["split", "process_type", "project_energy_type", "project_id"]).reset_index(drop=True)


def write_outputs(splits: pd.DataFrame, dry_run: bool) -> None:
    print("Split counts:")
    print(pd.crosstab(splits["split"], splits["process_type"]).to_string())
    print("\nEnergy counts:")
    print(pd.crosstab([splits["split"], splits["process_type"]], splits["project_energy_type"]).to_string())
    print(f"\nIRR rows: {int(splits['irr_required'].sum())}")
    print(pd.crosstab(splits["process_type"], splits["irr_required"]).to_string())

    if dry_run:
        print("\n[dry-run] not writing outputs")
        return

    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    splits.to_parquet(SPLITS_PATH, index=False)
    print(f"\nWrote: {SPLITS_PATH}")

    for split_name, group in splits.groupby("split"):
        csv_path = GOLD_OUTPUT_DIR / f"{split_name}.csv"
        ids_path = GOLD_OUTPUT_DIR / f"{split_name}_ids.txt"
        manifest_path = GOLD_OUTPUT_DIR / f"{split_name}_manifest.json"
        group.to_csv(csv_path, index=False)
        ids_path.write_text("\n".join(group["project_id"].astype(str).tolist()) + "\n")
        manifest = {
            "split": split_name,
            "n": int(len(group)),
            "ids_hash": group["split_id_hash"].iloc[0],
            "sample_seed": SEED,
            "source_input_hashes": json.loads(group["source_input_hashes"].iloc[0]),
            "created_at": group["split_created_at"].iloc[0],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(f"Wrote: {csv_path}")
        print(f"Wrote: {ids_path}")
        print(f"Wrote: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D4 gold-standard sample splits.")
    parser.add_argument("--dry-run", action="store_true", help="Print summaries without writing outputs.")
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow replacing existing split outputs. Use only when intentionally regenerating.",
    )
    args = parser.parse_args()

    splits = build_gold_splits()
    _check_immutability(splits, overwrite_existing=args.overwrite_existing)
    write_outputs(splits, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
