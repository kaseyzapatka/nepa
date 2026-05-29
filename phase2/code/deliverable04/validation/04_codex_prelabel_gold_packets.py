"""
Create Codex-labeled review packet copies for gold-standard timeline review.

This is a prelabeling pass, not final human-verified gold. It fills gold_* fields
where the refreshed D4 candidate/selection outputs provide candidate-backed
evidence, and marks no-candidate projects for source-document review.

Outputs:
    phase2/output/deliverable04/gold/codex_labels/<split>_batchNNN_projects_codex_labeled.csv
    phase2/output/deliverable04/gold/codex_labels/<split>_batchNNN_candidates_codex_labeled.csv
    phase2/output/deliverable04/gold/codex_labels/<split>_codex_label_summary.csv

Usage:
    python 13_codex_prelabel_gold_packets.py --split diagnostic_balanced_v2
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
PHASE2 = ROOT / "phase2"
TIMELINE_DIR = PHASE2 / "data" / "analysis" / "timeline"
PACKET_DIR = PHASE2 / "output" / "deliverable04" / "gold" / "review_packets"
OUTPUT_DIR = PHASE2 / "output" / "deliverable04" / "gold" / "codex_labels"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"

ROLE_TO_TYPE = {
    "clear_initiation": "clear",
    "proxy_initiation": "proxy",
    "clear_decision": "clear",
    "proxy_decision": "proxy",
}

CANDIDATE_ERROR_CATEGORY = {
    "historical": "historical_project",
    "review": "specialist_review",
    "reject": "other",
    "unknown": "other",
}


def _clean(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _iso_date(value: object) -> str:
    text = _clean(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def _confidence(value: object) -> str:
    text = _clean(value).lower()
    if text in {"high", "medium", "low"}:
        return text
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "low"
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


def _project_packet_paths(split: str) -> list[Path]:
    return sorted(PACKET_DIR.glob(f"{split}_batch*_projects.csv"))


def _candidate_path_for(project_path: Path) -> Path:
    return project_path.with_name(project_path.name.replace("_projects.csv", "_candidates.csv"))


def _packet_ids_with_context() -> set[str]:
    if not PACKETS_PATH.exists():
        return set()
    packets = pd.read_parquet(PACKETS_PATH, columns=["project_id"])
    return set(packets["project_id"].astype(str))


def _selected_candidate(cands: pd.DataFrame, role: str) -> pd.Series | None:
    if cands.empty:
        return None
    selected_col = f"selected_for_{role}"
    role_values = [f"clear_{role}", f"proxy_{role}"]
    if selected_col in cands.columns:
        sub = cands[
            cands[selected_col].fillna(False).astype(bool)
            & cands["candidate_role"].isin(role_values)
        ].copy()
        if not sub.empty:
            return sub.sort_values("ranking_score", ascending=False).iloc[0]
    sub = cands[cands["candidate_role"].isin(role_values)].copy()
    if sub.empty:
        return None
    return sub.sort_values("ranking_score", ascending=False).iloc[0]


def _fill_role_from_candidate(row: pd.Series, cand: pd.Series, role: str) -> dict:
    candidate_role = _clean(cand.get("candidate_role"))
    return {
        f"gold_{role}_date": _iso_date(cand.get("parsed_date")),
        f"gold_{role}_granularity": _clean(cand.get("date_granularity")) or "unknown",
        f"gold_{role}_type": ROLE_TO_TYPE.get(candidate_role, "proxy"),
        f"gold_{role}_source_type": _clean(cand.get("candidate_source_type")) or _clean(cand.get("source_tier")),
        f"gold_{role}_candidate_id": _clean(cand.get("candidate_id")),
        f"gold_{role}_document_id": _clean(cand.get("document_id")),
        f"gold_{role}_page_number": _clean(cand.get("page_number")),
        f"gold_{role}_evidence_text": _clean(cand.get("context_text")),
        f"gold_{role}_confidence": _confidence(cand.get("role_confidence")),
        f"gold_{role}_missing_reason": "",
    }


def _fill_role_from_project(row: pd.Series, role: str) -> dict | None:
    date_val = row.get(f"{role}_date")
    if not _clean(date_val):
        return None
    is_proxy = bool(row.get(f"{role}_is_proxy", False))
    evidence = _clean(row.get(f"{role}_evidence_text"))
    if not evidence:
        return None
    return {
        f"gold_{role}_date": _iso_date(date_val),
        f"gold_{role}_granularity": _clean(row.get(f"{role}_date_granularity")) or "unknown",
        f"gold_{role}_type": "proxy" if is_proxy else "clear",
        f"gold_{role}_source_type": _clean(row.get(f"{role}_source_type")),
        f"gold_{role}_candidate_id": "",
        f"gold_{role}_document_id": _clean(row.get(f"{role}_document_id")),
        f"gold_{role}_page_number": _clean(row.get(f"{role}_page_number")),
        f"gold_{role}_evidence_text": evidence,
        f"gold_{role}_confidence": _confidence(row.get(f"{role}_confidence")),
        f"gold_{role}_missing_reason": "",
    }


def _missing_reason(project_id: str, cands: pd.DataFrame, packet_project_ids: set[str]) -> str:
    if project_id not in packet_project_ids:
        return "retrieval_miss"
    if cands.empty:
        return "parser_miss"
    return "no_evidence"


def _fill_missing_role(project_id: str, cands: pd.DataFrame, packet_project_ids: set[str], role: str) -> dict:
    return {
        f"gold_{role}_date": "",
        f"gold_{role}_granularity": "unknown",
        f"gold_{role}_type": "missing",
        f"gold_{role}_source_type": "",
        f"gold_{role}_candidate_id": "",
        f"gold_{role}_document_id": "",
        f"gold_{role}_page_number": "",
        f"gold_{role}_evidence_text": "",
        f"gold_{role}_confidence": "low",
        f"gold_{role}_missing_reason": _missing_reason(project_id, cands, packet_project_ids),
    }


def label_projects(projects: pd.DataFrame, candidates: pd.DataFrame, packet_project_ids: set[str]) -> pd.DataFrame:
    out = projects.copy()
    label_cols = [
        col for col in out.columns
        if col.startswith("gold_")
        or col in {"reviewer", "review_status", "reconciler", "adjudication_reason", "adjudicated_at"}
    ]
    for col in label_cols:
        out[col] = out[col].fillna("").astype(object)

    cand_groups = {
        str(pid): group.copy()
        for pid, group in candidates.groupby("project_id", sort=False)
    } if not candidates.empty else {}

    label_statuses = []
    for idx, row in out.iterrows():
        project_id = str(row["project_id"])
        cands = cand_groups.get(project_id, pd.DataFrame())
        role_status = {}

        for role in ["initiation", "decision"]:
            selected = _selected_candidate(cands, role)
            values = None
            if selected is not None:
                values = _fill_role_from_candidate(row, selected, role)
                role_status[role] = "candidate_backed"
            else:
                values = _fill_role_from_project(row, role)
                if values is not None:
                    role_status[role] = "project_selection_backed"
                else:
                    values = _fill_missing_role(project_id, cands, packet_project_ids, role)
                    role_status[role] = values[f"gold_{role}_missing_reason"]

            for col, value in values.items():
                out.at[idx, col] = value

        out.at[idx, "reviewer"] = "codex"
        out.at[idx, "gold_ambiguity_flag"] = "|".join(
            sorted({v for v in role_status.values() if v not in {"candidate_backed", "project_selection_backed"}})
        )
        if cands.empty:
            out.at[idx, "review_status"] = "needs_source_review_no_candidates"
            out.at[idx, "gold_notes"] = (
                "Codex prelabel: no extracted timeline candidates after D4 stages 02-04; "
                "source-document review required."
            )
        elif "no_evidence" in role_status.values():
            out.at[idx, "review_status"] = "codex_prelabeled_partial"
            out.at[idx, "gold_notes"] = (
                "Codex prelabel from selected D4 candidates where present; missing role has no candidate evidence. "
                "Human verification required."
            )
        else:
            out.at[idx, "review_status"] = "codex_prelabeled_candidate_backed"
            out.at[idx, "gold_notes"] = (
                "Codex prelabel from selected D4 candidates. Human verification required."
            )

    return out


def label_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    if out.empty:
        return out
    for col in [
        "gold_candidate_role", "gold_selected_for", "gold_error_category",
        "gold_candidate_notes", "reviewer", "candidate_review_status",
    ]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(object)

    out["gold_candidate_role"] = out["candidate_role"].fillna("").astype(str)
    out["gold_selected_for"] = "none"
    out.loc[out["selected_for_initiation"].fillna(False).astype(bool), "gold_selected_for"] = "initiation"
    out.loc[out["selected_for_decision"].fillna(False).astype(bool), "gold_selected_for"] = "decision"
    alt_mask = (
        out["gold_selected_for"].eq("none")
        & out["candidate_role"].isin([
            "clear_initiation", "proxy_initiation", "clear_decision", "proxy_decision",
        ])
    )
    out.loc[alt_mask, "gold_selected_for"] = "alternate_valid"
    out["gold_error_category"] = out["candidate_role"].map(CANDIDATE_ERROR_CATEGORY).fillna("")
    out["gold_candidate_notes"] = "Codex prelabel from D4 candidate role and selected flags; human verification required."
    out["reviewer"] = "codex"
    out["candidate_review_status"] = "codex_prelabeled"
    return out


def run(split: str) -> None:
    packet_project_ids = _packet_ids_with_context()
    project_paths = _project_packet_paths(split)
    if not project_paths:
        raise FileNotFoundError(f"No project packets found for split {split!r} in {PACKET_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    all_project_parts = []
    all_candidate_parts = []
    for project_path in project_paths:
        candidate_path = _candidate_path_for(project_path)
        projects = pd.read_csv(project_path)
        candidates = pd.read_csv(candidate_path) if candidate_path.exists() else pd.DataFrame()

        labeled_projects = label_projects(projects, candidates, packet_project_ids)
        labeled_candidates = label_candidates(candidates)
        all_project_parts.append(labeled_projects)
        if not labeled_candidates.empty:
            all_candidate_parts.append(labeled_candidates)

        batch_id = project_path.name.replace("_projects.csv", "")
        project_out = OUTPUT_DIR / f"{batch_id}_projects_codex_labeled.csv"
        candidate_out = OUTPUT_DIR / f"{batch_id}_candidates_codex_labeled.csv"
        labeled_projects.to_csv(project_out, index=False)
        labeled_candidates.to_csv(candidate_out, index=False)

        summary_rows.append({
            "batch_id": batch_id,
            "n_projects": len(labeled_projects),
            "n_candidate_rows": len(labeled_candidates),
            "n_projects_with_candidates": int(candidates["project_id"].nunique()) if not candidates.empty else 0,
            "n_candidate_backed_projects": int(
                (labeled_projects["review_status"] == "codex_prelabeled_candidate_backed").sum()
            ),
            "n_partial_projects": int((labeled_projects["review_status"] == "codex_prelabeled_partial").sum()),
            "n_no_candidate_projects": int(
                (labeled_projects["review_status"] == "needs_source_review_no_candidates").sum()
            ),
            "project_output": str(project_out),
            "candidate_output": str(candidate_out),
        })
        print(f"Wrote: {project_out}")
        print(f"Wrote: {candidate_out}")

    summary = pd.DataFrame(summary_rows)
    combined_projects = pd.concat(all_project_parts, ignore_index=True)
    combined_candidates = (
        pd.concat(all_candidate_parts, ignore_index=True)
        if all_candidate_parts else pd.DataFrame()
    )
    combined_projects_path = OUTPUT_DIR / f"{split}_projects_codex_labeled.csv"
    combined_candidates_path = OUTPUT_DIR / f"{split}_candidates_codex_labeled.csv"
    combined_projects.to_csv(combined_projects_path, index=False)
    combined_candidates.to_csv(combined_candidates_path, index=False)

    summary_path = OUTPUT_DIR / f"{split}_codex_label_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote: {combined_projects_path}")
    print(f"Wrote: {combined_candidates_path}")
    print(f"Wrote: {summary_path}")
    print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prelabel D4 gold review packets with Codex labels.")
    parser.add_argument("--split", required=True, choices=["diagnostic_balanced_v2", "train_enriched_v1", "test_representative_v1"])
    args = parser.parse_args()
    run(args.split)


if __name__ == "__main__":
    main()
