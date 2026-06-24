import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "extract"))
from mitigation_conditions import extract_condition_rows  # noqa: E402

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, normalize_space, utc_now, write_parquet  # noqa: E402


PACKETS_PATH = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
SPANS_PATH = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
ASSIGNMENTS_PATH = D6_ANALYSIS_DIR / "project_action_archetypes.parquet"
ACTIONS_PATH = D6_ANALYSIS_DIR / "fonsi_actions.parquet"
CONDITIONS_PATH = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
REVIEW_PATH = D6_OUTPUT_DIR / "fonsi_extraction_review.csv"

SCALE_PATTERNS = {
    "acres": re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:acres?|ac\.)\b", re.I),
    "miles": re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:miles?|mi\.)\b", re.I),
    "megawatts": re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:mw|megawatts?)\b", re.I),
    "kilovolts": re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\s*(?:kv|kilovolts?)\b", re.I),
    "wells": re.compile(r"\b(\d[\d,]*)\s+(?:exploratory\s+|production\s+|injection\s+)?wells?\b", re.I),
}
ROAD_CONSTRAINT_RE = re.compile(
    r"\b(?:no|without)\s+(?:construction of\s+)?new access roads?\b|"
    r"\b(?:no|without)\s+(?:expansion|widening)\s+of\s+existing roads?\b",
    re.I,
)


def extract_scales(text: str) -> dict[str, list[float]]:
    scales = {}
    for name, pattern in SCALE_PATTERNS.items():
        values = sorted({float(match.replace(",", "")) for match in pattern.findall(text)})
        if values:
            scales[name] = values
    return scales


def primary_assignments() -> pd.DataFrame:
    if not ASSIGNMENTS_PATH.exists():
        return pd.DataFrame(columns=["project_id", "archetype_id", "assignment_method", "assignment_confidence"])
    assignments = pd.read_parquet(ASSIGNMENTS_PATH)
    return assignments.loc[
        assignments["is_primary_archetype"],
        ["project_id", "archetype_id", "assignment_method", "assignment_confidence"],
    ].drop_duplicates("project_id")


def action_rows(packets: pd.DataFrame, run_at: str) -> pd.DataFrame:
    assignments = primary_assignments()
    records = []
    for packet in packets.itertuples(index=False):
        text = normalize_space(packet.action_text or packet.analysis_text)
        scales = extract_scales(text)
        records.append(
            {
                "project_id": packet.project_id,
                "canonical_fonsi_document_id": packet.canonical_fonsi_document_id,
                "action_description": text[:12_000],
                "scale_values": json.dumps(scales, sort_keys=True),
                "max_acres": max(scales.get("acres", [float("nan")])),
                "max_miles": max(scales.get("miles", [float("nan")])),
                "max_megawatts": max(scales.get("megawatts", [float("nan")])),
                "max_kilovolts": max(scales.get("kilovolts", [float("nan")])),
                "max_wells": max(scales.get("wells", [float("nan")])),
                "has_no_new_access_road_constraint": bool(ROAD_CONSTRAINT_RE.search(text)),
                "action_extraction_method": "metadata_regex_and_packet_text",
                "action_confidence": "medium" if text else "low",
                "action_extraction_run_at": run_at,
                "action_llm_run_at": "",
                "llm_provider": "",
                "llm_model": "",
                "prompt_version": "",
            }
        )
    return pd.DataFrame(records).merge(assignments, how="left", on="project_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract structured D6 actions and conditions.")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--project-id", nargs="+", default=None)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-provider", choices=["ollama", "anthropic"], default="")
    parser.add_argument("--llm-model", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    run_at = utc_now()
    packets = pd.read_parquet(PACKETS_PATH)
    spans = pd.read_parquet(SPANS_PATH)
    if args.project_id:
        packets = packets.loc[packets["project_id"].astype(str).isin(args.project_id)]
    if args.sample:
        packets = packets.sample(min(args.sample, len(packets)), random_state=42)
    spans = spans.loc[spans["project_id"].isin(packets["project_id"])]
    condition_spans = spans.loc[
        spans["span_type"].isin(["condition", "boundary", "finding", "fallback"])
    ]
    actions = action_rows(packets, run_at)
    conditions = extract_condition_rows(
        condition_spans,
        use_llm=args.use_llm,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
    )
    write_parquet(actions, ACTIONS_PATH)
    write_parquet(conditions, CONDITIONS_PATH)
    review = actions.merge(
        conditions.groupby("project_id").size().rename("condition_row_count"),
        how="left",
        on="project_id",
    )
    review["condition_row_count"] = review["condition_row_count"].fillna(0).astype(int)
    review.sort_values(["condition_row_count", "project_id"]).head(150).to_csv(REVIEW_PATH, index=False)
    print(f"wrote {len(actions):,} action rows and {len(conditions):,} condition rows")


if __name__ == "__main__":
    main()

