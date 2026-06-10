import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import json
from html import escape

import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now


CANDIDATES_PATH = D6_ANALYSIS_DIR / "fonsi_candidate_categories.parquet"
PACKETS_PATH = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"
ACTIONS_PATH = D6_ANALYSIS_DIR / "fonsi_actions.parquet"
CONDITIONS_PATH = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
CROSSWALK_PATH = D6_ANALYSIS_DIR / "ce_crosswalk.parquet"
SHORTLIST_HTML = D6_OUTPUT_DIR / "fonsi_candidate_shortlist.html"
DOSSIER_DIR = D6_OUTPUT_DIR / "dossiers"


def table(df: pd.DataFrame, columns: list[str]) -> str:
    keep = [col for col in columns if col in df.columns]
    return df[keep].fillna("").to_html(index=False, escape=True, classes="data")


def page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 2rem; color: #222; max-width: 110rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.88rem; }}
th, td {{ border: 1px solid #ddd; padding: 0.42rem; vertical-align: top; }}
th {{ background: #f2f4f6; text-align: left; }}
.note {{ color: #444; max-width: 75rem; }} code {{ white-space: pre-wrap; }}
</style></head><body>{body}</body></html>"""


def representative_ids(value: str) -> list[str]:
    try:
        return [str(item) for item in json.loads(value)]
    except Exception:
        return []


def dossier(candidate: pd.Series, packets: pd.DataFrame, actions: pd.DataFrame, conditions: pd.DataFrame, crosswalk: pd.DataFrame) -> str:
    ids = representative_ids(candidate["representative_project_ids"])
    project_packets = packets.loc[packets["project_id"].astype(str).isin(ids)].copy()
    project_actions = actions.loc[actions["project_id"].astype(str).isin(ids)].copy()
    project_conditions = conditions.loc[conditions["project_id"].astype(str).isin(ids)].copy()
    ce_rows = crosswalk.loc[crosswalk["archetype_id"].eq(candidate["archetype_id"])].sort_values(
        ["retrieval_rank"]
    ).head(12)
    return page(
        f"D6 dossier: {candidate['archetype_label']}",
        f"""
<h1>{escape(candidate['archetype_label'])}</h1>
<p class="note">Stage A candidate dossier generated {escape(utc_now())}. This is a CATF
review aid. CE Explorer results are discovery links and require verification against
canonical agency materials. The heuristic tier is not a legal sufficiency conclusion.</p>
<h2>Candidate Summary</h2>
{table(pd.DataFrame([candidate]), [
    "recommendation_tier", "gating_flags", "archetype_description",
    "n_projects_classified", "n_fonsi_projects", "fonsi_share", "ce_share",
    "ea_share", "eis_share", "mitigation_dependence_share", "bmp_dependence_share",
    "monitoring_dependence_share", "repeated_limitations_and_scales", "qa_notes"
])}
<h2>Representative Projects</h2>
{table(project_packets, [
    "project_id", "project_title", "lead_agency_harmonized", "project_state",
    "canonical_fonsi_document_id", "evidence_span_count", "action_text", "boundary_text"
])}
<h2>Structured Actions</h2>
{table(project_actions, [
    "project_id", "archetype_id", "scale_values", "has_no_new_access_road_constraint",
    "action_description", "action_confidence"
])}
<h2>Condition Evidence</h2>
{table(project_conditions.head(150), [
    "project_id", "document_id", "page_number", "resource_area", "condition_role",
    "obligation_level", "condition_text", "confidence", "source_span_sha256"
])}
<h2>Existing CE Retrieval</h2>
{table(ce_rows, [
    "retrieval_rank", "match_type", "agency_unit", "structured_id", "ce_description",
    "canonical_source_url", "retrieval_score", "lexical_score", "embedding_cosine",
    "manual_verification_status"
])}
<h2>Open Review Gates</h2>
<p>{escape(candidate["gating_flags"])}</p>
""",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render CATF-facing D6 shortlist materials.")
    parser.add_argument("--limit", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_d6_dirs()
    DOSSIER_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(CANDIDATES_PATH)
    packets = pd.read_parquet(PACKETS_PATH)
    actions = pd.read_parquet(ACTIONS_PATH)
    conditions = pd.read_parquet(CONDITIONS_PATH)
    crosswalk = pd.read_parquet(CROSSWALK_PATH)
    tier_order = pd.CategoricalDtype(["advance", "review", "deprioritize"], ordered=True)
    candidates["tier_order"] = candidates["recommendation_tier"].astype(tier_order)
    shortlist = candidates.sort_values(
        ["tier_order", "assessment_total", "n_fonsi_projects"],
        ascending=[True, False, False],
    ).head(args.limit)
    dossier_links = {}
    for _, candidate in candidates.iterrows():
        filename = f"{candidate['archetype_id']}.html"
        (DOSSIER_DIR / filename).write_text(
            dossier(candidate, packets, actions, conditions, crosswalk),
            encoding="utf-8",
        )
        dossier_links[candidate["archetype_id"]] = (
            f'<li><a href="dossiers/{escape(filename)}">{escape(candidate["archetype_label"])}</a></li>'
        )
    links = [dossier_links[archetype_id] for archetype_id in shortlist["archetype_id"]]
    SHORTLIST_HTML.write_text(
        page(
            "D6 CATF candidate shortlist",
            f"""
<h1>D6 CATF Candidate Shortlist</h1>
<p class="note">Stage A shortlist generated {escape(utc_now())}. Select 2-4 candidates for
deeper policy and legal review before Stage B dossier substantiation.</p>
<ul>{''.join(links)}</ul>
{table(shortlist, [
    "archetype_label", "recommendation_tier", "gating_flags", "n_fonsi_projects",
    "fonsi_share", "ce_share", "eis_share", "assessment_total",
    "strongest_same_agency_ce", "strongest_other_agency_ce"
])}
""",
        ),
        encoding="utf-8",
    )
    print(
        f"wrote {len(shortlist):,}-candidate shortlist and "
        f"{len(candidates):,} review dossiers -> {DOSSIER_DIR}"
    )


if __name__ == "__main__":
    main()
