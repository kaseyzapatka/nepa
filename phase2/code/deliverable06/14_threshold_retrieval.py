"""D6 v2 — 14: retrieve significance-threshold language from evidence spans (item #44).

span_type=='boundary' is nearly empty (18 rows), so this searches finding/condition/
resource spans for threshold PHRASES instead — deterministic regex retrieval only,
no LLM call. Feeds the report's threshold-coverage quantification.

Input:
  data/analysis/deliverable06/fonsi_evidence_spans.parquet (large — read via DuckDB)

Output:
  output/deliverable06/threshold_candidates.csv
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import re

import duckdb
import pandas as pd

from common import D6_ANALYSIS_DIR, D6_OUTPUT_DIR, ensure_d6_dirs, utc_now

SPANS = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
OUT = D6_OUTPUT_DIR / "threshold_candidates.csv"

SPAN_TYPES = ("finding", "condition", "resource")

# canonical phrase label -> compiled regex (case-insensitive)
THRESHOLD_PATTERNS = {
    "would be significant if": re.compile(r"would be significant if", re.IGNORECASE),
    "would require an eis": re.compile(
        r"would require an eis"
        r"|require an environmental impact statement"
        r"|preparation of an environmental impact statement",
        re.IGNORECASE,
    ),
    "not to exceed": re.compile(r"not to exceed", re.IGNORECASE),
    "no new access road": re.compile(r"no new access road", re.IGNORECASE),
    "within existing right-of-way": re.compile(
        r"within (?:the )?existing right-of-way|within existing right of way", re.IGNORECASE
    ),
    "extraordinary circumstance": re.compile(r"extraordinary circumstances?", re.IGNORECASE),
}

# a single ILIKE-friendly filter to push the row selection down into DuckDB
DUCKDB_WHERE = " OR ".join(
    f"span_text ILIKE '%{needle}%'"
    for needle in (
        "would be significant if",
        "would require an eis",
        "require an environmental impact statement",
        "preparation of an environmental impact statement",
        "not to exceed",
        "no new access road",
        "within existing right-of-way",
        "within existing right of way",
        "extraordinary circumstance",
    )
)


def make_snippet(text: str, match_start: int, match_end: int, pad: int = 120) -> str:
    start = max(0, match_start - pad)
    end = min(len(text), match_end + pad)
    snippet = text[start:end]
    return re.sub(r"\s+", " ", snippet).strip()


def main() -> None:
    ensure_d6_dirs()
    run_at = utc_now()

    span_types_sql = ", ".join(f"'{t}'" for t in SPAN_TYPES)
    query = f"""
        SELECT project_id, span_type, heading_title, span_text
        FROM read_parquet('{SPANS}')
        WHERE span_type IN ({span_types_sql})
          AND ({DUCKDB_WHERE})
    """
    matched = duckdb.connect().execute(query).df()

    total_projects = duckdb.connect().execute(
        f"SELECT COUNT(DISTINCT project_id) AS n FROM read_parquet('{SPANS}')"
    ).df()["n"].iloc[0]

    rows = []
    for r in matched.itertuples(index=False):
        text = r.span_text or ""
        for phrase, pattern in THRESHOLD_PATTERNS.items():
            m = pattern.search(text)
            if not m:
                continue
            rows.append({
                "project_id": r.project_id,
                "span_type": r.span_type,
                "matched_phrase": phrase,
                "heading_title": r.heading_title,
                "snippet": make_snippet(text, m.start(), m.end()),
                "threshold_extraction_run_at": run_at,
                "threshold_llm_run_at": "",
            })

    out = pd.DataFrame(rows, columns=[
        "project_id", "span_type", "matched_phrase", "heading_title", "snippet",
        "threshold_extraction_run_at", "threshold_llm_run_at",
    ])
    out.to_csv(OUT, index=False)

    n_projects_matched = out["project_id"].nunique() if len(out) else 0
    print(f"total candidate rows: {len(out)}")
    print(f"distinct projects with >=1 threshold candidate: {n_projects_matched} (of {total_projects} in corpus)")
    print("counts by matched_phrase:")
    print(out["matched_phrase"].value_counts().to_string() if len(out) else "  (none)")
    print("counts by span_type:")
    print(out["span_type"].value_counts().to_string() if len(out) else "  (none)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
