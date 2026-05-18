import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# D03 STEP 1: IDENTIFY VISUAL IMPACT CANDIDATES
# --------------------------
# Converts the reusable document section layer (build_document_sections.py)
# into a D03-ready visual impact extraction table. Run this before
# 02_build_nepa_reviews.py --section-layer.
#
# The extraction unit is the full detected section, not a small keyword window.
# Keyword/density signals only rank fallback candidates.
#
# Usage:
#   conda run -n nepa python phase2/code/deliverable03/01_identify_visual_impact_candidates.py

import argparse
import time
from datetime import datetime
from pathlib import Path

import duckdb


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
D03_ANALYSIS_DIR = ANALYSIS_DIR / "deliverable03"
OUTPUT_DIR = BASE_DIR / "output" / "deliverable03"
VALIDATION_DIR = BASE_DIR / "output" / "validation"

DEFAULT_SECTIONS = ANALYSIS_DIR / "document_sections.parquet"
DEFAULT_REVIEWS = D03_ANALYSIS_DIR / "projects_nepa_reviews.parquet"
DEFAULT_OUTPUT = D03_ANALYSIS_DIR / "visual_impact_sections_from_document_sections.parquet"
DEFAULT_PROJECT_TEXT = D03_ANALYSIS_DIR / "projects_visual_text_from_document_sections.parquet"
DEFAULT_PROJECT_COVERAGE = OUTPUT_DIR / "visual_impact_section_project_coverage.csv"
DEFAULT_COVERAGE_SUMMARY = OUTPUT_DIR / "visual_impact_section_coverage_summary.csv"
DEFAULT_QA = VALIDATION_DIR / "visual_impact_sections_from_document_sections_qa.csv"

VISUAL_HEADING_RE = (
    r"\b(aesthetics?|visual resources?|visual impacts?|visual effects?|"
    r"scenic resources?|scenery|viewsheds?|view[ -]?sheds?|"
    r"landscape character|glare|shadow flicker|light pollution|night sky|dark sky)\b"
)

COMBINED_RESOURCE_RE = (
    r"\b(land use|recreation|shoreline|wild and scenic).{0,80}"
    r"(aesthetics?|visual|scenic)\b|"
    r"\b(aesthetics?|visual|scenic).{0,80}"
    r"(land use|recreation|shoreline|wild and scenic)\b"
)


_START = time.time()


def log(msg: str) -> None:
    elapsed = (time.time() - _START) / 60
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ({elapsed:.1f}m) {msg}", flush=True)


def sql_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def unlink_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def build_visual_candidates(
    sections: Path,
    reviews: Path,
    output: Path,
    project_text_output: Path,
    project_coverage_output: Path,
    coverage_summary_output: Path,
    qa_output: Path,
    max_fallback_per_project: int = 5,
) -> None:
    for path in [
        output,
        project_text_output,
        project_coverage_output,
        coverage_summary_output,
        qa_output,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        unlink_if_exists(path)

    con = duckdb.connect()
    section_sql = f"read_parquet('{sql_quote(sections)}')"
    review_sql = f"read_parquet('{sql_quote(reviews)}')"
    visual_re = sql_quote(VISUAL_HEADING_RE)
    combined_re = sql_quote(COMBINED_RESOURCE_RE)

    log("building section-level visual impact candidates")
    con.execute(f"""
        CREATE TEMP TABLE visual_candidates AS
        WITH scored AS (
            SELECT
                s.*,
                lower(
                    coalesce(s.heading_title, '') || ' ' ||
                    coalesce(s.parent_heading_title, '') || ' ' ||
                    coalesce(s.heading_raw, '')
                ) AS heading_blob,
                1000.0 * s.visual_term_count / greatest(s.section_words, 1) AS visual_terms_per_1000,
                1000.0 * s.impact_term_count / greatest(s.section_words, 1) AS impact_terms_per_1000,
                CASE
                    WHEN s.section_topic_guess = 'visual' THEN 1
                    WHEN regexp_extract(lower(coalesce(s.heading_title, '')), '{visual_re}') != '' THEN 1
                    WHEN regexp_extract(lower(coalesce(s.parent_heading_title, '')), '{visual_re}') != '' THEN 2
                    WHEN regexp_extract(
                        lower(coalesce(s.heading_title, '') || ' ' || coalesce(s.parent_heading_title, '')),
                        '{combined_re}'
                    ) != '' THEN 3
                    WHEN s.section_topic_guess IN ('land_use', 'recreation')
                     AND s.visual_term_count >= 5
                     AND s.impact_term_count >= 2
                     AND 1000.0 * s.visual_term_count / greatest(s.section_words, 1) >= 0.75
                     AND s.section_words BETWEEN 50 AND 20000 THEN 5
                    WHEN s.visual_term_count >= 5
                     AND s.impact_term_count >= 2
                     AND 1000.0 * s.visual_term_count / greatest(s.section_words, 1) >= 1.0
                     AND s.section_words BETWEEN 50 AND 20000 THEN 8
                    ELSE NULL
                END AS candidate_priority,
                CASE
                    WHEN s.section_topic_guess = 'visual' THEN 'primary_visual_heading'
                    WHEN regexp_extract(lower(coalesce(s.heading_title, '')), '{visual_re}') != '' THEN 'visual_heading_regex'
                    WHEN regexp_extract(lower(coalesce(s.parent_heading_title, '')), '{visual_re}') != '' THEN 'visual_parent_heading'
                    WHEN regexp_extract(
                        lower(coalesce(s.heading_title, '') || ' ' || coalesce(s.parent_heading_title, '')),
                        '{combined_re}'
                    ) != '' THEN 'combined_resource_heading'
                    WHEN s.section_topic_guess IN ('land_use', 'recreation')
                     AND s.visual_term_count >= 5
                     AND s.impact_term_count >= 2
                     AND 1000.0 * s.visual_term_count / greatest(s.section_words, 1) >= 0.75
                     AND s.section_words BETWEEN 50 AND 20000 THEN 'mixed_resource_high_signal'
                    WHEN s.visual_term_count >= 5
                     AND s.impact_term_count >= 2
                     AND 1000.0 * s.visual_term_count / greatest(s.section_words, 1) >= 1.0
                     AND s.section_words BETWEEN 50 AND 20000 THEN 'high_density_fallback'
                    ELSE NULL
                END AS candidate_reason
            FROM {section_sql} s
            WHERE s.process_type IN ('EA', 'EIS')
              AND s.energy_group IN ('Decarbonization', 'Fossil Fuel')
              AND s.section_words >= 30
              AND NOT s.very_long_section
              AND NOT s.large_page_span
        ),
        candidates AS (
            SELECT *
            FROM scored
            WHERE candidate_priority IS NOT NULL
        ),
        ranked AS (
            SELECT
                candidates.*,
                min(candidate_priority) OVER (PARTITION BY project_id) AS project_best_priority,
                row_number() OVER (
                    PARTITION BY project_id
                    ORDER BY
                        candidate_priority,
                        visual_term_count DESC,
                        impact_term_count DESC,
                        section_words ASC,
                        page_start
                ) AS candidate_rank
            FROM candidates
        )
        SELECT
            project_id,
            document_id,
            process_type,
            energy_group,
            tech_group,
            lead_agency_harmonized,
            document_title,
            source,
            page_start,
            page_end,
            line_start,
            line_end,
            char_start,
            char_end,
            heading_raw,
            heading_number,
            heading_title,
            heading_level,
            parent_heading_number,
            parent_heading_title,
            section_topic_guess,
            section_words,
            section_chars,
            visual_term_count,
            impact_term_count,
            visual_terms_per_1000,
            impact_terms_per_1000,
            visual_impact_signal,
            candidate_reason,
            candidate_priority,
            candidate_rank,
            CASE
                WHEN candidate_priority <= 3 THEN 'structural_heading'
                ELSE 'full_section_fallback'
            END AS extraction_unit,
            section_text
        FROM ranked
        WHERE candidate_priority <= 3
           OR (
               project_best_priority > 3
               AND candidate_rank <= {max_fallback_per_project}
           )
    """)

    con.execute(f"""
        COPY (
            SELECT *
            FROM visual_candidates
            ORDER BY process_type, energy_group, project_id, candidate_priority, page_start
        )
        TO '{sql_quote(output)}' (FORMAT PARQUET)
    """)

    log("building project-level concatenated visual text")
    con.execute(f"""
        COPY (
            SELECT
                project_id,
                any_value(process_type) AS process_type,
                any_value(energy_group) AS energy_group,
                any_value(tech_group) AS tech_group,
                any_value(lead_agency_harmonized) AS lead_agency_harmonized,
                count(*) AS n_visual_sections,
                sum(section_words) AS visual_section_words,
                min(candidate_priority) AS best_candidate_priority,
                string_agg(
                    '[[' || candidate_reason || ' | ' ||
                    coalesce(heading_number, '') || ' ' || coalesce(heading_title, '') ||
                    ' | pages ' || page_start::VARCHAR || '-' || page_end::VARCHAR || ']]' ||
                    chr(10) || section_text,
                    chr(10) || chr(10)
                    ORDER BY candidate_priority, page_start
                ) AS visual_section_text
            FROM visual_candidates
            GROUP BY project_id
            ORDER BY process_type, energy_group, project_id
        )
        TO '{sql_quote(project_text_output)}' (FORMAT PARQUET)
    """)

    log("writing project coverage outputs")
    con.execute(f"""
        COPY (
            WITH universe AS (
                SELECT DISTINCT
                    project_id,
                    process_type,
                    energy_group,
                    tech_group,
                    lead_agency_harmonized
                FROM {review_sql}
                WHERE process_type IN ('EA', 'EIS')
                  AND energy_group IN ('Decarbonization', 'Fossil Fuel')
            ),
            candidate_counts AS (
                SELECT
                    project_id,
                    count(*) AS n_visual_sections,
                    sum(section_words) AS visual_section_words,
                    min(candidate_priority) AS best_candidate_priority,
                    string_agg(DISTINCT candidate_reason, '; ' ORDER BY candidate_reason) AS candidate_reasons
                FROM visual_candidates
                GROUP BY project_id
            )
            SELECT
                u.*,
                coalesce(c.n_visual_sections, 0) AS n_visual_sections,
                coalesce(c.visual_section_words, 0) AS visual_section_words,
                c.best_candidate_priority,
                c.candidate_reasons,
                coalesce(c.n_visual_sections, 0) > 0 AS has_visual_candidate
            FROM universe u
            LEFT JOIN candidate_counts c USING (project_id)
            ORDER BY process_type, energy_group, project_id
        )
        TO '{sql_quote(project_coverage_output)}' (HEADER, DELIMITER ',')
    """)

    con.execute(f"""
        COPY (
            SELECT
                process_type,
                energy_group,
                count(*) AS n_projects,
                sum(has_visual_candidate::INTEGER) AS projects_with_visual_candidate,
                round(100.0 * sum(has_visual_candidate::INTEGER) / count(*), 1) AS pct_with_visual_candidate,
                median(visual_section_words) FILTER (WHERE has_visual_candidate) AS median_visual_section_words
            FROM read_csv_auto('{sql_quote(project_coverage_output)}')
            GROUP BY 1, 2
            ORDER BY 1, 2
        )
        TO '{sql_quote(coverage_summary_output)}' (HEADER, DELIMITER ',')
    """)

    log("writing candidate QA sample")
    con.execute(f"""
        COPY (
            WITH qa AS (
                SELECT
                    project_id,
                    document_id,
                    process_type,
                    energy_group,
                    tech_group,
                    candidate_reason,
                    candidate_priority,
                    candidate_rank,
                    extraction_unit,
                    heading_raw,
                    heading_number,
                    heading_title,
                    parent_heading_title,
                    page_start,
                    page_end,
                    section_words,
                    visual_term_count,
                    impact_term_count,
                    round(visual_terms_per_1000, 2) AS visual_terms_per_1000,
                    round(impact_terms_per_1000, 2) AS impact_terms_per_1000,
                    substr(section_text, 1, 700) AS section_start_excerpt,
                    substr(section_text, greatest(length(section_text) - 699, 1), 700) AS section_end_excerpt,
                    row_number() OVER (
                        PARTITION BY candidate_reason
                        ORDER BY random()
                    ) AS reason_sample_rank
                FROM visual_candidates
            )
            SELECT * EXCLUDE(reason_sample_rank)
            FROM qa
            WHERE reason_sample_rank <= 40
            ORDER BY candidate_priority, candidate_reason, project_id, page_start
        )
        TO '{sql_quote(qa_output)}' (HEADER, DELIMITER ',')
    """)

    summary = con.execute("""
        SELECT candidate_reason, count(*) AS n_sections, count(DISTINCT project_id) AS n_projects
        FROM visual_candidates
        GROUP BY 1
        ORDER BY min(candidate_priority), n_sections DESC
    """).fetchdf()
    coverage = con.execute(f"SELECT * FROM read_csv_auto('{sql_quote(coverage_summary_output)}')").fetchdf()

    log(f"wrote candidates -> {output}")
    log(f"wrote project text -> {project_text_output}")
    log("candidate summary:\n" + summary.to_string(index=False))
    log("coverage summary:\n" + coverage.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build D03 visual-impact full-section candidates from document_sections.parquet."
    )
    parser.add_argument("--sections", type=Path, default=DEFAULT_SECTIONS)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--project-text-output", type=Path, default=DEFAULT_PROJECT_TEXT)
    parser.add_argument("--project-coverage-output", type=Path, default=DEFAULT_PROJECT_COVERAGE)
    parser.add_argument("--coverage-summary-output", type=Path, default=DEFAULT_COVERAGE_SUMMARY)
    parser.add_argument("--qa-output", type=Path, default=DEFAULT_QA)
    parser.add_argument("--max-fallback-per-project", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_visual_candidates(
        sections=args.sections,
        reviews=args.reviews,
        output=args.output,
        project_text_output=args.project_text_output,
        project_coverage_output=args.project_coverage_output,
        coverage_summary_output=args.coverage_summary_output,
        qa_output=args.qa_output,
        max_fallback_per_project=args.max_fallback_per_project,
    )
