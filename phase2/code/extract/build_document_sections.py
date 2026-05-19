import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

# --------------------------
# PROJECT DOCUMENT SECTION LAYER
# --------------------------
# Builds a reusable section-span index for processed EA/EIS project documents.
#
# The immediate use case is D03 visual-impact extraction, but the output is
# general-purpose: one row per detected document section, with page/line/char
# spans and a lightweight topic guess.
#
# Usage:
#   conda run -n nepa python phase2/code/extract/build_document_sections.py --process EA EIS --main-only --sample 100
#   conda run -n nepa python phase2/code/extract/build_document_sections.py --process EIS --project-id <project_id>
#   conda run -n nepa python phase2/code/extract/build_document_sections.py --process EA EIS --main-only

import argparse
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import duckdb
import pandas as pd


# --------------------------
# PATHS
# --------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # nepa/phase2/
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
D03_REVIEWS = ANALYSIS_DIR / "deliverable03" / "projects_nepa_reviews.parquet"

DEFAULT_OUTPUT = ANALYSIS_DIR / "document_sections.parquet"
DEFAULT_QA_OUTPUT = BASE_DIR / "output" / "validation" / "document_sections_qa.csv"


# --------------------------
# REGEXES
# --------------------------

NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:(?:section|chapter)\s+)?"
    r"(?P<number>[A-Z]?\d+(?:[.\-]\d+){0,6})"
    r"(?P<sep>[.)]?)\s+"
    r"(?P<title>[A-Za-z][A-Za-z0-9 /,&():';+\-\u2013\u2014]{1,140})\s*$",
    re.IGNORECASE,
)

ROMAN_HEADING_RE = re.compile(
    r"^\s*(?P<number>[IVXLCM]{1,8})[.)]\s+"
    r"(?P<title>[A-Za-z][A-Za-z0-9 /,&():';+\-\u2013\u2014]{1,140})\s*$",
    re.IGNORECASE,
)

SHORT_TITLE_RE = re.compile(
    r"^\s*(?P<title>[A-Z][A-Za-z0-9 /,&():';+\-\u2013\u2014]{2,80})\s*$"
)

TOC_DOT_RE = re.compile(r"\.{3,}")
PAGE_NUM_RE = re.compile(r"^\s*(?:page\s*)?\d{1,4}\s*$", re.IGNORECASE)
STANDALONE_SECTION_NUM_RE = re.compile(r"^\s*[A-Z]?\d+(?:[.\-]\d+){1,6}\s*$")
LOWERCASE_START_RE = re.compile(r"^[a-z]")
BULLET_START_RE = re.compile(r"^[\u2022\u2023\u25e6\u2043\u2219\-\*]\s")
CAPTION_START_RE = re.compile(
    r"^\s*(?:table|figure|fig\.|map|photo|photograph|appendix table)\s+[\w.\-]+",
    re.IGNORECASE,
)
STRUCTURAL_PAGE_RE = re.compile(
    r"\btable\s+of\s+contents\b"
    r"|\blist\s+of\s+(?:figures|tables|preparers?|authors?|contributors?)\b"
    r"|\bchapter\s+\d+[\s\-–]+(?:preparers?|references?|bibliography)\b"
    r"|\bsection\s+\d+[\s\-–]+(?:preparers?|references?|bibliography)\b"
    r"|\blist\s+of\s+references\b"
    r"|\bprepared\s+by\b",
    re.IGNORECASE,
)

KNOWN_UNNUMBERED_HEADINGS = {
    "affected environment",
    "environmental consequences",
    "environmental impacts",
    "direct and indirect effects",
    "cumulative impacts",
    "mitigation measures",
    "mitigation",
    "no action alternative",
    "proposed action",
    "alternatives",
    "purpose and need",
    "need for action",
    "decision",
    "finding of no significant impact",
    "record of decision",
    "visual resources",
    "aesthetics",
    "visual resource management",
    "land use",
    "recreation",
    "cultural resources",
    "biological resources",
    "wildlife",
    "vegetation",
    "water resources",
    "air quality",
    "noise",
    "traffic and transportation",
    "socioeconomics",
    "visual impacts",
    "visual effects",
    "scenic resources",
    "scenic quality",
    "viewshed",
    "viewshed analysis",
    "landscape character",
    "glare",
    "shadow flicker",
    "wild and scenic rivers",
    "scenic backways and byways",
    "floodplain encroachment",
    "airport compatibility",
    "federally listed species",
    "natural resources",
}

SENTENCEY_TITLE_RE = re.compile(
    r"\b(?:determined|shows?|would|could|should|shall|will|may|because|during|"
    r"associated|anticipated|located|cross(?:es|ing)?|consistency|acres?|miles?|"
    r"percent|gpm|definition|determination\s+key|within|due|btu|jlglg|"
    r"reported|provided|estimated|identified|concluded|developed|exhibit|"
    r"upstream|downstream|adjacent|pursuant|whereas|therefore)\b",
    re.IGNORECASE,
)

# Lines that look like mailing addresses, office locations, or regulatory citations.
# These slip through numbered-heading detection because they start with a building
# number followed by title-case text.
ADDRESS_CITATION_RE = re.compile(
    r"\b(?:Room|Suite|Ste\.?|Rm\.?|Floor)\s+\w+|"
    # Street address: building number + 1-3 words + street type
    # Catches "20 M Street SE", "464 W 4th Street", "777 East Tahquitz Canyon Way"
    r"\b\d+(?:\s+\w+){1,3}\s+(?:Street|St\.|Avenue|Ave\.|Boulevard|Blvd\.|Drive|Dr\.|Road|Rd\.|Lane|Ln\.|Court|Ct\.|Place|Pl\.)\b|"
    r"\b(?:[A-Z]{2,6}\s+)?Part\s+\d+[)\s]|"              # AAC Part 75), CFR Part 50
    r"\b(?:Section|§)\s+\d+\.\d+",                        # Section 7.2, § 9.3
    re.IGNORECASE,
)

WEAK_TITLE_START_RE = re.compile(
    r"^(?:and|or|but|as|of|to|for|by|with|from|under|presents?)\b",
    re.IGNORECASE,
)

MAP_TEXT_RE = re.compile(r"(?:\bN\s*A\s*V\s*A\s*J\s*O\b|^[A-Z](?:\s+[A-Z]){3,}$)")
CHAPTER_TITLE_RE = re.compile(
    r"^(?:introduction|purpose and need|proposed action|alternatives?|"
    r"affected environment|environmental consequences|environmental impacts|"
    r"consultation|coordination|references?|appendix|mitigation|decision|"
    r"cumulative impacts?)\b",
    re.IGNORECASE,
)

TOPIC_PATTERNS = [
    ("visual", re.compile(
        r"\bvisual\s+(?:resources?|impacts?|effects?|quality|character|contrast|"
        r"sensitivity|setting|resource\s+management)\b|"
        r"\baesthetics?\b|\bscenic\b|\bscenery\b|\bviewsheds?\b|"
        r"\bview[ -]?sheds?\b|\bVRM\b|\bVQO\b|\blandscape\s+character\b|"
        r"\bglare\b|\bglint\b|\bshadow\s+flicker\b|\blight pollution\b|"
        r"\bnight sky\b|\bdark sky\b",
        re.IGNORECASE,
    )),
    ("environmental_justice", re.compile(r"\benvironmental justice\b|\bEJ\b", re.IGNORECASE)),
    ("cumulative_impacts", re.compile(r"\bcumulative (?:impacts?|effects?)\b", re.IGNORECASE)),
    ("cultural_resources", re.compile(r"\bcultural resources?\b|\bhistoric properties\b|\barchaeolog", re.IGNORECASE)),
    ("biological_resources", re.compile(r"\bbiological resources?\b|\bwildlife\b|\bvegetation\b|\bthreatened\b|\bendangered\b", re.IGNORECASE)),
    ("water_resources", re.compile(r"\bwater resources?\b|\bwetlands?\b|\bfloodplains?\b|\bsurface water\b|\bgroundwater\b", re.IGNORECASE)),
    ("air_quality", re.compile(r"\bair quality\b|\bemissions?\b", re.IGNORECASE)),
    ("greenhouse_gas", re.compile(r"\bgreenhouse gas\b|\bGHG\b|\bclimate change\b", re.IGNORECASE)),
    ("land_use", re.compile(r"\bland use\b|\bpublic lands?\b|\bfarmland\b", re.IGNORECASE)),
    ("recreation", re.compile(r"\brecreation\b|\brecreational\b", re.IGNORECASE)),
    ("noise", re.compile(r"\bnoise\b|\bacoustic\b", re.IGNORECASE)),
    ("traffic_transportation", re.compile(r"\btraffic\b|\btransportation\b|\broads?\b", re.IGNORECASE)),
    ("socioeconomics", re.compile(r"\bsocioeconomics?\b|\beconomics?\b|\benvironmental justice\b", re.IGNORECASE)),
    ("geology_soils", re.compile(r"\bgeology\b|\bsoils?\b|\bgeologic\b", re.IGNORECASE)),
    ("mitigation", re.compile(r"\bmitigation\b|\bmitigate\b|\bbest management practices\b|\bBMPs?\b", re.IGNORECASE)),
    ("alternatives", re.compile(r"\balternatives?\b|\bno action\b|\bproposed action\b", re.IGNORECASE)),
    ("purpose_need", re.compile(r"\bpurpose and need\b|\bneed for action\b", re.IGNORECASE)),
]

VISUAL_TERM_RE = re.compile(
    r"\bvisual\b|\baesthetics?\b|\bscenic\b|\bscenery\b|\bviews?\b|"
    r"\bviewsheds?\b|\bview[ -]?sheds?\b|\blandscape\b|\bglare\b|"
    r"\bglint\b|\bshadow\s+flicker\b|\blight pollution\b|\bnight sky\b|"
    r"\bdark sky\b",
    re.IGNORECASE,
)

IMPACT_TERM_RE = re.compile(
    r"\bimpacts?\b|\beffects?\b|\baffects?\b|\badverse\b|\bsignificant\b|"
    r"\bmitigat(?:e|es|ed|ion)\b|\bminimi[sz]e\b|\breduce\b|\bavoid\b|"
    r"\benhance\b|\bcontrast\b|\bobtrusive\b|\bnoticeable\b",
    re.IGNORECASE,
)


# --------------------------
# DATA CLASSES
# --------------------------

@dataclass
class LineRec:
    page_idx: int
    page_num: int
    line_idx: int
    char_start: int
    char_end: int
    text: str


@dataclass
class HeadingRec:
    page_idx: int
    page_num: int
    line_idx: int
    char_start: int
    char_end: int
    heading_raw: str
    heading_number: str
    heading_title: str
    heading_level: int
    heading_confidence: float
    is_numbered_heading: bool


# --------------------------
# HELPERS
# --------------------------

_START = time.time()


def log(msg: str) -> None:
    elapsed = (time.time() - _START) / 60
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] ({elapsed:.1f}m) {msg}", flush=True)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_title(title: str) -> str:
    title = normalize_space(title).strip(" .:-")
    title = re.sub(r"\s*/\s*", " / ", title)
    return title


def heading_level(number: str, title: str, is_numbered: bool) -> int:
    if is_numbered and number:
        if re.search(r"\d", number):
            return len(re.split(r"[.\-]", number))
        return 1

    title_norm = normalize_title(title).lower()
    if title_norm in {
        "affected environment",
        "environmental consequences",
        "environmental impacts",
        "direct and indirect effects",
        "cumulative impacts",
        "mitigation measures",
        "mitigation",
    }:
        return 3
    if title_norm in KNOWN_UNNUMBERED_HEADINGS:
        return 2
    return 99


def is_title_case_or_caps(text: str) -> bool:
    letters = re.sub(r"[^A-Za-z]+", "", text)
    if len(letters) < 3:
        return False
    if letters.isupper():
        return True
    words = [w for w in re.split(r"\s+", text) if re.search(r"[A-Za-z]", w)]
    if not words:
        return False
    titled = sum(1 for w in words if w[:1].isupper() or w.lower() in {"and", "or", "of", "the", "to", "in", "for"})
    return titled / len(words) >= 0.75


def is_known_or_clear_heading_title(title: str) -> bool:
    title_norm = normalize_title(title).lower()
    if title_norm in KNOWN_UNNUMBERED_HEADINGS:
        return True
    if SENTENCEY_TITLE_RE.search(title):
        return False
    words = title.split()
    if len(words) > 10:
        return False
    return is_title_case_or_caps(title)


def rejection_reason(line: str, page_text: str, repeated_lines: set[str]) -> Optional[str]:
    raw = str(line).strip()
    clean = normalize_space(raw)
    lower = clean.lower()
    if not clean:
        return "blank"
    if clean in repeated_lines:
        return "repeated_header_footer"
    if PAGE_NUM_RE.match(clean):
        return "page_number"
    if TOC_DOT_RE.search(raw):
        return "toc_dot_leader"
    if STRUCTURAL_PAGE_RE.search(page_text):
        return "structural_page"
    if BULLET_START_RE.match(clean):
        return "bullet"
    if LOWERCASE_START_RE.match(clean):
        return "lowercase_fragment"
    if CAPTION_START_RE.match(clean):
        return "caption"
    if len(clean) > 160:
        return "too_long"
    if clean.count("|") >= 2 or clean.count("\t") >= 2:
        return "table_row"
    if re.search(r"\b\d+(?:\.\d+)?\s*x\s*10\d?\s*(?:btu|gpm|mgd|kw|mw)\b", clean, re.IGNORECASE):
        return "measurement_row"
    if re.fullmatch(r"[A-Z]?\d+\s+to", clean, re.IGNORECASE):
        return "range_fragment"
    if MAP_TEXT_RE.search(clean):
        return "map_text"
    if clean.endswith(".") and len(clean.split()) > 8:
        return "sentence"
    if lower.startswith(("source:", "note:", "notes:", "continued", "appendix ")):
        return "caption_or_note"
    if ADDRESS_CITATION_RE.search(clean):
        return "address_or_citation"
    return None


def parse_heading(line: str, page_text: str, repeated_lines: set[str]) -> tuple[Optional[HeadingRec], Optional[str]]:
    reason = rejection_reason(line, page_text, repeated_lines)
    if reason:
        return None, reason

    clean = normalize_space(line)

    for regex in (NUMBERED_HEADING_RE, ROMAN_HEADING_RE):
        m = regex.match(clean)
        if not m:
            continue
        title = normalize_title(m.group("title"))
        if not title or len(title.split()) > 16:
            return None, "title_too_long"
        number = m.group("number").strip(".-")
        sep = m.groupdict().get("sep", "")
        title_norm = normalize_title(title).lower()

        if (
            (LOWERCASE_START_RE.match(title) or WEAK_TITLE_START_RE.match(title))
            and title_norm not in KNOWN_UNNUMBERED_HEADINGS
        ):
            return None, "weak_title_start"

        # Single-level numbered list items and years are the largest source of
        # false headings in OCR text (e.g., "7) Naturalness ..." or
        # "2014) determined ..."). Keep true chapter-style headings only when
        # the title looks like a resource/section title.
        if re.fullmatch(r"\d{4}", number):
            return None, "year_list_item"
        # 5+ digit integers are asset IDs, permit numbers, parcel numbers, etc.
        if re.fullmatch(r"\d{5,}", number) and title_norm not in KNOWN_UNNUMBERED_HEADINGS:
            return None, "large_number_not_section"
        if re.fullmatch(r"\d+", number):
            if sep == ")":
                return None, "numbered_list_item"
            if sep == "." and not CHAPTER_TITLE_RE.search(title) and title_norm not in KNOWN_UNNUMBERED_HEADINGS:
                return None, "numbered_list_item"
            if not is_known_or_clear_heading_title(title):
                return None, "weak_single_number_heading"

        if re.fullmatch(r"[A-Z]?\d+", number) and not CHAPTER_TITLE_RE.search(title):
            # Avoid single-number table/list fragments such as "0 Old" or
            # "9 Tazlina River Bridge" becoming document-level sections.
            if title_norm not in KNOWN_UNNUMBERED_HEADINGS and len(title.split()) <= 4:
                return None, "weak_single_number_heading"

        # Page headers often look like "4-568 Visual Resources"; these are not
        # section starts. Real hyphenated section numbers are usually short.
        if "-" in number:
            parts = number.split("-")
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) >= 100:
                return None, "page_prefixed_heading"

        if SENTENCEY_TITLE_RE.search(title) and title_norm not in KNOWN_UNNUMBERED_HEADINGS:
            return None, "sentencey_title"

        level = heading_level(number, title, True)
        return HeadingRec(
            page_idx=-1,
            page_num=-1,
            line_idx=-1,
            char_start=-1,
            char_end=-1,
            heading_raw=clean,
            heading_number=number,
            heading_title=title,
            heading_level=level,
            heading_confidence=0.95,
            is_numbered_heading=True,
        ), None

    m = SHORT_TITLE_RE.match(clean)
    if not m:
        return None, "not_heading_pattern"

    title = normalize_title(m.group("title"))
    title_norm = title.lower()
    if title_norm not in KNOWN_UNNUMBERED_HEADINGS:
        return None, "not_title_case"
    if SENTENCEY_TITLE_RE.search(title) and title_norm not in KNOWN_UNNUMBERED_HEADINGS:
        return None, "sentencey_title"
    if len(title.split()) > 8 and title_norm not in KNOWN_UNNUMBERED_HEADINGS:
        return None, "unnumbered_too_long"
    if title_norm not in KNOWN_UNNUMBERED_HEADINGS and len(title.split()) < 2:
        return None, "weak_unnumbered_heading"

    level = heading_level("", title, False)
    confidence = 0.75 if title_norm in KNOWN_UNNUMBERED_HEADINGS else 0.55
    return HeadingRec(
        page_idx=-1,
        page_num=-1,
        line_idx=-1,
        char_start=-1,
        char_end=-1,
        heading_raw=clean,
        heading_number="",
        heading_title=title,
        heading_level=level,
        heading_confidence=confidence,
        is_numbered_heading=False,
    ), None


def iter_lines(pages: list[tuple[int, str]]) -> list[LineRec]:
    lines: list[LineRec] = []
    char_pos = 0
    for page_idx, (page_num, page_text) in enumerate(pages):
        raw_lines = str(page_text).splitlines()
        for line_idx, line in enumerate(raw_lines):
            start = char_pos
            end = start + len(line)
            lines.append(LineRec(
                page_idx=page_idx,
                page_num=page_num,
                line_idx=line_idx,
                char_start=start,
                char_end=end,
                text=line,
            ))
            char_pos = end + 1
    return lines


def collect_repeated_lines(pages: list[tuple[int, str]], threshold: int = 20) -> set[str]:
    counts: Counter[str] = Counter()
    for _, page_text in pages:
        for raw in str(page_text).splitlines():
            clean = normalize_space(raw)
            if 3 <= len(clean) <= 90:
                counts[clean] += 1
    return {line for line, n in counts.items() if n > threshold}


def clean_section_text(text: str, repeated_lines: set[str]) -> str:
    text = re.sub(r"-\n(?=[a-z])", "", text)
    kept: list[str] = []
    for raw in text.splitlines():
        clean = normalize_space(raw)
        if not clean:
            continue
        if clean in repeated_lines:
            continue
        if PAGE_NUM_RE.match(clean):
            continue
        if STANDALONE_SECTION_NUM_RE.match(clean):
            continue
        kept.append(clean)
    return "\n".join(kept)


def guess_topic(title: str) -> str:
    title = normalize_title(title)
    for label, pat in TOPIC_PATTERNS:
        if pat.search(title):
            return label
    return "other"


def count_pattern(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text or ""))


def visual_signal_label(visual_terms: int, impact_terms: int, section_words: int) -> str:
    if visual_terms == 0:
        return "none"
    per_1000 = visual_terms / max(section_words, 1) * 1000
    if impact_terms >= 2 and per_1000 >= 2:
        return "visual_impact_language"
    if impact_terms >= 1:
        return "visual_with_some_impact_terms"
    return "visual_resource_description"


def build_document_text(lines: list[LineRec]) -> str:
    return "\n".join(line.text for line in lines)


def find_parent(heading: HeadingRec, prior: list[HeadingRec]) -> tuple[str, str]:
    for prev in reversed(prior):
        if prev.heading_level < heading.heading_level:
            return prev.heading_number, prev.heading_title
    return "", ""


# --------------------------
# PHASE 1: DUCKDB CANDIDATE EXTRACTION
# --------------------------

def _make_candidate_sql(
    pages_path: Path,
    docs_path: Path,
    source: str,
    main_filter: str,
    target_join: str,
) -> str:
    """
    Build the SQL that splits every page into lines, detects repeated lines
    (headers/footers) and structural pages (TOC etc.), then returns only the
    small fraction of lines that could plausibly be headings.

    Uses placeholder substitution to avoid conflicts with regex {n,m} syntax.
    """
    sql = """
    WITH pages_raw AS (
        SELECT
            '__SRC__' AS source,
            r.project_id,
            r.energy_group,
            r.tech_group,
            r.process_type,
            r.lead_agency_harmonized,
            d.document_id,
            d.document_title,
            COALESCE(
                TRY_CAST(regexp_extract(
                    CAST(p.page_number AS VARCHAR), '(\\d+)', 1
                ) AS INTEGER),
                1000000000
            ) AS page_num,
            p.page_text
        FROM read_parquet('__PP__') p
        JOIN read_parquet('__DP__') d USING (document_id)
        JOIN read_parquet('__RP__') r
          ON d.project_id.value = r.project_id
         AND r.process_type = '__SRC__'
        __TJ__
        WHERE length(p.page_text) > 50
          __MF__
    ),
    -- Pages that are purely structural (TOC, preparers, references lists).
    -- All lines from these pages are excluded.
    structural AS (
        SELECT DISTINCT document_id, page_num
        FROM pages_raw
        WHERE regexp_extract(page_text,
            '(?i)table\\s+of\\s+contents'
            '|list\\s+of\\s+(figures|tables|preparers?|authors?|contributors?)'
            '|list\\s+of\\s+references'
            '|prepared\\s+by'
        ) != ''
    ),
    -- Split each page into individual lines and record the 0-based line index
    -- within the page. DuckDB zips parallel unnest() calls.
    split AS (
        SELECT
            source, project_id, energy_group, tech_group, process_type,
            lead_agency_harmonized, document_id, document_title, page_num,
            string_split(page_text, chr(10)) AS lines_arr
        FROM pages_raw
    ),
    all_lines AS (
        SELECT
            source, project_id, energy_group, tech_group, process_type,
            lead_agency_harmonized, document_id, document_title, page_num,
            unnest(lines_arr)                         AS line_text,
            unnest(range(array_length(lines_arr)))    AS line_idx
        FROM split
    ),
    -- Normalise whitespace for all non-blank lines.
    normed AS (
        SELECT *,
            regexp_replace(trim(line_text), '\\s+', ' ', 'g') AS line_norm
        FROM all_lines
        WHERE length(trim(line_text)) >= 3
    ),
    -- Lines that appear > 20 times within a document are running headers/footers.
    repeated AS (
        SELECT document_id, line_norm
        FROM normed
        WHERE length(line_norm) BETWEEN 3 AND 90
        GROUP BY document_id, line_norm
        HAVING count(*) > 20
    )
    SELECT
        n.source, n.project_id, n.energy_group, n.tech_group, n.process_type,
        n.lead_agency_harmonized, n.document_id, n.document_title,
        n.page_num, n.line_idx, n.line_norm AS line_clean
    FROM normed n
    LEFT JOIN structural s
        ON n.document_id = s.document_id AND n.page_num = s.page_num
    LEFT JOIN repeated r
        ON n.document_id = r.document_id AND n.line_norm = r.line_norm
    WHERE s.document_id IS NULL               -- drop structural pages
      AND r.line_norm IS NULL                 -- drop repeated header/footer lines
      AND length(n.line_norm) BETWEEN 3 AND 160
      AND regexp_extract(n.line_norm, '^\\d\\d?\\d?\\d?$') = ''        -- bare page numbers
      AND regexp_extract(n.line_text, '\\.{3,}') = ''                 -- TOC dot leaders
      AND regexp_extract(n.line_norm, '^[a-z]') = ''                  -- lowercase start
      AND regexp_extract(n.line_norm, '^[-*]\\s') = ''                -- bullet items
      AND regexp_extract(n.line_norm,
            '(?i)^(table|figure|fig\\.|map|photo|photograph)\\s+[\\w.\\-]+') = ''
      AND regexp_extract(n.line_norm,
            '(?i)^(source:|note:|notes:|continued|appendix )') = ''
      AND NOT (
            length(n.line_norm) - length(replace(n.line_norm, '|', '')) >= 2
          )                                   -- table rows with pipe separators
    ORDER BY n.document_id, n.page_num, n.line_idx
    """
    return (
        sql
        .replace("__SRC__", source)
        .replace("__PP__", sql_quote(pages_path))
        .replace("__DP__", sql_quote(docs_path))
        .replace("__RP__", sql_quote(D03_REVIEWS))
        .replace("__TJ__", target_join)
        .replace("__MF__", main_filter)
    )


def fetch_all_candidates(
    conn: duckdb.DuckDBPyConnection,
    processes: list[str],
    main_only: bool,
    target_projects: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Phase 1: run the vectorised DuckDB candidate query for each process type
    and return a single DataFrame of surviving lines (~3-5% of all lines).
    """
    frames: list[pd.DataFrame] = []
    for source in processes:
        pages_path = PROCESSED_DIR / source.lower() / "pages.parquet"
        docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"
        main_filter = (
            "AND coalesce(nullif(d.main_document, ''), 'YES') <> 'NO'"
            if main_only else ""
        )
        target_join = (
            "JOIN target_projects tp ON r.project_id = tp.project_id"
            if target_projects is not None else ""
        )
        sql = _make_candidate_sql(pages_path, docs_path, source, main_filter, target_join)
        log(f"  scanning {source} pages for heading candidates...")
        df = conn.execute(sql).fetchdf()
        log(f"  {source}: {len(df):,} candidates from {df['document_id'].nunique():,} documents")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------
# PHASE 2: HEADING CLASSIFICATION
# --------------------------

def classify_headings(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 2: apply parse_heading() to each surviving candidate line.
    The set is small (~2-5 M rows for a full corpus) so a plain Python loop
    is fast here — the expensive per-line iteration already happened in DuckDB.
    Structural-page and repeated-line rejection are skipped (DuckDB did them).
    """
    records: list[dict] = []
    for row in candidates_df.itertuples(index=False):
        h, _ = parse_heading(row.line_clean, "", set())
        if h is None:
            continue
        records.append({
            "document_id": row.document_id,
            "page_num": row.page_num,
            "line_idx": row.line_idx,
            "heading_raw": h.heading_raw,
            "heading_number": h.heading_number,
            "heading_title": h.heading_title,
            "heading_level": h.heading_level,
            "heading_confidence": h.heading_confidence,
            "is_numbered_heading": h.is_numbered_heading,
        })
    return pd.DataFrame(records) if records else pd.DataFrame(columns=[
        "document_id", "page_num", "line_idx",
        "heading_raw", "heading_number", "heading_title",
        "heading_level", "heading_confidence", "is_numbered_heading",
    ])


# --------------------------
# PHASE 3: SECTION EXTRACTION (fast path — headings already known)
# --------------------------

def build_sections_fast(
    doc_meta: dict,
    doc_headings: pd.DataFrame,
    pages: list[tuple[int, str]],
    run_at: str,
) -> list[dict]:
    """
    Phase 3: given pre-detected headings for a single document, reconstruct
    the document text and slice out each section.  Replaces the slow
    build_sections_for_document() which detected headings by iterating every
    line in Python.
    """
    if doc_headings.empty or not pages:
        return []

    repeated = collect_repeated_lines(pages)
    lines = iter_lines(pages)
    if not lines:
        return []

    # Build lookup (page_num, line_idx_within_page) -> LineRec so we can
    # recover Python-computed char positions for each pre-detected heading.
    line_lookup: dict[tuple[int, int], LineRec] = {
        (lr.page_num, lr.line_idx): lr for lr in lines
    }

    headings: list[HeadingRec] = []
    for row in doc_headings.sort_values(["page_num", "line_idx"]).itertuples(index=False):
        lr = line_lookup.get((row.page_num, row.line_idx))
        if lr is None:
            continue
        headings.append(HeadingRec(
            page_idx=lr.page_idx,
            page_num=lr.page_num,
            line_idx=lr.line_idx,
            char_start=lr.char_start,
            char_end=lr.char_end,
            heading_raw=row.heading_raw,
            heading_number=row.heading_number,
            heading_title=row.heading_title,
            heading_level=row.heading_level,
            heading_confidence=row.heading_confidence,
            is_numbered_heading=row.is_numbered_heading,
        ))

    if not headings:
        return []

    document_text = build_document_text(lines)
    hard_cap_pages = 75 if doc_meta["process_type"] == "EIS" else 25
    sections: list[dict] = []
    prior: list[HeadingRec] = []

    for idx, h in enumerate(headings):
        end_heading: Optional[HeadingRec] = None
        end_note = ""
        for nxt in headings[idx + 1:]:
            if nxt.heading_level <= h.heading_level:
                end_heading = nxt
                end_note = "next_same_or_shallower_heading"
                break

        start_char = h.char_end + 1
        if end_heading is not None:
            end_char = max(start_char, end_heading.char_start)
            page_end = end_heading.page_num
            line_end = max(0, end_heading.line_idx - 1)
        else:
            max_page_idx = min(h.page_idx + hard_cap_pages, len(pages) - 1)
            end_line = max(
                (ln for ln in lines if ln.page_idx <= max_page_idx),
                key=lambda x: x.char_end,
            )
            end_char = end_line.char_end
            page_end = end_line.page_num
            line_end = end_line.line_idx
            end_note = "hard_cap_or_document_end"

        raw_section = document_text[start_char:end_char]
        section_text = clean_section_text(raw_section, repeated)
        section_words = len(section_text.split())
        if section_words < 10:
            prior.append(h)
            continue

        parent_num, parent_title = find_parent(h, prior)
        visual_term_count = count_pattern(VISUAL_TERM_RE, section_text)
        impact_term_count = count_pattern(IMPACT_TERM_RE, section_text)
        sections.append({
            "project_id": doc_meta["project_id"],
            "document_id": doc_meta["document_id"],
            "process_type": doc_meta["process_type"],
            "energy_group": doc_meta["energy_group"],
            "tech_group": doc_meta["tech_group"],
            "lead_agency_harmonized": doc_meta["lead_agency_harmonized"],
            "document_title": doc_meta["document_title"],
            "source": doc_meta["source"],
            "page_start": h.page_num,
            "page_end": page_end,
            "line_start": h.line_idx,
            "line_end": line_end,
            "char_start": start_char,
            "char_end": end_char,
            "heading_raw": h.heading_raw,
            "heading_number": h.heading_number,
            "heading_title": h.heading_title,
            "heading_level": h.heading_level,
            "parent_heading_number": parent_num,
            "parent_heading_title": parent_title,
            "section_text": section_text,
            "section_words": section_words,
            "section_chars": len(section_text),
            "section_topic_guess": guess_topic(h.heading_title),
            "visual_term_count": visual_term_count,
            "impact_term_count": impact_term_count,
            "visual_impact_signal": visual_signal_label(
                visual_term_count, impact_term_count, section_words
            ),
            "heading_confidence": h.heading_confidence,
            "is_numbered_heading": h.is_numbered_heading,
            "is_toc_like": False,
            "extraction_notes": end_note,
            "section_run_at": run_at,
            # QA diagnostic flags — use downstream to suppress bad rows without losing them
            "short_section": section_words < 50,
            "very_long_section": section_words > 10_000,
            "large_page_span": (page_end - h.page_num) > 50,
            "suspicious_heading": h.heading_confidence < 0.8,
        })
        prior.append(h)

    return sections


# Legacy single-document entry point kept for reference / unit tests.
def build_sections_for_document(
    doc_meta: dict,
    pages: list[tuple[int, str]],
    run_at: str,
    reject_counts: Counter[str],
) -> list[dict]:
    repeated = collect_repeated_lines(pages)
    lines = iter_lines(pages)
    if not lines:
        return []

    headings: list[HeadingRec] = []
    page_text_by_idx = {i: text for i, (_, text) in enumerate(pages)}
    for line in lines:
        h, reason = parse_heading(line.text, page_text_by_idx[line.page_idx], repeated)
        if h is None:
            if reason not in {"blank", "not_heading_pattern"}:
                reject_counts[reason or "unknown"] += 1
            continue
        h.page_idx = line.page_idx
        h.page_num = line.page_num
        h.line_idx = line.line_idx
        h.char_start = line.char_start
        h.char_end = line.char_end
        headings.append(h)

    if not headings:
        return []

    document_text = build_document_text(lines)
    hard_cap_pages = 75 if doc_meta["process_type"] == "EIS" else 25
    sections: list[dict] = []
    prior: list[HeadingRec] = []

    for idx, h in enumerate(headings):
        end_heading: Optional[HeadingRec] = None
        end_note = ""
        for nxt in headings[idx + 1:]:
            if nxt.heading_level <= h.heading_level:
                end_heading = nxt
                end_note = "next_same_or_shallower_heading"
                break

        start_char = h.char_end + 1
        if end_heading is not None:
            end_char = max(start_char, end_heading.char_start)
            page_end = end_heading.page_num
            line_end = max(0, end_heading.line_idx - 1)
        else:
            max_page_idx = min(h.page_idx + hard_cap_pages, len(pages) - 1)
            end_line = max((ln for ln in lines if ln.page_idx <= max_page_idx), key=lambda x: x.char_end)
            end_char = end_line.char_end
            page_end = end_line.page_num
            line_end = end_line.line_idx
            end_note = "hard_cap_or_document_end"

        raw_section = document_text[start_char:end_char]
        section_text = clean_section_text(raw_section, repeated)
        section_words = len(section_text.split())
        if section_words < 10:
            prior.append(h)
            continue

        parent_num, parent_title = find_parent(h, prior)
        visual_term_count = count_pattern(VISUAL_TERM_RE, section_text)
        impact_term_count = count_pattern(IMPACT_TERM_RE, section_text)
        sections.append({
            "project_id": doc_meta["project_id"],
            "document_id": doc_meta["document_id"],
            "process_type": doc_meta["process_type"],
            "energy_group": doc_meta["energy_group"],
            "tech_group": doc_meta["tech_group"],
            "lead_agency_harmonized": doc_meta["lead_agency_harmonized"],
            "document_title": doc_meta["document_title"],
            "source": doc_meta["source"],
            "page_start": h.page_num,
            "page_end": page_end,
            "line_start": h.line_idx,
            "line_end": line_end,
            "char_start": start_char,
            "char_end": end_char,
            "heading_raw": h.heading_raw,
            "heading_number": h.heading_number,
            "heading_title": h.heading_title,
            "heading_level": h.heading_level,
            "parent_heading_number": parent_num,
            "parent_heading_title": parent_title,
            "section_text": section_text,
            "section_words": section_words,
            "section_chars": len(section_text),
            "section_topic_guess": guess_topic(h.heading_title),
            "visual_term_count": visual_term_count,
            "impact_term_count": impact_term_count,
            "visual_impact_signal": visual_signal_label(visual_term_count, impact_term_count, section_words),
            "heading_confidence": h.heading_confidence,
            "is_numbered_heading": h.is_numbered_heading,
            "is_toc_like": False,
            "extraction_notes": end_note,
            "section_run_at": run_at,
        })
        prior.append(h)

    return sections


def unwrap_project_id(value) -> str:
    if isinstance(value, dict):
        return value.get("value", "")
    return str(value)


def target_project_ids(
    conn: duckdb.DuckDBPyConnection,
    processes: list[str],
    sample: Optional[int],
    project_ids: Optional[list[str]],
) -> Optional[pd.DataFrame]:
    keep: list[str] = []
    if project_ids:
        keep.extend(project_ids)

    if not sample:
        if not keep:
            return None
        return pd.DataFrame({"project_id": sorted(set(keep))})

    if sample:
        processes_sql = ", ".join(f"'{p}'" for p in processes)
        ids = conn.execute(f"""
            SELECT DISTINCT project_id
            FROM read_parquet('{D03_REVIEWS}')
            WHERE process_type IN ({processes_sql})
            ORDER BY project_id
        """).fetchdf()["project_id"].tolist()
        random.seed(42)
        keep.extend(random.sample(ids, min(sample, len(ids))))

    if not keep:
        return None
    return pd.DataFrame({"project_id": sorted(set(keep))})


def page_reader(
    conn: duckdb.DuckDBPyConnection,
    source: str,
    batch_size: int,
    main_only: bool,
    target_projects: Optional[pd.DataFrame],
):
    pages_path = PROCESSED_DIR / source.lower() / "pages.parquet"
    docs_path = PROCESSED_DIR / source.lower() / "documents.parquet"
    main_filter = "AND coalesce(nullif(d.main_document, ''), 'YES') <> 'NO'" if main_only else ""
    target_join = "JOIN target_projects tp ON r.project_id = tp.project_id" if target_projects is not None else ""

    query = f"""
        SELECT
            '{source}' AS source,
            r.project_id,
            r.energy_group,
            r.tech_group,
            r.process_type,
            r.lead_agency_harmonized,
            d.document_id,
            d.document_title,
            d.main_document,
            COALESCE(
                TRY_CAST(regexp_extract(CAST(p.page_number AS VARCHAR), '(\\d+)', 1) AS INTEGER),
                1000000000
            ) AS page_num,
            p.page_text
        FROM read_parquet('{pages_path}') p
        JOIN read_parquet('{docs_path}') d USING (document_id)
        JOIN read_parquet('{D03_REVIEWS}') r
          ON d.project_id.value = r.project_id
         AND r.process_type = '{source}'
        {target_join}
        WHERE length(p.page_text) > 50
          {main_filter}
        ORDER BY r.project_id, d.document_id, page_num
    """
    return conn.execute(query).fetch_record_batch(rows_per_batch=batch_size)


def flush_document_fast(
    current_key: Optional[tuple[str, str]],
    doc_meta: Optional[dict],
    doc_pages: list[tuple[int, str]],
    doc_headings: pd.DataFrame,
    run_at: str,
) -> list[dict]:
    if current_key is None or doc_meta is None or not doc_pages:
        return []
    doc_pages = sorted(doc_pages, key=lambda x: x[0])
    return build_sections_fast(doc_meta, doc_headings, doc_pages, run_at)


def build_qa(
    sections: pd.DataFrame,
    n: int = 120,
    focus_project_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    if sections.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    def add(df: pd.DataFrame, reason: str, limit: int) -> None:
        if df.empty:
            return
        sample_n = min(limit, len(df))
        out = df.sample(sample_n, random_state=42).copy() if len(df) > sample_n else df.copy()
        out["qa_reason"] = reason
        frames.append(out)

    if focus_project_ids:
        add(sections[sections["project_id"].isin(focus_project_ids)], "project_focus", max(n, 200))
    add(sections, "random", max(20, n // 4))
    add(sections.sort_values("section_words", ascending=False).head(30), "longest", 30)
    add(sections.sort_values("section_words", ascending=True).head(20), "shortest", 20)
    add(sections[sections["section_topic_guess"] == "visual"], "visual", 30)
    add(sections[sections["heading_level"] >= 99], "unknown_level", 20)
    add(sections[sections["extraction_notes"] == "hard_cap_or_document_end"], "hard_cap_or_doc_end", 20)

    qa = pd.concat(frames, ignore_index=True) if frames else sections.head(0).copy()
    qa = qa.drop_duplicates(subset=["project_id", "document_id", "heading_raw", "page_start"])

    qa["section_start_excerpt"] = qa["section_text"].fillna("").str.slice(0, 500)
    qa["section_end_excerpt"] = qa["section_text"].fillna("").str.slice(-500)
    cols = [
        "project_id", "document_id", "process_type", "energy_group", "tech_group",
        "heading_raw", "heading_number", "heading_title", "heading_level",
        "parent_heading_title", "page_start", "page_end", "line_start", "line_end",
        "section_words", "section_topic_guess", "visual_term_count",
        "impact_term_count", "visual_impact_signal", "extraction_notes",
        "section_start_excerpt", "section_end_excerpt", "qa_reason",
    ]
    return qa[[c for c in cols if c in qa.columns]].head(n)


def sql_quote(value: str | Path) -> str:
    return str(value).replace("'", "''")


def write_section_fragment(records: list[dict], fragment_dir: Path, fragment_idx: int) -> Path:
    path = fragment_dir / f"sections_{fragment_idx:05d}.parquet"
    pd.DataFrame(records).to_parquet(path, index=False)
    return path


def combine_fragments(conn: duckdb.DuckDBPyConnection, fragments: list[Path], output: Path) -> None:
    if output.exists():
        output.unlink()
    if len(fragments) == 1:
        pd.read_parquet(fragments[0]).to_parquet(output, index=False)
        return
    pattern = sql_quote(fragments[0].parent / "*.parquet")
    conn.execute(f"""
        COPY (
            SELECT *
            FROM read_parquet('{pattern}')
        )
        TO '{sql_quote(output)}' (FORMAT PARQUET)
    """)


def build_qa_from_output(
    conn: duckdb.DuckDBPyConnection,
    output: Path,
    qa_output: Path,
    n: int = 200,
    focus_project_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    source = f"read_parquet('{sql_quote(output)}')"
    frames: list[pd.DataFrame] = []
    cols = """
        project_id, document_id, process_type, energy_group, tech_group,
        document_title, lead_agency_harmonized,
        heading_raw, heading_number, heading_title, heading_level,
        heading_confidence, is_numbered_heading,
        parent_heading_title, page_start, page_end,
        (page_end - page_start) AS page_span,
        line_start, line_end,
        section_words, section_topic_guess, visual_term_count,
        impact_term_count, visual_impact_signal, extraction_notes,
        short_section, very_long_section, large_page_span, suspicious_heading,
        substr(section_text, 1, 500) AS section_start_excerpt,
        substr(section_text, greatest(length(section_text) - 499, 1), 500) AS section_end_excerpt
    """

    def add(query: str, reason: str) -> None:
        df = conn.execute(query).fetchdf()
        if df.empty:
            return
        df["qa_reason"] = reason
        frames.append(df)

    # Diagnostic buckets first so they're guaranteed rows before head(n) cuts in.
    # Buckets are ordered: failures → content-specific → general.
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE very_long_section = true
        ORDER BY section_words DESC
        LIMIT 25
    """, "very_long")
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE large_page_span = true
        ORDER BY page_span DESC
        LIMIT 20
    """, "large_page_span")
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE suspicious_heading = true
          AND very_long_section = false
        ORDER BY random()
        LIMIT 20
    """, "suspicious_heading")
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE section_topic_guess = 'visual'
          AND very_long_section = false
        ORDER BY random()
        LIMIT 40
    """, "visual")
    add(f"""
        SELECT {cols}
        FROM {source}
        ORDER BY random()
        LIMIT {max(20, n // 5)}
    """, "random")
    add(f"""
        SELECT {cols}
        FROM {source}
        ORDER BY section_words ASC
        LIMIT 20
    """, "shortest")
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE heading_level >= 99
        ORDER BY random()
        LIMIT 15
    """, "unknown_level")
    add(f"""
        SELECT {cols}
        FROM {source}
        WHERE extraction_notes = 'hard_cap_or_document_end'
        ORDER BY random()
        LIMIT 15
    """, "hard_cap_or_doc_end")

    if focus_project_ids:
        quoted = ", ".join(f"'{sql_quote(pid)}'" for pid in focus_project_ids)
        add(f"""
            SELECT {cols}
            FROM {source}
            WHERE project_id IN ({quoted})
            ORDER BY project_id, document_id, page_start, line_start
            LIMIT {max(n, 200)}
        """, "project_focus")

    if not frames:
        qa = pd.DataFrame()
    else:
        qa = pd.concat(frames, ignore_index=True)
        qa = qa.drop_duplicates(subset=["project_id", "document_id", "heading_raw", "page_start"])
        qa = qa.head(n)

    qa.to_csv(qa_output, index=False)
    return qa


def build_document_sections(
    processes: list[str],
    main_only: bool,
    sample: Optional[int],
    project_ids: Optional[list[str]],
    output: Path,
    qa_output: Path,
    batch_size: int,
    flush_sections: int,
) -> None:
    """
    3-phase pipeline:
      Phase 1 — DuckDB vectorised scan: split pages into lines, detect
                repeated lines and structural pages, return ~3-5% of lines
                as heading candidates.
      Phase 2 — Python classify: run parse_heading() on the small candidate
                set to produce a headings table indexed by document_id.
      Phase 3 — Stream pages: same streaming loop as before but section
                extraction is fast because headings are already known.
    """
    run_at = datetime.now(timezone.utc).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    qa_output.parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect()
    target_projects = target_project_ids(conn, processes, sample, project_ids)
    if target_projects is not None:
        conn.register("target_projects", target_projects)
        sample_msg = f"sample={sample}" if sample else "sample=None"
        focus_msg = f", project_ids={len(project_ids):,}" if project_ids else ""
        log(f"{sample_msg}{focus_msg} -> {len(target_projects):,} target projects")

    # ------------------------------------------------------------------
    # Phase 1: DuckDB candidate extraction
    # ------------------------------------------------------------------
    log("Phase 1: extracting heading candidates via DuckDB (vectorised)...")
    candidates_df = fetch_all_candidates(conn, processes, main_only, target_projects)
    log(
        f"Phase 1 done: {len(candidates_df):,} candidate lines from "
        f"{candidates_df['document_id'].nunique() if not candidates_df.empty else 0:,} documents"
    )

    # ------------------------------------------------------------------
    # Phase 2: heading classification
    # ------------------------------------------------------------------
    log("Phase 2: classifying candidates with parse_heading()...")
    headings_df = classify_headings(candidates_df)
    log(
        f"Phase 2 done: {len(headings_df):,} headings in "
        f"{headings_df['document_id'].nunique() if not headings_df.empty else 0:,} documents"
    )
    del candidates_df  # free memory before the page-streaming pass

    # Index headings by document_id for O(1) lookup during page streaming.
    headings_by_doc: dict[str, pd.DataFrame] = (
        {doc_id: grp for doc_id, grp in headings_df.groupby("document_id")}
        if not headings_df.empty else {}
    )

    # ------------------------------------------------------------------
    # Phase 3: stream pages, extract sections
    # ------------------------------------------------------------------
    log("Phase 3: streaming pages and extracting section text...")
    section_count = 0

    with TemporaryDirectory(prefix=f"{output.stem}_fragments_", dir=output.parent) as tmp:
        fragment_dir = Path(tmp)
        fragments: list[Path] = []
        pending_sections: list[dict] = []

        def add_sections(records: list[dict]) -> None:
            nonlocal pending_sections, section_count
            if not records:
                return
            pending_sections.extend(records)
            section_count += len(records)
            if len(pending_sections) >= flush_sections:
                fragments.append(write_section_fragment(
                    pending_sections, fragment_dir, len(fragments) + 1
                ))
                log(f"  flushed {section_count:,} sections")
                pending_sections = []

        for source in processes:
            log(f"  reading {source} pages ({'main only' if main_only else 'all documents'})")
            reader = page_reader(conn, source, batch_size, main_only, target_projects)
            current_key: Optional[tuple[str, str]] = None
            current_meta: Optional[dict] = None
            current_pages: list[tuple[int, str]] = []
            n_pages = 0
            n_docs = 0

            for batch in reader:
                pdf = batch.to_pandas()
                n_pages += len(pdf)
                for row in pdf.itertuples(index=False):
                    key = (row.project_id, row.document_id)
                    if current_key is not None and key != current_key:
                        doc_headings = headings_by_doc.get(current_key[1], pd.DataFrame())
                        add_sections(flush_document_fast(
                            current_key, current_meta, current_pages,
                            doc_headings, run_at,
                        ))
                        n_docs += 1
                        current_pages = []

                    if key != current_key:
                        current_key = key
                        current_meta = {
                            "source": row.source,
                            "project_id": row.project_id,
                            "document_id": row.document_id,
                            "process_type": row.process_type,
                            "energy_group": row.energy_group,
                            "tech_group": row.tech_group,
                            "lead_agency_harmonized": row.lead_agency_harmonized,
                            "document_title": row.document_title,
                        }

                    current_pages.append((int(row.page_num), row.page_text))

            # flush the last document for this source
            if current_key is not None:
                doc_headings = headings_by_doc.get(current_key[1], pd.DataFrame())
                add_sections(flush_document_fast(
                    current_key, current_meta, current_pages,
                    doc_headings, run_at,
                ))
                n_docs += 1
            log(f"  {source}: processed {n_pages:,} pages across {n_docs:,} documents")

        if pending_sections:
            fragments.append(write_section_fragment(
                pending_sections, fragment_dir, len(fragments) + 1
            ))
            pending_sections = []

        if not fragments:
            log("WARNING: no sections detected")
            pd.DataFrame().to_parquet(output, index=False)
            pd.DataFrame().to_csv(qa_output, index=False)
            return

        log(f"combining {len(fragments):,} section fragments...")
        combine_fragments(conn, fragments, output)

    qa = build_qa_from_output(conn, output, qa_output, focus_project_ids=project_ids)

    log(f"wrote {section_count:,} sections -> {output}")
    log(f"wrote {len(qa):,} QA rows -> {qa_output}")

    summary = conn.execute(f"""
        SELECT process_type, section_topic_guess, count(*) AS n_sections
        FROM read_parquet('{sql_quote(output)}')
        GROUP BY 1, 2
        QUALIFY row_number() OVER (
            PARTITION BY process_type
            ORDER BY count(*) DESC
        ) <= 8
        ORDER BY process_type, n_sections DESC
    """).fetchdf()
    log("topic summary:\n" + summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a reusable EA/EIS project document section-span layer."
    )
    parser.add_argument(
        "--process",
        nargs="+",
        choices=["EA", "EIS"],
        default=["EA", "EIS"],
        help="Process types/sources to scan.",
    )
    parser.add_argument(
        "--main-only",
        action="store_true",
        default=True,
        help="Scan main documents only (default).",
    )
    parser.add_argument(
        "--include-supporting",
        action="store_true",
        help="Scan supporting documents too.",
    )
    parser.add_argument("--sample", type=int, default=None, help="Random project sample size.")
    parser.add_argument(
        "--project-id",
        nargs="+",
        default=None,
        help="One or more project IDs to include for focused QA/debug runs.",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument(
        "--flush-sections",
        type=int,
        default=50_000,
        help="Number of extracted section rows to hold before flushing a temp parquet fragment.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qa-output", type=Path, default=DEFAULT_QA_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_document_sections(
        processes=args.process,
        main_only=not args.include_supporting,
        sample=args.sample,
        project_ids=args.project_id,
        output=args.output,
        qa_output=args.qa_output,
        batch_size=args.batch_size,
        flush_sections=args.flush_sections,
    )
