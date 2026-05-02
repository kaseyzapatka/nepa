"""
Extract structured co-agency names for already-flagged NEPA co-agency projects.

This is a sidecar to extract_coagency.py. The existing extractor identifies
projects/pages with high-confidence co-agency language; this script parses names
from those hit pages and from CE multi-department metadata.

Inputs:
- phase1/data/analysis/projects_combined.parquet
- phase1/data/analysis/coagency_projects.parquet
- phase1/data/analysis/coagency_hits.parquet
- phase1/data/processed/{ea,eis}/pages.parquet

Outputs:
- phase1/data/analysis/coagency_name_hits.parquet
- phase1/data/analysis/coagency_project_agencies.parquet
- phase1/data/analysis/coagency_department_pairs.parquet
- phase1/data/analysis/coagency_projects_with_names.parquet

Usage:
  conda run -n nepa python phase1/code/extract/extract_coagency_names.py --run
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd


if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")


BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_TABLES_DIR = BASE_DIR / "output" / "deliverable4" / "tables"

DEFAULT_PROJECTS_PATH = ANALYSIS_DIR / "projects_combined.parquet"
DEFAULT_COAGENCY_PROJECTS_PATH = ANALYSIS_DIR / "coagency_projects.parquet"
DEFAULT_COAGENCY_HITS_PATH = ANALYSIS_DIR / "coagency_hits.parquet"
DEFAULT_EA_PAGES_PATH = PROCESSED_DIR / "ea" / "pages.parquet"
DEFAULT_EIS_PAGES_PATH = PROCESSED_DIR / "eis" / "pages.parquet"

DEFAULT_OUTPUT_NAME_HITS = ANALYSIS_DIR / "coagency_name_hits.parquet"
DEFAULT_OUTPUT_PROJECT_AGENCIES = ANALYSIS_DIR / "coagency_project_agencies.parquet"
DEFAULT_OUTPUT_DEPARTMENT_PAIRS = ANALYSIS_DIR / "coagency_department_pairs.parquet"
DEFAULT_OUTPUT_PROJECTS_WITH_NAMES = ANALYSIS_DIR / "coagency_projects_with_names.parquet"

NO_RECOVERY_PROJECT_IDS = {
    # Explicitly no cooperating agencies or only related-project co-agency language.
    "30facdd6985c7dc939acffd2795b6d22",
    "76e3a68920145380360951dee72a9aad",
    "7a5d326c5742295d7809d757902d9075",
}


TEXT_ROLES = {
    "responsible_federal_agency": re.compile(
        r"\b(?:responsible\s+)?federal\s+agency\s*:", re.I
    ),
    "lead": re.compile(r"\blead\s+agenc(?:y|ies)\s*:", re.I),
    "joint_lead": re.compile(r"\bjoint\s+lead\s+agenc(?:y|ies)\s*:", re.I),
    "co_lead": re.compile(r"\bco\s*[- ]\s*lead\s+agenc(?:y|ies)\s*:", re.I),
    "cooperating": re.compile(r"\bcooperating\s+agenc(?:y|ies)\s*:", re.I),
    "participating": re.compile(r"\bparticipating\s+agenc(?:y|ies)\s*:", re.I),
}

ROLE_WORDING = {
    "responsible_federal_agency": re.compile(
        r"\b(?:responsible\s+)?federal\s+agency\b", re.I
    ),
    "lead": re.compile(r"\blead\s+agenc(?:y|ies)\b", re.I),
    "joint_lead": re.compile(r"\bjoint\s+lead\s+agenc(?:y|ies)\b", re.I),
    "co_lead": re.compile(r"\bco\s*[- ]\s*lead\s+agenc(?:y|ies)\b", re.I),
    "cooperating": re.compile(r"\bcooperating\s+agenc(?:y|ies)\b", re.I),
    "participating": re.compile(r"\bparticipating\s+agenc(?:y|ies)\b", re.I),
}

PROSE_LIST_PATTERNS = {
    "cooperating": re.compile(
        r"\b(?P<label>(?:the\s+)?cooperating\s+agenc(?:y|ies)"
        r"(?:\s+that\s+have\s+been\s+engaged\s+in\s+the\s+(?:EIS|EA)\s+process"
        r"(?:\s+for\s+this\s+project)?)?)\s+"
        r"(?:are|include|includes|included)\s+(?P<segment>[^.]{0,900})\.",
        re.I,
    ),
    "cooperating_including": re.compile(
        r"\b(?P<label>(?:the\s+)?cooperating\s+agenc(?:y|ies))\s*,?\s+"
        r"including\s+(?P<segment>[^.]{0,900})\.",
        re.I,
    ),
    "cooperating_agreed": re.compile(
        r"\b(?P<label>(?:these\s+)?federal\s+agencies\s+have\s+agreed\s+to\s+"
        r"participate\s+as\s+cooperating\s+agencies\b[^:]{0,250})\s*:\s*"
        r"(?P<segment>[^.]{0,900})\.",
        re.I,
    ),
    "participating": re.compile(
        r"\b(?P<label>(?:the\s+)?participating\s+agenc(?:y|ies))\s+"
        r"(?:are|include|includes|included)\s+(?P<segment>[^.]{0,900})\.",
        re.I,
    ),
}

REVERSE_COAGENCY_PATTERNS = (
    re.compile(
        r"(?:^|[.;]\s+)(?P<segment>[A-Z][^.]{1,350}?)\s*,?\s+"
        r"as\s+cooperating\s+agenc(?:y|ies)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|[.;]\s+)(?P<segment>[A-Z][^.]{1,350}?)\s+"
        r"(?:participated|signed\s+on)\s+as\s+cooperating\s+agenc(?:y|ies)\b",
        re.I,
    ),
    re.compile(
        r"(?:^|[.;]\s+)(?P<segment>(?:the\s+)?[A-Z][^.;]{1,220}?)\s+"
        r"(?:is|was)\s+a\s+cooperating\s+agency\b",
        re.I,
    ),
    re.compile(
        r"(?:^|[.;]\s+)(?P<segment>(?:the\s+)?[A-Z][^.;]{1,220}?)\s+"
        r"acted\s+as\s+a\s+cooperating\s+agency\b",
        re.I,
    ),
)

FALLBACK_NAME_PAGE_RE = (
    r"(?i)(cooperating\s+agenc(?:y|ies)[^.]{0,500}\b(?:are|include|includes|included)\b|"
    r"cooperating\s+agenc(?:y|ies)\s*,?\s+including\b|"
    r"federal\s+agencies\s+have\s+agreed\s+to\s+participate\s+as\s+cooperating\s+agencies\b|"
    r"\b(?:participated|signed\s+on)\s+as\s+cooperating\s+agenc(?:y|ies)\b|"
    r"\bas\s+cooperating\s+agenc(?:y|ies)\b|"
    r"the\s+cooperating\s+agencies\s+that\s+have\s+been\s+engaged|"
    r"participating\s+agenc(?:y|ies)[^.]{0,500}\b(?:are|include|includes|included)\b)"
)

STOP_RE = re.compile(
    r"\b(?:title|location|abstract|applicant|project\s+location|date|"
    r"for\s+further\s+information|for\s+additional\s+information|"
    r"questions?|contact|prepared\s+by|comments?|subject|purpose|"
    r"mailing\s+and\s+email\s+addresses|date\s+(?:draft|final)\s+eis\s+filed)\s*:",
    re.I,
)

NONE_RE = re.compile(r"^\s*(?:none|n/?a|not\s+applicable|no\s+cooperating\s+agencies)\b", re.I)

NO_AGENCY_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"cooperating\s+agenc(?:y|ies)\s*:\s*none|"
    r"there\s+are\s+no\s+cooperating\s+agenc(?:y|ies)|"
    r"did\s+not\s+identify\s+any\s+(?:federal,\s*state,\s*or\s*local\s+)?agencies\s+as\s+cooperating\s+agenc(?:y|ies)|"
    r"no\s+activities\s+that\s+would\s+make\s+it\s+necessary\s+for\s+another\s+agency\s+to\s+become\s+a\s+cooperating\s+agency"
    r")\b",
    re.I,
)


@dataclass(frozen=True)
class AgencyAlias:
    agency: str
    department: str
    is_federal: bool
    patterns: tuple[str, ...]


AGENCY_ALIASES: tuple[AgencyAlias, ...] = (
    AgencyAlias(
        "Bureau of Land Management",
        "Department of the Interior",
        True,
        (r"\bBureau\s+of\s+Land\s+Management\b", r"\bBLM\b"),
    ),
    AgencyAlias(
        "Bureau of Ocean Energy Management",
        "Department of the Interior",
        True,
        (r"\bBureau\s+of\s+Ocean\s+Energy\s+Management\b", r"\bBOEM\b"),
    ),
    AgencyAlias(
        "Bureau of Reclamation",
        "Department of the Interior",
        True,
        (r"\bBureau\s+of\s+Reclamation\b", r"\bReclamation\b"),
    ),
    AgencyAlias(
        "National Park Service",
        "Department of the Interior",
        True,
        (r"\bNational\s+Park\s+Service\b", r"\bNPS\b"),
    ),
    AgencyAlias(
        "U.S. Fish and Wildlife Service",
        "Department of the Interior",
        True,
        (
            r"\bU\.?S\.?\s+Fish\s+and\s+Wildlife\s+Service\b",
            r"\bFish\s+and\s+Wildlife\s+Service\b",
            r"\bUSFWS\b",
            r"\bFWS\b",
        ),
    ),
    AgencyAlias(
        "Office of Surface Mining Reclamation and Enforcement",
        "Department of the Interior",
        True,
        (
            r"\bOffice\s+of\s+Surface\s+Mining\s+Reclamation\s+and\s+Enforcement\b",
            r"\bOSMRE\b",
        ),
    ),
    AgencyAlias(
        "U.S. Forest Service",
        "Department of Agriculture",
        True,
        (r"\bU\.?S\.?\s+Forest\s+Service\b", r"\bForest\s+Service\b", r"\bUSFS\b"),
    ),
    AgencyAlias(
        "Department of Energy",
        "Department of Energy",
        True,
        (r"\bU\.?S\.?\s+Department\s+of\s+Energy\b", r"\bDepartment\s+of\s+Energy\b", r"\bDOE\b"),
    ),
    AgencyAlias(
        "Western Area Power Administration",
        "Department of Energy",
        True,
        (r"\bWestern\s+Area\s+Power\s+Administration\b", r"\bWAPA\b", r"\bWestern\b"),
    ),
    AgencyAlias(
        "Bonneville Power Administration",
        "Department of Energy",
        True,
        (r"\bBonneville\s+Power\s+Administration\b", r"\bBPA\b"),
    ),
    AgencyAlias(
        "U.S. Army Corps of Engineers",
        "Department of Defense",
        True,
        (
            r"\bU\.?S\.?\s+Army\s+Corps\s+of\s+Engineers\b",
            r"\bArmy\s+Corps\s+of\s+Engineers\b",
            r"\bUSACE\b",
        ),
    ),
    AgencyAlias(
        "Department of Defense",
        "Department of Defense",
        True,
        (r"\bDepartment\s+of\s+Defense\b", r"\bDOD\b"),
    ),
    AgencyAlias("U.S. Navy", "Department of Defense", True, (r"\bU\.?S\.?\s+Navy\b", r"\bNavy\b")),
    AgencyAlias("U.S. Army", "Department of Defense", True, (r"\bU\.?S\.?\s+Army\b",)),
    AgencyAlias("U.S. Air Force", "Department of Defense", True, (r"\bU\.?S\.?\s+Air\s+Force\b",)),
    AgencyAlias(
        "U.S. Environmental Protection Agency",
        "Environmental Protection Agency",
        True,
        (
            r"\bU\.?S\.?\s+Environmental\s+Protection\s+Agency\b",
            r"\bEnvironmental\s+Protection\s+Agency\b",
            r"\bUSEPA\b",
            r"\bEPA\b",
        ),
    ),
    AgencyAlias(
        "U.S. Coast Guard",
        "Department of Homeland Security",
        True,
        (r"\bU\.?S\.?\s+Coast\s+Guard\b", r"\bUSCG\b", r"\bCoast\s+Guard\b"),
    ),
    AgencyAlias(
        "Federal Energy Regulatory Commission",
        "Federal Energy Regulatory Commission",
        True,
        (r"\bFederal\s+Energy\s+Regulatory\s+Commission\b", r"\bFERC\b"),
    ),
    AgencyAlias(
        "Federal Aviation Administration",
        "Department of Transportation",
        True,
        (r"\bFederal\s+Aviation\s+Administration\b", r"\bFAA\b"),
    ),
    AgencyAlias(
        "Federal Highway Administration",
        "Department of Transportation",
        True,
        (r"\bFederal\s+Highway\s+Administration\b", r"\bFHWA\b"),
    ),
    AgencyAlias(
        "Federal Railroad Administration",
        "Department of Transportation",
        True,
        (r"\bFederal\s+Railroad\s+Administration\b", r"\bFRA\b"),
    ),
    AgencyAlias(
        "Federal Transit Administration",
        "Department of Transportation",
        True,
        (r"\bFederal\s+Transit\s+Administration\b", r"\bFTA\b"),
    ),
    AgencyAlias(
        "Federal Communications Commission",
        "Federal Communications Commission",
        True,
        (r"\bFederal\s+Communications\s+Commission\b", r"\bFCC\b"),
    ),
    AgencyAlias(
        "National Marine Fisheries Service",
        "Department of Commerce",
        True,
        (
            r"\bNational\s+Marine\s+Fisheries\s+Service\b",
            r"\bNMFS\b",
            r"\bNOAA\s+Fisheries\b",
        ),
    ),
    AgencyAlias(
        "National Oceanic and Atmospheric Administration",
        "Department of Commerce",
        True,
        (r"\bNational\s+Oceanic\s+and\s+Atmospheric\s+Administration\b", r"\bNOAA\b"),
    ),
    AgencyAlias(
        "Advisory Council on Historic Preservation",
        "Advisory Council on Historic Preservation",
        True,
        (r"\bAdvisory\s+Council\s+on\s+Historic\s+Preservation\b", r"\bACHP\b"),
    ),
    AgencyAlias(
        "Council on Environmental Quality",
        "Council on Environmental Quality",
        True,
        (r"\bCouncil\s+on\s+Environmental\s+Quality\b", r"\bCEQ\b"),
    ),
    AgencyAlias(
        "General Services Administration",
        "General Services Administration",
        True,
        (r"\bGeneral\s+Services\s+Administration\b", r"\bGSA\b"),
    ),
    AgencyAlias(
        "Tennessee Valley Authority",
        "Other Independent Agencies",
        True,
        (r"\bTennessee\s+Valley\s+Authority\b", r"\bTVA\b"),
    ),
    AgencyAlias(
        "Nuclear Regulatory Commission",
        "Nuclear Regulatory Commission",
        True,
        (r"\bNuclear\s+Regulatory\s+Commission\b", r"\bNRC\b"),
    ),
    AgencyAlias(
        "Department of State",
        "Department of State",
        True,
        (r"\bU\.?S\.?\s+Department\s+of\s+State\b", r"\bDepartment\s+of\s+State\b"),
    ),
)


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _normalize_text(text: object) -> str:
    if not isinstance(text, str):
        return ""
    normalized = re.sub(r"\s+", " ", text).strip()
    # Keep sentence-bounded prose regexes from stopping at common agency abbreviations.
    normalized = re.sub(r"\bU\.\s*S\.", "US", normalized)
    return normalized


def _json_list(values: Iterable[object]) -> str:
    out = sorted({str(v) for v in values if pd.notna(v) and str(v).strip()})
    return json.dumps(out)


def _parse_jsonish_vector(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    if "|" in text:
        return [p.strip() for p in text.split("|") if p.strip()]
    return [text]


def _department_from_prefix(raw: str) -> str | None:
    for prefix in (
        "Department of Energy",
        "Department of the Interior",
        "Department of Agriculture",
        "Department of Defense",
        "Department of Homeland Security",
        "Department of Transportation",
        "Department of Commerce",
        "Department of State",
        "Department of Veterans Affairs",
        "General Services Administration",
        "Major Independent Agencies",
        "Other Independent Agencies",
    ):
        if raw.startswith(prefix):
            return prefix
    return None


def _match_agencies(segment: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    seen: set[str] = set()
    for alias in AGENCY_ALIASES:
        best = None
        for pattern in alias.patterns:
            found = re.search(pattern, segment, re.I)
            if found and (best is None or found.start() < best.start()):
                best = found
        if best and alias.agency not in seen:
            seen.add(alias.agency)
            matches.append(
                {
                    "agency_raw": best.group(0).strip(),
                    "agency_normalized": alias.agency,
                    "department": alias.department,
                    "is_federal": alias.is_federal,
                    "match_start": best.start(),
                    "evidence_text": segment[max(0, best.start() - 80) : best.end() + 120].strip(),
                }
            )
    matches.sort(key=lambda row: int(row["match_start"]))
    for row in matches:
        row.pop("match_start", None)
    return matches


def _looks_like_unmatched_agency(token: str) -> bool:
    token = token.strip()
    if not token or len(token) < 2:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9&.-]{1,12}", token):
        return True
    return bool(
        re.search(
            r"\b(?:agency|department|office|division|bureau|commission|council|"
            r"authority|association|county|city|tribe|state|district|service|program)\b",
            token,
            re.I,
        )
    )


def _unmatched_agency_tokens(segment: str) -> list[str]:
    tokens = re.split(r"\s*;\s*|\s*,\s*|\s+\band\b\s+|\s+\balong\s+with\b\s+", segment)
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        clean = re.sub(r"\s+", " ", token).strip(" .;:")
        clean = re.sub(r"^(?:the|and)\s+", "", clean, flags=re.I).strip(" .;:")
        clean = re.sub(r"\s*\([^)]{1,80}\)\s*$", "", clean).strip(" .;:")
        clean = re.sub(r"\b(?:respectively|among others)$", "", clean, flags=re.I).strip(" .;:")
        clean = re.sub(r"\s+(?:participated|signed\s+on)$", "", clean, flags=re.I).strip(" .;:")
        if not clean or clean in seen:
            continue
        if _match_agencies(clean):
            continue
        if _looks_like_unmatched_agency(clean):
            seen.add(clean)
            out.append(clean)
    return out


def _county_reference(text: str, upto: int | None = None) -> str | None:
    search_text = text[:upto] if upto is not None else text
    pattern = re.compile(
        r"\b(?P<county>(?:County\s+of\s+[A-Z][A-Za-z .'-]+|[A-Z][A-Za-z .'-]+\s+County))"
        r"(?:,\s*[A-Z][A-Za-z .'-]+(?:\s+\d+)?)?\s*\(\s*(?:the\s+)?County\s*\)",
        re.I,
    )
    matches = list(pattern.finditer(search_text))
    if not matches:
        return None
    return re.sub(r"\s+", " ", matches[-1].group("county")).strip(" ,.;:")


def _resolve_local_references(segment: str, text: str, start: int | None = None) -> str:
    county = _county_reference(text, start)
    if not county:
        return segment
    return re.sub(r"\b(?:the\s+)?County(?:['’]s)?\b", county, segment, flags=re.I)


def _find_role_segments(text: str) -> list[tuple[str, str, str]]:
    markers: list[tuple[int, int, str, str]] = []
    for role, pattern in TEXT_ROLES.items():
        for match in pattern.finditer(text):
            markers.append((match.start(), match.end(), "role", role))
    for match in STOP_RE.finditer(text):
        markers.append((match.start(), match.end(), "stop", "stop"))
    markers.sort(key=lambda item: (item[0], item[1]))

    segments: list[tuple[str, str, str]] = []
    for i, (start, end, kind, role) in enumerate(markers):
        if kind != "role":
            continue
        next_start = len(text)
        for later_start, _later_end, _later_kind, _later_role in markers[i + 1 :]:
            if later_start > end:
                next_start = later_start
                break
        segment = text[end:next_start].strip(" :;-")
        if segment:
            label = text[start:end].strip()
            segments.append((role, label, segment[:1200]))
    return segments


def _prose_list_segments(text: str) -> list[tuple[str, str, str]]:
    segments: list[tuple[str, str, str]] = []
    for role, pattern in PROSE_LIST_PATTERNS.items():
        for match in pattern.finditer(text):
            segment = _resolve_local_references(
                match.group("segment").strip(" :;-"),
                text,
                match.start("segment"),
            )
            label = match.group("label").strip()
            if segment:
                parsed_role = "cooperating" if role.startswith("cooperating") else role
                segments.append((parsed_role, label, segment[:1200]))
    for pattern in REVERSE_COAGENCY_PATTERNS:
        for match in pattern.finditer(text):
            if NO_AGENCY_CONTEXT_RE.search(match.group(0)):
                continue
            segment = _resolve_local_references(
                match.group("segment").strip(" :;-"),
                text,
                match.start("segment"),
            )
            if segment:
                segments.append(("cooperating", "cooperating agency prose", segment[:1200]))
    return segments


def _fallback_role_segments(text: str) -> list[tuple[str, str, str]]:
    """Handle prose cues without colons, such as 'agreed to act as joint lead agency with BLM'."""
    segments: list[tuple[str, str, str]] = []
    for role in ("joint_lead", "co_lead", "cooperating"):
        for match in ROLE_WORDING[role].finditer(text):
            context = text[max(0, match.start() - 180) : match.end() + 360].strip()
            if context:
                segments.append((role, match.group(0), context))
    return segments


def _rows_from_segment(
    base: dict[str, object],
    role: str,
    role_label: str,
    segment: str,
    extraction_method: str,
    preserve_unmatched: bool = False,
) -> list[dict[str, object]]:
    segment_norm = _normalize_text(segment)
    if not segment_norm or NONE_RE.match(segment_norm):
        return [
            {
                **base,
                "role": role,
                "role_label": role_label,
                "agency_raw": "None",
                "agency_normalized": pd.NA,
                "department": pd.NA,
                "is_federal": False,
                "extraction_method": "explicit_none",
                "evidence_text": segment_norm[:500],
            }
        ]

    matches = _match_agencies(segment_norm)
    if matches:
        rows = [
            {
                **base,
                "role": role,
                "role_label": role_label,
                "agency_raw": match["agency_raw"],
                "agency_normalized": match["agency_normalized"],
                "department": match["department"],
                "is_federal": match["is_federal"],
                "extraction_method": extraction_method,
                "evidence_text": match["evidence_text"],
            }
            for match in matches
        ]
        if preserve_unmatched:
            for token in _unmatched_agency_tokens(segment_norm):
                rows.append(
                    {
                        **base,
                        "role": role,
                        "role_label": role_label,
                        "agency_raw": token,
                        "agency_normalized": token,
                        "department": pd.NA,
                        "is_federal": False,
                        "extraction_method": extraction_method,
                        "evidence_text": segment_norm[:500],
                    }
                )
        return rows

    if preserve_unmatched:
        tokens = _unmatched_agency_tokens(segment_norm)
        if tokens:
            return [
                {
                    **base,
                    "role": role,
                    "role_label": role_label,
                    "agency_raw": token,
                    "agency_normalized": token,
                    "department": pd.NA,
                    "is_federal": False,
                    "extraction_method": extraction_method,
                    "evidence_text": segment_norm[:500],
                }
                for token in tokens
            ]

    return [
        {
            **base,
            "role": role,
            "role_label": role_label,
            "agency_raw": segment_norm[:300],
            "agency_normalized": pd.NA,
            "department": pd.NA,
            "is_federal": False,
            "extraction_method": "unmatched_segment",
            "evidence_text": segment_norm[:500],
        }
    ]


def _parse_text_hit_rows(
    text_pages: pd.DataFrame,
    source: str = "document_text",
    extraction_method: str = "label_segment",
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in text_pages.to_dict("records"):
        page_text = _normalize_text(record.get("page_text"))
        base = {
            "project_id": record.get("project_id"),
            "dataset_source": record.get("dataset_source"),
            "process_type": record.get("process_type"),
            "project_title": record.get("project_title"),
            "project_department": record.get("project_department"),
            "document_id": record.get("document_id"),
            "page_number": record.get("page_number"),
            "cue_name": record.get("cue_name"),
            "source": source,
        }
        segment_specs = [
            (role, role_label, segment, extraction_method, False)
            for role, role_label, segment in _find_role_segments(page_text)
        ]
        segment_specs.extend(
            (
                role,
                role_label,
                segment,
                f"{extraction_method}_prose_list",
                True,
            )
            for role, role_label, segment in _prose_list_segments(page_text)
        )
        if not segment_specs:
            segment_specs = [
                (role, role_label, segment, extraction_method, False)
                for role, role_label, segment in _fallback_role_segments(page_text)
            ]
        for role, role_label, segment, method, preserve_unmatched in segment_specs:
            rows.extend(
                _rows_from_segment(
                    base,
                    role,
                    role_label,
                    segment,
                    method,
                    preserve_unmatched=preserve_unmatched,
                )
            )

    cols = _name_hit_columns()
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).drop_duplicates().loc[:, cols]


def _parse_metadata_rows(ce_projects: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for record in ce_projects.to_dict("records"):
        agencies = _parse_jsonish_vector(record.get("lead_agency"))
        for agency in agencies:
            segment = agency
            matches = _match_agencies(segment)
            if matches:
                for match in matches:
                    rows.append(
                        {
                            "project_id": record.get("project_id"),
                            "dataset_source": record.get("dataset_source"),
                            "process_type": record.get("process_type"),
                            "project_title": record.get("project_title"),
                            "project_department": record.get("project_department"),
                            "document_id": pd.NA,
                            "page_number": pd.NA,
                            "cue_name": "lead_agency_metadata",
                            "source": "lead_agency_metadata",
                            "role": "lead_metadata",
                            "role_label": "lead_agency",
                            "agency_raw": agency,
                            "agency_normalized": match["agency_normalized"],
                            "department": match["department"],
                            "is_federal": match["is_federal"],
                            "extraction_method": "metadata_alias",
                            "evidence_text": agency,
                        }
                    )
            else:
                department = _department_from_prefix(agency) or record.get("project_department")
                rows.append(
                    {
                        "project_id": record.get("project_id"),
                        "dataset_source": record.get("dataset_source"),
                        "process_type": record.get("process_type"),
                        "project_title": record.get("project_title"),
                        "project_department": record.get("project_department"),
                        "document_id": pd.NA,
                        "page_number": pd.NA,
                        "cue_name": "lead_agency_metadata",
                        "source": "lead_agency_metadata",
                        "role": "lead_metadata",
                        "role_label": "lead_agency",
                        "agency_raw": agency,
                        "agency_normalized": agency,
                        "department": department,
                        "is_federal": bool(isinstance(department, str) and department),
                        "extraction_method": "metadata_prefix",
                        "evidence_text": agency,
                    }
                )
    cols = _name_hit_columns()
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).drop_duplicates().loc[:, cols]


def _name_hit_columns() -> list[str]:
    return [
        "project_id",
        "dataset_source",
        "process_type",
        "project_title",
        "project_department",
        "document_id",
        "page_number",
        "cue_name",
        "source",
        "role",
        "role_label",
        "agency_raw",
        "agency_normalized",
        "department",
        "is_federal",
        "extraction_method",
        "evidence_text",
    ]


def _read_targets(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    try:
        target_query = f"""
        WITH base AS (
            SELECT
                p.project_id,
                p.dataset_source,
                p.process_type,
                p.project_energy_type,
                p.project_title,
                p.project_department,
                p.lead_agency,
                p.project_multi_department,
                co.project_multi_agency,
                co.project_has_coagency_signal_high_conf,
                co.project_coagency_signal_source
            FROM read_parquet('{_sql_path(args.projects)}') p
            INNER JOIN read_parquet('{_sql_path(args.coagency_projects)}') co
                ON p.project_id = co.project_id
               AND p.dataset_source = co.dataset_source
            WHERE p.project_energy_type = 'Clean'
              AND co.project_multi_agency
        ),
        hit_pages AS (
            SELECT DISTINCT
                b.project_id,
                b.dataset_source,
                b.process_type,
                b.project_title,
                b.project_department,
                h.document_id,
                h.page_number,
                h.cue_name
            FROM base b
            INNER JOIN read_parquet('{_sql_path(args.coagency_hits)}') h
                ON b.project_id = h.project_id
               AND b.dataset_source = h.dataset_source
            WHERE b.process_type IN ('EA', 'EIS')
              AND h.is_high_conf_match
        ),
        pages AS (
            SELECT
                'EA' AS dataset_source,
                document_id,
                try_cast(page_number AS INTEGER) AS page_number,
                page_text
            FROM read_parquet('{_sql_path(args.ea_pages)}')
            UNION ALL
            SELECT
                'EIS' AS dataset_source,
                document_id,
                try_cast(page_number AS INTEGER) AS page_number,
                page_text
            FROM read_parquet('{_sql_path(args.eis_pages)}')
        )
        SELECT
            hp.*,
            p.page_text
        FROM hit_pages hp
        LEFT JOIN pages p
            ON hp.dataset_source = p.dataset_source
           AND hp.document_id = p.document_id
           AND hp.page_number = p.page_number
        ORDER BY hp.process_type, hp.project_id, hp.document_id, hp.page_number, hp.cue_name
        """
        ce_query = f"""
        SELECT
            p.project_id,
            p.dataset_source,
            p.process_type,
            p.project_title,
            p.project_department,
            p.lead_agency,
            p.project_multi_department,
            co.project_multi_agency,
            co.project_has_coagency_signal_high_conf,
            co.project_coagency_signal_source
        FROM read_parquet('{_sql_path(args.projects)}') p
        INNER JOIN read_parquet('{_sql_path(args.coagency_projects)}') co
            ON p.project_id = co.project_id
           AND p.dataset_source = co.dataset_source
        WHERE p.project_energy_type = 'Clean'
          AND co.project_multi_agency
          AND p.process_type = 'CE'
          AND p.project_multi_department
        ORDER BY p.project_id
        """
        text_pages = con.execute(target_query).df()
        ce_projects = con.execute(ce_query).df()
    finally:
        con.close()
    return text_pages, ce_projects


def _projects_without_parsed_agencies(
    target_projects: pd.DataFrame,
    name_hits: pd.DataFrame,
) -> pd.DataFrame:
    parsed = name_hits[
        name_hits["agency_normalized"].notna()
        & (name_hits["agency_normalized"].astype(str).str.len() > 0)
    ][["project_id", "dataset_source"]].drop_duplicates()

    return (
        target_projects[target_projects["process_type"].isin(["EA", "EIS"])]
        .merge(parsed.assign(has_parsed_agency=True), on=["project_id", "dataset_source"], how="left")
        .loc[lambda df: df["has_parsed_agency"].isna()]
        .drop(columns=["has_parsed_agency"])
        .drop_duplicates()
    )


def _read_fallback_pages(args: argparse.Namespace, fallback_targets: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "project_id",
        "dataset_source",
        "process_type",
        "project_title",
        "project_department",
        "document_id",
        "page_number",
        "cue_name",
        "page_text",
    ]
    if fallback_targets.empty:
        return pd.DataFrame(columns=cols)
    fallback_targets = fallback_targets[
        ~fallback_targets["project_id"].isin(NO_RECOVERY_PROJECT_IDS)
    ].copy()
    if fallback_targets.empty:
        return pd.DataFrame(columns=cols)

    register_cols = ["project_id", "dataset_source", "process_type", "project_title", "project_department"]
    con = duckdb.connect()
    try:
        con.register("fallback_targets", fallback_targets[register_cols].drop_duplicates())
        fallback_query = f"""
        WITH target_docs AS (
            SELECT DISTINCT
                ft.project_id,
                ft.dataset_source,
                ft.process_type,
                ft.project_title,
                ft.project_department,
                h.document_id
            FROM fallback_targets ft
            INNER JOIN read_parquet('{_sql_path(args.coagency_hits)}') h
                ON ft.project_id = h.project_id
               AND ft.dataset_source = h.dataset_source
            WHERE h.is_high_conf_match
        ),
        pages AS (
            SELECT
                'EA' AS dataset_source,
                document_id,
                try_cast(page_number AS INTEGER) AS page_number,
                page_text
            FROM read_parquet('{_sql_path(args.ea_pages)}')
            UNION ALL
            SELECT
                'EIS' AS dataset_source,
                document_id,
                try_cast(page_number AS INTEGER) AS page_number,
                page_text
            FROM read_parquet('{_sql_path(args.eis_pages)}')
        ),
        candidates AS (
            SELECT
                td.project_id,
                td.dataset_source,
                td.process_type,
                td.project_title,
                td.project_department,
                td.document_id,
                p.page_number,
                'fallback_cooperating_agency_name_scan' AS cue_name,
                p.page_text,
                row_number() OVER (
                    PARTITION BY td.project_id, td.dataset_source, td.document_id, p.page_number
                    ORDER BY p.page_number
                ) AS page_rank
            FROM target_docs td
            INNER JOIN pages p
                ON td.dataset_source = p.dataset_source
               AND td.document_id = p.document_id
            WHERE regexp_matches(
                regexp_replace(coalesce(p.page_text, ''), '\\s+', ' ', 'g'),
                '{FALLBACK_NAME_PAGE_RE.replace("'", "''")}'
            )
        )
        SELECT {", ".join(cols)}
        FROM candidates
        WHERE page_rank = 1
        ORDER BY process_type, project_id, document_id, page_number
        """
        return con.execute(fallback_query).df()
    finally:
        con.close()


def _build_project_rollup(name_hits: pd.DataFrame, target_projects: pd.DataFrame) -> pd.DataFrame:
    valid = name_hits[
        name_hits["agency_normalized"].notna()
        & (name_hits["agency_normalized"].astype(str).str.len() > 0)
    ].copy()

    if valid.empty:
        rollup = target_projects[["project_id", "dataset_source", "process_type"]].copy()
        for col in (
            "coagency_agencies",
            "coagency_departments",
            "coagency_lead_agencies",
            "coagency_partner_agencies",
            "coagency_partner_departments",
        ):
            rollup[col] = "[]"
        rollup["coagency_name_extraction_count"] = 0
        return rollup

    lead_roles = {"lead", "responsible_federal_agency", "joint_lead", "co_lead", "lead_metadata"}
    partner_roles = {"cooperating", "participating"}
    group_cols = ["project_id", "dataset_source", "process_type"]
    rows = []
    for keys, group in valid.groupby(group_cols, dropna=False):
        role = group["role"].fillna("")
        partners = group[role.isin(partner_roles)]
        leads = group[role.isin(lead_roles)]
        rows.append(
            {
                "project_id": keys[0],
                "dataset_source": keys[1],
                "process_type": keys[2],
                "coagency_agencies": _json_list(group["agency_normalized"]),
                "coagency_departments": _json_list(group["department"]),
                "coagency_lead_agencies": _json_list(leads["agency_normalized"]),
                "coagency_partner_agencies": _json_list(partners["agency_normalized"]),
                "coagency_partner_departments": _json_list(partners["department"]),
                "coagency_name_extraction_count": int(group["agency_normalized"].nunique()),
            }
        )
    rollup = pd.DataFrame(rows)
    return target_projects[["project_id", "dataset_source", "process_type"]].merge(
        rollup, on=group_cols, how="left"
    ).fillna(
        {
            "coagency_agencies": "[]",
            "coagency_departments": "[]",
            "coagency_lead_agencies": "[]",
            "coagency_partner_agencies": "[]",
            "coagency_partner_departments": "[]",
            "coagency_name_extraction_count": 0,
        }
    )


def _dedupe_pair_rows(rows: list[dict[str, object]]) -> pd.DataFrame:
    cols = [
        "project_id",
        "dataset_source",
        "process_type",
        "project_title",
        "source_process",
        "relationship_type",
        "department_1",
        "department_2",
        "department_pair_key",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows)
    df["department_1_norm"] = df[["department_1", "department_2"]].min(axis=1)
    df["department_2_norm"] = df[["department_1", "department_2"]].max(axis=1)
    df["department_1"] = df["department_1_norm"]
    df["department_2"] = df["department_2_norm"]
    df["department_pair_key"] = df["department_1"] + " | " + df["department_2"]

    relationship_priority = {
        "metadata_department_pair": 0,
        "joint_or_co_lead_pair": 1,
        "lead_to_partner": 2,
    }
    df["relationship_priority"] = df["relationship_type"].map(relationship_priority).fillna(99)
    df = df.sort_values(
        [
            "project_id",
            "dataset_source",
            "process_type",
            "department_1",
            "department_2",
            "relationship_priority",
            "relationship_type",
        ],
        kind="stable",
    )
    return (
        df.drop_duplicates(
            subset=[
                "project_id",
                "dataset_source",
                "process_type",
                "department_1",
                "department_2",
            ],
            keep="first",
        )
        .loc[:, cols]
        .reset_index(drop=True)
    )


def _build_department_pairs(name_hits: pd.DataFrame, target_projects: pd.DataFrame) -> pd.DataFrame:
    valid = name_hits[
        name_hits["department"].notna()
        & (name_hits["department"].astype(str).str.len() > 0)
        & name_hits["is_federal"].fillna(False).astype(bool)
    ].copy()
    rows: list[dict[str, object]] = []
    lead_roles = {"lead", "responsible_federal_agency", "joint_lead", "co_lead", "lead_metadata"}
    partner_roles = {"cooperating", "participating"}

    for project in target_projects.to_dict("records"):
        project_hits = valid[
            (valid["project_id"] == project["project_id"])
            & (valid["dataset_source"] == project["dataset_source"])
        ]
        if project_hits.empty:
            continue
        process_type = project["process_type"]
        title = project.get("project_title")
        source_process = "metadata_ce" if process_type == "CE" else "text_signal"

        if process_type == "CE":
            depts = sorted(set(project_hits["department"].dropna().astype(str)))
            for d1, d2 in combinations(depts, 2):
                rows.append(
                    {
                        "project_id": project["project_id"],
                        "dataset_source": project["dataset_source"],
                        "process_type": process_type,
                        "project_title": title,
                        "source_process": source_process,
                        "relationship_type": "metadata_department_pair",
                        "department_1": d1,
                        "department_2": d2,
                        "department_pair_key": " | ".join(sorted((d1, d2))),
                    }
                )
            continue

        leads = sorted(
            set(project_hits.loc[project_hits["role"].isin(lead_roles), "department"].dropna().astype(str))
        )
        partners = sorted(
            set(project_hits.loc[project_hits["role"].isin(partner_roles), "department"].dropna().astype(str))
        )
        if not leads and isinstance(project.get("project_department"), str):
            leads = [project["project_department"]]

        for d1, d2 in combinations(leads, 2):
            rows.append(
                {
                    "project_id": project["project_id"],
                    "dataset_source": project["dataset_source"],
                    "process_type": process_type,
                    "project_title": title,
                    "source_process": source_process,
                    "relationship_type": "joint_or_co_lead_pair",
                    "department_1": d1,
                    "department_2": d2,
                    "department_pair_key": " | ".join(sorted((d1, d2))),
                }
            )
        for lead in leads:
            for partner in partners:
                if lead == partner:
                    continue
                rows.append(
                    {
                        "project_id": project["project_id"],
                        "dataset_source": project["dataset_source"],
                        "process_type": process_type,
                        "project_title": title,
                        "source_process": source_process,
                        "relationship_type": "lead_to_partner",
                        "department_1": lead,
                        "department_2": partner,
                        "department_pair_key": " | ".join(sorted((lead, partner))),
                    }
                )

    return _dedupe_pair_rows(rows)


def _target_projects_from_inputs(text_pages: pd.DataFrame, ce_projects: pd.DataFrame) -> pd.DataFrame:
    text_cols = ["project_id", "dataset_source", "process_type", "project_title", "project_department"]
    text_targets = text_pages[text_cols].drop_duplicates()
    ce_targets = ce_projects[text_cols].drop_duplicates()
    return pd.concat([text_targets, ce_targets], ignore_index=True).drop_duplicates()


def _write_qa_tables(
    args: argparse.Namespace,
    target_projects: pd.DataFrame,
    name_hits: pd.DataFrame,
    department_pairs: pd.DataFrame,
) -> None:
    if not args.qa_dir:
        return
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    target_projects.groupby("process_type", dropna=False).size().reset_index(name="target_projects").to_csv(
        args.qa_dir / "coagency_name_target_composition.csv", index=False
    )
    coverage = (
        target_projects.merge(
            name_hits[
                name_hits["agency_normalized"].notna()
                & (name_hits["agency_normalized"].astype(str).str.len() > 0)
            ][["project_id", "dataset_source"]].drop_duplicates().assign(has_parsed_agency=True),
            on=["project_id", "dataset_source"],
            how="left",
        )
        .assign(has_parsed_agency=lambda df: df["has_parsed_agency"].fillna(False))
        .groupby(["process_type", "has_parsed_agency"], dropna=False)
        .size()
        .reset_index(name="projects")
    )
    coverage.to_csv(args.qa_dir / "coagency_name_coverage_by_process.csv", index=False)

    failures = target_projects.merge(
        name_hits[
            name_hits["agency_normalized"].notna()
            & (name_hits["agency_normalized"].astype(str).str.len() > 0)
        ][["project_id", "dataset_source"]].drop_duplicates().assign(has_parsed_agency=True),
        on=["project_id", "dataset_source"],
        how="left",
    )
    failures = failures[failures["has_parsed_agency"].isna()].drop(columns=["has_parsed_agency"])
    failures.to_csv(args.qa_dir / "coagency_name_projects_without_parsed_agency.csv", index=False)

    department_pairs.groupby(
        ["process_type", "department_pair_key"], dropna=False
    ).size().reset_index(name="projects").sort_values(
        ["process_type", "projects"], ascending=[True, False]
    ).to_csv(args.qa_dir / "coagency_name_top_department_pairs.csv", index=False)

    successes = name_hits[
        name_hits["agency_normalized"].notna()
        & (name_hits["agency_normalized"].astype(str).str.len() > 0)
    ].head(50)
    successes.to_csv(args.qa_dir / "coagency_name_success_examples.csv", index=False)
    name_hits[name_hits["extraction_method"].isin(["unmatched_segment", "explicit_none"])].head(50).to_csv(
        args.qa_dir / "coagency_name_failure_examples.csv", index=False
    )


def run(args: argparse.Namespace) -> None:
    _require_file(args.projects, "projects parquet")
    _require_file(args.coagency_projects, "coagency projects parquet")
    _require_file(args.coagency_hits, "coagency hits parquet")
    _require_file(args.ea_pages, "EA pages parquet")
    _require_file(args.eis_pages, "EIS pages parquet")

    print("=== Coagency Name Extraction ===")
    print(f"Projects: {args.projects}")
    print(f"Coagency projects: {args.coagency_projects}")
    print(f"Coagency hits: {args.coagency_hits}")
    print(f"EA pages: {args.ea_pages}")
    print(f"EIS pages: {args.eis_pages}")

    text_pages, ce_projects = _read_targets(args)
    print(f"Text hit page rows: {len(text_pages):,}")
    print(f"CE metadata target rows: {len(ce_projects):,}")

    text_rows = _parse_text_hit_rows(text_pages)
    metadata_rows = _parse_metadata_rows(ce_projects)
    name_hits = pd.concat([text_rows, metadata_rows], ignore_index=True)
    if not name_hits.empty:
        name_hits = name_hits.drop_duplicates().reset_index(drop=True)

    target_projects = _target_projects_from_inputs(text_pages, ce_projects)
    fallback_targets = _projects_without_parsed_agencies(target_projects, name_hits)
    fallback_pages = pd.DataFrame()
    if args.fallback_scan and not fallback_targets.empty:
        fallback_pages = _read_fallback_pages(args, fallback_targets)
        print(f"Fallback no-name target projects: {len(fallback_targets):,}")
        print(f"Fallback candidate page rows: {len(fallback_pages):,}")
        fallback_rows = _parse_text_hit_rows(
            fallback_pages,
            source="document_text_fallback",
            extraction_method="fallback_project_scan",
        )
        if not fallback_rows.empty:
            name_hits = pd.concat([name_hits, fallback_rows], ignore_index=True)
            name_hits = name_hits.drop_duplicates().reset_index(drop=True)

    project_rollup = _build_project_rollup(name_hits, target_projects)
    department_pairs = _build_department_pairs(name_hits, target_projects)

    projects_with_names = pd.read_parquet(args.coagency_projects).merge(
        project_rollup.drop(columns=["process_type"], errors="ignore"),
        on=["project_id", "dataset_source"],
        how="left",
    )
    for col in (
        "coagency_agencies",
        "coagency_departments",
        "coagency_lead_agencies",
        "coagency_partner_agencies",
        "coagency_partner_departments",
    ):
        projects_with_names[col] = projects_with_names[col].fillna("[]")
    projects_with_names["coagency_name_extraction_count"] = (
        projects_with_names["coagency_name_extraction_count"].fillna(0).astype(int)
    )

    for path in (
        args.output_name_hits,
        args.output_project_agencies,
        args.output_department_pairs,
        args.output_projects_with_names,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    name_hits.to_parquet(args.output_name_hits, index=False)
    project_rollup.to_parquet(args.output_project_agencies, index=False)
    department_pairs.to_parquet(args.output_department_pairs, index=False)
    projects_with_names.to_parquet(args.output_projects_with_names, index=False)
    _write_qa_tables(args, target_projects, name_hits, department_pairs)

    print("\n=== Summary ===")
    print("Target projects by process:")
    print(target_projects.groupby("process_type").size().to_string())
    print(f"Parsed agency rows: {len(name_hits):,}")
    print(f"Project rollup rows: {len(project_rollup):,}")
    print(f"Department pair rows: {len(department_pairs):,}")
    print(f"Saved: {args.output_name_hits}")
    print(f"Saved: {args.output_project_agencies}")
    print(f"Saved: {args.output_department_pairs}")
    print(f"Saved: {args.output_projects_with_names}")
    if args.qa_dir:
        print(f"Saved QA tables under: {args.qa_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract structured co-agency names from flagged projects.")
    parser.add_argument("--run", action="store_true", help="Run extraction.")
    parser.add_argument("--projects", type=Path, default=DEFAULT_PROJECTS_PATH)
    parser.add_argument("--coagency-projects", type=Path, default=DEFAULT_COAGENCY_PROJECTS_PATH)
    parser.add_argument("--coagency-hits", type=Path, default=DEFAULT_COAGENCY_HITS_PATH)
    parser.add_argument("--ea-pages", type=Path, default=DEFAULT_EA_PAGES_PATH)
    parser.add_argument("--eis-pages", type=Path, default=DEFAULT_EIS_PAGES_PATH)
    parser.add_argument("--output-name-hits", type=Path, default=DEFAULT_OUTPUT_NAME_HITS)
    parser.add_argument("--output-project-agencies", type=Path, default=DEFAULT_OUTPUT_PROJECT_AGENCIES)
    parser.add_argument("--output-department-pairs", type=Path, default=DEFAULT_OUTPUT_DEPARTMENT_PAIRS)
    parser.add_argument("--output-projects-with-names", type=Path, default=DEFAULT_OUTPUT_PROJECTS_WITH_NAMES)
    parser.add_argument("--qa-dir", type=Path, default=OUTPUT_TABLES_DIR)
    parser.add_argument(
        "--no-fallback-scan",
        dest="fallback_scan",
        action="store_false",
        help="Disable targeted page scan for flagged EA/EIS projects with no parsed agency names.",
    )
    parser.set_defaults(fallback_scan=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.run:
        parser.print_help()
        return
    run(args)


if __name__ == "__main__":
    main()
