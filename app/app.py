#!/usr/bin/env python3
"""Streamlit browser for NEPA decarbonization and fossil fuel projects and document text."""

from __future__ import annotations

import ast
import html
import json
import os
import re
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlencode

import duckdb
import pandas as pd
import streamlit as st


MAX_RESULTS = 500
ENERGY_PREFIX = "Renewable Energy Production - "
DATASET_URL = "https://huggingface.co/datasets/PNNL/NEPATEC2.0"
PAPER_URL = "https://www.pnnl.gov/sites/default/files/media/file/PNNL_PermitAI_NEPATECv2_Public_Release_20_08_25.pdf"
DEFAULT_DB_FILENAME = "nepa_reader.duckdb"

HF_DB_REPO_ENV = "NEPA_DB_HF_REPO"
HF_DB_FILENAME_ENV = "NEPA_DB_HF_FILENAME"
HF_DB_SUBDIR_ENV = "NEPA_DB_HF_SUBDIR"
HF_DB_REVISION_ENV = "NEPA_DB_HF_REVISION"

QP_TITLE = "q"
QP_PROJECT_ID = "pid"
QP_CATEGORY = "cat"
QP_PROCESS = "proc"
QP_AGENCY = "agency"
QP_STATE = "state"
QP_ENERGY = "energy"

# Deep-link params for restoring a specific view (distinct from the search
# filters above). QP_PROJECT_ID ("pid") stays a *filter*; these open a record.
QP_VIEW = "view"
QP_OPEN_PROJECT = "p"
QP_OPEN_DOC = "d"
QP_OPEN_PAGE = "pg"

# Param groups for the on-page "Share" links (kept in sync with build_query_params).
FILTER_QP_KEYS = (QP_TITLE, QP_PROJECT_ID, QP_CATEGORY, QP_PROCESS, QP_AGENCY, QP_STATE, QP_ENERGY)
DEEPLINK_QP_KEYS = (QP_VIEW, QP_OPEN_PROJECT, QP_OPEN_DOC, QP_OPEN_PAGE)

# Display labels for the raw project_energy_type values. The user-facing framing
# is "Decarb" (not "Clean Energy").
ENERGY_CATEGORY_LABELS = {"Clean": "Decarb", "Fossil": "Fossil Fuel", "Other": "Other"}

# Browse-tab sort options; the first is the default (matches legacy behavior).
BROWSE_SORT_OPTIONS = [
    "Title (A–Z)",
    "# Documents (high–low)",
    "Category",
    "Agency",
]

BULLET_RE = re.compile(r"^(?:[-*•]\s+|[0-9]+[.)]\s+|[a-z][.)]\s+)")
SECTION_HEADING_RE = re.compile(
    r"^(?:chapter|section|appendix|part)\s+[A-Za-z0-9IVXLC]+",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[A-Z]")
LETTERED_HEADING_RE = re.compile(r"^[A-Z][.)]\s+[A-Z]")
ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,/&()'\".\-:]{4,}$")
TOC_HEADING_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)
TOC_ENTRY_START_RE = re.compile(r"\b\d+\.\d+(?:\.\d+){0,3}\s+")
TOC_PAGE_REF_RE = re.compile(r"\s+([0-9]{1,4}(?:-[0-9]{1,4})?)$")
SECTION_NUMBER_ONLY_RE = re.compile(r"^\d+\.\d+(?:\.\d+){0,3}$")
LETTER_MARKER_ONLY_RE = re.compile(r"^[A-Z][.)]$")


def env_text(name: str, default: str = "") -> str:
    value = os.getenv(name, default)
    return value.strip() if isinstance(value, str) else default.strip()


# Public, client-shareable base URL for on-page share links. Intentionally the
# production HF Space URL even on localhost (the links are meant to be shared);
# override via NEPA_PUBLIC_APP_URL when the app is re-hosted elsewhere.
PUBLIC_APP_URL = env_text("NEPA_PUBLIC_APP_URL") or "https://kaseyzapatka-nepa-document-explorer.hf.space"


def resolve_db_path() -> Path:
    """Find the DuckDB file in local and deployment layouts."""
    here = Path(__file__).resolve().parent
    candidates = [
        here / "data" / "rag" / "nepa_reader.duckdb",         # HF layout with app.py at repo root
        here.parent / "data" / "rag" / "nepa_reader.duckdb",  # local layout with app/app.py
        Path.cwd() / "data" / "rag" / "nepa_reader.duckdb",   # fallback for unusual working dirs
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1] if here.name == "app" else candidates[0]


DB_PATH = resolve_db_path()


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def ensure_database_available() -> Path:
    """Return a local DB path, downloading from a HF dataset repo if configured."""
    local_path = resolve_db_path()
    if local_path.exists():
        return local_path

    repo_id = env_text(HF_DB_REPO_ENV)
    if not repo_id:
        return local_path

    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - import guarded by runtime config
        raise RuntimeError(
            "NEPA_DB_HF_REPO is set but huggingface_hub is unavailable. "
            "Install app requirements first."
        ) from exc

    filename = env_text(HF_DB_FILENAME_ENV, DEFAULT_DB_FILENAME) or DEFAULT_DB_FILENAME
    subdir = env_text(HF_DB_SUBDIR_ENV)
    revision = env_text(HF_DB_REVISION_ENV) or None
    remote_filename = f"{subdir.strip('/')}/{filename}" if subdir else filename

    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=remote_filename,
        revision=revision,
        local_dir=str(local_path.parent),
    )
    return Path(downloaded)


def init_state() -> None:
    defaults = {
        "view": "search",  # search | project | document
        "project_id": None,
        "project_title": None,
        "document_id": None,
        "document_name": None,
        "current_page": 1,
        "search_term": "",
        "search_project_id": "",
        "doc_search_term": "",
        "filter_category": [],
        "filter_process": [],
        "filter_energy": [],
        "filter_agency": [],
        "filter_state": [],
        "text_view_mode": "Readable",
        "global_search_term": "",
        "browse_sort": BROWSE_SORT_OPTIONS[0],
        "browse_page": 1,
        "_browse_sig": "",
        "_last_doc_search": "",
        "_query_params_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path, read_only=True)

    extension_dir = Path(db_path).resolve().parent / ".duckdb_extensions"
    if extension_dir.exists():
        try:
            con.execute(f"SET extension_directory='{sql_path(extension_dir)}'")
        except duckdb.Error:
            pass

    try:
        con.execute("LOAD fts")
    except duckdb.Error:
        # Global text search falls back to LIKE if FTS is unavailable.
        pass

    return con


def run_df(query: str, params: Sequence[object] | None = None) -> pd.DataFrame:
    con = get_connection(str(DB_PATH))
    if params is None:
        return con.execute(query).df()
    return con.execute(query, list(params)).df()


def encode_query_list(values: Sequence[str]) -> str:
    return json.dumps([clean_scalar(v) for v in values if clean_scalar(v)], separators=(",", ":"))


def decode_query_list(raw_value: str) -> list[str]:
    text = clean_scalar(raw_value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [clean_scalar(v) for v in parsed if clean_scalar(v)]
    except Exception:
        pass

    return [token for token in parse_multi_value(text) if token]


def get_query_value(key: str) -> str:
    value = st.query_params.get(key, "")
    if isinstance(value, list):
        return clean_scalar(value[0]) if value else ""
    return clean_scalar(value)


def clean_scalar(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return ""
    return text


def display_value(value: object, default: str = "-") -> str:
    text = clean_scalar(value)
    return text if text else default


def category_display_value(raw_value: object, default: str = "-") -> str:
    """Map a raw project_energy_type value to its user-facing category label."""
    text = clean_scalar(raw_value)
    if not text:
        return default
    return ENERGY_CATEGORY_LABELS.get(text) or ENERGY_CATEGORY_LABELS.get(
        text.title(), default
    )


def apply_query_params_to_state_once() -> None:
    if st.session_state.get("_query_params_loaded"):
        return

    st.session_state["_query_params_loaded"] = True

    title = get_query_value(QP_TITLE)
    project_id = get_query_value(QP_PROJECT_ID)
    category = decode_query_list(get_query_value(QP_CATEGORY))
    process = decode_query_list(get_query_value(QP_PROCESS))
    agency = decode_query_list(get_query_value(QP_AGENCY))
    state = decode_query_list(get_query_value(QP_STATE))
    energy = decode_query_list(get_query_value(QP_ENERGY))

    if title:
        st.session_state["search_term"] = title
    if project_id:
        st.session_state["search_project_id"] = project_id
    if category:
        st.session_state["filter_category"] = category
    if process:
        st.session_state["filter_process"] = process
    if agency:
        st.session_state["filter_agency"] = agency
    if state:
        st.session_state["filter_state"] = state
    if energy:
        st.session_state["filter_energy"] = energy

    # Deep links: restore a specific project/document/page view. These are
    # independent of the search filters above. Old filter-only URLs (no "view"
    # param) simply leave the view on "search".
    view_qp = get_query_value(QP_VIEW)
    open_project = get_query_value(QP_OPEN_PROJECT)
    open_doc = get_query_value(QP_OPEN_DOC)
    open_page = get_query_value(QP_OPEN_PAGE)

    if view_qp == "project" and open_project:
        st.session_state["project_id"] = open_project
        try:
            proj_df = get_project(str(DB_PATH), open_project)
            if not proj_df.empty:
                st.session_state["project_title"] = clean_scalar(proj_df.iloc[0].get("project_title"))
        except Exception:
            pass
        st.session_state["view"] = "project"
    elif view_qp == "document" and open_doc:
        st.session_state["document_id"] = open_doc
        try:
            doc_df = run_df(
                """
                SELECT d.file_name, d.project_id, pr.project_title
                FROM documents d
                JOIN projects pr USING (project_id)
                WHERE d.document_id = ?
                """,
                [open_doc],
            )
            if not doc_df.empty:
                st.session_state["document_name"] = clean_scalar(doc_df.iloc[0]["file_name"])
                st.session_state["project_id"] = clean_scalar(doc_df.iloc[0]["project_id"])
                st.session_state["project_title"] = clean_scalar(doc_df.iloc[0]["project_title"])
        except Exception:
            pass
        if open_page:
            try:
                st.session_state["current_page"] = max(1, int(open_page))
            except (TypeError, ValueError):
                pass
        st.session_state["view"] = "document"


def build_query_params() -> dict[str, str]:
    """Single source of truth for shareable params (filters + view deep links).

    Both sync_query_params_from_state (URL bar) and the on-page Share links read
    from this, so they can never drift apart.
    """
    desired: dict[str, str] = {}

    title = clean_scalar(st.session_state.get("search_term"))
    if title:
        desired[QP_TITLE] = title

    project_id = clean_scalar(st.session_state.get("search_project_id"))
    if project_id:
        desired[QP_PROJECT_ID] = project_id

    for key, state_key in (
        (QP_CATEGORY, "filter_category"),
        (QP_PROCESS, "filter_process"),
        (QP_AGENCY, "filter_agency"),
        (QP_STATE, "filter_state"),
        (QP_ENERGY, "filter_energy"),
    ):
        values = st.session_state.get(state_key, [])
        encoded = encode_query_list(values)
        if encoded and encoded != "[]":
            desired[key] = encoded

    # Deep-link params reflect the current view so the URL is always shareable;
    # on the search view they are absent.
    view = st.session_state.get("view", "search")
    if view == "project":
        open_project = clean_scalar(st.session_state.get("project_id"))
        if open_project:
            desired[QP_VIEW] = "project"
            desired[QP_OPEN_PROJECT] = open_project
    elif view == "document":
        open_doc = clean_scalar(st.session_state.get("document_id"))
        if open_doc:
            desired[QP_VIEW] = "document"
            desired[QP_OPEN_DOC] = open_doc
            open_project = clean_scalar(st.session_state.get("project_id"))
            if open_project:
                desired[QP_OPEN_PROJECT] = open_project
            current_page = st.session_state.get("current_page")
            try:
                if current_page:
                    desired[QP_OPEN_PAGE] = str(int(current_page))
            except (TypeError, ValueError):
                pass

    return desired


def sync_query_params_from_state() -> None:
    desired = build_query_params()
    params = st.query_params

    for key in list(params.keys()):
        if key not in desired:
            del params[key]

    for key, value in desired.items():
        if get_query_value(key) != value:
            params[key] = value


def build_share_url(keys: Sequence[str]) -> str:
    """Absolute, client-shareable URL carrying only the given param keys."""
    desired = build_query_params()
    subset = {key: desired[key] for key in keys if key in desired}
    if not subset:
        return PUBLIC_APP_URL
    return f"{PUBLIC_APP_URL}?{urlencode(subset)}"


def render_share_link(label: str, keys: Sequence[str], caption: str) -> None:
    with st.expander(label, expanded=False):
        st.code(build_share_url(keys), language=None)
        st.caption(caption)


def reset_search_filters() -> None:
    st.session_state["search_term"] = ""
    st.session_state["search_project_id"] = ""
    st.session_state["filter_category"] = []
    st.session_state["filter_process"] = []
    st.session_state["filter_agency"] = []
    st.session_state["filter_state"] = []
    st.session_state["filter_energy"] = []
    st.session_state["global_search_term"] = ""
    st.session_state["global_search_term_input"] = ""
    st.session_state["view"] = "search"


def unique_preserve_order(values: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)


def normalize_energy_token(token: str) -> str:
    cleaned = clean_scalar(token)
    if not cleaned:
        return ""
    if cleaned.lower().startswith(ENERGY_PREFIX.lower()):
        cleaned = cleaned[len(ENERGY_PREFIX):].strip()
    return cleaned


def parse_multi_value(raw_value: object, *, split_delimiters: bool = True) -> tuple[str, ...]:
    text = clean_scalar(raw_value)
    if not text:
        return tuple()

    parsed_items: list[object] | None = None

    if text.startswith("[") and text.endswith("]"):
        for parser in (json.loads, ast.literal_eval):
            try:
                candidate = parser(text)
                if isinstance(candidate, (list, tuple, set)):
                    parsed_items = list(candidate)
                    break
            except Exception:
                continue

    if parsed_items is None:
        if split_delimiters:
            parsed_items = re.split(r"[;,|]", text)
        else:
            parsed_items = [text]

    tokens: list[str] = []
    for item in parsed_items:
        token = clean_scalar(item)
        if not token:
            continue

        token = token.strip("[]").strip("\"").strip("'")
        token = re.sub(r"\s+", " ", token).strip()

        if not token or token.lower() in {"none", "nan", "null"}:
            continue
        tokens.append(token)

    if not tokens and text:
        fallback = re.sub(r"[\[\]\"']", "", text).strip()
        if fallback:
            tokens.append(fallback)

    return unique_preserve_order(tokens)


def parse_energy_values(raw_value: object) -> tuple[str, ...]:
    raw_tokens = parse_multi_value(raw_value)
    normalized = [normalize_energy_token(token) for token in raw_tokens]
    return unique_preserve_order([token for token in normalized if token])


def format_energy_value(raw_value: object, default: str = "-") -> str:
    tokens = parse_energy_values(raw_value)
    if not tokens:
        return default
    return ", ".join(tokens)


def format_multi_value(
    raw_value: object,
    default: str = "-",
    *,
    split_delimiters: bool = True,
) -> str:
    tokens = parse_multi_value(raw_value, split_delimiters=split_delimiters)
    if not tokens:
        return default
    return ", ".join(tokens)


def agency_display_value(lead_agency_harmonized: object, lead_agency: object) -> str:
    harmonized = format_multi_value(
        lead_agency_harmonized,
        default="",
        split_delimiters=False,
    )
    if harmonized:
        return harmonized
    original = format_multi_value(
        lead_agency,
        default="",
        split_delimiters=False,
    )
    if original:
        return original
    return "-"


def line_is_heading(line: str) -> bool:
    if not line:
        return False
    if SECTION_HEADING_RE.match(line):
        return True
    if NUMBERED_HEADING_RE.match(line):
        return True
    if LETTERED_HEADING_RE.match(line):
        return True
    if line.endswith(":") and len(line) <= 120:
        return True

    words = line.split()
    if 2 <= len(words) <= 14 and ALL_CAPS_HEADING_RE.match(line):
        letters = sum(ch.isalpha() for ch in line)
        uppercase = sum(ch.isupper() for ch in line)
        if letters > 0 and (uppercase / letters) >= 0.85:
            return True
    return False


def line_is_table_like(line: str) -> bool:
    if not line:
        return False

    if "|" in line and line.count("|") >= 2:
        return True

    tokens = line.split()
    if len(tokens) < 5:
        return False

    numeric = 0
    for token in tokens:
        cleaned = token.strip(",.%()")
        if cleaned and re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", cleaned):
            numeric += 1

    return numeric >= 3 and (numeric / len(tokens)) >= 0.45


def strip_bullet_prefix(line: str) -> str:
    return BULLET_RE.sub("", line, count=1).strip()


def join_wrapped_lines(lines: list[str]) -> str:
    if not lines:
        return ""

    merged = ""
    for line in lines:
        if not merged:
            merged = line
            continue

        if merged.endswith("-") and line and line[0].islower():
            merged = merged[:-1] + line
        else:
            merged = f"{merged} {line}"

    merged = re.sub(r"\s+([,.;:!?])", r"\1", merged)
    merged = re.sub(r"\(\s+", "(", merged)
    merged = re.sub(r"\s+\)", ")", merged)
    merged = re.sub(r"\s{2,}", " ", merged).strip()
    return merged


def format_toc_paragraph(text: str) -> str | None:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return None

    has_toc_heading = bool(TOC_HEADING_RE.search(compact))
    starts = list(TOC_ENTRY_START_RE.finditer(compact))

    # Require clear TOC signal to avoid false positives on normal narrative text.
    if len(starts) < 5:
        return None
    if not has_toc_heading and compact.count("...") < 3 and compact.count("..") < 6:
        return None

    preamble = ""
    toc_heading = ""
    body = compact

    if has_toc_heading:
        heading_match = TOC_HEADING_RE.search(compact)
        if heading_match:
            preamble = compact[: heading_match.start()].strip()
            toc_heading = heading_match.group(0).upper()
            body = compact[heading_match.end() :].strip()
            starts = list(TOC_ENTRY_START_RE.finditer(body))
            if len(starts) < 3:
                return None

    positions = [match.start() for match in starts]
    positions.append(len(body))

    lines: list[str] = []
    if preamble:
        lines.extend([preamble, ""])
    if toc_heading:
        lines.extend([toc_heading, ""])

    for idx in range(len(starts)):
        segment = body[positions[idx] : positions[idx + 1]].strip()
        if not segment:
            continue

        segment = re.sub(r"\s+", " ", segment).strip()
        if not segment:
            continue

        page_match = TOC_PAGE_REF_RE.search(segment)
        if page_match:
            title = segment[: page_match.start()].rstrip()
            page_ref = page_match.group(1)
            if title:
                # Preserve dot leaders when present; otherwise keep simple spacing.
                if re.search(r"\.{3,}\s*$", title):
                    segment = f"{title} {page_ref}"
                else:
                    segment = f"{title}  {page_ref}"

        lines.append(segment)

    meaningful = [line for line in lines if line.strip()]
    if len(meaningful) < 4:
        return None

    return "\n".join(lines).strip()


def merge_orphan_section_markers(lines: list[str]) -> list[str]:
    merged: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line:
            merged.append("")
            i += 1
            continue

        marker_only = bool(SECTION_NUMBER_ONLY_RE.fullmatch(line) or LETTER_MARKER_ONLY_RE.fullmatch(line))
        if marker_only:
            j = i + 1
            while j < n and not lines[j]:
                j += 1

            if j < n:
                nxt = lines[j]
                next_is_marker = bool(
                    SECTION_NUMBER_ONLY_RE.fullmatch(nxt) or LETTER_MARKER_ONLY_RE.fullmatch(nxt)
                )
                if nxt and not next_is_marker:
                    merged.append(f"{line} {nxt}".strip())
                    i = j + 1
                    continue

        merged.append(line)
        i += 1

    return merged


FORM_BLANK_RE = re.compile(r"_{3,}|\.{5,}")
FORM_CHECKBOX_RE = re.compile(r"\[\s*[xX ]?\s*\]|\(\s*[xX ]?\s*\)|[\u2610\u2612\u25a2\u25a0\u25fb\u25fc]")
FORM_YESNO_RE = re.compile(r"\byes\b\s*/?\s*\bno\b", re.IGNORECASE)


def page_is_form_like(lines: list[str]) -> bool:
    """Heuristic: is this page a fill-in form / label list rather than narrative?

    Conservative on purpose \u2014 narrative (the EA/EIS common case, which defaults to
    Readable) must NOT trip this, so we require dominant short lines and low prose
    density before firing. Signals: short-line fraction, colon-terminated label
    fraction, form markers (underscore/dot blanks, checkboxes, Yes/No), low words
    per line.
    """
    content = [line for line in lines if line.strip()]
    n = len(content)
    if n < 8:
        return False

    short = sum(1 for line in content if len(line) <= 45)
    colon = sum(1 for line in content if line.rstrip().endswith(":"))
    markers = sum(
        1
        for line in content
        if FORM_BLANK_RE.search(line) or FORM_CHECKBOX_RE.search(line) or FORM_YESNO_RE.search(line)
    )
    avg_words = sum(len(line.split()) for line in content) / n
    short_frac = short / n
    colon_frac = colon / n
    marker_frac = markers / n

    # A form/label/table page needs a strong non-prose signal (form markers, or a
    # high colon-label density) in addition to short, low-word-count lines. Short
    # lines ALONE are usually OCR-wrapped narrative, which must keep merging, so
    # empirically (CE/EA/EIS samples) requiring a strong signal keeps the EIS/EA
    # false-positive rate near ~1.5% while still catching real forms/tables.
    strong_signal = marker_frac >= 0.10 or colon_frac >= 0.30
    return strong_signal and short_frac >= 0.60 and avg_words <= 8.0


def prepare_page_text(raw_text: str, view_mode: str) -> str:
    text = (raw_text or "").replace("\r\n", "\n").replace("\r", "\n")
    if view_mode != "Readable":
        return text

    raw_lines = text.split("\n")
    normalized_lines = [
        re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()
        for line in raw_lines
    ]
    normalized_lines = merge_orphan_section_markers(normalized_lines)

    # Form guard: preserve the line structure of form-like pages instead of
    # merging wrapped lines into paragraphs (line joining would destroy them).
    if page_is_form_like(normalized_lines):
        kept: list[str] = []
        for line in normalized_lines:
            if line:
                kept.append(line)
            elif kept and kept[-1] != "":
                kept.append("")
        readable = "\n".join(kept).strip()
        return readable if readable else text

    blocks: list[tuple[str, str]] = []
    paragraph_lines: list[str] = []
    current_bullet_index: int | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if paragraph_lines:
            merged = join_wrapped_lines(paragraph_lines)
            if merged:
                toc_formatted = format_toc_paragraph(merged)
                if toc_formatted:
                    blocks.append(("toc", toc_formatted))
                else:
                    blocks.append(("paragraph", merged))
            paragraph_lines = []

    for line in normalized_lines:
        if not line:
            flush_paragraph()
            current_bullet_index = None
            continue

        if line_is_heading(line):
            flush_paragraph()
            blocks.append(("heading", line))
            current_bullet_index = None
            continue

        if BULLET_RE.match(line):
            flush_paragraph()
            bullet_text = strip_bullet_prefix(line)
            if bullet_text:
                blocks.append(("bullet", bullet_text))
                current_bullet_index = len(blocks) - 1
            continue

        if line_is_table_like(line):
            flush_paragraph()
            blocks.append(("table", line))
            current_bullet_index = None
            continue

        if (
            current_bullet_index is not None
            and current_bullet_index < len(blocks)
            and blocks[current_bullet_index][0] == "bullet"
            and (line[0].islower() or len(line.split()) <= 6)
        ):
            kind, existing = blocks[current_bullet_index]
            blocks[current_bullet_index] = (kind, join_wrapped_lines([existing, line]))
            continue

        current_bullet_index = None
        paragraph_lines.append(line)

    flush_paragraph()

    output_lines: list[str] = []
    for idx, (kind, value) in enumerate(blocks):
        if kind == "heading":
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(value)
            output_lines.append("")
        elif kind == "bullet":
            output_lines.append(f"- {value}")
        elif kind == "table":
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(value)
        elif kind == "toc":
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.extend(value.split("\n"))
        else:
            if output_lines and output_lines[-1] != "":
                output_lines.append("")
            output_lines.append(value)

        if idx == len(blocks) - 1 and output_lines and output_lines[-1] == "":
            output_lines.pop()

    readable = "\n".join(output_lines).strip()
    toc_page = format_toc_paragraph(readable)
    if toc_page:
        readable = toc_page
    return readable if readable else text


def highlight_text(text: str, term: str) -> str:
    if not term or not term.strip():
        return html.escape(text)

    pattern = re.compile(re.escape(term), re.IGNORECASE)
    parts: list[str] = []
    last_index = 0

    for match in pattern.finditer(text):
        parts.append(html.escape(text[last_index:match.start()]))
        parts.append(f"<mark>{html.escape(match.group(0))}</mark>")
        last_index = match.end()

    parts.append(html.escape(text[last_index:]))
    return "".join(parts)


def render_page_text(text: str, search_term: str, view_mode: str = "Readable") -> None:
    term = clean_scalar(search_term)
    if term:
        html_text = highlight_text(text, term)
    else:
        html_text = html.escape(text)

    if view_mode == "Readable":
        # Comfortable reading column: constrained measure, generous leading,
        # real paragraph spacing. Split on blank lines into <p> blocks (a <mark>
        # highlight never spans a blank line, so this preserves highlighting);
        # single newlines within a block stay <br>.
        paragraphs = []
        for block in html_text.split("\n\n"):
            block_html = block.replace("\n", "<br>")
            if block_html.strip():
                paragraphs.append(f"<p style='margin:0 0 0.85em 0;'>{block_html}</p>")
        body = "".join(paragraphs) if paragraphs else html_text.replace("\n", "<br>")
        st.markdown(
            (
                "<div style='max-width:75ch; line-height:1.6; font-size:1.05rem; "
                "font-family: \"Source Sans Pro\", sans-serif;'>"
                f"{body}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        return

    # Source mode: keep the compact, structure-faithful rendering.
    html_text = html_text.replace("\n", "<br>")
    st.markdown(
        (
            "<div style='line-height:1.45; font-family: \"Source Sans Pro\", "
            "sans-serif; white-space: normal;'>"
            f"{html_text}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


@st.cache_data
def get_project_index(db_path: str) -> pd.DataFrame:
    del db_path  # cache key only

    project_columns = set(run_df("PRAGMA table_info('projects')")["name"].tolist())
    if "lead_agency_harmonized" in project_columns:
        harmonized_expr = "p.lead_agency_harmonized"
    else:
        harmonized_expr = "NULL AS lead_agency_harmonized"

    if "project_energy_type" in project_columns:
        energy_type_expr = "p.project_energy_type"
    else:
        energy_type_expr = "NULL AS project_energy_type"

    base = run_df(
        f"""
        SELECT
            p.project_id,
            p.project_title,
            p.lead_agency,
            {harmonized_expr},
            p.project_state,
            p.process_type,
            p.project_type,
            {energy_type_expr},
            COUNT(d.document_id) AS n_documents
        FROM projects p
        LEFT JOIN documents d USING (project_id)
        GROUP BY 1,2,3,4,5,6,7,8
        """
    )

    if base.empty:
        return base

    base = base.copy()
    base["project_title"] = base["project_title"].fillna("").astype(str)
    base["process_type"] = base["process_type"].fillna("").astype(str).str.strip()

    base["agency_display"] = base.apply(
        lambda row: agency_display_value(row.get("lead_agency_harmonized"), row.get("lead_agency")),
        axis=1,
    )
    base["state_values"] = base["project_state"].apply(parse_multi_value)
    base["state_display"] = base["project_state"].apply(format_multi_value)

    base["energy_values"] = base["project_type"].apply(parse_energy_values)
    base["energy_display"] = base["project_type"].apply(format_energy_value)

    base["category_display"] = base["project_energy_type"].apply(category_display_value)

    return base


@st.cache_data
def get_filter_options(
    db_path: str,
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    index_df = get_project_index(db_path)

    if index_df.empty:
        return [], [], ["CE", "EA", "EIS"], [], []

    agencies = sorted(
        value for value in index_df["agency_display"].dropna().unique().tolist() if value and value != "-"
    )
    states = sorted({state for vals in index_df["state_values"] for state in vals})

    process_types = sorted(
        value for value in index_df["process_type"].dropna().unique().tolist() if value
    )

    energy_types = sorted({energy for vals in index_df["energy_values"] for energy in vals})

    categories = sorted(
        value for value in index_df["category_display"].dropna().unique().tolist() if value and value != "-"
    )

    return agencies, states, process_types, energy_types, categories


@st.cache_data
def get_corpus_stats(db_path: str) -> dict[str, object]:
    """Cached corpus-wide counts for the Browse header (computed from the DB)."""
    del db_path  # cache key only

    n_projects = int(run_df("SELECT COUNT(*) AS n FROM projects")["n"].iloc[0])
    n_documents = int(run_df("SELECT COUNT(*) AS n FROM documents")["n"].iloc[0])
    n_pages = int(run_df("SELECT COUNT(*) AS n FROM pages")["n"].iloc[0])

    project_columns = set(run_df("PRAGMA table_info('projects')")["name"].tolist())
    categories: dict[str, int] = {}
    if "project_energy_type" in project_columns:
        cat_df = run_df(
            "SELECT project_energy_type, COUNT(*) AS n FROM projects GROUP BY 1"
        )
        for _, row in cat_df.iterrows():
            label = category_display_value(row["project_energy_type"])
            if label == "-":
                continue
            categories[label] = categories.get(label, 0) + int(row["n"])

    return {
        "projects": n_projects,
        "documents": n_documents,
        "pages": n_pages,
        "categories": categories,
    }


def format_corpus_stats_line(stats: dict[str, object]) -> str:
    parts = [f"{int(stats['projects']):,} projects"]
    categories = stats.get("categories", {}) or {}
    for label in ("Decarb", "Fossil Fuel", "Other"):
        if label in categories:
            parts.append(f"{categories[label]:,} {label}")
    parts.append(f"{int(stats['documents']):,} documents")

    pages = int(stats["pages"])
    if pages >= 1_000_000:
        pages_str = f"{pages / 1_000_000:.1f}M searchable pages"
    else:
        pages_str = f"{pages:,} searchable pages"
    parts.append(pages_str)
    return " · ".join(parts)


def sort_browse_results(df: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if df.empty:
        return df
    if sort_choice == "# Documents (high–low)":
        return df.sort_values(
            by=["n_documents", "project_title", "project_id"],
            ascending=[False, True, True],
            na_position="last",
        )
    if sort_choice == "Category":
        return df.sort_values(
            by=["category_display", "project_title", "project_id"], na_position="last"
        )
    if sort_choice == "Agency":
        return df.sort_values(
            by=["agency_display", "project_title", "project_id"], na_position="last"
        )
    # Default: Title (A–Z) — matches legacy behavior.
    return df.sort_values(by=["project_title", "project_id"], na_position="last")


def sanitize_filename(name: object, default: str = "document") -> str:
    stem = Path(clean_scalar(name)).stem or default
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or default
    return f"{stem}.txt"


@st.cache_data
def get_document_text(db_path: str, document_id: str) -> str:
    """Full document text with page separators, cached per document."""
    del db_path  # cache key only

    pages_df = get_page_list(str(DB_PATH), document_id)
    if pages_df.empty:
        return ""

    blocks: list[str] = []
    for row in pages_df.itertuples(index=False):
        page_num = int(row.page_number_int) if pd.notna(row.page_number_int) else 0
        label = str(page_num) if page_num > 0 else str(row.page_number)
        body = str(row.page_text or "")
        blocks.append(f"----- Page {label} -----\n\n{body}")
    return "\n\n".join(blocks)


@st.cache_data
def get_document_source(db_path: str, document_id: str) -> str:
    """dataset_source ('CE'/'EA'/'EIS') for a document, cached."""
    del db_path  # cache key only
    df = run_df(
        "SELECT dataset_source FROM documents WHERE document_id = ?", [document_id]
    )
    if df.empty:
        return ""
    return clean_scalar(df.iloc[0]["dataset_source"]).upper()


@st.cache_data
def search_projects(
    db_path: str,
    project_id_query: str,
    title_query: str,
    categories: tuple[str, ...],
    process_types: tuple[str, ...],
    energy_types: tuple[str, ...],
    agencies: tuple[str, ...],
    states: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    index_df = get_project_index(db_path)

    if index_df.empty:
        return index_df, index_df, 0

    mask = pd.Series(True, index=index_df.index)

    project_id_query = clean_scalar(project_id_query)
    if project_id_query:
        mask &= index_df["project_id"].astype(str).str.contains(
            project_id_query, case=False, regex=False, na=False
        )

    title_query = clean_scalar(title_query)
    if title_query:
        mask &= index_df["project_title"].astype(str).str.contains(
            title_query, case=False, regex=False, na=False
        )

    if categories:
        category_set = set(categories)
        mask &= index_df["category_display"].isin(category_set)

    if process_types:
        process_set = set(process_types)
        mask &= index_df["process_type"].isin(process_set)

    if agencies:
        agency_set = set(agencies)
        mask &= index_df["agency_display"].isin(agency_set)

    if energy_types:
        energy_set = set(energy_types)
        mask &= index_df["energy_values"].apply(lambda vals: bool(energy_set.intersection(vals)))

    if states:
        state_set = set(states)
        mask &= index_df["state_values"].apply(lambda vals: bool(state_set.intersection(vals)))

    filtered = index_df.loc[mask].sort_values(by=["project_title", "project_id"], na_position="last")
    total_matches = len(filtered)

    filtered_all = filtered.reset_index(drop=True)
    filtered_view = filtered_all.head(MAX_RESULTS).copy()
    return filtered_view, filtered_all, total_matches


@st.cache_data
def get_project(db_path: str, project_id: str) -> pd.DataFrame:
    del db_path  # cache key only
    return run_df("SELECT * FROM projects WHERE project_id = ?", [project_id])


@st.cache_data
def get_documents(db_path: str, project_id: str) -> pd.DataFrame:
    del db_path  # cache key only
    return run_df(
        """
        SELECT
            document_id,
            file_name,
            document_type_clean,
            document_type_category,
            main_document,
            total_pages,
            dataset_source
        FROM documents
        WHERE project_id = ?
        ORDER BY
            CASE WHEN main_document = 'YES' THEN 0 ELSE 1 END,
            total_pages DESC NULLS LAST
        """,
        [project_id],
    )


@st.cache_data
def get_page_list(db_path: str, document_id: str) -> pd.DataFrame:
    del db_path  # cache key only
    return run_df(
        """
        SELECT
            row_number() OVER (ORDER BY page_number_int, page_number, rowid) AS page_ordinal,
            page_number,
            page_number_int,
            page_text
        FROM pages
        WHERE document_id = ?
        ORDER BY page_number_int, page_number, rowid
        """,
        [document_id],
    )


@st.cache_data
def search_within_document(db_path: str, document_id: str, term: str) -> tuple[int, ...]:
    del db_path  # cache key only

    if not term or len(term.strip()) < 2:
        return tuple()

    results = run_df(
        """
        WITH ordered_pages AS (
            SELECT
                row_number() OVER (ORDER BY page_number_int, page_number, rowid) AS page_ordinal,
                page_text
            FROM pages
            WHERE document_id = ?
        )
        SELECT page_ordinal
        FROM ordered_pages
        WHERE LOWER(page_text) LIKE LOWER(?)
        ORDER BY page_ordinal
        """,
        [document_id, f"%{term}%"],
    )

    return tuple(int(x) for x in results["page_ordinal"].tolist())


def make_snippet(page_text: object, term: str, max_len: int = 280) -> str:
    text = re.sub(r"\s+", " ", clean_scalar(page_text)).strip()
    if not text:
        return ""

    term_clean = clean_scalar(term)
    if term_clean:
        lower_text = text.lower()
        lower_term = term_clean.lower()
        idx = lower_text.find(lower_term)
        if idx >= 0:
            start = max(0, idx - 100)
            end = min(len(text), idx + len(term_clean) + 160)
            snippet = text[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(text):
                snippet = snippet + "..."
            return snippet

    return text if len(text) <= max_len else f"{text[:max_len].rstrip()}..."


@st.cache_data
def search_global_pages(
    db_path: str,
    term: str,
    filtered_pids: tuple[str, ...] | None = None,
    display_limit: int = 120,
) -> tuple[pd.DataFrame, str]:
    """Full-text (or LIKE-fallback) page search.

    When ``filtered_pids`` is provided (any sidebar filter is active), the FTS
    ranked scan is widened to 2000 rows, inner-merged against the filtered
    project_id set in pandas, then trimmed to ``display_limit``. ``filtered_pids``
    participates in the cache key so distinct filter sets cache separately. When
    it is ``None`` (no filters), the original ``display_limit`` scan is used.
    """
    del db_path  # cache key only

    query_term = clean_scalar(term)
    if len(query_term) < 2:
        return pd.DataFrame(), "none"

    fetch_limit = 2000 if filtered_pids is not None else display_limit

    fts_query = """
        WITH scored AS (
            SELECT
                p.document_id,
                p.page_number,
                p.page_number_int,
                p.page_text,
                fts_main_pages.match_bm25(p.rowid, ?) AS score
            FROM pages p
        ),
        ranked AS (
            SELECT *
            FROM scored
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        )
        SELECT
            r.document_id,
            r.page_number,
            r.page_number_int,
            r.page_text,
            r.score,
            d.project_id,
            d.file_name,
            d.dataset_source,
            d.document_type_clean,
            pr.project_title,
            pr.lead_agency,
            pr.lead_agency_harmonized
        FROM ranked r
        JOIN documents d USING (document_id)
        JOIN projects pr USING (project_id)
        ORDER BY r.score DESC
    """

    like_query = """
        SELECT
            p.document_id,
            p.page_number,
            p.page_number_int,
            p.page_text,
            NULL::DOUBLE AS score,
            d.project_id,
            d.file_name,
            d.dataset_source,
            d.document_type_clean,
            pr.project_title,
            pr.lead_agency,
            pr.lead_agency_harmonized
        FROM pages p
        JOIN documents d USING (document_id)
        JOIN projects pr USING (project_id)
        WHERE LOWER(p.page_text) LIKE LOWER(?)
        LIMIT ?
    """

    mode = "fts"
    try:
        results = run_df(fts_query, [query_term, fetch_limit])
    except duckdb.Error:
        mode = "like"
        results = run_df(like_query, [f"%{query_term}%", fetch_limit])

    if results.empty:
        return results, mode

    # Filter-aware: keep only pages whose project is in the active filtered set,
    # then trim to the display limit (results are already score-ordered for FTS).
    if filtered_pids is not None:
        keep = set(filtered_pids)
        results = results[results["project_id"].astype(str).isin(keep)]

    results = results.head(display_limit)
    if results.empty:
        return results, mode

    results = results.copy()
    results["agency_display"] = results.apply(
        lambda row: agency_display_value(row.get("lead_agency_harmonized"), row.get("lead_agency")),
        axis=1,
    )
    results["page_label"] = results.apply(
        lambda row: str(int(row["page_number_int"]))
        if pd.notna(row["page_number_int"]) and int(row["page_number_int"]) > 0
        else clean_scalar(row["page_number"]),
        axis=1,
    )
    results["snippet"] = results["page_text"].apply(lambda value: make_snippet(value, query_term))
    return results, mode


def extract_selected_row(event: object) -> int | None:
    if event is None:
        return None

    selection = getattr(event, "selection", None)
    if selection is not None:
        rows = getattr(selection, "rows", None)
        if rows:
            return int(rows[0])

    if isinstance(event, dict):
        rows = event.get("selection", {}).get("rows", [])
        if rows:
            return int(rows[0])

    return None


def resolve_page_ordinal(document_id: str, page_number_int: object, page_number: object) -> int:
    pages_df = get_page_list(str(DB_PATH), document_id)
    if pages_df.empty:
        return 1

    if pd.notna(page_number_int):
        try:
            page_num = int(page_number_int)
            match = pages_df[pages_df["page_number_int"] == page_num]
            if not match.empty:
                return int(match.iloc[0]["page_ordinal"])
        except Exception:
            pass

    raw_page = clean_scalar(page_number)
    if raw_page:
        match = pages_df[pages_df["page_number"].astype(str) == raw_page]
        if not match.empty:
            return int(match.iloc[0]["page_ordinal"])

    return 1


def render_document_search_tab(
    filtered_pids: tuple[str, ...] | None, filters_active: bool
) -> None:
    if filters_active:
        st.caption(
            "Full-text search across every document page, limited to the projects "
            "that match your current sidebar filters. Select a result row to open that page."
        )
    else:
        st.caption(
            "Full-text search across every document page in the corpus. "
            "Select a result row to open that page."
        )

    if "global_search_term_input" not in st.session_state:
        st.session_state["global_search_term_input"] = st.session_state.get("global_search_term", "")

    query_input = st.text_input(
        "Search term",
        key="global_search_term_input",
        placeholder="e.g. wetlands mitigation",
    )

    button_col1, button_col2 = st.columns([1, 1])
    do_search = button_col1.button("Run Text Search", use_container_width=True)
    do_clear = button_col2.button("Clear Text Search", use_container_width=True)

    if do_clear:
        st.session_state["global_search_term"] = ""
        st.session_state["global_search_term_input"] = ""
        st.rerun()

    if do_search:
        st.session_state["global_search_term"] = clean_scalar(query_input)
        st.rerun()

    active_term = clean_scalar(st.session_state.get("global_search_term"))
    if not active_term:
        return

    results, mode = search_global_pages(str(DB_PATH), active_term, filtered_pids)
    if results.empty:
        if filters_active:
            st.info("No matches for this term within the current filters. Try broadening the filters.")
        else:
            st.info("No cross-document matches found for this term.")
        return

    mode_label = "FTS" if mode == "fts" else "LIKE fallback"
    scope = " (within current filters)" if filters_active else ""
    st.caption(f"{len(results):,} matches shown{scope} ({mode_label}).")

    display_cols = {
        "project_id": "Project ID",
        "project_title": "Project Title",
        "file_name": "Document",
        "dataset_source": "Source",
        "page_label": "Page",
        "snippet": "Snippet",
    }

    if "score" in results.columns and results["score"].notna().any():
        display_cols["score"] = "Score"

    display_df = results[list(display_cols.keys())].rename(columns=display_cols)
    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=360,
        on_select="rerun",
        selection_mode="single-row",
    )

    row_idx = extract_selected_row(event)
    if row_idx is None:
        return

    selected = results.iloc[row_idx]
    st.session_state["project_id"] = selected["project_id"]
    st.session_state["project_title"] = selected["project_title"]
    st.session_state["document_id"] = selected["document_id"]
    st.session_state["document_name"] = selected["file_name"]
    st.session_state["current_page"] = resolve_page_ordinal(
        selected["document_id"],
        selected["page_number_int"],
        selected["page_number"],
    )
    st.session_state["view"] = "document"
    st.rerun()


def sync_multiselect_values(key: str, options: Sequence[str]) -> None:
    current = st.session_state.get(key, [])
    st.session_state[key] = [value for value in current if value in options]


def render_data_citation() -> None:
    with st.sidebar.expander("Data Source & Citation", expanded=False):
        st.markdown(
            "This explorer uses the **NEPATEC v2.0** dataset curated by "
            "Pacific Northwest National Laboratory (PNNL)."
        )
        st.markdown(
            "Citation: Munikoti, S. et al. (2025). *NEPATEC v2.0: Standardized "
            "Metadata and Text Corpus of National Environmental Policy Act Documents*."
        )
        st.markdown(f"[Dataset page]({DATASET_URL})")
        st.markdown(f"[PNNL release paper]({PAPER_URL})")
        st.caption("License: CC0 1.0 (public domain).")


def render_sidebar() -> tuple[str, str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    st.sidebar.header("Filter Projects")

    if st.sidebar.button("Reset All Filters", use_container_width=True):
        reset_search_filters()
        st.rerun()

    agencies, states, process_types, energy_types, categories = get_filter_options(str(DB_PATH))

    sync_multiselect_values("filter_category", categories)
    sync_multiselect_values("filter_process", process_types)
    sync_multiselect_values("filter_energy", energy_types)
    sync_multiselect_values("filter_agency", agencies)
    sync_multiselect_values("filter_state", states)

    project_id_query = st.sidebar.text_input(
        "Project ID contains",
        key="search_project_id",
        placeholder="e.g. 2024-12345",
    )

    title_query = st.sidebar.text_input(
        "Project title contains",
        key="search_term",
    )

    selected_category = st.sidebar.multiselect(
        "Category",
        options=categories,
        key="filter_category",
    )

    selected_process = st.sidebar.multiselect(
        "Process type",
        options=process_types,
        key="filter_process",
    )

    selected_agency = st.sidebar.multiselect(
        "Agency (harmonized)",
        options=agencies,
        key="filter_agency",
    )

    selected_state = st.sidebar.multiselect(
        "State",
        options=states,
        key="filter_state",
    )

    selected_energy = st.sidebar.multiselect(
        "Energy type",
        options=energy_types,
        key="filter_energy",
    )

    render_data_citation()

    return (
        project_id_query,
        title_query,
        tuple(selected_category),
        tuple(selected_process),
        tuple(selected_energy),
        tuple(selected_agency),
        tuple(selected_state),
    )


def render_browse_tab(results_all: pd.DataFrame, filter_sig: str) -> None:
    stats = get_corpus_stats(str(DB_PATH))
    st.markdown(format_corpus_stats_line(stats))

    # Share the current filtered search — only when at least one filter is active.
    if build_share_url(FILTER_QP_KEYS) != PUBLIC_APP_URL:
        render_share_link(
            "🔗 Share this search",
            FILTER_QP_KEYS,
            "Anyone with this link opens these same filters.",
        )

    sort_choice = st.selectbox("Sort by", BROWSE_SORT_OPTIONS, key="browse_sort")

    # Reset to page 1 whenever the filter set or sort changes.
    sig = f"{filter_sig}||{sort_choice}"
    if st.session_state.get("_browse_sig") != sig:
        st.session_state["_browse_sig"] = sig
        st.session_state["browse_page"] = 1

    sorted_all = sort_browse_results(results_all, sort_choice)
    total = len(sorted_all)

    if total == 0:
        st.info("No projects found. Try broadening your filters.")
        return

    # CSV export always covers the FULL filtered set.
    export_cols = [
        "project_id",
        "project_title",
        "category_display",
        "agency_display",
        "state_display",
        "process_type",
        "energy_display",
        "n_documents",
    ]
    export_df = sorted_all[export_cols].rename(
        columns={
            "project_id": "Project ID",
            "project_title": "Project Title",
            "category_display": "Category",
            "agency_display": "Agency",
            "state_display": "State",
            "process_type": "Type",
            "energy_display": "Energy Type",
            "n_documents": "# Docs",
        }
    )
    st.download_button(
        "Download Filtered Projects CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="nepa_filtered_projects.csv",
        mime="text/csv",
        use_container_width=False,
    )

    n_pages = (total + MAX_RESULTS - 1) // MAX_RESULTS
    page = min(max(1, int(st.session_state.get("browse_page", 1))), n_pages)
    st.session_state["browse_page"] = page
    start = (page - 1) * MAX_RESULTS
    end = min(start + MAX_RESULTS, total)

    st.write(f"**Showing {start + 1:,}–{end:,} of {total:,} projects.**")
    st.caption("Select a row to open a project.")

    if total > MAX_RESULTS:
        nav_prev, nav_mid, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if st.button("<- Prev page", disabled=(page <= 1), key="browse_prev", use_container_width=True):
                st.session_state["browse_page"] = page - 1
                st.rerun()
        with nav_mid:
            st.markdown(
                f"<div style='text-align:center'>Page {page} of {n_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button("Next page ->", disabled=(page >= n_pages), key="browse_next", use_container_width=True):
                st.session_state["browse_page"] = page + 1
                st.rerun()

    page_slice = sorted_all.iloc[start:end].reset_index(drop=True)

    display_cols = {
        "project_id": "Project ID",
        "project_title": "Project Title",
        "category_display": "Category",
        "agency_display": "Agency",
        "state_display": "State",
        "process_type": "Type",
        "energy_display": "Energy Type",
        "n_documents": "# Docs",
    }
    display_df = page_slice[list(display_cols.keys())].rename(columns=display_cols)

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
        on_select="rerun",
        selection_mode="single-row",
    )

    row_idx = extract_selected_row(event)
    if row_idx is None:
        return

    selected = page_slice.iloc[row_idx]
    st.session_state["project_id"] = selected["project_id"]
    st.session_state["project_title"] = selected["project_title"]
    st.session_state["view"] = "project"
    st.rerun()


def render_search(
    project_id_query: str,
    title_query: str,
    categories: tuple[str, ...],
    process_types: tuple[str, ...],
    energy_types: tuple[str, ...],
    agencies: tuple[str, ...],
    states: tuple[str, ...],
) -> None:
    st.title("NEPA Document Explorer")
    st.caption("Browse decarbonization and fossil fuel environmental review projects and document text.")

    _results_view, results_all, _total_matches = search_projects(
        str(DB_PATH),
        project_id_query,
        title_query,
        categories,
        process_types,
        energy_types,
        agencies,
        states,
    )

    filters_active = any(
        [
            clean_scalar(project_id_query),
            clean_scalar(title_query),
            categories,
            process_types,
            energy_types,
            agencies,
            states,
        ]
    )
    filtered_pids = (
        tuple(sorted(results_all["project_id"].astype(str).tolist()))
        if filters_active and not results_all.empty
        else None
    )
    filter_sig = repr(
        (
            clean_scalar(project_id_query),
            clean_scalar(title_query),
            tuple(categories),
            tuple(process_types),
            tuple(energy_types),
            tuple(agencies),
            tuple(states),
        )
    )

    tab_browse, tab_docs = st.tabs(["Browse Projects", "Search Documents"])

    with tab_browse:
        render_browse_tab(results_all, filter_sig)

    with tab_docs:
        render_document_search_tab(filtered_pids, filters_active)

    st.divider()
    st.caption(
        "This explorer runs on a shared server and streams a large document "
        "database; it may take a few minutes to wake after a period of inactivity."
    )


def render_project() -> None:
    if st.button("<- Back to Search"):
        st.session_state["view"] = "search"
        st.rerun()

    project_id = st.session_state.get("project_id")
    if not project_id:
        st.session_state["view"] = "search"
        st.rerun()

    project_df = get_project(str(DB_PATH), project_id)
    if project_df.empty:
        st.error("Project not found in database.")
        if st.button("Return to Search"):
            st.session_state["view"] = "search"
            st.rerun()
        return

    proj = project_df.iloc[0]

    agency_display = agency_display_value(proj.get("lead_agency_harmonized"), proj.get("lead_agency"))
    state_display = format_multi_value(proj.get("project_state"))
    energy_display = format_energy_value(proj.get("project_type"))
    category_display = category_display_value(proj.get("project_energy_type"))

    st.title(display_value(proj.get("project_title"), default="(untitled project)"))
    st.caption(f"Project ID: {display_value(proj.get('project_id'))}")

    render_share_link(
        "🔗 Share this page",
        DEEPLINK_QP_KEYS,
        "Anyone with this link opens this exact page.",
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Category", category_display)
    col2.metric("Agency", agency_display)
    col3.metric("State", state_display)

    col4, col5, col6 = st.columns(3)
    col4.metric("Process Type", display_value(proj.get("process_type")))
    col5.metric("Energy Type", energy_display)
    col6.metric("Department", display_value(proj.get("project_department")))

    description = display_value(proj.get("project_description"), default="")
    if description:
        with st.expander("Project Description", expanded=False):
            st.write(description)

    st.divider()
    st.subheader("Documents")

    docs = get_documents(str(DB_PATH), project_id)
    if docs.empty:
        st.warning("No documents found for this project.")
        return

    docs_export = docs.copy()
    docs_export.insert(0, "project_id", project_id)
    st.download_button(
        "Download Project Documents CSV",
        data=docs_export.to_csv(index=False).encode("utf-8"),
        file_name=f"nepa_project_{project_id}_documents.csv",
        mime="text/csv",
        use_container_width=False,
    )

    display_cols = {
        "file_name": "File Name",
        "document_type_clean": "Type",
        "document_type_category": "Category",
        "main_document": "Main Doc",
        "total_pages": "Pages",
        "dataset_source": "Source",
    }
    display_df = docs[list(display_cols.keys())].rename(columns=display_cols)

    event = st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    row_idx = extract_selected_row(event)
    if row_idx is None:
        return

    selected = docs.iloc[row_idx]
    st.session_state["document_id"] = selected["document_id"]
    st.session_state["document_name"] = selected["file_name"]
    st.session_state["current_page"] = 1
    st.session_state["doc_search_term"] = ""
    st.session_state["view"] = "document"
    st.rerun()


def _on_goto_page_change() -> None:
    """Copy the 'Go to page' selectbox value into the current_page source of truth."""
    value = st.session_state.get("goto_page_select")
    if value is not None:
        st.session_state["current_page"] = int(value)


def _on_jump_page_change() -> None:
    """Copy the 'Jump to page' number_input value into the current_page source of truth."""
    value = st.session_state.get("jump_page_input")
    if value is not None:
        st.session_state["current_page"] = int(value)


def _on_text_mode_change() -> None:
    """Copy the 'Text display' radio value into the app-owned text_view_mode."""
    value = st.session_state.get("text_view_mode_widget")
    if value:
        st.session_state["text_view_mode"] = value


def render_document() -> None:
    project_title = st.session_state.get("project_title") or "project"
    if st.button(f"<- Back to {project_title}"):
        st.session_state["view"] = "project"
        st.rerun()

    doc_id = st.session_state.get("document_id")
    if not doc_id:
        st.session_state["view"] = "project"
        st.rerun()

    pages_df = get_page_list(str(DB_PATH), doc_id)
    if pages_df.empty:
        st.warning("No page text found for this document.")
        return

    total_pages = len(pages_df)
    doc_name = st.session_state.get("document_name") or doc_id
    st.markdown(f"**{doc_name}** - {total_pages} pages")

    st.download_button(
        "Download document text (.txt)",
        data=get_document_text(str(DB_PATH), doc_id).encode("utf-8"),
        file_name=sanitize_filename(doc_name),
        mime="text/plain",
        use_container_width=False,
    )

    render_share_link(
        "🔗 Share this page",
        DEEPLINK_QP_KEYS,
        "Anyone with this link opens this exact page.",
    )

    # Smart default text mode per document: CE = Source (forms/tables), EA/EIS =
    # Readable (narrative). Applied only when the rendered document changes, so a
    # manual radio change sticks for that document. Pre-sync the keyed radio's
    # state *before* it is instantiated (same source-of-truth pattern as page nav).
    if st.session_state.get("_text_mode_doc_id") != doc_id:
        st.session_state["_text_mode_doc_id"] = doc_id
        doc_source = get_document_source(str(DB_PATH), doc_id)
        st.session_state["text_view_mode"] = "Source" if doc_source == "CE" else "Readable"

    current_page = int(st.session_state.get("current_page", 1))
    if current_page < 1 or current_page > total_pages:
        current_page = 1
        st.session_state["current_page"] = current_page

    # Ordinal -> printed page label, for the selectbox / match list.
    page_labels: dict[int, str] = {}
    for row in pages_df.itertuples(index=False):
        page_num = int(row.page_number_int) if pd.notna(row.page_number_int) else 0
        page_labels[int(row.page_ordinal)] = str(page_num) if page_num > 0 else str(row.page_number)

    toc_col, reader_col = st.columns([1, 3])

    with toc_col:
        st.subheader("Pages")

        doc_search = st.text_input(
            "Search in document",
            key="doc_search_term",
            placeholder="e.g. wetlands",
        )

        matching_pages = (
            sorted(search_within_document(str(DB_PATH), doc_id, doc_search))
            if doc_search
            else []
        )

        # Jump to the first match only when the search term just changed, so the
        # match-navigation buttons aren't fought on every rerun.
        if doc_search != st.session_state.get("_last_doc_search"):
            st.session_state["_last_doc_search"] = doc_search
            if matching_pages and st.session_state["current_page"] not in matching_pages:
                st.session_state["current_page"] = matching_pages[0]
                st.rerun()

        if doc_search:
            st.caption(f"{len(matching_pages)} page(s) match")
            if matching_pages:
                shown = matching_pages[:200]
                if len(matching_pages) > 200:
                    st.caption("First 200 matching pages shown.")
                for page_idx in shown:
                    button_type = "primary" if page_idx == st.session_state["current_page"] else "secondary"
                    if st.button(
                        f"Page {page_labels.get(page_idx, page_idx)}",
                        key=f"match_{page_idx}",
                        use_container_width=True,
                        type=button_type,
                    ):
                        st.session_state["current_page"] = page_idx
                        st.rerun()
            else:
                st.info("No pages match. Use the reader on the right to browse.")
        else:
            ordinals = [int(o) for o in pages_df["page_ordinal"].tolist()]
            # current_page is the single source of truth: force the keyed widget to
            # follow it *before* instantiation, so Prev/Next/jump navigation sticks
            # instead of being reverted by stale widget state. on_change writes the
            # user's own selection back into current_page.
            if current_page in ordinals:
                st.session_state["goto_page_select"] = current_page
            st.selectbox(
                "Go to page",
                options=ordinals,
                format_func=lambda o: f"Page {page_labels.get(o, o)}",
                key="goto_page_select",
                on_change=_on_goto_page_change,
            )

    with reader_col:
        current_page = int(st.session_state["current_page"])
        current_idx = current_page - 1

        if doc_search and matching_pages:
            prev_matches = [p for p in matching_pages if p < current_page]
            prev_target = prev_matches[-1] if prev_matches else matching_pages[-1]
            next_matches = [p for p in matching_pages if p > current_page]
            next_target = next_matches[0] if next_matches else matching_pages[0]

            match_left, match_center, match_right = st.columns([1, 2, 1])
            with match_left:
                if st.button(
                    "<- Prev match",
                    key="prev_match",
                    disabled=(prev_target == current_page),
                    use_container_width=True,
                ):
                    st.session_state["current_page"] = prev_target
                    st.rerun()
            with match_center:
                if current_page in matching_pages:
                    position = matching_pages.index(current_page) + 1
                    label = f"Match {position} of {len(matching_pages)}"
                else:
                    label = f"{len(matching_pages)} matching pages"
                st.markdown(
                    f"<div style='text-align:center'>{label}</div>",
                    unsafe_allow_html=True,
                )
            with match_right:
                if st.button(
                    "Next match ->",
                    key="next_match",
                    disabled=(next_target == current_page),
                    use_container_width=True,
                ):
                    st.session_state["current_page"] = next_target
                    st.rerun()

        nav_left, nav_center, nav_right = st.columns([1, 2, 1])
        with nav_left:
            if st.button("<- Prev", disabled=(current_idx <= 0)):
                st.session_state["current_page"] = current_page - 1
                st.rerun()

        with nav_center:
            st.markdown(
                f"<div style='text-align:center'>Page {current_page} of {total_pages}</div>",
                unsafe_allow_html=True,
            )

        with nav_right:
            if st.button("Next ->", disabled=(current_idx >= total_pages - 1)):
                st.session_state["current_page"] = current_page + 1
                st.rerun()

        # Same source-of-truth pattern as the selectbox: pre-sync the keyed widget
        # to current_page, then let on_change write user edits back.
        if 1 <= current_page <= total_pages:
            st.session_state["jump_page_input"] = current_page
        st.number_input(
            "Jump to page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key="jump_page_input",
            on_change=_on_jump_page_change,
        )

        # text_view_mode is app-owned (set to the smart default on document change,
        # above). Pre-sync the keyed radio to it each run so a manual change sticks
        # across page navigation instead of snapping back to the widget default.
        st.session_state["text_view_mode_widget"] = st.session_state.get("text_view_mode", "Readable")
        st.radio(
            "Text display",
            options=["Readable", "Source"],
            horizontal=True,
            key="text_view_mode_widget",
            on_change=_on_text_mode_change,
        )
        text_view_mode = st.session_state["text_view_mode"]

        page_row = pages_df.iloc[current_idx]
        source_page = int(page_row["page_number_int"]) if pd.notna(page_row["page_number_int"]) else 0
        source_display = str(source_page) if source_page > 0 else str(page_row["page_number"])

        st.markdown(f"---\n**Page {source_display}**\n\n---")

        raw_text = str(page_row["page_text"] or "")
        prepared = prepare_page_text(raw_text, text_view_mode)
        render_page_text(prepared, st.session_state["doc_search_term"], text_view_mode)


def main() -> None:
    st.set_page_config(
        page_title="NEPA Document Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_state()

    global DB_PATH
    should_download = (not resolve_db_path().exists()) and bool(env_text(HF_DB_REPO_ENV))

    try:
        if should_download:
            with st.spinner(
                "Downloading the 12 GB document database from Hugging Face — "
                "the first load after idle usually takes a few minutes."
            ):
                DB_PATH = ensure_database_available()
        else:
            DB_PATH = ensure_database_available()
    except Exception as exc:
        st.error(f"Database download failed: {exc}")
        st.stop()

    if not DB_PATH.exists():
        st.error(f"Database not found: {DB_PATH}")
        st.markdown("Run the build script locally or configure dataset download:")
        st.code(
            "python app/build_text_store.py\n\n"
            "# Optional remote DB fallback\n"
            "export NEPA_DB_HF_REPO='kaseyzapatka/nepa-document-explorer-db'\n"
            "export NEPA_DB_HF_FILENAME='nepa_reader.duckdb'\n"
            "streamlit run app/app.py"
        )
        st.stop()

    try:
        _ = get_connection(str(DB_PATH))
    except duckdb.Error as exc:
        st.error(f"Could not open DuckDB database: {exc}")
        st.stop()

    # Read query params after the DB is available so deep links can complete
    # their project/document lookups (project_title, document_name).
    apply_query_params_to_state_once()

    sidebar_values = render_sidebar()
    sync_query_params_from_state()
    view = st.session_state["view"]

    if view == "search":
        render_search(*sidebar_values)
    elif view == "project":
        render_project()
    elif view == "document":
        render_document()
    else:
        st.session_state["view"] = "search"
        st.rerun()


if __name__ == "__main__":
    main()
