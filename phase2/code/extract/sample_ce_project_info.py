#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
CE_DIR = ROOT / "data" / "processed" / "ce"
DEFAULT_OUTPUT = ROOT / "output" / "timeline" / "sandbox" / "ce_project_info_sample_50.csv"

PROJECT_INFO_HEADER_RE = re.compile(
    r"(?im)^[ \t]*(?:PART[ \t]*)?(?:[IVX]+|[0-9]+)?[.):]?[ \t]*PROJECT INFORMATION\b[^\n]*"
)

PROJECT_INFO_END_RES = [
    re.compile(r"(?im)^[ \t]*(?:PART[ \t]*)?(?:II|2)[.):]?[ \t]*PLAN CONFORMANCE REVIEW\b"),
    re.compile(
        r"(?im)^[ \t]*(?:PART[ \t]*)?(?:II|2|III|3|IV|4|V|5)[.):]?[ \t]*"
        r"(?:PLAN CONFORMANCE REVIEW|RESOURCE PROGRAM CONSULTATION(?:\s*&\s*COORDINATION)?|"
        r"CATEGORICAL EXCLUSION REVIEW|DECISION|SIGNATURE|EXTRAORDINARY CIRCUMSTANCES)\b"
    ),
    re.compile(r"(?im)^[ \t]*(?:[A-Z]\.)[ \t]*(?:SIGNATURE|CONTACT PERSON)\b"),
    re.compile(r"(?im)^[ \t]*ATTACHMENTS?\b"),
]

COORDINATOR_ANCHOR_RE = re.compile(
    r"(?is)(?:Signature of (?:the )?NEPA Coordinator:|NEPA Coordinator:)"
)

DATE_RE = re.compile(
    r"""
    (?:
        \b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b |
        \b\d{1,2}\.\d{1,2}\.\d{2,4}\b |
        \b\d{4}\.\d{2}\.\d{2}(?:\s+\d{2}:\d{2}:\d{2})?(?:\s*[-+]\d{2}'?\d{2}'?)?\b |
        \b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|
           Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+
           \d{1,2},\s+\d{4}\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

NOISE_LINE_RES = [
    re.compile(r"^\d+$"),
    re.compile(r"^Page \d+ of \d+$", re.IGNORECASE),
    re.compile(r"^DOI-BLM-[A-Z0-9-]+-CX$", re.IGNORECASE),
    re.compile(r"^NEPA COMPLIANCE RECORD$", re.IGNORECASE),
    re.compile(r"^CATEGORICAL EXCLUSION(?: REVIEW)?(?: \(CX\))?$", re.IGNORECASE),
    re.compile(r"^CX NEPA Compliance Record$", re.IGNORECASE),
    re.compile(r"^Bureau of Land Management$", re.IGNORECASE),
    re.compile(r"^Department of the Interior$", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample CE projects and extract PROJECT INFORMATION text and NEPA Coordinator dates."
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def unwrap_struct(value: object) -> object:
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def page_sort_key(page_number: object) -> tuple[int, int, str]:
    text = "" if page_number is None else str(page_number)
    numbers = re.findall(r"\d+", text)
    if not numbers:
        return (10**9, 10**9, text)
    first = int(numbers[0])
    second = int(numbers[1]) if len(numbers) > 1 else 0
    return (first, second, text)


def clean_text_block(text: str) -> str:
    text = text.replace("\r", "\n")
    text = text.replace("\f", "\n")
    lines: list[str] = []

    for raw_line in text.split("\n"):
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if any(pattern.match(line) for pattern in NOISE_LINE_RES):
            continue
        lines.append(line)

    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if line == "":
            if current:
                paragraphs.append(" ".join(current).strip())
                current = []
            continue

        is_list_item = bool(re.match(r"^(?:[-*•]|\[[ xX]\]|\d+[.)])\s+", line))
        if is_list_item and current:
            paragraphs.append(" ".join(current).strip())
            current = [line]
            continue

        current.append(line)

    if current:
        paragraphs.append(" ".join(current).strip())

    cleaned = "\n\n".join(p for p in paragraphs if p)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def extract_project_information(doc_text: str) -> tuple[str | None, str | None]:
    match = PROJECT_INFO_HEADER_RE.search(doc_text)
    if not match:
        return None, None

    start = match.end()
    tail = doc_text[start:]
    end_positions = [m.start() for pattern in PROJECT_INFO_END_RES if (m := pattern.search(tail))]
    end = min(end_positions) if end_positions else len(tail)

    section = clean_text_block(tail[:end])
    if not section:
        return None, match.group(0).strip()
    return section, match.group(0).strip()


def extract_nepa_coordinator_date(doc_text: str) -> tuple[str | None, str | None]:
    first_context: str | None = None

    for match in COORDINATOR_ANCHOR_RE.finditer(doc_text):
        window = doc_text[match.start() : match.start() + 500]
        cleaned_window = clean_text_block(window)
        if first_context is None:
            first_context = cleaned_window

        date_match = DATE_RE.search(window)
        if date_match:
            return date_match.group(0), cleaned_window

    return None, first_context


def load_sample_projects(sample_size: int, seed: int) -> pd.DataFrame:
    projects = pd.read_parquet(CE_DIR / "projects.parquet", columns=["project_id", "project_title"]).copy()
    projects["project_id"] = projects["project_id"].map(unwrap_struct)
    projects = projects.drop_duplicates(subset=["project_id"]).reset_index(drop=True)
    sample_n = min(sample_size, len(projects))
    sampled = projects.sample(n=sample_n, random_state=seed).reset_index(drop=True)
    sampled.insert(0, "sample_rank", range(1, len(sampled) + 1))
    return sampled


def load_docs_and_pages(project_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    ids_sql = ", ".join("'" + pid.replace("'", "''") + "'" for pid in project_ids)

    docs_query = f"""
        SELECT
            project_id.value::VARCHAR AS project_id,
            document_id,
            document_title,
            file_name,
            total_pages,
            main_document
        FROM read_parquet('{(CE_DIR / "documents.parquet").as_posix()}')
        WHERE project_id.value::VARCHAR IN ({ids_sql})
    """
    docs_df = con.execute(docs_query).fetchdf()

    if docs_df.empty:
        return docs_df, pd.DataFrame(columns=["document_id", "page_number", "page_text"])

    doc_ids_sql = ", ".join("'" + doc_id.replace("'", "''") + "'" for doc_id in docs_df["document_id"].tolist())
    pages_query = f"""
        SELECT document_id, page_number, page_text
        FROM read_parquet('{(CE_DIR / "pages.parquet").as_posix()}')
        WHERE document_id IN ({doc_ids_sql})
    """
    pages_df = con.execute(pages_query).fetchdf()
    return docs_df, pages_df


def build_document_texts(pages_df: pd.DataFrame) -> dict[str, str]:
    texts: dict[str, str] = {}
    if pages_df.empty:
        return texts

    for document_id, doc_pages in pages_df.groupby("document_id"):
        ordered = doc_pages.copy()
        ordered["_page_sort_key"] = ordered["page_number"].map(page_sort_key)
        ordered = ordered.sort_values("_page_sort_key")
        page_texts = [text for text in ordered["page_text"].fillna("").tolist() if text]
        texts[document_id] = "\n\n".join(page_texts)
    return texts


def pick_project_documents(docs_df: pd.DataFrame, project_id: str) -> pd.DataFrame:
    project_docs = docs_df[docs_df["project_id"] == project_id].copy()
    if project_docs.empty:
        return project_docs

    project_docs["main_rank"] = (project_docs["main_document"].fillna("").str.upper() == "YES").map(
        lambda is_main: 0 if is_main else 1
    )
    project_docs = project_docs.sort_values(
        by=["main_rank", "total_pages", "document_title", "document_id"],
        ascending=[True, False, True, True],
    )
    return project_docs.drop(columns=["main_rank"])


def build_output(sampled_projects: pd.DataFrame, docs_df: pd.DataFrame, pages_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    doc_texts = build_document_texts(pages_df)
    rows: list[dict[str, object]] = []

    for project in sampled_projects.itertuples(index=False):
        project_docs = pick_project_documents(docs_df, project.project_id)

        project_info = {
            "found": False,
            "document_id": None,
            "document_title": None,
            "main_document": None,
            "header": None,
            "text": None,
        }
        coordinator = {
            "found": False,
            "document_id": None,
            "document_title": None,
            "main_document": None,
            "date": None,
            "context": None,
        }

        for doc in project_docs.itertuples(index=False):
            doc_text = doc_texts.get(doc.document_id)
            if not doc_text:
                continue

            if not project_info["found"]:
                section_text, header_text = extract_project_information(doc_text)
                if section_text:
                    project_info.update(
                        {
                            "found": True,
                            "document_id": doc.document_id,
                            "document_title": doc.document_title,
                            "main_document": doc.main_document,
                            "header": header_text,
                            "text": section_text,
                        }
                    )

            if not coordinator["found"]:
                date_text, context_text = extract_nepa_coordinator_date(doc_text)
                if context_text and coordinator["context"] is None:
                    coordinator["context"] = context_text
                    coordinator["document_id"] = doc.document_id
                    coordinator["document_title"] = doc.document_title
                    coordinator["main_document"] = doc.main_document
                if date_text:
                    coordinator.update(
                        {
                            "found": True,
                            "document_id": doc.document_id,
                            "document_title": doc.document_title,
                            "main_document": doc.main_document,
                            "date": date_text,
                            "context": context_text,
                        }
                    )

            if project_info["found"] and coordinator["found"]:
                break

        rows.append(
            {
                "sample_seed": seed,
                "sample_rank": project.sample_rank,
                "project_id": project.project_id,
                "project_title": project.project_title,
                "project_information_found": project_info["found"],
                "project_information_document_id": project_info["document_id"],
                "project_information_document_title": project_info["document_title"],
                "project_information_main_document": project_info["main_document"],
                "project_information_header": project_info["header"],
                "project_information_text": project_info["text"],
                "nepa_coordinator_date_found": coordinator["found"],
                "nepa_coordinator_document_id": coordinator["document_id"],
                "nepa_coordinator_document_title": coordinator["document_title"],
                "nepa_coordinator_main_document": coordinator["main_document"],
                "nepa_coordinator_date": coordinator["date"],
                "nepa_coordinator_context": coordinator["context"],
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    sampled_projects = load_sample_projects(args.sample_size, args.seed)
    docs_df, pages_df = load_docs_and_pages(sampled_projects["project_id"].tolist())
    output_df = build_output(sampled_projects, docs_df, pages_df, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output, index=False)

    print(f"Wrote {len(output_df)} rows to {args.output}")
    print(f"PROJECT INFORMATION found: {int(output_df['project_information_found'].sum())}")
    print(f"NEPA Coordinator date found: {int(output_df['nepa_coordinator_date_found'].sum())}")
    either = output_df["project_information_found"] | output_df["nepa_coordinator_date_found"]
    print(f"Either field found: {int(either.sum())}")


if __name__ == "__main__":
    main()
