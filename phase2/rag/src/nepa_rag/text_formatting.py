from __future__ import annotations

import re

BULLET_RE = re.compile(r"^(?:[-*•]\s+|[0-9]+[.)]\s+|[a-z][.)]\s+)")
SECTION_HEADING_RE = re.compile(
    r"^(?:chapter|section|appendix|part)\s+[A-Za-z0-9IVXLC]+",
    re.IGNORECASE,
)
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+){0,3}\s+[A-Z]")
LETTERED_HEADING_RE = re.compile(r"^[A-Z][.)]\s+[A-Z]")
ALL_CAPS_HEADING_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,/&()'\".\-:]{4,}$")
TOC_HEADING_RE = re.compile(r"\bTABLE OF CONTENTS\b", re.IGNORECASE)


def clean_text(text: object) -> str:
    if text is None:
        return ""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ")
    return value.strip()


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.replace("\u00a0", " ")).strip()


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
        return letters > 0 and (uppercase / letters) >= 0.85
    return False


def paragraph_blocks(text: object) -> list[str]:
    """Return paragraph-like blocks while preserving section-heading breaks."""
    raw = clean_text(text)
    if not raw:
        return []

    blocks: list[str] = []
    current: list[str] = []

    def flush() -> None:
        if current:
            merged = " ".join(current)
            merged = re.sub(r"\s+", " ", merged).strip()
            if merged:
                blocks.append(merged)
            current.clear()

    for raw_line in raw.split("\n"):
        line = normalize_line(raw_line)
        if not line:
            flush()
            continue
        if line_is_heading(line):
            flush()
            blocks.append(line)
            continue
        current.append(BULLET_RE.sub("", line, count=1).strip() or line)

    flush()
    return blocks


def estimate_tokens(text: str) -> int:
    # Cheap approximation good enough for chunk budgeting.
    return max(1, int(len(re.findall(r"\S+", text)) / 0.75))
