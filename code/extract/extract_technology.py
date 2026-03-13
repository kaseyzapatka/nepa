"""
Technology-specific feature extraction for NEPA projects.

This module centralizes extraction of:
- transmission flags and lengths
- geothermal flags/phases
- pipeline flags and lengths

It can be imported by extract_data.py and also run directly with CLI tags
for one or more technology domains.
"""

from __future__ import annotations

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import argparse
import ast
import json
import re
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


# --------------------------
# CONSTANTS
# --------------------------

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

# Column name prefixes that belong in projects_transmission.parquet, not projects_combined.parquet.
TX_OUTPUT_PREFIXES = (
    "project_is_transmission",
    "project_has_transmission",
    "project_transmission",
    "project_tx_",
)

MILES_RE = re.compile(r"(?<![a-z0-9])(\d{1,4}(?:,\d{3})*(?:\.\d+)?)\s*(mile|miles|mi)\b", re.IGNORECASE)
FEET_RE = re.compile(r"(?<![a-z0-9])(\d{1,6}(?:,\d{3})*(?:\.\d+)?)\s*(foot|feet|ft)\b", re.IGNORECASE)

# Matches width-context words that indicate a feet value is ROW/easement width, not line length.
WIDTH_CONTEXT_RE = re.compile(
    r"\b(wide|width|in\s+width|corridor\s+width|easement\s+width|right.of.way\s+width)\b",
    re.IGNORECASE,
)

# Matches when a mile value is immediately followed by a cardinal direction,
# indicating a geographic distance ("26 miles north of Helena"), not a line length.
# Applied to the text AFTER the miles match.
LOCATION_DIRECTION_RE = re.compile(
    r"^\s*(?:north|south|east|west|northeast|northwest|southeast|southwest)\b",
    re.IGNORECASE,
)

# Matches sentences with explicit total-length language ("11.7 miles long").
# Candidates in such sentences get a +2 hint_score bonus.
TOTAL_LENGTH_RE = re.compile(
    r"\b(?:miles?\s+long|miles?\s+in\s+length|total\s+(?:length|distance)\s+of"
    r"|overall\s+length|would\s+be\s+\d[\d,.]*\s*miles?)\b",
    re.IGNORECASE,
)

# Matches context indicating a mile value is a partial land-crossing extent,
# not the total line length ("cross public lands for 4.7 miles", "1.61 miles on public land").
# Two directional regexes to detect partial land-crossing extent around a miles match.
# Applied separately to text BEFORE and AFTER the match to avoid false positives
# (e.g., "3.14 miles of this powerline, where it crosses federal lands" is NOT a partial crossing
# because the crossing language is not immediately linked to the 3.14 number).
#
# BEFORE match: "cross public lands for [X miles]"
PARTIAL_CROSSING_BEFORE_RE = re.compile(
    r"cross(?:es|ing)?\s+(?:\w+\s+){0,4}(?:public|federal|BLM|National\s+Forest|state|tribal)\s+lands?\s+for\s*$",
    re.IGNORECASE,
)
# AFTER match: "[X miles] on/of/within public land"
PARTIAL_CROSSING_AFTER_RE = re.compile(
    r"^\s*(?:on|of|within|across)\s+(?:public|federal|BLM|state|tribal)\s+lands?\b",
    re.IGNORECASE,
)

# Projects matching this are vegetation management / routine maintenance,
# not infrastructure construction — flagged as project_is_transmission_maintenance.
TRANSMISSION_MAINTENANCE_RE = re.compile(
    r"\b(?:"
    r"vegetation\s+(?:management|inspection|control|removal|clearing|trimming)|"
    r"integrated\s+vegetation\s+management|"
    r"weed\s+(?:control|management)|herbicide(?:\s+(?:treatment|application))?|"
    r"brush\s+(?:clearing|control|management|removal)|"
    r"tree\s+(?:trimming|removal|cutting)|"
    r"right.of.way\s+(?:maintenance|mowing|spraying)|"
    r"routine\s+(?:maintenance|inspection|survey)|"
    r"road\s+maintenance|dust\s+abatement|reclaim(?:ation)?"
    r")\b",
    re.IGNORECASE,
)

# Matches action verbs in a candidate sentence that signal a proposed build action.
CANDIDATE_BUILD_VERB_RE = re.compile(
    r"\b(construct(?:ion|ed)?|build(?:ing)?|install(?:ation|ed)?|upgrade(?:d|s)?"
    r"|rebuild(?:ing)?|proposed?|approv(?:e|ed|al)|authoriz(?:e|ed|ation)|develop(?:ment|ed)?)\b",
    re.IGNORECASE,
)

# ----- Project-level action type classification regexes -----
# Applied to full project text (title + description) to classify the primary action.
# Categories: new_build | upgrade | maintenance | fiber_optic | renewal | acquisition

_TX_ACT_NEW_BUILD_RE = re.compile(
    # Require explicitly "new" infrastructure — avoid matching "construction" alone
    # because renewal/upgrade/acquisition projects also use "permit to construct" boilerplate.
    r"\b(?:new\s+(?:transmission\s+)?line|new\s+double.circuit|new\s+single.circuit"
    r"|new\s+substation|new\s+support\s+structure|new\s+overhead|new\s+underground"
    r"|new\s+segment\s+of\s+(?:line|transmission)|re.alignment\s+of"
    r"|substation\s+construction|construct(?:ion|ing|ed)?\s+of\s+(?:a\s+|the\s+)?new"
    r"|build(?:ing)?\s+(?:a\s+|the\s+)?new"
    r"|switchyard|tap\s+line|tie\s+line|interconnection)\b",
    re.IGNORECASE,
)
_TX_ACT_UPGRADE_RE = re.compile(
    r"\b(?:replac(?:e|es|ed|ing|ement)(?!\s+in\s+kind)|rebuild(?:ing|s|t)?|reconductor(?:ing|ed)?"
    r"|upgrad(?:e|es|ed|ing)|reconstruct(?:ion|ed|ing)?"
    r"|component\s+replacement|structure\s+replacement|hardware\s+replacement"
    r"|crossarm|insulator|shield\s+wire|spacer.damper)\b",
    re.IGNORECASE,
)
# Narrow on purpose: road/vegetation/herbicide are in TRANSMISSION_MAINTENANCE_RE (exclusion filter).
# This only labels on-line structural maintenance in projects that pass the exclusion filter.
_TX_ACT_MAINTENANCE_RE = re.compile(
    r"\b(?:hazard\s+tree|structure\s+inspection|line\s+inspection"
    r"|routine\s+inspection|pole\s+inspection|tower\s+inspection)\b",
    re.IGNORECASE,
)
_TX_ACT_FIBER_OPTIC_RE = re.compile(
    r"\b(?:fiber\s+optic|opgw|optical\s+ground\s+wire|replace\s+ogw"
    r"|overhead\s+fiber|fiber\s+cable|fiber\s+poles|telecom(?:munication)?"
    r"|communication\s+facilit(?:y|ies)|vault)\b",
    re.IGNORECASE,
)
_TX_ACT_RENEWAL_RE = re.compile(
    # Require ROW/authorization context — bare "renewal" fires too broadly on upgrade projects
    r"\b(?:right.of.way\s+(?:grant|renewal|application|amendment)"
    r"|row\s+(?:grant|renewal|application|amendment)"
    r"|re.authoriz(?:e|ation|ing)|re.licens(?:e|ing)"
    r"|short.term\s+row|row\s+expires|issued\s+in\s+\d{4})\b",
    re.IGNORECASE,
)
_TX_ACT_ACQUISITION_RE = re.compile(
    # "easement rights" removed — appears in nearly every transmission project description.
    # Keep only explicit acquisition/transfer actions.
    r"\b(?:acqui(?:re|res|red|ring|sition)|disposition|dispos(?:e|al)"
    r"|convey(?:ance|ed|ing)?|transfer\s+of\s+(?:easement|land|property|ownership))\b",
    re.IGNORECASE,
)

_TX_ACTION_REGEXES: List[Tuple[str, re.Pattern]] = [
    ("new_build",   _TX_ACT_NEW_BUILD_RE),
    ("upgrade",     _TX_ACT_UPGRADE_RE),
    ("maintenance", _TX_ACT_MAINTENANCE_RE),
    ("fiber_optic", _TX_ACT_FIBER_OPTIC_RE),
    ("renewal",     _TX_ACT_RENEWAL_RE),
    ("acquisition", _TX_ACT_ACQUISITION_RE),
]

# Per-candidate regexes (subset) — used only to split mileage by new_build vs upgrade.
# Deliberately broader than the project-level regexes to catch sentence-level cues.
_TX_CAND_NEW_BUILD_RE = re.compile(
    r"\b(?:new\s+(?:transmission\s+)?line|new\s+double.circuit|new\s+single.circuit"
    r"|new\s+substation|switchyard|tap\s+line|tie\s+line|interconnection"
    r"|new\s+overhead|new\s+underground)"
    r"|\bconstruct(?:ion)?\s+of\s+(?:a\s+|the\s+|new\s+)?(?:\d[\d,.]*\s*(?:mile|mi|km)s?\s+(?:of\s+)?)?(?:new\s+)?(?:transmission|power.?line)",
    re.IGNORECASE,
)
_TX_CAND_UPGRADE_RE = re.compile(
    r"\b(?:upgrad(?:e|ed|ing|es)|reconductor(?:ing|ed)?|rebuild(?:ing|s|t)?|rebuilt"
    r"|reconstruct(?:ion|ed|ing)?|replac(?:e|es|ed|ing|ement)|crossarm|insulator|shield\s+wire)\b",
    re.IGNORECASE,
)

TRANSMISSION_HINTS = (
    "transmission",
    "powerline",
    "power line",
    "kV transmission",
    "electric line",
    "line route",
)

PIPELINE_HINTS = (
    "pipeline",
    "pipelines",
    "right-of-way",
    "row",
    "buried line",
    "flowline",
)

TRANSMISSION_BUILD_RE = re.compile(
    r"(?:new\s+transmission\s+line|"
    r"\btransmission\s+line\s+(?:project|route|corridor)\b|"
    # Transmission project/corridor without the word "line" (e.g. "Gateway West Transmission Project")
    r"\btransmission\s+(?:project|corridor|facility)\b|"
    r"\b(?:construct(?:ion|ed)?|build(?:ing)?|install(?:ation|ed)?|upgrade(?:d|s)?|rebuild(?:ing)?)\s+"
    r"(?:of\s+)?(?:new\s+)?(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line\b|"
    r"double-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"single-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"\b\d{2,4}\s*-?\s*k\s?v\s+(?:transmission\s+line|line)\b|"
    # HVDC lines
    r"\bHVDC\b|high.voltage\s+direct\s+current\b|"
    # Generator-tie lines (new-build interconnection from generator to grid)
    r"\bgen.tie\s+(?:line|transmission)\b|\bgenerating\s+tie\s+line\b|"
    # ROW branch: narrowed to require "new" so plain ROW renewals don't pass
    r"right-?of-?way\s+(?:\w+\s+){0,3}new\s+transmission\s+line|"
    r"new\s+transmission\s+line\s+(?:\w+\s+){0,3}right-?of-?way)",
    re.IGNORECASE,
)

# Pipeline new-build gate: construction/project language in title + description (context_text).
# Applied analogously to TRANSMISSION_BUILD_RE but tuned for pipeline vocabulary.
# "pipeline project/route/corridor/segment/lateral" are treated as new-build signals because
# these phrases are far more commonly used for new infrastructure than for operational reviews.
PIPELINE_BUILD_RE = re.compile(
    r"\b(?:"
    r"new\s+(?:natural\s+gas\s+|gas\s+|oil\s+|carbon\s+|co2\s+|hydrogen\s+|water\s+|crude\s+)?pipeline|"
    r"(?:construct(?:ion|ed)?|build(?:ing)?|install(?:ation|ed)?|lay(?:ing)?)\s+"
    r"(?:a\s+|the\s+|new\s+)?(?:gas\s+|oil\s+|carbon\s+|hydrogen\s+|water\s+|crude\s+)?pipeline|"
    r"pipeline\s+(?:project|route|corridor|expansion|extension|segment|lateral|alignment|interconnect(?:ion)?)|"
    r"buried\s+pipeline|"
    r"(?:gathering\s+system|gathering\s+line|flowline)\s+(?:project|construction|installation|expansion)|"
    r"pipeline\s+(?:facility|system)\s+(?:project|construction|development)"
    r")\b",
    re.IGNORECASE,
)

# Pipeline maintenance exclusion gate: applied to title only (same design as
# TRANSMISSION_MAINTENANCE_RE — maintenance language in descriptions is often incidental).
PIPELINE_MAINTENANCE_RE = re.compile(
    r"\b(?:"
    r"pipeline\s+(?:inspection|survey|monitoring)|"
    r"cathodic\s+protection|"
    r"in-?line\s+inspection|internal\s+inspection|"
    r"pigging|"
    r"pipeline\s+(?:repair|maintenance|replacement)|"
    r"right.of.way\s+(?:maintenance|mowing|spraying|herbicide)|"
    r"routine\s+(?:maintenance|inspection)|"
    r"annual\s+(?:maintenance|inspection|survey)|"
    r"leak\s+(?:detection|survey|repair)|"
    r"(?:recoating|coating|lining)\s+(?:of\s+)?(?:the\s+)?pipeline|"
    r"emergency\s+repair|"
    r"pipeline\s+safety\s+(?:program|rule|regulation|compliance)|"
    r"integrity\s+management\s+(?:plan|program)"
    r")\b",
    re.IGNORECASE,
)

TRANSMISSION_ALTERNATIVE_RE = re.compile(
    r"\b("
    r"alternative(?:s)?|"
    r"route\s+alternative(?:s)?|"
    r"alignment\s+alternative(?:s)?|"
    r"option\s+[a-z0-9]+|"
    r"either\s+route|"
    r"one\s+of\s+(?:two|three|several)\s+routes"
    r")\b",
    re.IGNORECASE,
)

TRANSMISSION_ADDITIVE_RE = re.compile(
    r"\b("
    r"also\s+included|"
    r"in\s+addition|"
    r"along\s+with|"
    r"plus|"
    r"combined|"
    r"new\s+build\s+section|"
    r"upgrade\s+section|"
    r"segment(?:s)?|"
    r"phase(?:s)?|"
    r"lateral(?:s)?|"
    r"spur(?:s)?|"
    r"tap(?:s)?"
    r")\b",
    re.IGNORECASE,
)

GEOTHERMAL_PHASE_PATTERNS = {
    "exploration": [
        r"\bexploration\b",
        r"\bexploratory\b",
        r"\bresource assessment\b",
        r"\bgeophysical survey\b",
        r"\btemperature gradient\b",
        r"\bseismic survey\b",
        r"\bgravity survey\b",
        r"\bmagnetic survey\b",
        r"\btemperature probe\b",
        r"\btest hole\b",
        r"\bslim.?hole\b",
        r"\bcore hole\b",
        r"\bresource characterization\b",
        r"\bgeoelectrical\b",
        r"\bfeasibility study\b",
        r"\bpre.?feasibility\b",
        r"\bgeothermal prospecting\b",
    ],
    "drilling": [
        r"\bdrilling\b",
        r"\bdrill pad\b",
        r"\bwell pad\b",
        r"\bproduction well\b",
        r"\binjection well\b",
        r"\bwell stimulation\b",
        r"\bwell field\b",
        r"\bwell program\b",
        r"\bgeothermal well\b",
        r"\bsteam well\b",
        r"\btest well\b",
        r"\bhydrothermal well\b",
        r"\bwell construction\b",
        r"\bwellhead\b",
        r"\bwellfield\b",
        r"\bhydraulic stimulation\b",
        r"\breservoir stimulation\b",
        r"\bpermit to drill\b",
        r"\bwell permit\b",
        r"\bnotice of intent to drill\b",
        r"\bwell abandonment\b",
        r"\bwell plugging\b",
        r"\bwell completion\b",
    ],
    "plant": [
        r"\bpower plant\b",
        r"\bgenerating station\b",
        r"\bsteam plant\b",
        r"\bbinary plant\b",
        r"\bflash plant\b",
        r"\bturbine\b",
        r"\binterconnection\b",
        r"\bpower generation\b",
        r"\belectric generation\b",
        r"\bgenerating facility\b",
        r"\bpower facility\b",
        r"\bgenerator\b",
        r"\bsubstation\b",
        r"\btransmission line\b",
        r"\bsteam gathering\b",
        r"\bpipeline system\b",
        r"\bcondenser\b",
        r"\bcooling tower\b",
        r"\bbinary cycle\b",
    ],
    "operations": [
        r"\bsteam supply\b",
        r"\bfluid management\b",
        r"\bmake-up well\b",
        r"\bmakeup well\b",
        r"\breinjection\b",
        r"\bworking fluid\b",
        r"\bgeothermal resource utilization\b",
    ],
}

# Labels and mappings for the ML phase classifier
GEO_PHASE_LABELS: List[str] = ["exploration", "drilling", "plant", "operations", "multi_phase"]
GEO_PHASE_LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(GEO_PHASE_LABELS)}
GEO_PHASE_ID2LABEL: Dict[int, str] = {i: l for i, l in enumerate(GEO_PHASE_LABELS)}
GEO_PHASE_DEFAULT_BASE_MODEL = "allenai/scibert_scivocab_uncased"
GEO_PHASE_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models" / "geothermal_phase_classifier"


# --------------------------
# TYPES
# --------------------------

@dataclass
class LengthAdjudication:
    selected_length_miles: float       # Final answer: LLM result if used, else rule-based
    rule_based_length_miles: float     # Always rule-based (for comparison with LLM)
    confidence: str
    source_text: str
    taxonomy: str
    selection_method: str
    selected_candidate_ids: List[str]
    candidate_count: int
    distinct_candidate_count: int
    llm_trigger: bool
    llm_used: bool
    llm_status: str
    llm_reasoning: str
    llm_model: str      # Model name used (e.g. CLAUDE_DEFAULT_MODEL) or "" if LLM not used
    llm_run_at: str     # ISO-8601 UTC timestamp when Claude returned, or "" if not used


# --------------------------
# HELPERS
# --------------------------

def _value_to_text(value) -> str:
    """Convert list/JSON/scalar values to a normalized plain-text string."""
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(str(v) for v in value if str(v).strip())
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return ""
        if v.startswith("[") and v.endswith("]"):
            for parser in (json.loads, ast.literal_eval):
                try:
                    parsed = parser(v)
                    if isinstance(parsed, (list, tuple)):
                        return " ".join(str(x) for x in parsed if str(x).strip())
                except Exception:
                    pass
        return v
    return str(value)


def _series_text(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series("", index=df.index)
    return df[col].apply(_value_to_text)


def _match_snippet(text: str, start: int, end: int, max_len: int = 500) -> str:
    """
    Build a bounded snippet centered on the matched token span so the extracted
    numeric mention is always visible in QA fields.
    """
    if not text:
        return ""
    if len(text) <= max_len:
        return text

    center = (start + end) // 2
    half = max_len // 2
    left = max(0, center - half)
    right = min(len(text), left + max_len)
    left = max(0, right - max_len)
    snippet = text[left:right]

    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def _extract_length_candidates(text: str, hints: Sequence[str], prefix: str) -> List[Dict]:
    """
    Extract all candidate linear lengths from text with source sentences.

    Returns candidate dicts that include sentence-level evidence for QA.
    """
    if not text:
        return []

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    sentences = re.split(r"(?<=[\.\?!;])\s+", normalized)
    candidates: List[Dict] = []
    cid = 1

    for s_idx, sent in enumerate(sentences):
        s_lower = sent.lower()
        if "mile post" in s_lower or "mp " in s_lower:
            continue

        matched_hints = [h for h in hints if h.lower() in s_lower]
        hint_score = len(matched_hints)
        if hint_score == 0:
            continue

        sent_has_build = bool(CANDIDATE_BUILD_VERB_RE.search(sent))

        sent_is_total_length = bool(TOTAL_LENGTH_RE.search(sent))

        for m in MILES_RE.finditer(sent):
            raw = m.group(1)
            val_mi = float(raw.replace(",", ""))
            if 0 < val_mi <= 5000:
                # Skip if the miles value is a geographic distance reference
                # (e.g., "26 miles north of Helena" = location, not line length).
                after_40 = sent[m.end(): m.end() + 40]
                if LOCATION_DIRECTION_RE.match(after_40):
                    continue

                match_start, match_end = m.span()

                # Detect partial land-crossing context linked to this match.
                # Check text before for "crosses public lands for [X miles]"
                # and text after for "[X miles] on/of/within public land".
                before_80 = sent[max(0, m.start() - 80): m.start()]
                after_80 = sent[m.end(): min(len(sent), m.end() + 80)]
                is_partial_crossing = (
                    bool(PARTIAL_CROSSING_BEFORE_RE.search(before_80))
                    or bool(PARTIAL_CROSSING_AFTER_RE.match(after_80))
                )

                # Bonus for sentences with explicit total-length language ("X miles long").
                total_bonus = 2 if sent_is_total_length else 0

                candidates.append(
                    {
                        "candidate_id": f"{prefix}_{cid:04d}",
                        "sentence_index": s_idx,
                        "value_miles": round(val_mi, 3),
                        "raw_value": raw,
                        "matched_text": m.group(0),
                        "raw_unit": m.group(2).lower(),
                        "unit_normalized": "miles",
                        "hint_score": hint_score + 2 + total_bonus,
                        "hint_terms": matched_hints,
                        "sentence_has_build_verb": sent_has_build,
                        "is_partial_crossing": is_partial_crossing,
                        "source_text": _match_snippet(sent, match_start, match_end, max_len=500),
                    }
                )
                cid += 1

        for m in FEET_RE.finditer(sent):
            raw = m.group(1)
            val_ft = float(raw.replace(",", ""))
            val_mi = val_ft / 5280.0
            if 0 < val_mi <= 5000:
                # Skip if sentence context indicates this is a width (ROW width, pole spacing, etc.)
                if WIDTH_CONTEXT_RE.search(sent):
                    continue
                match_start, match_end = m.span()
                candidates.append(
                    {
                        "candidate_id": f"{prefix}_{cid:04d}",
                        "sentence_index": s_idx,
                        "value_miles": round(val_mi, 3),
                        "raw_value": raw,
                        "matched_text": m.group(0),
                        "raw_unit": m.group(2).lower(),
                        "unit_normalized": "miles_from_feet",
                        "hint_score": hint_score + 1,
                        "hint_terms": matched_hints,
                        "sentence_has_build_verb": sent_has_build,
                        "is_partial_crossing": False,
                        "source_text": _match_snippet(sent, match_start, match_end, max_len=500),
                    }
                )
                cid += 1

    return candidates


def _collapse_candidates_by_value(candidates: List[Dict], tol: float = 0.01) -> List[Dict]:
    """Group near-equal values to avoid treating unit-conversion duplicates as distinct."""
    if not candidates:
        return []

    groups: List[Dict] = []
    for c in sorted(candidates, key=lambda x: x["value_miles"]):
        placed = False
        for g in groups:
            if abs(c["value_miles"] - g["value_miles"]) <= tol:
                g["members"].append(c)
                best = g["best_candidate"]
                if (c["hint_score"], c["value_miles"]) > (best["hint_score"], best["value_miles"]):
                    g["best_candidate"] = c
                placed = True
                break
        if not placed:
            groups.append(
                {
                    "value_miles": c["value_miles"],
                    "members": [c],
                    "best_candidate": c,
                }
            )

    # Snap group value to best candidate value for consistency.
    for g in groups:
        g["value_miles"] = g["best_candidate"]["value_miles"]

    return groups


def _best_single_candidate(candidates: List[Dict]) -> Tuple[float, str, str]:
    """Legacy-compatible best single length choice."""
    if not candidates:
        return np.nan, "none", ""

    best = sorted(candidates, key=lambda x: (x["hint_score"], x["value_miles"]), reverse=True)[0]
    confidence = "high" if best["hint_score"] >= 4 else "medium"
    return best["value_miles"], confidence, best["source_text"]


def _call_claude_api(prompt: str, model: str, timeout: int, max_retries: int = 3) -> Dict:
    """
    Call Claude via the Anthropic messages API (same pattern as extract_timeline.py).
    Returns a dict with 'response' (str) and 'error' (str|None).
    """
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"response": "", "error": "ANTHROPIC_API_KEY not set"}

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                CLAUDE_API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 200,
                    "temperature": 0.1,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=timeout,
            )
            if resp.status_code == 200:
                text = resp.json().get("content", [{}])[0].get("text", "")
                return {"response": text, "error": None}
            elif resp.status_code == 429:
                retry_after = int(resp.headers.get("retry-after", 30))
                _time.sleep(retry_after)
                continue
            else:
                msg = resp.json().get("error", {}).get("message", resp.text[:200])
                return {"response": "", "error": f"claude_api_error ({resp.status_code}): {msg}"}
        except requests.exceptions.Timeout:
            return {"response": "", "error": "claude_timeout"}
        except Exception as e:
            return {"response": "", "error": f"claude_error: {e}"}

    return {"response": "", "error": "claude_rate_limit_exhausted"}


def _run_llm_transmission_adjudication(
    candidates: List[Dict],
    timeout: int = 120,
) -> Dict | None:
    """
    Adjudicate among competing transmission line length candidates using Claude API.

    Returns a result dict on success, or None so the caller falls back to rule-based logic.
    Includes 'reasoning' field so the LLM's logic can be audited.
    """
    nontrivial = [c for c in candidates if c["value_miles"] >= 0.25]
    if not nontrivial:
        return None

    cand_lines_parts = []
    for i, c in enumerate(nontrivial[:8]):
        label = " [PARTIAL CROSSING — not total length]" if c.get("is_partial_crossing") else ""
        cand_lines_parts.append(
            f"[{i + 1}] {c['value_miles']:.2f} miles{label} — \"{c['source_text'][:300]}\""
        )
    cand_lines = "\n".join(cand_lines_parts)

    prompt = (
        "NEPA transmission line review. Pick the ONE candidate = total length of the proposed line.\n\n"
        f"Candidates:\n{cand_lines}\n\n"
        "Rules:\n"
        "1. PREFER candidates whose sentence says 'X miles long' or 'X miles in length' — these are explicit total lengths.\n"
        "2. IGNORE candidates labeled [PARTIAL CROSSING] — these measure how far the line crosses a land type, not the total length.\n"
        "3. 'X miles north/south/east/west of [place]' = geographic location, NOT line length — skip it.\n"
        "4. Prefer the length of the line being built/upgraded/installed, not existing reference lines.\n"
        "5. If segments clearly add up to a stated total, use the total.\n\n"
        "Return ONLY valid JSON with these fields:\n"
        "{\"selected_index\": <1-based int>, \"selected_length_miles\": <number>, "
        "\"confidence\": \"high|medium|low\", \"reasoning\": \"<one sentence explanation>\"}\n\nJSON:"
    )

    # --- call Claude API ---
    result = _call_claude_api(prompt, model=CLAUDE_DEFAULT_MODEL, timeout=timeout)
    if result.get("error"):
        print(f"  [Claude error] {result['error']}")
        return None
    raw = result["response"]

    # --- parse JSON response (bracket-matching to handle reasoning w/ braces) ---
    try:
        parsed = json.loads(raw.strip())
    except (json.JSONDecodeError, ValueError):
        start = raw.find("{")
        if start == -1:
            print(f"  [LLM parse] No JSON in response: {raw[:120]!r}")
            return None
        depth = 0
        end = -1
        for k, ch in enumerate(raw[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = k + 1
                    break
        if end == -1:
            print(f"  [LLM parse] Unmatched braces: {raw[:120]!r}")
            return None
        try:
            parsed = json.loads(raw[start:end])
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  [LLM parse] JSON error: {exc} | raw: {raw[:120]!r}")
            return None

    try:
        idx = int(parsed.get("selected_index", 0)) - 1
        if 0 <= idx < len(nontrivial):
            chosen = nontrivial[idx]
        else:
            stated = float(parsed.get("selected_length_miles", 0))
            if stated <= 0:
                return None
            chosen = min(nontrivial, key=lambda c: abs(c["value_miles"] - stated))
        return {
            "selected_length_miles": chosen["value_miles"],
            "confidence": str(parsed.get("confidence", "medium")),
            "source_text": chosen["source_text"],
            "selected_candidate_ids": [chosen["candidate_id"]],
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception as exc:
        print(f"  [LLM parse] Result extraction failed: {exc}")
        return None


def _rule_based_length_selection(
    groups: List[Dict],
    has_alternative: bool,
    has_additive: bool,
) -> Dict:
    """
    Pure rule-based selection from collapsed candidate groups.
    Always computed — used both as fallback and as the comparison baseline
    when the LLM overrides the selection.
    Returns a dict with: selected_length_miles, confidence, source_text,
    taxonomy, selection_method, selected_candidate_ids.
    """
    distinct_count = len(groups)

    if distinct_count <= 1:
        best = groups[0]["best_candidate"]
        confidence = "high" if best["hint_score"] >= 4 else "medium"
        return dict(
            selected_length_miles=best["value_miles"],
            confidence=confidence,
            source_text=best["source_text"],
            taxonomy="do_not_sum",
            selection_method="rule",
            selected_candidate_ids=[best["candidate_id"]],
        )

    if has_alternative:
        chosen = sorted(
            (g["best_candidate"] for g in groups),
            key=lambda x: (x["value_miles"], x["hint_score"]),
            reverse=True,
        )[0]
        confidence = "high" if chosen["hint_score"] >= 4 else "medium"
        return dict(
            selected_length_miles=chosen["value_miles"],
            confidence=confidence,
            source_text=chosen["source_text"],
            taxonomy="choose_alternative",
            selection_method="rule",
            selected_candidate_ids=[chosen["candidate_id"]],
        )

    if has_additive:
        selected = sorted(
            [g["best_candidate"] for g in groups],
            key=lambda x: (x["value_miles"], x["hint_score"]),
            reverse=True,
        )
        value = round(sum(c["value_miles"] for c in selected), 3)
        source = " || ".join(c["source_text"] for c in selected)[:2000]
        confidence = "high" if len(selected) >= 2 else "medium"
        return dict(
            selected_length_miles=value,
            confidence=confidence,
            source_text=source,
            taxonomy="sum",
            selection_method="rule",
            selected_candidate_ids=[c["candidate_id"] for c in selected],
        )

    # Ambiguous multi-candidate: prefer non-partial candidates, then build verb or take_max.
    effective_groups = (
        [g for g in groups if not g["best_candidate"].get("is_partial_crossing", False)]
        or groups
    )
    build_candidates = [
        g["best_candidate"] for g in effective_groups
        if g["best_candidate"].get("sentence_has_build_verb", False)
    ]
    if len(build_candidates) == 1:
        chosen = build_candidates[0]
        taxonomy = "build_verb_winner"
    else:
        chosen = sorted(
            (g["best_candidate"] for g in effective_groups),
            key=lambda x: (x["hint_score"], x["value_miles"]),
            reverse=True,
        )[0]
        taxonomy = "take_max"

    confidence = "high" if chosen["hint_score"] >= 4 else "medium"
    return dict(
        selected_length_miles=chosen["value_miles"],
        confidence=confidence,
        source_text=chosen["source_text"],
        taxonomy=taxonomy,
        selection_method="rule",
        selected_candidate_ids=[chosen["candidate_id"]],
    )


def _adjudicate_transmission_length(
    full_text: str,
    candidates: List[Dict],
    use_llm: bool = False,
    timeout: int = 120,
) -> LengthAdjudication:
    groups = _collapse_candidates_by_value(candidates)
    candidate_count = len(candidates)
    distinct_count = len(groups)

    nontrivial = [g for g in groups if g["value_miles"] >= 0.25]
    non_partial_nontrivial = [
        g for g in nontrivial
        if not g["best_candidate"].get("is_partial_crossing", False)
    ]
    effective_nontrivial = non_partial_nontrivial if non_partial_nontrivial else nontrivial

    llm_trigger = len(effective_nontrivial) >= 2

    if candidate_count == 0:
        return LengthAdjudication(
            selected_length_miles=np.nan,
            rule_based_length_miles=np.nan,
            confidence="none",
            source_text="",
            taxonomy="none",
            selection_method="none",
            selected_candidate_ids=[],
            candidate_count=0,
            distinct_candidate_count=0,
            llm_trigger=False,
            llm_used=False,
            llm_status="not_triggered",
            llm_reasoning="",
            llm_model="",
            llm_run_at="",
        )

    text_lower = (full_text or "").lower()
    has_alternative = bool(TRANSMISSION_ALTERNATIVE_RE.search(text_lower))
    has_additive = bool(TRANSMISSION_ADDITIVE_RE.search(text_lower))

    # Always compute rule-based selection first — stored in project_transmission_length_miles
    # as the comparison baseline regardless of whether the LLM also runs.
    rule = _rule_based_length_selection(groups, has_alternative, has_additive)

    llm_used = False
    llm_status = "not_requested" if llm_trigger else "not_triggered"
    llm_run_at = ""
    llm_result = None
    if llm_trigger and use_llm:
        try:
            llm_result = _run_llm_transmission_adjudication(
                candidates, timeout=timeout
            )
            if llm_result:
                llm_used = True
                llm_status = "success"
                llm_run_at = datetime.now(timezone.utc).isoformat()
            else:
                llm_status = "failed_fallback_rule"
        except Exception:
            llm_used = False
            llm_status = "failed_fallback_rule"

    if llm_result:
        # LLM succeeded: selected_length_miles = LLM answer,
        # rule_based_length_miles = what rules would have chosen (for comparison).
        return LengthAdjudication(
            selected_length_miles=llm_result["selected_length_miles"],
            rule_based_length_miles=rule["selected_length_miles"],
            confidence=llm_result["confidence"],
            source_text=llm_result["source_text"],
            taxonomy="llm",
            selection_method="llm",
            selected_candidate_ids=llm_result["selected_candidate_ids"],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
            llm_reasoning=llm_result.get("reasoning", ""),
            llm_model=CLAUDE_DEFAULT_MODEL,
            llm_run_at=llm_run_at,
        )

    # No LLM: both columns are the rule-based result.
    return LengthAdjudication(
        selected_length_miles=rule["selected_length_miles"],
        rule_based_length_miles=rule["selected_length_miles"],
        confidence=rule["confidence"],
        source_text=rule["source_text"],
        taxonomy=rule["taxonomy"],
        selection_method=rule["selection_method"],
        selected_candidate_ids=rule["selected_candidate_ids"],
        candidate_count=candidate_count,
        distinct_candidate_count=distinct_count,
        llm_trigger=llm_trigger,
        llm_used=llm_used,
        llm_status=llm_status,
        llm_reasoning="",
        llm_model="",
        llm_run_at=llm_run_at,
    )


def _classify_project_transmission_action(text: str) -> str:
    """
    Classify the primary action type from full project title+description text.

    Returns: new_build | upgrade | maintenance | fiber_optic | renewal | acquisition | mixed | unknown
    Applied at the project level; stored in project_transmission_action.
    """
    hits = [label for label, rx in _TX_ACTION_REGEXES if rx.search(text or "")]
    if len(hits) == 0:  return "unknown"
    if len(hits) == 1:  return hits[0]
    return "mixed"


def _classify_candidate_action(sentence: str) -> str:
    """
    Classify the action type from a single candidate sentence (new_build or upgrade only).
    Used to split mileage into project_transmission_new_build_miles / upgrade_miles.
    """
    is_new = bool(_TX_CAND_NEW_BUILD_RE.search(sentence))
    is_upg = bool(_TX_CAND_UPGRADE_RE.search(sentence))
    if is_new and is_upg:   return "mixed"
    if is_new:              return "new_build"
    if is_upg:              return "upgrade"
    return "unknown"


def _miles_by_action(candidates: List[Dict], action: str) -> float:
    """
    Sum distinct length values (>= 0.25 mi) for a given action type.

    Uses the same near-equality collapse as adjudication to avoid double-counting
    repeated mentions of the same length across sentences.
    """
    relevant = [c for c in candidates if c.get("candidate_action_type") == action and c.get("value_miles", 0) >= 0.25]
    if not relevant:
        return np.nan
    groups = _collapse_candidates_by_value(relevant)
    return round(sum(g["value_miles"] for g in groups), 3)


def _classify_geothermal_phase(text: str) -> tuple:
    """Return (phase, matched_phases).

    phase is one of: none, exploration, drilling, plant, operations,
    multi_phase, unknown.

    matched_phases is the list of phase keys whose pattern set fired.
    Empty for 'none' and 'unknown'; a single-element list for single-phase
    matches; two or more elements when phase == 'multi_phase'.  This lets
    callers decompose multi_phase rows into their constituent phases.
    """
    txt = (text or "").lower()
    geo_keyword = re.search(r"\b(geothermal|enhanced geothermal|egs)\b", txt)
    if not geo_keyword:
        return "none", []

    matches = []
    for phase, patterns in GEOTHERMAL_PHASE_PATTERNS.items():
        if any(re.search(p, txt, flags=re.IGNORECASE) for p in patterns):
            matches.append(phase)

    if len(matches) == 0:
        return "unknown", []
    if len(matches) == 1:
        return matches[0], matches
    return "multi_phase", matches


# --------------------------
# PAGE-LEVEL LENGTH RECOVERY
# --------------------------

def _extract_tx_length_from_pages(
    out: pd.DataFrame,
    processed_dir: Path,
    max_ea_eis_pages: int = 10,
    use_llm: bool = False,
    timeout: int = 120,
    workers: int = 1,
) -> pd.DataFrame:
    """
    Recover transmission line lengths for projects that passed the build-text gate
    but have no extractable mileage in title/description text.

    Uses DuckDB to efficiently join documents and pages parquet files, restricting
    the search to only the projects that need recovery. For CE projects all pages
    are read (each CE document is a single page blob). For EA/EIS only the first
    `max_ea_eis_pages` pages of main documents are searched — the Proposed Action
    and Project Description sections almost always appear in the opening pages.

    Recovered lengths are written back into the standard transmission length columns.
    A new boolean column `project_transmission_length_from_pages` is set True for
    any project whose length was recovered by this step.
    """
    try:
        import duckdb
    except ImportError:
        print("  [page-recovery] duckdb not installed — skipping. Install with: pip install duckdb")
        out["project_transmission_length_from_pages"] = False
        return out

    needs_mask = (
        out["project_has_transmission_type_tag"].fillna(False)
        & out["project_has_transmission_build_text"].fillna(False)
        & ~out["project_is_transmission_maintenance"].fillna(False)
        & (out["project_transmission_length_final"].isna() | (out["project_transmission_length_final"] < 1.0))
    )
    n_needs = int(needs_mask.sum())

    out["project_transmission_length_from_pages"] = False

    if n_needs == 0:
        print("  [page-recovery] No projects need length recovery — skipping.")
        return out

    print(f"  [page-recovery] {n_needs} projects need page-level length recovery")

    needs_df = out.loc[needs_mask, ["project_id", "process_type"]].copy()
    needs_df["pid_clean"] = needs_df["project_id"].astype(str).str.replace("-", "", regex=False)

    con = duckdb.connect()
    recovered: Dict[str, LengthAdjudication] = {}
    recovered_candidates: Dict[str, List[Dict]] = {}

    for ptype in ["CE", "EA", "EIS"]:
        ptype_lower = ptype.lower()
        ptype_df = needs_df[needs_df["process_type"] == ptype]
        if ptype_df.empty:
            continue

        docs_path = str(processed_dir / ptype_lower / "documents.parquet")
        pages_path = str(processed_dir / ptype_lower / "pages.parquet")

        if not Path(docs_path).exists() or not Path(pages_path).exists():
            print(f"  [page-recovery] {ptype}: parquet not found at {docs_path} — skipping")
            continue

        target_ids = list(ptype_df["pid_clean"].unique())
        print(f"  [page-recovery] {ptype}: querying {len(target_ids)} projects")

        # Register target IDs as an in-memory table for the IN-clause join.
        con.register("_target_ids", pd.DataFrame({"pid": target_ids}))

        try:
            if ptype == "CE":
                # CE documents are single-page blobs — no page-count filter needed.
                query = """
                    SELECT
                        replace(d.project_id.value, '-', '') AS pid,
                        p.page_text
                    FROM read_parquet(?) d
                    JOIN read_parquet(?) p ON p.document_id = d.document_id
                    WHERE replace(d.project_id.value, '-', '') IN (SELECT pid FROM _target_ids)
                """
                page_df = con.execute(query, [docs_path, pages_path]).df()
            else:
                # EA/EIS: restrict to main documents and the first N pages.
                # ROW_NUMBER orders by page_number string; lexicographic ordering is
                # sufficient to capture the opening sections where line length appears.
                query = """
                    SELECT pid, page_text FROM (
                        SELECT
                            replace(d.project_id.value, '-', '') AS pid,
                            p.page_text,
                            ROW_NUMBER() OVER (
                                PARTITION BY d.document_id
                                ORDER BY p.page_number
                            ) AS rn
                        FROM read_parquet(?) d
                        JOIN read_parquet(?) p ON p.document_id = d.document_id
                        WHERE replace(d.project_id.value, '-', '') IN (SELECT pid FROM _target_ids)
                          AND d.main_document = 'YES'
                    )
                    WHERE rn <= ?
                """
                page_df = con.execute(query, [docs_path, pages_path, max_ea_eis_pages]).df()
        except Exception as exc:
            print(f"  [page-recovery] {ptype}: DuckDB error — {exc}")
            continue

        if page_df.empty:
            print(f"  [page-recovery] {ptype}: no pages matched")
            continue

        # Concatenate all recovered pages per project into one text blob.
        project_texts = (
            page_df.groupby("pid")["page_text"]
            .apply(lambda parts: " ".join(str(p) for p in parts if p))
            .to_dict()
        )

        print(f"  [page-recovery] {ptype}: extracting from {len(project_texts)} projects")

        def _recover_one(args):
            pid, text = args
            cands = _extract_length_candidates(text, TRANSMISSION_HINTS, prefix="tx")
            for c in cands:
                c["candidate_action_type"] = _classify_candidate_action(c["source_text"])
            adj = _adjudicate_transmission_length(
                text, cands, use_llm=use_llm, timeout=timeout
            )
            return pid, cands, adj

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_recover_one, item): item[0] for item in project_texts.items()}
            for future in as_completed(futures):
                pid, cands, adj = future.result()
                if not np.isnan(adj.selected_length_miles) and adj.selected_length_miles >= 1.0:
                    recovered[pid] = adj
                    recovered_candidates[pid] = cands

    con.close()

    if not recovered:
        print("  [page-recovery] No lengths >= 1 mi recovered from document pages.")
        return out

    print(f"  [page-recovery] Recovered lengths for {len(recovered)} projects — writing back")

    # Build index -> pid_clean map for the writeback loop.
    orig_to_pid = dict(zip(needs_df.index, needs_df["pid_clean"]))

    for idx in out.loc[needs_mask].index:
        pid = orig_to_pid.get(idx)
        if pid is None or pid not in recovered:
            continue

        adj = recovered[pid]
        cands = recovered_candidates[pid]

        out.at[idx, "project_transmission_length_final"] = adj.selected_length_miles
        out.at[idx, "project_transmission_length_miles"] = adj.rule_based_length_miles
        out.at[idx, "project_transmission_length_confidence"] = adj.confidence
        out.at[idx, "project_transmission_length_source_text"] = adj.source_text
        out.at[idx, "project_transmission_length_taxonomy"] = adj.taxonomy
        out.at[idx, "project_transmission_length_selection_method"] = adj.selection_method
        out.at[idx, "project_transmission_length_candidate_count"] = adj.candidate_count
        out.at[idx, "project_transmission_length_distinct_candidate_count"] = adj.distinct_candidate_count
        out.at[idx, "project_transmission_length_candidates_json"] = json.dumps(cands, ensure_ascii=False)
        out.at[idx, "project_transmission_length_selected_candidate_ids"] = json.dumps(adj.selected_candidate_ids)
        out.at[idx, "project_transmission_length_llm_trigger"] = adj.llm_trigger
        out.at[idx, "project_transmission_length_llm_used"] = adj.llm_used
        out.at[idx, "project_transmission_length_llm_status"] = adj.llm_status
        out.at[idx, "project_transmission_length_llm_reasoning"] = adj.llm_reasoning
        out.at[idx, "project_transmission_length_llm_model"] = adj.llm_model
        out.at[idx, "project_tx_llm_run_at"] = adj.llm_run_at
        out.at[idx, "project_transmission_new_build_miles"] = _miles_by_action(cands, "new_build")
        out.at[idx, "project_transmission_upgrade_miles"] = _miles_by_action(cands, "upgrade")
        out.at[idx, "project_transmission_length_from_pages"] = True

    return out


# --------------------------
# DOMAIN EXTRACTORS
# --------------------------

def _add_transmission_columns(
    df: pd.DataFrame,
    full_text: pd.Series,
    context_text: pd.Series,
    type_text: pd.Series,
    title_text: pd.Series,
    use_llm: bool = False,
    timeout: int = 120,
    workers: int = 1,
    processed_dir: Path | None = None,
    max_ea_eis_pages: int = 10,
) -> pd.DataFrame:
    extraction_run_at = datetime.now(timezone.utc).isoformat()
    out = df.copy()
    lower_text = full_text.str.lower()
    lower_context = context_text.str.lower()

    out["project_is_transmission_broad"] = (
        lower_text.str.contains(r"\belectricity transmission\b", regex=True)
        | lower_text.str.contains(r"\btransmission line\b", regex=True)
        | lower_text.str.contains(r"\btransmission\b", regex=True)
    )

    out["project_has_transmission_type_tag"] = type_text.str.lower().str.contains(
        r"\belectricity transmission\b", regex=True
    )
    out["project_has_transmission_build_text"] = lower_context.str.contains(TRANSMISSION_BUILD_RE)

    # Flag maintenance projects early — before expensive extraction — so they are
    # excluded from candidate extraction, adjudication, and the strict definition.
    # Title-only check: maintenance language in the description is often incidental
    # (e.g., "access road maintenance" in a pole-replacement project). Checking only
    # the title avoids excluding genuine transmission projects where maintenance work
    # is a minor supporting activity, not the primary purpose.
    out["project_is_transmission_maintenance"] = title_text.apply(
        lambda x: bool(TRANSMISSION_MAINTENANCE_RE.search(str(x) if x else ""))
    ).fillna(False)

    # Only extract candidates for broad, non-maintenance transmission rows.
    is_broad = out["project_is_transmission_broad"].values
    is_maintenance = out["project_is_transmission_maintenance"].values
    is_active = is_broad & ~is_maintenance
    n_broad = int(is_broad.sum())
    n_active = int(is_active.sum())
    texts = full_text.tolist()

    candidates: List[List[Dict]] = []
    for txt, active in zip(texts, is_active):
        if active:
            candidates.append(_extract_length_candidates(txt, TRANSMISSION_HINTS, prefix="tx"))
        else:
            candidates.append([])

    # Classify action type on each candidate sentence — used only for mileage split.
    for cand_list in candidates:
        for c in cand_list:
            c["candidate_action_type"] = _classify_candidate_action(c["source_text"])

    # Adjudicate lengths in parallel (workers > 1 speeds up the LLM calls).
    n_skipped = n_broad - n_active
    llm_label = f"Claude API ({CLAUDE_DEFAULT_MODEL})" if use_llm else "LLM off"
    print(f"  {len(texts):,} rows | {n_broad:,} broad-tx | {n_active:,} active (excl. {n_skipped} maintenance) | workers={workers} | {llm_label}")

    def _adjudicate_one(args):
        i, txt, cands = args
        return i, _adjudicate_transmission_length(
            txt, cands, use_llm=use_llm, timeout=timeout
        )

    indexed = [(i, txt, cands) for i, (txt, cands) in enumerate(zip(texts, candidates))]
    adjudications_map: Dict[int, LengthAdjudication] = {}
    llm_trigger_count = 0

    try:
        from tqdm import tqdm as _tqdm
        _pbar = _tqdm(total=len(indexed), desc="Adjudicating", unit="row")
    except ImportError:
        _pbar = None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_adjudicate_one, item): item[0] for item in indexed}
        for future in as_completed(futures):
            i, adj = future.result()
            adjudications_map[i] = adj
            if adj.llm_trigger:
                llm_trigger_count += 1
                if use_llm:
                    print(f"  [LLM] row {i}: status={adj.llm_status} length={adj.selected_length_miles:.1f}mi")
            if _pbar:
                _pbar.update(1)

    if _pbar:
        _pbar.close()

    adjudications = [adjudications_map[i] for i in range(len(texts))]
    print(f"  LLM-trigger rows: {llm_trigger_count}"
          + (" | LLM called on all triggers" if use_llm else " | rerun with --use-llm to adjudicate"))

    out["project_transmission_length_candidates_json"] = [json.dumps(c, ensure_ascii=False) for c in candidates]
    out["project_transmission_length_candidate_count"] = [a.candidate_count for a in adjudications]
    out["project_transmission_length_distinct_candidate_count"] = [a.distinct_candidate_count for a in adjudications]
    out["project_transmission_length_taxonomy"] = [a.taxonomy for a in adjudications]
    out["project_transmission_length_selection_method"] = [a.selection_method for a in adjudications]
    out["project_transmission_length_selected_candidate_ids"] = [
        json.dumps(a.selected_candidate_ids) for a in adjudications
    ]
    out["project_transmission_length_llm_trigger"] = [a.llm_trigger for a in adjudications]
    out["project_transmission_length_llm_used"] = [a.llm_used for a in adjudications]
    out["project_transmission_length_llm_status"] = [a.llm_status for a in adjudications]
    out["project_transmission_length_llm_reasoning"] = [a.llm_reasoning for a in adjudications]
    out["project_transmission_length_llm_model"] = [a.llm_model for a in adjudications]
    out["project_tx_llm_run_at"] = [a.llm_run_at for a in adjudications]

    # project_transmission_length_miles  = rule-based only (comparison baseline)
    # project_transmission_length_final  = LLM result when llm_used=True, else rule-based
    out["project_transmission_length_miles"] = [a.rule_based_length_miles for a in adjudications]
    out["project_transmission_length_final"]  = [a.selected_length_miles  for a in adjudications]
    out["project_transmission_length_confidence"] = [a.confidence for a in adjudications]
    out["project_transmission_length_source_text"] = [a.source_text for a in adjudications]

    # Project-level action type from full title+description text.
    out["project_transmission_action"] = [
        _classify_project_transmission_action(txt) for txt in context_text.tolist()
    ]
    # Separate mileage by action type (NaN when no candidates of that type exist).
    out["project_transmission_new_build_miles"] = [_miles_by_action(cands, "new_build") for cands in candidates]
    out["project_transmission_upgrade_miles"]   = [_miles_by_action(cands, "upgrade")   for cands in candidates]

    # Keep only broad transmission projects populated.
    non_broad = ~out["project_is_transmission_broad"]
    out.loc[non_broad, "project_transmission_length_miles"] = np.nan
    out.loc[non_broad, "project_transmission_length_final"]  = np.nan
    out.loc[non_broad, "project_transmission_length_confidence"] = "none"
    out.loc[non_broad, "project_transmission_length_source_text"] = ""
    out.loc[non_broad, "project_transmission_length_candidate_count"] = 0
    out.loc[non_broad, "project_transmission_length_distinct_candidate_count"] = 0
    out.loc[non_broad, "project_transmission_length_taxonomy"] = "none"
    out.loc[non_broad, "project_transmission_length_selection_method"] = "none"
    out.loc[non_broad, "project_transmission_length_selected_candidate_ids"] = "[]"
    out.loc[non_broad, "project_transmission_length_llm_trigger"] = False
    out.loc[non_broad, "project_transmission_length_llm_used"] = False
    out.loc[non_broad, "project_transmission_length_llm_status"] = "not_triggered"
    out.loc[non_broad, "project_transmission_length_llm_reasoning"] = ""
    out.loc[non_broad, "project_transmission_length_llm_model"] = ""
    out.loc[non_broad, "project_tx_llm_run_at"] = ""
    # Stamp every row with when this extraction ran; LLM timestamp is per-row (empty if not called).
    out["project_tx_extraction_run_at"] = extraction_run_at
    out.loc[non_broad, "project_transmission_length_candidates_json"] = "[]"
    out.loc[non_broad, "project_transmission_action"] = "none"
    # Broad-tx projects excluded as maintenance get a consistent "maintenance" label
    # regardless of what _classify_project_transmission_action found in the text.
    broad_maintenance = out["project_is_transmission_broad"] & out["project_is_transmission_maintenance"].fillna(False)
    out.loc[broad_maintenance, "project_transmission_action"] = "maintenance"
    out.loc[non_broad, "project_transmission_new_build_miles"] = np.nan
    out.loc[non_broad, "project_transmission_upgrade_miles"] = np.nan

    # Page-level length recovery: search document body text for projects that
    # passed the build-text gate but have no mileage in title/description.
    if processed_dir is not None:
        out = _extract_tx_length_from_pages(
            out, processed_dir,
            max_ea_eis_pages=max_ea_eis_pages,
            use_llm=use_llm, timeout=timeout, workers=workers,
        )
    else:
        out["project_transmission_length_from_pages"] = False

    out["project_is_transmission"] = (
        out["project_has_transmission_type_tag"]
        & out["project_has_transmission_build_text"]
        & (out["project_transmission_length_final"] >= 1.0)
        & ~out["project_is_transmission_maintenance"]
    )

    return out


def _add_pipeline_columns(
    df: pd.DataFrame,
    full_text: pd.Series,
    context_text: pd.Series,
    title_txt: pd.Series,
) -> pd.DataFrame:
    out = df.copy()
    lower_text = full_text.str.lower()

    # project_is_pipeline: entry gate — any project describing pipeline infrastructure.
    # Searches: project_title + project_description + project_type (structured metadata only,
    # not full NEPA document pages).
    # Broadened from original \bpipelines?\b to also catch flowlines and gathering lines,
    # which are common synonyms in CCS, gas gathering, and hydrogen conveyance projects.
    out["project_is_pipeline"] = lower_text.str.contains(
        r"\bpipelines?\b|\bflowlines?\b|\bgathering lines?\b", regex=True
    )

    # Carbon/CCS pipeline: pipeline flag + any carbon-related keyword.
    # Includes CCS acronym and "carbon capture" phrase which often appear instead of "carbon pipeline".
    out["project_is_carbon_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\b(?:carbon|co2|carbon dioxide|ccs|carbon capture|carbon sequestration)\b", regex=True
    )
    # Hydrogen pipeline: pipeline flag + hydrogen keyword.
    out["project_is_hydrogen_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bhydrogen\b", regex=True
    )
    # Natural gas pipeline: pipeline flag + natural gas / gas line keywords.
    out["project_is_natural_gas_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bnatural gas\b|\bgas pipeline\b|\bgas gathering\b|\bgas line\b", regex=True
    )

    # New-build filter — analogous to the transmission 4-gate filter.
    # Gate A (build text): PIPELINE_BUILD_RE on title + description (context_text, not full_text
    # which includes doc titles and NOI text that can add noise).
    # Gate B (maintenance exclusion): PIPELINE_MAINTENANCE_RE on title only — maintenance language
    # in descriptions is often incidental (e.g. "access road maintenance" in a construction project).
    # No length gate: pipeline length coverage is much sparser than transmission, so requiring a
    # minimum length would discard many genuine new-build projects that lack an extractable length.
    lower_context = context_text.str.lower()
    lower_title = title_txt.str.lower()
    out["project_pipeline_has_build_text"] = lower_context.str.contains(
        PIPELINE_BUILD_RE, na=False
    )
    out["project_pipeline_is_maintenance"] = lower_title.str.contains(
        PIPELINE_MAINTENANCE_RE, na=False
    )
    # Carbon and hydrogen pipelines are exempted from the build-text gate: these
    # technologies have no established infrastructure base, so virtually all NEPA
    # reviews are for new construction or major new projects rather than operational
    # maintenance.  Natural gas, oil/petroleum, and other pipeline groups still
    # require explicit build-text to pass (their large existing infrastructure
    # generates many routine operational/maintenance NEPA filings).
    out["project_is_pipeline_new_build"] = (
        out["project_is_pipeline"]
        & (
            out["project_pipeline_has_build_text"]
            | out["project_is_carbon_pipeline"]
            | out["project_is_hydrogen_pipeline"]
        )
        & ~out["project_pipeline_is_maintenance"]
    )

    candidates = [
        _extract_length_candidates(txt, PIPELINE_HINTS, prefix="pl")
        for txt in full_text.tolist()
    ]

    best = [_best_single_candidate(cands) for cands in candidates]
    out["project_pipeline_length_candidates_json"] = [json.dumps(c, ensure_ascii=False) for c in candidates]
    out["project_pipeline_length_candidate_count"] = [len(c) for c in candidates]
    out["project_pipeline_length_distinct_candidate_count"] = [len(_collapse_candidates_by_value(c)) for c in candidates]
    out["project_pipeline_length_miles"] = [x[0] for x in best]
    out["project_pipeline_length_confidence"] = [x[1] for x in best]
    out["project_pipeline_length_source_text"] = [x[2] for x in best]

    non_pipeline = ~out["project_is_pipeline"]
    out.loc[non_pipeline, "project_pipeline_length_miles"] = np.nan
    out.loc[non_pipeline, "project_pipeline_length_confidence"] = "none"
    out.loc[non_pipeline, "project_pipeline_length_source_text"] = ""
    out.loc[non_pipeline, "project_pipeline_length_candidate_count"] = 0
    out.loc[non_pipeline, "project_pipeline_length_distinct_candidate_count"] = 0
    out.loc[non_pipeline, "project_pipeline_length_candidates_json"] = "[]"

    out["project_pipeline_group"] = np.select(
        [
            out["project_is_carbon_pipeline"],
            out["project_is_hydrogen_pipeline"],
            out["project_is_natural_gas_pipeline"],
            out["project_is_pipeline"],
        ],
        [
            "carbon_pipeline",
            "hydrogen_pipeline",
            "natural_gas_pipeline",
            "other_pipeline",
        ],
        default="none",
    )

    return out


def _add_geothermal_columns(df: pd.DataFrame, full_text: pd.Series) -> pd.DataFrame:
    out = df.copy()
    # Use project_type tags as the geothermal inclusion rule.
    type_text = _series_text(out, "project_type").str.lower()
    geothermal_re = r"\b(?:geothermal|enhanced geothermal|egs)\b"
    out["project_is_geothermal"] = type_text.str.contains(geothermal_re, regex=True)

    phase_results = [_classify_geothermal_phase(text) for text in full_text]
    out["project_geothermal_phase"] = [r[0] for r in phase_results]
    # Matched phases as a JSON array string so R can parse it with safe_fromJSON.
    # For single-phase rows: '["drilling"]'; for multi_phase: '["exploration","drilling"]';
    # for none/unknown: '[]'.
    out["project_geothermal_matched_phases"] = [
        json.dumps(r[1]) for r in phase_results
    ]
    return out


# --------------------------
# GEOTHERMAL PHASE ML CLASSIFIER
# --------------------------

def _load_geo_page_texts(
    df: pd.DataFrame,
    processed_dir: Path,
    max_pages: int = 3,
) -> Dict[str, str]:
    """Return {pid_clean: first-N-pages text} for rows in *df*, using DuckDB parquets.

    Mirrors the page-loading pattern used for transmission length recovery.
    Returns an empty dict if duckdb is unavailable or processed_dir is missing.
    """
    try:
        import duckdb
    except ImportError:
        print("  [geo-pages] duckdb not available — skipping page text enrichment")
        return {}

    if "project_id" not in df.columns:
        return {}

    pid_series = df["project_id"].astype(str).str.replace("-", "", regex=False)

    # Determine which process_type each row belongs to
    ptype_col = next((c for c in ("process_type", "dataset_source") if c in df.columns), None)

    result: Dict[str, str] = {}
    con = duckdb.connect()

    for ptype in ["CE", "EA", "EIS"]:
        if ptype_col is not None:
            submask = df[ptype_col].str.upper().str.strip() == ptype
            target_pids = pid_series[submask].unique().tolist()
        else:
            target_pids = pid_series.unique().tolist()

        if not target_pids:
            continue

        ptype_lower = ptype.lower()
        docs_path  = str(processed_dir / ptype_lower / "documents.parquet")
        pages_path = str(processed_dir / ptype_lower / "pages.parquet")

        if not Path(docs_path).exists() or not Path(pages_path).exists():
            continue

        con.register("_geo_pids", pd.DataFrame({"pid": target_pids}))
        try:
            if ptype == "CE":
                query = """
                    SELECT replace(d.project_id.value, '-', '') AS pid, p.page_text
                    FROM read_parquet(?) d
                    JOIN read_parquet(?) p ON p.document_id = d.document_id
                    WHERE replace(d.project_id.value, '-', '') IN (SELECT pid FROM _geo_pids)
                """
                page_df = con.execute(query, [docs_path, pages_path]).df()
            else:
                query = """
                    SELECT pid, page_text FROM (
                        SELECT
                            replace(d.project_id.value, '-', '') AS pid,
                            p.page_text,
                            ROW_NUMBER() OVER (
                                PARTITION BY d.document_id ORDER BY p.page_number
                            ) AS rn
                        FROM read_parquet(?) d
                        JOIN read_parquet(?) p ON p.document_id = d.document_id
                        WHERE replace(d.project_id.value, '-', '') IN (SELECT pid FROM _geo_pids)
                          AND d.main_document = 'YES'
                    ) WHERE rn <= ?
                """
                page_df = con.execute(query, [docs_path, pages_path, max_pages]).df()
        except Exception as exc:
            print(f"  [geo-pages] {ptype}: DuckDB error — {exc}")
            continue

        if page_df.empty:
            continue

        for pid, grp in page_df.groupby("pid"):
            blob = " ".join(str(t) for t in grp["page_text"] if t)
            result[pid] = " ".join(blob.split()[:300])  # cap at 300 words

        n_loaded = len(page_df["pid"].unique())
        print(f"  [geo-pages] {ptype}: loaded page text for {n_loaded:,} projects")

    return result


def _geo_phase_text(row: "pd.Series", page_texts: "Dict[str, str] | None" = None) -> str:
    """Compose input text for the ML phase classifier.

    Uses title + project_type + first 100 words of description, plus up to 300
    words of page text when *page_texts* is provided and a match exists.
    """
    title = str(row.get("project_title_txt", "") or row.get("project_title", "") or "")
    ptype = str(row.get("project_type", "") or "")
    desc  = str(row.get("project_description_txt", "") or row.get("project_description", "") or "")
    desc_short = " ".join(desc.split()[:100])

    parts = [p for p in [title, ptype, desc_short] if p.strip()]

    if page_texts is not None:
        pid = str(row.get("project_id", "") or "").replace("-", "")
        page = page_texts.get(pid, "")
        if page:
            parts.append(page)

    return " ".join(parts)


def train_geothermal_phase_classifier(
    input_path: Path,
    model_dir: Path,
    base_model: str = GEO_PHASE_DEFAULT_BASE_MODEL,
    epochs: int = 5,
    test_size: float = 0.2,
    batch_size: int = 16,
    processed_dir: Path | None = None,
    self_training_rounds: int = 1,
    self_training_threshold: float = 0.70,
    max_pages: int = 3,
) -> None:
    """Fine-tune a sequence classifier on labeled geothermal phase rows.

    Improvements over the basic version:
    - **Page text**: if *processed_dir* is given, the first *max_pages* pages of
      each project's main document are appended to the input text.
    - **Weighted loss**: class weights are computed from training label frequencies
      so that rare phases (operations, exploration) are not ignored.
    - **Self-training**: after the initial fit, the model is applied to all
      unknown rows; predictions above *self_training_threshold* are added as
      pseudo-labels and the model is retrained for *self_training_rounds* extra
      rounds.

    The trained model and tokenizer are saved to *model_dir*.
    """
    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch
        import torch.nn as nn
        from torch.utils.data import Dataset as _TorchDataset
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report
        from sklearn.utils.class_weight import compute_class_weight
    except ImportError as exc:
        raise ImportError(
            "Training requires: pip install transformers torch scikit-learn accelerate. "
            f"Missing package: {exc}"
        ) from exc

    df = pd.read_parquet(input_path)

    # ── Page text enrichment ──────────────────────────────────────────────────
    page_texts: Dict[str, str] = {}
    if processed_dir is not None and Path(processed_dir).exists():
        print(f"Loading page text from: {processed_dir}")
        page_texts = _load_geo_page_texts(df, Path(processed_dir), max_pages=max_pages)
        print(f"  Page text loaded for {len(page_texts):,} projects")
    else:
        print("No processed_dir provided — using title + type + description only")

    # ── Labeled examples ─────────────────────────────────────────────────────
    labeled = df[
        df.get("project_is_geothermal", pd.Series(False, index=df.index)).fillna(False)
        & df.get("project_geothermal_phase", pd.Series("unknown", index=df.index))
             .isin(GEO_PHASE_LABELS)
    ].copy()

    if len(labeled) == 0:
        raise ValueError(
            "No labeled geothermal rows found. "
            "Run --run geothermal first to populate project_geothermal_phase."
        )

    labeled["_text"]     = labeled.apply(lambda r: _geo_phase_text(r, page_texts), axis=1)
    labeled["_label_id"] = labeled["project_geothermal_phase"].map(GEO_PHASE_LABEL2ID)
    labeled = labeled.dropna(subset=["_label_id"])
    labeled["_label_id"] = labeled["_label_id"].astype(int)

    print(f"\nLabeled examples: {len(labeled):,}")
    print(labeled.groupby("project_geothermal_phase").size().rename("n").to_string())

    if len(labeled) < 10:
        raise ValueError("Too few labeled examples to train (need ≥10).")

    train_df, val_df = train_test_split(
        labeled, test_size=test_size, stratify=labeled["_label_id"], random_state=42
    )
    print(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}")

    # ── Unknown rows for self-training ────────────────────────────────────────
    unknown_mask = (
        df.get("project_is_geothermal", pd.Series(False, index=df.index)).fillna(False)
        & (df.get("project_geothermal_phase", pd.Series("", index=df.index)) == "unknown")
    )
    unknown_df = df[unknown_mask].copy()
    unknown_df["_text"] = unknown_df.apply(lambda r: _geo_phase_text(r, page_texts), axis=1)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # ── Dataset helper ────────────────────────────────────────────────────────
    class _GeoDataset(_TorchDataset):
        def __init__(self, texts: List[str], labels: List[int]) -> None:
            enc = tokenizer(
                texts,
                truncation=True, padding="max_length",
                max_length=256, return_tensors="pt",
            )
            self.input_ids      = enc["input_ids"]
            self.attention_mask = enc["attention_mask"]
            self.labels         = torch.tensor(labels, dtype=torch.long)

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Dict:
            return {
                "input_ids":      self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels":         self.labels[idx],
            }

    # ── Trainer with weighted cross-entropy ───────────────────────────────────
    def _make_weighted_trainer(
        train_dataset: _TorchDataset,
        val_dataset: _TorchDataset,
        train_labels: List[int],
        n_epochs: int,
    ) -> "Trainer":
        raw_weights = compute_class_weight(
            "balanced",
            classes=np.arange(len(GEO_PHASE_LABELS)),
            y=train_labels,
        )
        cw = torch.FloatTensor(raw_weights)

        class _WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels  = inputs.get("labels")
                outputs = model(**inputs)
                logits  = outputs.get("logits")
                loss = nn.CrossEntropyLoss(weight=cw.to(logits.device))(
                    logits.view(-1, model.config.num_labels),
                    labels.view(-1),
                )
                return (loss, outputs) if return_outputs else loss

        args = TrainingArguments(
            output_dir=str(model_dir),
            num_train_epochs=n_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            logging_steps=20,
            report_to="none",
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            base_model,
            num_labels=len(GEO_PHASE_LABELS),
            id2label=GEO_PHASE_ID2LABEL,
            label2id=GEO_PHASE_LABEL2ID,
        )
        return _WeightedTrainer(
            model=model, args=args,
            train_dataset=train_dataset, eval_dataset=val_dataset,
        )

    # ── Round 0: initial training ─────────────────────────────────────────────
    print(f"\n=== Training round 0 (base labeled data, {epochs} epochs) ===")
    train_texts  = train_df["_text"].tolist()
    train_labels = train_df["_label_id"].tolist()

    val_dataset   = _GeoDataset(val_df["_text"].tolist(), val_df["_label_id"].tolist())
    train_dataset = _GeoDataset(train_texts, train_labels)

    trainer = _make_weighted_trainer(train_dataset, val_dataset, train_labels, epochs)
    trainer.train()

    # ── Self-training rounds ──────────────────────────────────────────────────
    for rnd in range(1, self_training_rounds + 1):
        if unknown_df.empty:
            print(f"\nSelf-training round {rnd}: no unknown rows — skipping")
            break

        print(f"\n=== Self-training round {rnd} ===")
        model_so_far = trainer.model
        model_so_far.eval()

        unk_texts = unknown_df["_text"].tolist()
        all_pred_ids: List[int]    = []
        all_confs:    List[float]  = []

        with torch.no_grad():
            for i in range(0, len(unk_texts), batch_size * 2):
                enc    = tokenizer(
                    unk_texts[i : i + batch_size * 2],
                    truncation=True, padding=True, max_length=256, return_tensors="pt",
                )
                logits = model_so_far(**enc).logits
                probs  = torch.softmax(logits, dim=-1)
                all_pred_ids.extend(probs.argmax(dim=-1).tolist())
                all_confs.extend(probs.max(dim=-1).values.tolist())

        pseudo = unknown_df.copy()
        pseudo["_label_id"] = all_pred_ids
        pseudo["_conf"]     = all_confs
        pseudo = pseudo[pseudo["_conf"] >= self_training_threshold]

        print(f"  High-confidence pseudo-labels (≥{self_training_threshold}): {len(pseudo):,}")
        if pseudo.empty:
            print("  No pseudo-labels above threshold — stopping self-training")
            break

        # Combine original labeled + pseudo-labeled for this round
        aug_texts  = train_texts + pseudo["_text"].tolist()
        aug_labels = train_labels + pseudo["_label_id"].tolist()

        train_dataset_aug = _GeoDataset(aug_texts, aug_labels)
        trainer = _make_weighted_trainer(
            train_dataset_aug, val_dataset, aug_labels,
            n_epochs=max(2, epochs // 2),
        )
        trainer.train()

    # ── Save and report ───────────────────────────────────────────────────────
    trainer.model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"\nModel saved: {model_dir}")

    preds    = trainer.predict(val_dataset)
    pred_ids = preds.predictions.argmax(axis=-1)
    print("\nValidation classification report:")
    print(classification_report(
        val_df["_label_id"].tolist(), pred_ids,
        target_names=GEO_PHASE_LABELS, zero_division=0,
    ))


def classify_geothermal_phase_ml(
    input_path: Path,
    output_path: Path,
    model_dir: Path = GEO_PHASE_MODEL_DIR,
    batch_size: int = 32,
    dry_run: bool = False,
    processed_dir: Path | None = None,
    max_pages: int = 3,
) -> None:
    """Apply the trained classifier to rows where phase == 'unknown'.

    Writes three new columns to the parquet:
    - ``project_geothermal_phase`` — updated from 'unknown' to the predicted label
    - ``project_geothermal_phase_ml_confidence`` — softmax score for the winning label
    - ``project_geothermal_phase_ml_classified`` — True for rows updated by this step
    """
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
    except ImportError as exc:
        raise ImportError(
            "Classification requires: pip install transformers torch. "
            f"Missing: {exc}"
        ) from exc

    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_dir}. "
            "Run --geothermal-phase-train first."
        )

    df = pd.read_parquet(input_path)

    mask = (
        df.get("project_is_geothermal", pd.Series(False, index=df.index)).fillna(False)
        & (df.get("project_geothermal_phase", pd.Series("", index=df.index)) == "unknown")
    )
    n_unknown = int(mask.sum())
    print(f"Geothermal rows with unknown phase: {n_unknown:,}")
    if n_unknown == 0:
        print("Nothing to classify.")
        return

    # Page text enrichment (mirrors training)
    page_texts: Dict[str, str] = {}
    if processed_dir is not None and Path(processed_dir).exists():
        print(f"Loading page text from: {processed_dir}")
        page_texts = _load_geo_page_texts(df[mask], Path(processed_dir), max_pages=max_pages)
        print(f"  Page text loaded for {len(page_texts):,} projects")

    texts = df[mask].apply(lambda r: _geo_phase_text(r, page_texts), axis=1).tolist()

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model     = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.eval()

    all_pred_ids: List[int]      = []
    all_confidences: List[float] = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc    = tokenizer(
                texts[i : i + batch_size],
                truncation=True, padding=True, max_length=256, return_tensors="pt",
            )
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1)
            all_pred_ids.extend(probs.argmax(dim=-1).tolist())
            all_confidences.extend(probs.max(dim=-1).values.tolist())

    predicted_labels = [GEO_PHASE_ID2LABEL[i] for i in all_pred_ids]

    print("\nPredicted phase distribution (unknowns only):")
    print(pd.Series(predicted_labels).value_counts().to_string())
    print(f"Mean confidence: {float(np.mean(all_confidences)):.3f}")

    if dry_run:
        print("\nDry run — no changes written.")
        return

    if "project_geothermal_phase_ml_classified" not in df.columns:
        df["project_geothermal_phase_ml_classified"] = False
    if "project_geothermal_phase_ml_confidence" not in df.columns:
        df["project_geothermal_phase_ml_confidence"] = np.nan

    df.loc[mask, "project_geothermal_phase"]               = predicted_labels
    df.loc[mask, "project_geothermal_phase_ml_confidence"] = all_confidences
    df.loc[mask, "project_geothermal_phase_ml_classified"] = True

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}  ({n_unknown:,} rows updated)")


# --------------------------
# PUBLIC API
# --------------------------

def normalize_run_targets(run: Sequence[str] | str) -> tuple[List[str], bool]:
    """Return (targets, use_llm). ``--run llm`` maps to transmission + LLM enabled."""
    if isinstance(run, str):
        targets = [run]
    else:
        targets = list(run)

    targets = [t.strip().lower() for t in targets if t and str(t).strip()]
    if not targets or "all" in targets:
        return ["transmission", "geothermal", "pipeline"], False

    use_llm = "llm" in targets
    # Replace "llm" token with "transmission"
    targets = ["transmission" if t == "llm" else t for t in targets]
    # Deduplicate while preserving order
    seen: set = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]

    allowed = {"transmission", "geothermal", "pipeline"}
    cleaned = [t for t in targets if t in allowed]
    if not cleaned:
        raise ValueError(f"No valid run targets in {targets}. Allowed: {sorted(allowed | {'llm'})}")
    return cleaned, use_llm


def add_technology_columns(
    df: pd.DataFrame,
    run: Sequence[str] | str = "all",
    use_llm: bool = False,
    timeout: int = 120,
    workers: int = 1,
    processed_dir: Path | None = None,
    max_ea_eis_pages: int = 10,
) -> pd.DataFrame:
    """
    Add technology-specific features to a project dataframe.

    Args:
        df: project dataframe
        run: one or more of transmission/geothermal/pipeline/all
        use_llm: enable Claude API adjudication for multi-candidate transmission rows
        timeout: seconds before an API request times out
        workers: parallel workers for adjudication

    Returns:
        DataFrame with requested technology columns added/updated.
    """
    targets, run_implies_llm = normalize_run_targets(run)
    use_llm = bool(use_llm or run_implies_llm)
    out = df.copy()

    title_txt = _series_text(out, "project_title")
    desc_txt = _series_text(out, "project_description")
    type_txt = _series_text(out, "project_type")

    # Option 1: noi_project_title (already in projects_combined, free join)
    noi_txt = _series_text(out, "noi_project_title")

    # Option 2: document titles from documents_combined.parquet, aggregated per project
    doc_title_txt = pd.Series("", index=out.index)
    if "project_id" in out.columns:
        docs_path = Path(__file__).resolve().parent.parent.parent / "data" / "analysis" / "documents_combined.parquet"
        if docs_path.exists():
            docs = pd.read_parquet(docs_path, columns=["project_id", "document_title"])
            docs = docs.dropna(subset=["document_title"])
            docs_agg = (
                docs.groupby("project_id")["document_title"]
                .apply(lambda x: " ".join(x.unique()))
                .reset_index()
                .rename(columns={"document_title": "_doc_titles"})
            )
            out = out.merge(docs_agg, on="project_id", how="left")
            doc_title_txt = out["_doc_titles"].fillna("").astype(str)
            out = out.drop(columns=["_doc_titles"])

    full_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
        + " "
        + type_txt.fillna("").astype(str)
        + " "
        + noi_txt.fillna("").astype(str)
        + " "
        + doc_title_txt
    ).str.strip()

    context_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
    ).str.strip()

    if "transmission" in targets:
        out = _add_transmission_columns(
            out, full_text, context_text, type_txt, title_txt,
            use_llm=use_llm, timeout=timeout, workers=workers,
            processed_dir=processed_dir, max_ea_eis_pages=max_ea_eis_pages,
        )

    if "geothermal" in targets:
        out = _add_geothermal_columns(out, full_text)

    if "pipeline" in targets:
        out = _add_pipeline_columns(out, full_text, context_text, title_txt)

    return out


# --------------------------
# CLI
# --------------------------

def _default_input_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "data" / "analysis" / "projects_combined.parquet"


def _default_transmission_output_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "data" / "analysis" / "projects_transmission.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract technology-specific project features")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["all"],
        choices=["all", "transmission", "geothermal", "pipeline", "llm"],
        help="Domains to run. Use 'llm' to run transmission + Claude adjudication (default: all)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_input_path(),
        help="Input parquet file (default: data/analysis/projects_combined.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_default_transmission_output_path(),
        help="Transmission-only output path (default: data/analysis/projects_transmission.parquet)",
    )
    parser.add_argument(
        "--projects-output",
        type=Path,
        default=None,
        help=(
            "Optional full-project output path. "
            "Defaults to --input (overwrite in place) so geothermal/pipeline "
            "columns persist to projects_combined.parquet."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds before a Claude API request times out (default: 120)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for Claude API adjudication (default: 1)",
    )
    parser.add_argument(
        "--page-search-max-pages",
        type=int,
        default=10,
        help="Max pages to search per EA/EIS main document during length recovery (default: 10)",
    )

    # -- Geothermal phase ML classifier --
    parser.add_argument(
        "--geothermal-phase-train",
        action="store_true",
        help=(
            "Fine-tune a DistilBERT classifier on the labeled geothermal phase rows "
            "already present in --input. Run this after --run geothermal has populated "
            "project_geothermal_phase for the non-unknown rows."
        ),
    )
    parser.add_argument(
        "--geothermal-phase-classify",
        action="store_true",
        help=(
            "Apply the trained geothermal phase classifier to all rows where "
            "project_geothermal_phase == 'unknown'. Writes results back to --input "
            "(or --projects-output if specified)."
        ),
    )
    parser.add_argument(
        "--geo-phase-model-dir",
        type=Path,
        default=GEO_PHASE_MODEL_DIR,
        help=f"Directory to save/load the trained phase model (default: {GEO_PHASE_MODEL_DIR})",
    )
    parser.add_argument(
        "--geo-phase-base-model",
        type=str,
        default=GEO_PHASE_DEFAULT_BASE_MODEL,
        help=f"HuggingFace base model for fine-tuning (default: {GEO_PHASE_DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--geo-phase-epochs",
        type=int,
        default=5,
        help="Training epochs for geothermal phase classifier (default: 5)",
    )
    parser.add_argument(
        "--geo-phase-test-size",
        type=float,
        default=0.2,
        help="Validation fraction for geothermal phase training (default: 0.2)",
    )
    parser.add_argument(
        "--geo-phase-batch-size",
        type=int,
        default=16,
        help="Batch size for training and inference (default: 16)",
    )
    parser.add_argument(
        "--geo-phase-dry-run",
        action="store_true",
        help="Classify but do not write any output (useful for previewing predictions)",
    )
    parser.add_argument(
        "--geo-phase-processed-dir",
        type=Path,
        default=None,
        help=(
            "Path to data/processed/ directory for page-text enrichment during "
            "training and classification. If omitted, uses title+type+description only."
        ),
    )
    parser.add_argument(
        "--geo-phase-max-pages",
        type=int,
        default=3,
        help="Max pages per document to include as input text (default: 3)",
    )
    parser.add_argument(
        "--geo-phase-self-training-rounds",
        type=int,
        default=1,
        help="Self-training rounds after initial fit (default: 1; 0 to disable)",
    )
    parser.add_argument(
        "--geo-phase-self-training-threshold",
        type=float,
        default=0.70,
        help="Minimum confidence for a self-training pseudo-label to be accepted (default: 0.70)",
    )

    return parser


def run_cli(args: argparse.Namespace) -> None:
    in_path = args.input
    tx_path = args.output
    projects_out_path = args.projects_output if args.projects_output is not None else in_path

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    # -- ML phase classifier: train (early exit) --
    if getattr(args, "geothermal_phase_train", False):
        print(f"=== Geothermal phase classifier: TRAIN ===")
        print(f"Input:      {in_path}")
        print(f"Base model: {args.geo_phase_base_model}")
        print(f"Model dir:  {args.geo_phase_model_dir}")
        _proc = getattr(args, "geo_phase_processed_dir", None)
        if _proc is None:
            _default = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
            _proc = _default if _default.exists() else None
        train_geothermal_phase_classifier(
            input_path=in_path,
            model_dir=args.geo_phase_model_dir,
            base_model=args.geo_phase_base_model,
            epochs=args.geo_phase_epochs,
            test_size=args.geo_phase_test_size,
            batch_size=args.geo_phase_batch_size,
            processed_dir=_proc,
            self_training_rounds=args.geo_phase_self_training_rounds,
            self_training_threshold=args.geo_phase_self_training_threshold,
            max_pages=args.geo_phase_max_pages,
        )
        return

    # -- ML phase classifier: classify (early exit) --
    if getattr(args, "geothermal_phase_classify", False):
        print(f"=== Geothermal phase classifier: CLASSIFY ===")
        print(f"Input:     {in_path}")
        print(f"Output:    {projects_out_path}")
        print(f"Model dir: {args.geo_phase_model_dir}")
        _proc = getattr(args, "geo_phase_processed_dir", None)
        if _proc is None:
            _default = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
            _proc = _default if _default.exists() else None
        classify_geothermal_phase_ml(
            input_path=in_path,
            output_path=projects_out_path,
            model_dir=args.geo_phase_model_dir,
            batch_size=args.geo_phase_batch_size,
            dry_run=args.geo_phase_dry_run,
            processed_dir=_proc,
            max_pages=args.geo_phase_max_pages,
        )
        return

    targets, use_llm = normalize_run_targets(args.run)
    print(f"Loading: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Rows loaded: {len(df):,}")
    print(f"Running targets: {', '.join(targets)}")
    llm_label = f"on (model={CLAUDE_DEFAULT_MODEL}, timeout={args.timeout}s, workers={args.workers})" if use_llm else "off"
    print(f"LLM mode: {llm_label}")

    # Page-length recovery is always enabled when processed/ dir is present.
    _default_processed = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    if _default_processed.exists():
        processed_dir: Path | None = _default_processed
        print(f"Page-length recovery: ON (max_pages={args.page_search_max_pages}, dir={processed_dir})")
    else:
        processed_dir = None
        print(f"Warning: processed dir not found at {_default_processed} — page recovery disabled")

    updated = add_technology_columns(
        df, run=targets, use_llm=use_llm,
        timeout=args.timeout, workers=args.workers,
        processed_dir=processed_dir, max_ea_eis_pages=args.page_search_max_pages,
    )
    projects_out_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(projects_out_path, index=False)
    print(f"Saved updated projects dataset: {projects_out_path}")

    tx_path.parent.mkdir(parents=True, exist_ok=True)

    if "transmission" in targets and any(
        c for c in updated.columns if c.startswith(TX_OUTPUT_PREFIXES)
    ):
        try:
            same_output = tx_path.resolve() == projects_out_path.resolve()
        except Exception:
            same_output = tx_path == projects_out_path
        if same_output:
            print(
                "Warning: transmission output path matches full projects output path; "
                "skipping transmission-only parquet write."
            )
            return
        tx_cols = ["project_id"] + [
            c for c in updated.columns if c.startswith(TX_OUTPUT_PREFIXES)
        ]
        updated[tx_cols].to_parquet(tx_path, index=False)
        print(f"Saved: {tx_path}  ({len(tx_cols) - 1} columns)")

    if "transmission" in targets and "project_is_transmission" in updated.columns:
        clean = updated[updated.get("project_energy_type", pd.Series("", index=updated.index)) == "Clean"] \
            if "project_energy_type" in updated.columns else updated
        n_tx = int(clean["project_is_transmission"].fillna(False).sum())
        n_llm_trigger = int(
            clean.loc[clean["project_is_transmission"].fillna(False),
                      "project_transmission_length_llm_trigger"].fillna(False).sum()
        )
        n_llm_used = int(
            clean.loc[clean["project_is_transmission"].fillna(False),
                      "project_transmission_length_llm_used"].fillna(False).sum()
        )
        print(f"Clean energy strict transmission projects: {n_tx:,}")
        print(f"  of which triggered LLM (2+ distinct candidates): {n_llm_trigger:,}")
        print(f"  of which ran Claude API successfully: {n_llm_used:,}")
        if "project_transmission_length_from_pages" in updated.columns:
            n_recovered = int(
                clean["project_transmission_length_from_pages"].fillna(False).sum()
            )
            print(f"  page-recovery: {n_recovered} projects had lengths recovered from document pages")


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    run_cli(cli_args)
