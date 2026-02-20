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

import argparse
import ast
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests


# --------------------------
# CONSTANTS
# --------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_LLM_MODEL = "llama3.2:3b-instruct-q4_K_M"

CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_DEFAULT_MODEL = "claude-haiku-4-5-20251001"

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
    r"|substation\s+construction|construct(?:ion|ing|ed)?\s+of\s+(?:a\s+|the\s+)?new"
    r"|build(?:ing)?\s+(?:a\s+|the\s+)?new"
    r"|switchyard|tap\s+line|tie\s+line|interconnection)\b",
    re.IGNORECASE,
)
_TX_ACT_UPGRADE_RE = re.compile(
    r"\b(?:replac(?:e|es|ed|ing|ement)|rebuild(?:ing|s|t)?|reconductor(?:ing|ed)?"
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
    r"\b(?:construct(?:ion|ed)?|build(?:ing)?|install(?:ation|ed)?|upgrade(?:d|s)?|rebuild(?:ing)?)\s+"
    r"(?:of\s+)?(?:new\s+)?(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line\b|"
    r"double-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"single-?circuit\s+(?:\d{2,4}\s*-?\s*k\s?v\s+)?transmission\s+line|"
    r"\b\d{2,4}\s*-?\s*k\s?v\s+(?:transmission\s+line|line)\b|"
    r"right-?of-?way.*transmission\s+line|transmission\s+line.*right-?of-?way)",
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
    ],
    "drilling": [
        r"\bdrilling\b",
        r"\bdrill pad\b",
        r"\bwell pad\b",
        r"\bproduction well\b",
        r"\binjection well\b",
        r"\bwell stimulation\b",
    ],
    "plant": [
        r"\bpower plant\b",
        r"\bgenerating station\b",
        r"\bsteam plant\b",
        r"\bbinary plant\b",
        r"\bflash plant\b",
        r"\bturbine\b",
        r"\binterconnection\b",
    ],
}


# --------------------------
# TYPES
# --------------------------

@dataclass
class LengthAdjudication:
    selected_length_miles: float
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


import time as _time


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
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
    provider: str = "ollama",
) -> Dict | None:
    """
    Adjudicate among competing transmission line length candidates using an LLM.

    Supports provider='ollama' (local Ollama) or provider='anthropic' (Claude Haiku).
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

    # --- call LLM (route by provider) ---
    raw = ""
    if provider == "anthropic":
        claude_model = CLAUDE_DEFAULT_MODEL
        result = _call_claude_api(prompt, model=claude_model, timeout=timeout)
        if result.get("error"):
            print(f"  [Claude error] {result['error']}")
            return None
        raw = result["response"]
    else:
        try:
            resp = requests.post(
                OLLAMA_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 120},
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
        except Exception:
            return None

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


def _adjudicate_transmission_length(
    full_text: str,
    candidates: List[Dict],
    use_llm: bool = False,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
    provider: str = "ollama",
) -> LengthAdjudication:
    groups = _collapse_candidates_by_value(candidates)
    candidate_count = len(candidates)
    distinct_count = len(groups)

    # Separate nontrivial groups into full-length and partial-crossing.
    # Partial-crossing candidates (e.g., "4.7 miles of public land") are
    # deprioritized when full-length alternatives exist, but kept as the
    # answer when they are the only option.
    nontrivial = [g for g in groups if g["value_miles"] >= 0.25]
    non_partial_nontrivial = [
        g for g in nontrivial
        if not g["best_candidate"].get("is_partial_crossing", False)
    ]
    effective_nontrivial = non_partial_nontrivial if non_partial_nontrivial else nontrivial

    # ------------------------------------------------------------------
    # LLM trigger logic:
    #   anthropic provider: trigger on ANY row with >= 2 nontrivial (>= 0.25 mi)
    #     distinct candidates — Claude reads all genuinely ambiguous rows.
    #     Sub-quarter-mile projects (tiny electric lines) are handled by rules.
    #   ollama provider: only fire when genuinely ambiguous among effective
    #     (non-partial) candidates (spread > 1.5x and no dominant build-verb winner)
    # ------------------------------------------------------------------
    if provider == "anthropic":
        llm_trigger = len(effective_nontrivial) >= 2
    elif len(effective_nontrivial) >= 2:
        nt_vals = [g["value_miles"] for g in effective_nontrivial]
        spread = max(nt_vals) / min(nt_vals) if min(nt_vals) > 0 else 1.0
        top_score = max(g["best_candidate"]["hint_score"] for g in effective_nontrivial)
        dominant = [
            g for g in effective_nontrivial
            if g["best_candidate"]["hint_score"] == top_score
            and g["best_candidate"].get("sentence_has_build_verb", False)
        ]
        llm_trigger = spread > 1.5 and not (len(dominant) == 1 and top_score >= 3)
    else:
        llm_trigger = False

    if candidate_count == 0:
        return LengthAdjudication(
            selected_length_miles=np.nan,
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
        )

    text_lower = (full_text or "").lower()
    has_alternative = bool(TRANSMISSION_ALTERNATIVE_RE.search(text_lower))
    has_additive = bool(TRANSMISSION_ADDITIVE_RE.search(text_lower))

    llm_used = False
    llm_status = "not_requested" if llm_trigger else "not_triggered"
    llm_result = None
    if llm_trigger and use_llm:
        try:
            llm_result = _run_llm_transmission_adjudication(
                candidates, model=model, timeout=timeout, provider=provider
            )
            if llm_result:
                llm_used = True
                llm_status = "success"
            else:
                llm_status = "failed_fallback_rule"
        except Exception:
            llm_used = False
            llm_status = "failed_fallback_rule"

    if llm_result:
        return LengthAdjudication(
            selected_length_miles=llm_result["selected_length_miles"],
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
        )

    if distinct_count <= 1:
        best = groups[0]["best_candidate"]
        confidence = "high" if best["hint_score"] >= 4 else "medium"
        return LengthAdjudication(
            selected_length_miles=best["value_miles"],
            confidence=confidence,
            source_text=best["source_text"],
            taxonomy="do_not_sum",
            selection_method="rule",
            selected_candidate_ids=[best["candidate_id"]],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
            llm_reasoning="",
        )

    # Rule order matters: alternatives should not be summed.
    if has_alternative:
        chosen = sorted(
            (g["best_candidate"] for g in groups),
            key=lambda x: (x["value_miles"], x["hint_score"]),
            reverse=True,
        )[0]
        confidence = "high" if chosen["hint_score"] >= 4 else "medium"
        return LengthAdjudication(
            selected_length_miles=chosen["value_miles"],
            confidence=confidence,
            source_text=chosen["source_text"],
            taxonomy="choose_alternative",
            selection_method="rule",
            selected_candidate_ids=[chosen["candidate_id"]],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
            llm_reasoning="",
        )

    if has_additive:
        selected = [g["best_candidate"] for g in groups]
        selected = sorted(selected, key=lambda x: (x["value_miles"], x["hint_score"]), reverse=True)
        value = round(sum(c["value_miles"] for c in selected), 3)
        source = " || ".join(c["source_text"] for c in selected)[:2000]
        confidence = "high" if len(selected) >= 2 else "medium"
        return LengthAdjudication(
            selected_length_miles=value,
            confidence=confidence,
            source_text=source,
            taxonomy="sum",
            selection_method="rule",
            selected_candidate_ids=[c["candidate_id"] for c in selected],
            candidate_count=candidate_count,
            distinct_candidate_count=distinct_count,
            llm_trigger=llm_trigger,
            llm_used=llm_used,
            llm_status=llm_status,
            llm_reasoning="",
        )

    # Ambiguous multi-candidate: among non-partial (preferred) candidates,
    # pick the sole one with a build verb; fall back to highest hint_score / take_max.
    # effective_groups = non-partial groups when available, else all groups.
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
    return LengthAdjudication(
        selected_length_miles=chosen["value_miles"],
        confidence=confidence,
        source_text=chosen["source_text"],
        taxonomy=taxonomy,
        selection_method="rule",
        selected_candidate_ids=[chosen["candidate_id"]],
        candidate_count=candidate_count,
        distinct_candidate_count=distinct_count,
        llm_trigger=llm_trigger,
        llm_used=llm_used,
        llm_status=llm_status,
        llm_reasoning="",
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


def _classify_geothermal_phase(text: str) -> str:
    """Return one of: none, exploration, drilling, plant, multi_phase, unknown."""
    txt = (text or "").lower()
    if "geothermal" not in txt:
        return "none"

    matches = []
    for phase, patterns in GEOTHERMAL_PHASE_PATTERNS.items():
        if any(re.search(p, txt, flags=re.IGNORECASE) for p in patterns):
            matches.append(phase)

    if len(matches) == 0:
        return "unknown"
    if len(matches) == 1:
        return matches[0]
    return "multi_phase"


# --------------------------
# DOMAIN EXTRACTORS
# --------------------------

def _add_transmission_columns(
    df: pd.DataFrame,
    full_text: pd.Series,
    context_text: pd.Series,
    type_text: pd.Series,
    use_llm: bool = False,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
    workers: int = 1,
    provider: str = "ollama",
) -> pd.DataFrame:
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
    out["project_is_transmission_maintenance"] = context_text.str.contains(
        TRANSMISSION_MAINTENANCE_RE, regex=False
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
    provider_label = f"provider={provider}" if use_llm else "LLM off"
    print(f"  {len(texts):,} rows | {n_broad:,} broad-tx | {n_active:,} active (excl. {n_skipped} maintenance) | workers={workers} | {provider_label}")

    def _adjudicate_one(args):
        i, txt, cands = args
        return i, _adjudicate_transmission_length(
            txt, cands, use_llm=use_llm, model=model, timeout=timeout, provider=provider
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

    out["project_transmission_length_miles"] = [a.selected_length_miles for a in adjudications]
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
    out.loc[non_broad, "project_transmission_length_candidates_json"] = "[]"
    out.loc[non_broad, "project_transmission_action"] = "none"
    out.loc[non_broad, "project_transmission_new_build_miles"] = np.nan
    out.loc[non_broad, "project_transmission_upgrade_miles"] = np.nan

    out["project_is_transmission_strict"] = (
        out["project_has_transmission_type_tag"]
        & out["project_has_transmission_build_text"]
        & (out["project_transmission_length_miles"] >= 1.0)
        & ~out["project_is_transmission_maintenance"]
    )
    out["project_is_transmission"] = out["project_is_transmission_strict"]

    return out


def _add_pipeline_columns(df: pd.DataFrame, full_text: pd.Series) -> pd.DataFrame:
    out = df.copy()
    lower_text = full_text.str.lower()

    out["project_is_pipeline"] = lower_text.str.contains(r"\bpipelines?\b", regex=True)
    out["project_is_carbon_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\b(?:carbon|co2|carbon dioxide)\b", regex=True
    )
    out["project_is_hydrogen_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bhydrogen\b", regex=True
    )
    out["project_is_natural_gas_pipeline"] = out["project_is_pipeline"] & lower_text.str.contains(
        r"\bnatural gas\b|\bgas pipeline\b", regex=True
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
    lower_text = full_text.str.lower()
    out["project_is_geothermal"] = lower_text.str.contains(r"\bgeothermal\b", regex=True)
    out["project_geothermal_phase"] = full_text.apply(_classify_geothermal_phase)
    return out


# --------------------------
# PUBLIC API
# --------------------------

def normalize_run_targets(run: Sequence[str] | str) -> List[str]:
    if isinstance(run, str):
        targets = [run]
    else:
        targets = list(run)

    targets = [t.strip().lower() for t in targets if t and str(t).strip()]
    if not targets or "all" in targets:
        return ["transmission", "geothermal", "pipeline"]

    allowed = {"transmission", "geothermal", "pipeline"}
    cleaned = [t for t in targets if t in allowed]
    if not cleaned:
        raise ValueError(f"No valid run targets in {targets}. Allowed: {sorted(allowed)}")
    return cleaned


def add_technology_columns(
    df: pd.DataFrame,
    run: Sequence[str] | str = "all",
    use_llm: bool = False,
    model: str = DEFAULT_LLM_MODEL,
    timeout: int = 120,
    workers: int = 1,
    provider: str = "ollama",
) -> pd.DataFrame:
    """
    Add technology-specific features to a project dataframe.

    Args:
        df: project dataframe
        run: one or more of transmission/geothermal/pipeline/all
        use_llm: optional LLM adjudication for multi-candidate transmission rows
        model: Ollama model name for LLM adjudication (ignored when provider='anthropic')
        timeout: seconds before an LLM request times out
        workers: parallel workers for adjudication (use 1 for anthropic to respect rate limits)
        provider: 'ollama' (local) or 'anthropic' (Claude Haiku)

    Returns:
        DataFrame with requested technology columns added/updated.
    """
    targets = normalize_run_targets(run)
    out = df.copy()

    title_txt = _series_text(out, "project_title")
    desc_txt = _series_text(out, "project_description")
    type_txt = _series_text(out, "project_type")

    full_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
        + " "
        + type_txt.fillna("").astype(str)
    ).str.strip()

    context_text = (
        title_txt.fillna("").astype(str)
        + " "
        + desc_txt.fillna("").astype(str)
    ).str.strip()

    if "transmission" in targets:
        out = _add_transmission_columns(
            out, full_text, context_text, type_txt,
            use_llm=use_llm, model=model, timeout=timeout, workers=workers, provider=provider,
        )

    if "geothermal" in targets:
        out = _add_geothermal_columns(out, full_text)

    if "pipeline" in targets:
        out = _add_pipeline_columns(out, full_text)

    return out


# --------------------------
# CLI
# --------------------------

def _default_projects_combined_path() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "data" / "analysis" / "projects_combined.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract technology-specific project features")
    parser.add_argument(
        "--run",
        nargs="+",
        default=["all"],
        choices=["all", "transmission", "geothermal", "pipeline"],
        help="Technology domains to run (default: all)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_default_projects_combined_path(),
        help="Input parquet file (default: data/analysis/projects_combined.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path (default: overwrite input)",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Attempt LLM adjudication for multi-candidate transmission rows",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_LLM_MODEL,
        help=f"Ollama model for LLM adjudication (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Seconds before an Ollama request times out (default: 120)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel workers for adjudication (default: 1; use 4 to speed up Ollama calls)",
    )
    parser.add_argument(
        "--provider",
        default="ollama",
        choices=["ollama", "anthropic"],
        help="LLM provider: 'ollama' (local) or 'anthropic' (Claude Haiku). "
             "Anthropic reads ALL multi-candidate rows; requires ANTHROPIC_API_KEY env var. "
             "(default: ollama)",
    )
    return parser


def run_cli(args: argparse.Namespace) -> None:
    in_path = args.input
    out_path = args.output or args.input

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    targets = normalize_run_targets(args.run)
    print(f"Loading: {in_path}")
    df = pd.read_parquet(in_path)
    print(f"Rows loaded: {len(df):,}")
    print(f"Running targets: {', '.join(targets)}")
    if args.use_llm:
        if args.provider == "anthropic":
            llm_label = f"on (provider=anthropic model={CLAUDE_DEFAULT_MODEL}, timeout={args.timeout}s, workers={args.workers})"
        else:
            llm_label = f"on (provider=ollama model={args.model}, timeout={args.timeout}s, workers={args.workers})"
    else:
        llm_label = "off"
    print(f"LLM mode: {llm_label}")

    updated = add_technology_columns(
        df, run=targets, use_llm=args.use_llm, model=args.model,
        timeout=args.timeout, workers=args.workers, provider=args.provider,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    if "transmission" in targets and "project_is_transmission" in updated.columns:
        n_tx = int(updated["project_is_transmission"].fillna(False).sum())
        n_multi = int((updated["project_transmission_length_distinct_candidate_count"].fillna(0) >= 2).sum())
        print(f"Transmission strict projects: {n_tx:,}")
        print(f"Transmission rows with multi distinct candidates: {n_multi:,}")


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    run_cli(cli_args)
