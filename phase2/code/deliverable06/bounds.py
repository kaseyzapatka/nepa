"""D6 — shared numeric-bound parser.

Extracts stated quantitative limits (acres, miles, kV, MW, wells) from free text.
Used to (a) parse an existing CE's stated bounds (04/06) and (b) compare them to
our FONSIs' extracted limits for the EXPAND test (07). Same regex family as 03.

Context-aware: a raw "N miles"/"N acres" in CE text is frequently NOT a size cap
on the regulated action — it can be an access-road length, a siting setback
("more than 10 nautical miles ... about 11.5 statute miles"), a unit conversion,
or a descriptor of a study/planning area. Comparing those against a project's
line length or footprint is a category error that manufactures false EXPAND
verdicts. Each candidate match is therefore screened against a window of
surrounding words and dropped if that window names a disqualifying context.
"""

from __future__ import annotations

import re

NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
RX = {
    "acres": re.compile(NUM + r"\s*acres?\b", re.IGNORECASE),
    "miles": re.compile(NUM + r"\s*miles?\b", re.IGNORECASE),
    "mw": re.compile(NUM + r"\s*(?:mw|megawatts?)\b", re.IGNORECASE),
    "kv": re.compile(NUM + r"\s*(?:kv|kilovolts?)\b", re.IGNORECASE),
    "wells": re.compile(NUM + r"\s*(?:wells?|borings?|boreholes?)\b", re.IGNORECASE),
}
# sanity caps (drop garbage grabs); shared with 03's intent
CAPS = {"acres": 100000, "miles": 1000, "mw": 10000, "kv": 1000, "wells": 1000}

# Context terms (in a window around the match) that mean the number is NOT a size
# cap on the regulated action. Screened case-insensitively.
DISQUALIFIERS = {
    # length values that describe an access road, a setback/proximity distance, or
    # a nautical→statute conversion — none of which bound the line/footprint we
    # measure in the FONSIs.
    # NOTE: bare "within" is excluded on purpose — "within existing
    # right-of-way" is ubiquitous in real transmission CEs; the proximity sense
    # ("within 5 miles of") is handled separately by _proximity_preceded().
    "miles": (
        "nautical", "statute", "access road", "access roads", "setback",
        "away from", "buffer", "radius", "distance", "proximity",
        "shoreline", "offshore", "airport", "residence", "boundary of",
        "of a park", "of a wilderness",
    ),
    # acreage that describes a study/planning/management area, not a disturbance cap.
    "acres": (
        "study area", "planning area", "analysis area", "project area",
        "watershed", "field office", "national forest", "allotment",
        "wilderness", "public land", "management area",
    ),
}
# how many characters of left/right context to inspect around each match
_WIN_LEFT, _WIN_RIGHT = 80, 20


# proximity preposition immediately before the number => setback, not a size cap
_PROXIMITY_RX = re.compile(r"(?:within|with[in]*\s+about)\s*$", re.IGNORECASE)


def _disqualified(metric: str, text: str, start: int, end: int) -> bool:
    if metric == "miles" and _PROXIMITY_RX.search(text[max(0, start - 14):start]):
        return True
    terms = DISQUALIFIERS.get(metric)
    if not terms:
        return False
    window = text[max(0, start - _WIN_LEFT): end + _WIN_RIGHT].lower()
    return any(t in window for t in terms)


def _vals(metric: str, text: str) -> list[float]:
    out = []
    text = text or ""
    for m in RX[metric].finditer(text):
        if _disqualified(metric, text, m.start(), m.end()):
            continue
        try:
            v = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        if 0 < v <= CAPS[metric]:
            out.append(v)
    return out


def parse_bounds(text: str) -> dict[str, float | None]:
    """Return the stated limit per metric (max in-range value), or None."""
    return {metric: (round(max(v), 2) if v else None) for metric in RX
            for v in [_vals(metric, text)]}
