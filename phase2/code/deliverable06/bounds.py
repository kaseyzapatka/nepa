"""D6 — shared numeric-bound parser.

Extracts stated quantitative limits (acres, miles, kV, MW, wells) from free text.
Used to (a) parse an existing CE's stated bounds (04/06) and (b) compare them to
our FONSIs' extracted limits for the EXPAND test (07). Same regex family as 03.
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


def _vals(metric: str, text: str) -> list[float]:
    out = []
    for m in RX[metric].findall(text or ""):
        try:
            v = float(m.replace(",", ""))
        except ValueError:
            continue
        if 0 < v <= CAPS[metric]:
            out.append(v)
    return out


def parse_bounds(text: str) -> dict[str, float | None]:
    """Return the stated limit per metric (max in-range value), or None."""
    return {metric: (round(max(v), 2) if v else None) for metric in RX
            for v in [_vals(metric, text)]}
