"""D6 #38 — lightweight federal agency crosswalk (parent department ↔ sub-agencies).

Used by 07's ADOPT annotation: an agency is only a real "gap" (a target that should adopt a CE) if
neither it, its parent department, nor a department sibling already holds an equivalent CE among the
matched CE's ranks 1–8. All agencies here are REAL federal bureaus (the same token set already in
07's OUR_AGENCY_ALIASES) — none invented.

ANNOTATE-ONLY: 07 uses this to write adopt_targets_net / adopt_targets_gross; it does NOT change the
`verdict`. The authoritative adopt→covered reclassification is owned by the eCFR read (A1/#37).
"""
from __future__ import annotations

# department -> its sub-agency tokens (tokens match 07.ce_agency_tokens / OUR_AGENCY_ALIASES)
DEPT_MEMBERS: dict[str, set[str]] = {
    "DOI": {"BLM", "BOR", "NPS", "USFWS", "BIA", "BOEM"},  # Interior
    "DOE": {"PMA", "NNSA", "WAPA", "BPA", "SWPA", "SEPA"}, # Energy (power-marketing admins + NNSA)
    "USDA": {"USFS"},                                       # Agriculture
    "DOD": {"USACE"},                                       # Defense (Army Corps)
}
# reverse: sub-agency token -> department
DEPT_OF: dict[str, str] = {sub: dept for dept, subs in DEPT_MEMBERS.items() for sub in subs}
# a department also "holds" a CE if the token itself is the department
for _d in DEPT_MEMBERS:
    DEPT_OF.setdefault(_d, _d)


def equivalents(token: str) -> set[str]:
    """The agency token plus its department and department siblings — the set that, if any holds a
    CE, means `token` is not a gap."""
    token = token.upper()
    dept = DEPT_OF.get(token)
    out = {token}
    if dept:
        out.add(dept)
        out |= DEPT_MEMBERS.get(dept, set())
    return out


def is_covered(our_token: str, ce_agency_tokens: set[str]) -> bool:
    """True if our_token (or its dept / a dept sibling) appears among the CE-holding agency tokens."""
    return bool(equivalents(our_token) & {t.upper() for t in ce_agency_tokens})
