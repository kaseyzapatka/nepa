"""
Extract date candidates from context packets for D4 timeline extraction.

Applies the full date regex suite from Phase 1/2 lessons, enriches each
candidate with context signals and role pre-labels, and writes the candidate
sidecar table.

Outputs:
    phase2/data/analysis/timeline/timeline_candidates.parquet

Usage:
    python 03_extract_candidates.py [--process CE EA EIS] [--sample-ids path]
"""

import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please activate conda env 'nepa'.")

import argparse
import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PHASE2 = ROOT / "phase2"
ANALYSIS_DIR = PHASE2 / "data" / "analysis"
TIMELINE_DIR = ANALYSIS_DIR / "timeline"
PACKETS_PATH = TIMELINE_DIR / "timeline_context_packets.parquet"
INDEX_PATH = TIMELINE_DIR / "timeline_document_index.parquet"
OUTPUT_PATH = TIMELINE_DIR / "timeline_candidates.parquet"

# The single human-labeling sample for SetFit training (emitted at end of a full run).
# Label assets are INPUTS, so they live under training/ (not output/, which is regenerable).
OUTPUT_DIR = PHASE2 / "output" / "deliverable04"
TRAINING_DIR = PHASE2 / "training" / "deliverable04"
LABELING_SAMPLE_PATH = TRAINING_DIR / "classifier.csv"   # was output/labeling_sample.csv
# Locked selection: the chosen candidate_ids are persisted here so re-runs reproduce the
# exact same candidates (delete this file to re-draw a fresh sample).
LABELING_SAMPLE_IDS_PATH = TRAINING_DIR / "classifier_ids.txt"
LABELING_SAMPLE_SIZE = 300  # total rows, stratified across process_type x candidate_role

RUN_DATE = datetime.now(timezone.utc).date()

# ---------------------------------------------------------------------------
# Date regexes — full suite from plan §4 / Phase 1 lessons
# ---------------------------------------------------------------------------
MONTHS_FULL = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
MONTHS_SHORT = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"

DATE_PATTERNS = [
    (rf"({MONTHS_FULL})\s+(\d{{1,2}}),?\s+(\d{{4}})", "MDY_full"),
    (rf"({MONTHS_SHORT})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})", "MDY_short"),
    # Ordinal day suffixes: "September 3rd, 2020" / "Jan 1st, 2021"
    (rf"({MONTHS_FULL})\s+(\d{{1,2}})(?:st|nd|rd|th),?\s+(\d{{4}})", "MDY_ordinal"),
    (rf"({MONTHS_SHORT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th),?\s+(\d{{4}})", "MDY_short_ordinal"),
    (rf"(\d{{1,2}})\s+({MONTHS_FULL})\s+(\d{{4}})", "DMY_full"),
    # Day-first short month, e.g. "30 Oct 2015" / "30 Oct. 2015". Without this the
    # bare MY_short pattern below captures only "Oct 2015" and drops the day.
    (rf"(\d{{1,2}})\s+({MONTHS_SHORT})\.?\s+(\d{{4}})", "DMY_short"),
    (r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{4})", "numeric_slash"),
    (r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{2})\b", "numeric_slash_2y"),
    (r"(\d{4})-(\d{1,2})-(\d{1,2})", "ISO"),
    (r"(\d{1,2})-(\d{1,2})-(\d{4})", "numeric_dash"),
    (r"(\d{4})\.(\d{2})\.(\d{2})", "digital_sig"),
    # Dotted dates: require a TWO-digit month (01-12), i.e. XX.XX.XX or XX.XX.XXXX.
    # A single-digit leading group is almost always a citation / section / lot number,
    # not a date (e.g. "PMC-EF2a (2.04.02)", "Lots 7.10.15") — those are now excluded.
    (r"(?<![\d.])(0[1-9]|1[0-2])\.(\d{2})\.(\d{2,4})(?![\d.])", "numeric_dot"),
    (rf"({MONTHS_FULL})\s+(\d{{4}})", "MY_full"),
    (rf"({MONTHS_SHORT})\.?\s+(\d{{4}})", "MY_short"),
    # NEPA case number year fallback: "DOI-BLM-WY-P070-2019-0035-CX" → year 2019
    # Region codes vary: 2-char state (WY), 4-char region (ORWA), or mixed (AK-020).
    # Last resort for CEs whose only date signal is the case number header.
    (r"DOI-[A-Z]{2,4}-[A-Z]{2,4}-[A-Z0-9]+-(\d{4})-\d{4,}", "nepa_case_year"),
]

COMPILED_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), ptype) for pattern, ptype in DATE_PATTERNS
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

# ---------------------------------------------------------------------------
# Anchored model context — the text the classifier (04) and labeler read.
# The target date is wrapped in [[ ]] markers so the model knows WHICH date it is
# scoring even when several dates share the window (common in EIS narrative and CE
# signature tables). The window is centered on the date so it survives the encoder's
# token-truncation, expanded to sentence boundaries, then capped per process type.
# Char caps approximate the token targets CE~100 / EA~150 / EIS~200 (≈4 chars/token).
# ---------------------------------------------------------------------------
# Per-process window size (chars) for the candidate-evidence context. Fuller than a
# single sentence, but bounded — never the whole document. EIS gets more (narrative);
# CE less (dense forms). Note: SetFit (all-MiniLM, 256 tokens ≈ ~1024 chars) truncates
# the far tail at inference, but the date is centered so it is always within budget.
MODEL_CONTEXT_CHARS = {"CE": 900, "EA": 1200, "EIS": 1500}
DATE_MARKER_OPEN = "[["
DATE_MARKER_CLOSE = "]]"
_SENT_BOUNDARY = re.compile(r"[.!?]\s+")


def _build_model_context(full_text: str, ds: int, de: int, process_type: str,
                         cap: int | None = None) -> str:
    """
    Build a date-centered window of ~`cap` chars (the candidate evidence), with the
    target date wrapped in [[ ]]. Bounded — it fills toward the cap around the date
    rather than collapsing to the single sentence the date sits in.
    full_text is the normalized packet text; [ds, de) is the date span within it.
    """
    if cap is None:
        cap = MODEL_CONTEXT_CHARS.get(process_type, 1200)
    n = len(full_text)
    if not (0 <= ds < de <= n):
        return full_text[:cap]

    # Center a cap-sized window on the date; clamp to text bounds.
    center = (ds + de) // 2
    s = max(0, center - cap // 2)
    e = min(n, s + cap)
    s = max(0, e - cap)

    rel_s, rel_e = ds - s, de - s
    body = full_text[s:e]
    if 0 <= rel_s < rel_e <= len(body):
        body = (body[:rel_s] + DATE_MARKER_OPEN + body[rel_s:rel_e]
                + DATE_MARKER_CLOSE + body[rel_e:])
    return ("..." if s > 0 else "") + body + ("..." if e < n else "")


def _suppress_contained(matches: list[tuple]) -> list[tuple]:
    """
    Drop matches whose character span is contained within a strictly longer match.
    Keeps the longest date at each position so "30 Oct 2015" wins over "Oct 2015".
    Each match tuple is (start, end, m, ptype, parsed, granularity).
    """
    kept: list[tuple] = []
    for cand in sorted(matches, key=lambda t: (t[1] - t[0]), reverse=True):
        cs, ce = cand[0], cand[1]
        if any(ks <= cs and ce <= ke and (ke - ks) > (ce - cs) for ks, ke, *_ in kept):
            continue
        kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Exclusion patterns (legal/bibliographic/map)
# ---------------------------------------------------------------------------
EXCLUSION_KEYWORDS = [
    # Law/statute references
    "act of 19", "act of 20", "act (19", "act (20",
    "policy act", "preservation act", "conservation act",
    "management act", "protection act", "improvement act", "reform act",
    "recovery act", "species act", "water act", "air act", "lands act",
    # "statute" removed: "Not Established by Statute" is a BLM CE form label, not a citation
    "u.s.c.", " usc ", "public law", "p.l.", "amended in",
    # Bibliographic references
    "accessed on", "retrieved on", "available at",
    "et al.", "et al,", "eds.", "vol.", "pp.", "journal", "doi:",
    "isbn", "issn", "proceedings", "report no.",
    # Expiration / validity dates — not decision dates (Phase 1: EXPIRATION_PATTERNS_STRONG)
    "expiration date", "valid until", "expires on", "expiry", "for a term of",
    "categorical exclusion expires", "re-authoriz",
    # Operational/interim management dates — duration of field actions, not NEPA decisions
    "remain in place until", "protective fencing",
    # URL references
    "http://", "https://",
    # Print-on-recycled boilerplate on document covers (false proxy dates)
    "printed on recycled",
    # OMB / form boilerplate (Phase 1: DECISION_BOILERPLATE_PATTERNS)
    "paperwork reduction", "omb control", "previous editions obsolete",
    "forms mgmt",
    # DOE/NETL form revision date stamps, e.g. "DOE F 540.1 (12/2010)"
    "doe f ", "netl f ",
    # Map preparation dates (complement to REJECT_CUES which has "map created/printed")
    "map prepared",
    # Engineering drawing sheet fields — "DATE:" fields in CAD/GIS sheets attached to EIS
    "drawn by", "checked by", "issued for bid", "issued for construction",
    "drawing number", "sheet number",
    # Comment response table headers — commenter date columns in EA/EIS appendices
    "commenter (name", "commenter name", "commenter organization",
]

# Regex-based exclusions — applied to context window around each date match.
# These catch technical citation formats that simple keyword matching misses.
# (Phase 1: EXCLUSION_PATTERNS and CITATION_PATTERNS)
EXCLUSION_RE = [
    re.compile(r'\b\d+\s*cfr\s*\d+', re.IGNORECASE),      # "40 CFR 1508"
    re.compile(r'\b\d+\s*fr\s*\d+', re.IGNORECASE),        # "80 FR 12345"
    re.compile(r'\b[A-Z][a-z]+\.\s*\d{4}\.'),              # "Smith. 2005."
    re.compile(r'\b[A-Z]{2,}\.\s*\d{4}\.'),                # "EPA. 2010."
]

# ---------------------------------------------------------------------------
# Role cue patterns — positive
# ---------------------------------------------------------------------------

# Clear initiation — strong cues
# Phase 1 patterns ported: designation form, initiator signature, doe initiator signature,
# consultation initiated, initiation of consultation, initiated on, intent to prepare EIS
CLEAR_INITIATION_STRONG = re.compile(
    r"\b("
    r"application\s+received|application\s+submitted|"
    r"submitted\s+(?:a|an|the)?\s*(?:completed\s+)?(?:right|application|permit|plan|request)|"
    r"blm\s+received\s+(?:a|an|the)\s+(?:row\s+)?application|"
    r"blm\s+received|agency\s+received|"
    r"(?:noi|notice\s+of\s+intent)\s+(?:was\s+)?(?:published|issued|submitted)|"
    r"notice\s+of\s+intent\s+to\s+prepare\s+(?:a|an|the|this)?\s*(?:supplemental\s+|revised\s+)?(?:environmental\s+impact\s+statement|eis)\s+(?:was\s+)?(?:published|issued|submitted|filed)|"
    r"initiated\s+the\s+scoping\s+process\s+by\s+publishing|"
    r"scoping\s+period\s+(?:began|started|initiated|opened)|"
    r"notice\s+of\s+intent\s+was\s+published|"
    r"(?:federal\s+register).*notice\s+of\s+intent|"
    r"environmental\s+review\s+(?:began|initiated|started)|"
    r"external\s+scoping\s+(?:was\s+)?(?:conducted|initiated|begun)|"
    r"posted\s+(?:on|to)\s+(?:the\s+)?(?:on[-\s]?line\s+)?nepa\s+register|"
    r"doe\s+initiator\s+signature|initiator\s+signature|"
    r"consultation\s+initiated|initiation\s+of\s+consultation|"
    r"initiated\s+on|"
    r"designation\s+form|"
    r"intent\s+to\s+prepare\s+(?:an?\s+)?environmental\s+impact\s+statement|"
    r"submitted\s+(?:a\s+)?(?:completed\s+)?right[-\s]of[-\s]way\s+application"
    r")\b",
    re.IGNORECASE,
)

# Clear initiation — medium cues
# Phase 1 additions: renewal application received, project proposed, nepa process started,
# nepa review began, request received, review was initiated
CLEAR_INITIATION_MED = re.compile(
    r"\b("
    r"(?:application|request|permit|apd|plan\s+of\s+development|pod|right[-\s]of[-\s]way|"
    r"license\s+application|row)\s+(?:date|filed|submitted|received)|"
    r"(?:date\s+(?:of\s+)?)?(?:application|request|submission|filing|receipt)|"
    r"doe\s+initiator|nepa\s+initiator|action\s+initiating|nepa\s+initiation|"
    r"renewal\s+application\s+received|"
    r"project\s+proposed|"
    r"nepa\s+(?:process|review)\s+(?:started|began|initiated)|"
    r"request\s+received|"
    r"review\s+was\s+initiated|"
    r"(?:distribution|review)\s+(?:was\s+)?initiated|"
    r"nepa\s+clause\s+prepared\s+by|"   # DOE EECBG form: "NEPA clause prepared by OCC [person date]"
    r"prepared\s+by\s+occ|"             # Office of Chief Counsel preparation date
    r"clause\s+prepared\s+by|"
    # Phase 1: INITIATION_PATTERNS additions not previously ported
    r"deemed\s+the\s+application\s+complete|"    # formal ROW/permit start of NEPA review
    r"amended\s+and\s+re[-\s]submitted|"         # resubmission = initiation
    r"re[-\s]submitted\s+(?:a|the)\s+application|"
    r"30[-\s]day\s+comment\s+period|"            # initiation-adjacent
    r"date\s+(?:created|prepared)|document\s+creation|"  # proxy initiation from document metadata
    r"drafted|"
    r"comment\s+period\s+(?:was|ran|began|started|opened|ended|closed)"  # scoping/comment period dates
    r")\b",
    re.IGNORECASE,
)

CE_INITIATOR_ROLE = re.compile(
    r"\b(doe\s+initiator|nepa\s+initiator|action\s+initiating\s+office|"
    r"project\s+(?:initiator|proponent|sponsor))\b",
    re.IGNORECASE,
)

# EA/EIS scoping & NOI initiation cues. The SetFit classifier already scores these dates as
# initiation (~0.86), but the regex roles them `unknown` because the exact phrasing isn't in the
# clear-init patterns. Keying on the *phrase* (not the probability) recovers the real scoping/NOI
# inits while excluding the comment-status / future / FONSI false positives. (Validated: matches the
# real scoping/NOI dates, rejects "no comments received as of", "Final EIS will be published", etc.)
SCOPING_NOI_INIT = re.compile(
    r"scoping\s+(was\s+|period\s+)?(conducted|held|beg[au]n|initiated|opened|started|between|from)"
    r"|public\s+scoping\s+between"
    r"|scoping\s+(document|notice|letters?)[^.]{0,30}(distributed|sent|mailed|issued|publish)"
    r"|notice\s+of\s+intent[^.]{0,40}(publish|issued|prepare)"
    r"|\bnoi\b[^.]{0,25}(publish|issued)"
    r"|uploaded\s+to[^.]{0,20}eplanning",
    re.IGNORECASE,
)

# EA/EIS application & FERC pre-filing initiation cues. Same idea as SCOPING_NOI_INIT, different
# vocabulary: a formal application filing or entry into FERC's pre-filing process is an initiation.
# "applied for" is also Fix B for CE; here it is extended to EA/EIS along with application-filing and
# pre-filing phrasings. Anchored to the date's clause like the scoping cue (no FP on "authorized on").
APPLICATION_PREFILING_INIT = re.compile(
    r"\bapplied\s+for\b"
    r"|filed\s+(a|an|the)?\s*application"
    r"|application\s+(was\s+)?(filed|received|submitted)"
    r"|submitted\s+(a|an|the)?\s*(application|proposal|request)"
    r"|entered\s+(the\s+)?pre[-\s]?filing|requested\s+to\s+use\s+the\s+pre[-\s]?filing"
    r"|pre[-\s]?filing\s+(process|request|period)|pre[-\s]?application\s+(process|request|filing)",
    re.IGNORECASE,
)

# Clear decision — strong cues
# Phase 1 additions: digitally signed by, /s/ signature notation, YYYY.MM.DD timestamp,
# NCO determination, authority and approval, decision memo(randum), ce determination date,
# selection of alternative (EIS), decision to implement, joint record of decision,
# field office manager determination, nepa compliance officer (standalone), concur+NCO
CLEAR_DECISION_STRONG = re.compile(
    r"\b("
    r"fonsi\s+(?:was\s+)?(?:signed|issued|approved|dated)|"
    r"finding\s+of\s+no\s+significant\s+impact\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"record\s+of\s+decision[,\s]+(?:was\s+)?(?:signed|issued|dated)|"
    r"joint\s+record\s+of\s+decision|"
    r"rod\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"(?:signed|issued)\s+(?:the\s+)?(?:rod|record\s+of\s+decision|fonsi|finding\s+of\s+no)|"
    r"decision\s+(?:record|notice|memo(?:randum)?)\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"decision\s+memo(?:randum)?|"
    r"categorical\s+exclusion\s+(?:determination|approved|signed)|"
    r"(?:ce|cx)\s+(?:determination|approved|signed)|"
    r"ce\s+determination\s+date|cx\s+determination\s+date|"
    r"(?:date\s+)?signed\s+(?:by|on).*(?:field\s+manager|district\s+manager|authorizing\s+official)|"
    r"signature\s+of\s+(?:authorized|authorizing|approving)\s+officer|"
    r"field\s+office\s+manager\s+determination|"
    r"nepa\s+compliance\s+officer.*(?:date|concur)|"
    r"concur.*nepa\s+compliance\s+officer|"
    r"nepa\s+compliance\s+officer|"
    r"NCO\s+determination|"
    r"authority\s+and\s+approval|determination\s+and\s+approval|"
    r"date\s+of\s+(?:decision|approval|determination)|"
    r"digitally\s+signed\s+by|"
    r"(?:selected|selection\s+of)\s+(?:the\s+)?(?:preferred\s+)?alternative|"
    r"decision\s+to\s+implement"
    r")\b"
    r"|/s/\s*\w+"           # digital signature notation (no word boundary needed)
    r"|\b\d{4}\.\d{2}\.\d{2}\b",  # YYYY.MM.DD timestamp in digital signatures
    re.IGNORECASE,
)

# Decision-keyword-only subset of CLEAR_DECISION_STRONG — excludes the generic /s/ and
# YYYY.MM.DD branches. Used to verify that a CLEAR_DECISION_STRONG hit was driven by
# actual decision language, not just a specialist's /s/ signature on a face sheet.
CLEAR_DECISION_KEYWORDS_RE = re.compile(
    r"\b("
    r"fonsi\s+(?:was\s+)?(?:signed|issued|approved|dated)|"
    r"finding\s+of\s+no\s+significant\s+impact\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"record\s+of\s+decision[,\s]+(?:was\s+)?(?:signed|issued|dated)|"
    r"joint\s+record\s+of\s+decision|"
    r"rod\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"(?:signed|issued)\s+(?:the\s+)?(?:rod|record\s+of\s+decision|fonsi|finding\s+of\s+no)|"
    r"decision\s+(?:record|notice|memo(?:randum)?)\s+(?:was\s+)?(?:signed|issued|dated)|"
    r"decision\s+memo(?:randum)?|"
    r"categorical\s+exclusion\s+(?:determination|approved|signed)|"
    r"(?:ce|cx)\s+(?:determination|approved|signed)|"
    r"ce\s+determination\s+date|cx\s+determination\s+date|"
    r"(?:date\s+)?signed\s+(?:by|on).*(?:field\s+manager|district\s+manager|authorizing\s+official)|"
    r"signature\s+of\s+(?:authorized|authorizing|approving)\s+officer|"
    r"field\s+office\s+manager\s+determination|"
    r"nepa\s+compliance\s+officer.*(?:date|concur)|"
    r"concur.*nepa\s+compliance\s+officer|"
    r"nepa\s+compliance\s+officer|"
    r"NCO\s+determination|"
    r"authority\s+and\s+approval|determination\s+and\s+approval|"
    r"date\s+of\s+(?:decision|approval|determination)|"
    r"digitally\s+signed\s+by|"
    r"(?:selected|selection\s+of)\s+(?:the\s+)?(?:preferred\s+)?alternative|"
    r"decision\s+to\s+implement"
    r")\b",
    re.IGNORECASE,
)

# EA-only (Phase C): approving-authority titles. A BLM/agency FONSI signature block lists
# recommenders/reviewers (specialists) AND the approving official together, so the generic
# specialist-sheet disambiguation wrongly downgrades the whole block — including the approving
# signature, which IS the decision — to `review`. When one of these decision-authority titles is
# present, an EA signature date is treated as a real decision. EA-scoped so CE/EIS specialist-sheet
# handling is untouched.
EA_DECISION_AUTHORITY_RE = re.compile(
    r"\b(?:"
    r"field\s+(?:office\s+)?manager|district\s+manager|"
    r"assistant\s+field\s+manager|acting\s+field\s+manager|"
    r"authorizing\s+official|approving\s+official|"
    r"(?:deputy\s+|associate\s+)?state\s+director|area\s+manager|"
    r"district\s+ranger|forest\s+supervisor"
    r")\b",
    re.IGNORECASE,
)

# Proxy decision: FEIS/EA publication dates serve as upper bound for ROD/FONSI
PROXY_DECISION_RE = re.compile(
    r"\b("
    r"(?:final\s+(?:environmental\s+impact\s+statement|eis)|feis)\s+(?:was\s+)?(?:published|released|issued|filed|available|signed)|"
    r"notice\s+of\s+availability\s+(?:for|of)\s+(?:the\s+)?(?:feis|final\s+eis|final\s+environmental)|"
    r"(?:published|issued|released)\s+(?:an?\s+)?(?:noa|notice\s+of\s+availability)\s+(?:for|of)\s+(?:the\s+)?(?:feis|final)|"
    r"(?:published|issued|released)\s+(?:the\s+)?(?:final\s+eis|feis|final\s+ea)|"
    r"(?:final\s+)?(?:ea|environmental\s+assessment)\s+(?:was\s+)?(?:signed|issued|approved|completed|finalized)|"
    r"(?:approved|signed)\s+(?:the\s+)?(?:final\s+)?(?:ea|environmental\s+assessment)"
    r")\b",
    re.IGNORECASE,
)

# Clear decision — medium cues
# Phase 1 additions: date determined, field office manager (standalone), final approval
CLEAR_DECISION_MED = re.compile(
    r"\b("
    r"(?:approved|signed|authorized|determined)\s+(?:by|on|this)|"
    r"(?:field\s+manager|district\s+manager|field\s+office\s+manager|"
    r"assistant\s+field\s+manager|authorizing\s+official|"
    r"nepa\s+compliance\s+officer|certifying\s+official)\s+(?:signature|date|signed)|"
    r"decision\s+date|date\s+approved|date\s+signed|"
    r"date\s+determined|"
    r"approval\s+date|date\s+of\s+approval|"
    r"final\s+approval|"
    # Phase 1: standalone ROD/FONSI language — catches cover pages and references where
    # the verb (signed/issued/dated) is absent. Lower confidence than the strong pattern.
    r"(?:final\s+)?record\s+of\s+decision|"
    r"finding\s+of\s+no\s+significant\s+impact|"
    r"\bfonsi\b|"
    r"\brod\s+(?:for|of|dated|for\s+the)"
    r")\b",
    re.IGNORECASE,
)

# Review / specialist signatures (not decision dates)
# Phase 1 additions: realty specialist, recreation planner/specialist, natural resource
# specialist, environmental coordinator, botanist, project officer, MOA, SME roles,
# environmental clearance memorandum, reviewer/initials table headers, nepa review completed
REVIEW_CUES = re.compile(
    r"\b("
    r"environmental\s+specialist|wildlife\s+biologist|archaeologist|archeologist|"
    r"cultural\s+resource(?:s)?\s+specialist|shpo|section\s+106|"
    r"review\s+completed|interim\s+review|phase\s+approval|"
    r"concurrence\s+(?:received|date)|coordination\s+date|"
    r"realty\s+specialist|"
    r"recreation\s+(?:planner|specialist)|outdoor\s+recreation\s+planner|"
    r"natural\s+resource\s+specialist|"
    r"environmental\s+coordinator|planning\s+(?:and|&)\s+environmental\s+coordinator|"
    r"botanist|"
    r"project\s+officer|"
    r"fisheries(?:/wildlife)?\s+biologist|"
    r"environmental\s+clearance\s+memorandum|"
    r"yes\s+no\s+reviewer|reviewer.*title.*initials|"
    r"initial\s+and\s+date|initials?\s*(?:&|and)\s*date|"
    r"nepa\s+review\s+completed|"
    r"\bmoa\b|memorandum\s+of\s+agreement|"
    r"subject\s+matter\s+expert|NEPA-SME|NEPA\s+SME"
    r")\b",
    re.IGNORECASE,
)

# Historical / legal — GENUINE past-event cues only.
# Removed bare lup / rmp / "resource management plan" / "land use plan" / bare "historical":
# they fire on BLM CE form boilerplate ("LUP Conformance & CX Confirmation" is on every BLM
# CE) and checklist labels ("Cultural or Historical [ ]"), which — because HISTORICAL_CUES is
# checked before CLEAR_DECISION_STRONG — was stamping real Field-Manager/NEPA-officer signature
# dates as historical. Audit (2026-06-02): only ~16% of historical candidates had genuine
# past-event phrasing; ~13% were pure form-boilerplate contamination. Keep only event cues
# that describe a PRIOR dated action.
HISTORICAL_CUES = re.compile(
    r"\b("
    r"prior\s+rod|previous\s+(?:eis|ea|rod|decision)|prior\s+(?:eis|ea|decision)|"
    r"old\s+(?:lease|plan)|"
    r"communication\s+site\s+established|lease\s+issued\s+on|"
    r"(?:was\s+)?granted\s+a\s+(?:row|right[-\s]of[-\s]way)|"
    r"(?:row|right[-\s]of[-\s]way)\s+(?:was\s+)?(?:granted|issued)\s+on|"
    r"previously\s+(?:authorized|approved|granted|issued|prepared)"
    r")\b",
    re.IGNORECASE,
)

# Sub-process consultation initiation — "tribal/cultural consultation was initiated on
# [date]" describes a NHPA Section 106 or ESA sub-process start, not NEPA initiation.
# Must fire BEFORE CLEAR_INITIATION_STRONG so "initiated on" doesn't misclassify these.
SUBCONSULTATION_INITIATED_RE = re.compile(
    r"\b(tribal|native\s+american|cultural\s+resource|section\s+106)\b"
    r".{0,200}\b(consultation|coordination)\b.{0,200}"
    r"\b(initiated|was\s+initiated|began|started)\b",
    re.IGNORECASE | re.DOTALL,
)

# Hard reject
REJECT_CUES = re.compile(
    r"\b("
    r"omb\s+(?:control|approval)|form\s+approved|prepared\s+by|"
    r"downloaded|accessed\s+on|retrieved\s+on|revision\s+date|revised\s+\d{4}|"
    r"map\s+(?:date|created|printed|prepared)|figure\s+\d+|table\s+\d+|"
    # Phase 1: INITIATION_EXCLUSION_PATTERNS — contexts that produce false initiation dates
    r"program\s+specific\s+guidance|"
    r"prepared\s+in\s+accordance\s+with.*guidance|"
    r"resource\s+management\s+plan|\brmp\b|land\s+use\s+plan|"
    r"conformance\s+with\s+the\s+applicable\s+lup|"
    r"plan\s+maintenance\s+action|"
    r"specialist\s+signature|"
    # Engineering drawing sheet metadata — CAD/GIS drawing title blocks attached to EIS
    r"(?:drawn|checked|approved)\s+by\s*:|issued\s+for\s+(?:bid|construction)|"
    r"drawing\s+(?:number|no\.?)\s*:|revision\s+(?:number|no\.?)?\s*:|"
    # Comment response table dates — appended letter/commenter tables in EA/EIS appendices
    r"commenter\s+(?:name|organization)|"
    r"(?:name|organization)[,;]\s+date[,;]\s+comment"
    r")\b",
    re.IGNORECASE,
)

# Signature/bottom-of-document signal
SIGNATURE_BLOCK_RE = re.compile(
    r"\b("
    r"signature|nepa\s+compliance\s+officer|authorizing\s+official|field\s+manager|"
    r"district\s+manager|certifying\s+official|approved\s+by|concurrence"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_match(match: re.Match, ptype: str) -> tuple[datetime | None, str]:
    """Return (datetime_or_None, granularity)."""
    try:
        g = match.groups()
        if ptype in ("MDY_full", "MDY_short", "MDY_ordinal", "MDY_short_ordinal"):
            month = MONTH_MAP.get(g[0].lower())
            return datetime(int(g[2]), month, int(g[1])), "day"
        if ptype in ("DMY_full", "DMY_short"):
            month = MONTH_MAP.get(g[1].lower())
            return datetime(int(g[2]), month, int(g[0])), "day"
        if ptype == "numeric_slash":
            return datetime(int(g[2]), int(g[0]), int(g[1])), "day"
        if ptype == "numeric_slash_2y":
            yr = int(g[2])
            yr = 2000 + yr if yr <= 30 else 1900 + yr
            return datetime(yr, int(g[0]), int(g[1])), "day"
        if ptype == "ISO":
            return datetime(int(g[0]), int(g[1]), int(g[2])), "day"
        if ptype == "numeric_dash":
            return datetime(int(g[2]), int(g[0]), int(g[1])), "day"
        if ptype == "digital_sig":
            return datetime(int(g[0]), int(g[1]), int(g[2])), "day"
        if ptype == "numeric_dot":
            yr = int(g[2])
            if yr < 100:
                yr = 2000 + yr if yr <= 30 else 1900 + yr
            if not (1970 <= yr <= 2035):
                return None, "unknown"
            return datetime(yr, int(g[0]), int(g[1])), "day"
        if ptype in ("MY_full", "MY_short"):
            month = MONTH_MAP.get(g[0].lower())
            return datetime(int(g[1]), month, 1), "month"
        if ptype == "nepa_case_year":
            yr = int(g[0])
            if not (1970 <= yr <= 2035):
                return None, "unknown"
            return datetime(yr, 7, 1), "year"
    except (ValueError, TypeError, KeyError):
        return None, "unknown"
    return None, "unknown"


def _should_reject_date(
    parsed_date: datetime,
    context: str,
    process_type: str,
    source_tier: str,
    date_span: tuple[int, int] | None = None,
) -> tuple[bool, str]:
    """
    Return (reject, reason) applying the plan §4 exclusion rules.
    Does not reject month-year candidates; those are granularity=month.

    When ``date_span`` (the date match's char offsets within ``context``) is supplied, the
    keyword/citation exclusions are scoped to a window around the date: CE ±60, EIS ±120
    (dense FEIS pages — a citation elsewhere must not kill a real publication/ROD date). For
    EIS the ``REJECT_CUES`` historical scan is also windowed. EA keeps the whole-block scan
    (window dict excludes it) so EA behavior is unchanged.
    """
    # Future date check
    if parsed_date.date() > RUN_DATE:
        return True, "future_date"

    # Process-specific year cutoffs (plan §4)
    if process_type in ("CE", "EA") and parsed_date.year < 1970:
        return True, "pre_1970_hard_reject"
    if process_type == "EIS" and parsed_date.year < 1970:
        # Soft reject: allow only with strong evidence (handled in scoring/selection)
        return True, "pre_1970_eis_reject"

    # Window the citation/keyword exclusions to the immediate neighborhood of the date.
    # CE ±60 (existing), EIS ±120 (Phase 3). EA falls through to whole-block (unchanged).
    _EXCL_WINDOW = {"CE": 60, "EIS": 120}
    if date_span is not None and process_type in _EXCL_WINDOW:
        ds, de = date_span
        w = _EXCL_WINDOW[process_type]
        excl_text = context[max(0, ds - w):de + w]
    else:
        excl_text = context
    excl_lower = excl_text.lower()

    # Legal/statutory citation exclusions
    for kw in EXCLUSION_KEYWORDS:
        if kw in excl_lower:
            return True, f"exclusion_keyword:{kw}"

    # Regex-based exclusions (CFR/FR citations, author-year bibliographic patterns)
    for pat in EXCLUSION_RE:
        if pat.search(excl_text):
            return True, "exclusion_regex"

    # Metadata-only sources bypass text-based exclusions
    if source_tier == "metadata":
        return False, ""

    # Reject/historical cues. Windowed for EIS (a historical sentence elsewhere on a dense
    # FEIS page must not reject an unrelated publication/signature date); whole-block for CE/EA.
    reject_scan = excl_text if (process_type == "EIS" and date_span is not None) else context
    if REJECT_CUES.search(reject_scan):
        return True, "reject_cue"

    return False, ""


def _prelabel_role(
    context: str,
    source_tier: str,
    retrieval_reason: str | None,
    ptype: str,
    document_type_category: str | None = None,
    process_type: str | None = None,
) -> tuple[str, float, list[str], list[str]]:
    """
    Return (candidate_role, role_confidence_float, positive_cue_flags, negative_cue_flags).
    Role confidence 0-5 maps to 5=high, 3=medium, 1=low.
    """
    pos_cues: list[str] = []
    neg_cues: list[str] = []

    # Agency register Tier A — authoritative government source, highest confidence
    if source_tier == "metadata" and "blm_register_decision" in (retrieval_reason or ""):
        return "clear_decision", 5.0, ["blm_register_tier_a"], []
    if source_tier == "metadata" and "blm_register_initiation" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["blm_register_tier_a"], []
    if source_tier == "metadata" and "doe_register_decision" in (retrieval_reason or ""):
        return "clear_decision", 5.0, ["doe_register_tier_a"], []
    if source_tier == "metadata" and "doe_register_initiation" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["doe_register_tier_a"], []
    if source_tier == "metadata" and "doe_cx_register_decision" in (retrieval_reason or ""):
        return "clear_decision", 5.0, ["doe_cx_register_tier_a"], []

    # Filename Tier A — date from decision document filename (e.g. fonsi-ea-...-2008-04-14.pdf)
    # Score 3.0: below authoritative register sources (5.0) but above text extraction.
    if source_tier == "metadata" and "filename_date_decision_doc" in (retrieval_reason or ""):
        return "clear_decision", 3.0, ["filename_date_tier_a"], []

    # Metadata / FR NOI
    if source_tier == "metadata" and "noi" in (retrieval_reason or ""):
        return "clear_initiation", 5.0, ["fr_noi_metadata"], []

    if source_tier in ("file_name", "title"):
        if document_type_category in ("final", "decision"):
            return "proxy_decision", 1.5, ["filename_or_title"], []
        return "proxy_initiation", 1.5, ["filename_or_title"], []

    # NEPA case number years are always last-resort fallback — never promote via text cues.
    # "Field Manager Date DOI-BLM-...-2015-..." has decision language but the date came from
    # the case number, so it must stay a low-confidence proxy regardless of surrounding text.
    if ptype == "nepa_case_year":
        return "proxy_decision", 0.5, ["nepa_case_number_year"], []

    # Historical cues: checked BEFORE decision cues because a past ROW grant / prior EIS
    # reference is definitionally non-current and can never be the active decision date,
    # even when the surrounding block also contains decision language.
    if HISTORICAL_CUES.search(context):
        neg_cues.append("historical_cue")
        return "historical", 0.0, pos_cues, neg_cues

    # Check strong decision cues
    if CLEAR_DECISION_STRONG.search(context):
        # Disambiguate specialist face sheets: the /s/ branch of CLEAR_DECISION_STRONG
        # fires on any signature, including multi-specialist review sheets. When the
        # decision signal came only from /s/ or YYYY.MM.DD (not decision keywords), treat
        # as a specialist sheet if: (a) 3+ /s/ patterns appear (multi-reviewer grid), or
        # (b) REVIEW_CUES confirms a specialist role is present.
        if not CLEAR_DECISION_KEYWORDS_RE.search(context):
            slash_s_count = len(re.findall(r"/s/", context, re.IGNORECASE))
            if slash_s_count >= 3 or REVIEW_CUES.search(context):
                # EA-only escape: a FONSI signature block lists specialists/recommenders AND the
                # approving official together; the approving-authority signature IS the decision.
                # When a decision-authority title is present, keep an EA signature date as a decision
                # rather than downgrading the whole block to review. CE/EIS handling is unchanged.
                if process_type == "EA" and EA_DECISION_AUTHORITY_RE.search(context):
                    pos_cues.append("ea_decision_authority")
                    return "clear_decision", 5.0, pos_cues, neg_cues
                neg_cues.append("specialist_sig_sheet")
                return "review", 2.0, pos_cues, neg_cues
        pos_cues.append("decision_strong")
        return "clear_decision", 5.0, pos_cues, neg_cues

    # Sub-process consultation: "tribal/cultural/Section 106 consultation was initiated"
    # is a NHPA/ESA subprocess start, not the NEPA process initiation itself.
    if SUBCONSULTATION_INITIATED_RE.search(context):
        neg_cues.append("subconsultation_initiated")
        return "historical", 0.0, pos_cues, neg_cues

    if CLEAR_INITIATION_STRONG.search(context):
        pos_cues.append("initiation_strong")
        return "clear_initiation", 5.0, pos_cues, neg_cues

    if REJECT_CUES.search(context):
        neg_cues.append("reject_cue")
        return "reject", 0.0, pos_cues, neg_cues

    if CLEAR_DECISION_MED.search(context):
        pos_cues.append("decision_med")
        return "clear_decision", 3.0, pos_cues, neg_cues

    if CLEAR_INITIATION_MED.search(context):
        pos_cues.append("initiation_med")
        return "clear_initiation", 3.0, pos_cues, neg_cues

    if REVIEW_CUES.search(context):
        pos_cues.append("review_cue")
        return "review", 2.0, pos_cues, neg_cues

    # Proxy decision from FEIS/EA publication language
    if PROXY_DECISION_RE.search(context):
        pos_cues.append("feis_pub_proxy")
        return "proxy_decision", 2.5, pos_cues, neg_cues

    # Month-year dates:
    #   decision doc, short context (≤18 words) → cover/signature date → clear_decision
    #   decision doc, long context → body-text reference → proxy_initiation (may be app date)
    #   final doc, short context → FEIS pub date → proxy_decision (FEIS ≠ ROD)
    #   everything else → proxy_initiation
    if ptype in ("MY_full", "MY_short"):
        if document_type_category == "decision" and len(context.split()) <= 18:
            # Short month-year in a decision doc: likely a cover/signature month, but no
            # explicit cue. Holding category for the classifier, not an auto clear_decision.
            pos_cues.append("body_text")
            return "body_text", 2.0, pos_cues, neg_cues
        if document_type_category == "final" and len(context.split()) <= 18:
            return "proxy_decision", 1.5, pos_cues, neg_cues
        return "proxy_initiation", 1.0, pos_cues, neg_cues

    # numeric_dot: require signature/form context for CE, else unknown
    if ptype == "numeric_dot":
        if SIGNATURE_BLOCK_RE.search(context):
            return "clear_decision", 3.0, ["signature_block"], neg_cues
        return "unknown", 1.0, pos_cues, neg_cues

    # Dates in decision-labeled documents without explicit text cues. There is NO
    # role evidence in the text — only the document type suggests a decision context.
    # Label these "body_text": a holding category for dates the regex cannot resolve.
    # The classifier (script 04) is responsible for promoting/demoting body_text
    # candidates; selection (script 05) only falls back to them as a last resort.
    # Guard: comment-period language in a FONSI/ROD is a public comment date, not a decision.
    if document_type_category == "decision":
        # Comment/scoping period dates inside a FONSI/CE are public-process dates,
        # not decision dates — regardless of whether they match a specific verb pattern.
        ctx_lower = context.lower()
        if "comment period" in ctx_lower or "scoping period" in ctx_lower or "notice of proposed action" in ctx_lower:
            neg_cues.append("comment_period_in_decision_doc")
            return "proxy_initiation", 1.0, pos_cues, neg_cues
        pos_cues.append("body_text")
        return "body_text", 2.0, pos_cues, neg_cues

    return "unknown", 1.5, pos_cues, neg_cues


def extract_candidates_from_packet(packet: dict) -> list[dict]:
    """
    Extract date candidates from a single context packet row.
    Returns a list of candidate dicts.
    """
    context = packet.get("context_text") or ""
    if not context:
        return []

    context_clean = " ".join(context.split())
    source_tier = packet.get("source_tier", "page_slice")
    process_type = packet.get("process_type", "")
    retrieval_reason = packet.get("retrieval_reason", "")
    document_type_clean = packet.get("document_type_clean")
    heading = packet.get("heading_title") or ""

    # Metadata tier: the whole context text IS the date value
    if source_tier == "metadata":
        # Try to extract date from the metadata string
        for compiled, ptype in COMPILED_PATTERNS:
            for m in compiled.finditer(context_clean):
                parsed, granularity = _parse_match(m, ptype)
                if parsed is None:
                    continue
                reject, reason = _should_reject_date(
                    parsed, context_clean, process_type, source_tier
                )
                if reject:
                    continue
                role, conf, pos_cues, neg_cues = _prelabel_role(
                    context_clean, source_tier, retrieval_reason, ptype,
                    document_type_category=packet.get("document_type_category"),
                    process_type=process_type,
                )
                candidate_id = hashlib.sha1(
                    f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{parsed.date()}|{context_clean[:50]}".encode()
                ).hexdigest()[:20]
                return [{
                    "candidate_id": candidate_id,
                    "project_id": packet["project_id"],
                    "process_type": process_type,
                    "document_id": packet.get("document_id"),
                    "page_number": packet.get("page_start"),
                    "section_id": packet.get("section_id"),
                    "context_packet_id": packet.get("context_packet_id"),
                    "source_tier": source_tier,
                    "retrieval_tier": packet.get("retrieval_tier"),
                    "candidate_source_type": "noi_notice" if "noi" in retrieval_reason else "metadata",
                    "raw_date_text": m.group(0),
                    "parsed_date": parsed.date().isoformat(),
                    "date_granularity": granularity,
                    "context_text": context_clean,
                    "context_cleaned": context_clean,
                    # Metadata candidates are Tier A (exempt from the classifier); keep the
                    # column present for schema consistency.
                    "model_context": context_clean,
                    "document_title": packet.get("document_title"),
                    "file_name": None,
                    "document_type_clean": document_type_clean,
                    "document_type_category": packet.get("document_type_category"),
                    "main_document": packet.get("main_document"),
                    "heading_title": heading,
                    "parent_heading_title": packet.get("parent_heading_title"),
                    "position_pct": None,
                    "section_position_pct": None,
                    "candidate_role": role,
                    "role_confidence": ("high" if conf >= 4 else "medium" if conf >= 2.5 else "low"),
                    "role_confidence_score": conf,
                    "rule_ids": "metadata_tier_a",
                    "positive_cue_flags": "|".join(pos_cues),
                    "negative_cue_flags": "|".join(neg_cues),
                    "classifier_label": None,
                    "classifier_score": None,
                    "api_label": None,
                    "api_call_id": None,
                    "context_window_hash": hashlib.sha1(" ".join(context_clean.split()).encode()).hexdigest()[:16],
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }]
        return []

    # Text-based tiers: extract from sentence contexts
    candidates: list[dict] = []
    seen_keys: set[str] = set()

    # Split context into sentences
    sentences = re.split(r"(?<=[.!?])\s+|\n", context_clean)
    # Also scan the whole context to catch multi-sentence date cues
    scan_blocks = sentences + [context_clean]

    for block in scan_blocks:
        block = block.strip()
        if not block:
            continue
        # Collect all date matches in this block across patterns, then drop any match
        # whose span sits inside a longer match (e.g. "Oct 2015" inside "30 Oct 2015")
        # so the most complete date wins instead of being artificially subset.
        raw_matches: list[tuple] = []
        for compiled, ptype in COMPILED_PATTERNS:
            for m in compiled.finditer(block):
                parsed, granularity = _parse_match(m, ptype)
                if parsed is None:
                    continue
                raw_matches.append((m.start(), m.end(), m, ptype, parsed, granularity))

        for _ms, _me, m, ptype, parsed, granularity in _suppress_contained(raw_matches):
                reject, _ = _should_reject_date(
                    parsed, block, process_type, source_tier, (_ms, _me)
                )
                if reject:
                    continue

                date_str = parsed.date().isoformat()
                block_norm = " ".join(block.split())[:100]
                dedup_key = f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{date_str}|{block_norm}"
                if dedup_key in seen_keys:
                    continue
                seen_keys.add(dedup_key)

                role, conf, pos_cues, neg_cues = _prelabel_role(
                    block, source_tier, retrieval_reason, ptype,
                    document_type_category=packet.get("document_type_category"),
                    process_type=process_type,
                )

                # Tag a "Date Determined: <date>" date (DOE CX form). Stays clear_decision
                # (so a lone Date-Determined is the decision), but 05 can recover it as a CE
                # initiation when a separate, later decision (e.g. NCO signature) also exists.
                if re.search(r"date\s+determined\s*:?\s*$",
                             block[max(0, _ms - 40):_ms], re.IGNORECASE):
                    pos_cues = pos_cues + ["date_determined"]

                # DOE Initiator signature date = CE initiation. A date in the "DOE Initiator
                # Signature" block (which precedes the "NEPA Compliance Officer" block) marks
                # the program office initiating the review — relabel as initiation even though
                # the surrounding CX boilerplate reads as a decision. Detection: the nearest
                # preceding "initiator" label is closer than any preceding "compliance officer".
                _pre = block[:_ms].lower()
                _init_pos = max(_pre.rfind("initiator signature"), _pre.rfind("doe initiator"))
                _nco_pos = max(_pre.rfind("compliance officer"), _pre.rfind("nepa compliance"))
                if _init_pos >= 0 and _init_pos > _nco_pos:
                    role = "clear_initiation"
                    conf = 5.0
                    pos_cues = [c for c in pos_cues
                                if c not in ("decision_strong", "decision_med", "doc_type_decision")]
                    pos_cues = pos_cues + ["doe_initiator_signature"]

                # CE application date = initiation. The dominant structure is
                # "On <date>, <applicant> applied for ..." — the date is immediately FOLLOWED by
                # "applied for". Anchor to the ~70 chars AFTER the date so it won't grab a prior
                # "authorized to ... on <date>" grant. Cut the look-ahead at the first sentence
                # boundary so "applied for" must be in the SAME clause as the date (drops cases like
                # "...expiration, May 6, 2018. ... applied for" tagging the wrong date). CE only;
                # never steals an existing decision role.
                _af_win = re.split(r"\.\s|\bCOC-|\bOn\s+[A-Z0-9]", block[_me:_me + 70])[0]
                if (process_type == "CE"
                        and role not in ("clear_decision", "proxy_decision")
                        and re.search(r"\bapplied\s+for\b", _af_win, re.IGNORECASE)):
                    role = "clear_initiation"
                    conf = 5.0
                    pos_cues = [c for c in pos_cues
                                if c not in ("decision_strong", "decision_med", "doc_type_decision")]
                    pos_cues = pos_cues + ["applied_for_application"]

                # EA/EIS scoping & NOI date = initiation. The scoping/NOI phrase can sit just before
                # or just after the date ("Scoping was conducted from <date>", "<date>, … uploaded to
                # ePlanning"), so check a tight window on both sides, cut at sentence boundaries.
                # Never steals an existing decision role; chronology (init<decision) is enforced in 05.
                if process_type in ("EA", "EIS") and role not in ("clear_decision", "proxy_decision"):
                    _pre = re.split(r"\.\s", block[max(0, _ms - 80):_ms])[-1]
                    _post = re.split(r"\.\s", block[_me:_me + 60])[0]
                    _init_cue = None
                    if SCOPING_NOI_INIT.search(_pre) or SCOPING_NOI_INIT.search(_post):
                        _init_cue = "scoping_noi_init"
                    elif APPLICATION_PREFILING_INIT.search(_pre) or APPLICATION_PREFILING_INIT.search(_post):
                        _init_cue = "application_prefiling_init"
                    if _init_cue:
                        role = "clear_initiation"
                        conf = 5.0
                        pos_cues = [c for c in pos_cues
                                    if c not in ("decision_strong", "decision_med", "doc_type_decision")]
                        pos_cues = pos_cues + [_init_cue]

                # Skip clear rejects
                if role == "reject" and not heading:
                    continue

                # Minimum context guard: very short contexts (<40 chars, non-metadata)
                # have no meaningful signal for BERT and are often just date + name fragments.
                # Allow short contexts only when they carry an explicit strong cue.
                if len(block) < 40 and role not in ("clear_decision", "clear_initiation") and source_tier != "metadata":
                    continue

                # Estimate position in document
                pos_pct: float | None = None
                if packet.get("page_start") is not None and packet.get("page_end") is not None:
                    try:
                        page_n = int(packet["page_start"])
                        # Use packet retrieval_score as proxy for importance
                        pos_pct = min(1.0, page_n / max(1.0, page_n + 10))
                    except (ValueError, TypeError):
                        pass

                # Anchored model context: locate this date in the full packet text and
                # build a centered, sentence-bounded, capped, [[marker]]-wrapped window.
                raw_date = m.group(0)
                blk_off = context_clean.find(block)
                abs_ds = blk_off + _ms if blk_off >= 0 else context_clean.find(raw_date)
                if abs_ds < 0:
                    abs_ds = context_clean.find(raw_date)
                abs_de = abs_ds + len(raw_date) if abs_ds >= 0 else -1
                model_ctx = (
                    _build_model_context(context_clean, abs_ds, abs_de, process_type)
                    if abs_ds >= 0 else block
                )

                candidate_id = hashlib.sha1(
                    f"{packet['project_id']}|{packet.get('document_id')}|{packet.get('page_start')}|{date_str}|{block_norm}".encode()
                ).hexdigest()[:20]
                # Stable hash of the context text itself — survives candidate_id changes
                # caused by pipeline re-runs. Used by gold label import as a secondary join key
                # so labels don't need to be re-done when page numbers or logic change.
                context_window_hash = hashlib.sha1(
                    " ".join(block.split()).encode()
                ).hexdigest()[:16]

                candidates.append({
                    "candidate_id": candidate_id,
                    "project_id": packet["project_id"],
                    "process_type": process_type,
                    "document_id": packet.get("document_id"),
                    "page_number": packet.get("page_start"),
                    "section_id": packet.get("section_id"),
                    "context_packet_id": packet.get("context_packet_id"),
                    "source_tier": source_tier,
                    "retrieval_tier": packet.get("retrieval_tier"),
                    "candidate_source_type": "document_text",
                    "raw_date_text": raw_date,
                    "parsed_date": date_str,
                    "date_granularity": granularity,
                    "context_text": block,
                    "context_cleaned": block_norm + ("..." if len(block) > 100 else ""),
                    "model_context": model_ctx,
                    "document_title": packet.get("document_title"),
                    "file_name": None,
                    "document_type_clean": document_type_clean,
                    "document_type_category": packet.get("document_type_category"),
                    "main_document": packet.get("main_document"),
                    "heading_title": heading,
                    "parent_heading_title": packet.get("parent_heading_title"),
                    "position_pct": pos_pct,
                    "section_position_pct": None,
                    "candidate_role": role,
                    "role_confidence": ("high" if conf >= 4 else "medium" if conf >= 2.5 else "low"),
                    "role_confidence_score": conf,
                    "rule_ids": ptype,
                    "positive_cue_flags": "|".join(pos_cues),
                    "negative_cue_flags": "|".join(neg_cues),
                    "classifier_label": None,
                    "classifier_score": None,
                    "api_label": None,
                    "api_call_id": None,
                    "context_window_hash": context_window_hash,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })

    return candidates


def add_repeated_mention_counts(candidates_df: pd.DataFrame) -> pd.DataFrame:
    """Count how many times the same parsed_date appears per project."""
    if candidates_df.empty:
        return candidates_df
    counts = (
        candidates_df.groupby(["project_id", "parsed_date"])
        .size()
        .reset_index(name="date_mention_count")
    )
    return candidates_df.merge(counts, on=["project_id", "parsed_date"], how="left")


def _hash_rank(cid: object) -> int:
    """Deterministic per-candidate rank (stable across runs, independent of df order)."""
    return int(hashlib.sha1(str(cid).encode()).hexdigest(), 16)


def emit_labeling_sample(df: pd.DataFrame, packets_df: pd.DataFrame | None = None,
                         path: Path = LABELING_SAMPLE_PATH,
                         ids_path: Path = LABELING_SAMPLE_IDS_PATH,
                         n: int = LABELING_SAMPLE_SIZE) -> None:
    """
    Write (or expand) the human-labeling sample used to train the SetFit classifier.

    REPLICABLE: Chosen candidate_ids are persisted to `ids_path`. If the file exists
    with fewer IDs than `n`, additional candidates are drawn using the same
    stratified hash-rank protocol and appended — the existing IDs are never
    replaced. Delete `ids_path` to start a completely fresh draw.

    SAFE: Existing labels and notes in labeling_sample.csv are carried forward by
    candidate_id on every re-run. Non-empty values are never overwritten.

    GROWABLE: Call with a larger `n` (e.g. via --emit-labeling-sample --labeling-sample-n 800)
    to expand the sample without re-running the full extraction pipeline.

    Reviewer fills the `label` column with: initiation | decision | neither.
    """
    if df.empty:
        return
    pool = df[df["rule_ids"] != "metadata_tier_a"].copy()
    if pool.empty:
        pool = df.copy()

    # 1) Load existing locked IDs (if any). Expand if current count < n.
    existing_ids: list[str] = []
    if ids_path.exists():
        existing_ids = [x.strip() for x in ids_path.read_text().splitlines() if x.strip()]

    if len(existing_ids) < n:
        # Draw additional candidates to reach n, skipping already-selected IDs.
        n_needed = n - len(existing_ids)
        existing_set = set(existing_ids)
        available = pool[~pool["candidate_id"].isin(existing_set)].copy()
        cells = available.groupby(["process_type", "candidate_role"], dropna=False)
        per_cell = max(3, n_needed // max(1, cells.ngroups))
        parts = []
        for _, g in cells:
            gg = g.assign(_h=g["candidate_id"].map(_hash_rank)).sort_values("_h")
            parts.append(gg.head(min(len(gg), per_cell)).drop(columns="_h"))
        new_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(new_df) > n_needed:
            new_df = (new_df.assign(_h=new_df["candidate_id"].map(_hash_rank))
                      .sort_values("_h").head(n_needed).drop(columns="_h").reset_index(drop=True))
        new_ids = list(new_df["candidate_id"].astype(str)) if not new_df.empty else []
        all_ids = existing_ids + new_ids
        ids_path.parent.mkdir(parents=True, exist_ok=True)
        ids_path.write_text("\n".join(all_ids))
        added = len(new_ids)
        if existing_ids:
            print(f"[labeling sample] Expanded: {len(existing_ids)} → {len(all_ids)} IDs "
                  f"({added} new, {n_needed - added} unavailable).")
        else:
            print(f"Wrote locked selection: {ids_path} ({len(all_ids)} ids) — "
                  "re-runs will reuse these exact candidates.")
    else:
        all_ids = existing_ids

    # 2) Pull the locked selection from pool; warn about missing candidates.
    sample = pool[pool["candidate_id"].isin(set(all_ids))].copy()
    missing = len(all_ids) - sample["candidate_id"].nunique()
    if missing:
        print(f"[labeling sample] {missing} previously-selected candidates no longer "
              "exist (regex changed); keeping the rest.")
    order = {cid: i for i, cid in enumerate(all_ids)}
    sample = (sample.assign(_o=sample["candidate_id"].map(order))
              .sort_values("_o").drop(columns="_o"))

    # 3) Rebuild the candidate-evidence context (bounded window centered on the date)
    #    from the full packet text, so the sample reflects the current windowing logic.
    pkt_map: dict = {}
    if packets_df is not None and "context_packet_id" in packets_df.columns:
        pk = packets_df[["context_packet_id", "context_text"]].drop_duplicates("context_packet_id")
        pkt_map = dict(zip(pk["context_packet_id"], pk["context_text"]))

    def _ctx(row) -> str:
        txt = " ".join(str(pkt_map.get(row.get("context_packet_id")) or row.get("context_text") or "").split())
        dt = str(row.get("raw_date_text") or "")
        i = txt.find(dt) if dt else -1
        if i < 0:
            return str(row.get("model_context") or "")
        return _build_model_context(txt, i, i + len(dt), row.get("process_type"))

    sample["model_context"] = sample.apply(_ctx, axis=1)

    # 4) Carry over existing labels AND notes by candidate_id. Never overwrite non-empty values.
    prev_labels: dict = {}
    prev_notes: dict = {}
    if path.exists():
        try:
            prev = pd.read_csv(path)
            for _, r in prev.iterrows():
                cid = str(r.get("candidate_id"))
                raw_lab = r.get("label")
                lab = "" if pd.isna(raw_lab) else str(raw_lab).strip()
                if lab and lab.lower() not in ("nan", "none"):
                    prev_labels[cid] = lab
                raw_note = r.get("notes")
                note = "" if pd.isna(raw_note) else str(raw_note).strip()
                if note and note.lower() not in ("nan", "none"):
                    prev_notes[cid] = note
        except Exception:
            pass
    sample["stratum"] = (sample["process_type"].astype(str) + "/"
                         + sample["candidate_role"].astype(str))
    sample["label"] = sample["candidate_id"].astype(str).map(prev_labels).fillna("")
    sample["notes"] = sample["candidate_id"].astype(str).map(prev_notes).fillna("")
    carried_labels = int((sample["label"].str.strip() != "").sum())
    carried_notes = int((sample["notes"].str.strip() != "").sum())

    cols = ["candidate_id", "project_id", "process_type", "candidate_role",
            "role_confidence_score", "parsed_date", "date_granularity",
            "document_type_clean", "heading_title", "raw_date_text",
            "model_context", "stratum", "label", "notes"]
    sample = sample[[c for c in cols if c in sample.columns]]

    path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(path, index=False)
    print(f"Wrote labeling sample: {path}  "
          f"({len(sample)} rows; {carried_labels} labels, {carried_notes} notes carried over)")
    print("  Read 'model_context' (candidate evidence) to label; fill 'label' with: "
          "initiation | decision | neither, then run 04_classify_candidates.py --train")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract date candidates from context packets.")
    parser.add_argument(
        "--process", nargs="+", choices=["CE", "EA", "EIS"], default=["CE", "EA", "EIS"]
    )
    parser.add_argument("--sample-ids", help="Path to a file with one project_id per line.")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output even if it already exists.")
    parser.add_argument("--run-dir", help="Override run directory (reads packets from here, writes candidates here).")
    parser.add_argument(
        "--emit-labeling-sample", action="store_true",
        help="Emit (or expand) the labeling sample from the existing candidates parquet "
             "without re-running extraction. Use --labeling-sample-n to set the target size.",
    )
    parser.add_argument(
        "--labeling-sample-n", type=int, default=None,
        help=f"Target total size for the labeling sample (default: {LABELING_SAMPLE_SIZE}). "
             "If the locked IDs file already has this many entries, nothing is added.",
    )
    args = parser.parse_args()

    # Standalone mode: emit/expand the labeling sample without touching extraction outputs.
    if args.emit_labeling_sample:
        if not OUTPUT_PATH.exists():
            raise FileNotFoundError(
                f"Candidates parquet not found: {OUTPUT_PATH}\n"
                "Run the full extraction pipeline first."
            )
        target_n = args.labeling_sample_n or LABELING_SAMPLE_SIZE
        print(f"Loading candidates: {OUTPUT_PATH}")
        df_cands = pd.read_parquet(OUTPUT_PATH)
        print(f"  {len(df_cands):,} candidates loaded.")
        emit_labeling_sample(df_cands, n=target_n)
        return

    # Resolve run directory — matches the logic in script 02.
    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif args.sample_ids:
        run_dir = TIMELINE_DIR / "sample_runs" / Path(args.sample_ids).stem
    else:
        run_dir = TIMELINE_DIR
    packets_path = run_dir / "timeline_context_packets.parquet"
    output_path = run_dir / "timeline_candidates.parquet"

    if output_path.exists() and not args.force and not args.sample_ids and not args.append:
        print(f"Output already exists: {output_path}")
        print("Re-run only when 02_retrieve.py output changes.")
        print("Pass --force to overwrite.")
        return

    if not packets_path.exists():
        raise FileNotFoundError(f"Context packets not found: {packets_path}\nRun 02_retrieve.py first.")

    project_ids: set[str] | None = None
    if args.sample_ids:
        with open(args.sample_ids) as f:
            project_ids = {line.strip() for line in f if line.strip()}
        print(f"Filtering to {len(project_ids)} sample project IDs.")

    print(f"Loading context packets: {packets_path}")
    packets_df = pd.read_parquet(packets_path)

    # Guard: refuse to write subset data to the main TIMELINE_DIR output.
    # A partial packets file (e.g. EIS-only from a --process run) would overwrite
    # the full-corpus candidates and silently lose CE/EA data.
    ALL_PROCESS_TYPES = {"CE", "EA", "EIS"}
    packets_process_types = set(packets_df["process_type"].unique())
    if run_dir == TIMELINE_DIR and packets_process_types != ALL_PROCESS_TYPES:
        raise SystemExit(
            f"[GUARD] Input packets contain only {packets_process_types}, not all process types.\n"
            f"Writing subset data to {output_path} would overwrite the full-corpus candidates.\n"
            f"Use --run-dir to isolate this run, or re-run 02_retrieve.py "
            f"without --process to restore full-corpus packets."
        )

    packets_df = packets_df[packets_df["process_type"].isin(args.process)]
    if project_ids:
        packets_df = packets_df[packets_df["project_id"].isin(project_ids)]
    print(f"  {len(packets_df):,} packets to process")

    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    all_candidates: list[dict] = []
    packets_records = packets_df.to_dict("records")
    for i, row in enumerate(packets_records):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i:,}/{len(packets_records):,} packets, {len(all_candidates):,} candidates so far...")
        cands = extract_candidates_from_packet(row)
        all_candidates.extend(cands)

    if not all_candidates:
        print("No candidates extracted.")
        return

    df = pd.DataFrame(all_candidates)
    df = add_repeated_mention_counts(df)

    # Initialize scoring columns to zero — populated by 05_select_dates.py
    for col in [
        "source_strength", "role_cue_strength", "document_priority",
        "section_priority", "page_priority", "position_signal",
        "classifier_signal", "chronology_signal", "repeated_mention_signal",
        "negative_penalty", "ranking_score",
    ]:
        df[col] = 0.0

    df["selected_for_initiation"] = False
    df["selected_for_decision"] = False
    df["is_proxy"] = df["candidate_role"].isin(["proxy_initiation", "proxy_decision"])

    # Deduplicate by candidate_id
    df = df.drop_duplicates("candidate_id")

    print(f"\nTotal candidates: {len(df):,}")
    print("Role distribution:")
    print(df["candidate_role"].value_counts().to_string())
    print("Granularity distribution:")
    print(df["date_granularity"].value_counts().to_string())

    run_dir.mkdir(parents=True, exist_ok=True)

    if args.append and output_path.exists():
        existing = pd.read_parquet(output_path)
        if project_ids:
            # Drop existing rows for this shard's projects so updated flags win.
            # Without this, drop_duplicates("candidate_id") keeps the first (old) row,
            # discarding any new positive_cue_flags (e.g. date_determined) added since
            # the last full run.
            existing = existing[~existing["project_id"].isin(project_ids)]
        df = pd.concat([existing, df], ignore_index=True).drop_duplicates("candidate_id")
        print(f"After merge with existing: {len(df):,}")

    df.to_parquet(output_path, index=False)
    print(f"\nWrote: {output_path}")

    # Emit the human-labeling sample only on a full-corpus run (run_dir == TIMELINE_DIR),
    # so the SetFit training sample is representative of the whole corpus.
    if run_dir == TIMELINE_DIR:
        emit_labeling_sample(df, packets_df)


if __name__ == "__main__":
    main()
