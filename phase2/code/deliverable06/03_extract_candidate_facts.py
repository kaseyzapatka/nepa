"""D6 v2 — 03: extract structured candidate facts.

Deterministic-first (runs now): numeric limits found *within* the assembled span
texts with category-specific sanity bounds; boolean siting constraints; and a
mitigation-dependence signal reused from the existing `fonsi_conditions.parquet`
(condition roles/obligations). An LLM pass is wired but gated behind `--use-llm`
(Gate 3) and is a no-op without an API key / the anthropic SDK.

Audit-timestamp convention (matches the rest of the pipeline):
  candidate_extraction_run_at on every row; candidate_llm_run_at set only on a
  successful LLM call, else "".

Outputs:
  - data/analysis/deliverable06/candidate_facts.parquet
  - output/deliverable06/candidate_extraction_review.csv
"""

import argparse
import json
import os
import re
from functools import lru_cache

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import duckdb
import pandas as pd

import embeddings
from prompts import build_facts_prompt
from common import (
    D6_ANALYSIS_DIR,
    D6_REVIEW_DIR,
    ensure_d6_dirs,
    normalize_space,
    sha256_text,
    utc_now,
    write_parquet,
)
from candidates import TAXONOMY_VERSION

PACKETS = D6_ANALYSIS_DIR / "candidate_evidence_packets.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
CONDITIONS = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
OUT = D6_ANALYSIS_DIR / "candidate_facts.parquet"
REVIEW = D6_REVIEW_DIR / "candidate_extraction_review.csv"
LLM_CACHE = D6_ANALYSIS_DIR / "candidate_facts_llm_cache.json"

SCHEMA_VERSION = "d6_facts_v1"
PROMPT_VERSION = "d6_facts_prompt_v1"
# Default model tier: Sonnet (workhorse). Escalate to claude-opus-4-8 for highest
# fidelity on the subtle calls; benchmark with 06 to confirm the lowest model
# that clears the accuracy bar. Haiku is available but not the default.
LLM_MODEL_DEFAULT = "claude-sonnet-4-6"

NUM = r"(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)"
RX = {
    "acres": re.compile(NUM + r"\s*acres?\b", re.IGNORECASE),
    "miles": re.compile(NUM + r"\s*miles?\b", re.IGNORECASE),
    "mw": re.compile(NUM + r"\s*(?:mw|megawatts?)\b", re.IGNORECASE),
    "kv": re.compile(NUM + r"\s*(?:kv|kilovolts?)\b", re.IGNORECASE),
    "wells": re.compile(NUM + r"\s*(?:wells?|borings?|boreholes?)\b", re.IGNORECASE),
}
RX_DURATION = re.compile(r"\b(\d+)\s*(year|month|week|day)s?\b", re.IGNORECASE)
RX_NO_ROAD = re.compile(
    r"no\s+new\s+(?:permanent\s+)?(?:access\s+)?roads?|no\s+(?:permanent\s+)?road\s+construction|"
    r"without\s+(?:the\s+)?(?:construction\s+of\s+)?new\s+roads?|no\s+new\s+or\s+improved\s+roads?",
    re.IGNORECASE,
)
RX_EXISTING_ROW = re.compile(
    r"within\s+(?:the\s+)?existing\s+(?:right[- ]of[- ]way|right of way|row|corridor|disturbed)",
    re.IGNORECASE,
)
RX_DISTURBED = re.compile(
    r"previously\s+disturbed|previously\s+developed|disturbed\s+(?:land|area|ground)|"
    r"brownfield|reclaimed|degraded\s+land",
    re.IGNORECASE,
)

DEFAULT_CAPS = {"acres": 100000, "miles": 1000, "mw": 10000, "kv": 1000, "wells": 1000}
CATEGORY_CAPS = {
    "solar": {"acres": 50000, "mw": 5000},
    "transmission_upgrade": {"acres": 100000, "miles": 1000, "kv": 1000},
    "geothermal_exploration": {"acres": 50000, "wells": 300},
    "temporary_resource_assessment": {"acres": 5000, "wells": 200},
    "wind_onshore": {"acres": 100000, "mw": 5000},
}


# --- context-aware numeric limits: disturbance footprint vs. planning/lease area ---
DISTURB_RX = re.compile(
    r"disturb|footprint|affected|impact|grad(?:e|ing)|clear(?:ed|ing)|temporar|"
    r"ground disturbance|surface disturbance|construction|right[- ]of[- ]way", re.IGNORECASE)
AREA_RX = re.compile(
    r"planning area|study area|project area|analysis area|lease|allotment|watershed|"
    r"public lands|administered|acreage of|management area|total of", re.IGNORECASE)

# --- previously-disturbed / sited-on-disturbed land (tightened: requires the
#     "previously" qualifier or a developed-site noun, not generic "disturbed area") ---
DISTURBED_LAND_RX = re.compile(
    r"previously disturbed|previously developed|already disturbed|brownfield|"
    r"reclaimed (?:land|area|mine)|degraded land|disturbed or developed|"
    r"former(?:ly)? (?:developed|industrial|mine|agricultural)|rooftop|parking (?:lot|area)|"
    r"landfill|capped|on existing|industrial site", re.IGNORECASE)

# --- extraordinary circumstances (CE-gating categories only) ---
# Narrowed to the rarer, genuinely CE-gating categories. Generic resource areas
# (wetlands, cultural, migratory birds) appear in nearly every EA and are not
# discriminating, so they are excluded. NOTE: this is a *mention scan*, not a
# determination that the resource is present AND impacted — that is an LLM task.
EXTRAORDINARY_RX = re.compile(
    r"extraordinary circumstance|critical habitat|threatened (?:or |and )?endangered species|"
    r"\besa[- ]listed\b|wilderness (?:area|study area)|wild and scenic|100[- ]year floodplain|"
    r"national (?:monument|register of historic)|prime (?:or unique )?farmland|"
    r"sole[- ]source aquifer", re.IGNORECASE)

# --- spelled-out small numbers (well counts are often written as words) ---
WORD_NUM = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
RX_WELLS_WORD = re.compile(
    r"\b(" + "|".join(WORD_NUM) + r")\b[\w\s,-]{0,40}?\b(?:wells?|borings?|boreholes?)\b", re.IGNORECASE)

# --- action-definition sentence selection (spaCy split + all-MiniLM rank) ---
ACTION_QUERIES = [
    "The proposed action is to construct and operate the facility.",
    "The applicant proposes to build the project.",
    "The project would consist of constructing and operating.",
    "BLM proposes to authorize the construction of.",
    "The purpose of the proposed action is to upgrade or develop.",
]
BOILERPLATE_RX = re.compile(
    r"no action alternative|table of contents|introduction and background|"
    r"this environmental assessment (?:has|was)|list of (?:figures|tables)|^page\b", re.IGNORECASE)
# header-like lines to skip outright (chapter/figure/table/section-number openers)
HEADER_RX = re.compile(r"^\s*(?:chapter|figure|table|appendix|section|exhibit)\b|^\s*\d+(?:\.\d+)*\s", re.IGNORECASE)
# require an action/proposal verb so we pick a real action sentence, not a heading
ACTION_VERB_RX = re.compile(
    r"\b(?:construct|install|operat|build|drill|upgrad|replac|reconductor|rebuild|develop|"
    r"propos|would|maintain|remov|expand|provid|authoriz|reinforc|reloc|decommission)\w*", re.IGNORECASE)


def _floats(rx: re.Pattern, text: str) -> list[float]:
    out = []
    for m in rx.findall(text):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            continue
    return out


def _capped_max(values: list[float], cap: float):
    kept = [v for v in values if 0 < v <= cap]
    return round(max(kept), 2) if kept else None


def extract_numbers(text: str) -> dict[str, list[float]]:
    return {metric: _floats(rx, text) for metric, rx in RX.items()}


def context_metric(metric: str, text: str) -> tuple[list[float], list[float]]:
    """Split a numeric metric's mentions into (disturbance-context, any) via a ±window.

    The CE-relevant number is the disturbance/work extent, not lease/planning-area
    or total-route totals. Applies to acres and miles.
    """
    disturb, anyv = [], []
    for m in RX[metric].finditer(text):
        try:
            val = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        anyv.append(val)
        window = text[max(0, m.start() - 70):m.end() + 45]
        if DISTURB_RX.search(window) and not AREA_RX.search(window):
            disturb.append(val)
    return disturb, anyv


def wells_from_words(text: str) -> list[int]:
    """Recover well counts written as words (e.g. 'up to twelve exploratory wells')."""
    return [WORD_NUM[m.group(1).lower()] for m in RX_WELLS_WORD.finditer(text)]


@lru_cache(maxsize=1)
def _sentencizer():
    import spacy
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    return nlp


def _good_sentence(s: str) -> bool:
    words = s.split()
    if not (6 <= len(words) <= 60) or s.isupper():
        return False
    if BOILERPLATE_RX.search(s) or HEADER_RX.search(s):
        return False
    return bool(ACTION_VERB_RX.search(s))  # must contain an action/proposal verb


def split_sentences(text: str) -> list[str]:
    return [normalize_space(s.text) for s in _sentencizer()(text[:8000]).sents]


def build_action_definitions(packets: pd.DataFrame, use_emb: bool) -> dict[str, str]:
    """Pick the sentence in each project's action_text closest to an action template."""
    per: dict[str, list[str]] = {}
    for r in packets.itertuples(index=False):
        sents = [s for s in split_sentences(normalize_space(getattr(r, "action_text", "") or ""))
                 if _good_sentence(s)]
        per[str(r.project_id)] = sents[:25]
    if not use_emb:
        return {pid: (s[0] if s else "") for pid, s in per.items()}
    flat = [(pid, s) for pid, sents in per.items() for s in sents]
    if not flat:
        return {pid: "" for pid in per}
    sims = embeddings.cosine(embeddings.embed([s for _, s in flat]),
                             embeddings.embed(ACTION_QUERIES)).max(axis=1)
    best: dict[str, str] = {}
    score: dict[str, float] = {}
    for (pid, s), sc in zip(flat, sims):
        if pid not in score or sc > score[pid]:
            score[pid] = float(sc)
            best[pid] = s
    for pid, sents in per.items():
        best.setdefault(pid, sents[0] if sents else "")
    return best


def mitigation_from_conditions(cond: pd.DataFrame) -> dict:
    """3-way mitigation dependence from reused condition roles/obligations."""
    if cond.empty:
        return {"mitigation_dependence": "none", "mitigation_summary": "",
                "mitigation_resource_areas": "", "mitigation_method": "no_condition_rows"}
    roles = set(cond["condition_role"])
    strong = cond[(cond["condition_role"] == "mitigation_commitment")
                  & (cond["obligation_level"].isin(["required", "committed"]))]
    if not strong.empty:
        dep = "case_specific_dependent"
    elif roles & {"baseline_design_feature", "best_management_practice"}:
        dep = "design_feature_only"
    elif roles & {"mitigation_commitment", "monitoring_requirement", "enforcement_or_permit_condition"}:
        dep = "design_feature_only"
    else:
        dep = "uncertain"
    areas = sorted({a for a in cond["resource_area"].dropna().astype(str) if a and a != "unknown"})
    sample = strong if not strong.empty else cond
    summary = " | ".join(normalize_space(t)[:160] for t in sample["condition_text"].head(3))
    return {"mitigation_dependence": dep, "mitigation_summary": summary[:500],
            "mitigation_resource_areas": ", ".join(areas[:8]), "mitigation_method": "conditions_reuse"}


def first_citation(span_provenance: str) -> dict:
    try:
        recs = json.loads(span_provenance)
    except (TypeError, json.JSONDecodeError):
        recs = []
    if not recs:
        return {"citation_document_id": "", "citation_section_id": "",
                "citation_evidence_span_id": "", "citation_page": None}
    r = recs[0]
    return {"citation_document_id": r.get("document_id", ""),
            "citation_section_id": r.get("section_id", ""),
            "citation_evidence_span_id": r.get("evidence_span_id", ""),
            "citation_page": r.get("page_start")}


def llm_extract(packet_text: str, category: str, model: str, cache: dict) -> dict | None:
    """Gated LLM pass (Gate 3). No-op without anthropic + ANTHROPIC_API_KEY."""
    key = sha256_text(f"{PROMPT_VERSION}|{SCHEMA_VERSION}|{model}|{category}|{packet_text}")
    if key in cache:
        return cache[key]
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    client = anthropic.Anthropic()
    prompt = build_facts_prompt(packet_text, category)
    try:
        msg = client.messages.create(model=model, max_tokens=700,
                                     messages=[{"role": "user", "content": prompt}])
        data = json.loads(msg.content[0].text)
    except Exception:
        return None
    cache[key] = data
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-llm", action="store_true", help="enable the gated LLM pass (Gate 3)")
    ap.add_argument("--model", default=LLM_MODEL_DEFAULT)
    args = ap.parse_args()

    ensure_d6_dirs()
    run_at = utc_now()

    packets = pd.read_parquet(PACKETS)
    packets["project_id"] = packets["project_id"].astype(str)
    corpus = pd.read_parquet(CORPUS)
    fonsi = corpus.loc[corpus["is_fonsi"]].copy()
    fonsi["project_id"] = fonsi["project_id"].astype(str)

    # reuse existing condition rows for the mitigation signal
    cond_by_project: dict[str, pd.DataFrame] = {}
    if CONDITIONS.exists():
        ids = ",".join(f"'{p}'" for p in packets["project_id"].unique())
        con = duckdb.connect()
        cond = con.execute(
            f"""select project_id, condition_role, obligation_level, resource_area, condition_text
                from read_parquet('{CONDITIONS}') where cast(project_id as varchar) in ({ids})"""
        ).df()
        cond["project_id"] = cond["project_id"].astype(str)
        cond_by_project = {pid: g for pid, g in cond.groupby("project_id")}

    cache = {}
    if args.use_llm and LLM_CACHE.exists():
        cache = json.loads(LLM_CACHE.read_text())

    # action definitions via spaCy sentence split + all-MiniLM ranking (local, cheap)
    use_emb = embeddings.available()
    action_defs = build_action_definitions(packets, use_emb)

    # per-project deterministic extraction (text is shared across a project's categories)
    per_project: dict[str, dict] = {}
    for r in packets.itertuples(index=False):
        pid = r.project_id
        text = " ".join(getattr(r, c, "") or "" for c in
                        ("action_text", "boundary_text", "finding_text", "condition_text"))
        # text for sensitive-resource / extraordinary-circumstance signals (incl. resources)
        ec_text = " ".join(getattr(r, c, "") or "" for c in
                           ("resource_text", "boundary_text", "finding_text", "condition_text"))
        nums = extract_numbers(text)
        nums["wells"] = nums["wells"] + [float(w) for w in wells_from_words(text)]  # recover word counts
        acres_disturb, acres_any = context_metric("acres", text)
        miles_disturb, miles_any = context_metric("miles", text)
        dur = RX_DURATION.search(text)
        cite = first_citation(r.span_provenance)
        mit = mitigation_from_conditions(cond_by_project.get(pid, pd.DataFrame(
            columns=["condition_role", "obligation_level", "resource_area", "condition_text"])))
        action_def = action_defs.get(pid) or normalize_space(getattr(r, "action_text", "") or "")[:300]
        quoted = normalize_space(
            (getattr(r, "action_text", "") or getattr(r, "finding_text", "") or ""))[:300]
        ec_terms = sorted({m.group(0).lower() for m in EXTRAORDINARY_RX.finditer(ec_text)})
        per_project[pid] = {
            "_nums": nums,
            "_acres_disturb": acres_disturb,
            "_acres_any": acres_any,
            "_miles_disturb": miles_disturb,
            "_miles_any": miles_any,
            "duration": dur.group(0) if dur else "",
            "within_existing_row": bool(RX_EXISTING_ROW.search(text)),
            "no_new_access_road": bool(RX_NO_ROAD.search(text)),
            "previously_disturbed_land": bool(DISTURBED_LAND_RX.search(text)),
            "has_sensitive_resource": bool(ec_terms),
            "extraordinary_circumstances": ", ".join(ec_terms[:10]),
            "action_definition": action_def,
            "quoted_span": quoted,
            "packet_source": getattr(r, "packet_source", "packet"),
            **cite, **mit,
        }

    # emit one row per (project, candidate_category) with category-capped numbers
    rows = []
    llm_hits = 0
    for fr in fonsi.itertuples(index=False):
        pid = fr.project_id
        if pid not in per_project:
            continue
        base = per_project[pid]
        caps = {**DEFAULT_CAPS, **CATEGORY_CAPS.get(fr.candidate_category, {})}
        nums = base["_nums"]
        # prefer disturbance-context acreage; fall back to any in-range acreage
        acres_disturb = _capped_max(base["_acres_disturb"], caps["acres"])
        acres_any = _capped_max(base["_acres_any"], caps["acres"])
        max_acres = acres_disturb if acres_disturb is not None else acres_any
        acres_basis = ("disturbance" if acres_disturb is not None
                       else ("area_or_unspecified" if acres_any is not None else "none"))
        # prefer disturbance-context miles; fall back to any in-range (drops total-route grabs)
        miles_disturb = _capped_max(base["_miles_disturb"], caps["miles"])
        miles_any = _capped_max(base["_miles_any"], caps["miles"])
        max_miles = miles_disturb if miles_disturb is not None else miles_any
        llm_run_at = ""
        method = "deterministic_regex+conditions"
        confidence = "low"
        llm_fields = {}
        if args.use_llm:
            pkt = packets.loc[packets["project_id"].eq(pid), "action_text"]
            data = llm_extract(pkt.iloc[0] if len(pkt) else "", fr.candidate_category, args.model, cache)
            if data:
                llm_hits += 1
                llm_run_at = run_at
                method = "deterministic+llm"
                confidence = "medium"
                llm_fields = {k: data.get(k) for k in
                              ("action_definition", "mitigation_dependence", "mitigation_summary")
                              if data.get(k) is not None}
        rows.append({
            "project_id": pid,
            "candidate_category": fr.candidate_category,
            "candidate_label": fr.candidate_label,
            "subtype": fr.subtype,
            "is_profile_subtype": bool(fr.is_profile_subtype),
            "candidate_role": fr.candidate_role,
            "action_definition": llm_fields.get("action_definition", base["action_definition"]),
            "max_acres": max_acres,
            "max_acres_any": acres_any,
            "acres_basis": acres_basis,
            "max_miles": max_miles,
            "max_megawatts": _capped_max(nums["mw"], caps["mw"]),
            "max_kilovolts": _capped_max(nums["kv"], caps["kv"]),
            "n_wells": (lambda v: int(v) if v is not None else None)(_capped_max(nums["wells"], caps["wells"])),
            "duration": base["duration"],
            "within_existing_row": base["within_existing_row"],
            "no_new_access_road": base["no_new_access_road"],
            "previously_disturbed_land": base["previously_disturbed_land"],
            "has_sensitive_resource": base["has_sensitive_resource"],
            "extraordinary_circumstances": base["extraordinary_circumstances"],
            "mitigation_dependence": llm_fields.get("mitigation_dependence", base["mitigation_dependence"]),
            "mitigation_summary": llm_fields.get("mitigation_summary", base["mitigation_summary"]),
            "mitigation_resource_areas": base["mitigation_resource_areas"],
            "finding_rationale": base["quoted_span"],
            "citation_document_id": base["citation_document_id"],
            "citation_section_id": base["citation_section_id"],
            "citation_evidence_span_id": base["citation_evidence_span_id"],
            "citation_page": base["citation_page"],
            "quoted_span": base["quoted_span"],
            "extraction_method": method,
            "confidence": confidence,
            "llm_provider": "anthropic" if llm_run_at else "",
            "llm_model": args.model if llm_run_at else "",
            "prompt_version": PROMPT_VERSION if llm_run_at else "",
            "schema_version": SCHEMA_VERSION,
            "taxonomy_version": TAXONOMY_VERSION,
            "candidate_extraction_run_at": run_at,
            "candidate_llm_run_at": llm_run_at,
        })

    facts = pd.DataFrame(rows)
    write_parquet(facts, OUT)
    if args.use_llm:
        LLM_CACHE.write_text(json.dumps(cache))

    review_cols = ["project_id", "candidate_category", "subtype", "is_profile_subtype",
                   "action_definition", "max_acres", "acres_basis", "max_miles", "max_megawatts",
                   "n_wells", "no_new_access_road", "within_existing_row", "previously_disturbed_land",
                   "has_sensitive_resource", "extraordinary_circumstances",
                   "mitigation_dependence", "mitigation_resource_areas", "confidence",
                   "citation_document_id", "citation_page"]
    facts[review_cols].sort_values(["candidate_category", "project_id"]).to_csv(REVIEW, index=False)

    print(f"[03] fact rows={len(facts):,} (project×category) embeddings={use_emb} llm_hits={llm_hits} -> {OUT}")
    print(f"[03] mitigation_dependence:\n{facts['mitigation_dependence'].value_counts().to_string()}")
    filled = {m: int(facts[f'max_{m}'].notna().sum()) for m in ('acres', 'miles', 'megawatts', 'kilovolts')}
    print(f"[03] numeric limit fill (rows): {filled}  no_road={int(facts['no_new_access_road'].sum())} "
          f"existing_row={int(facts['within_existing_row'].sum())} "
          f"disturbed={int(facts['previously_disturbed_land'].sum())}")
    print(f"[03] acres_basis:\n{facts['acres_basis'].value_counts().to_string()}")
    print(f"[03] n_wells filled: {int(facts['n_wells'].notna().sum())}  "
          f"sensitive_resource flagged: {int(facts['has_sensitive_resource'].sum())} of {len(facts)}")


if __name__ == "__main__":
    main()
