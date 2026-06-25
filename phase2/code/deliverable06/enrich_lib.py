"""D6 — shared enrichment helpers (un-numbered, importable).

Common logic for the production enrichment pass (03_enrich_llm.py) and the model
benchmark (benchmark_models.py): the BALANCED span-based evidence packet, the
stratified pilot sample, the tool-use (structured-output) call, span-ref quote
verification, and the Keychain key loader. Numbered scripts can't be imported, so
this lives here.

Key change vs v1: the packet is built from typed SPAN rows with per-section
budgets (action / finding / condition / boundary / resource) — not an action-first
string truncated at 8k, which used to starve the model of finding/mitigation text.
Each excerpt is tagged [S#] with its page/document so the model cites span_refs.
"""

from __future__ import annotations

import json
import os
import re
import subprocess

import duckdb
import pandas as pd

from common import D6_ANALYSIS_DIR, normalize_space
from prompts import ENRICHMENT_FIELDS, build_enrichment_prompt, enrichment_tool_schema

PACKETS = D6_ANALYSIS_DIR / "fonsi_project_packets.parquet"     # per-project metadata + typed text
SPANS = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"        # typed span rows (verbatim + page/role)
INVENTORY = D6_ANALYSIS_DIR / "fonsi_document_inventory.parquet"
CORPUS = D6_ANALYSIS_DIR / "candidate_corpus.parquet"
ENRICH_MAX_TOKENS = 4096   # generous: 37 fields + quote arrays (avoids truncation)
ENRICH_MAX_RETRIES = 8     # SDK exponential backoff on 429 / 500 / 503 / 529 overloads
TOOL_NAME = "emit_fonsi_enrichment"

PRICING = {  # input, output USD per 1M tokens (claude-api skill table; verify before billing)
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}
JSON_FIELDS = {"key_activities", "mitigation_resource_areas", "key_impacts", "significance_factors",
               "significance_thresholds", "evidence", "referenced_ce_citations", "cooperating_agencies",
               "evidence_cited"}

# balanced packet plan: (span_type, max excerpts, preference regex). boundary is
# nearly empty in the data, so finding/condition carry the significance signal.
SECTION_PLAN = [
    ("action", 3, r"proposed action|would (construct|install|replace|build|rebuild|reconductor|upgrade)|"
                  r"project (would|includes|consists)"),
    ("finding", 3, r"no significant impact|finding of no|would be significant|significant (impact|effect)"),
    ("condition", 3, r"shall|will be required|mitigation|committed|conservation measure|best management|monitor"),
    ("boundary", 2, None),
    ("resource", 2, None),
]
FALLBACK_PLAN = [("fallback", 2, None)]   # used only if SECTION_PLAN yields nothing
FALLBACK_TEXT_FIELDS = [("action", "action_text"), ("finding", "finding_text"),
                        ("condition", "condition_text"), ("boundary", "boundary_text"),
                        ("resource", "resource_text")]   # last-resort: typed packet text
PER_SPAN_CHARS = 700
WIN_BEFORE, WIN_AFTER = 250, 450   # window around the matched phrase


# --- Anthropic key (same as D4 timeline: macOS Keychain 'nepa-anthropic') ---
KEYCHAIN_SERVICE = "nepa-anthropic"
_key_cache: dict[str, str] = {}


def get_anthropic_key(allow_keychain: bool = True) -> str | None:
    if "key" in _key_cache:
        return _key_cache["key"]
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key and allow_keychain:
        try:
            out = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                                 capture_output=True, text=True, timeout=60)
            if out.returncode == 0 and out.stdout.strip():
                key = out.stdout.strip()
        except Exception:
            key = None
    if key:
        _key_cache["key"] = key   # cache positive hits only -> at most one Keychain prompt
    return key


def pricing_for(model: str) -> tuple[float, float]:
    for prefix, rate in PRICING.items():
        if model.startswith(prefix):
            return rate
    return (3.0, 15.0)


def write_json_atomic(path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)   # atomic on POSIX — never leaves a half-written cache


def load_clean_packets() -> pd.DataFrame:
    pk = pd.read_parquet(PACKETS)
    pk = pk[pk["project_energy_type"].astype(str) == "Clean"].copy()
    pk["project_id"] = pk["project_id"].astype(str)
    return pk.sort_values("project_id").reset_index(drop=True)   # deterministic order


def _norm(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _doc_role(manifest_role: str) -> str:
    r = str(manifest_role).lower()
    return "FONSI" if "fonsi" in r else ("EA" if "ea" in r else manifest_role)


_META = [("title", "project_title"), ("type", "project_type"), ("tech", "tech_group"),
         ("agency", "lead_agency_harmonized"), ("state", "project_state"),
         ("fonsi_doc", "canonical_fonsi_document_id")]


def _excerpt(text: str, kw: str | None) -> str:
    """Emit a window AROUND the matched phrase (not just the span start), so the
    sentence that caused the span to be selected is actually shown."""
    if kw:
        m = re.search(kw, text, re.I)
        if m:
            s = max(0, m.start() - WIN_BEFORE)
            e = min(len(text), m.start() + WIN_AFTER)
            return ("…" if s > 0 else "") + text[s:e] + ("…" if e < len(text) else "")
    return text[:PER_SPAN_CHARS] + ("…" if len(text) > PER_SPAN_CHARS else "")


def _select_from_spans(sp: pd.DataFrame, plan, start_i: int = 1) -> tuple[list, dict, int]:
    blocks, tag_map, i = [], {}, start_i
    for stype, k, kw in plan:
        sub = sp[sp["span_type"] == stype]
        if sub.empty:
            continue
        match = sub["ntext"].map(lambda t: bool(re.search(kw, t, re.I))) if kw else pd.Series(True, index=sub.index)
        sub = sub.assign(mtch=match).sort_values(["mtch", "nlen"], ascending=[False, False])
        for r in sub.head(k).itertuples(index=False):
            tag = f"S{i}"; i += 1
            page = int(r.page_start) if pd.notna(r.page_start) else None
            role = _doc_role(r.manifest_role)
            head = _norm(getattr(r, "heading_title", ""))[:60]
            blocks.append(f"[{tag}] ({role}, p.{page if page is not None else '?'}, {stype}"
                          + (f', "{head}"' if head else "") + f"): {_excerpt(r.ntext, kw)}")
            tag_map[tag] = {"span_id": getattr(r, "evidence_span_id", None), "page": page,
                            "document_role": role, "document_id": r.document_id, "text": r.ntext}
    return blocks, tag_map, i


def _select_from_meta(meta_row, start_i: int = 1) -> tuple[list, dict]:
    """Last-resort packet from the typed packet text (for projects with no usable spans)."""
    blocks, tag_map, i = [], {}, start_i
    did = getattr(meta_row, "canonical_fonsi_document_id", None)
    for stype, col in FALLBACK_TEXT_FIELDS:
        t = _norm(getattr(meta_row, col, ""))
        if not t:
            continue
        tag = f"S{i}"; i += 1
        blocks.append(f"[{tag}] (FONSI, p.?, {stype}, packet-fallback): {t[:PER_SPAN_CHARS]}")
        tag_map[tag] = {"span_id": None, "page": None, "document_role": "FONSI",
                        "document_id": did, "text": t}
    return blocks, tag_map


def build_evidence_packet(meta_row, spans: pd.DataFrame) -> tuple[str, dict]:
    """Balanced, span-tagged evidence packet for one project. Returns (packet_text,
    tag_map) where tag_map[S#] = {span_id, page, document_role, document_id, text}.
    Never returns an empty packet: falls back to fallback-type spans, then to the
    typed packet text, so no paid row is metadata-only."""
    meta = "; ".join(f"{lab}={normalize_space(getattr(meta_row, col, ''))}"
                     for lab, col in _META if normalize_space(getattr(meta_row, col, "")))
    blocks, tag_map = [], {}
    if spans is not None and not spans.empty:
        sp = spans.assign(ntext=spans["span_text"].map(_norm))
        sp = sp.assign(nlen=sp["ntext"].str.len())
        blocks, tag_map, _ = _select_from_spans(sp, SECTION_PLAN)
        if not tag_map:                                  # only odd span types (e.g. 'fallback')
            blocks, tag_map, _ = _select_from_spans(sp, FALLBACK_PLAN)
    if not tag_map:                                      # no usable spans at all
        blocks, tag_map = _select_from_meta(meta_row)
    return f"PROJECT METADATA: {meta}" + ("\n\n" + "\n\n".join(blocks) if blocks else ""), tag_map


def load_spans_and_main(project_ids) -> tuple[dict, dict]:
    ids = "', '".join(str(p) for p in project_ids)
    spans = duckdb.connect().execute(
        f"select cast(project_id as varchar) project_id, evidence_span_id, span_type, span_text, "
        f"page_start, heading_title, manifest_role, document_id "
        f"from read_parquet('{SPANS}') where cast(project_id as varchar) in ('{ids}')").df()
    spans_by_pid = {pid: g for pid, g in spans.groupby("project_id")}
    inv = pd.read_parquet(INVENTORY, columns=["document_id", "main_document"])
    # main_document is a STRING ("YES"/"NO"/"") — astype(bool) would make every
    # non-empty value True (~99.7%). Compare to "YES".
    main_by_doc = dict(zip(inv["document_id"], inv["main_document"].astype(str).str.upper().eq("YES")))
    return spans_by_pid, main_by_doc


# --- stratified pilot sample (Codex spec) ----------------------------------
def pilot_sample(seed: int = 42) -> pd.DataFrame:
    """Representative pilot: profile + non-profile + off-scope + non-candidate FONSIs,
    so the pilot tests the schema across real diversity (not head(N) ordering)."""
    corp = pd.read_parquet(CORPUS)
    corp["project_id"] = corp["project_id"].astype(str)
    f = corp[corp["is_fonsi"]].copy()
    clean = set(load_clean_packets()["project_id"])
    candidate_ids = set(f["project_id"])

    chosen: dict[str, str] = {}

    def take(df, n, stratum):
        d = df.drop_duplicates("project_id")
        d = d.sample(min(n, len(d)), random_state=seed) if len(d) > n else d
        for pid in d["project_id"]:
            chosen.setdefault(pid, stratum)

    tx = f[f["candidate_category"] == "transmission_upgrade"]
    take(tx[tx["subtype"] == "standalone_upgrade"], 9, "transmission_profile")
    take(tx[tx["subtype"] == "off_scope_misclassified"], 4, "transmission_offscope")
    take(tx[~tx["subtype"].isin(["standalone_upgrade", "off_scope_misclassified"])], 4, "transmission_nonprofile")
    take(f[(f["candidate_category"] == "solar") & (f["subtype"] == "disturbed_developed")], 8, "solar_profile")
    take(f[(f["candidate_category"] == "solar") & (f["subtype"] != "disturbed_developed")], 3, "solar_other")
    take(f[(f["candidate_category"] == "geothermal_exploration") & (f["subtype"] == "exploration")], 7, "geothermal_profile")
    take(f[f["candidate_category"] == "temporary_resource_assessment"], 2, "temp_resource")
    take(f[f["candidate_category"] == "wind_onshore"], 10, "wind_contrast")
    noncand = pd.DataFrame({"project_id": sorted(clean - candidate_ids)})
    take(noncand, 18, "non_candidate")

    return pd.DataFrame({"project_id": list(chosen), "stratum": list(chosen.values())})


# --- structured-output call (tool-use) -------------------------------------
def make_client(key: str | None = None):
    """Anthropic client with built-in exponential backoff on 429/500/503/529, so a
    parallel run rides out transient overloads instead of erroring. Share one client
    across threads (it is thread-safe)."""
    import anthropic
    return anthropic.Anthropic(api_key=key or get_anthropic_key(), max_retries=ENRICH_MAX_RETRIES)


def call_enrichment(text: str, model: str, client=None) -> dict:
    """One enrichment call via tool-use (forces schema-valid JSON). Returns a dict:
    parsed|None, raw, in_tok, out_tok, stop_reason, error. Never raises (safe to run
    in a thread pool). Overloads are retried inside the client's max_retries budget."""
    blank = {"parsed": None, "raw": "", "in_tok": 0, "out_tok": 0, "stop_reason": "", "error": ""}
    key = get_anthropic_key()
    if not key:
        return {**blank, "error": "no_key"}
    try:
        import anthropic
    except ImportError:
        return {**blank, "error": "no_sdk"}
    c = client or anthropic.Anthropic(api_key=key, max_retries=ENRICH_MAX_RETRIES)
    tool = {"name": TOOL_NAME, "description": "Return the structured FONSI enrichment.",
            "input_schema": enrichment_tool_schema()}
    try:
        msg = c.messages.create(
            model=model, max_tokens=ENRICH_MAX_TOKENS, temperature=0,
            tools=[tool], tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=[{"role": "user", "content": build_enrichment_prompt(text)}],
        )
        block = next((b for b in msg.content if getattr(b, "type", None) == "tool_use"), None)
        parsed = block.input if block else None
        return {"parsed": parsed, "raw": json.dumps(parsed) if parsed is not None else "",
                "in_tok": msg.usage.input_tokens, "out_tok": msg.usage.output_tokens,
                "stop_reason": msg.stop_reason, "error": "" if parsed is not None else "no_tool_block"}
    except Exception as e:  # invalid model id, rate limit, SDK error, etc. — record it
        return {**blank, "error": f"{type(e).__name__}: {e}"}


def preflight(model: str, client=None) -> dict:
    """One cheap call to verify the model id + tool-use work before a real run."""
    return call_enrichment("PROJECT METADATA: title=test\n\n[S1] (EA, p.1, action): "
                           "The proposed action is a test reconductoring of an existing line.", model, client)


def coerce(parsed: dict) -> dict:
    out = {}
    for name, _t, _d in ENRICHMENT_FIELDS:
        v = parsed.get(name)
        out[name] = json.dumps(v, ensure_ascii=False) if name in JSON_FIELDS else v
    return out


# --- span-ref quote verification + citation --------------------------------
def _resolve(span_ref, quote, tag_map: dict, main_by_doc: dict) -> dict:
    miss = {"verified": False, "page": None, "document_role": None, "document_id": None,
            "is_main_document": None, "span_id": None}
    # strip any synthetic leading/trailing ellipsis the model may have copied from the excerpt
    q = re.sub(r"^[….\s]+|[….\s]+$", "", _norm(quote)) if quote else ""
    t = tag_map.get(str(span_ref or "").strip().strip("[]"))
    if t is None:                                   # fall back to searching all excerpts
        t = next((v for v in tag_map.values() if len(q) >= 12 and q in v["text"]), None)
        if t is None:
            return miss
    verified = (q in t["text"]) if q else True
    return {"verified": bool(verified), "page": t["page"], "document_role": t["document_role"],
            "document_id": t["document_id"], "is_main_document": bool(main_by_doc.get(t["document_id"], False)),
            "span_id": t["span_id"]}


def cite_quotes(parsed: dict, tag_map: dict, main_by_doc: dict) -> list[dict]:
    """Resolve each quoted claim's span_ref to page/document, and verify the quote is
    actually in that excerpt (catches hallucinated/misattributed text)."""
    out = []
    for ev in (parsed.get("evidence") or []):
        out.append({"claim": ev.get("claim"), "span_ref": ev.get("span_ref"), "quote": ev.get("quote"),
                    **_resolve(ev.get("span_ref"), ev.get("quote", ""), tag_map, main_by_doc)})
    for th in (parsed.get("significance_thresholds") or []):
        out.append({"claim": "significance_threshold", "span_ref": th.get("span_ref"),
                    "quote": th.get("statement"),
                    **_resolve(th.get("span_ref"), th.get("statement", ""), tag_map, main_by_doc)})
    for rc in (parsed.get("referenced_ce_citations") or []):
        q = rc.get("context") or rc.get("citation")
        out.append({"claim": "ce_citation", "span_ref": rc.get("span_ref"), "quote": q,
                    **_resolve(rc.get("span_ref"), q or "", tag_map, main_by_doc)})
    cdl = parsed.get("ce_development_language")
    if cdl:
        out.append({"claim": "ce_development_language", "span_ref": None, "quote": cdl,
                    **_resolve(None, cdl, tag_map, main_by_doc)})
    return out
