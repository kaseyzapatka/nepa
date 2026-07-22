"""D6 #47 — re-tag condition -> resource_area (Tier-1 heading rules + scoped Haiku multi-label).

STANDALONE-BILLABLE prerequisite (same pattern as 03_enrich_llm.py / 10_action_label.py): runs
OUTSIDE _run.py's deterministic chain, rebuilds fonsi_conditions.parquet in place; the chain's 05/09
and D2's join then consume it unchanged. Keyed cache => re-runs are $0.

WHY (see phase2/notes/deliverable06/pilot47_findings.md): fonsi_conditions.resource_area is pure
keyword-counting (~51% 'unknown', ~33% mis-attribution on mitigation commitments). That caps D2's
resource-level mitigation F1 at ~0.43. This does NOT move any D6 headline (mitigated-FONSI share,
mitigation_dependence, CE verdicts never read these tags); it is a D2-facing fix delivered from D6.

TWO TIERS:
  Tier-1 (free, deterministic): a condition the keyword dict leaves 'unknown' inherits the resource
    area from its section HEADING (mitigation_conditions.classify_resource_area_with_heading).
  Tier-2 (billable Haiku, deduped): for the mitigation_commitment rows (the ones feeding D2's join),
    a scoped multi-label pass tags resource_areas from the SHARED 12. Deduped by condition_text sha,
    so ~11,246 unique calls cover 14,072 rows; cached => re-run $0.

VOCABULARY: emits only the 12 shared areas + 'unknown' (identical to D2 significance_taxonomy
.SHARED_RESOURCE_AREAS and D6 mitigation_conditions.RESOURCE_AREAS). 'vegetation' (a project-level
enrichment-only value) is mapped to biological — never introduced into fonsi_conditions, or it would
break D2's RESOURCE_CROSSWALK lookup.

KEY: $ANTHROPIC_API_KEY else macOS Keychain 'nepa-anthropic' (one prompt). --dry-run never touches it.

USAGE
  python retag_condition_resources.py --dry-run     # Tier-1 preview + EXACT metered Haiku cost, no key
  python retag_condition_resources.py --run         # apply Tier-1 + Tier-2, rewrite fonsi_conditions (billable)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import duckdb
import pandas as pd

from common import D6_ANALYSIS_DIR, D6_RAW_DIR, ensure_d6_dirs, sha256_text, utc_now, write_parquet
import enrich_lib

sys.path.insert(0, str((D6_ANALYSIS_DIR.parents[2] / "code" / "extract")))
from mitigation_conditions import (  # noqa: E402
    classify_resource_area,
    classify_resource_area_with_heading,
)

CONDITIONS = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
SPANS = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
CACHE = D6_RAW_DIR / "resource_retag_cache.json"

# The 12 shared resource areas + 'unknown' — identical to D2 significance_taxonomy.SHARED_RESOURCE_AREAS
# and D6 mitigation_conditions.RESOURCE_AREAS. Enforced downstream by qa_deliverable06.py (enum subset).
SHARED_12 = [
    "air_quality", "water", "biological", "cultural", "visual", "noise",
    "soils_geology", "socioeconomic", "transportation", "land_use",
    "climate_ghg", "public_health",
]
VALID = set(SHARED_12) | {"unknown"}
ALIAS = {"vegetation": "biological"}  # enrichment-only value -> shared area (never emit 'vegetation')

# roles that feed D2's mitigation join (extract_common / 02_extract_fonsi_significance). Scope the paid
# pass to these; leave enforcement/legal/boilerplate/uncertain as 'unknown' (they have no resource area).
LLM_ROLES = ("mitigation_commitment",)

DEFAULT_MODEL = "claude-haiku-4-5"


def _heading_map() -> dict[str, str]:
    """section_id -> heading_title (join key for Tier-1)."""
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT section_id, heading_title FROM '{SPANS}' "
        "WHERE heading_title IS NOT NULL AND heading_title <> ''"
    ).fetchall()
    return {sid: h for sid, h in rows}


def _prompt(text: str) -> str:
    return (
        "You tag a NEPA mitigation/condition sentence with the environmental resource area(s) it "
        "protects. Return JSON only: {\"resource_areas\": [...]}. Each value MUST be one of: "
        f"{SHARED_12}. Use multiple only when the sentence genuinely covers multiple resources. "
        "If none applies, return an empty list. Do NOT invent values.\nSentence: " + text
    )


def _parse(obj_text: str) -> list[str]:
    import re
    m = re.search(r"\{.*\}", obj_text, re.S)
    if not m:
        return []
    try:
        areas = json.loads(m.group(0)).get("resource_areas", [])
    except Exception:
        return []
    out = []
    for a in areas:
        a = ALIAS.get(str(a).strip(), str(a).strip())
        if a in VALID and a != "unknown":
            out.append(a)
    # dedupe preserving order
    seen, dedup = set(), []
    for a in out:
        if a not in seen:
            seen.add(a); dedup.append(a)
    return dedup


def _call_haiku(text: str, model: str, client) -> list[str]:
    msg = client.messages.create(
        model=model, max_tokens=80,
        messages=[{"role": "user", "content": _prompt(text)}],
    )
    return _parse(msg.content[0].text)


def main() -> None:
    ap = argparse.ArgumentParser(description="D6 #47 condition resource re-tag (Tier-1 + scoped Haiku).")
    ap.add_argument("--run", action="store_true", help="apply Tier-1 + Tier-2 and rewrite fonsi_conditions (billable)")
    ap.add_argument("--dry-run", action="store_true", help="Tier-1 preview + exact Haiku cost, NO key, NO write")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--use-tier1", action="store_true",
                    help="re-enable Tier-1 heading inheritance (DEFAULT OFF: gold validation measured its "
                         "precision at 0.20 — it stamps labels onto procedural text under generic headings)")
    args = ap.parse_args()
    if not args.run:
        args.dry_run = True  # default is safe: never call the API unless --run is explicit
    ensure_d6_dirs()

    cond = pd.read_parquet(CONDITIONS)
    n = len(cond)
    base_unknown = int((cond["resource_area"] == "unknown").sum())
    print(f"[retag] fonsi_conditions rows={n}  baseline unknown={base_unknown} ({100*base_unknown/n:.1f}%)")

    # --- baseline tag, recomputed from text (idempotent: NOT read off the possibly-already-retagged
    #     resource_area column, so re-runs are stable) ---
    cond["_base_kw"] = [classify_resource_area(str(t)) for t in cond["condition_text"]]
    # Tier-1 heading inheritance is DISABLED by default. The 80-row gold validation
    # (retag_validation_score.md) measured its precision at 0.20 (8/10 wrong) — it stamps resource labels
    # onto procedural text sitting under generic headings; the pilot's 0.72 estimate did NOT hold up.
    # Non-Haiku rows therefore use the pure keyword baseline. Re-enable only with --use-tier1.
    if args.use_tier1:
        hmap = _heading_map()
        cond["_base"] = [
            classify_resource_area_with_heading(str(t), hmap.get(sid, ""))
            for t, sid in zip(cond["condition_text"], cond["section_id"])
        ]
        n_relabel = int((cond["_base"] != cond["_base_kw"]).sum())
        print(f"[retag] Tier-1 heading inheritance ENABLED (--use-tier1): {n_relabel} rows relabeled from heading")
    else:
        cond["_base"] = cond["_base_kw"]
        print("[retag] Tier-1 heading inheritance DISABLED (default; gold precision 0.20). "
              "Non-Haiku rows fall back to the keyword baseline.")

    # --- Tier-2 scope: the mitigation_commitment rows, deduped by condition_text ---
    scope = cond[cond["condition_role"].isin(LLM_ROLES)].copy()
    scope["_key"] = scope["condition_text"].map(sha256_text)
    uniq = scope.drop_duplicates("_key")
    n_scope, n_uniq = len(scope), len(uniq)
    print(f"[retag] Tier-2 scope: {n_scope} {LLM_ROLES} rows -> {n_uniq} unique texts "
          f"({100*n_uniq/max(n_scope,1):.1f}%; dedupe saves {n_scope-n_uniq})")

    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    todo = uniq[~uniq["_key"].isin(cache)]
    n_todo = len(todo)

    # --- exact metered cost projection (from the ACTUAL uncached texts) ---
    in_rate, out_rate = enrich_lib.pricing_for(args.model)
    PROMPT_OVERHEAD_TOK = 210          # fixed instruction tokens per call (measured from _prompt scaffold)
    OUT_TOK = 22                        # multi-label JSON
    in_tok = sum(PROMPT_OVERHEAD_TOK + max(1, len(str(t)) // 4) for t in todo["condition_text"])
    out_tok = OUT_TOK * n_todo
    cost = in_tok / 1e6 * in_rate + out_tok / 1e6 * out_rate
    print(f"\n[retag] Tier-2 metered projection ({args.model} @ ${in_rate}/${out_rate} per M):")
    print(f"        uncached unique calls = {n_todo}  (~{in_tok:,} in tok, ~{out_tok:,} out tok)")
    print(f"        EXACT PROJECTED COST  = ${cost:.2f}")

    if args.dry_run:
        print("\n[retag] --dry-run: no API call, no write. To execute (BILLABLE, user-launched):")
        print(f"        conda run -n nepa python phase2/code/deliverable06/retag_condition_resources.py --run --workers {args.workers}")
        return

    # ---------------- BILLABLE PATH (Phase 6, user-launched) ----------------
    if n_todo:
        import anthropic
        client = anthropic.Anthropic(api_key=enrich_lib.get_anthropic_key(), max_retries=4)
        results: dict[str, list[str]] = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_call_haiku, str(text), args.model, client): key
                    for text, key in zip(todo["condition_text"], todo["_key"])}
            for i, fut in enumerate(as_completed(futs), 1):
                k = futs[fut]
                try:
                    results[k] = fut.result()
                except Exception as e:  # noqa: BLE001
                    results[k] = []
                    print(f"[retag] call failed ({k[:8]}): {e}")
                if i % 500 == 0:
                    print(f"[retag] {i}/{n_todo}")
        cache.update({k: {"resource_areas": v} for k, v in results.items()})
        CACHE.write_text(json.dumps(cache))
        print(f"[retag] cache updated -> {CACHE} ({len(cache)} entries)")

    # --- apply: Haiku (verbatim) for commitment cache-hits; baseline tag otherwise ---
    run_at = utc_now()
    cond["_key"] = cond["condition_text"].map(sha256_text)
    def resolve(key, role, base):
        # commitment rows Haiku actually looked at: trust Haiku verbatim — an EMPTY list means Haiku
        # judged there is no resource area, which we now respect (was previously overridden by Tier-1).
        if role in LLM_ROLES and key in cache:
            return cache[key].get("resource_areas", [])
        # everything else: the baseline tag (keyword; keyword+heading only under --use-tier1)
        return [base] if base != "unknown" else []
    cond["resource_areas_multi"] = [
        ",".join(resolve(k, r, b)) for k, r, b in zip(cond["_key"], cond["condition_role"], cond["_base"])
    ]
    cond["resource_area"] = [m.split(",")[0] if m else "unknown" for m in cond["resource_areas_multi"]]
    cond["resource_retag_extraction_run_at"] = run_at
    cond["resource_retag_llm_run_at"] = [
        (run_at if (r in LLM_ROLES and k in cache) else "")
        for r, k in zip(cond["condition_role"], cond["_key"])
    ]
    cond = cond.drop(columns=["_base_kw", "_base", "_key"])
    final_unknown = int((cond["resource_area"] == "unknown").sum())
    write_parquet(cond, CONDITIONS)
    print(f"[retag] rewrote {CONDITIONS} — unknown {base_unknown} -> {final_unknown} "
          f"({100*final_unknown/n:.1f}%). Next: D6 05->09->08.R, then D2 02 join + 05 (all $0).")


if __name__ == "__main__":
    main()
