"""D2 Phase 3 — FONSI significance extraction (plan v2.11 §5).

Pipeline: deterministic candidate generation (candidate_gen) -> frozen mitigation page-window
join (mitigation_signal_matches) -> determination records (+ threshold child) via the shared
`extract_common` assembly -> optional LLM adjudication on bounded windows.

Two run modes:
  --dry-run  (KEY-FREE): builds candidates, the mitigation join, and a deterministic
             determination table (extraction_method='regex') so the whole pipeline is runnable
             WITHOUT the billable API. This is what CI / the assistant runs.
  (real)     without --dry-run: adjudicates each window with the pinned Claude model
             (extraction_method='regex+llm'), temp 0. REQUIRES the Anthropic key
             (keychain 'nepa-anthropic' / ANTHROPIC_API_KEY) + budget approval. The USER runs it.

Run (key-free):  conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --dry-run
Run (billable):  conda run -n nepa python phase2/code/deliverable02/02_extract_fonsi_significance.py --model claude-haiku-4-5-20251001
"""
from __future__ import annotations

import argparse

import pandas as pd

import common as C
import extract_common as X
from candidate_gen import OBLIG_SQL, ROLES_SQL, generate_fonsi_candidates


# --- D2-2: condition <-> impact resource matching rule (D6 #47) -------------------------------
# D6's re-tag (retag_condition_resources.py) gave every mitigation condition a MULTI-LABEL
# `resource_areas_multi` alongside the legacy single-label `resource_area`. A commitment to
# "prevent degradation of adjacent water sources and fisheries habitat" is genuinely water AND
# biological; the old scalar column had to pick one and dropped the other.
#
#   "any-overlap" (DEFAULT) — a condition contributes ALL of its resource areas to the window's
#       set, so a determination matches if ANY of the condition's areas is the impact's area.
#       This is the honest rule under multi-label tagging: it stops a correctly-broad commitment
#       from being penalised for the arbitrary choice of which label happened to be primary.
#   "primary"  — legacy behaviour: only the single `resource_area` scalar counts. Kept so the
#       before/after effect of the re-tag is measurable rather than asserted.
#
# NOTE this widens the window's area set, so it can only ADD matches, never remove them. That
# direction is intentional but it is NOT self-validating: a wider set also converts a wrong tag
# into a wrong match. The condition-side precision that justifies it is measured by
# phase2/code/deliverable06/build_retag_validation_sample.py, which is UNLABELED as of this
# writing. Until those human labels exist, treat any resource-level movement here as mechanical.
MATCHING_RULES = ("any-overlap", "primary")
DEFAULT_MATCHING_RULE = "any-overlap"


def _multi_sql(rule: str) -> str:
    """SQL expression yielding the condition's contributed resource areas under `rule`."""
    if rule == "primary":
        return "resource_area"
    # coalesce: rows outside the re-tag's scope have an empty resource_areas_multi
    return ("CASE WHEN resource_areas_multi IS NULL OR resource_areas_multi = '' "
            "THEN resource_area ELSE resource_areas_multi END")


def mitigation_signal_matches(rule: str = DEFAULT_MATCHING_RULE) -> pd.DataFrame:
    """cue finding-span x qualifying condition-row, same-section OR page-window +/-2 (frozen §3)."""
    if rule not in MATCHING_RULES:
        raise ValueError(f"matching rule must be one of {MATCHING_RULES}, got {rule!r}")
    MULTI_SQL = _multi_sql(rule)
    sql = f"""
    WITH corpus_ea AS (
        SELECT project_id FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') WHERE process_type = 'EA'
    ),
    cue AS (
        SELECT project_id, document_id, section_id, evidence_span_id, page_start, page_end
        FROM read_parquet('{C.FONSI_SPANS}')
        JOIN corpus_ea USING (project_id)
        WHERE span_type = 'finding' AND manifest_role IN ('linked_ea','canonical_fonsi','supporting_fonsi')
    ),
    qc AS (
        SELECT project_id, document_id, section_id, page_number, resource_area,
               {MULTI_SQL} AS resource_areas_multi,
               condition_role, obligation_level, source_span_sha256, condition_text
        FROM read_parquet('{C.FONSI_CONDITIONS}')
        WHERE condition_role IN ({ROLES_SQL}) AND obligation_level IN ({OBLIG_SQL})
    )
    SELECT cue.evidence_span_id AS cue_evidence_span_id, cue.project_id, cue.document_id,
           cue.section_id AS cue_section_id, cue.page_start AS cue_page_start,
           cue.page_end AS cue_page_end,
           qc.section_id AS condition_section_id, qc.page_number AS condition_page_number,
           qc.resource_area, qc.resource_areas_multi, qc.condition_role, qc.obligation_level,
           qc.source_span_sha256, qc.condition_text,
           CASE WHEN cue.section_id = qc.section_id THEN 'same_section' ELSE 'windowed' END AS match_type
    FROM cue JOIN qc
      ON cue.project_id = qc.project_id AND cue.document_id = qc.document_id
     AND (cue.section_id = qc.section_id
          OR qc.page_number BETWEEN cue.page_start - 2 AND cue.page_end + 2)
    """
    m = C.q(sql)
    if m.empty:
        return m
    m["condition_row_id"] = [
        C.sha256_join(p, d, s, pg, ss, cr, ob, ra) for p, d, s, pg, ss, cr, ob, ra in zip(
            m.project_id, m.document_id, m.condition_section_id, m.condition_page_number,
            m.source_span_sha256, m.condition_role, m.obligation_level, m.resource_area)]
    m["condition_text_sha256"] = m["condition_text"].map(C.sha256_text)
    return m.drop(columns=["condition_text"])


def mitigation_summary(matches: pd.DataFrame) -> pd.DataFrame:
    """Per cue finding-span (=source_unit_id): matched count + role/obligation sets."""
    if matches.empty:
        return pd.DataFrame(columns=["source_unit_id", "matched_condition_row_count",
                                     "condition_role_set", "obligation_level_set",
                                     "mitigation_resource_areas",
                                     "mitigation_resource_areas_primary",
                                     "mitigation_same_section"])
    g = matches.groupby("cue_evidence_span_id")

    def _union_areas(s: pd.Series) -> str:
        """Union every area each matched condition contributes (D2-2 any-overlap).

        Under rule='primary' each cell holds one scalar, so this reduces to the legacy behaviour;
        under 'any-overlap' a cell may hold 'water,biological' and both join the window's set.
        """
        out: set[str] = set()
        for v in s:
            out.update(a.strip().lower() for a in str(v or "").split(",") if a.strip())
        return ",".join(sorted(out))

    return pd.DataFrame({
        "matched_condition_row_count": g["condition_text_sha256"].nunique(),
        "condition_role_set": g["condition_role"].apply(lambda s: ",".join(sorted(set(s)))),
        "obligation_level_set": g["obligation_level"].apply(lambda s: ",".join(sorted(set(s)))),
        "mitigation_resource_areas": g["resource_areas_multi"].apply(_union_areas),
        # legacy single-label union, carried alongside so the rules stay directly comparable
        "mitigation_resource_areas_primary": g["resource_area"].apply(_union_areas),
        "mitigation_same_section": g["match_type"].apply(lambda s: "same_section" in set(s)),
    }).reset_index().rename(columns={"cue_evidence_span_id": "source_unit_id"})


def project_context() -> pd.DataFrame:
    return C.q(f"""
    SELECT c.project_id, c.doc_type AS doc_type, c.process_type, c.agency, c.agency_scope_status,
           c.agency_scope_rule, c.time_scope_status, c.analysis_scope,
           r.decision_date, r.decision_source_type, r.decision_confidence, r.decision_is_proxy,
           r.decision_period, r.applicability_period, r.fra_overlay, r.regime_assignment_status,
           coh.cohort_by_date
    FROM read_parquet('{C.SIGNIFICANCE_CORPUS}') c
    LEFT JOIN read_parquet('{C.PROJECT_REGIME}') r USING (project_id)
    LEFT JOIN read_parquet('{C.PROJECT_COHORTS}') coh USING (project_id)
    WHERE c.process_type = 'EA'
    """)


def rejoin_mitigation(rule: str = DEFAULT_MATCHING_RULE,
                      dep_rule: str = X.DEFAULT_MITIGATION_DEP_RULE) -> None:
    """Recompute ONLY the mitigation columns on the existing determinations. $0, key-free.

    `dep_rule` selects how `mitigation_dependent` (the labeled SCREENING metric) is derived —
    the tightened T5 rule by default (#53), or the legacy any-overlap rule for comparison. It does
    NOT touch `mitigation_resource_matched` (the any-overlap resource-match reporting column that
    the report's aggregate finding reads) — only the screening flag changes.

    WHY THIS MODE EXISTS
    --------------------
    D6's #47 re-tag changed `fonsi_conditions.resource_area`, so D2's impact<->mitigation join has
    to be recomputed. But the two normal ways to rebuild `significance_determinations.parquet` are
    both unacceptable here:

      --dry-run     rebuilds determinations from regex only. It would overwrite 7,249 rows of
                    LLM-adjudicated output (extraction_method='regex+llm') with regex guesses —
                    destructive, and unrecoverable without re-buying the whole pass.
      --batch-fetch re-reads the Message Batch results. Retrieval is free, but it opens the
                    Anthropic client (macOS Keychain prompt) and batch results expire after ~29
                    days, so it is not a reliable $0 path months later.

    The mitigation columns are a DETERMINISTIC function of the join plus fields already stored on
    each determination row, so they can be recomputed in place without touching the LLM at all:

        resource_mitigation_match = mitigation_flag AND (scope=='project_overall' OR resource in areas)
        mitigation_dependent      = derive_mitigation_dependent(...)   # single source of truth in
            extract_common.py, imported here so the scorer and the shipped column cannot diverge.
            Default rule 't5-specific-2cond' (#53): (literal same-resource overlap AND >=2 matched
            conditions) OR class=='less_than_significant_with_mitigation'. The legacy
            'baseline-any-overlap' formula (resource_mitigation_match OR class==LTS, precision ~0.41)
            survives only as a scoring variant, not as this path's default.

    Everything the LLM produced (determination_class, shared_resource_area, rationale, thresholds)
    is read, never written. `determination_instance_id` does not hash any mitigation field, so ids
    are stable across this operation.
    """
    matches = mitigation_signal_matches(rule)
    C.write_parquet(matches, C.MITIGATION_SIGNAL_MATCHES, "mitigation matches")
    mit = mitigation_summary(matches)

    dets = pd.read_parquet(C.SIGNIFICANCE_DETERMINATIONS)
    before = {
        "mitigation_flag": int(dets["mitigation_flag"].sum()),
        "mitigation_resource_matched": int(dets["mitigation_resource_matched"].sum()),
        "mitigation_dependent": int(dets["mitigation_dependent"].sum()),
    }
    method = dets["extraction_method"].value_counts().to_dict()

    drop = [c for c in ("matched_condition_row_count", "condition_role_set", "obligation_level_set",
                        "mitigation_resource_areas", "mitigation_resource_areas_primary",
                        "mitigation_same_section") if c in dets.columns]
    out = dets.drop(columns=drop).merge(mit, on="source_unit_id", how="left")
    out["matched_condition_row_count"] = out["matched_condition_row_count"].fillna(0).astype(int)
    for col in ("condition_role_set", "obligation_level_set", "mitigation_resource_areas",
                "mitigation_resource_areas_primary"):
        out[col] = out[col].fillna("")
    out["mitigation_flag"] = out["matched_condition_row_count"] > 0

    areas = [{t.strip().lower() for t in str(v or "").replace(";", ",").split(",") if t.strip()}
             for v in out["mitigation_resource_areas"]]
    out["mitigation_resource_matched"] = [
        bool(f) and (sc == "project_overall" or res in a)
        for f, sc, res, a in zip(out["mitigation_flag"], out["determination_scope"],
                                 out["shared_resource_area"], areas)]
    # labeled SCREENING metric: T5 tightening by default (#53), legacy rule behind dep_rule flag.
    # Shared source of truth with extract_common so the scorer and the shipped column can't diverge.
    out["mitigation_dependent"] = [
        X.derive_mitigation_dependent(
            dclass=c, resource_mitigation_match=m, mitigation_flag=f,
            shared_resource_area=res, mitigation_resource_areas_set=a,
            matched_condition_row_count=cnt, rule=dep_rule)
        for c, m, f, res, a, cnt in zip(
            out["determination_class"], out["mitigation_resource_matched"], out["mitigation_flag"],
            out["shared_resource_area"], areas, out["matched_condition_row_count"])]
    out["mitigation_enforceability"] = ["permit_condition" if m else "none"
                                        for m in out["mitigation_resource_matched"]]

    assert out["extraction_method"].value_counts().to_dict() == method, \
        "rejoin must not change extraction_method — LLM output was touched"
    assert len(out) == len(dets), f"row count moved {len(dets)} -> {len(out)}"

    # preserve the original column order, appending the comparison column once (it is already
    # present when re-joining a file this mode has written before — do not duplicate it)
    cols = dets.columns.tolist()
    if "mitigation_resource_areas_primary" not in cols:
        cols.append("mitigation_resource_areas_primary")
    C.write_parquet(out[cols], C.SIGNIFICANCE_DETERMINATIONS, "determinations (mitigation re-join)")
    print(f"\n[02 rejoin] matching-rule={rule}  mitigation-dep-rule={dep_rule}  rows={len(out):,}  "
          f"extraction_method preserved: {method}")
    for k, v0 in before.items():
        v1 = int(out[k].sum())
        print(f"[02 rejoin]   {k}: {v0:,} -> {v1:,} ({v1 - v0:+,})")

    # Keep the run manifest's content hashes in sync with the files this rejoin just rewrote.
    # Rejoin never calls the LLM, so the LLM provenance (mode='llm', the extraction model) is
    # preserved from the prior manifest — only the integrity hashes/n_bytes and run_at are refreshed.
    # Fixes the stale-hash gap where a rejoin updated determinations/mitigation_matches but left the
    # manifest pinned at the original LLM-run timestamp.
    prior_model = X.DEFAULT_MODEL
    if C.RUN_MANIFEST.exists():
        try:
            _prior = pd.read_parquet(C.RUN_MANIFEST)
            if len(_prior) and str(_prior["model"].iloc[0]):
                prior_model = str(_prior["model"].iloc[0])
        except Exception:
            pass
    X.write_manifest({
        "significance_section_candidates": "data/analysis/deliverable02/significance_section_candidates.parquet",
        "mitigation_signal_matches": "data/analysis/deliverable02/mitigation_signal_matches.parquet",
        "significance_determinations": "data/analysis/deliverable02/significance_determinations.parquet",
        "determination_thresholds": "data/analysis/deliverable02/determination_thresholds.parquet",
    }, dry_run=False, model=prior_model)
    print(f"[02 rejoin] manifest refreshed ({C.RUN_MANIFEST.name}); model={prior_model} preserved")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="key-free deterministic pass")
    ap.add_argument("--model", default=X.DEFAULT_MODEL)
    ap.add_argument("--sample", type=int, default=0, help="limit candidates (debug)")
    ap.add_argument("--rejoin-mitigation", action="store_true",
                    help="$0 KEY-FREE: recompute only the mitigation columns on the existing "
                         "determinations (after a D6 condition re-tag). Preserves all LLM output.")
    ap.add_argument("--matching-rule", choices=MATCHING_RULES, default=DEFAULT_MATCHING_RULE,
                    help="how a condition's resource area matches an impact's (D2-2). "
                         "'any-overlap' (default) unions the D6 multi-label tags; "
                         "'primary' is the legacy single-label behaviour, kept for comparison.")
    ap.add_argument("--mitigation-dep-rule", choices=X.MITIGATION_DEP_RULES,
                    default=X.DEFAULT_MITIGATION_DEP_RULE,
                    help="how the labeled SCREENING metric `mitigation_dependent` is derived (#53). "
                         "'t5-specific-2cond' (default) is the tightened rule (F1 0.622/prec 0.53); "
                         "'baseline-any-overlap' is the legacy rule (F1 0.566/prec 0.41), kept for "
                         "comparison. Does not affect the any-overlap resource-match reporting column.")
    ap.add_argument("--batch-run", action="store_true",
                    help="ONE-PASSWORD batch: submit + poll + fetch + build, all in this process")
    ap.add_argument("--batch-submit", action="store_true",
                    help="submit windows as Message Batch(es) (50%% price) and exit")
    ap.add_argument("--batch-fetch", action="store_true",
                    help="retrieve the submitted batch and build determinations")
    ap.add_argument("--wait", action="store_true", help="with --batch-fetch: poll until ended")
    args = ap.parse_args()
    mode = ("REJOIN-MITIGATION ($0, key-free, LLM output preserved)" if args.rejoin_mitigation
            else "BATCH-RUN (one password, submit+poll+fetch)" if args.batch_run
            else "BATCH-SUBMIT" if args.batch_submit else "BATCH-FETCH" if args.batch_fetch
            else "DRY-RUN (regex only, key-free)" if args.dry_run else f"LLM sync ({args.model})")
    print(f"D2 Phase 3: FONSI significance extraction — {mode}")

    if args.rejoin_mitigation:
        rejoin_mitigation(args.matching_rule, args.mitigation_dep_rule)
        return

    if args.batch_fetch:
        cand, results, model = X.fetch_batch("fonsi", args.wait)
    else:
        cand = generate_fonsi_candidates()
        if args.sample:
            cand = cand.sort_values("evidence_text_sha256").head(args.sample)
        C.write_parquet(cand, C.SIGNIFICANCE_SECTION_CANDIDATES, "candidates")
        results, model = None, args.model

    matches = mitigation_signal_matches(args.matching_rule)
    print(f"[02] condition<->impact matching rule: {args.matching_rule}")
    C.write_parquet(matches, C.MITIGATION_SIGNAL_MATCHES, "mitigation matches")

    if args.batch_submit or args.batch_run:
        X.submit_batch(cand, args.model, "fonsi")   # key read once & cached for this process
        if not args.batch_run:
            return
        cand, results, model = X.fetch_batch("fonsi", wait=True)

    dets, thr = X.build_determinations(cand, mitigation_summary(matches),
                                       project_context(), args.dry_run, model,
                                       llm_results=results)
    C.write_parquet(dets, C.SIGNIFICANCE_DETERMINATIONS, "determinations")
    C.write_parquet(thr, C.DETERMINATION_THRESHOLDS, "threshold child")

    X.write_manifest({
        "significance_section_candidates": "data/analysis/deliverable02/significance_section_candidates.parquet",
        "mitigation_signal_matches": "data/analysis/deliverable02/mitigation_signal_matches.parquet",
        "significance_determinations": "data/analysis/deliverable02/significance_determinations.parquet",
        "determination_thresholds": "data/analysis/deliverable02/determination_thresholds.parquet",
    }, args.dry_run, args.model)

    print(f"\ncandidates={len(cand):,}  mitigation_matches={len(matches):,}  "
          f"determinations={len(dets):,}  threshold_rows={len(thr):,}")
    print(f"mitigation_flag=TRUE: {int(dets['mitigation_flag'].sum()):,}")
    print("\ndetermination_class:\n" + dets["determination_class"].value_counts().to_string())
    if args.dry_run:
        print("\n[dry-run] every row extraction_method='regex', needs_human_review=TRUE. "
              "Run WITHOUT --dry-run (billable) to adjudicate with the LLM.")


if __name__ == "__main__":
    main()
