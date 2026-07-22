"""D6 #47 — validation gate for the condition -> resource_area re-tag.

WHY THIS EXISTS
---------------
`retag_condition_resources.py` replaced the pure keyword-dictionary `resource_area` tags on
`fonsi_conditions.parquet` with Tier-1 (section-heading inheritance) + Tier-2 (scoped Haiku
multi-label) tags. That moved mitigation-commitment unknowns 5,117 -> 1,993 and added the
`resource_areas_multi` column.

Nothing has yet established whether the NEW tags are actually *right*. The existing "33% precision"
figure came from a 6-row eyeball; the "F1 ~= 0.43 cap" quoted in the plan is an estimate, never a
measured quantity (see the honest-accounting note at the bottom of this docstring). D2's gold set
(`deliverable02/gold/significance_gold.parquet`) labels the IMPACT/determination side only — it has
no column describing which resource a *mitigation condition* protects. So there is no existing
ground truth that can score this re-tag, and none can be derived from what we have.

This script builds that missing ground truth: a deterministic, stratified hand-labeling sheet, plus
the scorer that reads it back. THE AGENT DOES NOT LABEL IT — a human does. Everything here is $0
and offline; no API key is ever touched.

WHAT IT DOES
------------
--build   Draws a stratified sample of mitigation_commitment conditions (the rows that feed D2's
          impact<->mitigation join) and writes a CSV + a readable markdown worksheet with EMPTY
          human columns, to phase2/notes/deliverable06/.
--score   Reads the filled sheet back and reports precision/recall/F1 overall and per stratum, plus
          an old-tag vs new-tag comparison so the user can see whether the re-tag actually helped.
--self-test  Fabricates labels on the drawn sample and runs the scorer end-to-end, so the scorer is
          known-working BEFORE the user spends an hour labeling. The synthetic labels are noise;
          only the fact that the scorer runs and its arithmetic is right is meaningful.

STRATA (reconstructed deterministically, no state needed)
---------------------------------------------------------
The old keyword tag is recomputed from `classify_resource_area(condition_text)` — the same pure
function that produced the pre-retag column — so "what changed" is derivable without a snapshot of
the old parquet. Tier-1 is recomputed the same way via `classify_resource_area_with_heading`.

  new_haiku      old was 'unknown'; Haiku supplied label(s)         (a pure gain if right)
  new_tier1      old was 'unknown'; section-heading rule supplied it (free tier, ~72% pilot precision)
  changed        old was tagged; the re-tag CHANGED the primary area (can help OR hurt)
  unchanged      old was tagged; the re-tag kept the same primary    (regression check)
  still_unknown  no tier resolved it; still 'unknown'                (is 'unknown' the right answer?)

Within each stratum the draw is spread across the common resource areas, and multi-label rows
(>=2 areas in `resource_areas_multi`) are deliberately over-sampled into their own cells so the
any-overlap matching rule D2 adopts can actually be evaluated.

ID STABILITY
------------
`condition_id` here is sha256 of (project_id, document_id, section_id, page_number,
source_span_sha256) — deliberately NOT including `resource_area`. D2's own `condition_row_id` DOES
hash `resource_area`, so it churns on every re-tag; this id does not, and a filled sheet stays
joinable across future re-tags.

HONEST ACCOUNTING (read before quoting any number from this)
------------------------------------------------------------
This sheet measures ONE thing: the accuracy of condition-side resource tagging. It does not, by
itself, produce D2's `mitigation_dependent_f1` (currently 0.612 overall / 0.623 holdout, scored
against D2's gold). Those are different metrics on different grains. A good score here is the
*precondition* for upgrading D2's resource-level mitigation caveat to a finding — it is not the
upgrade itself.

USAGE
  conda run -n nepa python phase2/code/deliverable06/build_retag_validation_sample.py --build
  conda run -n nepa python phase2/code/deliverable06/build_retag_validation_sample.py --self-test
  conda run -n nepa python phase2/code/deliverable06/build_retag_validation_sample.py --score
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

import duckdb
import numpy as np
import pandas as pd

from common import D6_ANALYSIS_DIR, D6_RAW_DIR, sha256_text

sys.path.insert(0, str((D6_ANALYSIS_DIR.parents[2] / "code" / "extract")))
from mitigation_conditions import (  # noqa: E402
    classify_resource_area,
    classify_resource_area_with_heading,
)

CONDITIONS = D6_ANALYSIS_DIR / "fonsi_conditions.parquet"
SPANS = D6_ANALYSIS_DIR / "fonsi_evidence_spans.parquet"
# written by retag_condition_resources.py; keyed by sha256(condition_text)
CACHE = D6_RAW_DIR / "resource_retag_cache.json"
NOTES_DIR = D6_ANALYSIS_DIR.parents[2] / "notes" / "deliverable06"
SHEET_CSV = NOTES_DIR / "retag_validation_sheet.csv"
SHEET_MD = NOTES_DIR / "retag_validation_worksheet.md"
SCORE_MD = NOTES_DIR / "retag_validation_score.md"

SEED = 20260722
TARGET_N = 80

SHARED_12 = [
    "air_quality", "water", "biological", "cultural", "visual", "noise",
    "soils_geology", "socioeconomic", "transportation", "land_use",
    "climate_ghg", "public_health",
]
VALID = set(SHARED_12) | {"unknown"}

# per-stratum target counts (sum = TARGET_N). still_unknown and changed get extra weight: they are
# the two strata where a wrong re-tag actively costs D2 something.
STRATUM_TARGETS = {
    "new_haiku": 24,
    "new_tier1": 10,
    "changed": 18,
    "unchanged": 14,
    "still_unknown": 14,
}
# of the new_haiku / changed allocations, this many are forced to be multi-label rows
MULTI_LABEL_FORCED = {"new_haiku": 8, "changed": 6}

HUMAN_COLS = ["gold_resource_areas", "is_correct", "notes"]


# --------------------------------------------------------------------------- build

def _heading_map() -> dict[str, str]:
    con = duckdb.connect()
    rows = con.execute(
        f"SELECT DISTINCT section_id, heading_title FROM '{SPANS}' "
        "WHERE heading_title IS NOT NULL AND heading_title <> ''"
    ).fetchall()
    return {sid: h for sid, h in rows}


def _condition_id(r) -> str:
    raw = "|".join(str(x) for x in (
        r["project_id"], r["document_id"], r["section_id"], r["page_number"], r["source_span_sha256"]))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def load_frame() -> pd.DataFrame:
    """mitigation_commitment rows, with old tag + tier provenance reconstructed."""
    cond = pd.read_parquet(CONDITIONS)
    df = cond[cond["condition_role"] == "mitigation_commitment"].copy()
    hmap = _heading_map()

    df["old_tag"] = [classify_resource_area(str(t)) for t in df["condition_text"]]
    df["tier1_tag"] = [
        old if old != "unknown" else classify_resource_area_with_heading(str(t), hmap.get(sid, ""))
        for old, t, sid in zip(df["old_tag"], df["condition_text"], df["section_id"])
    ]
    df["new_multi"] = df["resource_areas_multi"].fillna("").astype(str)
    df["new_primary"] = df["resource_area"].astype(str)
    df["n_labels"] = [len([a for a in m.split(",") if a]) for m in df["new_multi"]]
    df["condition_id"] = df.apply(_condition_id, axis=1)
    df["text_sha"] = df["condition_text"].map(sha256_text)

    # Provenance, read straight off the Haiku cache rather than inferred. retag_condition_resources
    # keys the cache by sha256(condition_text) and applies the cached labels when the entry exists
    # AND is non-empty; otherwise it falls back to [tier1_tag] when that is not 'unknown'. So a row
    # is Haiku-sourced iff its text_sha has a non-empty cache entry. Inferring this from the label
    # values instead silently folds every Tier-1 resolution into the Haiku stratum.
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    if not cache:
        print(f"[build] WARNING no retag cache at {CACHE} — Haiku/Tier-1 provenance unavailable")
    df["from_haiku"] = [bool(cache.get(k, {}).get("resource_areas")) for k in df["text_sha"]]

    def stratum(row) -> str:
        if row["new_primary"] == "unknown":
            return "still_unknown"
        if row["old_tag"] == "unknown":
            return "new_haiku" if row["from_haiku"] else "new_tier1"
        return "unchanged" if row["new_primary"] == row["old_tag"] else "changed"

    df["stratum"] = [stratum(r) for _, r in df.iterrows()]
    # Dedupe on identical condition text BEFORE drawing: labeling the same sentence twice buys
    # nothing, and deduping after the draw silently undershoots the target sample size.
    df = df.drop_duplicates("text_sha").reset_index(drop=True)
    return df


def draw(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic stratified draw: by stratum, spread over resource areas, multi-label quota."""
    rng = np.random.RandomState(SEED)
    picked: list[pd.DataFrame] = []

    for stratum, target in STRATUM_TARGETS.items():
        pool = df[df["stratum"] == stratum]
        if pool.empty:
            print(f"[build] WARNING stratum '{stratum}' is empty — skipping")
            continue
        n_multi = MULTI_LABEL_FORCED.get(stratum, 0)
        chosen_idx: list = []

        if n_multi:
            mpool = pool[pool["n_labels"] >= 2]
            take = min(n_multi, len(mpool))
            if take:
                chosen_idx += list(_spread(mpool, take, rng))
            if take < n_multi:
                print(f"[build] note: '{stratum}' had only {len(mpool)} multi-label rows "
                      f"(wanted {n_multi})")

        rest = pool.drop(index=chosen_idx)
        need = target - len(chosen_idx)
        if need > 0 and len(rest):
            chosen_idx += list(_spread(rest, min(need, len(rest)), rng))
        picked.append(pool.loc[chosen_idx].assign(stratum=stratum))

    # population is already deduped on text_sha in load_frame(), so no post-draw shrink here
    return pd.concat(picked).sort_values(["stratum", "condition_id"]).reset_index(drop=True)


def _spread(pool: pd.DataFrame, k: int, rng) -> list:
    """Pick k rows spread as evenly as possible across primary resource areas."""
    groups = {a: list(g.index) for a, g in pool.groupby("new_primary")}
    for a in groups:
        rng.shuffle(groups[a])
    order = sorted(groups, key=lambda a: (-len(groups[a]), a))  # deterministic
    out: list = []
    while len(out) < k:
        progressed = False
        for a in order:
            if groups[a] and len(out) < k:
                out.append(groups[a].pop())
                progressed = True
        if not progressed:
            break
    return out


INSTRUCTIONS = """\
# D6 #47 — condition resource-area re-tag: hand-labeling worksheet

**You are the ground truth.** Nothing in this file has been labeled by a model. Fill the three
`gold_*` / `is_correct` / `notes` columns in `retag_validation_sheet.csv` (open it in Excel/Numbers
or any CSV editor), then run the scorer. Budget ~45-60 minutes for ~80 rows.

## What you are judging

Each row is one **mitigation commitment** sentence pulled from a FONSI. Your job: decide which
environmental **resource area(s)** that commitment is protecting. You are NOT judging whether the
mitigation is good, enforceable, or well-written — only *what it protects*.

## The 12 resource areas (the only allowed values, plus `unknown`)

| value | means |
|---|---|
| `air_quality` | air emissions, dust, fugitive particulates, odor |
| `water` | surface water, groundwater, wetlands, stormwater, water quality, hydrology |
| `biological` | wildlife, fish, plants, vegetation, habitat, T&E species, migratory birds |
| `cultural` | historic properties, archaeology, tribal/sacred sites, Section 106 |
| `visual` | scenic quality, viewshed, lighting/glare, aesthetics |
| `noise` | acoustics, sound levels, vibration |
| `soils_geology` | soil, erosion, sediment control, geology, seismicity, paleontology |
| `socioeconomic` | jobs, housing, environmental justice, community/economic effects |
| `transportation` | traffic, roads, access, haul routes, parking |
| `land_use` | zoning, land ownership, easements, recreation, farmland, right-of-way |
| `climate_ghg` | greenhouse gases, carbon, climate resilience |
| `public_health` | human health/safety, hazardous materials, contamination, spill response, waste |
| `unknown` | **no** resource area applies (see below) |

## How to fill each column

**`gold_resource_areas`** — comma-separated, no spaces, lowercase. Examples:
- `biological` (single)
- `water,biological` (multi — see the multi-label rule)
- `unknown` (see the unknown rule)

**Multi-label rule.** List *every* area the commitment genuinely protects, not just the most
prominent one. A commitment to "prevent degradation of adjacent water sources and fisheries
habitat" is `water,biological` — both, because a downstream match on **either** should count.
But do not pad: only list an area if the sentence actually commits to protecting it. Incidental
mentions don't count ("the access road near the wetland will be graded" is `soils_geology` or
`transportation`, not `water`, unless it commits to protecting the wetland).

**Unknown rule.** Write `unknown` when the sentence has **no** resource area — it is procedural,
legal, administrative, or boilerplate. Real examples of correct `unknown`:
- "The applicant shall indemnify the agency against all claims."
- "An EIS is not required for this action."
- "This decision may be appealed within 30 days."
`unknown` is a **legitimate right answer**, not a cop-out. Marking a boilerplate row `unknown` when
the pipeline also said `unknown` is a *correct* prediction and the scorer credits it as such.
If the sentence is truncated or too garbled to judge, put `unknown` and say so in `notes` — the
scorer reports those separately so they don't silently distort precision.

**`is_correct`** — your holistic verdict on the pipeline's `new_tags` for this row. One of:
- `yes` — new_tags is right (exactly, or close enough that a downstream match would be correct)
- `partial` — new_tags gets some areas right but misses one, or adds one that doesn't belong
- `no` — new_tags is wrong
This is redundant with `gold_resource_areas` on purpose: it's a sanity check on the scorer's
set-arithmetic, and it catches "technically overlapping but substantively wrong" cases.

**`notes`** — free text, optional. Use it for anything ambiguous, and especially when you disagree
with the taxonomy itself (e.g. a commitment that protects something none of the 12 cover).

## Reading the pre-filled columns

- `old_tag` — what the OLD pure-keyword dictionary said (single label, `unknown` if no keyword hit)
- `new_tags` — what the re-tag produced (comma-separated, possibly multi-label)
- `stratum` — which change-type this row was sampled from:
  - `new_haiku` — was `unknown`, Haiku gave it label(s)
  - `new_tier1` — was `unknown`, the free section-heading rule gave it a label
  - `changed` — was tagged, the re-tag changed the primary area
  - `unchanged` — was tagged, the re-tag agreed
  - `still_unknown` — still `unknown` after both tiers

**Label blind if you can.** The honest way to do this is to read `condition_text`, decide, and only
*then* look at `old_tag`/`new_tags`. If you read the prediction first you will anchor on it and the
precision estimate will come out too high. Consider hiding those columns while you work.

## When you are done

Save the CSV (keep it as CSV, keep the header row), then run:

```
conda run -n nepa python phase2/code/deliverable06/build_retag_validation_sample.py --score
```

The scorer prints precision / recall / F1 overall and per stratum, and an old-tag vs new-tag
comparison so you can see whether the re-tag actually helped. It writes the same report to
`phase2/notes/deliverable06/retag_validation_score.md`.

---
"""


def build() -> None:
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_frame()
    print(f"[build] mitigation_commitment rows: {len(df)}")
    print("[build] stratum sizes in the full population:")
    print(df["stratum"].value_counts().to_string())
    print(f"[build] multi-label (>=2 areas) rows: {int((df['n_labels'] >= 2).sum())}")

    s = draw(df)
    sheet = pd.DataFrame({
        "condition_id": s["condition_id"],
        "stratum": s["stratum"],
        "old_tag": s["old_tag"],
        "new_tags": s["new_multi"].where(s["new_multi"] != "", "unknown"),
        "condition_text": s["condition_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip(),
        "gold_resource_areas": "",
        "is_correct": "",
        "notes": "",
        "project_id": s["project_id"],
        "text_sha": s["text_sha"],
    })
    sheet.to_csv(SHEET_CSV, index=False)
    print(f"\n[build] wrote {SHEET_CSV}  ({len(sheet)} rows)")
    print(sheet["stratum"].value_counts().to_string())

    # readable worksheet
    lines = [INSTRUCTIONS, f"\n**Sheet:** `{SHEET_CSV}` — {len(sheet)} rows to label.\n"]
    for stratum in STRATUM_TARGETS:
        sub = sheet[sheet["stratum"] == stratum]
        if sub.empty:
            continue
        lines.append(f"\n## Stratum: `{stratum}` ({len(sub)} rows)\n")
        for _, r in sub.iterrows():
            lines.append(
                f"### `{r.condition_id}`\n\n"
                f"> {r.condition_text}\n\n"
                f"- old_tag: `{r.old_tag}`\n"
                f"- new_tags: `{r.new_tags}`\n"
                f"- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______\n"
            )
    SHEET_MD.write_text("\n".join(lines))
    print(f"[build] wrote {SHEET_MD}")
    print("\n[build] NEXT: a human fills gold_resource_areas / is_correct / notes in the CSV, then:")
    print("        conda run -n nepa python phase2/code/deliverable06/build_retag_validation_sample.py --score")


# --------------------------------------------------------------------------- score

def _parse_set(v: object) -> set[str]:
    """'water,biological' -> {'water','biological'}; 'unknown'/'' -> empty set."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return set()
    parts = [p.strip().lower() for p in str(v).split(",")]
    return {p for p in parts if p and p != "unknown"}


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and (p + r)) else float("nan")
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn}


def _score_col(df: pd.DataFrame, pred_col: str) -> dict:
    """Micro P/R/F1 over (row, resource_area) pairs, 'unknown' excluded from both sides."""
    tp = fp = fn = 0
    exact = 0
    for _, r in df.iterrows():
        g, p = _parse_set(r["gold_resource_areas"]), _parse_set(r[pred_col])
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
        exact += int(g == p)
    m = _prf(tp, fp, fn)
    m["exact_set_match"] = round(exact / len(df), 3) if len(df) else float("nan")
    m["n"] = len(df)
    return m


def _unknown_metrics(df: pd.DataFrame, pred_col: str) -> dict:
    """How well does the pipeline decide 'this row has no resource area' at all?"""
    gold_unk = df["gold_resource_areas"].map(lambda v: len(_parse_set(v)) == 0)
    pred_unk = df[pred_col].map(lambda v: len(_parse_set(v)) == 0)
    both = int((gold_unk & pred_unk).sum())
    n_gu, n_pu = int(gold_unk.sum()), int(pred_unk.sum())
    return {
        "gold_unknown_rows": n_gu,
        "pred_unknown_rows": n_pu,
        # recall: of the rows that genuinely have no resource area, how many did we leave untagged.
        # Reported alone this is misleading — a pipeline that tagged NOTHING would score 1.0 — so
        # precision (of what we left untagged, how much genuinely had no area) is reported with it.
        "unknown_recall": round(both / n_gu, 3) if n_gu else float("nan"),
        "unknown_precision": round(both / n_pu, 3) if n_pu else float("nan"),
        "over_tagged": int((gold_unk & ~pred_unk).sum()),   # gold says none, pipeline invented one
        "under_tagged": int((~gold_unk & pred_unk).sum()),  # gold says something, pipeline said none
    }


def _any_overlap_rate(df: pd.DataFrame, pred_col: str) -> float:
    """Share of rows where >=1 predicted area is genuinely correct — the D2 any-overlap rule."""
    ok = 0
    n = 0
    for _, r in df.iterrows():
        g, p = _parse_set(r["gold_resource_areas"]), _parse_set(r[pred_col])
        if not g and not p:
            continue  # both 'unknown' — counted in the unknown metrics instead
        n += 1
        ok += int(bool(g & p))
    return round(ok / n, 3) if n else float("nan")


def score(path: Path = SHEET_CSV) -> None:
    if not path.exists():
        raise SystemExit(f"[score] no sheet at {path} — run --build first")
    df = pd.read_csv(path, dtype=str).fillna("")
    labeled = df[df["gold_resource_areas"].str.strip() != ""].copy()
    n_all, n_lab = len(df), len(labeled)
    out: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        out.append(s)

    emit("# D6 #47 re-tag validation — score report")
    emit()
    emit(f"Sheet: `{path}` — {n_lab}/{n_all} rows labeled.")
    if n_lab == 0:
        raise SystemExit("[score] nothing labeled yet.")
    if n_lab < n_all:
        emit(f"**{n_all - n_lab} rows are unlabeled and excluded from every number below.**")
    emit()

    # vocabulary check on the human labels
    bad: dict[str, set[str]] = {}
    for _, r in labeled.iterrows():
        strays = {a for a in _parse_set(r["gold_resource_areas"]) if a not in VALID}
        if strays:
            bad[r["condition_id"]] = strays
    if bad:
        emit("## WARNING — labels outside the shared 12 vocabulary")
        emit()
        for cid, strays in bad.items():
            emit(f"- `{cid}`: {sorted(strays)}")
        emit()
        emit("These rows are still scored, but the stray values can never match a prediction.")
        emit()

    ungradeable = labeled["notes"].str.contains("truncat|garbl|unreadable", case=False, na=False).sum()
    if ungradeable:
        emit(f"_{ungradeable} row(s) flagged in `notes` as truncated/garbled — still scored, "
             f"but treat them as a floor on achievable precision._")
        emit()

    # ---- headline: new vs old
    emit("## Overall — new tags vs old keyword tags")
    emit()
    emit("Micro-averaged over (row, resource_area) pairs; `unknown` excluded from both sides "
         "(it is scored separately below).")
    emit()
    new_m = _score_col(labeled, "new_tags")
    old_m = _score_col(labeled, "old_tag")
    emit("| metric | OLD (keyword) | NEW (tier1+haiku) | delta |")
    emit("|---|---|---|---|")
    for k in ("precision", "recall", "f1", "exact_set_match"):
        o, nw = old_m[k], new_m[k]
        d = round(nw - o, 3) if (o == o and nw == nw) else float("nan")
        emit(f"| {k} | {o} | {nw} | {d:+.3f} |")
    emit(f"| tp / fp / fn | {old_m['tp']} / {old_m['fp']} / {old_m['fn']} "
         f"| {new_m['tp']} / {new_m['fp']} / {new_m['fn']} | |")
    emit()

    emit("## Any-overlap rate (the rule D2 adopts)")
    emit()
    emit("Share of rows with a real resource area where **at least one** predicted area is correct. "
         "This is what D2's impact<->mitigation join actually depends on.")
    emit()
    emit(f"- OLD: **{_any_overlap_rate(labeled, 'old_tag')}**")
    emit(f"- NEW: **{_any_overlap_rate(labeled, 'new_tags')}**")
    emit()

    emit("## 'unknown' handling")
    emit()
    for lbl, col in (("OLD", "old_tag"), ("NEW", "new_tags")):
        u = _unknown_metrics(labeled, col)
        emit(f"- **{lbl}** — gold-unknown rows {u['gold_unknown_rows']}, "
             f"predicted-unknown {u['pred_unknown_rows']}; "
             f"unknown precision {u['unknown_precision']} / recall {u['unknown_recall']}; "
             f"over-tagged {u['over_tagged']} (gold says none, pipeline invented one), "
             f"under-tagged {u['under_tagged']} (gold says something, pipeline said none)")
    emit()

    emit("## Per stratum (new tags)")
    emit()
    emit("| stratum | n | precision | recall | f1 | exact | any-overlap |")
    emit("|---|---|---|---|---|---|---|")
    for stratum in list(STRATUM_TARGETS) + sorted(set(labeled["stratum"]) - set(STRATUM_TARGETS)):
        sub = labeled[labeled["stratum"] == stratum]
        if sub.empty:
            continue
        m = _score_col(sub, "new_tags")
        emit(f"| `{stratum}` | {m['n']} | {m['precision']} | {m['recall']} | {m['f1']} "
             f"| {m['exact_set_match']} | {_any_overlap_rate(sub, 'new_tags')} |")
    emit()

    emit("## Per stratum — did the re-tag help? (old vs new f1)")
    emit()
    emit("| stratum | n | old f1 | new f1 | delta |")
    emit("|---|---|---|---|---|")
    for stratum in list(STRATUM_TARGETS):
        sub = labeled[labeled["stratum"] == stratum]
        if sub.empty:
            continue
        o, nw = _score_col(sub, "old_tag")["f1"], _score_col(sub, "new_tags")["f1"]
        d = round(nw - o, 3) if (o == o and nw == nw) else float("nan")
        emit(f"| `{stratum}` | {len(sub)} | {o} | {nw} | {d:+.3f} |")
    emit()

    # ---- holistic human verdict
    if labeled["is_correct"].str.strip().ne("").any():
        emit("## Human holistic verdict (`is_correct`)")
        emit()
        vc = labeled["is_correct"].str.strip().str.lower().replace("", "(blank)").value_counts()
        for k, v in vc.items():
            emit(f"- `{k}`: {v} ({100*v/n_lab:.0f}%)")
        emit()
        emit("| stratum | yes | partial | no |")
        emit("|---|---|---|---|")
        for stratum in list(STRATUM_TARGETS):
            sub = labeled[labeled["stratum"] == stratum]
            if sub.empty:
                continue
            c = sub["is_correct"].str.strip().str.lower().value_counts()
            emit(f"| `{stratum}` | {c.get('yes',0)} | {c.get('partial',0)} | {c.get('no',0)} |")
        emit()

    emit("## How to read this")
    emit()
    emit("- **precision** is the number that gates D2's claim upgrade. High precision means a "
         "predicted resource area can be trusted when it fires.")
    emit("- **any-overlap** is the closest single number to what D2's join needs.")
    emit("- **`still_unknown` precision is not meaningful** — that stratum's predictions are all "
         "empty. Read its `unknown` agreement instead: high agreement means leaving them untagged "
         "was right; low agreement means there is recoverable signal we are still missing.")
    emit("- A **negative delta on `unchanged`** would mean the re-tag broke rows that were already "
         "right — check that before shipping.")
    emit()
    emit("**This does not by itself move D2's `mitigation_dependent_f1` (0.612 overall / 0.623 "
         "holdout).** That metric is scored against D2's own gold set, which labels the impact "
         "side only. A strong score here is the precondition for upgrading D2's resource-level "
         "mitigation caveat to a finding — not the upgrade itself.")

    SCORE_MD.write_text("\n".join(out) + "\n")
    print(f"\n[score] wrote {SCORE_MD}")


# --------------------------------------------------------------------------- self-test

def self_test() -> None:
    """Fabricate labels on the drawn sample and run the scorer, to prove the scorer works.

    The synthetic labels are DELIBERATE NOISE — they agree with the prediction most of the time,
    drop/add an area sometimes, and occasionally say 'unknown'. Only the scorer's mechanics are
    being tested; none of the resulting numbers mean anything about the real re-tag quality.
    """
    if not SHEET_CSV.exists():
        raise SystemExit("[self-test] run --build first")
    rng = np.random.RandomState(SEED + 1)
    df = pd.read_csv(SHEET_CSV, dtype=str).fillna("")
    golds, verdicts, notes = [], [], []
    for _, r in df.iterrows():
        pred = [a for a in str(r["new_tags"]).split(",") if a and a != "unknown"]
        u = rng.rand()
        if not pred:
            # still_unknown: 70% of the time 'unknown' really is right
            g = [] if u < 0.7 else [SHARED_12[rng.randint(len(SHARED_12))]]
        elif u < 0.55:
            g = list(pred)                                   # exact agreement
        elif u < 0.75:
            g = list(pred) + [SHARED_12[rng.randint(len(SHARED_12))]]   # human adds one
        elif u < 0.9:
            g = list(pred)[:-1] or list(pred)                # human drops one
        else:
            g = [SHARED_12[rng.randint(len(SHARED_12))]]     # outright disagreement
        g = sorted(set(g))
        golds.append(",".join(g) if g else "unknown")
        pset, gset = set(pred), set(g)
        verdicts.append("yes" if pset == gset else ("partial" if pset & gset else "no"))
        notes.append("SYNTHETIC — self-test only")
    df["gold_resource_areas"] = golds
    df["is_correct"] = verdicts
    df["notes"] = notes

    tmp = NOTES_DIR / "_selftest_retag_validation_sheet.csv"
    df.to_csv(tmp, index=False)
    print(f"[self-test] wrote synthetic sheet {tmp}\n"
          f"[self-test] scoring it — numbers below are MEANINGLESS, only mechanics are tested\n")
    score(tmp)
    # keep the real score report from being overwritten by synthetic numbers
    SCORE_MD.unlink(missing_ok=True)
    print(f"\n[self-test] OK — scorer ran end-to-end on {len(df)} synthetic rows.")
    print(f"[self-test] removed the synthetic {SCORE_MD} so it cannot be mistaken for a real result.")
    print(f"[self-test] the synthetic sheet is left at {tmp} for inspection; delete it freely.")


def refresh_tags(path: Path = SHEET_CSV) -> None:
    """Refresh ONLY the `new_tags` prediction column of an already-labeled sheet from the CURRENT
    fonsi_conditions.parquet (joined by text_sha), preserving every human column (gold_resource_areas
    / is_correct / notes) and the stratum. Run after a re-tag rebuild, then --score. This keeps the
    scorer honest: it scores the live pipeline, not a stale snapshot."""
    if not path.exists():
        raise SystemExit(f"[refresh] no sheet at {path}")
    sheet = pd.read_csv(path, dtype=str).fillna("")
    cond = pd.read_parquet(CONDITIONS)
    cond["text_sha"] = cond["condition_text"].map(sha256_text)
    # per text: the current multi-label prediction ('' => unknown). Deterministic per text for the
    # mitigation_commitment rows the sheet is drawn from, so first() is unambiguous.
    live = (cond.groupby("text_sha")["resource_areas_multi"].first()
                .fillna("").astype(str).to_dict())
    new_vals, missing, changed = [], 0, 0
    for _, r in sheet.iterrows():
        sha = r["text_sha"]
        if sha in live:
            v = live[sha] if live[sha] else "unknown"
        else:
            v = r["new_tags"]; missing += 1
        changed += int(v != r["new_tags"])
        new_vals.append(v)
    sheet["new_tags"] = new_vals
    sheet.to_csv(path, index=False)
    print(f"[refresh] updated new_tags from live fonsi_conditions: {changed} rows changed, "
          f"{missing} not found (kept old). Human columns untouched. -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="D6 #47 re-tag validation gate (build / score).")
    ap.add_argument("--build", action="store_true", help="draw the stratified sample + write the sheet")
    ap.add_argument("--score", action="store_true", help="score a human-filled sheet")
    ap.add_argument("--refresh-tags", action="store_true",
                    help="refresh new_tags from current fonsi_conditions (after a re-tag rebuild); preserves gold")
    ap.add_argument("--self-test", action="store_true", help="fabricate labels and exercise the scorer")
    ap.add_argument("--sheet", default=str(SHEET_CSV), help="path to the sheet (score mode)")
    args = ap.parse_args()
    if args.build:
        build()
    elif args.self_test:
        self_test()
    elif args.refresh_tags:
        refresh_tags(Path(args.sheet))
    elif args.score:
        score(Path(args.sheet))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
