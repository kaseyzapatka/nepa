import os

if os.environ.get("CONDA_DEFAULT_ENV") != "nepa":
    raise SystemExit("Please run in conda env 'nepa' (e.g., `conda run -n nepa python ...`).")

"""
01_build_supervision_samples.py

For each process type (CE, EA, EIS), pull 200 labeled examples for manual review:
  - 4 classes × 50 examples each = 200 per source
  - Per class: 20 "clear correct" (confident, diverse) + 30 "likely mislabeled" (conflicting signals)

The goal is to give the BERT model high-quality supervised examples covering a diversity
of real-world patterns — not just what weak supervision got right, but especially the
systematic errors it makes.

Outputs:
    data/manual_supervision/review_CE.csv
    data/manual_supervision/review_EA.csv
    data/manual_supervision/review_EIS.csv

Workflow:
    1. Run this script
    2. Open each CSV. For each row:
         - Read 'context' and 'why' to understand the case
         - Set 'correct_label' to: initiation | decision | review | other
         - Leave 'correct_label' blank if you are unsure
    3. Run: python code/manual_supervision/02_apply_supervision.py
    4. Retrain: --bert-generate → --bert-train --source CE (or EA, EIS)
"""

import re
import sys
import random
import duckdb
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "code" / "extract"))
from extract_timeline import auto_label_context  # noqa: E402

CACHE_DIR   = BASE_DIR / "data" / "analysis"
OUTPUT_DIR  = BASE_DIR / "data" / "manual_supervision"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = ["CE", "EA", "EIS"]
CLASSES = ["decision", "initiation", "review", "other"]
N_CORRECT   = 20   # clear-correct examples per class
N_INCORRECT = 30   # likely-mislabeled examples per class
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


# ── Pattern helpers ────────────────────────────────────────────────────────────

def _c(text: str) -> str:
    """Lowercase, collapse whitespace."""
    return re.sub(r'\s+', ' ', str(text or '')).lower()


def _has(pattern: str, text: str) -> bool:
    return bool(re.search(pattern, _c(text)))


def has_initiator(ctx):
    return _has(r'doe initiator|nepa initiator|action initiating|project initiator', ctx)

def has_application_intake(ctx):
    return _has(r'application date|date received|date filed|date submitted|application received|'
                r'blm.*application|right.of.way application|row application|gonepa|ef2a', ctx)

def has_noi_scoping(ctx):
    return _has(r'notice of intent|scoping notice|initiat.*scoping|scoping.*period', ctx)

def has_strong_decision(ctx):
    return _has(r'fonsi|finding of no significant impact|record of decision|\brод\b|'
                r'authorizing official|nepa compliance officer|ce determination|'
                r'categorical exclusion determination', ctx)

def has_reviewer_pattern(ctx):
    return _has(r'environmental coordinator|reviewing official|reviewing officer|'
                r'reviewed by|concurrence|concurs', ctx)

def has_digital_sig(ctx):
    return _has(r'digitally signed|electronically signed|digital signature', ctx)

def has_reference_signal(ctx):
    return _has(r'see also|refer to|pursuant to|in accordance with|as of|as amended', ctx)

def has_any_initiation(ctx):
    return has_initiator(ctx) or has_application_intake(ctx) or has_noi_scoping(ctx)


# ── "Why" explanation generator ───────────────────────────────────────────────

def _build_why(row, auto_label: str, conflict_note: str = "") -> str:
    """
    One-sentence explanation of why auto_label was assigned and/or why it might be wrong.
    """
    ctx   = str(row.get("context", ""))
    sec   = str(row.get("section_label") or "")
    verb  = str(row.get("dep_verb") or "")
    sig   = bool(row.get("sig_flag"))
    ner   = bool(row.get("ner_decision_signal"))

    signals = []
    if sec:
        signals.append(f"section={sec}")
    if verb:
        signals.append(f"verb={verb}")
    if sig:
        signals.append("sig_block=Y")
    if ner:
        signals.append("ner_decision=Y")
    if has_initiator(ctx):
        signals.append("has_initiator_text")
    if has_application_intake(ctx):
        signals.append("has_application_intake")
    if has_noi_scoping(ctx):
        signals.append("has_noi_scoping")
    if has_strong_decision(ctx):
        signals.append("has_decision_text")
    if has_reviewer_pattern(ctx):
        signals.append("has_reviewer_text")
    if has_digital_sig(ctx):
        signals.append("has_digital_sig")

    label_str = f"auto={auto_label}" if auto_label else "auto=None(unlabeled)"
    sig_str   = " | ".join(signals) if signals else "no strong signals"
    base      = f"{label_str} | {sig_str}"
    return f"{base} || CONFLICT: {conflict_note}" if conflict_note else base


# ── Conflict finders (return sub-DataFrames of likely-mislabeled rows) ─────────

def _find_conflicts(df: pd.DataFrame, target_class: str, source: str) -> pd.DataFrame:
    """
    For each target class, find rows whose auto_label is likely WRONG and the
    correct label is probably `target_class`.

    Conflict rows are tagged with a `conflict_note` column explaining the issue.
    """
    ctx = df["context"]
    al  = df["auto_label"]
    sec = df["section_label"].fillna("")

    frames = []

    if target_class == "initiation":
        # Labeled decision/review but context has initiator form field (CE)
        m1 = df[al.isin(["decision", "review"]) & ctx.apply(has_initiator)].copy()
        m1["conflict_note"] = "labeled non-initiation but has DOE/NEPA initiator text"

        # Labeled review/decision but context has application intake field
        m2 = df[al.isin(["decision", "review", "other"]) & ctx.apply(has_application_intake)].copy()
        m2["conflict_note"] = "labeled non-initiation but has application/intake date field"

        # Unlabeled (auto=None) but context has NOI/scoping or application signals
        m3 = df[al.isna() & ctx.apply(has_any_initiation)].copy()
        m3["conflict_note"] = "auto_label=None but context has clear initiation signal"

        # Section says noi/signature_block but labeled non-initiation
        m4 = df[sec.isin(["noi"]) & ~al.isin(["initiation"])].copy()
        m4["conflict_note"] = "section=noi but labeled non-initiation"

        frames = [m1, m2, m3, m4]

    elif target_class == "decision":
        # Labeled review but has FONSI / ROD / authorizing official
        m1 = df[al.isin(["review", "other"]) & ctx.apply(has_strong_decision)].copy()
        m1["conflict_note"] = "labeled non-decision but has FONSI/ROD/authorizing text"

        # Labeled initiation but section is ce_determination / fonsi / rod
        m2 = df[al.isin(["initiation", "other"]) &
                sec.isin(["ce_determination", "fonsi", "rod", "final_eis"])].copy()
        m2["conflict_note"] = "labeled non-decision but section is ce_determination/fonsi/rod"

        # sig_flag=True + dep_verb is sign/approve/authorize, but labeled non-decision
        sig_decision = df[
            df["sig_flag"].astype(bool) &
            df["dep_verb"].str.lower().isin(["sign", "approve", "authorize", "execute"]) &
            ~al.isin(["decision"])
        ].copy()
        sig_decision["conflict_note"] = "sig_block + decision verb but labeled non-decision"

        frames = [m1, m2, sig_decision]

    elif target_class == "review":
        # Labeled decision but has reviewer patterns (could be intermediate sign-off)
        m1 = df[al.isin(["decision"]) & ctx.apply(has_reviewer_pattern) &
                ~ctx.apply(has_strong_decision)].copy()
        m1["conflict_note"] = "labeled decision but has reviewer sign-off text without strong decision cue"

        # Section is review_checklist but labeled non-review
        m2 = df[sec.isin(["review_checklist"]) & ~al.isin(["review"])].copy()
        m2["conflict_note"] = "section=review_checklist but labeled non-review"

        frames = [m1, m2]

    elif target_class == "other":
        # Labeled decision but no strong decision signal (just incidental date)
        m1 = df[al.isin(["decision"]) &
                ~ctx.apply(has_strong_decision) &
                ~ctx.apply(has_reviewer_pattern) &
                ~df["sig_flag"].astype(bool)].copy()
        m1["conflict_note"] = "labeled decision but no strong decision/reviewer/sig signal"

        # Section is references or legal_citations but labeled non-other
        m2 = df[sec.isin(["references", "legal_citations"]) & ~al.isin(["other"])].copy()
        m2["conflict_note"] = "section=references/legal_citations but labeled non-other"

        # Has reference signal (see also / pursuant to) and labeled decision
        m3 = df[al.isin(["decision"]) & ctx.apply(has_reference_signal)].copy()
        m3["conflict_note"] = "labeled decision but context is a reference/citation sentence"

        frames = [m1, m2, m3]

    if not frames:
        return pd.DataFrame(columns=df.columns.tolist() + ["conflict_note"])

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["project_id", "date"])
    return combined


# ── Diversity sampler ──────────────────────────────────────────────────────────

def _sample_diverse(df: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Sample n rows, maximising diversity across project_id, doc_type, and year.
    Strategy: one per project first, then fill remaining randomly.
    """
    if df.empty:
        return df

    # One row per project (earliest date wins for stability)
    df = df.copy()
    df["_year"] = df["date"].astype(str).str[:4]
    one_per_project = (
        df.sort_values("date")
          .drop_duplicates(subset="project_id", keep="first")
    )

    if len(one_per_project) >= n:
        # Sample to spread across years and doc_types
        return one_per_project.sample(min(n, len(one_per_project)), random_state=seed)

    # Not enough unique projects — allow multiple rows per project to fill
    remaining_n = n - len(one_per_project)
    extras = df[~df.index.isin(one_per_project.index)]
    if not extras.empty:
        extras_sample = extras.sample(min(remaining_n, len(extras)), random_state=seed)
        return pd.concat([one_per_project, extras_sample], ignore_index=True)

    return one_per_project


# ── Main per-source builder ────────────────────────────────────────────────────

def build_source_samples(source: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"regex_candidates_{source.lower()}.parquet"
    if not cache_path.exists():
        print(f"  SKIP {source}: cache not found at {cache_path}")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  {source} — loading cache ({cache_path.name})")

    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{cache_path}')").df()
    print(f"  Loaded {len(df):,} rows")

    # Apply weak supervision
    print("  Applying auto_label_context()...")
    df["auto_label"] = df["context"].apply(lambda c: auto_label_context(c, source))

    print("  auto_label distribution:")
    print("   ", df["auto_label"].value_counts(dropna=False).to_dict())

    # Normalise types
    df["section_label"]       = df["section_label"].fillna("").astype(str)
    df["dep_verb"]            = df["dep_verb"].fillna("").astype(str)
    df["sig_flag"]            = df["sig_flag"].fillna(False).astype(bool)
    df["ner_decision_signal"] = df["ner_decision_signal"].fillna(False).astype(bool)

    all_rows = []

    for cls in CLASSES:
        print(f"\n  Class: {cls}")

        # ── Correct examples ──────────────────────────────────────────────────
        correct_pool = df[df["auto_label"] == cls].copy()

        # Prefer rows with confirming structural features
        if cls == "decision":
            strong = correct_pool[
                correct_pool["context"].apply(has_strong_decision) |
                correct_pool["sig_flag"] |
                correct_pool["ner_decision_signal"]
            ]
        elif cls == "initiation":
            strong = correct_pool[
                correct_pool["context"].apply(has_any_initiation) |
                correct_pool["section_label"].isin(["noi", "signature_block"])
            ]
        elif cls == "review":
            strong = correct_pool[
                correct_pool["context"].apply(has_reviewer_pattern) |
                correct_pool["section_label"].isin(["review_checklist"])
            ]
        else:  # other
            strong = correct_pool[
                ~correct_pool["context"].apply(has_strong_decision) &
                ~correct_pool["context"].apply(has_any_initiation)
            ]

        if len(strong) < N_CORRECT:
            strong = correct_pool  # fall back to full pool

        correct_sample = _sample_diverse(strong, N_CORRECT)
        correct_sample["sample_type"] = "correct"
        correct_sample["conflict_note"] = ""
        print(f"    correct:   {len(correct_sample):>3} (pool: {len(strong):,})")

        # ── Conflict examples ─────────────────────────────────────────────────
        conflict_pool = _find_conflicts(df, cls, source)
        conflict_sample = _sample_diverse(conflict_pool, N_INCORRECT)
        conflict_sample["sample_type"] = "likely_mislabeled"
        print(f"    conflicts: {len(conflict_sample):>3} (pool: {len(conflict_pool):,})")

        combined = pd.concat([correct_sample, conflict_sample], ignore_index=True)
        combined["target_class"] = cls

        all_rows.append(combined)

    if not all_rows:
        return pd.DataFrame()

    result = pd.concat(all_rows, ignore_index=True)

    # Build "why" column
    result["why"] = result.apply(
        lambda r: _build_why(r, r["auto_label"], r.get("conflict_note", "")),
        axis=1,
    )

    # Truncate context for readability
    result["context"] = result["context"].astype(str).str[:400].str.replace("\n", " ").str.strip()

    # Output columns
    output_cols = [
        "project_id", "date", "auto_label", "section_label", "dep_verb",
        "doc_type", "target_class", "sample_type", "context", "why",
        "correct_label",
    ]
    result["correct_label"] = ""   # user fills this in

    return result[output_cols].drop_duplicates(subset=["project_id", "date"])


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("Building manual supervision samples...")
    print(f"Target: {N_CORRECT} correct + {N_INCORRECT} mislabeled per class × {len(CLASSES)} classes = "
          f"{(N_CORRECT + N_INCORRECT) * len(CLASSES)} rows per source\n")

    for source in SOURCES:
        df = build_source_samples(source)
        if df.empty:
            continue

        out_path = OUTPUT_DIR / f"review_{source}.csv"
        df.to_csv(out_path, index=False)

        print(f"\n  Saved {len(df):,} rows → {out_path}")
        print("  sample_type distribution:", df["sample_type"].value_counts().to_dict())
        print("  target_class distribution:", df["target_class"].value_counts().to_dict())

    print("\nDone. Review each CSV, fill in 'correct_label', then run:")
    print("  python code/manual_supervision/02_apply_supervision.py")


if __name__ == "__main__":
    main()
