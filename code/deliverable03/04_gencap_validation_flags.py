# --------------------------
# DELIVERABLE 3: GENERATION CAPACITY VALIDATION FLAGS + SAMPLE
# --------------------------
# Create lightweight flags for likely false positives / non-generation cases
# and output a small audit sample with snippets.

import re
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
OUTPUT_DIR = BASE_DIR / "output" / "deliverable3"

# Simple patterns
DATE_NEAR_UNIT = re.compile(r"\b(?:MW|kW|GW)\b[^\n]{0,30}\b\d{1,2}/\d{1,2}/\d{2,4}\b", re.IGNORECASE)
INITIALS_DATE = re.compile(r"\b[A-Z]{1,3},?\s*\d{1,2}/\d{1,2}/\d{2,4}\b")

NON_GEN_TERMS = re.compile(r"beam power|accelerator|kwh|tes|thermal storage|battery storage", re.IGNORECASE)
NON_BUILD_TERMS = re.compile(r"removal|decommission|retire|demolition|dismantle|replace|upgrade|maintenance|repair", re.IGNORECASE)
EQUIPMENT_LIST_HINT = re.compile(r"\*\s+|\bincludes:\b|\bthe equipment\b", re.IGNORECASE)


def flag_row(title: str, context: str) -> dict:
    text = " ".join([str(title or ""), str(context or "")])
    return {
        "gencap_flag_initials_date": bool(INITIALS_DATE.search(text) or DATE_NEAR_UNIT.search(text)),
        "gencap_flag_non_generation": bool(NON_GEN_TERMS.search(text)),
        "gencap_flag_non_build": bool(NON_BUILD_TERMS.search(text)),
        "gencap_flag_equipment_list": bool(EQUIPMENT_LIST_HINT.search(text)),
    }


def main():
    gencap_path = ANALYSIS_DIR / "projects_gencap.parquet"
    if not gencap_path.exists():
        raise FileNotFoundError(f"Missing: {gencap_path}")

    df = pd.read_parquet(gencap_path)

    # Clean energy only
    if "project_energy_type" in df.columns:
        df = df[df["project_energy_type"] == "Clean"]

    # Build flags using title + context
    flags = df.apply(lambda r: flag_row(r.get("project_title"), r.get("project_gencap_context")), axis=1, result_type="expand")
    df = pd.concat([df, flags], axis=1)

    # Save flagged dataset
    out_flags = ANALYSIS_DIR / "projects_gencap_flagged.parquet"
    df.to_parquet(out_flags)
    print(f"Saved flagged dataset: {out_flags}")

    # Create small audit sample from projects with capacity
    with_cap = df[df["project_gencap_value"].notna()].copy()
    # Oversample suspicious cases
    suspicious = with_cap[with_cap[[
        "gencap_flag_initials_date",
        "gencap_flag_non_generation",
        "gencap_flag_non_build",
        "gencap_flag_equipment_list",
    ]].any(axis=1)]

    sample = pd.concat([
        suspicious.sample(min(15, len(suspicious)), random_state=42),
        with_cap.sample(min(15, len(with_cap)), random_state=7),
    ], ignore_index=True).drop_duplicates(subset=["project_id"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = OUTPUT_DIR / "gencap_validation_quick_sample.csv"

    cols = [
        "project_id", "project_title", "dataset_source",
        "project_gencap_value", "project_gencap_unit",
        "project_gencap_context", "project_gencap_matches",
        "gencap_flag_initials_date", "gencap_flag_non_generation",
        "gencap_flag_non_build", "gencap_flag_equipment_list",
    ]
    cols = [c for c in cols if c in sample.columns]
    sample[cols].to_csv(sample_path, index=False)
    print(f"Saved quick audit sample: {sample_path}")


if __name__ == "__main__":
    main()
