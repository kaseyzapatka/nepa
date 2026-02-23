# --------------------------
# DELIVERABLE 3: MERGE REGEX + LLM RESULTS
# --------------------------
# Merge LLM capacity outputs back into the regex dataset

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
POWER_UNITS = {"GW", "MW", "kW"}


def normalize_power_unit(unit):
    """Normalize common power unit variants to GW/MW/kW."""
    if unit is None or (isinstance(unit, float) and pd.isna(unit)):
        return None
    text = str(unit).strip().lower().replace(" ", "")
    mapping = {
        "gw": "GW",
        "gwe": "GW",
        "gwac": "GW",
        "gwdc": "GW",
        "gigawatt": "GW",
        "gigawatts": "GW",
        "mw": "MW",
        "mwe": "MW",
        "mwt": "MW",
        "mwth": "MW",
        "mwac": "MW",
        "mwdc": "MW",
        "mwp": "MW",
        "megawatt": "MW",
        "megawatts": "MW",
        "kw": "kW",
        "kwe": "kW",
        "kwac": "kW",
        "kwdc": "kW",
        "kilowatt": "kW",
        "kilowatts": "kW",
    }
    return mapping.get(text, str(unit).strip())


def load_llm_results():
    files = {
        "CE": ANALYSIS_DIR / "gencap_ce_llm.parquet",
        "EA": ANALYSIS_DIR / "gencap_ea_llm.parquet",
        "EIS": ANALYSIS_DIR / "gencap_eis_llm.parquet",
    }
    frames = []
    for source, path in files.items():
        if path.exists():
            df = pd.read_parquet(path)
            df["dataset_source"] = source
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    regex_path = ANALYSIS_DIR / "projects_gencap_flagged.parquet"
    if not regex_path.exists():
        regex_path = ANALYSIS_DIR / "projects_gencap.parquet"

    regex = pd.read_parquet(regex_path)
    if {"project_id", "dataset_source"}.issubset(regex.columns):
        regex = regex.drop_duplicates(subset=["project_id", "dataset_source"])
    llm = load_llm_results()

    if llm.empty:
        print("No LLM outputs found. Exiting.")
        return

    # Select and rename LLM columns for merge
    llm_cols = [
        "project_id",
        "dataset_source",
        "capacity_value",
        "capacity_unit",
        "confidence",
        "source_quote",
        "extraction_method",
        "pages_scanned",
        "candidates_found",
        "num_candidates",
        "llm_run_completed_at_utc",
        "llm_model_used",
        "llm_trigger_mode",
    ]
    llm_cols = [c for c in llm_cols if c in llm.columns]
    llm = llm[llm_cols].copy()

    llm = llm.rename(columns={
        "capacity_value": "llm_capacity_value",
        "capacity_unit": "llm_capacity_unit",
        "confidence": "llm_confidence",
        "source_quote": "llm_source_quote",
        "extraction_method": "llm_extraction_method",
        "pages_scanned": "llm_pages_scanned",
        "candidates_found": "llm_candidates_found",
        "num_candidates": "llm_num_candidates",
    })

    merged = regex.merge(llm, on=["project_id", "dataset_source"], how="left")
    if {"project_id", "dataset_source"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["project_id", "dataset_source"])

    # LLM validation for merge override
    merged["llm_capacity_unit_norm"] = merged["llm_capacity_unit"].apply(normalize_power_unit)
    merged["llm_is_valid_power"] = (
        merged["llm_capacity_value"].notna()
        & (pd.to_numeric(merged["llm_capacity_value"], errors="coerce") > 0)
        & merged["llm_capacity_unit_norm"].isin(POWER_UNITS)
    )
    merged["llm_is_rejected_method"] = merged["llm_extraction_method"].isin(
        ["no_candidates", "no_numeric_candidates", "llm_rejected_no_quote", "llm_error", "llm_timeout"]
    )
    merged["llm_should_override_regex"] = merged["llm_is_valid_power"] & ~merged["llm_is_rejected_method"]

    # Choose final capacity: valid LLM override, else regex
    merged["project_gencap_final_value"] = merged["project_gencap_value"]
    merged["project_gencap_final_unit"] = merged["project_gencap_unit"]
    merged["project_gencap_final_source"] = merged.get("project_gencap_source")
    merged["project_gencap_final_confidence"] = merged["project_gencap_confidence"]
    merged["project_gencap_final_quote"] = merged["project_gencap_context"]

    llm_override_mask = merged["llm_should_override_regex"]
    merged.loc[llm_override_mask, "project_gencap_final_value"] = merged.loc[llm_override_mask, "llm_capacity_value"]
    merged.loc[llm_override_mask, "project_gencap_final_unit"] = merged.loc[llm_override_mask, "llm_capacity_unit_norm"]
    merged.loc[llm_override_mask, "project_gencap_final_source"] = merged.loc[llm_override_mask, "llm_extraction_method"]
    merged.loc[llm_override_mask, "project_gencap_final_confidence"] = merged.loc[llm_override_mask, "llm_confidence"]
    merged.loc[llm_override_mask, "project_gencap_final_quote"] = merged.loc[llm_override_mask, "llm_source_quote"]

    merged["llm_merge_decision"] = "regex_no_llm"
    merged.loc[merged["llm_capacity_value"].notna() & ~llm_override_mask, "llm_merge_decision"] = "regex_invalid_or_rejected_llm"
    merged.loc[llm_override_mask & merged["project_gencap_value"].notna(), "llm_merge_decision"] = "llm_override_regex"
    merged.loc[llm_override_mask & merged["project_gencap_value"].isna(), "llm_merge_decision"] = "llm_only_fill"
    merged.loc[
        merged["project_gencap_final_value"].isna() & merged["llm_capacity_value"].isna(),
        "llm_merge_decision",
    ] = "no_capacity"

    out_path = ANALYSIS_DIR / "projects_gencap_merged.parquet"
    merged.to_parquet(out_path)
    print(f"Saved merged dataset: {out_path}")


if __name__ == "__main__":
    main()
