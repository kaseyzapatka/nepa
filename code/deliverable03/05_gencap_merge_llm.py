# --------------------------
# DELIVERABLE 3: MERGE REGEX + LLM RESULTS
# --------------------------
# Merge LLM capacity outputs back into the regex dataset

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"


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

    # Choose final capacity: LLM if present, else regex
    merged["project_gencap_final_value"] = merged["llm_capacity_value"].combine_first(merged["project_gencap_value"])
    merged["project_gencap_final_unit"] = merged["llm_capacity_unit"].combine_first(merged["project_gencap_unit"])
    merged["project_gencap_final_source"] = merged["llm_extraction_method"].where(merged["llm_capacity_value"].notna(), merged.get("project_gencap_source"))
    merged["project_gencap_final_confidence"] = merged["llm_confidence"].combine_first(merged["project_gencap_confidence"])
    merged["project_gencap_final_quote"] = merged["llm_source_quote"].combine_first(merged["project_gencap_context"])

    out_path = ANALYSIS_DIR / "projects_gencap_merged.parquet"
    merged.to_parquet(out_path)
    print(f"Saved merged dataset: {out_path}")


if __name__ == "__main__":
    main()
