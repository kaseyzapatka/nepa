import pandas as pd
from datetime import date

def _dates_match(p_date, g_date, g_gran):
    if pd.isna(p_date) or pd.isna(g_date) or str(p_date) == "None" or str(g_date) == "None":
        return False
    try:
        p = pd.Timestamp(p_date).date()
        g = pd.Timestamp(g_date).date()
    except Exception:
        return False
    gran = str(g_gran or "day").strip().lower()
    if gran == "day":
        return p == g
    if gran == "month":
        return p.year == g.year and p.month == g.month
    if gran == "year":
        return p.year == g.year
    return p == g

gold = pd.read_parquet("phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet")
pipe = pd.read_parquet("phase2/data/analysis/timeline/timeline_project_dates.parquet")

df = gold.merge(pipe, on=["project_id", "process_type"], how="left")

init_match = 0
dec_match = 0
init_total = 0
dec_total = 0

report = []

for _, row in df.iterrows():
    gi_date = row.get("gold_initiation_date")
    gi_gran = row.get("gold_initiation_granularity")
    pi_date = row.get("initiation_date")
    
    gd_date = row.get("gold_decision_date")
    gd_gran = row.get("gold_decision_granularity")
    pd_date = row.get("decision_date")

    has_gi = pd.notna(gi_date) and str(gi_date).strip() not in ("", "none", "nan", "None")
    has_gd = pd.notna(gd_date) and str(gd_date).strip() not in ("", "none", "nan", "None")
    
    if has_gi:
        init_total += 1
        im = _dates_match(pi_date, gi_date, gi_gran)
        if im: init_match += 1
        else:
            report.append({
                "project_id": row["project_id"], "process_type": row["process_type"],
                "role": "initiation", "gold": gi_date, "gold_gran": gi_gran, "pred": pi_date,
                "confidence": row.get("initiation_confidence")
            })

    if has_gd:
        dec_total += 1
        dm = _dates_match(pd_date, gd_date, gd_gran)
        if dm: dec_match += 1
        else:
            report.append({
                "project_id": row["project_id"], "process_type": row["process_type"],
                "role": "decision", "gold": gd_date, "gold_gran": gd_gran, "pred": pd_date,
                "confidence": row.get("decision_confidence")
            })

print(f"End-to-End Initiation Accuracy: {init_match}/{init_total} = {init_match/init_total if init_total else 0:.1%}")
print(f"End-to-End Decision Accuracy: {dec_match}/{dec_total} = {dec_match/dec_total if dec_total else 0:.1%}")

if report:
    err_df = pd.DataFrame(report)
    err_df.to_csv("phase2/output/deliverable04/timeline_gold_errors.csv", index=False)
    print("Errors saved to phase2/output/deliverable04/timeline_gold_errors.csv")
