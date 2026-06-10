import pandas as pd

gold = pd.read_parquet("phase2/data/analysis/timeline/gold/timeline_gold_projects.parquet")
gold = gold.sample(n=50, random_state=42)
pipe = pd.read_parquet("phase2/data/analysis/timeline/timeline_project_dates.parquet")
cands = pd.read_parquet("phase2/data/analysis/timeline/timeline_candidates.parquet")

df = gold.merge(pipe, on=["project_id", "process_type"], how="left")

md = ["# Timeline Selection Review (50 Gold Projects)\n\n"]
md.append("This artifact compares the model's selected dates against the gold standard labels, and shows the top candidates the model was choosing from.\n\n")

for _, row in df.iterrows():
    pid = row["project_id"]
    pt = row["process_type"]
    
    gi = row.get("gold_initiation_date", "None") or "None"
    gd = row.get("gold_decision_date", "None") or "None"
    
    pi = row.get("initiation_date", "None") or "None"
    pd_date = row.get("decision_date", "None") or "None"
    
    md.append(f"## Project: {pid} ({pt})\n")
    md.append(f"- **Gold Initiation:** {gi} | **Selected Initiation:** {pi}")
    if str(gi) == str(pi):
        md.append(" ✅\n")
    else:
        md.append(" ❌\n")
        
    md.append(f"- **Gold Decision:** {gd} | **Selected Decision:** {pd_date}")
    if str(gd) == str(pd_date):
        md.append(" ✅\n\n")
    else:
        md.append(" ❌\n\n")
        
    proj_cands = cands[cands["project_id"] == pid]
    
    # Initiation candidates
    init_cands = proj_cands[proj_cands["candidate_role"].str.contains("initiation")].copy()
    if not init_cands.empty:
        if "ranking_score" in init_cands.columns:
            init_cands = init_cands.sort_values("ranking_score", ascending=False).head(3)
        md.append("**Top Initiation Candidates:**\n")
        for _, c in init_cands.iterrows():
            d = c.get("parsed_date")
            s = c.get("ranking_score", 0.0)
            doc = str(c.get("document_type_clean", ""))[:30]
            ctx = str(c.get("context_text", "")).replace('\n', ' ')[:100]
            role = c.get("candidate_role", "")
            md.append(f"1. **{d}** (Score: {s:.2f}) [{role}] - {doc} | *\"{ctx}...\"*\n")
    else:
        md.append("**No Initiation Candidates.**\n")
        
    md.append("\n")
    
    # Decision candidates
    dec_cands = proj_cands[proj_cands["candidate_role"].str.contains("decision")].copy()
    if not dec_cands.empty:
        if "ranking_score" in dec_cands.columns:
            dec_cands = dec_cands.sort_values("ranking_score", ascending=False).head(3)
        md.append("**Top Decision Candidates:**\n")
        for _, c in dec_cands.iterrows():
            d = c.get("parsed_date")
            s = c.get("ranking_score", 0.0)
            doc = str(c.get("document_type_clean", ""))[:30]
            ctx = str(c.get("context_text", "")).replace('\n', ' ')[:100]
            role = c.get("candidate_role", "")
            md.append(f"1. **{d}** (Score: {s:.2f}) [{role}] - {doc} | *\"{ctx}...\"*\n")
    else:
        md.append("**No Decision Candidates.**\n")
        
    md.append("\n---\n")

with open("/Users/Dora/.gemini/antigravity-ide/brain/ac8954e7-8be6-4e3d-a093-6a01d6caba01/artifacts/timeline_selection_review.md", "w") as f:
    f.write("".join(md))

print("Artifact created!")
