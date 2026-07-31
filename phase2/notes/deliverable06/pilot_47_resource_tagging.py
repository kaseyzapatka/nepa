"""
#47 SIZING PILOT — condition->resource_area re-tagging (LOCAL, $0, no LLM/API).

Estimates how much two free tiers would cut the ~51% 'unknown' resource_area rate in
fonsi_conditions.parquet, and triangulates precision (no gold set exists).

  Tier 1 = section-heading inheritance: join condition.section_id -> evidence_spans.heading_title,
           classify the HEADING with the resource keyword dict; resource-specific headings resolve.
  Tier 2 = local embeddings (all-MiniLM-L6-v2): for conditions still unknown, cosine-match
           condition_text against prototype sentences per resource area; thresholds 0.45 / 0.55.

Proxy precision (no gold):
  (a) enrichment cross-check: agreement = predicted area is in that project's LLM
      mitigation_resource_areas list (fonsi_enrichment.parquet), for baseline / tier1 / tier2.
  (b) 20 random re-tagged examples printed for human eyeball.

REPRODUCE:  conda run -n nepa python pilot_47_resource_tagging.py
Outputs (this dir): pilot47_summary.txt, pilot47_examples.csv, pilot47_tier_tags.parquet
Runtime: ~1-3 min (Tier-2 embeds ~30k unknown conditions on CPU).
"""
from __future__ import annotations
import json, os, random, sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]
COND = REPO / "phase2/data/analysis/deliverable06/fonsi_conditions.parquet"
SPANS = REPO / "phase2/data/analysis/deliverable06/fonsi_evidence_spans.parquet"
ENRICH = REPO / "phase2/data/analysis/deliverable06/fonsi_enrichment.parquet"
sys.path.insert(0, str(REPO / "phase2/code/deliverable06"))
sys.path.insert(0, str(REPO / "phase2/code/extract"))

from mitigation_conditions import RESOURCE_AREAS, classify_resource_area  # baseline keyword dict

OUT = []
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line); OUT.append(line)

# ---------------------------------------------------------------- load + baseline
con = duckdb.connect()
df = con.execute(f"""
    SELECT a.project_id, a.section_id, a.condition_text, a.resource_area AS baseline_area,
           a.condition_role, b.heading_title
    FROM '{COND}' a
    LEFT JOIN (SELECT DISTINCT section_id, heading_title FROM '{SPANS}') b
      ON a.section_id = b.section_id
""").df()
N = len(df)
base_unknown = (df.baseline_area == "unknown").sum()
log(f"[baseline] rows={N}  unknown={base_unknown} ({100*base_unknown/N:.1f}%)")
log("[baseline] resource_area dist:")
for area, cnt in df.baseline_area.value_counts().items():
    log(f"    {area:14s} {cnt:6d} ({100*cnt/N:.1f}%)")

# ---------------------------------------------------------------- Tier 1: heading inheritance
# Classify the HEADING text with the same resource keyword dict. Generic structural headings
# ("Environmental Consequences", "Mitigation Measures", "Decision", "FONSI") contain no resource
# keyword -> stay unknown; resource-specific headings ("Wildlife", "Cultural Resources") resolve.
def classify_heading(h: str) -> str:
    if not isinstance(h, str) or not h.strip():
        return "unknown"
    return classify_resource_area(h)

df["heading_area"] = df.heading_title.map(classify_heading)
# tier1 tag = baseline if known, else heading inheritance
df["tier1_area"] = np.where(df.baseline_area != "unknown", df.baseline_area, df.heading_area)
t1_unknown = (df.tier1_area == "unknown").sum()
newly_t1 = ((df.baseline_area == "unknown") & (df.tier1_area != "unknown")).sum()
log(f"\n[tier1] heading inheritance resolved {newly_t1} of {base_unknown} unknowns "
    f"({100*newly_t1/base_unknown:.1f}% of unknowns)")
log(f"[tier1] unknown now {t1_unknown} ({100*t1_unknown/N:.1f}%)")
log("[tier1] top headings that RESOLVED a previously-unknown condition:")
res = df[(df.baseline_area == "unknown") & (df.tier1_area != "unknown")]
for (h, ar), cnt in res.groupby(["heading_title", "tier1_area"]).size().sort_values(ascending=False).head(15).items():
    log(f"    {cnt:5d}  [{ar:12s}] {str(h)[:55]}")

# ---------------------------------------------------------------- Tier 2: local embeddings
PROTOTYPES = {
    "biological": [
        "Timing restrictions will avoid disturbing nesting migratory birds during breeding season.",
        "Pre-construction wildlife and raptor surveys will be conducted before ground disturbance.",
        "Equipment will be washed to prevent the spread of invasive and noxious weeds.",
        "Vegetation clearing will be minimized and disturbed areas will be reseeded with native species.",
        "Biological monitors will be present to protect special-status species and habitat.",
    ],
    "water": [
        "A stormwater pollution prevention plan will control runoff and sediment to surface waters.",
        "Best management practices will protect wetlands, streams, and water quality.",
        "A spill prevention and response plan will be maintained for fuels and hazardous liquids.",
        "Erosion and sediment controls will be installed to prevent discharge into drainages.",
        "Work will avoid impacts to groundwater and floodplains.",
    ],
    "cultural": [
        "If unanticipated cultural or archaeological resources are discovered, work will stop and a monitor consulted.",
        "A cultural resources monitor will be present during ground-disturbing activities.",
        "Tribal consultation and Section 106 procedures will be followed for historic properties.",
        "Discovered human remains or artifacts will be protected and reported to the tribe.",
        "Paleontological resources will be avoided and recorded.",
    ],
    "soils_geology": [
        "Topsoil will be salvaged, stockpiled, and respread during reclamation.",
        "Erosion control measures will limit soil compaction and loss on disturbed slopes.",
        "Reclamation will recontour and stabilize disturbed ground to prevent erosion.",
        "Construction will avoid prime farmland and minimize surface disturbance.",
        "Soil stabilization and revegetation will follow ground disturbance.",
    ],
    "air_quality": [
        "Dust suppression via watering will control fugitive dust during construction.",
        "Equipment will comply with emission standards to limit criteria pollutants.",
        "Fugitive dust and particulate matter will be minimized on unpaved roads.",
        "Idling of diesel equipment will be limited to reduce air emissions.",
        "A dust control plan will be implemented for earthwork.",
    ],
    "noise": [
        "Construction noise will be limited to daytime hours near sensitive receptors.",
        "Equipment mufflers will reduce operational noise levels.",
        "Blasting and vibration will be scheduled to minimize disturbance.",
        "Noise levels at nearby residences will be kept below the applicable decibel limit.",
        "Noise-generating activities will avoid nighttime hours.",
    ],
    "transportation": [
        "Haul routes and traffic control will minimize disruption on public roads.",
        "Access roads will be maintained and dust-controlled during use.",
        "Truck trips will follow designated routes to limit traffic impacts.",
        "A traffic management plan will govern construction vehicle movements.",
        "Roads damaged by construction traffic will be repaired.",
    ],
    "visual": [
        "Structures will be painted non-reflective colors to reduce visual contrast.",
        "Lighting will be shielded and downcast to minimize night sky impacts.",
        "Facilities will be sited to reduce visibility from scenic viewpoints.",
        "Visual screening will limit views of the project from key observation points.",
        "Surface treatments will blend the facility with the surrounding landscape.",
    ],
    "public_health": [
        "Hazardous materials will be stored and handled to prevent contamination and exposure.",
        "A health and safety plan will manage risks to workers and the public.",
        "Contaminated soils will be characterized and disposed of properly.",
        "Fire prevention measures will reduce wildfire risk.",
        "Toxic and hazardous waste will be managed per applicable regulations.",
    ],
    "land_use": [
        "Work will remain within the existing right-of-way and designated easements.",
        "The project will be consistent with the applicable land use plan and zoning.",
        "Surface use will be coordinated with grazing and adjacent land uses.",
        "Disturbance will be confined to previously authorized corridors.",
        "Reclamation will return land to its prior use.",
    ],
    "socioeconomic": [
        "Local hiring and procurement will support the surrounding community.",
        "Environmental justice communities will not bear disproportionate impacts.",
        "Public services and housing demand impacts will be monitored.",
        "Coordination with local governments will address community concerns.",
        "Compensation will be provided for affected landowners.",
    ],
    "climate_ghg": [
        "Greenhouse gas emissions will be minimized through equipment efficiency.",
        "Methane and carbon dioxide emissions will be reduced where feasible.",
        "The project will limit its net greenhouse gas and climate footprint.",
        "Vehicle idling limits will reduce carbon emissions.",
        "Emission reduction measures will address climate change concerns.",
    ],
}

try:
    import embeddings as emb  # D6 helper (all-MiniLM-L6-v2)
    HAVE_EMB = emb.available()
except Exception as e:
    HAVE_EMB = False
    log(f"[tier2] embeddings unavailable: {e}")

df["tier2_area"] = df["tier1_area"].copy()
df["tier2_sim"] = np.nan
THRESHOLDS = [0.45, 0.55]
tier2_stats = {}
if HAVE_EMB:
    areas = list(PROTOTYPES.keys())
    proto_texts = [p for a in areas for p in PROTOTYPES[a]]
    proto_area = [a for a in areas for _ in PROTOTYPES[a]]
    log(f"\n[tier2] embedding {len(proto_texts)} prototypes + unknown conditions (all-MiniLM-L6-v2)...")
    P = np.asarray(emb.embed(proto_texts))               # (P, d) normalized
    unk_mask = df.tier1_area == "unknown"
    unk_idx = df.index[unk_mask].tolist()
    texts = df.loc[unk_idx, "condition_text"].fillna("").astype(str).tolist()
    log(f"[tier2] {len(texts)} still-unknown conditions to embed")
    C = np.asarray(emb.embed(texts))                      # (U, d)
    sims = C @ P.T                                        # (U, P)
    # best prototype per condition, aggregated to area = max over that area's prototypes
    area_of = np.array(proto_area)
    best_area, best_sim = [], []
    for row in sims:
        # max sim per area
        per = {}
        for a, s in zip(area_of, row):
            if s > per.get(a, -1): per[a] = s
        ba = max(per, key=per.get)
        best_area.append(ba); best_sim.append(per[ba])
    best_area = np.array(best_area); best_sim = np.array(best_sim)
    df.loc[unk_idx, "tier2_cand_area"] = best_area
    df.loc[unk_idx, "tier2_sim"] = best_sim
    for th in THRESHOLDS:
        assign = best_sim >= th
        resolved = int(assign.sum())
        tier2_stats[th] = resolved
        log(f"[tier2] threshold {th}: resolves {resolved} of {len(texts)} still-unknown "
            f"({100*resolved/max(len(texts),1):.1f}%)  -> combined unknown "
            f"{100*(t1_unknown-resolved)/N:.1f}%")
    # materialize tier2 tag at the 0.45 threshold (report both, use 0.45 as the working combined)
    TH_MAIN = 0.45
    unk_idx_arr = np.array(unk_idx)
    sel_mask = best_sim >= TH_MAIN
    df.loc[unk_idx_arr[sel_mask], "tier2_area"] = best_area[sel_mask]
else:
    log("[tier2] SKIPPED (sentence-transformers not available in this env)")

# ---------------------------------------------------------------- proxy precision (a): enrichment
enr = con.execute(f"SELECT project_id, mitigation_resource_areas FROM '{ENRICH}'").df()
def parse_list(v):
    try:
        x = json.loads(v) if isinstance(v, str) else v
        return set(str(i) for i in x) if isinstance(x, (list, tuple)) else set()
    except Exception:
        return set()
proj_areas = {r.project_id: parse_list(r.mitigation_resource_areas) for r in enr.itertuples()}
VEG_ALIAS = {"biological", "soils_geology"}  # enrichment 'vegetation' ~ these baseline areas

def agree(pred, plist):
    if pred == "unknown" or not plist:
        return None
    if pred in plist:
        return True
    if pred in VEG_ALIAS and "vegetation" in plist:  # fair-credit vegetation
        return True
    return False

df["_pl"] = df.project_id.map(lambda p: proj_areas.get(p, set()))
df["_has_enr"] = df.project_id.isin(proj_areas)
sub = df[df._has_enr].copy()
log(f"\n[proxy-a] enrichment cross-check universe: {len(sub)} conditions in {sub.project_id.nunique()} enriched projects")

def agreement(col, frame):
    a = frame.apply(lambda r: agree(r[col], r["_pl"]), axis=1)
    a = a.dropna()
    return (a.mean() if len(a) else float("nan")), len(a)

for col in ["baseline_area", "tier1_area", "tier2_area"]:
    rate, n = agreement(col, sub)
    log(f"[proxy-a] overall agreement  {col:14s}: {rate:.3f}  (n non-unknown tagged = {n})")

# newly-resolved agreement (the real signal): conditions baseline left unknown
res1 = sub[(sub.baseline_area == "unknown") & (sub.tier1_area != "unknown")]
r1, n1 = agreement("tier1_area", res1)
log(f"[proxy-a] NEWLY-RESOLVED agreement (tier1 heading): {r1:.3f}  (n={n1})")
if HAVE_EMB:
    res2 = sub[(sub.tier1_area == "unknown") & (sub.tier2_area != "unknown")]
    r2, n2 = agreement("tier2_area", res2)
    log(f"[proxy-a] NEWLY-RESOLVED agreement (tier2 embed @0.45): {r2:.3f}  (n={n2})")

# ---------------------------------------------------------------- proxy precision (b): eyeball 20
random.seed(47)
ex_rows = []
pool1 = df[(df.baseline_area == "unknown") & (df.tier1_area != "unknown")]
pool2 = df[(df.tier1_area == "unknown") & (df.tier2_area != "unknown")] if HAVE_EMB else df.iloc[0:0]
for tier, pool in [("tier1", pool1), ("tier2", pool2)]:
    take = pool.sample(min(10, len(pool)), random_state=47) if len(pool) else pool
    for _, r in take.iterrows():
        newtag = r.tier1_area if tier == "tier1" else r.tier2_area
        ex_rows.append({
            "tier": tier, "old_tag": r.baseline_area, "new_tag": newtag,
            "sim": round(r.tier2_sim, 3) if (tier == "tier2" and pd.notna(r.tier2_sim)) else "",
            "heading": str(r.heading_title)[:40],
            "condition_text": str(r.condition_text)[:180].replace("\n", " "),
        })
ex = pd.DataFrame(ex_rows)
ex.to_csv(HERE / "pilot47_examples.csv", index=False)
log(f"\n[proxy-b] wrote 20-example eyeball table -> pilot47_examples.csv ({len(ex)} rows)")

# save tags for reuse
df[["project_id", "section_id", "baseline_area", "heading_area", "tier1_area",
    "tier2_area", "tier2_sim"]].to_parquet(HERE / "pilot47_tier_tags.parquet", index=False)

# ---------------------------------------------------------------- residual cost projection
if HAVE_EMB:
    resid = int((df.tier2_area == "unknown").sum())
    log(f"\n[residual] after tier1+tier2@0.45: {resid} unknown ({100*resid/N:.1f}%) -> LLM residual pass")
    # ~250 input tokens/condition (text + short prompt) + ~15 output; Haiku $0.80/M in, $4/M out
    in_tok = resid * 250; out_tok = resid * 15
    hk = in_tok/1e6*0.80 + out_tok/1e6*4.0
    sn = in_tok/1e6*3.0 + out_tok/1e6*15.0
    log(f"[residual] rough LLM cost for {resid} rows: Haiku ~${hk:.1f}, Sonnet ~${sn:.1f} "
        f"(batchable; ~250 in / 15 out tok each)")

(HERE / "pilot47_summary.txt").write_text("\n".join(OUT) + "\n")
print(f"\n[done] summary -> {HERE/'pilot47_summary.txt'}")
