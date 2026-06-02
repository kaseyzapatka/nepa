# Notes: CE Solar Projects > 500 MW (Figure 15)

**Context:** CATF client inquiry about solar projects appearing in the CE band above 500 MW in the
capacity distribution violin/box figure (06_capacity_distribution_violin_box_solar.png). Keith and
a PNNL colleague asked for examples; these may warrant a footnote in the report.

**Data source:** `phase1/data/analysis/projects_gencap.parquet`, filtered to:
- `project_type` contains `"Renewable Energy Production - Solar"`
- `dataset_source == "CE"`
- `capacity_mw > 500` and `<= 5000`
- Capacity = `coalesce(project_gencap_final_value, project_gencap_value)`, unit-converted to MW

**Result:** 8 records across 7 distinct projects (Ponderosa appears twice — likely a duplicate).
All 8 confirmed to have the Solar tag on `project_type`.

---

## Category 1: Gen-tie / transmission actions referencing a larger solar project's capacity

The CE covers only transmission infrastructure, but the capacity figure is extracted from language
describing the upstream solar project the line serves — not from what is being categorically excluded.

| project_id | Title | Capacity | State | Agency |
|---|---|---|---|---|
| `4dfe91d0-2a23-5dc1-c090-0956fbd91c18` | Bellefield Gen-Tie Aerial Crossing | 1,500 MW | CA | BLM |
| `4ee5541e-421b-51f2-f39d-b3812f4b34b9` | Bellefield Aerial Crossing Assignment (Additional Entities) | 1,500 MW | CA | BLM |
| `b8414c51-dc3a-ccaa-daa6-42dba1fc7418` | Wind Tests of Transmission Line Towers | 600 MW | TX | DOE |

**Bellefield (4dfe91d0, 4ee5541e):** Both CEs cover a gen-tie line crossing BLM land; the 1,500 MW
figure comes from describing the private-land solar project the line supports. The CE action is the
aerial crossing, not the solar facility.

> *"The gen-tie would support a 1,500-megawatt private land solar project."*

**ON Line / Wind Tests (b8414c51):** CE covers wind load testing of transmission towers. The 600 MW
is the first phase of a larger transmission project (ON Line), mentioned as context.

> *"The ON Line project is the first 600 MW phase of a larger project that when completed is
> expected to carry approximately..."*

---

## Category 2: Minor modifications to existing facilities

The CE covers a small operational change; capacity extracted is the overall facility's rated size,
not the scope of the CE action.

| project_id | Title | Capacity | State | Agency |
|---|---|---|---|---|
| `f9b01763-9935-c36e-3d36-ebc593ec7c1c` | Amend ROW grant to include a temporary SODAR unit | 740 MW | AZ | BLM |
| `25c9686f-6540-8c73-9a4e-9ef10171f4df` | Ponderosa Substation Breaker and Transformer Reconfiguration | 600 MW | OR | DOE |
| `ea26e714-b78e-2a17-42b6-60c0e9fc121c` | Ponderosa Substation Breaker and Transformer Reconfiguration | 600 MW | OR | DOE |

**SODAR unit (f9b01763):** CE adds a temporary wind-measurement sensor (SODAR) to Mohave County
Wind Farm. The 740 MW is the farm's total rated capacity, not the SODAR action. Note: this project
is primarily tagged Wind, Onshore — Solar is a secondary tag.

**Ponderosa (25c9686f, ea26e714):** CE reconfigures a BPA substation breaker and transformer
layout. The 600 MW is the solar input the reconfiguration would *facilitate* from Ponderosa Solar
LLC's proposed generating facility — not directly approved capacity. These two records appear to be
the same underlying project (possible duplicate in source data).

> *"...would facilitate an additional 600 MW of electrical load input from Ponderosa Solar LLC's
> proposed solar generating facility."*

---

## Category 3: Administrative or non-construction actions

| project_id | Title | Capacity | State | Agency |
|---|---|---|---|---|
| `80df880f-7b79-9e11-4a6b-c54dadb04468` | Riverside County Solar Farm Land Lease Market Study | 1,535 MW | CA | BLM |
| `7c4865d1-aabf-7de9-2c3d-5152cbbb4793` | Project Eagle Direct Wafer Manufacturing Plants | 980 MW | MA | DOE |

**Riverside County Market Study (80df880f):** CE is a land lease *market study* — no construction.
The 1,535 MW is the aggregate capacity of sites being appraised for their underlying land value.

> *"In aggregate the sites total 14,955 acres and are improved with up to 1,535 MW of solar energy
> power. Only the underlying land as described below is being valued..."*

**Project Eagle (7c4865d1):** CE covers a solar *wafer manufacturing* plant (1366 Technologies,
Lexington MA). The "980 MW" is silicon wafer production capacity per year — a measurement unit
mismatch where MW refers to solar product throughput, not electrical generation output. Tagged
primarily as Manufacturing + Semiconductors; Solar tag reflects the product type.

> *"...and 980 MW of silicon wafers annually at an existing facility..."*

---

## Summary for footnote

Every project in the CE > 500 MW band falls into one of three patterns:

1. **Transmission/gen-tie actions** where the large capacity figure describes an adjacent solar
   project mentioned in the CE document, not the action being categorically excluded.
2. **Minor modifications** (equipment additions, substation reconfigurations) where the extracted
   capacity is the existing or planned facility size, not the CE scope.
3. **Administrative or non-generation actions** (market studies, manufacturing plants) where the
   MW figure is either appraised-facility capacity or a unit-mismatched production metric.

None represent a large-scale solar generation project being fully approved under a CE. Keith's and
PNNL's intuition — minor modifications or initial administrative actions — is accurate.

**Possible footnote language:** *"CEs with reported capacities above 500 MW reflect administrative
actions, minor modifications, or transmission approvals where the capacity figure describes an
associated facility rather than the scope of the CE action itself."*

---

## Potential data quality flags

- **Ponderosa duplicate:** `25c9686f` and `ea26e714` appear to be the same project with identical
  title, sponsor, agency, capacity, and source quote — worth investigating in source data.
- **SODAR / Wind Tests:** Solar tag is incidental (secondary type); primary activity is wind or
  transmission. May warrant review of multi-type project tagging logic.
- **Project Eagle:** 980 MW is wafer production throughput, not electrical output — a known
  limitation of unit-agnostic capacity extraction from free text.
