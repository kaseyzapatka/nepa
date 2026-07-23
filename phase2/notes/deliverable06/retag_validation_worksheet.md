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


**Sheet:** `/Users/Dora/git/consulting/nepa/phase2/notes/deliverable06/retag_validation_sheet.csv` — 80 rows to label.


## Stratum: `new_haiku` (24 rows)

### `1bd3cc7daee176ab`

> If construction activities could not avoid direct impacts to aquatic resources, appropriate permits would be obtained prior to any disturbance.

- old_tag: `unknown`
- new_tags: `water,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `329c7c6bc4ff72d6`

> RDF 121 Yes Maintain a properly functioning overflow to prevent water from flowing onto the pad and surrounding area, to eliminate or minimize pooling of water that is attractive to breeding mosquitoes.

- old_tag: `unknown`
- new_tags: `water,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `414aa807ac2afe5e`

> This alternative would minimize the potential for the public to encounters AML-related dangers while still providing non-mechanized travel and relatively high levels of motorized travel near AML sites.

- old_tag: `unknown`
- new_tags: `public_health,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `421840a7b71751eb`

> Health and Safety For the safety and protection of the surface and surrounding area, the operator must keep the area clear of trash and other debris as much as possible to avoid damaging or contaminating the human and environmental health surrounding the well pad location.

- old_tag: `unknown`
- new_tags: `public_health,soils_geology,water`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `42d7731b21b8f877`

> Droplet size and pressure of the herbicide applicator will be controlled carefully to minimize particle drift.

- old_tag: `unknown`
- new_tags: `air_quality,water,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `6008da2257eafc5c`

> The use of the film crew’s polar bear interaction plan, project design features, project specific ROPS, and ROPs from the 2022 NPR-A Integrated Activity Plan would limit disturbances and mitigate impacts from filming activities to polar bears.

- old_tag: `unknown`
- new_tags: `biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `611d1b064e433e6f`

> They are limited in the amount of material accepted, and recycling programs are present to minimize waste.

- old_tag: `unknown`
- new_tags: `soils_geology,socioeconomic`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `6ca190a8ed6d5976`

> Project siting, design, and landscape features will avoid impacts in this instance and support such departure.

- old_tag: `unknown`
- new_tags: `visual,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `706c50b775571cbe`

> Project design implements strategies to protect special status and sensitive resources in special use areas to avoid adverse impacts to those resources.

- old_tag: `unknown`
- new_tags: `biological,cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `74092726fdeb3dac`

> Mitigation Measures: The following measures are included in an effort to minimize the impacts ofthe proposed project to social and natural environmental resources.

- old_tag: `unknown`
- new_tags: `socioeconomic,biological,water,air_quality,soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `7a1862637afe213e`

> This would be partially mitigated by replanting this area with shrubs.

- old_tag: `unknown`
- new_tags: `biological,visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `7a69b644de079296`

> The exploration POO also proposes to complete concurrent reclamation which would minimize the total active disturbance acreage during exploration.

- old_tag: `unknown`
- new_tags: `land_use,soils_geology,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `905ee32316b80a72`

> Recommended mitigation for indirect impacts to recreation include the requirement that larger permitted vehicles clearly display a copy of the authorization that they are allowed to be there.

- old_tag: `unknown`
- new_tags: `socioeconomic,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `952ca1de45cd3967`

> NOI-4: If blasting becomes necessary, efforts will be made to restrict the peak overpressures to less than 120 dB at the source to minimize effects to surrounding areas.

- old_tag: `unknown`
- new_tags: `noise,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `96431838f247f81d`

> Chapter 2, Section 2.1 details the designs proposed for these structures that would minimize their visibility.

- old_tag: `unknown`
- new_tags: `visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `a4567add74d04343`

> During the life of the development, aU disturbed areas not needed for active support of production operations should undergo "interim" reclamation in order to minimize the environmental impacts ofdevelopment on other resources and used.

- old_tag: `unknown`
- new_tags: `soils_geology,land_use,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b0056aa290023621`

> Practical Grazing Managment to Maintain or Restore Riparian Functions and Values on Rangeland.

- old_tag: `unknown`
- new_tags: `water,biological,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b643c01002f8a87e`

> Each turbine is fitted with a lightning protection system (arrestor) to minimize the fire risk.

- old_tag: `unknown`
- new_tags: `public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `c914327f98a041cc`

> The wells would be "dual purpose" to minimize the number of wells needed.

- old_tag: `unknown`
- new_tags: `land_use,water`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `db3a7959adc40a7f`

> After construction (BPA) Coordinate with local agencies to avoid construction activities that could conflict with their own construction activities.

- old_tag: `unknown`
- new_tags: `transportation`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `dbfeae31c5d9ae2b`

> Minimize idling construction equipment, if feasible.

- old_tag: `unknown`
- new_tags: `air_quality,noise`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `efe2b41dc78b18b4`

> In efforts to reduce existing and future unauthorized uses, Mitigation Measure REC-1 is provided below.

- old_tag: `unknown`
- new_tags: `land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `fad48bd08353dc15`

> Also, shipping distances for many of the materials would be minimized to the extent practicable to meet LEED standards, thus fuel use would be reduced.

- old_tag: `unknown`
- new_tags: `climate_ghg,air_quality`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `fbbf04e0cd814b9c`

> If previously unknown sites are discovered, the BLM would avoid the sites through establishment of exclusion areas.

- old_tag: `unknown`
- new_tags: `cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______


## Stratum: `new_tier1` (10 rows)

### `094d4ff26566947b`

> Appendix B for the proposed action contains management practices to minimize the degree of potential negative impacts, to the extent possible.

- old_tag: `unknown`
- new_tags: `soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `14a22d0335f6c5e2`

> Regular site visits, currently part of the BLM's LTC management program, would be critical for continuing to inform and educate permit holders on these values and how to mitigate impacts.

- old_tag: `unknown`
- new_tags: `water`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `207c324fe4948e6e`

> Protection or relocation of essential infrastructure will be required to avoid effects to these features.

- old_tag: `unknown`
- new_tags: `transportation`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `5f7d308ba0501e7d`

> Resource avoidance associated with the widening and resurfacing of CR 120 may not be feasible due to engineering and/or ROW restrictions.

- old_tag: `unknown`
- new_tags: `cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `6b6f369459ba7b28`

> As a result, mitigation to resolve those adverse effects would be necessary.

- old_tag: `unknown`
- new_tags: `cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `844a1cf8f516521d`

> Census Bureau as Black or African American, American Indian and Alaska Native, Asian, Native Hawaiian and other Pacific Islander, Hispanic or Latino, and those classified under “two or more races.” Hispanics may be of any race and are excluded from the totals for individual races to avoid double counting.

- old_tag: `unknown`
- new_tags: `socioeconomic`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `90e226c191dc4bce`

> With the adherence to the following Conservation Measures direct impacts should be avoided, and indirect impacts should be minimized.

- old_tag: `unknown`
- new_tags: `biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `949a52018b09b46f`

> Further, FCC regulations require that the operators of these devices mitigate such interference.

- old_tag: `unknown`
- new_tags: `public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `aa2d2578fedef9e7`

> The degree of potential direct and indirect effects from these actions is dependent on the duration of the action, and the types of protective measures used to minimize adverse effects.

- old_tag: `unknown`
- new_tags: `biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `aca23b077a535e32`

> Mitigation and precautions apply to the proposed action alternative.

- old_tag: `unknown`
- new_tags: `visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______


## Stratum: `changed` (18 rows)

### `0774404c461a5e97`

> Well pad preparation well pad and access roads Grading of pads and access roads Yes sump Excavations for sump Yes Installation of well well drilling None anticipated (drilling - diameter <18") No pumpjack & equipment Excavation & drilling (>18" diameter) for foundation Yes Installation of pipelines Above-ground sleepers None anticipated (drilling for sleeper support foundations - diameter <18") No Below ground pipelines Trenching for subgrade pipelines Yes Condensate Pot None anticipated (earthwork not anticipated) No Installation of electrical lines Pole stringing None anticipated (above ground work only) No Transformer bank upgrades None anticipated (above ground work only) No New service & bank poles Drilling for pole foundations - diameter >24" Yes * small diameter drilling (<18”), mechanical compaction, and hydroexcavation cannot be feasibly mitigated for paleontological resources ‡ earthwork impacting previously undocumented artificial fill deposits is not required for paleontological mitigation

- old_tag: `transportation`
- new_tags: `soils_geology,cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `1ad80d3f4019d0fd`

> Furthermore, burn units are typically demarcated by already defensible controls such as roads, thus the length of hand line creation is further minimized.

- old_tag: `transportation`
- new_tags: `soils_geology,air_quality`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `208c7a2e34f3ce56`

> However, because of the extensive risk management practices that would be implemented to manage biological hazards in the HTRL workplace, impacts to the health and safety of laboratory workers would be minimized.

- old_tag: `biological`
- new_tags: `public_health,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `2efc697cb9f6b99b`

> The Proposed Action’s contribution to global warming would be minor because the amount of tree clearing would be small and because low-growing vegetation would naturally revegetate cleared areas.

- old_tag: `biological`
- new_tags: `climate_ghg,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `349d1b529f86c646`

> To mitigate this potential conflict, additional signing would be placed during winter months to warn users to stay off the designated winter sled dog trails during winter. • Homeless and Illegal Campers- The proposed action would have no effect to existing homeless and illegal camping on CT. • Wildlife Encounters- The proposed action would temporarily increase the likelihood of a negative wildlife encounter until wildlife become habituated to humans in the previously undisturbed forested landscape. • Users in Aviation and Administrative Areas- The proposed action would have no effect to the impacts from users in aviation and administrative areas.

- old_tag: `biological`
- new_tags: `transportation,biological,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `361db75af712f229`

> These impacts wodd be mitigated by taking appropriate erosion conmol measures, and the construction of a new storrnwatir retention basin.

- old_tag: `soils_geology`
- new_tags: `water,soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `381c4a49d384ec27`

> Mitigation for the destruction of the mining claim monuments shall be the responsibility of the Developers/Owners and would consist of the erection of a witness monument outside of the proposed right of way for each mining claim monument destroyed.  Any affected mining claimant(s) shall be notified by the Developers/Owners, the BLM would provide the mining claimant contact information, of the proposed replacement of the mining claim monument(s) with witness monument(s) prior to the destruction of the original monument(s).  Each witness monument would be fitted with an embossed brass or aluminum tag indicating the relative location of the original mining claim monument from the witness monument.

- old_tag: `land_use`
- new_tags: `cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `45a4f4311697c729`

> In places such as the Tuttletown and Glory Hole Recreation Areas, implementation of REC-SOP-1, along with SOPs for aesthetics, air quality, noise, and public health, would ensure that conflicts with established recreational areas would be minimized to acceptable levels (see Table 2.4-1).

- old_tag: `air_quality`
- new_tags: `visual,air_quality,noise,public_health,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `50fe2fe24773aaba`

> The mitigation measures are organized around the key issues developed for the analysis: Effects on the wild character of the Gulkana National Wild River.

- old_tag: `water`
- new_tags: `visual,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `528c53eb2600e21d`

> The following describes the practices and the procedures they would follow to mitigate the impact of contaminants from refueling operations on fish, wildlife and the environment.

- old_tag: `biological`
- new_tags: `water,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `6299f22b9173ad59`

> Air emissions would be below de minimis levels; no new soil disturbance would occur and design features would minimize erosion and instability; and .

- old_tag: `soils_geology`
- new_tags: `air_quality,soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `7928a27bc9536ce0`

> Areas overlapping with these viewsheds represent areas bighorn sheep would avoid with increased human use associated with actions in the various alternatives.

- old_tag: `visual`
- new_tags: `biological,visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `a6366f92e294f018`

> The analysis found that impacts of this specific project would be moderate to major and localized and that the mitigation measures in Appendix B and Appendix C, and project specific stipulations from Section 4.4 would reduce adverse effects to Acoustical Environment, Subsistence, Sociocultural Systems, and Environmental Justice to the greatest degree feasible.

- old_tag: `cultural`
- new_tags: `noise,socioeconomic,cultural,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `a7f59aab97cd70b1`

> No GPL-REC- 6 If designated vehicle routes are directly impacted by activities (includes modification of existing route to accommodate industrial equipment, restricted access or full closure of designated route, pull outs, and staging area’s to the public, etc.), mitigation will include the development of alternative routes to allow for continued vehicular access with proper signage, with a similar recreation experience.

- old_tag: `transportation`
- new_tags: `land_use,socioeconomic,visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b0dc1b5d0505c7f3`

> To mitigate the potential impacts identified above, the following mitigation measures would be implemented: • Affected farmers would receive compensation for lost crop production caused by the construction of the project. • Equipment operators and the construction crew would be instructed to close gates to avoid disturbances to livestock, and to stay within the ROW to minimize impacts to crops. • To minimize the establishment of noxious weeds, construction crews would wash equipment and vehicles before entering construction areas. • Marker balls would be installed on the conductor as it crosses the North Santiam River to make it more visible to pilots. • BPA would compensate landowners to disc or till soil to reduce soil compaction from equipment once construction is completed. • Conduct construction activities in coordination with agricultural activities.

- old_tag: `soils_geology`
- new_tags: `socioeconomic,land_use,soils_geology,visual,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b71f08f7be3a7119`

> KSU has committed to obtain and comply with appropriate federal, state, and local permits required for the Proposed Project, and to minimize or avoid potential environmental effects to land uses, biological resources, cultural resources, visual resources, transportation, and the health and safety of construction workers, faculty, students, and the public through the implementation of the protection measures detailed in section 2.5 of the EA.

- old_tag: `visual`
- new_tags: `land_use,biological,cultural,visual,transportation,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `d1317d9c60208840`

> Mitigation measures (see below) would be implemented to help restore similar plant communities to the extent possible, and the existing functional values of the wetland buffer for habitat and water quality improvement would be altered for the duration of the project, a low to moderate effect.

- old_tag: `water`
- new_tags: `biological,water`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `f86fca49d0ed14c8`

> Statutes govern the generation, treatment, storage and disposal of hazardous materials, substances, and waste, and the investigation and mitigation of waste releases, air and water quality, human health, and land use.

- old_tag: `public_health`
- new_tags: `air_quality,water,public_health,land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______


## Stratum: `unchanged` (14 rows)

### `0126567d34e4f4eb`

> Any direct effect would consist of short-term turbidity due to construction activity, which would minimally affect fish downstream with work performed within the in-water work window and by implementing conservation measures to minimize any potential effects.

- old_tag: `water`
- new_tags: `water,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `15f46eb480aee6f1`

> The BLM must conduct a project-specific NEPA analysis and determine whether the proposed project should be approved, rejected, or approved with modifications, and if additional mitigation is needed.

- old_tag: `socioeconomic`
- new_tags: `socioeconomic`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `1b6e7938efc15395`

> Because none of the significance criteria would be met by the implementation of the Proposed Action, no mitigation measures specific to visual resources are recommended.

- old_tag: `visual`
- new_tags: `visual`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `28262eceb41da13d`

> Much depends on the rate at which temperature will continue to rise and whether global emissions of greenhouse gases can be mitigated before serious ecological thresholds are reached.

- old_tag: `climate_ghg`
- new_tags: `climate_ghg,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `33b472d81ee68c53`

> Based on the analysis of the potential effects of the proposed Project, this study concludes that with Mitigation Measures in place, the impacts associated with geology and soils resources will be less than significant and that the Project: a.

- old_tag: `soils_geology`
- new_tags: `soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `3d080bc43aa0736a`

> Trenching through stream banks and channels alters habitat and substrate characteristics, and therefore their productivity and should be avoided.

- old_tag: `water`
- new_tags: `water,biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `55679ff5eddbc4ff`

> All mitigation measures have been resolved by stipulations attached to the right-of-way grant.

- old_tag: `land_use`
- new_tags: `land_use`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `7670567b11f726f7`

> Implementation of the management strategies, actions, and minimization measures selected for incorporation into the plan would improve public health or safety.

- old_tag: `public_health`
- new_tags: `public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `86e6d2d5ef3b27df`

> IWRB will also minimize interference with, disturbance to, and damage of all nesting birds granted protection by the MBTA, and will not destroy any occupied migratory bird nests.

- old_tag: `biological`
- new_tags: `biological`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `91e5a98ce68cee3a`

> Because the proposed construction actions are short-term and small in scale with minimal potential to affect air quality, noise, or public safety, and the ongoing actions produce minimal effects and are consistent with those on surrounding lands, the overall effect of the Proposed Action on air quality, noise, and public safety would be low, and would be mitigated by the application of the measures in Section 2.3 “Mitigation Measures.” Effects of the No Action Alternative The No Action Alternative would cease operations at the existing facilities, eliminating all current sources of impacts to air quality and noise.

- old_tag: `air_quality`
- new_tags: `air_quality,noise,public_health`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `9594b658ef2e9fab`

> These concerns were discussed, and the NETL's plans to minimize traffic impacts were clearly presented during the meeting.

- old_tag: `transportation`
- new_tags: `transportation`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `9678ba2945667634`

> While significant impacts to neighboring areas would not be anticipated, mitigation measures would be implemented to minimize construction noise impacts.

- old_tag: `noise`
- new_tags: `noise`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `c366dc3a671fd532`

> With implementation of these measures, impacts to cultural resources will be minimized.

- old_tag: `cultural`
- new_tags: `cultural`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `e794291f3a4dbfed`

> Following construction activities, all temporarily disturbed areas would be revegetated, resulting in the permanent removal of 2.77 acres of vegetation, with 1.40 of these acres on BLM-administered public land.

- old_tag: `biological`
- new_tags: `biological,land_use,soils_geology`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______


## Stratum: `still_unknown` (14 rows)

### `1f40193ac5919170`

> These impacts are minimized and/or avoided using the Design Criteria (p10), Management Restrictions and Standard Operating Procedures (Appendix E, p75), and ID/swMT ARMPA Management Decisions and FONSI DOI-BLM-ID-I020-2018-0004-EA Required Design Features (Appendix F, p81) found in the EA.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `1f7f785ba0cc326d`

> The Field Manager is the responsible officer who will decide one of the following: • To approve issuance of the ROW grant with the design features as submitted; • To approve issuance of the ROW grant as proposed with design features and additional mitigation added; • To analyze the effects of the Proposed Action in an EIS; or • To deny issuance of the ROW grant.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `2157da3adce92441`

> This MAP is for the Proposed Action and includes all of the integral elements and commitments made in the Environmental Assessment (EA) to mitigate any potential adverse environmental impacts.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `2721ae657b89d376`

> None of the direct, indirect, or cumulative effects associated with implementation of the management strategies, actions, and minimization measures selected for incorporation into the plan are considered significant, either individually or cumulatively, based on the analyses provided in the EA.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `3d165dbe969044eb`

> The EA describes typical exploration and development activities that could occur on a federal lease (RFDS) along with the potential impacts from those activities as well as mitigation measures designed to minimize or eliminate impacts (EA Section 3.0).

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `3d565e03cdc0700e`

> The BLM has analyzed and approved numerous projects involving SRPs for winter recreational use and the use of explosives for avalanche mitigation as well as routine ROW grants, and the effects of this project are not considered to be controversial, nor is there scientific dispute about these effects.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `45853acb4bc98f59`

> MITIGATION MEASURES The mitigation measures in Table 1 have been identified to reduce potential impacts to environmental resources from the project.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `8af54f8e549a7511`

> COMMENT: Commentors want mitigation measures more extensively discussed in the PEA rather than assuming adequate mitigation will occur through the site-specific approval process and stipulations in lease agreements.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `8ee556fe9e411727`

> Additional information regarding mitigation can be found in Section 8.4.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `981be4378935d711`

> The following mitigation measures would be implemented to reduce impacts from the proposed action or Alternative B.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `a396d29153e539be`

> Roche Moutonnee Temporary Easement/Bridge /Culvert Environmental Assessment DOI-BLM-AK-F030–2017–0007-EA Serial No. - F-97202 Based on the analysis of potential environmental impacts (per Environmental Assessment DOI-BLM-AK-F030–2017–0007), I have determined that the proposed action with the mitigation measures attached to the authorization will not have any significant impacts on the environment and an environmental impact statement is not required.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b21083d19d08b3fd`

> When no option is available, the CDT will consult with Level 1 Teams to identify adequate avoidance and minimization measures for the site.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `b93953dee651e3b1`

> Further mitigation measures could be identified at the APD stage.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______

### `c004c9a26f34b7d1`

> After mitigation, the direct and indirect effects of the proposed action would be negligible and therefore would not considerably contribute to any regional cumulative effects that may be occurring to such resources.

- old_tag: `unknown`
- new_tags: `unknown`
- **gold_resource_areas:** ______   **is_correct:** ______   **notes:** ______
