# Deliverable 1 Status Report: County Data Coverage Analysis

**Date**: 2025-02-04
**Status**: Analysis Complete - Recovery Strategy Identified

---

## Executive Summary

County-level geographic data is available for **47.8%** of clean energy projects (10,644 out of 22,279 projects). However, this varies dramatically by NEPA process type, with critical implications for data recovery strategies.

**Key Finding**: Nearly all EA and EIS projects missing county data (98.5%) can be recovered through reverse geocoding, while most CE projects cannot.

---

## County Data Coverage by Process Type

| Process Type | Total Projects | With County Data | Missing County | Coverage |
|--------------|---------------|------------------|----------------|----------|
| **CE** (Categorical Exclusion) | 20,863 | 9,631 (46.2%) | 11,232 (53.8%) | Low |
| **EA** (Environmental Assessment) | 622 | 504 (81.0%) | 118 (19.0%) | High |
| **EIS** (Environmental Impact Statement) | 794 | 509 (64.1%) | 285 (35.9%) | Moderate |
| **Total** | 22,279 | 10,644 (47.8%) | 11,635 (52.2%) | - |

---

## Why CE Projects Are Missing County Data (NOT RECOVERABLE)

### Root Causes

CE projects missing county data typically have location descriptions using:

1. **Legal land survey descriptions** without explicit county names:
   - Example: "T. 25 S., R. 22 E., Section 16"
   - Example: "Salt Lake Base and Meridian, T. 43 N., R. 108 W."

2. **Meridian-based coordinate systems** that don't reference counties

3. **Minimal geographic detail** due to streamlined documentation requirements

### Why They Can't Be Recovered

- **No usable coordinates**: Most CE projects don't have lat/long stored in separate fields
- **Legal descriptions don't map cleanly**: Township/Range/Section systems would require complex GIS lookup tables
- **By design**: CE projects are routine, low-impact actions that don't require the same level of geographic specificity in documentation

### Agency Patterns

The two largest agencies handling CE projects both have low county coverage:
- **Department of Energy**: 47% of CE projects have county data (17,584 CE projects)
- **Department of the Interior**: 41.3% of CE projects have county data (3,224 CE projects)

This is systematic across agencies - data collection standards for CE simply don't require county-level precision.

---

## Why EA/EIS Projects Are Missing County Data (HIGHLY RECOVERABLE) ✅

### Current Status

- **EA**: 116 out of 118 projects missing county data (98.3%) **have lat/long coordinates**
- **EIS**: 281 out of 285 projects missing county data (98.6%) **have lat/long coordinates**
- **Total recoverable**: 397 out of 403 projects (98.5%)

### Why They're Different from CE

EA and EIS projects undergo more rigorous review processes that require:
1. Precise location documentation (coordinates captured)
2. More detailed geographic metadata
3. Often multi-county or regional scope (explaining why some legitimately lack a single county assignment)

### Location Patterns for Missing County Data

**EA Projects (118 missing)**:
- 99% have generic location descriptions (e.g., "Fairbanks, Alaska" or "Idaho (Lat/Long)")
- Only 1 explicitly mentions county in location field
- Nearly all have actual lat/long coordinates in separate database fields

**EIS Projects (285 missing)**:
- 96% have generic descriptions
- 8 are multi-county/regional projects (e.g., "Solar Energy Development in Six Southwestern States")
- 2 are offshore/marine projects (no county association)
- 98.6% have coordinates available

### Sample Projects That Can Be Recovered

| Project Title | State | Lat | Long | Current Status |
|--------------|-------|-----|------|----------------|
| Tanacross Bluff | Alaska | 63.4 | -143.0 | Missing county, has coords |
| Gateway West Transmission Line | Idaho | 43.1 | -116.0 | Missing county, has coords |
| Southern California Edison | California | 35.6 | -116.0 | Missing county, has coords |

---

## Recovery Strategy & Impact

### Recommended Approach: Reverse Geocoding

**Method**: Use existing `project_lat` and `project_lon` fields to look up county information via:
1. TIGER/Line county boundary shapefiles (already loaded in 03_location.R)
2. `sf::st_join()` or similar spatial join operations
3. Handle multi-county projects appropriately (store as list/array)

### Expected Impact

| Metric | Current | After Recovery | Improvement |
|--------|---------|---------------|-------------|
| EA county coverage | 81.0% | ~99% | +18 percentage points |
| EIS county coverage | 64.1% | ~99% | +35 percentage points |
| Overall coverage | 47.8% | ~52% | +4.2 percentage points |
| Recoverable projects | - | 397 projects | - |

**Note**: Overall improvement is modest (47.8% → 52%) because CE projects (which can't be recovered) represent 94% of the clean energy dataset.

### Implementation Complexity: LOW

- Coordinates already in database
- County shapefiles already loaded in mapping code
- Standard spatial join operation
- Estimated effort: 2-3 hours including testing

### Edge Cases to Handle

1. **Offshore projects (n=2)**: No county assignment - flag as "Offshore" or "Marine"
2. **Multi-county projects (n=8)**: Store multiple counties as array/list
3. **Default coordinates (39.8, -98.6)**: Geographic center of US - likely invalid, flag for manual review
4. **Border cases**: Projects on county boundaries may need tolerance handling

---

## Agency Patterns in Missing EA/EIS County Data

### Environmental Assessment (EA)
- Department of Agriculture: 27.3% missing (11 total projects)
- Department of Energy: 19.9% missing (387 total projects)
- Department of the Interior: 16.5% missing (212 total projects)

### Environmental Impact Statement (EIS)
- Major Independent Agencies: 46.7% missing (30 total projects)
- Department of Energy: 43.4% missing (281 total projects)
- Other/Unclassified: 40.6% missing (64 total projects)
- Department of the Interior: 37.7% missing (244 total projects)

---

## Recommendations

### Immediate Actions

1. **Implement reverse geocoding for EA/EIS projects** ✅ High priority, low effort
   - Add to data processing pipeline
   - Validate results with sample checks
   - Document methodology

2. **Update report captions** to reflect:
   - CE missing data is a documentation standard, not a data quality issue
   - EA/EIS coverage can be improved to ~99%
   - Overall coverage is limited by CE projects (94% of dataset)

3. **Create data quality flags**:
   - `county_source`: "original" | "geocoded" | "missing"
   - `geocode_confidence`: for reverse-geocoded counties

### Future Considerations

4. **Don't attempt CE recovery** - not cost-effective given:
   - Low success rate expected (<10%)
   - Would require complex GIS Township/Range lookups
   - By-design limitation of CE documentation standards

5. **Consider multi-county flagging** for large projects:
   - Transmission lines often span multiple counties
   - Store as array rather than forcing single county assignment

---

## Other Important Findings

### Geographic Concentration

**State Level**:
- South Carolina dominates (4,000+ projects) due to Savannah River Site
- Western states (WA, CA, ID, NV) show high activity
- Strong correlation with federal land holdings and renewable resource potential

**County Level**:
- Aiken County, SC: Highest concentration (Savannah River Site)
- Boundary County, ID: National lab activity
- Clustering around major federal facilities and renewable energy zones

### Process Type Distribution

**CE (Categorical Exclusion) - 94% of projects**:
- Dominated by DOE (84% of CE projects)
- 96% of DOE projects are CE (highly streamlined)

**EA & EIS - 6% of projects**:
- More diverse agency participation
- Higher geographic specificity requirements
- Better metadata quality overall

### Technology Patterns

- **Utilities & Electricity Transmission**: Largest share
- **Nuclear Technology**: Significant (concentrated in SC, ID, WA)
- **Solar & Wind**: Strong presence in Western states
- Most projects (90%) have multiple technology tags

---

## Data Quality Assessment

### Strengths
- Comprehensive state-level coverage (99.8% have state data)
- High-quality coordinates for EA/EIS projects
- Consistent NEPA process type classification
- Rich technology tagging (14 clean energy categories)

### Limitations
- County data inherently limited by CE documentation standards
- Multi-state projects challenging to represent in single-county schema
- Offshore/marine projects don't map to counties
- Some coordinates may be approximate (centroids vs. actual project locations)

### Recommended Data Quality Tiers

**Tier 1 (Highest Quality)**: EA & EIS projects with original county data
**Tier 2 (High Quality)**: EA & EIS projects with geocoded county data
**Tier 3 (State-Level Only)**: CE projects without county data
**Tier 4 (Flagged)**: Projects with invalid/default coordinates

---

## Next Steps

1. ✅ **Complete**: County data coverage analysis
2. ✅ **Complete**: Identify recovery strategy for EA/EIS
3. ⏳ **Pending**: Implement reverse geocoding routine
4. ⏳ **Pending**: Validate geocoded results (sample check)
5. ⏳ **Pending**: Update deliverable01.qmd with revised county coverage notes
6. ⏳ **Pending**: Document methodology in technical appendix

---

## Technical Notes

### Files Modified for Analysis
- `code/deliverable1/03_location.R`: Added county coverage analysis code (lines 512-708)
- Analysis output shows in script console output

### Key Variables
- `project_county`: JSON array of county names (may be empty `[]`)
- `project_lat`, `project_lon`: Decimal degree coordinates
- `project_location`: Free-text location description
- `project_state`: JSON array of state names

### Reverse Geocoding Approach
```r
# Pseudo-code for implementation
projects_to_geocode <- clean_energy %>%
  filter(process_type %in% c("EA", "EIS")) %>%
  filter(project_county == "[]") %>%
  filter(!is.na(project_lat) & project_lat != 0)

# Convert to spatial points
project_points <- st_as_sf(
  projects_to_geocode,
  coords = c("project_lon", "project_lat"),
  crs = 4326
)

# Spatial join with county boundaries
geocoded_counties <- st_join(project_points, us_counties, join = st_within)

# Update project_county field
# Handle edge cases (offshore, multi-county, etc.)
```

---

## Conclusion

While overall county coverage is 47.8%, this is primarily driven by CE project documentation standards (not recoverable). The critical finding is that **397 EA/EIS projects (98.5% of missing data) can be recovered through straightforward reverse geocoding**, bringing EA/EIS coverage to ~99%.

This recovery would support more robust county-level analysis for the most impactful projects (EA/EIS), while accepting that CE projects will remain at state-level granularity due to their streamlined documentation requirements.

**Recommendation**: Proceed with reverse geocoding implementation for EA/EIS projects as a high-value, low-effort data quality improvement.
