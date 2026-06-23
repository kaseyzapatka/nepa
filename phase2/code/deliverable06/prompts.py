"""D6 — shared LLM prompt definitions (un-numbered helper).

Lives here (not in a numbered step) so both the extraction step
(03_extract_candidate_facts.py) and the standalone benchmark (benchmark_models.py)
import the *identical* production prompt. Numbered scripts can't be imported
(module names can't start with a digit), so shared functions live in helpers.
"""

from __future__ import annotations


def build_facts_prompt(packet_text: str, category: str) -> str:
    """The production fact-extraction prompt. Shared with the model benchmark so it
    measures the exact prompt used in production."""
    return (
        "Extract CE-relevant facts from this NEPA FONSI/EA evidence as strict JSON with keys: "
        "action_definition, max_acres, max_miles, max_megawatts, within_existing_row, "
        "no_new_access_road, previously_disturbed_land, mitigation_dependence "
        "(one of none/design_feature_only/case_specific_dependent), mitigation_summary, "
        "extraordinary_circumstances. "
        f"Category hint: {category}. Use null when unknown.\n\nEVIDENCE:\n" + packet_text[:6000]
    )
