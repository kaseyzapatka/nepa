# --------------------------
# ENERGY TYPE CLASSIFICATION
# --------------------------
# Functions to classify projects as Clean, Fossil, or Other energy types

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import CLEAN_ENERGY_TYPES, FOSSIL_ENERGY_TYPES, AMBIGUOUS_WITH_UTILITIES


def normalize_type(t):
    """Normalize a project type string for comparison."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return None
    return str(t).strip().lower()


def get_type_set(project_types):
    """
    Convert project_type value to a set of normalized types.
    Handles: list, numpy array, single string, None, NaN
    """
    if project_types is None:
        return set()
    if isinstance(project_types, float) and np.isnan(project_types):
        return set()
    if isinstance(project_types, str):
        return {normalize_type(project_types)} - {None}
    if isinstance(project_types, (list, np.ndarray)):
        return {normalize_type(t) for t in project_types} - {None}
    return set()


def classify_energy_type(project_types):
    """
    Classify a project's energy type based on its project_type tags.

    Rules:
    1. If ANY fossil fuel tag exists -> "Fossil" (fossil takes precedence)
    2. Else if ANY clean energy tag exists -> "Clean"
    3. Else -> "Other"

    Args:
        project_types: list, array, or string of project type values

    Returns:
        str: "Clean", "Fossil", or "Other"
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return "Other"

    # Check for fossil fuel tags (takes precedence)
    has_fossil = bool(types_set & FOSSIL_ENERGY_TYPES)
    if has_fossil:
        return "Fossil"

    # Check for clean energy tags
    has_clean = bool(types_set & CLEAN_ENERGY_TYPES)
    if has_clean:
        return "Clean"

    return "Other"


def flag_energy_questions(project_types):
    """
    Flag projects that need manual review for energy classification.

    Flags True if:
    - Utilities tag present WITH Broadband AND no other energy tags
    - Both clean AND fossil tags present (for manual verification)

    Args:
        project_types: list, array, or string of project type values

    Returns:
        bool: True if project needs manual review
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return False

    # Check for both clean and fossil (conflict)
    has_fossil = bool(types_set & FOSSIL_ENERGY_TYPES)
    has_clean = bool(types_set & CLEAN_ENERGY_TYPES)

    if has_fossil and has_clean:
        return True

    # Check for Utilities + Broadband only (ambiguous)
    utilities_tag = "utilities (electricity, gas, telecommunications)"
    has_utilities = utilities_tag in types_set
    has_ambiguous = bool(types_set & AMBIGUOUS_WITH_UTILITIES)

    if has_utilities and has_ambiguous:
        # Check if there are any other tags besides utilities and ambiguous ones
        other_tags = types_set - {utilities_tag} - AMBIGUOUS_WITH_UTILITIES
        if not other_tags:
            return True

    return False


def count_energy_tags(project_types):
    """
    Count different types of energy tags in a project.

    Args:
        project_types: list, array, or string of project type values

    Returns:
        tuple: (total_count, clean_count, fossil_count)
    """
    types_set = get_type_set(project_types)

    total_count = len(types_set)
    clean_count = len(types_set & CLEAN_ENERGY_TYPES)
    fossil_count = len(types_set & FOSSIL_ENERGY_TYPES)

    return total_count, clean_count, fossil_count


def is_utilities_broadband_only(project_types):
    """
    Check if project has ONLY Utilities + Broadband tags (no other tags).

    These are likely telecommunications projects, not clean energy.

    Args:
        project_types: list, array, or string of project type values

    Returns:
        bool: True if project is Utilities + Broadband only
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return False

    utilities_tag = "utilities (electricity, gas, telecommunications)"
    broadband_tag = "broadband"

    # Must have utilities
    if utilities_tag not in types_set:
        return False

    # Must have broadband
    if broadband_tag not in types_set:
        return False

    # Must have ONLY these two tags
    other_tags = types_set - {utilities_tag, broadband_tag}
    return len(other_tags) == 0


def is_nuclear_tech_only(project_types):
    """
    Check if project has Nuclear Technology but NOT Conventional Energy Production - Nuclear.

    These are likely waste management or R&D projects, not power generation.

    Args:
        project_types: list, array, or string of project type values

    Returns:
        bool: True if project has Nuclear Technology without nuclear power production
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return False

    nuclear_tech_tag = "nuclear technology"
    nuclear_production_tag = "conventional energy production - nuclear"

    # Has nuclear technology tag
    has_nuclear_tech = nuclear_tech_tag in types_set

    # Does NOT have nuclear production tag
    has_nuclear_production = nuclear_production_tag in types_set

    return has_nuclear_tech and not has_nuclear_production


def is_utilities_to_filter_out(project_types):
    """
    Identify projects that have ONLY Utilities + a non-energy infrastructure tag.

    This filter flags projects where "Utilities (electricity, gas, telecommunications)"
    co-occurs with ONLY one of the following tags (and no other tags):
    - Broadband: likely telecommunications infrastructure, not energy
    - Waste Management: likely utility-adjacent but not energy production
    - Land Development (Housing, Other, Urban): likely development projects with
      utility connections, not energy projects

    Projects with additional tags beyond Utilities + one of these are NOT flagged,
    as the additional tags suggest the project has broader scope.

    Use this filter to exclude projects that are likely not primarily about clean energy.

    Args:
        project_types: list, array, or string of project type values

    Returns:
        bool: True if project has ONLY Utilities + one non-energy tag (no other tags)
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return False

    utilities_tag = "utilities (electricity, gas, telecommunications)"

    # Must have utilities tag
    if utilities_tag not in types_set:
        return False

    # Tags that when combined with Utilities (and nothing else) suggest non-energy projects
    non_energy_cooccur_tags = {
        "broadband",
        "waste management",
        "land development - housing",
        "land development - other",
        "land development - urban",
    }

    # Check if any of the non-energy co-occurring tags are present
    cooccur_tags_present = types_set & non_energy_cooccur_tags
    if not cooccur_tags_present:
        return False

    # Check if there are any OTHER tags beyond utilities and the non-energy tags
    other_tags = types_set - {utilities_tag} - non_energy_cooccur_tags

    # Only flag if there are NO other tags (i.e., ONLY utilities + non-energy tag(s))
    return len(other_tags) == 0


def classify_energy_type_strict(project_types):
    """
    Classify a project's energy type using STRICT criteria.

    Excludes borderline cases:
    - Utilities + Broadband only projects (likely telecom)
    - Nuclear Technology without Nuclear Production (likely waste/R&D)

    Rules:
    1. If Utilities + Broadband ONLY -> "Other" (excluded)
    2. If Nuclear Technology without Nuclear Production -> "Other" (excluded)
    3. If ANY fossil fuel tag exists -> "Fossil" (fossil takes precedence)
    4. Else if ANY clean energy tag exists -> "Clean"
    5. Else -> "Other"

    Args:
        project_types: list, array, or string of project type values

    Returns:
        str: "Clean", "Fossil", or "Other"
    """
    types_set = get_type_set(project_types)

    if not types_set:
        return "Other"

    # Exclusion 1: Utilities + Broadband only -> Other
    if is_utilities_broadband_only(project_types):
        return "Other"

    # Exclusion 2: Nuclear Technology only (no nuclear production) -> Other
    if is_nuclear_tech_only(project_types):
        return "Other"

    # Standard classification logic
    # Check for fossil fuel tags (takes precedence)
    has_fossil = bool(types_set & FOSSIL_ENERGY_TYPES)
    if has_fossil:
        return "Fossil"

    # Check for clean energy tags
    has_clean = bool(types_set & CLEAN_ENERGY_TYPES)
    if has_clean:
        return "Clean"

    return "Other"


def add_energy_columns(df):
    """
    Add all energy classification columns to a projects DataFrame.

    Adds columns:
    - project_energy_type: "Clean", "Fossil", or "Other" (broad definition)
    - project_energy_type_strict: "Clean", "Fossil", or "Other" (excludes borderline cases)
    - project_energy_type_questions: Boolean flag for manual review
    - project_is_utilities_broadband_only: Boolean flag for Utilities+Broadband only projects
    - project_is_nuclear_tech_only: Boolean flag for Nuclear Tech without production
    - project_utilities_to_filter_out: Boolean flag for projects with ONLY Utilities +
      Broadband/Waste Management/Land Development (no other tags) - likely not clean energy
    - project_type_count: Total count of project_type values
    - project_type_count_clean: Count of clean energy tags
    - project_type_count_fossil: Count of fossil fuel tags

    Args:
        df: pandas DataFrame with 'project_type' column

    Returns:
        pandas DataFrame with new columns added
    """
    df = df.copy()

    # Classify energy type (broad)
    df["project_energy_type"] = df["project_type"].apply(classify_energy_type)

    # Classify energy type (strict - excludes borderline cases)
    df["project_energy_type_strict"] = df["project_type"].apply(classify_energy_type_strict)

    # Flag for manual review
    df["project_energy_type_questions"] = df["project_type"].apply(flag_energy_questions)

    # Specific exclusion flags
    df["project_is_utilities_broadband_only"] = df["project_type"].apply(is_utilities_broadband_only)
    df["project_is_nuclear_tech_only"] = df["project_type"].apply(is_nuclear_tech_only)

    # Utilities filter: flags projects with ONLY Utilities + Broadband/Waste Management/Land Development
    # These are likely not primarily clean energy projects
    df["project_utilities_to_filter_out"] = df["project_type"].apply(is_utilities_to_filter_out)

    # Count tags
    counts = df["project_type"].apply(count_energy_tags)
    df["project_type_count"] = counts.apply(lambda x: x[0])
    df["project_type_count_clean"] = counts.apply(lambda x: x[1])
    df["project_type_count_fossil"] = counts.apply(lambda x: x[2])

    return df


# --------------------------
# TESTING
# --------------------------

if __name__ == "__main__":
    # Test cases for broad classification
    # (input, expected_type, expected_flag)
    test_cases = [
        (["Utilities (electricity, gas, telecommunications)"], "Clean", False),
        (["Utilities (electricity, gas, telecommunications)", "Broadband"], "Clean", True),
        (["Renewable Energy Production - Solar"], "Clean", False),
        (["Conventional Energy Production - Land-based Oil & Gas"], "Fossil", False),
        (["Utilities (electricity, gas, telecommunications)",
          "Conventional Energy Production - Land-based Oil & Gas"], "Fossil", True),
        (["Broadband"], "Other", False),
        (["Agriculture"], "Other", False),
        (None, "Other", False),
        ([], "Other", False),
    ]

    print("Testing BROAD energy classification...")
    for types, expected_type, expected_flag in test_cases:
        result_type = classify_energy_type(types)
        result_flag = flag_energy_questions(types)

        type_ok = result_type == expected_type
        flag_ok = result_flag == expected_flag

        status = "PASS" if (type_ok and flag_ok) else "FAIL"
        print(f"{status}: {types}")
        if not type_ok:
            print(f"  Expected type: {expected_type}, Got: {result_type}")
        if not flag_ok:
            print(f"  Expected flag: {expected_flag}, Got: {result_flag}")

    # Test cases for strict classification
    # (input, expected_strict_type)
    strict_test_cases = [
        # Utilities + Broadband only -> Other (excluded)
        (["Utilities (electricity, gas, telecommunications)", "Broadband"], "Other"),
        # Utilities alone -> Clean (not excluded)
        (["Utilities (electricity, gas, telecommunications)"], "Clean"),
        # Utilities + Broadband + Solar -> Clean (has other tag)
        (["Utilities (electricity, gas, telecommunications)", "Broadband",
          "Renewable Energy Production - Solar"], "Clean"),
        # Nuclear Technology alone -> Other (excluded)
        (["Nuclear Technology"], "Other"),
        # Nuclear Technology + Nuclear Production -> Clean (not excluded)
        (["Nuclear Technology", "Conventional Energy Production - Nuclear"], "Clean"),
        # Regular clean energy -> Clean
        (["Renewable Energy Production - Solar"], "Clean"),
        # Fossil -> Fossil
        (["Conventional Energy Production - Land-based Oil & Gas"], "Fossil"),
    ]

    print("\nTesting STRICT energy classification...")
    for types, expected_strict in strict_test_cases:
        result_strict = classify_energy_type_strict(types)
        status = "PASS" if result_strict == expected_strict else "FAIL"
        print(f"{status}: {types}")
        if result_strict != expected_strict:
            print(f"  Expected: {expected_strict}, Got: {result_strict}")

    # Test cases for utilities to filter out
    # (input, expected_filter)
    utilities_filter_cases = [
        # Utilities + Broadband ONLY -> True (filter out)
        (["Utilities (electricity, gas, telecommunications)", "Broadband"], True),
        # Utilities + Waste Management ONLY -> True (filter out)
        (["Utilities (electricity, gas, telecommunications)", "Waste Management"], True),
        # Utilities + Land Development - Housing ONLY -> True (filter out)
        (["Utilities (electricity, gas, telecommunications)", "Land Development - Housing"], True),
        # Utilities + Land Development - Other ONLY -> True (filter out)
        (["Utilities (electricity, gas, telecommunications)", "Land Development - Other"], True),
        # Utilities + Land Development - Urban ONLY -> True (filter out)
        (["Utilities (electricity, gas, telecommunications)", "Land Development - Urban"], True),
        # Utilities + Broadband + Solar -> False (has other tag, keep it)
        (["Utilities (electricity, gas, telecommunications)", "Broadband",
          "Renewable Energy Production - Solar"], False),
        # Utilities + Waste Management + Agriculture -> False (has other tag, keep it)
        (["Utilities (electricity, gas, telecommunications)", "Waste Management",
          "Agriculture"], False),
        # Utilities alone -> False (no co-occurring non-energy tag)
        (["Utilities (electricity, gas, telecommunications)"], False),
        # Utilities + Solar -> False (Solar is energy, not a flagged tag)
        (["Utilities (electricity, gas, telecommunications)",
          "Renewable Energy Production - Solar"], False),
        # Broadband alone (no utilities) -> False
        (["Broadband"], False),
        # Waste Management alone (no utilities) -> False
        (["Waste Management"], False),
        # Empty -> False
        ([], False),
    ]

    print("\nTesting UTILITIES TO FILTER OUT...")
    for types, expected_filter in utilities_filter_cases:
        result_filter = is_utilities_to_filter_out(types)
        status = "PASS" if result_filter == expected_filter else "FAIL"
        print(f"{status}: {types}")
        if result_filter != expected_filter:
            print(f"  Expected: {expected_filter}, Got: {result_filter}")

    print("\nDone.")
