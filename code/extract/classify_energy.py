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


def add_energy_columns(df):
    """
    Add all energy classification columns to a projects DataFrame.

    Adds columns:
    - project_energy_type: "Clean", "Fossil", or "Other"
    - project_energy_type_questions: Boolean flag for manual review
    - project_type_count: Total count of project_type values
    - project_type_count_clean: Count of clean energy tags
    - project_type_count_fossil: Count of fossil fuel tags

    Args:
        df: pandas DataFrame with 'project_type' column

    Returns:
        pandas DataFrame with new columns added
    """
    df = df.copy()

    # Classify energy type
    df["project_energy_type"] = df["project_type"].apply(classify_energy_type)

    # Flag for manual review
    df["project_energy_type_questions"] = df["project_type"].apply(flag_energy_questions)

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
    # Test cases
    test_cases = [
        # (input, expected_type, expected_flag)
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

    print("Testing energy classification...")
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

    print("\nDone.")
