# --------------------------
# LOCATION PARSING
# --------------------------
# Functions to extract state and county from project_location field

import re
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.config import US_STATES, STATE_ABBREVIATIONS, STATE_ABBREV_TO_NAME


def extract_lat_long(location_str):
    """
    Extract latitude and longitude from location string.

    Args:
        location_str: String containing location info

    Returns:
        tuple: (lat, lon) or (None, None) if not found
    """
    if not location_str or not isinstance(location_str, str):
        return None, None

    # Pattern: Lat/Lon: 67.2500, -150.1750
    pattern = r"Lat/Lon:\s*([-]?\d+\.?\d*),\s*([-]?\d+\.?\d*)"
    match = re.search(pattern, location_str)

    if match:
        try:
            lat = float(match.group(1))
            lon = float(match.group(2))
            return lat, lon
        except ValueError:
            pass

    return None, None


def extract_states_from_text(location_str):
    """
    Extract US states from location text using regex patterns.

    Handles formats like:
    - "Kern County, CA"
    - "Alaska"
    - "Arizona, California, Colorado"
    - "New Mexico (Lat/Lon: 35.3, -107.0)"

    Args:
        location_str: String containing location info

    Returns:
        list: List of state names (full names, title case)
    """
    if not location_str or not isinstance(location_str, str):
        return []

    states_found = set()
    text_lower = location_str.lower()

    # First, look for state abbreviations (2 letters)
    # Pattern: ", XX" or ", XX " or "County, XX" etc.
    abbrev_pattern = r"(?:,\s*|\s)([A-Z]{2})(?:\s|$|[;,\)])"
    for match in re.finditer(abbrev_pattern, location_str):
        abbrev = match.group(1)
        if abbrev in STATE_ABBREVIATIONS:
            states_found.add(STATE_ABBREV_TO_NAME[abbrev])

    # Then, look for full state names
    for state_name, abbrev in US_STATES.items():
        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(state_name) + r'\b'
        if re.search(pattern, text_lower):
            states_found.add(state_name.title())

    return sorted(list(states_found))


def extract_counties_from_text(location_str):
    """
    Extract county names from location text.

    Handles formats like:
    - "Kern County, CA"
    - "La Paz County, Arizona"
    - "Maricopa County, Pima County, Pinal County, AZ"

    Args:
        location_str: String containing location info

    Returns:
        list: List of county names (without "County" suffix)
    """
    if not location_str or not isinstance(location_str, str):
        return []

    counties_found = set()

    # Pattern: Word(s) + "County"
    # Handles: "Kern County", "La Paz County", "San Juan County"
    pattern = r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+County"
    for match in re.finditer(pattern, location_str):
        county = match.group(1).strip()
        counties_found.add(county)

    return sorted(list(counties_found))


def parse_location_list(location_value):
    """
    Parse location from a list or single value.

    Args:
        location_value: list, array, string, or None

    Returns:
        str: Combined location string for parsing
    """
    if location_value is None:
        return ""
    if isinstance(location_value, float) and np.isnan(location_value):
        return ""
    if isinstance(location_value, str):
        return location_value
    if isinstance(location_value, (list, np.ndarray)):
        return " ; ".join(str(v) for v in location_value if v)
    return str(location_value)


def parse_project_location(location_value):
    """
    Parse a project_location value and extract structured location data.

    Args:
        location_value: The project_location field value (list, string, etc.)

    Returns:
        dict: {
            'project_state': list of states,
            'project_county': list of counties,
            'project_lat': float or None,
            'project_lon': float or None,
            'project_location_needs_geocoding': bool
        }
    """
    location_str = parse_location_list(location_value)

    states = extract_states_from_text(location_str)
    counties = extract_counties_from_text(location_str)
    lat, lon = extract_lat_long(location_str)

    # Flag for geocoding if we have coords but no states
    needs_geocoding = (lat is not None and lon is not None and len(states) == 0)

    return {
        'project_state': states,
        'project_county': counties,
        'project_lat': lat,
        'project_lon': lon,
        'project_location_needs_geocoding': needs_geocoding
    }


def add_location_columns(df):
    """
    Add location columns to a projects DataFrame.

    Adds columns:
    - project_state: List of states
    - project_county: List of counties
    - project_lat: Latitude (if available)
    - project_lon: Longitude (if available)
    - project_location_needs_geocoding: Flag for entries needing geocoding

    Args:
        df: pandas DataFrame with 'project_location' column

    Returns:
        pandas DataFrame with new columns added
    """
    df = df.copy()

    # Parse all locations
    parsed = df["project_location"].apply(parse_project_location)

    df["project_state"] = parsed.apply(lambda x: x['project_state'])
    df["project_county"] = parsed.apply(lambda x: x['project_county'])
    df["project_lat"] = parsed.apply(lambda x: x['project_lat'])
    df["project_lon"] = parsed.apply(lambda x: x['project_lon'])
    df["project_location_needs_geocoding"] = parsed.apply(
        lambda x: x['project_location_needs_geocoding']
    )

    return df


# --------------------------
# GEOCODING (optional, for fallback)
# --------------------------

def geocode_coordinates(lat, lon, geocoder=None):
    """
    Reverse geocode lat/lon to get state and county.

    Note: This requires geopy library and makes API calls.
    Use sparingly and with rate limiting.

    Args:
        lat: Latitude
        lon: Longitude
        geocoder: Optional pre-configured geocoder instance

    Returns:
        dict: {'state': str or None, 'county': str or None}
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut

        if geocoder is None:
            geocoder = Nominatim(user_agent="nepa_project_analysis")

        location = geocoder.reverse(f"{lat}, {lon}", exactly_one=True, language="en")

        if location and location.raw.get('address'):
            addr = location.raw['address']
            state = addr.get('state')
            county = addr.get('county', '').replace(' County', '')
            return {'state': state, 'county': county if county else None}

    except ImportError:
        print("Warning: geopy not installed. Run: pip install geopy")
    except GeocoderTimedOut:
        print(f"Geocoding timed out for {lat}, {lon}")
    except Exception as e:
        print(f"Geocoding error for {lat}, {lon}: {e}")

    return {'state': None, 'county': None}


# --------------------------
# TESTING
# --------------------------

if __name__ == "__main__":
    test_locations = [
        "[Coldfoot, Yukon-Koyukuk Census Area, AK (Lat/Lon: 67.2500, -150.1750)]",
        "[Kern Front Oil Field, Kern County, CA (Lat/Lon: 35.48, -119.05)]",
        "[New Mexico (Lat/Lon: 35.3, -107.0)]",
        "[Arizona, California, Colorado, Nevada, New Mexico]",
        "[Maricopa County, Pima County, Pinal County, AZ]",
        "[Grand County, UT; Uintah County, UT (Lat/Lon: 39.0, -109.5)]",
        "[]",
        None,
    ]

    print("Testing location parsing...")
    for loc in test_locations:
        result = parse_project_location(loc)
        print(f"\nInput: {loc}")
        print(f"  States: {result['project_state']}")
        print(f"  Counties: {result['project_county']}")
        print(f"  Lat/Lon: {result['project_lat']}, {result['project_lon']}")
        print(f"  Needs geocoding: {result['project_location_needs_geocoding']}")

    print("\nDone.")
