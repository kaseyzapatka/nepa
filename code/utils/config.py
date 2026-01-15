# --------------------------
# ENERGY TYPE CLASSIFICATION CONSTANTS
# --------------------------
# Based on ea_project_type.txt classification rules

# Clean Energy project types (case-insensitive matching)
CLEAN_ENERGY_TYPES = {
    "carbon capture and sequestration",
    "conventional energy production - nuclear",
    "conventional energy production - other",  # includes hydroelectric power
    "renewable energy production - biomass",
    "renewable energy production - energy storage",
    "renewable energy production - geothermal",
    "renewable energy production - hydrokinetic",
    "renewable energy production - hydropower",
    "renewable energy production - other",
    "renewable energy production - solar",
    "renewable energy production - wind, offshore",
    "renewable energy production - wind, onshore",
    "nuclear technology",
    "electricity transmission",
    "utilities (electricity, gas, telecommunications)",
}

# Fossil Fuel project types (case-insensitive matching)
FOSSIL_ENERGY_TYPES = {
    "conventional energy production - coal",
    "conventional energy production - land-based oil & gas",
    "conventional energy production - rural energy",
    "conventional energy production - offshore oil and gas",
    "pipelines",  # includes natural gas and oil (may also include carbon/hydrogen)
}

# Non-energy tags that when combined with Utilities should flag for review
AMBIGUOUS_WITH_UTILITIES = {
    "broadband",
}

# US States for location parsing
US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC",
}

# State abbreviations to full names (reverse lookup)
STATE_ABBREV_TO_NAME = {v: k.title() for k, v in US_STATES.items()}

# Common state abbreviations found in location strings
STATE_ABBREVIATIONS = set(US_STATES.values())

# Generation capacity units and their standard forms
GENCAP_UNITS = {
    "mw": "MW",
    "megawatt": "MW",
    "megawatts": "MW",
    "gw": "GW",
    "gigawatt": "GW",
    "gigawatts": "GW",
    "kw": "kW",
    "kilowatt": "kW",
    "kilowatts": "kW",
    "mwh": "MWh",
    "kwh": "kWh",
    "gwh": "GWh",
}
