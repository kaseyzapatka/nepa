

# --------------------------
# AUTHENTICATION
# --------------------------
#hf auth login - NEED TO RUN ON TERMINAL AND DROP IN HUGGINGFACE KEY

# --------------------------
# LIBRARIES
# --------------------------
from datasets import load_dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
from itertools import chain
import re
from pathlib import Path

# set panda options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
np.set_printoptions(threshold=np.inf)

# --------------------------
# FILE PATHS
# --------------------------

#
# Set folder file paths
# ----------------------------------------
PROCESSED_DIR = Path("data/processed")
RAW_DIR = Path("data/raw")

#
# Identify specific files
# ----------------------------------------
PROJECTS_FILE = PROCESSED_DIR / "projects.parquet"
PROCESSES_FILE = PROCESSED_DIR / "processes.parquet"
DOCUMENTS_FILE = PROCESSED_DIR / "documents.parquet"
PAGES_FILE = PROCESSED_DIR / "pages.parquet"



# --------------------------
# LOAD DATA
# --------------------------
# Data are accessible here: https://huggingface.co/datasets/PNNL/NEPATEC2.0
# See the accompanying paper for more details: https://www.pnnl.gov/sites/default/files/media/file/PNNL_PermitAI_NEPATECv2_Public_Release_20_08_25.pdf

# load database from hugging face
ds = load_dataset(
    "PNNL/NEPATEC2.0",
    data_files="EIS/*/*.jsonl", # Environmental Impact Statement (EIS)
    split="train"
)

# --------------------------
# CREATE DATABASES
# --------------------------

#
# Projects
# ----------------------------------------
# create empty projects list 
projects = []

# loop over specific items in to create list
for item in ds:
    p = item["project"]
    projects.append({
        "project_id": p.get("project_ID"),
        "project_title": p.get("project_title", {}).get("value", ""),
        "project_sector": p.get("project_sector", {}).get("value", ""),
        "project_type": p.get("project_type", {}).get("value", ""),
        "project_description": p.get("project_description", {}).get("value", ""),
        "project_sponsor": p.get("project_sponsor", {}).get("value", ""),
        "project_location": p.get("location", {}).get("value", "")
    })

# convert to pandas dataframe
projects_df = pd.DataFrame(projects)
projects_df.head()

# save
projects_df.to_parquet(PROJECTS_FILE)


#
# Processes
# ----------------------------------------
# create empty projects list 
process_rows = []

# loop over specific items in to create list
for item in ds:
    process = item["process"]
    process_rows.append({
        "project_id": item["project"]["project_ID"],
        "process_family": process.get("process_family", {}).get("value", ""),
        "process_type": process.get("process_type", {}).get("value", ""),
        "lead_agency": process.get("lead_agency", {}).get("value", "")
    })

# convert to pandas dataframe
process_df = pd.DataFrame(process_rows)
process_df

# save
process_df.to_parquet(PROCESSES_FILE)

#
# Documents
# ----------------------------------------
documents = []

for item in ds:
    pid = item["project"]["project_ID"]
    for doc in item["documents"]:
        meta = doc.get("metadata", {})
        doc_meta = meta.get("document_metadata", {})
        file_meta = meta.get("file_metadata", {})

        documents.append({
            "project_id": pid,
            "document_id": doc_meta.get("document_ID", {}).get("value", ""),
            "document_type": doc_meta.get("document_type", {}).get("value", ""),
            "document_title": doc_meta.get("document_title", {}).get("value", ""),
            "prepared_by": doc_meta.get("prepared_by", {}).get("value", ""),
            "ce_category": doc_meta.get("ce_category", {}).get("value", ""),
            "file_id": file_meta.get("file_ID", {}).get("value", ""),
            "file_name": file_meta.get("file_name", {}).get("value", ""),
            "total_pages": file_meta.get("total_pages", {}).get("value", ""),
            "main_document": file_meta.get("main_document", {}).get("value", "")
        })

# convert to pandas dataframe
documents_df = pd.DataFrame(documents)
documents_df.head()

# save
documents_df.to_parquet(DOCUMENTS_FILE)

#
# Pages
# ----------------------------------------
# create empty projects list 
pages = []

# loop over specific items in to create list
for item in ds:
    pid = item["project"]["project_ID"]
    for doc in item["documents"]:
        doc_meta = doc["metadata"]["document_metadata"]
        doc_id = doc_meta["document_ID"]["value"]
        for pg in doc.get("pages", []):
            pages.append({
                "document_id": doc_id,
                "page_number": pg.get("page number"),
                "page_text": pg.get("page text", "")
            })

# convert to pandas dataframe
pages_df = pd.DataFrame(pages)
pages_df.head()

# save
pages_df.to_parquet(PAGES_FILE)



# --------------------------
# EXPLORATORY ANALYSES VIZ
# --------------------------

# 
# Check out unique values of a few columns
# --------------------------------------------------
from itertools import chain

all_sectors = list(chain.from_iterable(projects_df["project_sector"]))
unique_sectors = sorted(set(all_sectors))
print(unique_sectors)

all_locations = list(chain.from_iterable(projects_df["project_location"]))
unique_locations = sorted(set(all_locations))
print(unique_locations)

all_types = list(chain.from_iterable(projects_df["project_type"]))
unique_types = sorted(set(all_types))
print(unique_types)



# 
# Bat chart of types
# --------------------------------------------------
# replicate the EIS barchart on pg 23 of the report

# Flatten the lists and count occurrences
all_types = list(chain.from_iterable(projects_df["project_type"]))
type_counts = Counter(all_types)

# Get the data for plotting
types = list(type_counts.keys())
counts = list(type_counts.values())

# Show only top 20 most common
top_n = 20
top_types = type_counts.most_common(top_n)
types = [t[0] for t in top_types]
counts = [t[1] for t in top_types]

plt.figure(figsize=(12, 8))
plt.barh(types, counts)
plt.xlabel('Count')
plt.ylabel('Project Type')
plt.title(f'Top {top_n} Most Common Project Types')
plt.tight_layout()
plt.show()


# 
# Map of all locations
# --------------------------------------------------
# replicate the EIS map on pg 26 of the report
# Parse locations
def parse_location(loc_string):
    match = re.search(r'(.*?)\s*\(Lat/Lon:\s*([-\d.]+),\s*([-\d.]+)\)', loc_string)
    if match:
        name = match.group(1).strip()
        lat = float(match.group(2))
        lon = float(match.group(3))
        return {'name': name, 'lat': lat, 'lon': lon}
    return None

parsed_locations = [parse_location(loc) for loc in unique_locations if parse_location(loc)]
locations_df = pd.DataFrame(parsed_locations)
locations_df


# Create interactive map
fig = px.scatter_geo(locations_df,
                     lat='lat',
                     lon='lon',
                     hover_name='name',
                     scope='usa',
                     title='NEPA Project Locations')

fig.update_geos(
    showland=True,
    landcolor='lightgray',
    showlakes=True,
    lakecolor='lightblue',
    showcountries=True,
    showcoastlines=True
)

fig.show()