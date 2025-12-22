
# load
from datasets import load_dataset
ds_ce = load_dataset(
    "PNNL/NEPATEC2.0",
    data_files="CE/*/*.jsonl",
    split="train"
)

# build project 
projects = []

for item in ds_ce:
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

import pandas as pd
projects_df = pd.DataFrame(projects)

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
projects_df.head()

import numpy as np
# Remove truncation for numpy arrays
np.set_printoptions(threshold=np.inf)

# Now run your command
projects_df["project_sector"].unique()
projects_df["project_location"].head(100)
projects_df.shape

projects_df.head()
process_df.head()

process_df



# Flatten all the lists to see every unique value
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

all_types


import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain

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

# processes
process_rows = []

for item in ds_ce:
    process = item["process"]
    process_rows.append({
        "project_id": item["project"]["project_ID"],
        "process_family": process.get("process_family", {}).get("value", ""),
        "process_type": process.get("process_type", {}).get("value", ""),
        "lead_agency": process.get("lead_agency", {}).get("value", "")
    })

process_df = pd.DataFrame(process_rows)
process_df[["process_type"]].unique()


# documents
documents = []

for item in ds_ce:
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

documents_df = pd.DataFrame(documents)
documents_df.head()

# pages
pages = []
for item in ds_ce:
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

pages_df = pd.DataFrame(pages)
pages_df.head()
pages_df.page_text[0]
pages_df.page_text[99]
pages_df.head()
