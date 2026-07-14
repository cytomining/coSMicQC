#!/usr/bin/env python
# coding: utf-8

# # Generate manifest file with paths to SQLite files for CytoTable processing
# 
# A manifest file is a simple CSV that tracks the plate names in the dataset.
# It contains the correct paths to their respective SQLite file.
# 
# Code derived from the `JUMP-single-cell` repository, which can be found [here](https://github.com/WayScience/JUMP-single-cell/blob/main/0.download_data/0.generate_jump_dataset_manifest.ipynb).

# In[1]:


import pathlib

import boto3
import pandas as pd
import s3fs
from botocore import UNSIGNED
from botocore.config import Config


# In[2]:


filename = "barcode_platemap.csv"
plate_name_path = f"./{filename}"
plate_namedf = pd.read_csv(plate_name_path)


# In[3]:


output_path = pathlib.Path("manifest")
output_path.mkdir(parents=True, exist_ok=True)


# In[4]:


dataset_name = "cpg0004-lincs"
source = "broad"
batch = "2016_04_01_a549_48hr_batch1"
data_locations = (
    f"s3://cellpainting-gallery/{dataset_name}/{source}/workspace/backend/{batch}"
)

# Initialize S3 filesystem (anonymous access)
fs = s3fs.S3FileSystem(anon=True)

# Use the directory names from the repo to specify the plate names
object_names = [item["Assay_Plate_Barcode"] for _, item in plate_namedf.iterrows()]

# Construct and filter only existing paths
missing_plates = []
existing_entries = []
for obj_name in object_names:
    sqlite_path = f"{data_locations}/{obj_name}/{obj_name}.sqlite"
    if fs.exists(sqlite_path):
        existing_entries.append({"plate": obj_name, "sqlite_file": sqlite_path})
    else:
        missing_plates.append(obj_name)

if missing_plates:
    print("\nSQLite files NOT found for the following plates:")
    for plate in missing_plates:
        print(f"- {plate}")
else:
    print("✅ All SQLite files found.")

# Print the number of plates found to have SQLite files
print(f"\nTotal plates with SQLite files: {len(existing_entries)}")

# Create the manifest only with valid paths
manifest_df = pd.DataFrame(existing_entries)
manifest_df.to_csv(output_path / "lincs_cp_output_location_manifest.csv", index=False)


# ## Download platemap files for downstream use

# In[ ]:


bucket = "cellpainting-gallery"
prefix = (
    "cpg0004-lincs/broad/workspace/metadata/platemaps/"
    "2016_04_01_a549_48hr_batch1/platemap/"
)
local_dir = pathlib.Path("platemaps")
local_dir.mkdir(exist_ok=True)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
paginator = s3.get_paginator("list_objects_v2")

for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith(".txt"):
            filename = pathlib.Path(key).name
            local_path = local_dir / filename
            if not local_path.exists():
                s3.download_file(bucket, key, str(local_path))
            else:
                print(f"Skipping (already exists): {filename}")

