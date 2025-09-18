#!/usr/bin/env python
# coding: utf-8

# ## Perform single cell quality control on the profiles
#
# We utilize the conditions used in the ALSF project repository.
# We detect poor quality nuclei segmentations.
#
# NOTE: We run this notebook via `papermill` through this [bash script](./run_single_cell_qc.sh).  # noqa: E501
# We found it was easier to run through all 136 plates in this format over a for loop.

# In[1]:


import pathlib
import time

# Ignore FutureWarnings from cytodataframe due to skimage deprecation
# (does not affect functionality)
import warnings

import pandas as pd

from cosmicqc import find_outliers

warnings.filterwarnings("ignore", category=FutureWarning)


# In[2]:


plate_id = "SQ00014812"


# In[3]:


# Parameters
plate_id = "SQ00015157"


# In[4]:


# Directory containing the converted profiles
data_dir = pathlib.Path("/media/NVME_4TB/LINCS_cytotable_output/data/")

# Directory to save labeled data
labeled_dir = pathlib.Path(
    "/home/jenna/mnt/bandicoot/LINCS_data/processed_profiles/qc_profiles"
)
labeled_dir.mkdir(exist_ok=True)

# Create an empty dictionary to store data frames for each plate
all_qc_data_frames = {}

# Set the compartment of choice to perform QC at the start (will change later)
compartment = "Nuclei"


# In[5]:


# Construct the file path for the given plate_id
file_path = data_dir / f"{plate_id}/{plate_id}.parquet"

if file_path.exists():
    start_time = time.time()  # Start timer for loading

    # Load the DataFrame with pandas
    plate_df = pd.read_parquet(file_path, engine="pyarrow")

    end_time = time.time()  # End timer for loading
    print(
        f"Loaded plate: {plate_id}, "
        f"Shape: {plate_df.shape}, "
        f"Time taken: {end_time - start_time:.2f} seconds"
    )
else:
    print(f"Parquet file for plate {plate_id} not found.")


# In[6]:


# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    f"{compartment}_Location_Center_X",
    f"{compartment}_Location_Center_Y",
]

# Define the QC features
qc_features = [
    "Nuclei_Intensity_IntegratedIntensity_DNA",
    "Nuclei_AreaShape_Solidity",
    "Nuclei_Intensity_MassDisplacement_DNA",
]

# Filter plate_df to only include metadata columns and QC features
filtered_plate_df = plate_df[metadata_columns + qc_features]

# Drop any rows with NaN values in the QC features
filtered_plate_df = filtered_plate_df.dropna(subset=qc_features)

# Print the first few rows of the filtered DataFrame
print("Filtered plate DataFrame shape:", filtered_plate_df.shape)
filtered_plate_df.head()


# In[ ]:


# Find large nuclei outliers for the current plate
nuclei_clustered_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set very low as to detect all instances of clustering nuclei
        "Nuclei_Intensity_MassDisplacement_DNA": 0.05,
        # Set higher than displacement to avoid false positives
        "Nuclei_Intensity_IntegratedIntensity_DNA": 1.5,
    },
)

# Convert to regular pandas DataFrame
nuclei_clustered_outliers = pd.DataFrame(nuclei_clustered_outliers)

print(nuclei_clustered_outliers.shape)
nuclei_clustered_outliers.sort_values(
    by="Nuclei_Intensity_MassDisplacement_DNA", ascending=True
).head(2)


# In[ ]:


# Find low nuclei solidity outliers for the current plate
solidity_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set at this point where it looks like it starts to detect good quality nuclei
        "Nuclei_AreaShape_Solidity": -1.6,
    },
)

# Convert to regular pandas DataFrame
solidity_nuclei_outliers = pd.DataFrame(solidity_nuclei_outliers)

print(solidity_nuclei_outliers.shape)
solidity_nuclei_outliers.sort_values(
    by="Nuclei_AreaShape_Solidity", ascending=False
).head(2)


# In[9]:


# Set compartment as cells
compartment = "Cells"

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    f"{compartment}_Location_Center_X",
    f"{compartment}_Location_Center_Y",
]

# Define the QC features
qc_features = ["Cells_Intensity_IntegratedIntensity_DNA"]

# Filter plate_df to only include metadata columns and QC features
filtered_plate_df = plate_df[metadata_columns + qc_features]

# Drop any rows with NaN values in the QC features
filtered_plate_df = filtered_plate_df.dropna(subset=qc_features)


# In[ ]:


# Find cell outliers for the current plate
cell_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set low to attempt to detect all instances of abnormally high int in nuclei
        # for whole cells
        "Cells_Intensity_IntegratedIntensity_DNA": 0.5,
    },
)

# Convert to regular pandas DataFrame
cell_outliers = pd.DataFrame(cell_outliers)

print(cell_outliers.shape)
cell_outliers.sort_values(
    by="Cells_Intensity_IntegratedIntensity_DNA", ascending=True
).head(2)


# In[11]:


# Add QC failure columns to plate_df based on outlier indices
plate_df["cqc.failed_clustered_nuclei"] = plate_df.index.isin(
    nuclei_clustered_outliers.index
)
plate_df["cqc.failed_low_solidity_nuclei"] = plate_df.index.isin(
    solidity_nuclei_outliers.index
)
plate_df["cqc.failed_cell_outlier"] = plate_df.index.isin(cell_outliers.index)

# Save the labeled dataframe to parquet
labeled_path = labeled_dir / f"{plate_id}_qc_labeled.parquet"
plate_df.to_parquet(labeled_path, index=False)
print(f"Labeled dataframe for plate {plate_id} saved to {labeled_path}")

# print the shape and head of the updated DataFrame
print(plate_df.shape)
plate_df.head()
