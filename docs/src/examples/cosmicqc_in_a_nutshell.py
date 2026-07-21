#!/usr/bin/env python
# coding: utf-8

# # `coSMicQC` in a nutshell
#
# This notebook demonstrates various capabilities of `coSMicQC` using examples.

# In[1]:


import pathlib

# In[2]:
from importlib.metadata import version

import pandas as pd
from cytodataframe import CytoDataFrame

import cosmicqc

version("cytodataframe")


# In[3]:


# set a path for the parquet-based dataset
# (in this case, CellProfiler SQLite data processed by CytoTable)
data_path = (
    "../../../tests/data/cytotable/NF1_cellpainting_data/"
    "Plate_2_with_image_data.parquet"
)

# set a context directory for images associated with the dataset
image_context_dir = pathlib.Path(data_path).parent / "Plate_2_images"
mask_context_dir = pathlib.Path(data_path).parent / "Plate_2_masks"

# create a cosmicqc CytoDataFrame (single-cell DataFrame)
scdf = CytoDataFrame(
    data=data_path,
    data_context_dir=image_context_dir,
    data_mask_context_dir=mask_context_dir,
)

# display the dataframe
scdf


# In[4]:


bbox_cols = [
    col for col in scdf.columns if "bbox" in col.lower() or "box" in col.lower()
]

print("bbox_col:", bbox_cols)
print("bbox_cols:")
for col in bbox_cols:
    print(col)


# In[5]:


# Identify which rows include outliers for a given threshold definition
# which references a column name and a z-score number which is considered
# the limit.
cosmicqc.analyze.identify_outliers(
    df=scdf,
    feature_thresholds={"Nuclei_AreaShape_Area": -1},
).sort_values()


# In[6]:


# Show the number of outliers given a column name and a specified threshold
# via the `feature_thresholds` parameter and the `find_outliers` function.
cosmicqc.analyze.find_outliers(
    df=scdf,
    metadata_columns=["Metadata_ImageNumber", "Image_Metadata_Plate_x"],
    feature_thresholds={"Nuclei_AreaShape_Area": -1},
)


# In[7]:


# create a labeled dataset which includes z-scores and whether those scores
# are interpreted as outliers or inliers. We use pre-defined threshold sets
# loaded from defaults (cosmicqc can accept user-defined thresholds too!).
labeled_scdf = cosmicqc.analyze.label_outliers(
    df=scdf, include_threshold_scores=True, feature_thresholds="large_nuclei"
)
labeled_scdf


# In[8]:


# show cropped images through CytoDataFrame from the dataset to help analyze outliers
# labeled_scdf._enbable_debug_mode()
labeled_scdf.sort_values(by="Metadata_cqc_large_nuclei_is_outlier", ascending=False)[
    [
        "Metadata_ImageNumber",
        "Metadata_Cells_Number_Object_Number",
        "Metadata_cqc_large_nuclei_is_outlier",
        "Image_FileName_GFP",
        "Image_FileName_RFP",
        "Image_FileName_DAPI",
    ]
]


# In[9]:


# One can convert from cosmicqc.CytoDataFrame to pd.DataFrame's
# (when or if needed!)
df = pd.DataFrame(scdf)
print(type(df))
