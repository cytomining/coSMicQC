#!/usr/bin/env python
# coding: utf-8

# # Generate plots for features used in contamination detector

# In[ ]:


from pathlib import Path

import numpy as np
import pandas as pd
from cytodataframe import CytoDataFrame
from plotnine import (
    aes,
    element_text,
    geom_density,
    ggplot,
    labs,
    scale_fill_manual,
    theme,
    theme_bw,
)
from plotnine.options import set_option
from sklearn.preprocessing import StandardScaler

from cosmicqc import find_outliers

# In[2]:


# Directory with data
data_dir = Path(
    "/media/18tbdrive/1.Github_Repositories/nf1_schwann_cell_painting_data/3.processing_features/data/converted_data"
)

# Directory to save cleaned data
figure_dir = Path("./figures")
figure_dir.mkdir(exist_ok=True)

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]


# In[3]:


# Load in converted plate data
plate_df = pd.read_parquet(f"{data_dir}/Plate_3.parquet")

print(plate_df.shape)
plate_df.head()


# In[4]:


# compartment for bounding boxes
compartment = "Cytoplasm"

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
    "Image_FileName_DAPI",
    "Image_PathName_DAPI",
    f"{compartment}_AreaShape_BoundingBoxMaximum_X",
    f"{compartment}_AreaShape_BoundingBoxMaximum_Y",
    f"{compartment}_AreaShape_BoundingBoxMinimum_X",
    f"{compartment}_AreaShape_BoundingBoxMinimum_Y",
]

# find irregular shaped nuclei
feature_thresholds = {
    # outlier threshold for only cytoplasm texture in nuclei channel
    "Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256": 1,
}

irregular_nuclei_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

irregular_nuclei_outliers_cdf = CytoDataFrame(
    data=irregular_nuclei_outliers,
)[
    [
        "Image_FileName_DAPI",
        "Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Image_Metadata_Plate",
    ]
]


# In[5]:


# find irregular shaped nuclei
feature_thresholds = {
    # outlier threshold for only cytoplasm granularity in nuclei channel
    "Cytoplasm_Granularity_2_DAPI": 1,
}

granularity_nuclei_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

granularity_nuclei_outliers_cdf = CytoDataFrame(
    data=granularity_nuclei_outliers,
)[
    [
        "Image_FileName_DAPI",
        "Cytoplasm_Granularity_2_DAPI",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Image_Metadata_Plate",
    ]
]


# In[6]:


# Add QC_status column to plate_df for irregular nuclei outliers only
plate_df["QC_status"] = "Single-cell passed QC"
plate_df.loc[list(irregular_nuclei_outliers.index), "QC_status"] = (
    "Single-cell failed QC"
)


# In[7]:


# Z-score normalize the cytoplasm texture for plotting
scaler = StandardScaler()
plate_df[["Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256"]] = scaler.fit_transform(
    plate_df[["Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256"]]
)

# Define colors
fill_colors = {"Single-cell passed QC": "#006400", "Single-cell failed QC": "#990090"}

# Set the figure size
height = 6
width = 7
set_option("figure_size", (width, height))

# Create the density plot
p = (
    ggplot(
        plate_df, aes(x="Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256", fill="QC_status")
    )
    + geom_density(alpha=0.5)
    + scale_fill_manual(values=fill_colors)
    + labs(
        x="Z-score (cytoplasm texture around nucleus)",
        y="Density",
        fill="Single-cell QC status",
    )
    + theme_bw()
    + theme(
        legend_position=(0.07, 0.96),
        legend_title=element_text(size=14),
        legend_text=element_text(size=13),
        axis_title=element_text(size=15),
        axis_text=element_text(size=14),
    )
)

# Save the plot
p.save(
    figure_dir / "texture_distribution_plot.png",
    dpi=600,
    width=width,
    height=height,
)

# Show plot
p.show()


# In[8]:


# Add QC_status column to plate_df for granularity nuclei outliers only
plate_df["QC_status"] = "Single-cell passed QC"
plate_df.loc[list(granularity_nuclei_outliers_cdf.index), "QC_status"] = (
    "Single-cell failed QC"
)


# In[9]:


scaler = StandardScaler()
plate_df[["Cytoplasm_Granularity_2_DAPI"]] = scaler.fit_transform(
    plate_df[["Cytoplasm_Granularity_2_DAPI"]]
)

# Take log10 of the cytoplasm granularity
plate_df["Log_Cytoplasm_Granularity"] = np.log10(
    plate_df["Cytoplasm_Granularity_2_DAPI"]
)


# Define colors
fill_colors = {"Single-cell passed QC": "#006400", "Single-cell failed QC": "#990090"}

# Set the figure size
height = 6
width = 7
set_option("figure_size", (width, height))

# Create the density plot
p = (
    ggplot(plate_df, aes(x="Log_Cytoplasm_Granularity", fill="QC_status"))
    + geom_density(alpha=0.5)
    + scale_fill_manual(values=fill_colors)
    + labs(
        x="Log10(z-score of cytoplasm granularity around the nucleus)",
        y="Density",
        fill="Single-cell QC status",
    )
    + theme_bw()
    + theme(
        legend_position=(0.07, 0.96),
        legend_title=element_text(size=14),
        legend_text=element_text(size=13),
        axis_title=element_text(size=15),
        axis_text=element_text(size=14),
    )
)

# Save the plot
p.save(
    figure_dir / "granularity_distribution_plot.png",
    dpi=600,
    width=width,
    height=height,
)

# Show plot
p.show()
