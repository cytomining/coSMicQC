#!/usr/bin/env python
# coding: utf-8

# # Generate distribution plots for features used to detect poor quality segmentations
#
# This code is derived from the `cellpainting_predicts_cardiac_fibrosis` repository.

# In[ ]:


import pathlib

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_density,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    theme,
    theme_bw,
    xlim,
)
from plotnine.options import set_option

from cosmicqc import find_outliers

# In[2]:


# Directory with data
data_dir = pathlib.Path(
    "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/converted_profiles/"
)

# Directory to save cleaned data
figure_dir = pathlib.Path("./figures")
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
plate_df = pd.read_parquet(f"{data_dir}/localhost230405150001_converted.parquet")

print(plate_df.shape)
plate_df.head()


# In[4]:


# Set outlier threshold that maximizes removing most technical outliers
outlier_threshold = 2

# find large nuclei and high intensity
feature_thresholds = {
    "Nuclei_AreaShape_Area": outlier_threshold,
    "Nuclei_Intensity_IntegratedIntensity_Hoechst": outlier_threshold,
}

large_nuclei_high_int_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

print(large_nuclei_high_int_outliers.shape)
pd.DataFrame(large_nuclei_high_int_outliers.head())


# In[5]:


# Set the default value to 'Single-cell passed QC'
plate_df["Outlier_Status"] = "Single-cell passed QC"

# Update Outlier_Status based on outliers index
plate_df.loc[
    plate_df.index.isin(large_nuclei_high_int_outliers.index), "Outlier_Status"
] = "Single-cell failed QC"

# Define colors
color_dict = {"Single-cell passed QC": "#006400", "Single-cell failed QC": "#990090"}

# Set the figure size
height = 6
width = 7
set_option("figure_size", (width, height))

# Create the plot
p = (
    ggplot(
        plate_df,
        aes(
            x="Nuclei_AreaShape_Area",
            y="Nuclei_Intensity_IntegratedIntensity_Hoechst",
            color="Outlier_Status",
        ),
    )
    + geom_point(alpha=0.3, size=3)
    + scale_color_manual(values=color_dict)
    + labs(
        x="Nuclei area",
        y="Nuclei integrated intensity (Hoechst)",
        color="Single-cell QC status",
    )
    + theme_bw()
    + theme(
        legend_position=(0.03, 0.97),
        legend_direction="vertical",
        legend_text=element_text(size=11),
        axis_title=element_text(size=12),
        axis_text=element_text(size=11),
    )
)

# Save figure
p.save(
    f"{figure_dir}/distribution_nuclei_outliers.png",
    dpi=500,
    width=width,
    height=height,
)

# Show plot
p.show()


# In[6]:


# Set threshold to detect outliers (abnormally small cells)
feature_thresholds = {
    "Cells_AreaShape_Area": -0.9,
}


small_cells_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

print(small_cells_outliers.shape)
pd.DataFrame(small_cells_outliers).sort_values(
    by="Cells_AreaShape_Area", ascending=False
).head()


# In[7]:


# Create Outlier_Status column
plate_df["Outlier_Status"] = "Single-cell passed QC"
plate_df.loc[plate_df.index.isin(small_cells_outliers.index), "Outlier_Status"] = (
    "Single-cell failed QC"
)

# Take log10 of the cell area
plate_df["Log_Cells_Area"] = np.log10(plate_df["Cells_AreaShape_Area"])

# Define colors
fill_colors = {"Single-cell passed QC": "#006400", "Single-cell failed QC": "#990090"}

# Create the density plot
p = (
    ggplot(plate_df, aes(x="Log_Cells_Area", fill="Outlier_Status"))
    + geom_density(alpha=0.5)
    + scale_fill_manual(values=fill_colors)
    + labs(x="Log10(Cells area)", y="Density", fill="Single-cell QC status")
    + theme_bw()
    + xlim(2.5, 5.5)
    + theme(
        legend_position=(0.97, 0.96),
        legend_text=element_text(size=11),
        axis_title=element_text(size=12),
        axis_text=element_text(size=11),
    )
)

# Save figure
p.save(
    f"{figure_dir}/distribution_cells_outliers.png", dpi=500, width=width, height=height
)

# Show plot
p.show()
