#!/usr/bin/env python
# coding: utf-8

# # Generate distribution plots of the features used in the contamination detector

# In[1]:


from pathlib import Path

import pandas as pd
from cytodataframe import CytoDataFrame
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    labs,
    scale_fill_gradientn,
    theme,
    theme_bw,
    ylim,
)
from plotnine.options import set_option

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
    display_options={
        "center_dot": False,
        "brightness": 35,
        "pixel_per_um": 3.1065,
        "scale_bar": {
            "length_um": 20,
            "location": "lower right",
            "color": (255, 255, 255),
            "thickness_px": 4,
            "margin_px": 5,
        },
        "offset_bounding_box": {
            "x_min": -60,
            "y_min": -60,
            "x_max": 60,
            "y_max": 60,
        },
    },
)[
    [
        "Image_FileName_DAPI",
        "Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Image_Metadata_Plate",
    ]
]

print(irregular_nuclei_outliers_cdf.shape)
irregular_nuclei_outliers_cdf.sort_values(
    by="Cytoplasm_Texture_InfoMeas1_DAPI_3_02_256", ascending=True
).head(6).T


# In[5]:


# find irregular shaped nuclei
feature_thresholds = {
    # outlier threshold for only cytoplasm texture in nuclei channel
    "Cytoplasm_Granularity_2_DAPI": 1,
}

extra_nuclei_outliers = find_outliers(
    df=plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds=feature_thresholds,
)

extra_nuclei_outliers_cdf = CytoDataFrame(
    data=extra_nuclei_outliers,
    display_options={
        "center_dot": False,
        "brightness": 35,
        "pixel_per_um": 3.1065,
        "scale_bar": {
            "length_um": 20,
            "location": "lower right",
            "color": (255, 255, 255),
            "thickness_px": 4,
            "margin_px": 5,
        },
        "offset_bounding_box": {
            "x_min": -60,
            "y_min": -60,
            "x_max": 60,
            "y_max": 60,
        },
    },
)[
    [
        "Image_FileName_DAPI",
        "Cytoplasm_Granularity_2_DAPI",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
        "Image_Metadata_Plate",
    ]
]

print(extra_nuclei_outliers_cdf.shape)
# Filter for wells in columns 5-8
extra_nuclei_outliers_cdf[
    extra_nuclei_outliers_cdf["Image_Metadata_Well"]
    .str.extract(r"(\d+)")
    .astype(int)[0]
    .between(5, 8)
].sort_values(by="Cytoplasm_Granularity_2_DAPI", ascending=False).sample(
    n=6, random_state=0
)


# In[6]:


# Set QC_status column based on outlier dataframes
plate_df["QC_status"] = "Passed QC"
failed_indices = irregular_nuclei_outliers.index.union(extra_nuclei_outliers.index)
plate_df.loc[failed_indices, "QC_status"] = "Single-cell failed QC"

# Count failed QC cells per well
failed_qc_wells = plate_df.loc[
    plate_df["QC_status"] == "Single-cell failed QC", "Image_Metadata_Well"
]

failed_qc_counts = failed_qc_wells.value_counts().reset_index()
failed_qc_counts.columns = ["Image_Metadata_Well", "Failed_QC_Count"]

# Get total cell count per well
total_cells_per_well = plate_df["Image_Metadata_Well"].value_counts().reset_index()
total_cells_per_well.columns = ["Image_Metadata_Well", "Total_Cell_Count"]

# Merge total cells with failed counts — this keeps all wells
plot_df = total_cells_per_well.merge(
    failed_qc_counts, on="Image_Metadata_Well", how="left"
)
plot_df["Failed_QC_Count"] = plot_df["Failed_QC_Count"].fillna(0)

# Calculate percentage of failed QC cells per well
plot_df["Failed_QC_Proportion"] = (
    plot_df["Failed_QC_Count"] / plot_df["Total_Cell_Count"] * 100
)

# Extract row letter and column number for sorting
plot_df["Row"] = plot_df["Image_Metadata_Well"].str.extract(r"([A-Z])")
plot_df["Column"] = plot_df["Image_Metadata_Well"].str.extract(r"(\d+)").astype(int)

# Sort by column, then row
plot_df = plot_df.sort_values(["Column", "Row"])

# Update well order factor for x-axis to match new order
plot_df["Image_Metadata_Well"] = pd.Categorical(
    plot_df["Image_Metadata_Well"],
    categories=plot_df["Image_Metadata_Well"],
    ordered=True,
)


# In[7]:


# Assign genotype (blinded) based on column number directly
plot_df["Metadata_genotype"] = plot_df["Column"].apply(
    lambda col: (
        "cell_line_1"  # WT
        if 1 <= col <= 4  # noqa: PLR2004
        else (
            "cell_line_2"  # HET
            if 5 <= col <= 8  # noqa: PLR2004
            else "cell_line_3"
            if 9 <= col <= 12  # noqa: PLR2004
            else "Unknown"  # Null
        )
    )
)

print(plot_df.shape)
plot_df.head()


# In[8]:


# Set facet order: cell_line_1, cell_line_2, cell_line_3
genotype_order = ["cell_line_1", "cell_line_2", "cell_line_3"]
plot_df["Metadata_genotype"] = pd.Categorical(
    plot_df["Metadata_genotype"], categories=genotype_order, ordered=True
)

# Set the figure size
height = 8
width = 8
set_option("figure_size", (width, height))

# Create bar plot of proportion failed per well
p = (
    ggplot(
        plot_df,
        aes(x="Image_Metadata_Well", y="Failed_QC_Proportion", fill="Total_Cell_Count"),
    )
    + geom_bar(stat="identity", width=0.7)
    + scale_fill_gradientn(colors=["#FEE0D2", "#FC9272", "#FB6A4A", "#FF0000"])
    + labs(x="Well", y="Proportion of contaminated cells", fill="Total cell\ncount")
    + theme_bw()
    + facet_wrap("Metadata_genotype", scales="free_x", ncol=1)
    + ylim(0, 100)
    + theme(
        axis_text_x=element_text(angle=45, hjust=1, size=11),
        axis_text_y=element_text(size=16),
        axis_title=element_text(size=20),
        legend_title=element_text(size=16),
        legend_text=element_text(size=15),
        strip_text=element_text(size=13),  # Increase facet font size
    )
)

# Save the plot
p.save(
    figure_dir / "barplot_partial_contamination.png",
    dpi=600,
    width=width,
    height=height,
)

p.show()
