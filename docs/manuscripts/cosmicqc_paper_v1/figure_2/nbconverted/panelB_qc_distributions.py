#!/usr/bin/env python
# coding: utf-8

# ## Generate plots to show the distributions of the failed versus passing cells
#
# We use the `PCCMA optimization` dataset.

# In[1]:


import re
from pathlib import Path

import pandas as pd
from cytodataframe import CytoDataFrame
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_bin2d,
    geom_histogram,
    geom_hline,
    geom_point,
    geom_vline,
    ggplot,
    ggtitle,
    guides,
    labs,
    scale_color_manual,
    scale_fill_gradient,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_bw,
    theme_light,
)
from plotnine.options import set_option
from pycytominer import annotate

from cosmicqc import find_outliers, identify_outliers

# In[2]:


# Set the figure size for all plots
height = 6
width = 8
set_option("figure_size", (width, height))

# Set output directory for figures
output_dir = Path("./figures")
output_dir.mkdir(parents=True, exist_ok=True)

# set plate to analyze
plate_id = "BR00145816"


# In[3]:


# Load the data to generate the plots
example_df = pd.read_parquet(
    Path(
        f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/3.preprocessing_features/data/converted_profiles/Round_2_data/{plate_id}_converted.parquet"
    )
)

# Print the shape of the dataframe
print(example_df.shape)


# In[4]:


# Load the data to generate the plots
example_df = pd.read_parquet(
    Path(
        f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/3.preprocessing_features/data/converted_profiles/Round_2_data/{plate_id}_converted.parquet"
    )
)

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
    "Image_FileName_OrigDNA",
    "Image_FileName_OrigAGP",
    "Image_PathName_OrigDNA",
    "Image_PathName_OrigAGP",
    "Nuclei_AreaShape_BoundingBoxMaximum_X",
    "Nuclei_AreaShape_BoundingBoxMaximum_Y",
    "Nuclei_AreaShape_BoundingBoxMinimum_X",
    "Nuclei_AreaShape_BoundingBoxMinimum_Y",
    "Image_Metadata_Row",
    "Image_Metadata_Col",
    "Metadata_Nuclei_Number_Object_Number",
]

# Define the QC features
qc_features = [
    "Nuclei_Intensity_IntegratedIntensity_CorrDNA",
    "Nuclei_AreaShape_Solidity",
    "Nuclei_Intensity_MassDisplacement_CorrDNA",
    "Cells_Intensity_IntegratedIntensity_CorrDNA",
]

# Filter plate_df to only include metadata columns and QC features
filtered_plate_df = example_df[metadata_columns + qc_features]

print(example_df.shape)
example_df.head()


# In[5]:


# Find large nuclei outliers for the current plate
identify_nuclei_clustered_outliers = identify_outliers(
    df=filtered_plate_df,
    feature_thresholds={
        # Set very low as to detect all instances of clustering nuclei
        "Nuclei_Intensity_MassDisplacement_CorrDNA": 0.05,
        # Set higher than displacement to avoid false positives
        "Nuclei_Intensity_IntegratedIntensity_CorrDNA": 1.5,
    },
    include_threshold_scores=True,
)

pd.DataFrame(identify_nuclei_clustered_outliers).head()


# In[ ]:


# Rename for easier access
rename_map = {
    "cqc.custom.Z_Score.Nuclei_Intensity_IntegratedIntensity_CorrDNA": "zscore_intensity",  # noqa: E501
    "cqc.custom.Z_Score.Nuclei_Intensity_MassDisplacement_CorrDNA": "zscore_displacement",  # noqa: E501
    "cqc.custom.is_outlier": "is_outlier",
}
df = identify_nuclei_clustered_outliers.rename(columns=rename_map)

# Make sure 'is_outlier' is a string or category
df["is_outlier"] = df["is_outlier"].map({True: "Outlier", False: "Not Outlier"})

# Set the figure size
set_option("figure_size", (width, height))

# Plot scatterplot with thresholds for over-segmented nuclei
p = (
    ggplot(df, aes(x="zscore_intensity", y="zscore_displacement", color="is_outlier"))
    + geom_point(alpha=0.3, size=0)
    + geom_bin2d(aes(fill="..count.."), bins=50, alpha=1.0, size=0.8)
    + scale_fill_gradient(
        name="Log10\n(Cell count)", trans="log10", low="#1b0064", high="#f8d125"
    )
    + geom_vline(xintercept=1.5, linetype="--", color="#800080", size=1.0)
    + geom_hline(yintercept=0.05, linetype="--", color="#800080", size=1.0)
    + scale_color_manual(values={"Outlier": "#A658A6", "Not Outlier": "#5C8F5C"})
    + labs(
        x="Z-score (nuclei intensity)",
        y="Z-score\n(nuclei mass displacement)",
        fill="Log10(Cell count)",
    )
    + guides(color=False)  # 🔹 drop outlier/not outlier legend
    + ggtitle("Condition 1: Over-segmented nuclei detection")
    + theme_light()
    + theme(
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
        legend_position="bottom",
        legend_direction="horizontal",
        axis_title=element_text(size=20),
        axis_text=element_text(size=15),
        plot_title=element_text(size=20),
        axis_title_y=element_text(
            angle=90,
            vjust=0.5,
            ha="center",
        ),
        plot_margin_left=0.03,
    )
)

# Save the plot
p.save(
    output_dir / "over_segmented_nuclei_plot.png", dpi=600, width=width, height=height
)

# Save version without legend
(p + theme(legend_position="none")).save(
    output_dir / "over_segmented_nuclei_plot_no_legend.png",
    dpi=600,
    width=width,
    height=height,
)

p.show()


# In[7]:


# Find large nuclei outliers for the current plate
identify_poor_nuclei_shape_outliers = identify_outliers(
    df=filtered_plate_df,
    # Set at this point where it looks like it starts to detect good quality nuclei
    feature_thresholds={
        "Nuclei_AreaShape_Solidity": -1.6,
    },
    include_threshold_scores=True,
)

pd.DataFrame(identify_poor_nuclei_shape_outliers).head()


# In[8]:


# Rename for easier access
df = identify_poor_nuclei_shape_outliers.rename(
    columns={
        "cqc.custom.Z_Score.Nuclei_AreaShape_Solidity": "zscore_solidity",
        "cqc.custom.is_outlier": "is_outlier",
    }
)

# Make sure 'is_outlier' is a string or category (for coloring)
df["is_outlier"] = df["is_outlier"].map({True: "Outlier", False: "Not Outlier"})

# Set the figure size
set_option("figure_size", (width, height))

# Plot histogram for poorly-segmented nuclei
p = (
    ggplot(df, aes(x="zscore_solidity", fill="is_outlier"))
    + geom_histogram(position="stack", bins=45, color="black")
    + scale_fill_manual(values={"Not Outlier": "#5C8F5C", "Outlier": "#A658A6"})
    + geom_vline(
        xintercept=df.loc[df["is_outlier"] == "Outlier", "zscore_solidity"].max(),
        linetype="dashed",
        color="#800080",
        size=1.0,
    )
    + labs(x="Z-score (nuclei solidity)", y="Single-cell count", fill="Outlier status")
    + ggtitle("Condition 2: Poorly-segmented nuclei detection")
    + theme_light()
    + theme(
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
        legend_position="right",
        axis_title=element_text(size=20),
        axis_text=element_text(size=15),
        plot_title=element_text(
            size=20,
        ),
    )
)

# Save the plot
p.save(
    output_dir / "poorly_segmented_nuclei_plot.png", dpi=600, width=width, height=height
)

# Save version without legend
(p + theme(legend_position="none")).save(
    output_dir / "poorly_segmented_nuclei_plot_no_legend.png",
    dpi=600,
    width=width,
    height=height,
)

p.show()


# In[9]:


# Find over-segmented cell outliers for the current plate
over_segmented_cells_outliers = identify_outliers(
    df=filtered_plate_df,
    feature_thresholds={
        # Set low to detect instances of abnormally high int in nuclei for whole cells
        "Cells_Intensity_IntegratedIntensity_CorrDNA": 0.5,
    },
    include_threshold_scores=True,
)

pd.DataFrame(over_segmented_cells_outliers).head()


# In[10]:


# Rename for easier access
df = over_segmented_cells_outliers.rename(
    columns={
        "cqc.custom.Z_Score.Cells_Intensity_IntegratedIntensity_CorrDNA": "zscore_integrated_intensity",  # noqa: E501
        "cqc.custom.is_outlier": "is_outlier",
    }
)

# Make sure 'is_outlier' is a string or category (for coloring)
df["is_outlier"] = df["is_outlier"].map({True: "Outlier", False: "Not Outlier"})

# Set the figure size
set_option("figure_size", (width, height))

p = (
    ggplot(df, aes(x="zscore_integrated_intensity", fill="is_outlier"))
    + geom_histogram(position="stack", bins=45, color="black")
    + scale_fill_manual(values={"Not Outlier": "#5C8F5C", "Outlier": "#A658A6"})
    + geom_vline(
        xintercept=df.loc[
            df["is_outlier"] == "Outlier", "zscore_integrated_intensity"
        ].min(),
        linetype="dashed",
        color="#800080",
        size=1.0,
    )
    + scale_x_continuous(trans="log1p")
    + labs(
        x="Z-score (log1p of nuclei total intensity in cells)",
        y="Single-cell count",
        fill="Outlier status",
    )
    + ggtitle("Condition 3: Over-segmented cells detection")
    + theme_light()
    + theme(
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
        legend_position="right",
        axis_title=element_text(size=20),
        axis_text=element_text(size=15),
        plot_title=element_text(
            size=20,
        ),
    )
)

# Save the plot
# Save the plot
p.save(
    output_dir / "over_segmented_cells_plot.png", dpi=600, width=width, height=height
)

# Save version without legend
(p + theme(legend_position="none")).save(
    output_dir / "over_segmented_cells_plot_no_legend.png",
    dpi=600,
    width=width,
    height=height,
)

p.show()


# ## Print out CytoDataFrame to visualize crops

# In[11]:


correct_parent = "/media/18tbdrive/ALSF_pilot_data"

for col in filtered_plate_df.columns:
    if "PathName" in col and "Illum" not in col:
        filtered_plate_df[col] = filtered_plate_df[col].apply(
            lambda x: (
                re.sub(r"^.*ALSF_pilot_data/", correct_parent + "/", x)
                if isinstance(x, str)
                else x
            )
        )

# Print example image path after fix
print(filtered_plate_df["Image_PathName_OrigDNA"].dropna().iloc[0])


# In[12]:


# create an outline and orig mapping dictionary to map original images to outlines
# note: we turn off formatting here to avoid the key-value pairing definition
# from being reformatted by black, which is normally preferred.
# fmt: off
compartment = "Nuclei"

outline_to_orig_mapping = {
    rf"{compartment}Outlines_{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}_{record['Image_Metadata_Site']}.tiff":
    rf"r{int(record['Image_Metadata_Row']):02d}c{int(record['Image_Metadata_Col']):02d}f{int(record['Image_Metadata_Site']):02d}p(\d{{2}})-ch\d+sk\d+fk\d+fl\d+\.tiff"
    for record in filtered_plate_df[
        [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Image_Metadata_Row",
            "Image_Metadata_Col",
        ]
    ].to_dict(orient="records")
}
# fmt: on

next(iter(outline_to_orig_mapping.items()))


# In[13]:


# Find large nuclei outliers for the current plate
nuclei_clustered_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_Intensity_MassDisplacement_CorrDNA": 0.05,
        "Nuclei_Intensity_IntegratedIntensity_CorrDNA": 1.5,
    },
)

# MUST SET DATA AS DATAFRAME FOR OUTLINE DIR TO WORK
nuclei_clustered_outliers_cdf = CytoDataFrame(
    data=pd.DataFrame(nuclei_clustered_outliers),
    data_outline_context_dir=f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/2.feature_extraction/sqlite_outputs/Round_2_data/{plate_id}",
    segmentation_file_regex=outline_to_orig_mapping,
    display_options={
        "center_dot": False,
        "outline_color": (180, 30, 180),  # magenta
        "brightness": 20,
    },
)[
    [
        "Nuclei_Intensity_MassDisplacement_CorrDNA",
        "Nuclei_Intensity_IntegratedIntensity_CorrDNA",
        "Image_FileName_OrigDNA",
    ]
]


print(nuclei_clustered_outliers_cdf.shape)
nuclei_clustered_outliers_cdf.sort_values(
    by="Nuclei_Intensity_MassDisplacement_CorrDNA", ascending=True
).sample(n=2, random_state=42)


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

# Convert to CytoDataFrame for outline viewing
solidity_nuclei_outliers_cdf = CytoDataFrame(
    data=pd.DataFrame(solidity_nuclei_outliers),
    data_outline_context_dir=f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/2.feature_extraction/sqlite_outputs/Round_2_data/{plate_id}",
    segmentation_file_regex=outline_to_orig_mapping,
    display_options={
        "center_dot": False,
        "brightness": 20,
        "outline_color": (180, 30, 180),
    },
)[
    [
        "Nuclei_AreaShape_Solidity",
        "Image_FileName_OrigDNA",
    ]
]


print(solidity_nuclei_outliers_cdf.shape)
solidity_nuclei_outliers_cdf.sort_values(
    by="Nuclei_AreaShape_Solidity", ascending=False
).head(4)


# In[15]:


# change compartment to cells
compartment = "Cells"

# metadata columns to include in output data frame
metadata_columns = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
    "Image_Metadata_Site",
    f"Metadata_{compartment}_Location_Center_X",
    f"Metadata_{compartment}_Location_Center_Y",
    "Image_FileName_OrigDNA",
    "Image_FileName_OrigAGP",
    "Image_PathName_OrigDNA",
    "Image_PathName_OrigAGP",
    f"{compartment}_AreaShape_BoundingBoxMaximum_X",
    f"{compartment}_AreaShape_BoundingBoxMaximum_Y",
    f"{compartment}_AreaShape_BoundingBoxMinimum_X",
    f"{compartment}_AreaShape_BoundingBoxMinimum_Y",
    "Image_Metadata_Row",
    "Image_Metadata_Col",
    "Metadata_Nuclei_Number_Object_Number",
]

# Define the QC features
qc_features = [
    "Nuclei_Intensity_IntegratedIntensity_CorrDNA",
    "Nuclei_AreaShape_Solidity",
    "Nuclei_Intensity_MassDisplacement_CorrDNA",
    "Cells_Intensity_IntegratedIntensity_CorrDNA",
]

# Filter plate_df to only include metadata columns and QC features
filtered_plate_df = example_df[metadata_columns + qc_features]

# create an outline and orig mapping dictionary to map original images to outlines
# note: we turn off formatting here to avoid the key-value pairing definition
# from being reformatted by black, which is normally preferred.
# fmt: off
outline_to_orig_mapping = {
    rf"{compartment}Outlines_{record['Image_Metadata_Plate']}_{record['Image_Metadata_Well']}_{record['Image_Metadata_Site']}.tiff":
    rf"r{int(record['Image_Metadata_Row']):02d}c{int(record['Image_Metadata_Col']):02d}f{int(record['Image_Metadata_Site']):02d}p(\d{{2}})-ch\d+sk\d+fk\d+fl\d+\.tiff"
    for record in filtered_plate_df[
        [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Image_Metadata_Row",
            "Image_Metadata_Col",
        ]
    ].to_dict(orient="records")
}
# fmt: on

next(iter(outline_to_orig_mapping.items()))


# In[16]:


correct_parent = "/media/18tbdrive/ALSF_pilot_data"

for col in filtered_plate_df.columns:
    if "PathName" in col and "Illum" not in col:
        filtered_plate_df[col] = filtered_plate_df[col].apply(
            lambda x: (
                re.sub(r"^.*ALSF_pilot_data/", correct_parent + "/", x)
                if isinstance(x, str)
                else x
            )
        )

# Print example image path after fix
print(filtered_plate_df["Image_PathName_OrigDNA"].dropna().iloc[0])


# In[ ]:


# Find cell outliers for the current plate
cell_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set low to attempt to detect all instances of abnormally high int in nuclei for whole cells  # noqa: E501
        "Cells_Intensity_IntegratedIntensity_CorrDNA": 0.5,
    },
)

# Convert to CytoDataFrame for outline viewing
cell_outliers_cdf = CytoDataFrame(
    data=pd.DataFrame(cell_outliers),
    data_outline_context_dir=f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/2.feature_extraction/sqlite_outputs/Round_2_data/{plate_id}",
    segmentation_file_regex=outline_to_orig_mapping,
    display_options={
        "center_dot": False,
        "brightness": 20,
        "outline_color": (180, 30, 180),
    },
)[
    [
        "Cells_Intensity_IntegratedIntensity_CorrDNA",
        "Image_FileName_OrigDNA",
    ]
]


print(cell_outliers_cdf.shape)
cell_outliers_cdf.sort_values(
    by="Cells_Intensity_IntegratedIntensity_CorrDNA", ascending=True
).sample(n=2, random_state=42)


# In[18]:


# platemap file path
platemap_file = Path(
    "/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/0.download_data/metadata/platemaps/Assay_Plate4_platemap.csv"
)
platemap_df = pd.read_csv(platemap_file)

# Annotate the filtered_plate_df with metadata from platemap
annotated_plate_df = annotate(
    profiles=filtered_plate_df,
    platemap=platemap_df,
    join_on=["Metadata_well", "Image_Metadata_Well"],
)
print(annotated_plate_df.shape)
annotated_plate_df.head()


# In[19]:


all_outlier_keys = set(
    pd.concat([nuclei_clustered_outliers, solidity_nuclei_outliers, cell_outliers])[
        [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Metadata_Nuclei_Number_Object_Number",
        ]
    ].apply(tuple, axis=1)
)

# Rename well and plate to just Metadata_ prefix
all_outlier_keys = {
    (plate, well, site, int(obj_num))
    for (plate, well, site, obj_num) in all_outlier_keys
}


# Create failed_qc column, default to False
annotated_plate_df["failed_qc"] = False
# Set to True where the cell matches the outlier dataframe
annotated_plate_df.loc[
    annotated_plate_df[
        [
            "Metadata_Plate",
            "Metadata_Well",
            "Image_Metadata_Site",
            "Metadata_Nuclei_Number_Object_Number",
        ]
    ]
    .apply(tuple, axis=1)
    .isin(all_outlier_keys),
    "failed_qc",
] = True

# Check result
annotated_plate_df["failed_qc"].value_counts()


# In[20]:


# Group by cell line and seeding density, and calculate
# total nuclei segmented and failed QC
failure_stats = (
    annotated_plate_df.groupby(
        [
            "Metadata_cell_line",
            "Metadata_seeding_density",
            "Metadata_condition",
            "Metadata_Plate",
        ]
    )
    .agg(
        total_nuclei_segmented=("failed_qc", "count"),
        total_failed_qc=("failed_qc", "sum"),
        percentage_failing_cells=("failed_qc", "mean"),
    )
    .reset_index()
)

# Blind the cell line names
cell_line_map = {
    "CHLA-10": "Cell line A",
    "CHLA-113": "Cell line B",
    "CHLA-218": "Cell line C",
    "CHLA-25": "Cell line D",
    "U2-OS": "Cell line E",
}
failure_stats["Metadata_cell_line"] = failure_stats["Metadata_cell_line"].map(
    cell_line_map
)

# Set the figure size
width = 16
height = 5
set_option("figure_size", (width, height))

# Plot using plotnine
p = (
    ggplot(
        failure_stats,
        aes(
            x="factor(Metadata_seeding_density)",
            y="percentage_failing_cells",
            fill="total_nuclei_segmented",
        ),
    )
    + geom_bar(stat="identity")
    + facet_wrap("~Metadata_cell_line", nrow=1, scales="free_x")
    + labs(
        x="Seeding density",
        y="Proportion of failed cells",
        fill="Cell count",
    )
    + scale_fill_gradient(low="#C6DBEF", high="#08306B")
    + theme_bw(base_size=16)
    + theme(axis_text_x=element_text(rotation=45, hjust=1))
)

# Save the plot
p.save(f"{output_dir}/failed_qc_summary.png", dpi=600, width=width, height=height)
p.show()
