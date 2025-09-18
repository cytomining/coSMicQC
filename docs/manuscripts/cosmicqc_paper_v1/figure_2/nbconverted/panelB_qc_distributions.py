#!/usr/bin/env python
# coding: utf-8

# ## Generate plots to show the distributions of the failed versus passing cells
#
# We use the `PCCMA optimization` dataset.

# In[1]:


from pathlib import Path

import pandas as pd
from plotnine import (
    aes,
    element_text,
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
    theme,
    theme_light,
)
from plotnine.options import set_option

from cosmicqc import identify_outliers

# In[2]:


# Set output directory for figures
output_dir = Path("./figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load the data to generate the plots
example_df = pd.read_parquet(
    Path(
        "/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/3.preprocessing_features/data/converted_profiles/Round_2_data/BR00145816_converted.parquet"
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
]

# Define the QC features
qc_features = [
    "Nuclei_Intensity_IntegratedIntensity_CorrDNA",
    "Nuclei_AreaShape_Solidity",
    "Nuclei_Intensity_MassDisplacement_CorrDNA",
]

# Filter plate_df to only include metadata columns and QC features
filtered_plate_df = example_df[metadata_columns + qc_features]

print(example_df.shape)
example_df.head()


# In[3]:


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


# In[4]:


# Rename for easier access
rename_map = {
    (
        "cqc.custom.Z_Score.Nuclei_Intensity_IntegratedIntensity_CorrDNA"
    ): "zscore_intensity",
    (
        "cqc.custom.Z_Score.Nuclei_Intensity_MassDisplacement_CorrDNA"
    ): "zscore_displacement",
    "cqc.custom.is_outlier": "is_outlier",
}
df = identify_nuclei_clustered_outliers.rename(columns=rename_map)

# Make sure 'is_outlier' is a string or category
df["is_outlier"] = df["is_outlier"].map({True: "Outlier", False: "Not Outlier"})

# Set the figure size
height = 6
width = 10  # a little wider for facets
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
    + ggtitle("Over-segmented nuclei detection")
    + theme_light()
    + theme(
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
        legend_position="right",
        legend_direction="vertical",
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

p.show()


# In[5]:


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


# In[6]:


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
    + ggtitle("Poorly-segmented nuclei detection")
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

p.show()
