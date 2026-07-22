#!/usr/bin/env python
# coding: utf-8

# # Generate example image montages from ALSF example plate

# In[1]:


import os
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from pycytominer import annotate

from cosmicqc import find_outliers


# In[2]:


def add_scale_bar(
    ax: plt.Axes, img_shape: tuple, bar_length_px: float, color: str = "white"
) -> None:
    """Draw a scale bar in the bottom-right corner of an imshow'd image.

    Args:
        ax (plt.Axes): The axes to draw the scale bar on.
        img_shape (tuple): The shape of the image (height, width).
        bar_length_px (float): The length of the scale bar in pixels.
        color (str): The color of the scale bar. Default is "white".
    """
    img_h, img_w = img_shape[0], img_shape[1]
    margin_x = img_w * 0.05
    margin_y = img_h * 0.05

    x_end = img_w - margin_x
    x_start = x_end - bar_length_px
    y_pos = img_h - margin_y

    ax.plot(
        [x_start, x_end],
        [y_pos, y_pos],
        color=color,
        linewidth=3,
        solid_capstyle="butt",
    )


def add_highlight_box(ax: plt.Axes, color: str = "red", linewidth: int = 4) -> None:
    """Draw a highlight box around the full axes (image) area.

    Args:
        ax (plt.Axes): The axes to draw the highlight box on.
        color (str): The color of the highlight box. Default is "red".
        linewidth (int): The width of the highlight box lines. Default is 4.
    """
    rect = mpatches.Rectangle(
        (0, 0),
        1,
        1,
        transform=ax.transAxes,
        fill=False,
        edgecolor=color,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(rect)


# In[3]:


# Set output directory for figures
output_dir = Path(
    "./figures"
)
output_dir.mkdir(parents=True, exist_ok=True)


# Set pixel size resolution for all plots
# Pixel size from Index.xml (ImageResolutionX/Y = 5.93376264949402E-07 m)
UM_PER_PIXEL = 0.593376264949402
SCALE_BAR_UM = 100
SCALE_BAR_PX = SCALE_BAR_UM / UM_PER_PIXEL  # ~168.5 px


# Blind the cell line names
cell_line_map = {
    "CHLA-10": "Cell line A",
    "CHLA-113": "Cell line B",
    "CHLA-218": "Cell line C",
    "CHLA-25": "Cell line D",
    "U2-OS": "Cell line E",
}

# set plate to analyze
plate_id = "BR00145816"


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


# In[6]:


# Find large nuclei outliers for the current plate
nuclei_clustered_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        "Nuclei_Intensity_MassDisplacement_CorrDNA": 0.05,
        "Nuclei_Intensity_IntegratedIntensity_CorrDNA": 1.5,
    },
)

# Find low nuclei solidity outliers for the current plate
solidity_nuclei_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set at this point where it looks like it starts to detect good quality nuclei
        "Nuclei_AreaShape_Solidity": -1.6,
    },
)

# Find cell outliers for the current plate
cell_outliers = find_outliers(
    df=filtered_plate_df,
    metadata_columns=metadata_columns,
    feature_thresholds={
        # Set low to attempt to detect all instances of abnormally high int in nuclei for whole cells  # noqa: E501
        "Cells_Intensity_IntegratedIntensity_CorrDNA": 0.5,
    },
)


# In[7]:


# platemap file path
platemap_file = Path(
    "/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/0.download_data/metadata/platemaps/Assay_Plate4_platemap.csv"
)
platemap_df = pd.read_csv(platemap_file)

# Rename Image_Metadata_Site to Metadata_Site prior to annotation
filtered_plate_df = filtered_plate_df.rename(
    columns={"Image_Metadata_Site": "Metadata_Site"}
)

# Annotate the filtered_plate_df with metadata from platemap
annotated_plate_df = annotate(
    profiles=filtered_plate_df,
    platemap=platemap_df,
    join_on=["Metadata_well", "Image_Metadata_Well"],
)
print(annotated_plate_df.shape)
annotated_plate_df.head()


# In[8]:


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
            "Metadata_Site",
            "Metadata_Nuclei_Number_Object_Number",
        ]
    ]
    .apply(tuple, axis=1)
    .isin(all_outlier_keys),
    "failed_qc",
] = True

# Check result
annotated_plate_df["failed_qc"].value_counts()


# In[9]:


# Find FOVs with highest failure rate per cell line
fov_failure_rates = (
    annotated_plate_df.groupby(["Metadata_cell_line", "Metadata_Well", "Metadata_Site"])
    .agg(
        total_cells=("failed_qc", "count"),
        failed_cells=("failed_qc", "sum"),
        failure_rate=("failed_qc", "mean"),
    )
    .reset_index()
    .sort_values(["Metadata_cell_line", "failure_rate"], ascending=[True, False])
)

# Get top 2 FOVs per cell line
top_fovs = fov_failure_rates.groupby("Metadata_cell_line").head(2)

# Get file paths for these FOVs
fov_images = annotated_plate_df.merge(
    top_fovs[
        [
            "Metadata_cell_line",
            "Metadata_Well",
            "Metadata_Site",
            "failed_cells",
            "total_cells",
            "failure_rate",
        ]
    ],
    on=["Metadata_cell_line", "Metadata_Well", "Metadata_Site"],
)[
    [
        "Metadata_cell_line",
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_seeding_density",
        "failed_cells",
        "total_cells",
        "failure_rate",
        "Image_PathName_OrigDNA",
        "Image_FileName_OrigDNA",
    ]
].drop_duplicates()

top_fovs


# In[10]:


# Create a 2x5 subplot grid
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

# Get unique cell lines and sort them
cell_lines = sorted(fov_images["Metadata_cell_line"].unique())

# Iterate through each cell line and display the first two images
for col, cell_line in enumerate(cell_lines):
    cell_line_df = (
        fov_images[fov_images["Metadata_cell_line"] == cell_line]
        .sort_values("failure_rate", ascending=False)
        .reset_index(drop=True)
    )
    blinded_name = cell_line_map.get(cell_line, cell_line)
    for row in range(min(2, len(cell_line_df))):
        ax = axes[row * 5 + col]
        image_path = cell_line_df.loc[row, "Image_PathName_OrigDNA"]
        image_filename = cell_line_df.loc[row, "Image_FileName_OrigDNA"]
        full_path = os.path.join(image_path, image_filename)
        failed = cell_line_df.loc[row, "failed_cells"]
        total = cell_line_df.loc[row, "total_cells"]
        pct_failed = 100 * failed / total if total > 0 else 0
        try:
            img = imread(full_path)
            ax.imshow(img, cmap="gray")
            add_scale_bar(ax, img.shape, SCALE_BAR_PX)
            seeding_density = cell_line_df.loc[row, "Metadata_seeding_density"]
            ax.set_title(
                (
                    f"{blinded_name} | Seeding: {seeding_density}\n"
                    f"{failed}/{total} failed ({pct_failed:.1f}%)"
                ),
                fontsize=12,
            )
            ax.axis("off")
        except Exception:
            ax.text(
                0.5,
                0.5,
                f"Error loading image\n{blinded_name}",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.axis("off")

# Hide unused subplots
for idx in range(2 * len(cell_lines), len(axes)):
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig(output_dir / "fov_images_by_cellline.png", dpi=150, bbox_inches="tight")
plt.show()


# In[11]:


# "Most optimal" combos to highlight with a red box (blinded name -> seeding density)
# NOTE: adjust these density values/format (e.g. 12000 vs "12k") to match
# whatever format Metadata_seeding_density actually uses in your dataframe.
highlight_pairs = {
    "Cell line A": 12000,
    "Cell line B": 1000,
    "Cell line C": 1000,
    "Cell line D": 2000,
    "Cell line E": 4000,
}

cell_lines = sorted(annotated_plate_df["Metadata_cell_line"].unique())[:5]
seeding_densities = sorted(annotated_plate_df["Metadata_seeding_density"].unique())[:5]
random_state = 0
montage_records = []
for cell_line in cell_lines:
    for density in seeding_densities:
        subset = annotated_plate_df[
            (annotated_plate_df["Metadata_cell_line"] == cell_line)
            & (annotated_plate_df["Metadata_seeding_density"] == density)
        ]
        if subset.empty:
            continue
        montage_records.append(subset.sample(n=1, random_state=random_state))

montage_df = pd.concat(montage_records, ignore_index=True)
montage_index = montage_df.set_index(
    ["Metadata_cell_line", "Metadata_seeding_density"], drop=False
)

fig, axes = plt.subplots(
    len(seeding_densities),
    len(cell_lines),
    figsize=(20, 20),
    squeeze=False,
)

for row_idx, density in enumerate(seeding_densities):
    for col_idx, cell_line in enumerate(cell_lines):
        ax = axes[row_idx, col_idx]
        key = (cell_line, density)
        blinded_name = cell_line_map.get(cell_line, cell_line)
        if key not in montage_index.index:
            ax.axis("off")
            continue
        record = montage_index.loc[key]
        if isinstance(record, pd.DataFrame):
            record = record.iloc[0]
        full_path = os.path.join(
            record["Image_PathName_OrigDNA"], record["Image_FileName_OrigDNA"]
        )
        try:
            img = imread(full_path)
            ax.imshow(img, cmap="gray")
            add_scale_bar(ax, img.shape, SCALE_BAR_PX)
        except Exception:
            ax.text(
                0.5,
                0.5,
                "Unable to load image",
                ha="center",
                va="center",
                fontsize=12,
            )
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(blinded_name, fontsize=14, pad=12)
        if col_idx == 0:
            ax.set_ylabel(str(density), fontsize=14, rotation=90, labelpad=20, va="center")
            ax.axis("on")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        if highlight_pairs.get(blinded_name) == density:
            add_highlight_box(ax)

plt.tight_layout()
fig.savefig(output_dir / "random_fov_montage_5x5.png", dpi=150, bbox_inches="tight")
plt.show()

