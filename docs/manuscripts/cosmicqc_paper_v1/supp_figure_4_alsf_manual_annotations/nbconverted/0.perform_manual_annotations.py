#!/usr/bin/env python
# coding: utf-8

# # Perform manual annotation of the subset of cells to evaluate QC performance on dataset

# In[1]:


import itertools
import re
from pathlib import Path

import ipywidgets as widgets
import numpy as np
import pandas as pd
from cytodataframe import CytoDataFrame
from IPython.display import clear_output, display
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    geom_hline,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    theme,
    theme_bw,
)
from plotnine.options import set_option
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# ## Helper functions to create manual annotation method

# In[2]:


def save_annotation(label: str) -> None:
    """
    Save the annotation for the current cell to the annotations DataFrame.

    Args:
        label (str): The annotation label for the current cell
            (e.g., "good", "bad", "unsure").
    """
    global annotations_df  # noqa: PLW0603

    row = pending_df.iloc[current_position]
    new_annotation = pd.DataFrame(
        {
            "sample_row_id": [int(row["sample_row_id"])],
            "manual_segmentation_label": [label],
            "manual_segmentation_is_bad": [label == "bad"],
        }
    )

    annotations_df = (
        pd.concat([annotations_df, new_annotation], ignore_index=True)
        .drop_duplicates("sample_row_id", keep="last")
        .sort_values("sample_row_id")
        .reset_index(drop=True)
    )
    annotations_df.to_parquet(annotation_path, index=False)


def print_cytodataframe_debug(row_df: pd.DataFrame) -> None:
    """
    Print debug for CytoDataFrame in case of issue with rendering crops.

    Args:
        row_df (pd.DataFrame): The DataFrame row being visualized in the CytoDataFrame.
    """
    print("CytoDataFrame debug")
    print("sample_row_id:", int(row_df["sample_row_id"].iloc[0]))
    print("outline_context_dir exists:", outline_context_dir.exists())
    print("outline_context_dir:", outline_context_dir)

    for filename_column, path_column in zip(image_filename_columns, image_path_columns):
        filename = str(row_df[filename_column].iloc[0])
        image_path = Path(str(row_df[path_column].iloc[0]), filename)
        print(f"{filename_column}: {image_path} | exists={image_path.exists()}")

        matching_outline_patterns = [
            outline_pattern
            for outline_pattern, image_pattern in outline_to_orig_mapping.items()
            if re.search(image_pattern, filename)
        ]
        print(f"matched outline patterns for {filename_column}:")
        for outline_pattern in matching_outline_patterns[:3]:
            outline_matches = [
                path.name
                for path in outline_context_dir.rglob("*")
                if re.search(outline_pattern, path.name)
            ]
            print(f"  {outline_pattern} -> {outline_matches[:3]}")

    if bounding_box_columns:
        print("bounding_box:", row_df[bounding_box_columns].iloc[0].to_dict())
    if center_xy_columns:
        print("center_xy:", row_df[center_xy_columns].iloc[0].to_dict())


def show_current_cell() -> None:
    """
    Display the current cell for annotation in the CytoDataFrame,
    along with progress information.
    If there are no pending cells left to annotate, display a completion message.
    """
    with output:
        clear_output(wait=True)

        if len(pending_df) == 0 or current_position >= len(pending_df):
            progress.value = "<b>All sampled cells have been annotated.</b>"
            print("All sampled cells have been annotated.")
            return

        progress.value = (
            f"<b>Current:</b> {current_position + 1} of {len(pending_df)} pending "
            f"| <b>Saved total:</b> {len(annotations_df)}"
        )
        row_df = pending_df.iloc[[current_position]].copy()
        if DEBUG_CYTODATAFRAME:
            print_cytodataframe_debug(row_df)

        row_cdf = CytoDataFrame(
            data=row_df[display_columns],
            data_image_paths=row_df[image_path_columns] if image_path_columns else None,
            data_bounding_box=(
                row_df[bounding_box_columns] if bounding_box_columns else None
            ),
            compartment_center_xy=(
                row_df[center_xy_columns] if center_xy_columns else False
            ),
            data_outline_context_dir=outline_context_dir,
            segmentation_file_regex=outline_to_orig_mapping,
            display_options={"brightness": 1},
        )
        if DEBUG_CYTODATAFRAME:
            row_cdf._enbable_debug_mode()
        display(row_cdf)


def annotate_and_advance(label: str) -> None:
    """
    Save the annotation for the current cell and advance to the next cell.

    Args:
        label (str): The annotation label for the current cell
            (e.g., "good", "bad", "unsure").
    """
    global current_position  # noqa: PLW0603

    if len(pending_df) == 0:
        show_current_cell()
        return

    save_annotation(label)
    current_position += 1

    show_current_cell()


# ## Load in stratified QC sample

# In[3]:


# set plate that was used for the stratified sample
plate_id = "BR00145816"

# Set path to the stratified sample
sample_path = Path("./data/stratified_qc_sample.parquet")

# Load the stratified sample for manual annotation
qc_sample_df = pd.read_parquet(sample_path)
print("Stratified sample loaded from:", sample_path)
print(qc_sample_df.shape)

preview_columns = [
    column
    for column in [
        "Metadata_cell_line",
        "Metadata_condition",
        "Metadata_Plate",
        "Metadata_Well",
        "Image_Metadata_Row",
        "Image_Metadata_Col",
    ]
    if column in qc_sample_df.columns
]
qc_sample_df[preview_columns].head()


# ## Manually annotate segmentation quality
# 
# The annotation widget below is intentionally blinded to the coSMicQC `failed_qc` result. It saves every button click to disk immediately and only serves cells that have not already been annotated.

# In[4]:


# Save manual annotations separately from the stratified sample
name_of_annotator = "test"  # change to your name or initials
annotation_path = Path(
    f"./data/manual_segmentation_annotations_{name_of_annotator}.parquet"
)
annotation_path.parent.mkdir(parents=True, exist_ok=True)

# Stable identifier for each sampled cell. This lets us resume safely even after
# shuffling and lets us merge with failed_qc only after manual annotation is done.
annotation_df = qc_sample_df.reset_index(names="sample_row_id").copy()

# Shuffle once for blinded review. Existing annotations are matched by sample_row_id,
# so resuming the notebook does not create duplicate work.
annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

if annotation_path.exists():
    annotations_df = pd.read_parquet(annotation_path)

    required_cols = {
        "sample_row_id",
        "manual_segmentation_label",
        "manual_segmentation_is_bad",
    }

    missing = required_cols - set(annotations_df.columns)

    if missing:
        raise ValueError(
            f"Annotation file is not valid. Missing columns: {missing}. "
            f"Likely wrong or overwritten file at {annotation_path}."
        )
else:
    annotations_df = pd.DataFrame(
        columns=[
            "sample_row_id",
            "manual_segmentation_label",
            "manual_segmentation_is_bad",
        ]
    )

annotations_df = annotations_df.drop_duplicates("sample_row_id", keep="last")
annotated_ids = set(annotations_df["sample_row_id"].astype(int))
pending_df = annotation_df.loc[
    ~annotation_df["sample_row_id"].astype(int).isin(annotated_ids)
].reset_index(drop=True)

print(f"Saved annotation file: {annotation_path}")
print(f"Already annotated: {len(annotations_df)}")
print(f"Remaining cells: {len(pending_df)}")


# In[5]:


# Create an outline-to-original-image regex mapping for CytoDataFrame.
# DNA images should display nuclei outlines; AGP images should display cell outlines.
# The regex values are matched against each Image_FileName_* value shown by CytoDataFrame.
outline_to_orig_mapping = {}

for record in pending_df[
    [
        "Metadata_Plate",
        "Metadata_Well",
        "Image_Metadata_Site",
        "Image_Metadata_Row",
        "Image_Metadata_Col",
    ]
].to_dict(orient="records"):
    image_prefix = (
        rf"r{int(record['Image_Metadata_Row']):02d}"
        rf"c{int(record['Image_Metadata_Col']):02d}"
        rf"f{int(record['Image_Metadata_Site']):02d}"
    )

    outline_to_orig_mapping[
        rf"NucleiOutlines_{record['Metadata_Plate']}_{record['Metadata_Well']}_{record['Image_Metadata_Site']}\.tiff"
    ] = rf"{image_prefix}p\d{{2}}-ch5sk\d+fk\d+fl\d+\.tiff"

    outline_to_orig_mapping[
        rf"CellsOutlines_{record['Metadata_Plate']}_{record['Metadata_Well']}_{record['Image_Metadata_Site']}\.tiff"
    ] = rf"{image_prefix}p\d{{2}}-ch3sk\d+fk\d+fl\d+\.tiff"

list(outline_to_orig_mapping.items())[:2]


# In[6]:


# CytoDataFrame display setup. The rendered table should only contain image
# filename columns. Image paths, bounding boxes, center coordinates, and outline
# mappings are passed to CytoDataFrame as rendering metadata.
DEBUG_CYTODATAFRAME = False

outline_context_dir = Path(
    f"/media/18tbdrive/1.Github_Repositories/pediatric_cancer_atlas_profiling/2.feature_extraction/sqlite_outputs/Round_2_data/{plate_id}"
).resolve()

image_filename_columns = [
    column
    for column in ["Image_FileName_OrigDNA", "Image_FileName_OrigAGP"]
    if column in pending_df.columns
]
image_path_columns = [
    column
    for column in [
        filename_column.replace("FileName", "PathName")
        for filename_column in image_filename_columns
    ]
    if column in pending_df.columns
]
bounding_box_columns = [
    column
    for column in [
        "Cells_AreaShape_BoundingBoxMinimum_X",
        "Cells_AreaShape_BoundingBoxMinimum_Y",
        "Cells_AreaShape_BoundingBoxMaximum_X",
        "Cells_AreaShape_BoundingBoxMaximum_Y",
    ]
    if column in pending_df.columns
]
center_xy_columns = [
    column
    for column in [
        "Metadata_Cells_Location_Center_X",
        "Metadata_Cells_Location_Center_Y",
    ]
    if column in pending_df.columns
]

# Only these columns will be visible in the CytoDataFrame display.
display_columns = image_filename_columns

required_render_columns = (
    image_filename_columns
    + image_path_columns
    + bounding_box_columns
    + center_xy_columns
)
missing_render_columns = [
    column for column in required_render_columns if column not in pending_df.columns
]

print("Visible CytoDataFrame columns:", display_columns)
print("Image path metadata columns:", image_path_columns)
print("Bounding box metadata columns:", bounding_box_columns)
print("Center XY metadata columns:", center_xy_columns)
print("Outline context exists:", outline_context_dir.exists(), outline_context_dir)
print("Outline regex mappings:", len(outline_to_orig_mapping))
if missing_render_columns:
    print("Missing render columns:", missing_render_columns)


# ### Fix path name root path to `bandicoot` path

# In[7]:


# Map original image paths to bandicoot mount paths for CytoDataFrame rendering
old_root = Path("/media/18tbdrive/ALSF_pilot_data/Round_2_data")

# Find the bandicoot mount anywhere under the user's home directory
bandicoot_root = next(p for p in Path.home().rglob("bandicoot") if p.is_dir())

# Set new root to find images in a common place
new_root = Path.home() / "mnt" / "bandicoot" / "Mike_manual_annotation_ALSF" / "images"

pending_df[image_path_columns] = pending_df[image_path_columns].apply(
    lambda col: col.map(
        lambda p: (str(new_root / Path(p).relative_to(old_root)) if pd.notna(p) else p)
    )
)

# Confirm it worked
pending_df[image_path_columns].head()


# In[8]:


# --- Set up interactive widgets for manual annotation and navigation ---
# Assign current position after loading existing annotations
current_position = 0
# Set output and progress display widgets
output = widgets.Output()
progress = widgets.HTML()

# Set up annotation buttons and their callbacks
good_button = widgets.Button(
    description="Good segmentation",
    button_style="success",
    icon="check",
)
bad_button = widgets.Button(
    description="Bad segmentation",
    button_style="danger",
    icon="times",
)
unsure_button = widgets.Button(
    description="Unsure / skip",
    button_style="warning",
    icon="question",
)

# Connect button clicks to annotation and navigation logic
good_button.on_click(lambda _: annotate_and_advance("good"))
bad_button.on_click(lambda _: annotate_and_advance("bad"))
unsure_button.on_click(lambda _: annotate_and_advance("unsure"))

# Initial display of progress and controls and complete manual annotation interface
display(progress)
display(widgets.HBox([good_button, bad_button, unsure_button]))
display(output)
show_current_cell()

