#!/usr/bin/env python
# coding: utf-8

# ## Generate labelled FOVs for failing or passing single-cells
# 
# Do not use `Run All` if you want to run in sequential order.
# We make manual changes in some images prior to running the last code cell.

# In[1]:


import os
import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import PyQt5
from skimage import io

# Must set the PyQt5 plugin path before importing napari to avoid plugin errors
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
    pathlib.Path(PyQt5.__file__).parent / "Qt" / "plugins" / "platforms"
)

import napari


# In[2]:


# Helper function to apply gamma correction
def apply_gamma(image: np.ndarray, gamma: float = 0.6) -> np.ndarray:
    """
    Apply gamma correction to an image for visualization of background.

    Args:
        image (np.ndarray): Input image array (any numeric type).
        gamma (float, optional): Gamma correction value. Defaults to 0.6.

    Returns:
        np.ndarray: Gamma-corrected image as float32 with values normalized
            between 0 and 1.
    """
    image = image.astype(np.float32)
    max_val = image.max()
    if max_val == 0:
        return image  # Avoid division by zero; return original image if max is 0
    image /= max_val
    return np.power(image, gamma)


# In[3]:


# Load in no QC normalized dataframe for CFReT example plate
no_QC_df = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_normalized_no_QC.parquet"
    )
)

# Load in cleaned dataframe for CFReT example plate
cleaned_df = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/cleaned_profiles/localhost230405150001_cleaned.parquet"
    )
)

# Rename columns to match
cleaned_df = cleaned_df.rename(
    columns={
        "Image_Metadata_Well": "Metadata_Well",
        "Image_Metadata_Site": "Metadata_Site",
    }
)

# Print shapes
print("no QC dataframe shape:", no_QC_df.shape[0])
print("cleaned dataframe shape:", cleaned_df.shape[0])


# In[4]:


# Create QC_status column with default value 'Failed'
no_QC_df["QC_status"] = "Failed QC"

# Define columns to match on
match_cols = [
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]

# Create a MultiIndex for fast matching
cleaned_index = cleaned_df.set_index(match_cols).index
no_QC_index = no_QC_df.set_index(match_cols).index

# Find matching indices
matching = no_QC_index.isin(cleaned_index)

# Set QC_status to 'Passed' where matches are found
no_QC_df.loc[matching, "QC_status"] = "Passed QC"
no_QC_df["QC_status"].value_counts()


# In[5]:


# Select a well and site to visualize after running `whole_FOV_outlines.cppipe`
# CellProfiler pipeline
well = "E10"
site = "f01"

# Select the outline to apply to the FOV
compartment = "Cells"

# Select the gamma correction value
if compartment == "Cells" and well == "G07":
    gamma = 0.6
elif compartment == "Cells" and well == "E10":
    gamma = 0.25
elif compartment == "Nuclei":
    gamma = 0.6

# Filter the no_QC_df for the selected well and site
filtered_df = no_QC_df[
    (no_QC_df["Metadata_Well"] == well) & (no_QC_df["Metadata_Site"] == site)
]


# In[6]:


# Load the images
cells_image = io.imread(
    f"./whole_FOV_examples/localhost230405150001_{well}{site}d0_illumcorrect_{compartment}Overlay.tiff"
)
outlines_image = io.imread(
    f"./whole_FOV_examples/localhost230405150001_{well}{site}d0_illumcorrect_{compartment}OverlayOnly.tiff"
)

# Split the RGB channels and keep only the green channel (layer 1)
green_channel = outlines_image[:, :, 1]

# Convert green_channel to 32-bit to work with cv2.floodFill
green_channel = green_channel.astype(np.int32)

# Create an empty labels array
labels = np.zeros_like(green_channel, dtype=np.int32)
# Debug: Print the number of connected regions
print(f"Number of connected regions: {labels.max()}")


# In[7]:


print(f"Shape of cells_image: {cells_image.shape}")


# In[8]:


# Define a tolerance for flood filling
tolerance = 10

# Iterate over the DataFrame and use flood fill to fill the labels based on QC_status
for _, row in filtered_df.iterrows():
    x = int(row["Metadata_Nuclei_Location_Center_X"])
    y = int(row["Metadata_Nuclei_Location_Center_Y"])
    qc_status = row["QC_status"]

    # Convert mask to required format with extra border
    mask = np.zeros((green_channel.shape[0] + 2, green_channel.shape[1] + 2), np.uint8)

    # Use OpenCV's flood fill
    if qc_status == "Failed QC":
        _, _, _, rect = cv2.floodFill(
            green_channel, mask, (x, y), 44, loDiff=(tolerance,), upDiff=(tolerance,)
        )
    elif qc_status == "Passed QC":
        _, _, _, rect = cv2.floodFill(
            green_channel, mask, (x, y), 55, loDiff=(tolerance,), upDiff=(tolerance,)
        )

    # Debug: Print information about the region filled
    print(f"Filled region at ({x}, {y}) with QC status {qc_status}")

# Initialize Napari viewer
viewer = napari.Viewer()

# Add the cells image
cells_layer = viewer.add_image(cells_image, name="Cells")

# Set gamma of cells layer
cells_layer.gamma = gamma

# Add the modified labels layer
viewer.add_labels(
    green_channel, name="Outlines with QC", opacity=0.35, blending="additive"
)

# Start Napari viewer
napari.run()


# In[9]:


# Apply gamma correction to background
gamma_corrected = apply_gamma(cells_image, gamma)

# Convert to RGB if grayscale
correction_value = 2
if gamma_corrected.ndim == correction_value:
    base_img = np.stack([gamma_corrected] * 3, axis=-1)
else:
    base_img = gamma_corrected

# Convert to uint8 for plotting
base_img = (base_img * 255).astype(np.uint8)

# Create a semi-transparent RGBA overlay (Napari style)
overlay = np.zeros((*green_channel.shape, 4), dtype=np.uint8)
# Define colors for QC statuses
failed_qc_color = 44
passed_qc_color = 55
overlay[green_channel == failed_qc_color] = [221, 102, 102, 90]  # Soft red
overlay[green_channel == passed_qc_color] = [102, 187, 102, 90]  # Soft green

# Add 200 uM scale bar to images (1 uM/pixel)
scale_bar_length = 200  # in micrometers
scale_bar_pixels = scale_bar_length  # 1 uM/pixel
# Coordinates for bottom-right placement
bar_height = 8  # thickness of the scale bar
y_start = overlay.shape[0] - 40  # 40 px above bottom
y_end = y_start + bar_height
x_end = overlay.shape[1] - 40  # 40 px from right
x_start = x_end - scale_bar_pixels

# Draw scale bar
overlay[y_start:y_end, x_start:x_end] = [255, 255, 255, 255]

# Plot the labelled FOV and save
plt.figure(figsize=(6, 6), dpi=600)
plt.imshow(base_img)
plt.imshow(overlay, interpolation="none")
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(
    f"figures/{well}{site}_{compartment}.png", bbox_inches="tight", pad_inches=0
)
plt.close()

