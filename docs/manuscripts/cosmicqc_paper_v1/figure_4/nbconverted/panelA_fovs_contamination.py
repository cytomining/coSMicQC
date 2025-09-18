#!/usr/bin/env python
# coding: utf-8

# # Find FOVs that are good examples of nucleus staining with contamination and not

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

np.random.seed(0)  # Set seed for reproducibility


# In[2]:


# path to the directory containing the profiles
profiles_dir = pathlib.Path(
    "/media/18tbdrive/1.Github_Repositories/nf1_schwann_cell_painting_data/3.processing_features/data/cleaned_profiles"
)

# read in the Plate 3 profile to find example FOVs
plate3_profile = pd.read_parquet(profiles_dir / "Plate_3_cleaned.parquet")

# display the first few rows of the profile
print(plate3_profile.shape)
plate3_profile.head()


# In[3]:


# select specific wells to take FOVs from
selected_wells = ["E7", "B8", "C11", "G4"]

# filter the profile for the selected wells
filtered_profile = plate3_profile[
    plate3_profile["Image_Metadata_Well"].isin(selected_wells)
]

# display the filtered profile
filtered_profile.head()


# In[ ]:


# For each well, randomly select one Image_Metadata_Site
fovs = []
for well in selected_wells:
    well_rows = filtered_profile[
        filtered_profile["Image_Metadata_Well"].astype(str) == str(well)
    ]
    # Filter for rows with at least 10 cells
    cell_count_threshold = 10
    well_rows = well_rows[
        well_rows["Metadata_number_of_singlecells"] >= cell_count_threshold
    ]
    if not well_rows.empty:
        site = np.random.choice(well_rows["Image_Metadata_Site"].unique())
        cell_row = well_rows[well_rows["Image_Metadata_Site"] == site].iloc[0]
        fovs.append(
            {
                "well": well,
                "site": site,
                "Image_PathName_DAPI": cell_row["Image_PathName_DAPI"],
                "Image_FileName_DAPI": cell_row["Image_FileName_DAPI"],
            }
        )

# Print the selected FOVs
for fov in fovs:
    print(
        f"Well: {fov['well']}, Site: {fov['site']}, "
        f"DAPI Path: {fov['Image_PathName_DAPI']}, "
        f"DAPI File: {fov['Image_FileName_DAPI']}"
    )


# In[5]:


fovs_df = pd.DataFrame(fovs)
fovs_df.head()


# In[ ]:


# Set pixel-to-micron conversion
microns_per_pixel = 3.1065
scalebar_length_um = 100
scalebar_length_px = int(scalebar_length_um * microns_per_pixel)

# Plot only the DAPI image for each FOV in well_rows in a 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.flatten()

for ax, (_, fov_row) in zip(axes, fovs_df.iterrows()):
    img_path = fov_row["Image_PathName_DAPI"]
    img_file = fov_row["Image_FileName_DAPI"]
    full_path = pathlib.Path(img_path) / img_file

    img = Image.open(full_path)
    img_arr = np.array(img)

    GRAYSCALE_DIM = 2
    if img_arr.ndim == GRAYSCALE_DIM:  # grayscale
        img_arr = img_arr.astype(np.float32)

        # Contrast stretch using 1st and 99th percentiles
        p1, p99 = np.percentile(img_arr, (1, 99))
        img_arr = np.clip(img_arr, p1, p99)
        img_arr = 255 * (img_arr - p1) / (p99 - p1 + 1e-5)
        img_arr = img_arr.astype(np.uint8)

        cyan_img = np.zeros((*img_arr.shape, 3), dtype=np.uint8)
        cyan_img[..., 1] = img_arr  # G
        cyan_img[..., 2] = img_arr  # B

        # Add scale bar directly to image
        bar_height = 12  # in pixels (thickness of the bar)
        x_offset = 10
        y_offset = 10

        x_start = img_arr.shape[1] - scalebar_length_px - x_offset
        y_start = img_arr.shape[0] - bar_height - y_offset
        x_end = x_start + scalebar_length_px
        y_end = y_start + bar_height

        cyan_img[y_start:y_end, x_start:x_end] = 255  # white bar

        ax.imshow(cyan_img)
    else:
        ax.imshow(img)

    # Save the displayed image as PNG
    save_name = f"Well_{fov_row['well']}_Site_{fov_row['site']}.png"
    fov_dir = pathlib.Path("./panelA_FOVs")
    fov_dir.mkdir(exist_ok=True)
    if img_arr.ndim == GRAYSCALE_DIM:
        Image.fromarray(cyan_img).save(f"{fov_dir}/{save_name}")

    ax.set_title(f"Well {fov_row['well']}, Site {fov_row['site']}")
    ax.axis("off")

plt.tight_layout()
plt.show()
