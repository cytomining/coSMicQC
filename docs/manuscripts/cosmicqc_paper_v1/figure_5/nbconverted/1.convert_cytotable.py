#!/usr/bin/env python
# coding: utf-8

# # Using the manifest, convert all plate SQLite files into parquet files for processing
#
# This code is derived from the `JUMP-single-cell` repository, which can be found [here](https://github.com/WayScience/JUMP-single-cell/blob/main/0.download_data/1.process_JUMP_plates_with_CytoTable.py).

# In[1]:


import pathlib
import shutil

import pandas as pd
from cytotable import convert, presets
from parsl.config import Config
from parsl.executors import ThreadPoolExecutor
from pyarrow import parquet

# In[2]:


# data directory for converted files (output to 4TB drive)
data_dir = pathlib.Path("/media/NVME_4TB/LINCS_cytotable_output/data")
data_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


preset = "cellprofiler_sqlite_cpg0016_jump"

# Start from the preset join string
joins = presets.config[preset]["CONFIG_JOINS"]

# Replace Metadata_Well and Metadata_Plate with Image_ prefix,
# include Image_Metadata_Col and Image_Count_Cells and
# add PathName columns
joins = (
    joins.replace("image.Metadata_Well,", "image.Image_Metadata_Well,")
    .replace("image.Metadata_Plate,", "image.Image_Metadata_Plate,")
    .replace(
        "Image_TableNumber,",
        "Image_TableNumber, Image_Metadata_Col, Image_Count_Cells, ",
    )
    .replace(
        "COLUMNS('Image_FileName_.*'),",
        "COLUMNS('Image_FileName_.*'),\n COLUMNS('Image_PathName_.*'),",
    )
)


# In[4]:


# process each plate from the manifest individually
for _, plate_name, plate_s3_path in pd.read_csv(
    "./manifest/lincs_cp_output_location_manifest.csv", header=0
).to_records():
    print("Processing plate ", plate_name)

    # create a folder for the plate
    plate_folder = pathlib.Path(f"{data_dir}/{plate_name}")
    plate_folder.mkdir(parents=True, exist_ok=True)

    cytotable_output_path = pathlib.Path(f"{plate_folder}/{plate_name}.parquet")

    # check if plate has already been processed
    if cytotable_output_path.is_file():
        print(f"Plate {plate_name} already processed, skipping.")
        continue

    # process plate using CytoTable
    cytotable_output_path = convert(
        source_path=plate_s3_path,
        dest_path=cytotable_output_path,
        dest_datatype="parquet",
        source_datatype="sqlite",
        chunk_size=8000,
        preset=preset,
        no_sign_request=True,
        local_cache_dir="./lincs_sqlite_s3_cache/",
        parsl_config=Config(
            executors=[ThreadPoolExecutor(label="tpe_for_lincs_processing")]
        ),
        sort_output=False,
        joins=joins,
    )

    # read only the metadata from parquet file
    meta = parquet.ParquetFile(cytotable_output_path).metadata
    print(
        "Finished processing plate",
        plate_name,
        "with output",
        cytotable_output_path,
        "which has shape (",
        meta.num_rows,
        ",",
        meta.num_columns,
        ").",
    )

# remove the SQLite plates only if at least one plate was processed
if not all(
    pathlib.Path(f"{data_dir}/{plate_name}/{plate_name}.parquet").is_file()
    for _, plate_name, _ in pd.read_csv(
        "./manifest/lincs_cp_output_location_manifest.csv", header=0
    ).to_records()
):
    shutil.rmtree("./lincs_sqlite_s3_cache/")
