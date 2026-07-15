#!/bin/bash

#########################################################################
# This script runs single cell quality control using [`papermill`](https://papermill.readthedocs.io/en/latest/)
# on all plate folders found in the specified parent directory.

# To run the script, use the command:
# bash run_single_cell_qc.sh <parent_folder>
# where <parent_folder> is the path to the directory containing plate subdirectories.
#########################################################################

# Check if parent folder argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <parent_folder>"
    exit 1
fi

# Define the path to the parent folder to generate list of plate IDs
# (example: /media/NVME_4TB/LINCS_cytotable_output/data)
PARENT_FOLDER="$1"

# Create an array of folder names (excluding files)
plates=($(find "$PARENT_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

# Print the count of folders
echo "Number of plates found: ${#plates[@]}"

# Using papermill, run single cell quality control on all plates
for plate in "${plates[@]}"; do
    uv run papermill \
    2.single_cell_qc.ipynb \
    2.single_cell_qc.ipynb \
    -p plate_id $plate
done

echo "Single cell QC completed for all plates."
