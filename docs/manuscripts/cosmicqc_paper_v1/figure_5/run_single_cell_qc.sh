#!/bin/bash

#########################################################################
# This script runs single cell quality control using papermill
# on all plate folders found in the specified parent directory.
#########################################################################

# Define the path to the parent folder to generate list of plate IDs
PARENT_FOLDER="/media/NVME_4TB/LINCS_cytotable_output/data/"

# Create an array of folder names (excluding files)
plates=($(find "$PARENT_FOLDER" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;))

# Print the count of folders
echo "Number of plates found: ${#plates[@]}"

# Using papermill, run single cell quality control on all plates
for plate in "${plates[@]}"; do
    poetry run papermill \
    2.single_cell_qc.ipynb \
    2.single_cell_qc.ipynb \
    -p plate_id $plate
done

echo "Single cell QC completed for all plates."
