#!/usr/bin/env python
# coding: utf-8

# # Preprocess the features into two sets of data per plate: Post- or Pre-QC
#
# The features will be preprocessed using [Pycytominer](https://github.com/cytomining/pycytominer).
# We use aggregate, annotate, normalize w/ MAD robustize for each plate.
# Then all plates are  merged together as one batch for feature selection and spherization.  # noqa: E501
#
# We generate these profiles for either the full profiles (pre-QC) and only the non-flagged cells (post-QC).  # noqa: E501

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from pycytominer import aggregate, annotate, feature_select, normalize
from pycytominer.cyto_utils import output

# ## Helper functions
#
# These functions comes from the LINCS profiling repository.

# In[2]:


def recode_dose(x: float, doses: list[float], return_level: bool = False) -> float:
    """Recode a dose value based on a list of predefined doses.

    Args:
        x (float): Dose value to be recoded.
        doses (list[float]): List of predefined dose values.
        return_level (bool, optional): If True, returns the level index (1-based)
            of the closest dose. If False, returns the closest dose value.
            Defaults to False.

    Returns:
        float: Either the closest dose value or its level index.
    """
    if np.isnan(x):
        return 0.0
    closest_index = np.argmin([np.abs(dose - x) for dose in doses])
    return float(closest_index + 1) if return_level else float(doses[closest_index])


def feature_selection(df_lvl4: pd.DataFrame, qc_status: str) -> pd.DataFrame:
    """
    Perform feature selection by dropping columns with null values
    (greater than 384 i.e. equivalent to one plate worth of cell profiles)
    and highly correlated values from the data.
    """
    PLATE_WELL_COUNT_THRESHOLD = 384  # Number of wells per plate
    metadata_columns = [x for x in df_lvl4.columns if (x.startswith("Metadata_"))]
    df_lvl4_metadata = df_lvl4[metadata_columns].copy()
    df_lvl4_features = df_lvl4.drop(metadata_columns, axis=1)
    null_cols = [
        col
        for col in df_lvl4_features.columns
        if df_lvl4_features[col].isnull().sum() > PLATE_WELL_COUNT_THRESHOLD
    ]
    df_lvl4_features.drop(null_cols, axis=1, inplace=True)

    for col in df_lvl4_features.columns:
        if df_lvl4_features[col].isnull().sum():
            df_lvl4_features[col].fillna(
                value=df_lvl4_features[col].mean(), inplace=True
            )

    if qc_status == "pre":
        meta_cols = [
            "Metadata_broad_sample",
            "Metadata_pert_id",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_broad_id",
            "Metadata_moa",
            "Metadata_dose_recode",
            "Metadata_sc_count",
        ]
    else:  # "post"
        meta_cols = [
            "Metadata_broad_sample",
            "Metadata_pert_id",
            "Metadata_Plate",
            "Metadata_Well",
            "Metadata_broad_id",
            "Metadata_moa",
            "Metadata_dose_recode",
            "Metadata_sc_count",
            "Metadata_sc_count_failed_qc",
            "Metadata_sc_count_passed_qc",
        ]
    df_meta_info = df_lvl4_metadata[meta_cols].copy()
    df_lvl4_new = pd.concat([df_meta_info, df_lvl4_features], axis=1)

    return df_lvl4_new


def merge_dataframe(
    df: pd.DataFrame, pertinfo_file: pathlib.Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function merge aligned L1000 and Cell painting Metadata information dataframe
    with the Level-4 data, change the values of the Metadata_dose_recode column
    and create a new column 'replicate_name' that represents each replicate in the
    dataset.
    """
    df_pertinfo = pd.read_csv(pertinfo_file)
    df_lvl4_new = df.merge(df_pertinfo, on="Metadata_broad_sample", how="outer")
    no_cpds_df = (
        df_lvl4_new[df_lvl4_new["pert_iname"].isnull()].copy().reset_index(drop=True)
    )
    df_lvl4_new.drop(
        df_lvl4_new[df_lvl4_new["pert_iname"].isnull()].index, inplace=True
    )
    df_lvl4_new.reset_index(drop=True, inplace=True)
    df_lvl4_new["Metadata_dose_recode"] = df_lvl4_new["Metadata_dose_recode"].map(
        {0.0: 0, 0.04: 1, 0.12: 2, 0.37: 3, 1.11: 4, 3.33: 5, 10.0: 6, 20.0: 7}
    )
    df_lvl4_new["replicate_name"] = [
        "replicate_" + str(x) for x in range(df_lvl4_new.shape[0])
    ]

    return df_lvl4_new, no_cpds_df


# ## Set constants

# In[3]:


# Set parameters for feature selection
feature_select_ops = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
]
na_cut = 0
corr_threshold = 0.95
full_blocklist_file = pathlib.Path("./utils/consensus_blocklist.txt").resolve(
    strict=True
)

# Set parameters for aggregation
aggregate_method = "median"
strata = [
    "Image_Metadata_Plate",
    "Image_Metadata_Well",
]
float_format = "%.5g"

# Set parameters for recoding the doses
primary_dose_mapping = [0, 0.04, 0.12, 0.37, 1.11, 3.33, 10, 20]

# Load in MOA file
moa_file = pathlib.Path(
    "./metadata/moa/repurposing_info_external_moa_map_resolved.tsv"
).resolve(strict=True)
moa_df = pd.read_csv(moa_file, sep="\t")

# Load in barcode platemap file
barcode_platemap_file = pathlib.Path("./metadata/barcode_platemap.csv").resolve(
    strict=True
)
barcode_platemap_df = pd.read_csv(barcode_platemap_file)

# Set path for output and input profiles main directory
profiles_dir = pathlib.Path("/home/jenna/mnt/bandicoot/LINCS_data/processed_profiles")

# Pertubation info file
pertinfo_file = pathlib.Path("./utils/aligned_moa_CP_L1000.csv").resolve(strict=True)

# Output path for single-cell profiles
output_dir = pathlib.Path(f"{profiles_dir}/single_cell_profiles")
output_dir.mkdir(parents=True, exist_ok=True)


# ## First, perform operations on each individual plate first
#
# 1. Aggregate (median)
# 2. Annotate
# 3. Normalize (MAD robustize) -> whole plate

# In[4]:


qc_profiles_dir = profiles_dir / "qc_profiles"
profile_files = qc_profiles_dir.rglob("*_qc_labeled.parquet")

# Lists to collect normalized data across all plates
all_pre_qc = []
all_post_qc = []

for profile_file in profile_files:
    plate_name = profile_file.stem.split("_")[0]
    print(f"Processing {plate_name}...")

    # Define all expected output files for this plate
    expected_outputs = [
        output_dir / f"{plate_name}_pre_qc_agg.parquet",
        output_dir / f"{plate_name}_post_qc_agg.parquet",
        output_dir / f"{plate_name}_pre_qc_agg_annotated.parquet",
        output_dir / f"{plate_name}_post_qc_agg_annotated.parquet",
        output_dir / f"{plate_name}_pre_qc_agg_normalized.parquet",
        output_dir / f"{plate_name}_post_qc_agg_normalized.parquet",
    ]

    # Skip plate if all outputs already exist
    if all(f.exists() for f in expected_outputs):
        print(f"Skipping {plate_name}, all outputs already exist.")
        continue

    # Load the profile data
    print("Loading profile data...")
    df = pd.read_parquet(profile_file, engine="pyarrow")

    # Drop columns with TableNumber in the name
    df = df.loc[:, ~df.columns.str.contains("TableNumber")]

    print("Starting aggregation...")
    # --- Pre-QC aggregation ---
    pre_qc_agg = aggregate(
        population_df=df,
        operation=aggregate_method,
        strata=strata,
        float_format=float_format,
    )

    # Add column for pre-QC data that says how many single-cells were in each well
    pre_qc_agg["Metadata_sc_count"] = pre_qc_agg["Image_Metadata_Well"].map(
        df.groupby("Image_Metadata_Well").size()
    )

    # --- Post-QC aggregation ---
    cqc_cols = [col for col in df.columns if col.startswith("cqc.")]
    post_qc_df = df[~df[cqc_cols].any(axis=1)]
    post_qc_agg = aggregate(
        population_df=post_qc_df,
        operation=aggregate_method,
        strata=strata,
        float_format=float_format,
    )

    # Add column for post-QC data that says how many single-cells were in each well
    # prior to QC filtering
    post_qc_agg["Metadata_sc_count"] = post_qc_agg["Image_Metadata_Well"].map(
        df.groupby("Image_Metadata_Well").size()
    )

    # Count failed QC cells per well
    post_qc_agg["Metadata_sc_count_failed_qc"] = post_qc_agg["Image_Metadata_Well"].map(
        df[df[cqc_cols].any(axis=1)].groupby("Image_Metadata_Well").size()
    )

    # Count passed QC cells per well
    post_qc_agg["Metadata_sc_count_passed_qc"] = post_qc_agg["Image_Metadata_Well"].map(
        df[~df[cqc_cols].any(axis=1)].groupby("Image_Metadata_Well").size()
    )

    output(
        df=pre_qc_agg,
        output_filename=output_dir / f"{plate_name}_pre_qc_agg.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    output(
        df=post_qc_agg,
        output_filename=output_dir / f"{plate_name}_post_qc_agg.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    del df, post_qc_df  # free memory

    print("Starting annotation...")
    # --- Annotate pre- and post-QC ---
    platemap_info = barcode_platemap_df.query("Assay_Plate_Barcode == @plate_name")
    if platemap_info.empty:
        raise FileNotFoundError(f"No platemap found for plate {plate_name}")
    txt_filename = platemap_info["Plate_Map_Name"].iloc[0]
    txt_path = pathlib.Path("./metadata/platemaps") / f"{txt_filename}.txt"
    platemap = pd.read_csv(txt_path, sep="\t")

    pre_annotated_df = annotate(
        profiles=pre_qc_agg,
        platemap=platemap,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
        float_format=float_format,
        format_broad_cmap=True,
        external_metadata=moa_df,
        external_join_left=["Metadata_broad_sample"],
        external_join_right=["Metadata_broad_sample"],
        cmap_args={"cell_id": "A549", "perturbation_mode": "chemical"},
    )

    post_annotated_df = annotate(
        profiles=post_qc_agg,
        platemap=platemap,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
        float_format=float_format,
        format_broad_cmap=True,
        external_metadata=moa_df,
        external_join_left=["Metadata_broad_sample"],
        external_join_right=["Metadata_broad_sample"],
        cmap_args={"cell_id": "A549", "perturbation_mode": "chemical"},
    )

    # Add dose recoding information to the annotated DataFrames
    pre_annotated_df = pre_annotated_df.assign(
        Metadata_dose_recode=(
            pre_annotated_df.Metadata_mmoles_per_liter.apply(
                lambda x: recode_dose(x, primary_dose_mapping, return_level=False)
            )
        )
    )

    post_annotated_df = post_annotated_df.assign(
        Metadata_dose_recode=(
            post_annotated_df.Metadata_mmoles_per_liter.apply(
                lambda x: recode_dose(x, primary_dose_mapping, return_level=False)
            )
        )
    )

    # Save the annotated DataFrames
    output(
        df=pre_annotated_df,
        output_filename=output_dir / f"{plate_name}_pre_qc_agg_annotated.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    output(
        df=post_annotated_df,
        output_filename=output_dir / f"{plate_name}_post_qc_agg_annotated.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    print("Starting normalization...")
    # --- Normalize pre- and post-QC ---
    pre_normalized_df = normalize(
        profiles=pre_annotated_df,
        samples="all",
        float_format=float_format,
        method="mad_robustize",
    )
    post_normalized_df = normalize(
        profiles=post_annotated_df,
        samples="all",
        float_format=float_format,
        method="mad_robustize",
    )

    # Append to the batch lists
    all_pre_qc.append(pre_normalized_df)
    all_post_qc.append(post_normalized_df)

    output(
        df=pre_normalized_df,
        output_filename=output_dir / f"{plate_name}_pre_qc_agg_normalized.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    output(
        df=post_normalized_df,
        output_filename=output_dir / f"{plate_name}_post_qc_agg_normalized.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    print(f"Finished processing {plate_name}.")


# In[5]:


# --- Merge all plates into single DataFrames for the batch ---
if all_pre_qc and all_post_qc:  # only run if both lists have data
    batch_pre_qc_df = pd.concat(all_pre_qc, ignore_index=True)
    batch_post_qc_df = pd.concat(all_post_qc, ignore_index=True)

    # Save the merged batch-level DataFrames
    output(
        df=batch_pre_qc_df,
        output_filename=output_dir / "whole_batch_pre_qc_norm.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    output(
        df=batch_post_qc_df,
        output_filename=output_dir / "whole_batch_post_qc_norm.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    print("Finished merging all plates into one batch.")
else:
    print("No new plates were processed. Skipping batch merge.")


# ## Perform preprocessing on merged data pre and post QC

# In[6]:


# --- Perform feature selection and spherization for whole batches ---
if all_pre_qc and all_post_qc:  # only run if both lists had data merged
    batch_pre_qc_fs_df = feature_select(
        profiles=batch_pre_qc_df,
        operation=feature_select_ops,
        na_cutoff=na_cut,
        corr_threshold=corr_threshold,
        blocklist_file=full_blocklist_file,
    )

    output(
        df=batch_pre_qc_fs_df,
        output_filename=output_dir / "whole_batch_pre_qc_agg_norm_fs.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    batch_post_qc_fs_df = feature_select(
        profiles=batch_post_qc_df,
        operation=feature_select_ops,
        na_cutoff=na_cut,
        corr_threshold=corr_threshold,
        blocklist_file=full_blocklist_file,
    )

    output(
        df=batch_post_qc_fs_df,
        output_filename=output_dir / "whole_batch_post_qc_agg_norm_fs.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    # --- Perform spherization for whole batches ---
    batch_pre_qc_spherized_df = normalize(
        profiles=batch_pre_qc_fs_df,
        features="infer",
        meta_features="infer",
        samples="Metadata_broad_sample == 'DMSO'",
        method="spherize",
    )

    output(
        df=batch_pre_qc_spherized_df,
        output_filename=output_dir / "whole_batch_pre_qc_agg_norm_fs_spherized.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    batch_post_qc_spherized_df = normalize(
        profiles=batch_post_qc_fs_df,
        features="infer",
        meta_features="infer",
        samples="Metadata_broad_sample == 'DMSO'",
        method="spherize",
    )

    output(
        df=batch_post_qc_spherized_df,
        output_filename=output_dir
        / "whole_batch_post_qc_agg_norm_fs_spherized.parquet",
        float_format=float_format,
        output_type="parquet",
    )

    print("Finished processing batch-level data.")
else:
    print(
        "No batch-level data to process. Skipping feature selection and spherization."
    )


# ## Output one dataframe to inspect

# In[7]:


# Path to the spherized file we want to inspect
spherized_file = output_dir / "whole_batch_post_qc_agg_norm_fs_spherized.parquet"

if spherized_file.exists():
    print("Spherized batch file already exists. Loading for inspection...")
    batch_post_qc_spherized_df = pd.read_parquet(spherized_file, engine="pyarrow")
    print(batch_post_qc_spherized_df.shape)
    display(batch_post_qc_spherized_df.head())  # noqa: F821
elif all_pre_qc and all_post_qc:  # only print after processing if new data exists
    print(batch_post_qc_spherized_df.shape)
    display(batch_post_qc_spherized_df.head())  # noqa: F821
else:
    print("No spherized batch data available to inspect.")


# ## Final preprocessing steps

# In[8]:


# Paths to the final merged files
pre_merged_file = output_dir / "whole_batch_pre_qc_cpd_replicates.parquet"
post_merged_file = output_dir / "whole_batch_post_qc_cpd_replicates.parquet"

# --- Perform feature selection and merge for pre-QC spherized batch ---
if pre_merged_file.exists():
    print("Pre-QC merged batch file already exists. Skipping processing.")
else:
    pre_selected_df = feature_selection(batch_pre_qc_spherized_df, qc_status="pre")
    pre_qc_cpd_replicates_merged_df, _ = merge_dataframe(pre_selected_df, pertinfo_file)
    output(
        df=pre_qc_cpd_replicates_merged_df,
        output_filename=pre_merged_file,
        float_format=float_format,
        output_type="parquet",
    )

# --- Perform feature selection and merge for post-QC spherized batch ---
if post_merged_file.exists():
    print("Post-QC merged batch file already exists. Skipping processing.")
else:
    post_selected_df = feature_selection(batch_post_qc_spherized_df, qc_status="post")
    post_qc_cpd_replicates_merged_df, _ = merge_dataframe(
        post_selected_df, pertinfo_file
    )
    output(
        df=post_qc_cpd_replicates_merged_df,
        output_filename=post_merged_file,
        float_format=float_format,
        output_type="parquet",
    )
