#!/usr/bin/env python
# coding: utf-8

# # Preprocess the features into two sets of data per plate: Post- or Pre-QC
# 
# The features will be preprocessed using [Pycytominer](https://github.com/cytomining/pycytominer).
# We use aggregate, annotate, normalize w/ MAD robustize for each plate.
# Then all plates are  merged together as one batch for feature selection and spherization.
# 
# We generate these profiles for either the full profiles (pre-QC) and only the non-flagged cells (post-QC).
# 
# This code was updated from the original code from the LINCS paper: https://github.com/broadinstitute/lincs-cell-painting

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

# Perturbation info file
pertinfo_file = pathlib.Path("./utils/aligned_moa_CP_L1000.csv").resolve(strict=True)

# Output path for single-cell profiles
output_dir = pathlib.Path(f"{profiles_dir}/single_cell_profiles")
output_dir.mkdir(parents=True, exist_ok=True)

# QC filtering mode: either keep post-QC passing cells or only failing cells
qc_filter_mode = "only_failed_cells"  # options: "post" or "only_failed_cells"
if qc_filter_mode not in {"post", "only_failed_cells"}:
    raise ValueError(
        "qc_filter_mode must be one of {'post', 'only_failed_cells'}"
    )
qc_suffix = "" if qc_filter_mode == "post" else "_only_failing_cells"
qc_filter_description = (
    "passing cells (post-QC)"
    if qc_filter_mode == "post"
    else "only failing cells"
)


# ## First, perform operations on each individual plate first
# 
# 1. Aggregate (median)
# 2. Annotate
# 3. Normalize (MAD robustize) -> whole plate

# In[4]:


qc_profiles_dir = profiles_dir / "qc_profiles"
profile_files = list(qc_profiles_dir.rglob("*_qc_labeled.parquet"))

# Separate lists for pre and post QC
all_pre_qc = []
all_post_qc = []

# ========== PRE-QC PROCESSING ==========
print("\n" + "="*50)
print("PHASE 1: PRE-QC PROCESSING")
print("="*50)

for profile_file in profile_files:
    plate_name = profile_file.stem.split("_")[0]
    print(f"\nProcessing {plate_name} (PRE-QC)...")

    pre_agg_path = output_dir / f"{plate_name}_pre_qc_agg.parquet"
    pre_annotated_path = output_dir / f"{plate_name}_pre_qc_agg_annotated.parquet"
    pre_normalized_path = output_dir / f"{plate_name}_pre_qc_agg_normalized.parquet"

    # Skip only if ALL outputs exist
    if (
        pre_agg_path.exists()
        and pre_annotated_path.exists()
        and pre_normalized_path.exists()
    ):
        print(f"  Skipping {plate_name}: all pre-QC outputs already exist.")
        continue

    # If nothing has been processed, then run
    print("  Loading profile data...")
    df = pd.read_parquet(profile_file, engine="pyarrow")
    df = df.loc[:, ~df.columns.str.contains("TableNumber")]

    print("  Starting aggregation...")
    pre_qc_agg = aggregate(
        population_df=df,
        operation=aggregate_method,
        strata=strata,
        float_format=float_format,
    )
    pre_qc_agg["Metadata_sc_count"] = pre_qc_agg["Image_Metadata_Well"].map(
        df.groupby("Image_Metadata_Well").size()
    )

    output(
        df=pre_qc_agg,
        output_filename=pre_agg_path,
        float_format=float_format,
        output_type="parquet",
    )

    print("  Starting annotation...")
    platemap_info = barcode_platemap_df.query("Assay_Plate_Barcode == @plate_name")
    if platemap_info.empty:
        raise FileNotFoundError(f"No platemap found for plate {plate_name}")
    txt_filename = platemap_info["Plate_Map_Name"].iloc[0]
    txt_path = pathlib.Path("./metadata/platemaps") / f"{txt_filename}.txt"
    platemap = pd.read_csv(txt_path, sep="	")

    pre_annotated_df = annotate(
        profiles=pre_qc_agg,
        platemap=platemap,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
        float_format=float_format,
        format_broad_cmap=True,
        external_metadata=moa_df,
        external_join_on=["Metadata_broad_sample"],
        cmap_args={"cell_id": "A549", "perturbation_mode": "chemical"},
    )

    pre_annotated_df = pre_annotated_df.assign(
        Metadata_dose_recode=(
            pre_annotated_df.Metadata_mmoles_per_liter.apply(
                lambda x: recode_dose(x, primary_dose_mapping, return_level=False)
            )
        )
    )

    output(
        df=pre_annotated_df,
        output_filename=pre_annotated_path,
        float_format=float_format,
        output_type="parquet",
    )

    print("  Starting normalization...")
    pre_normalized_df = normalize(
        profiles=pre_annotated_df,
        samples="all",
        float_format=float_format,
        method="mad_robustize",
    )

    output(
        df=pre_normalized_df,
        output_filename=pre_normalized_path,
        float_format=float_format,
        output_type="parquet",
    )

    all_pre_qc.append(pre_normalized_df)
    print(f"  Finished processing {plate_name} (PRE-QC).")


# In[5]:


# ========== POST-QC PROCESSING ==========
print("\n" + "="*50)
print("PHASE 2: POST-QC PROCESSING")
print(f"Mode: {qc_filter_mode} {qc_filter_description}")
print("="*50)

for profile_file in profile_files:
    plate_name = profile_file.stem.split("_")[0]
    print(f"\nProcessing {plate_name} (POST-QC{qc_suffix})...")

    post_agg_path = output_dir / f"{plate_name}_post_qc_agg{qc_suffix}.parquet"
    post_annotated_path = output_dir / f"{plate_name}_post_qc_agg_annotated{qc_suffix}.parquet"
    post_normalized_path = output_dir / f"{plate_name}_post_qc_agg_normalized{qc_suffix}.parquet"

    # Skip only if ALL outputs exist
    if (
        post_agg_path.exists()
        and post_annotated_path.exists()
        and post_normalized_path.exists()
    ):
        print(f"  Skipping {plate_name}: all post-QC outputs already exist.")
        continue

    print("  Loading profile data...")
    df = pd.read_parquet(profile_file, engine="pyarrow")
    df = df.loc[:, ~df.columns.str.contains("TableNumber")]

    print("  Starting aggregation...")
    cqc_cols = [col for col in df.columns if col.startswith("cqc.")]
    post_qc_df = (
        df[~df[cqc_cols].any(axis=1)]
        if qc_filter_mode == "post"
        else df[df[cqc_cols].any(axis=1)]
    )
    print(f"  Filtering for {qc_filter_description}...")

    post_qc_agg = aggregate(
        population_df=post_qc_df,
        operation=aggregate_method,
        strata=strata,
        float_format=float_format,
    )

    post_qc_agg["Metadata_sc_count"] = post_qc_agg["Image_Metadata_Well"].map(
        df.groupby("Image_Metadata_Well").size()
    )
    post_qc_agg["Metadata_sc_count_failed_qc"] = post_qc_agg["Image_Metadata_Well"].map(
        df[df[cqc_cols].any(axis=1)].groupby("Image_Metadata_Well").size()
    )
    post_qc_agg["Metadata_sc_count_passed_qc"] = post_qc_agg["Image_Metadata_Well"].map(
        df[~df[cqc_cols].any(axis=1)].groupby("Image_Metadata_Well").size()
    )

    output(
        df=post_qc_agg,
        output_filename=post_agg_path,
        float_format=float_format,
        output_type="parquet",
    )

    print("  Starting annotation...")

    platemap_info = barcode_platemap_df.query(
        "Assay_Plate_Barcode == @plate_name"
    )
    if platemap_info.empty:
        raise FileNotFoundError(
            f"No platemap found for plate {plate_name}"
        )

    txt_filename = platemap_info["Plate_Map_Name"].iloc[0]
    txt_path = pathlib.Path("./metadata/platemaps") / f"{txt_filename}.txt"
    platemap = pd.read_csv(txt_path, sep="\t")

    post_annotated_df = annotate(
        profiles=post_qc_agg,
        platemap=platemap,
        join_on=["Metadata_well_position", "Image_Metadata_Well"],
        float_format=float_format,
        format_broad_cmap=True,
        external_metadata=moa_df,
        external_join_on=["Metadata_broad_sample"],
        cmap_args={
            "cell_id": "A549",
            "perturbation_mode": "chemical",
        },
    )

    post_annotated_df = post_annotated_df.assign(
        Metadata_dose_recode=(
            post_annotated_df.Metadata_mmoles_per_liter.apply(
                lambda x: recode_dose(
                    x,
                    primary_dose_mapping,
                    return_level=False,
                )
            )
        )
    )

    output(
        df=post_annotated_df,
        output_filename=post_annotated_path,
        float_format=float_format,
        output_type="parquet",
    )

    print("  Starting normalization...")

    post_normalized_df = normalize(
        profiles=post_annotated_df,
        samples="all",
        float_format=float_format,
        method="mad_robustize",
    )

    output(
        df=post_normalized_df,
        output_filename=post_normalized_path,
        float_format=float_format,
        output_type="parquet",
    )

    all_post_qc.append(post_normalized_df)

    print(f"  Finished processing {plate_name} (POST-QC).")

print(f"\nPre-QC plates collected: {len(all_pre_qc)}")
print(f"Post-QC plates collected: {len(all_post_qc)}")


# In[6]:


# --- Merge all plates into single DataFrames for the batch ---
if all_pre_qc:
    print("\nMerging pre-QC plates into single batch...")
    batch_pre_qc_df = pd.concat(all_pre_qc, ignore_index=True)
    output(
        df=batch_pre_qc_df,
        output_filename=output_dir / "whole_batch_pre_qc_norm.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    print("Finished merging pre-QC plates.")
else:
    print("No pre-QC plates to merge.")
    batch_pre_qc_df = None

if all_post_qc:
    print(f"\nMerging post-QC plates into single batch ({qc_filter_description})...")
    batch_post_qc_df = pd.concat(all_post_qc, ignore_index=True)
    output(
        df=batch_post_qc_df,
        output_filename=output_dir / f"whole_batch_post_qc_norm{qc_suffix}.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    print("Finished merging post-QC plates.")
else:
    print("No post-QC plates to merge.")
    batch_post_qc_df = None


# ## Perform preprocessing on merged data pre and post QC

# In[7]:


# --- Perform feature selection and spherization for whole batches ---
# PRE-QC BATCH PROCESSING
if batch_pre_qc_df is not None:
    print("\n" + "="*50)
    print("BATCH PRE-QC PROCESSING")
    print("="*50)
    
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
    print("Finished pre-QC batch processing.")
else:
    print("No pre-QC batch data to process.")
    batch_pre_qc_spherized_df = None

# POST-QC BATCH PROCESSING
if batch_post_qc_df is not None:
    print("\n" + "="*50)
    print(f"BATCH POST-QC PROCESSING ({qc_filter_description})")
    print("="*50)
    
    batch_post_qc_fs_df = feature_select(
        profiles=batch_post_qc_df,
        operation=feature_select_ops,
        na_cutoff=na_cut,
        corr_threshold=corr_threshold,
        blocklist_file=full_blocklist_file,
    )

    output(
        df=batch_post_qc_fs_df,
        output_filename=output_dir / f"whole_batch_post_qc_agg_norm_fs{qc_suffix}.parquet",
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
        / f"whole_batch_post_qc_agg_norm_fs_spherized{qc_suffix}.parquet",
        float_format=float_format,
        output_type="parquet",
    )
    print("Finished post-QC batch processing.")
else:
    print("No post-QC batch data to process.")
    batch_post_qc_spherized_df = None


# ## Output one dataframe to inspect

# In[8]:


# Path to the spherized file we want to inspect
spherized_file = output_dir / f"whole_batch_post_qc_agg_norm_fs_spherized{qc_suffix}.parquet"

if spherized_file.exists():
    print("Spherized batch file already exists. Loading for inspection...")
    batch_post_qc_spherized_df = pd.read_parquet(spherized_file, engine="pyarrow")
    print(batch_post_qc_spherized_df.shape)
    display(batch_post_qc_spherized_df.head())
elif all_pre_qc and all_post_qc:  # only print after processing if new data exists
    print(batch_post_qc_spherized_df.shape)
    display(batch_post_qc_spherized_df.head())
else:
    print("No spherized batch data available to inspect.")


# ## Final preprocessing steps

# In[9]:


# --- Final preprocessing: feature selection and merge ---

# PRE-QC FINAL PROCESSING
print("\n" + "="*50)
print("FINAL PRE-QC PROCESSING")
print("="*50)

pre_merged_file = output_dir / "whole_batch_pre_qc_cpd_replicates.parquet"
if batch_pre_qc_spherized_df is not None:
    if pre_merged_file.exists():
        print("Pre-QC final file already exists. Skipping.")
    else:
        print("Processing pre-QC spherized batch...")
        pre_selected_df = feature_selection(batch_pre_qc_spherized_df, qc_status="pre")
        pre_qc_cpd_replicates_merged_df, _ = merge_dataframe(pre_selected_df, pertinfo_file)
        output(
            df=pre_qc_cpd_replicates_merged_df,
            output_filename=pre_merged_file,
            float_format=float_format,
            output_type="parquet",
        )
        print("Finished pre-QC final processing.")
else:
    print("No pre-QC spherized data to process.")

# POST-QC FINAL PROCESSING
print("\n" + "="*50)
print(f"FINAL POST-QC PROCESSING ({qc_filter_description})")
print("="*50)

post_merged_file = output_dir / f"whole_batch_post_qc_cpd_replicates{qc_suffix}.parquet"
if batch_post_qc_spherized_df is not None:
    if post_merged_file.exists():
        print("Post-QC final file already exists. Skipping.")
    else:
        print("Processing post-QC spherized batch...")
        post_selected_df = feature_selection(batch_post_qc_spherized_df, qc_status="post")
        post_qc_cpd_replicates_merged_df, _ = merge_dataframe(post_selected_df, pertinfo_file)
        output(
            df=post_qc_cpd_replicates_merged_df,
            output_filename=post_merged_file,
            float_format=float_format,
            output_type="parquet",
        )
        print("Finished post-QC final processing.")
else:
    print("No post-QC spherized data to process.")

