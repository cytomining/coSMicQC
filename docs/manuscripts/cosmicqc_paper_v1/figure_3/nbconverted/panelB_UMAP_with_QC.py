#!/usr/bin/env python
# coding: utf-8

# ## Generate UMAP embeddings from data after QC and PyOD ECOD, and plot facet UMAP

# In[1]:


from pathlib import Path

import pandas as pd
import umap
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_point,
    ggplot,
    guide_legend,
    guides,
    labs,
    scale_color_brewer,
    theme,
    theme_bw,
)
from plotnine.options import set_option
from pycytominer.cyto_utils import infer_cp_features


# In[2]:


# Set constants
umap_random_seed = 0
umap_n_components = 2

# Set output directories
figure_dir = Path("./figures")
output_dir = Path("./umap_embeddings")
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


# Load in coSMicQC (post-QC) normalized dataframe for CFReT example plate
cosmicqc_df = pd.read_parquet(
    Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected.parquet"
    )
)

# Drop any NaN rows from columns that are not metadata (contain Metadata_ prefix)
cosmicqc_df = cosmicqc_df.dropna(
    subset=[col for col in cosmicqc_df.columns if not col.startswith("Metadata_")]
).reset_index(drop=True)

# Blind treatments for UMAP visualization
# (DMSO = control, TGFRi = treatment2, drug_X = treatment3)
treatment_mapping = {
    "DMSO": "control",
    "TGFRi": "treatment1",
    "drug_x": "treatment2",
}
cosmicqc_df["Metadata_treatment"] = cosmicqc_df["Metadata_treatment"].map(
    treatment_mapping
)

# Change cell type "failing" to "diseased" for better interpretability in the figure
cell_type_mapping = {"failing": "diseased"}
cosmicqc_df["Metadata_cell_type"] = cosmicqc_df["Metadata_cell_type"].map(
    lambda x: cell_type_mapping.get(x, x)
)

# Create new column for treatment cell type ID for each unique combo
cosmicqc_df["Metadata_Treatment_CellType_ID"] = (
    cosmicqc_df["Metadata_treatment"] + "_" + cosmicqc_df["Metadata_cell_type"]
)

# Update treatment-cell type IDs for plain English formatting
cosmicqc_df["Metadata_Treatment_CellType_ID"] = cosmicqc_df[
    "Metadata_Treatment_CellType_ID"
].replace(
    {
        "control_diseased": "DMSO-control diseased",
        "treatment1_diseased": "Treatment 1 diseased",
        "treatment2_diseased": "Treatment 2 diseased",
        "control_healthy": "DMSO-control healthy",
        "treatment1_healthy": "Treatment 1 healthy",
        "treatment2_healthy": "Treatment 2 healthy",
    }
)

# Print shape of the DataFrame
print(cosmicqc_df.shape)
cosmicqc_df.head()


# In[4]:


# Process cosmicqc_df to separate features and metadata
cp_features = infer_cp_features(cosmicqc_df)
meta_features = infer_cp_features(cosmicqc_df, metadata=True)

# Make sure to reinitialize UMAP instance per plate
umap_fit = umap.UMAP(
    random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
)

# Fit UMAP and convert to pandas DataFrame
embeddings = pd.DataFrame(
    umap_fit.fit_transform(cosmicqc_df.loc[:, cp_features]),
    columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
)
print(f"{embeddings.shape} UMAP embeddings generated")

# Combine with metadata
cosmicqc_umap_df = pd.concat([cosmicqc_df.loc[:, meta_features], embeddings], axis=1)

# Save UMAP with metadata DataFrame
cosmicqc_umap_df.to_parquet(output_dir / "post_QC_umap_embeddings.parquet")


# In[5]:


# Load in PyOD ECOD normalized dataframe for CFReT example plate
ecod_df = pd.read_parquet(
    Path(
        "./data/ecod_retransplant_norm_fs.parquet"
    )
)

# Drop any NaN rows from columns that are not metadata (contain Metadata_ or Image_ prefix)
ecod_df = ecod_df.dropna(
    subset=[
        col for col in ecod_df.columns if not col.startswith(("Metadata_", "Image_"))
    ]
).reset_index(drop=True)

# Blind treatments for UMAP visualization
ecod_df["Metadata_treatment"] = ecod_df["Metadata_treatment"].map(treatment_mapping)

# Change cell type "failing" to "diseased" for better interpretability in the figure
ecod_df["Metadata_cell_type"] = ecod_df["Metadata_cell_type"].map(
    lambda x: cell_type_mapping.get(x, x)
)

# Create new column for treatment cell type ID for each unique combo
ecod_df["Metadata_Treatment_CellType_ID"] = (
    ecod_df["Metadata_treatment"] + "_" + ecod_df["Metadata_cell_type"]
)

# Update treatment-cell type IDs for plain English formatting
ecod_df["Metadata_Treatment_CellType_ID"] = ecod_df[
    "Metadata_Treatment_CellType_ID"
].replace(
    {
        "control_diseased": "DMSO-control diseased",
        "treatment1_diseased": "Treatment 1 diseased",
        "treatment2_diseased": "Treatment 2 diseased",
        "control_healthy": "DMSO-control healthy",
        "treatment1_healthy": "Treatment 1 healthy",
        "treatment2_healthy": "Treatment 2 healthy",
    }
)

# Print shape of the DataFrame
print(ecod_df.shape)
ecod_df.head()


# In[6]:


# Process ecod_df to separate features and metadata
ecod_cp_features = infer_cp_features(ecod_df)
ecod_meta_features = infer_cp_features(ecod_df, metadata=True)

# Make sure to reinitialize UMAP instance per plate
ecod_umap_fit = umap.UMAP(
    random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
)

# Fit UMAP and convert to pandas DataFrame
ecod_embeddings = pd.DataFrame(
    ecod_umap_fit.fit_transform(ecod_df.loc[:, ecod_cp_features]),
    columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
)
print(f"{ecod_embeddings.shape} UMAP embeddings generated")

# Combine with metadata
ecod_umap_df = pd.concat([ecod_df.loc[:, ecod_meta_features], ecod_embeddings], axis=1)

# Save UMAP with metadata DataFrame
ecod_umap_df.to_parquet(output_dir / "post_ECOD_umap_embeddings.parquet")


# In[7]:


# Tag each dataframe with its QC method
cosmicqc_umap_df = cosmicqc_umap_df.copy()
cosmicqc_umap_df["Metadata_QC_method"] = "coSMicQC"

ecod_umap_df = ecod_umap_df.copy()
ecod_umap_df["Metadata_QC_method"] = "PyOD ECOD"

# Combine both into one dataframe for faceting
combined_umap_df = pd.concat(
    [cosmicqc_umap_df, ecod_umap_df],
    axis=0,
    ignore_index=True,
)

# Force facet order: coSMicQC first, then PyOD ECOD
combined_umap_df["Metadata_QC_method"] = pd.Categorical(
    combined_umap_df["Metadata_QC_method"],
    categories=["coSMicQC", "PyOD ECOD"],
    ordered=True,
)

# Set the figure size (widened slightly to accommodate the right-side legend)
height = 8
width = 18
set_option("figure_size", (width, height))

# Plot with custom color palette, faceted by QC method
p = (
    ggplot(
        combined_umap_df,
        aes(x="UMAP0", y="UMAP1", color="Metadata_Treatment_CellType_ID"),
    )
    + labs(
        color="Cell type & treatment",
    )
    + geom_point(alpha=0.2, size=2)
    + facet_wrap("~Metadata_QC_method", scales="free")
    + theme_bw()
    + theme(
        axis_title=element_text(size=32),
        axis_text=element_text(size=26),
        legend_title=element_text(size=28),
        legend_text=element_text(size=24),
        legend_position="right",
        strip_text=element_text(size=20),  # Adjust facet label size
    )
    + scale_color_brewer(type="qual", palette="Dark2")
    + guides(
        color=guide_legend(
            override_aes={
                "alpha": 1,
                "size": 5,
            },
            ncol=1,  # single column, since legend is now on the right
        )
    )
)

# Save the plot
p.save(
    figure_dir / "facet_umap_coSMicQC_PyOD_ECOD_plot.png",
    dpi=200,
    width=width,
    height=height,
)

p.show()

