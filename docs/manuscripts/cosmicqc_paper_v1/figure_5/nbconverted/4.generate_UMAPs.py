#!/usr/bin/env python
# coding: utf-8

# # Generate UMAP embeddings and plots
# 
# We replicate the same process of generating UMAP embeddings that was done in the original LINCS paper, but with and without QC.

# In[1]:


import pathlib
import numpy as np
import pandas as pd
import umap

from sklearn.decomposition import PCA
from pycytominer.cyto_utils import infer_cp_features

from plotnine import (
    ggplot,
    aes,
    geom_point,
    ggtitle,
    xlab,
    ylab,
    guides,
    guide_legend,
    theme_bw,
    scale_size_manual,
    scale_alpha_manual,
    scale_color_manual,
    scale_color_gradient,
    scale_shape_manual,
)


# In[2]:


np.random.seed(42)


# In[3]:


# Output file info
output_dir = pathlib.Path("./embeddings")
output_dir.mkdir(parents=True, exist_ok=True)
pre_qc_output_file = pathlib.Path(f"{output_dir}/whole_batch_pre_qc_embeddings.parquet")
post_qc_output_file = pathlib.Path(
    f"{output_dir}/whole_batch_post_qc_embeddings.parquet"
)

# Figure output directory
figure_output_dir = pathlib.Path("./figures")
figure_output_dir.mkdir(parents=True, exist_ok=True)

# Input path for single-cell profiles
input_dir = pathlib.Path(
    "/home/jenna/mnt/bandicoot/LINCS_data/processed_profiles/single_cell_profiles"
)


# ## Load in pre-QC and post-QC profiles

# In[4]:


# Load in pre-QC cell painting profile
pre_qc_file = pathlib.Path(input_dir, "whole_batch_pre_qc_cpd_replicates.parquet")
pre_qc_df = pd.read_parquet(pre_qc_file)

# Load in post-QC cell painting profile
post_qc_file = pathlib.Path(input_dir, "whole_batch_post_qc_cpd_replicates.parquet")
post_qc_df = pd.read_parquet(post_qc_file)

# Align "Metadata_sc_count_failed_qc" from post-QC df to pre-QC df using
# Metadata_Plate and Metadata_Well
pre_qc_df = pre_qc_df.merge(
    post_qc_df[["Metadata_Plate", "Metadata_Well", "Metadata_sc_count_failed_qc"]],
    on=["Metadata_Plate", "Metadata_Well"],
    how="left",
    suffixes=("", "_postqc"),
)
pre_qc_df["Metadata_sc_count_failed_qc"] = pre_qc_df[
    "Metadata_sc_count_failed_qc"
].fillna(0)

# Print out pre_qc_df shape and head
print(f"Pre-QC df shape: {pre_qc_df.shape}")
pre_qc_df.head()


# In[5]:


# Confirm both dataframes have the single cell count columns
assert "Metadata_sc_count_failed_qc" in pre_qc_df.columns
assert "Metadata_sc_count" in pre_qc_df.columns
assert "Metadata_sc_count_failed_qc" in post_qc_df.columns
assert "Metadata_sc_count" in post_qc_df.columns


# ## Pre-QC UMAPs (Panels A and B)

# In[6]:


cp_features = infer_cp_features(pre_qc_df)
meta_features = [
    *infer_cp_features(pre_qc_df, metadata=True),
    "broad_id",
    "pert_iname",
    "moa",
    "replicate_name",
]

# Transform PCA to top 50 components
n_components = 50
pca = PCA(n_components=n_components)

pre_qc_pca_df = pca.fit_transform(pre_qc_df.loc[:, cp_features])
pre_qc_pca_df = pd.DataFrame(pre_qc_pca_df)
pre_qc_pca_df.columns = [f"PCA_{x}" for x in range(0, n_components)]

print(pre_qc_pca_df.shape)
pre_qc_pca_df.head()


# In[7]:


# Fit UMAP directly on CellProfiler features
reducer = umap.UMAP(random_state=123, min_dist=0.1, n_neighbors=20, metric="euclidean")
pre_qc_embedding_df = reducer.fit_transform(
    pre_qc_pca_df.drop(["PCA_0"], axis="columns")
)

pre_qc_embedding_df = pd.DataFrame(pre_qc_embedding_df)
pre_qc_embedding_df.columns = ["UMAP_0", "UMAP_1"]
pre_qc_embedding_df = pd.concat(
    [pre_qc_df.loc[:, meta_features], pre_qc_embedding_df], axis="columns"
)

pre_qc_embedding_df.head()


# In[8]:


# --- Select MOA targets (key = lowercase name, value = display label) ---
moa_targets = {
    "aurora kinase inhibitor": "Aurora kinase inhibitor",
    "plk inhibitor": "PLK inhibitor",
    "proteasome inhibitor": "Proteasome inhibitor",
    "exportin antagonist": "Exportin antagonist",
    "maternal embryonic leucine zipper kinase inhibitor": "MELK inhibitor",
    "tubulin inhibitor": "Tubulin inhibitor",
    "hsp inhibitor": "HSP inhibitor",
    "xiap inhibitor": "XIAP inhibitor",
    "other": "Other",
}

# Add DMSO label to dataframe
pre_qc_embedding_df = pre_qc_embedding_df.assign(dmso_label="DMSO")
pre_qc_embedding_df.loc[
    pre_qc_embedding_df.Metadata_broad_sample != "DMSO", "dmso_label"
] = "compound"

# --- Create compounds to highlight ---
# Use display labels directly (values of the dict)
moa_labels = list(moa_targets.values())

moa_targets_size_values = dict.fromkeys(moa_labels[:-1], 1)
moa_targets_size_values[moa_labels[-1]] = 0.1

moa_targets_alpha_values = dict.fromkeys(moa_labels[:-1], 0.5)
moa_targets_alpha_values[moa_labels[-1]] = 0.1

# --- Add highlight_moa column ---
pre_qc_embedding_df = pre_qc_embedding_df.assign(
    highlight_moa=pre_qc_embedding_df["moa"].str.lower().map(moa_targets)
)
pre_qc_embedding_df["highlight_moa"] = pre_qc_embedding_df["highlight_moa"].fillna(
    "Other"
)

pre_qc_embedding_df.head()


# In[9]:


# Define colors for MOA categories
moa_colors = {
    "Aurora kinase inhibitor": "#ff75cf",
    "PLK inhibitor": "#332288",
    "Proteasome inhibitor": "#117733",
    "Exportin antagonist": "#88CCEE",
    "MELK inhibitor": "#4bf507",
    "Tubulin inhibitor": "#CC6677",
    "HSP inhibitor": "#FF9A00",
    "XIAP inhibitor": "#882255",
    "Other": "grey",
}

# 1. Create columns for plotting
pre_qc_embedding_df["is_control"] = pre_qc_embedding_df["dmso_label"] == "DMSO"

# Make a plotting MOA column so DMSO is separate from "Other"
pre_qc_embedding_df["plot_moa"] = pre_qc_embedding_df["highlight_moa"]
pre_qc_embedding_df.loc[pre_qc_embedding_df["dmso_label"] == "DMSO", "plot_moa"] = (
    "DMSO"
)

# Split datasets for layering
other_df = pre_qc_embedding_df.query("highlight_moa == 'Other' & dmso_label != 'DMSO'")
main_df = pre_qc_embedding_df.query("highlight_moa != 'Other' | dmso_label == 'DMSO'")

# 2. Build plot
p = (
    ggplot()
    # Layer 1: Other points in back, faint and small
    + geom_point(
        data=other_df,
        mapping=aes(x="UMAP_0", y="UMAP_1"),
        color=moa_colors["Other"],
        alpha=0.1,
        size=1,
        show_legend=False,
    )
    # Layer 2: All other MOAs + DMSO for correct legends
    + geom_point(
        data=main_df,
        mapping=aes(x="UMAP_0", y="UMAP_1", color="plot_moa", shape="is_control"),
    )
    # Titles and labels
    + xlab("UMAP_0")
    + ylab("UMAP_1")
    # Scales
    + scale_color_manual(name="MOA", values={**moa_colors, "DMSO": "#D80000"})
    + scale_shape_manual(name="Control", values={True: "+", False: "o"})
    # Guides
    + guides(
        shape=guide_legend(override_aes={"alpha": 1, "size": 2}),
        color=guide_legend(override_aes={"alpha": 1, "size": 2}),
    )
    # Theme
    + theme_bw()
)
p.save(filename=figure_output_dir / "pre_qc_moa_umap.png", dpi=600)
p.show()


# In[10]:


pre_qc_embedding_df["prop_failed_qc"] = (
    pre_qc_embedding_df["Metadata_sc_count_failed_qc"]
    / pre_qc_embedding_df["Metadata_sc_count"]
)
# Drop any rows with NaN in the proportion column
pre_qc_embedding_df = pre_qc_embedding_df.dropna(subset=["prop_failed_qc"])

p_failed_qc = (
    ggplot(pre_qc_embedding_df, aes(x="UMAP_0", y="UMAP_1", color="prop_failed_qc"))
    + geom_point(size=1, alpha=0.5)
    + xlab("UMAP_0")
    + ylab("UMAP_1")
    + scale_color_gradient(name="Proportion\nFailed QC", low="#56B1F7", high="#CA0020")
    + theme_bw()
)
p_failed_qc.save(filename=figure_output_dir / "pre_qc_failed_qc_umap.png", dpi=600)
p_failed_qc.show()


# ## UMAP for post-QC

# In[11]:


cp_features = infer_cp_features(post_qc_df)
meta_features = [
    *infer_cp_features(post_qc_df, metadata=True),
    "broad_id",
    "pert_iname",
    "moa",
    "replicate_name",
]

# Transform PCA to top 50 components
n_components = 50
pca = PCA(n_components=n_components)

post_qc_pca_df = pca.fit_transform(post_qc_df.loc[:, cp_features])
post_qc_pca_df = pd.DataFrame(post_qc_pca_df)
post_qc_pca_df.columns = [f"PCA_{x}" for x in range(0, n_components)]

print(post_qc_pca_df.shape)
post_qc_pca_df.head()


# In[12]:


# Fit UMAP
reducer = umap.UMAP(random_state=123, min_dist=0.1, n_neighbors=20, metric="euclidean")
post_qc_embedding_df = reducer.fit_transform(
    post_qc_pca_df.drop(["PCA_0"], axis="columns")
)

post_qc_embedding_df = pd.DataFrame(post_qc_embedding_df)
post_qc_embedding_df.columns = ["UMAP_0", "UMAP_1"]
post_qc_embedding_df = pd.concat(
    [post_qc_df.loc[:, meta_features], post_qc_embedding_df], axis="columns"
)

post_qc_embedding_df.head()


# In[13]:


# --- Select MOA targets (key = lowercase name, value = display label) ---
moa_targets = {
    "aurora kinase inhibitor": "Aurora kinase inhibitor",
    "plk inhibitor": "PLK inhibitor",
    "proteasome inhibitor": "Proteasome inhibitor",
    "exportin antagonist": "Exportin antagonist",
    "maternal embryonic leucine zipper kinase inhibitor": "MELK inhibitor",
    "tubulin inhibitor": "Tubulin inhibitor",
    "hsp inhibitor": "HSP inhibitor",
    "xiap inhibitor": "XIAP inhibitor",
    "other": "Other",
}

# Add DMSO label to dataframe
post_qc_embedding_df = post_qc_embedding_df.assign(dmso_label="DMSO")
post_qc_embedding_df.loc[
    post_qc_embedding_df.Metadata_broad_sample != "DMSO", "dmso_label"
] = "compound"

# --- Create compounds to highlight ---
# Use display labels directly (values of the dict)
moa_labels = list(moa_targets.values())

moa_targets_size_values = dict.fromkeys(moa_labels[:-1], 1)
moa_targets_size_values[moa_labels[-1]] = 0.1

moa_targets_alpha_values = dict.fromkeys(moa_labels[:-1], 0.5)
moa_targets_alpha_values[moa_labels[-1]] = 0.1

# --- Add highlight_moa column ---
post_qc_embedding_df = post_qc_embedding_df.assign(
    highlight_moa=post_qc_embedding_df["moa"].str.lower().map(moa_targets)
)
post_qc_embedding_df["highlight_moa"] = post_qc_embedding_df["highlight_moa"].fillna(
    "Other"
)


# In[14]:


# 1. Create columns for plotting
post_qc_embedding_df["is_control"] = post_qc_embedding_df["dmso_label"] == "DMSO"

# Make a plotting MOA column so DMSO is separate from "Other"
post_qc_embedding_df["plot_moa"] = post_qc_embedding_df["highlight_moa"]
post_qc_embedding_df.loc[post_qc_embedding_df["dmso_label"] == "DMSO", "plot_moa"] = (
    "DMSO"
)

# Split datasets for layering
other_df = post_qc_embedding_df.query("highlight_moa == 'Other' & dmso_label != 'DMSO'")
main_df = post_qc_embedding_df.query("highlight_moa != 'Other' | dmso_label == 'DMSO'")

# 2. Build plot
p = (
    ggplot()
    # Layer 1: Other points in back, faint and small
    + geom_point(
        data=other_df,
        mapping=aes(x="UMAP_0", y="UMAP_1"),
        color=moa_colors["Other"],
        alpha=0.1,
        size=1,
        show_legend=False,
    )
    # Layer 2: All other MOAs + DMSO for correct legends
    + geom_point(
        data=main_df,
        mapping=aes(x="UMAP_0", y="UMAP_1", color="plot_moa", shape="is_control"),
    )
    # Titles and labels
    + xlab("UMAP_0")
    + ylab("UMAP_1")
    # Scales
    + scale_color_manual(name="MOA", values={**moa_colors, "DMSO": "#D80000"})
    + scale_shape_manual(name="Control", values={True: "+", False: "o"})
    # Guides
    + guides(
        shape=guide_legend(override_aes={"alpha": 1, "size": 2}),
        color=guide_legend(override_aes={"alpha": 1, "size": 2}),
    )
    # Theme
    + theme_bw()
)
p.save(filename=figure_output_dir / "post_qc_moa_umap.png", dpi=600)
p.show()

