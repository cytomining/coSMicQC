#!/usr/bin/env python
# coding: utf-8

# # Compute HDSBCAN clusters for pre- and post-QC feature spaces
#
# We apply PCA first to the feature spaces before computing HDBSCAN.

# In[1]:


import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_bar,
    ggplot,
    labs,
    position_stack,
    scale_fill_manual,
    theme,
)
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA

# Ignore warning about deprecated argument name in sklearn
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")


# In[2]:


# Output dir for figure
output_dir = pathlib.Path("./figures")
output_dir.mkdir(exist_ok=True, parents=True)

# Load in pre- and post-QC feature selected profiles
pre_QC_df = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected_no_QC.parquet"
    )
)
post_QC_df = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected.parquet"
    )
)

# Add QC status to pre_QC_df based on matching rows in post_QC_df using metadata
# Set all rows to failed first
pre_QC_df["Metadata_QC_status"] = "failed"
# Find rows in pre_QC_df that are in post_QC_df based on metadata set to passing
metadata_cols = [
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]
# Create a MultiIndex for fast lookup
qc_index = post_QC_df.set_index(metadata_cols).index
mask = pre_QC_df.set_index(metadata_cols).index.isin(qc_index)
pre_QC_df.loc[mask, "Metadata_QC_status"] = "passed"  # rows matching post-QC are passed

# Set min cluster size for HDBSCAN
min_cluster_size = 50


# ## Compute individual scores

# ### Pre-QC UMAP scores

# In[3]:


# Isolate just the feature space from Metadata in dataframe
pre_X = pre_QC_df.loc[:, ~pre_QC_df.columns.str.startswith("Metadata_")].values

# Run PCA on pre-QC feature space
pca = PCA(n_components=5, random_state=0)
pre_QC_pca = pca.fit_transform(pre_X)

# Print the total explained variance percentage for the first 5 PCs
explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total explained variance by first 5 PCs: {explained_variance:.2f}%")

# Create scree plot to show variance explained by each PC
plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker="o",
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid()
plt.tight_layout()
plt.show()


# In[4]:


# Run HDBSCAN using sklearn compatible interface
clusterer = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
pre_cluster_labels = clusterer.fit_predict(pre_QC_pca)

# Attach cluster labels to the original dataframe
pre_QC_df["cluster"] = pre_cluster_labels

# Info about clusters
print("Unique cluster labels (-1 is noise):", np.unique(pre_cluster_labels))

# Make a dataframe for enrichment analysis: only metadata + cluster label
metadata_cols = [col for col in pre_QC_df.columns if col.startswith("Metadata_")]
pre_enrichment_df = pre_QC_df[[*metadata_cols, "cluster"]].copy()

# Exclude noise points if you want enrichment only on real clusters
pre_enrichment_df = pre_enrichment_df[pre_enrichment_df["cluster"] != -1]
print(pre_enrichment_df.shape)
pre_enrichment_df.head()


# ### Post-QC UMAP scores

# In[5]:


# Isolate just the feature space from Metadata in dataframe
post_X = post_QC_df.loc[:, ~post_QC_df.columns.str.startswith("Metadata_")].values

# Run PCA on post-QC feature space
pca = PCA(n_components=5, random_state=0)
post_QC_pca = pca.fit_transform(post_X)

# Print the total explained variance percentage for the first 5 PCs
explained_variance = np.sum(pca.explained_variance_ratio_) * 100
print(f"Total explained variance by first 5 PCs: {explained_variance:.2f}%")

# Create scree plot to show variance explained by each PC
plt.figure(figsize=(8, 5))
plt.plot(
    np.arange(1, len(pca.explained_variance_ratio_) + 1),
    pca.explained_variance_ratio_,
    marker="o",
)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance Ratio")
plt.title("Scree Plot")
plt.grid()
plt.tight_layout()
plt.show()


# In[6]:


# Run HDBSCAN using sklearn compatible interface
clusterer = HDBSCAN(min_cluster_size=min_cluster_size, n_jobs=-1)
post_cluster_labels = clusterer.fit_predict(post_QC_pca)

# Attach cluster labels to the original dataframe
post_QC_df["cluster"] = post_cluster_labels

# Info about clusters
print("Unique cluster labels (-1 is noise):", np.unique(post_cluster_labels))

# Make a dataframe for enrichment analysis: only metadata + cluster label
metadata_cols = [col for col in post_QC_df.columns if col.startswith("Metadata_")]
post_enrichment_df = post_QC_df[[*metadata_cols, "cluster"]].copy()

# Exclude noise points if you want enrichment only on real clusters
post_enrichment_df = post_enrichment_df[post_enrichment_df["cluster"] != -1]
print(post_enrichment_df.shape)
post_enrichment_df.head()


# # Visualize where passing or failing cells are within each cluster

# In[7]:


# Attach cluster labels to full dataframe
pre_QC_clusters_df = pre_QC_df.copy()
pre_QC_clusters_df["cluster"] = pre_cluster_labels

# Drop noise
pre_QC_clusters_df = pre_QC_clusters_df[pre_QC_clusters_df["cluster"] != -1]

# Convert labels to string for nicer plotting
pre_QC_clusters_df["cluster"] = pre_QC_clusters_df["cluster"].astype(str)

# Print number of cells in each cluster
print(pre_QC_clusters_df["cluster"].value_counts().sort_index())

# Calculate proportions for each cluster
cluster_counts = (
    pre_QC_clusters_df.groupby(["cluster", "Metadata_QC_status"])
    .size()
    .unstack(fill_value=0)
)
cluster_props = (
    cluster_counts.div(cluster_counts.sum(axis=1), axis=0)
    .reset_index()
    .melt(id_vars="cluster", var_name="QC_status", value_name="proportion")
)

# Plot
p = (
    ggplot(cluster_props, aes(x="cluster", y="proportion", fill="QC_status"))
    + geom_bar(stat="identity", position=position_stack())
    + scale_fill_manual(values={"passed": "#0072B2", "failed": "#D55E00"})
    + labs(
        x="Cluster",
        y="Proportion of cells",
        fill="QC status",
    )
    + theme(
        figure_size=(8, 5),
        axis_text_x=element_text(size=11),
        axis_text_y=element_text(size=11),
        axis_title_x=element_text(size=13),
        axis_title_y=element_text(size=13),
        legend_title=element_text(size=12),
        legend_text=element_text(size=10),
    )
)

p.save(output_dir / "pre_QC_cluster_qc_status_proportions.png", dpi=300)
p.show()


# In[8]:


# Attach cluster labels to full dataframe
post_QC_clusters_df = post_QC_df.copy()
post_QC_clusters_df["cluster"] = post_cluster_labels

# Drop noise
post_QC_clusters_df = post_QC_clusters_df[post_QC_clusters_df["cluster"] != -1]

# Convert labels to string for nicer plotting
post_QC_clusters_df["cluster"] = post_QC_clusters_df["cluster"].astype(str)

# Print number of cells in each cluster
print(post_QC_clusters_df["cluster"].value_counts().sort_index())

# Prepare data
plot_df = (
    post_QC_clusters_df.groupby(["cluster", "Metadata_treatment", "Metadata_cell_type"])
    .size()
    .reset_index(name="count")
)

# Calculate proportion per cluster
plot_df["proportion"] = plot_df.groupby("cluster")["count"].transform(
    lambda x: x / x.sum()
)

# Blind treatments
treatment_blind_map = {
    "DMSO": "treatment1",
    "TGFRi": "treatment2",
    "drug_x": "treatment3",
}
plot_df["Metadata_treatment"] = plot_df["Metadata_treatment"].map(treatment_blind_map)

# Update treatment_celltype
plot_df["treatment_celltype"] = (
    plot_df["Metadata_treatment"] + " | " + plot_df["Metadata_cell_type"]
)

# Update palette dictionary with blinded names
palette_dict = {
    "treatment1 | healthy": "#66c2a5",
    "treatment2 | healthy": "#41ae76",
    "treatment3 | healthy": "#238b45",
    "treatment1 | failing": "#fc8d62",
    "treatment2 | failing": "#e34a33",
    "treatment3 | failing": "#b30000",
}

# Set categorical order with blinded names
desired_order = [
    "treatment1 | healthy",
    "treatment2 | healthy",
    "treatment3 | healthy",
    "treatment1 | failing",
    "treatment2 | failing",
    "treatment3 | failing",
]
plot_df["treatment_celltype"] = pd.Categorical(
    plot_df["treatment_celltype"], categories=desired_order, ordered=True
)


# Plot
p = (
    ggplot(plot_df, aes(x="cluster", y="proportion", fill="treatment_celltype"))
    + geom_bar(stat="identity", position=position_stack())
    + scale_fill_manual(values=palette_dict, name="Treatment | Cell type")
    + labs(
        x="Cluster",
        y="Proportion of cells",
    )
    + theme(
        figure_size=(10, 6),
        axis_text_x=element_text(size=14),
        axis_text_y=element_text(size=14),
        axis_title_x=element_text(size=16),
        axis_title_y=element_text(size=16),
        legend_title=element_text(size=14),
        legend_text=element_text(size=12),
    )
)

p.save(output_dir / "post_QC_cluster_treatment_celltype.png", dpi=300)
p
