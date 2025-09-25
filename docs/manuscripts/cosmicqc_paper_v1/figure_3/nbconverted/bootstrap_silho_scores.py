#!/usr/bin/env python
# coding: utf-8

# # Calculate Silhouette scores with bootstrapping method
# 
# Determine if there is a significant improvement in clustering after QC.

# In[1]:


import pathlib
import warnings

import hdbscan
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import silhouette_score

# Ignore warning about deprecated argument name in sklearn
warnings.filterwarnings("ignore", message="'force_all_finite' was renamed")


# In[2]:


def _silhouette_once(
    X: np.ndarray,
    min_cluster_size: int,
    sample_frac: float,
    max_samples: int,
    rng_seed: int,
) -> None:
    """Run one bootstrap iteration for silhouette score.

    Args:
        X (np.ndarray): Data points.
        min_cluster_size (int): Minimum cluster size for HDBSCAN.
        sample_frac (float): Fraction of samples to use for each bootstrap.
        max_samples (int): Cap on subsample size for speed.
        rng_seed (int): Random seed for reproducibility.
    """
    rng = np.random.default_rng(rng_seed)
    sample_size = min(int(len(X) * sample_frac), max_samples)
    sample_idx = rng.choice(len(X), size=sample_size, replace=True)
    X_sample = X[sample_idx]

    # <-- suppress warning inside worker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size).fit(X_sample)

    labels = clusterer.labels_

    if len(np.unique(labels[labels != -1])) > 1:
        return silhouette_score(X_sample[labels != -1], labels[labels != -1])
    return None


def bootstrap_silhouette(  # noqa: PLR0913
    X: np.ndarray,
    min_cluster_size: int = 50,
    n_bootstraps: int = 1000,
    sample_frac: float = 0.8,
    max_samples: int = 5000,
    random_state: int = 0,
    n_jobs: int = -1,
) -> np.ndarray:
    """Perform bootstrapping to compute silhouette scores with parallelization.

    Args:
        X (np.ndarray): Data points.
        min_cluster_size (int, optional): Minimum cluster size for HDBSCAN.
        n_bootstraps (int, optional): Number of bootstrap iterations.
        sample_frac (float, optional): Fraction of samples to use for each bootstrap.
        max_samples (int, optional): Cap on subsample size for speed. Defaults to 5000.
        random_state (int, optional): Random seed for reproducibility.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1 (all cores).

    Returns:
        np.ndarray: Silhouette scores for each bootstrap iteration.
    """
    rng = np.random.default_rng(random_state)
    seeds = rng.integers(0, 1e9, size=n_bootstraps)

    sil_scores = Parallel(n_jobs=n_jobs)(
        delayed(_silhouette_once)(X, min_cluster_size, sample_frac, max_samples, seed)
        for seed in seeds
    )

    return np.array([s for s in sil_scores if s is not None])


# In[3]:


# Output dir for figure
output_dir = pathlib.Path("./figures")
output_dir.mkdir(exist_ok=True, parents=True)

# Load in pre- and post-QC UMAP embeddings DataFrames
pre_QC_umap_df = pd.read_parquet(
    "../figure_3/umap_embeddings/pre_QC_umap_embeddings.parquet"
)
post_QC_umap_df = pd.read_parquet(
    "../figure_3/umap_embeddings/post_QC_umap_embeddings.parquet"
)

# Set min cluster size for HDBSCAN
min_cluster_size = 100


# ## Compute individual Silhouette scores

# In[4]:


# Calculate and print silhouette scores for pre-QC datasets
pre_X = pre_QC_umap_df[["UMAP0", "UMAP1"]].values

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
pre_cluster_labels = clusterer.fit_predict(pre_X)

# Info about clusters
print("Unique cluster labels (-1 is noise):", np.unique(pre_cluster_labels))
print("Cluster persistence:", clusterer.cluster_persistence_)

# Silhouette score (exclude noise points labeled as -1)
mask = pre_cluster_labels != -1
X_clustered = pre_X[mask]
labels_clustered = pre_cluster_labels[mask]
pre_QC_score = silhouette_score(X_clustered, labels_clustered)
print("Silhouette score (excluding noise):", pre_QC_score)


# # Output which cells go with which cluster (plot bar plot proportion of cells removed on y and the x is the cluster label)
# 
# Is there a way to show that the cluster cells that did not get removed moved to a different cluster?

# In[5]:


import matplotlib.pyplot as plt

# Attach cluster labels to full dataframe
pre_QC_clusters_df = pre_QC_umap_df.copy()
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
cluster_props = cluster_counts.div(cluster_counts.sum(axis=1), axis=0)

# Plot
ax = cluster_props.plot(
    kind="bar", stacked=True, color=["tomato", "mediumseagreen"], figsize=(8, 5)
)
ax.set_ylabel("Proportion of Cells")
ax.set_xlabel("Cluster Label")
ax.set_title("Proportion of Cells Passed/Failed QC by Cluster")
ax.legend(["Failed QC", "Passed QC"], title="QC Status")
plt.tight_layout()
plt.savefig(output_dir / "pre_QC_cluster_qc_status_proportions.png", dpi=300)
plt.show()


# In[6]:


# Quantify number of failed vs passed cells in pre-QC cluster 0
cluster0_df = pre_QC_clusters_df[pre_QC_clusters_df["cluster"] == "0"]
failed_count = (cluster0_df["Metadata_QC_status"] == "failed").sum()
passed_count = (cluster0_df["Metadata_QC_status"] == "passed").sum()

print(f"Cluster 0 - Failed cells: {failed_count}")
print(f"Cluster 0 - Passed cells: {passed_count}")


# In[7]:


# Calculate and print silhouette scores for post-QC datasets
post_X = post_QC_umap_df[["UMAP0", "UMAP1"]].values

# Run HDBSCAN
clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
post_cluster_labels = clusterer.fit_predict(post_X)

# Info about clusters
print("Unique cluster labels (-1 is noise):", np.unique(post_cluster_labels))
print("Cluster persistence:", clusterer.cluster_persistence_)

# Silhouette score (exclude noise points labeled as -1)
mask = post_cluster_labels != -1
post_X_clustered = post_X[mask]
labels_clustered = post_cluster_labels[mask]
post_QC_score = silhouette_score(post_X_clustered, labels_clustered)
print("Silhouette score (excluding noise):", post_QC_score)


# ## Perform bootstrapping method to compute Silhouette scores
# 
# Bootstrap method uses replacement over 1000 iterations.
# Applied to pre-QC and post_QC datasets to evaluate significance in difference.

# In[8]:


# Perform bootstrapping on pre-QC and post-QC datasets
pre_bootstrap_scores = bootstrap_silhouette(
    pre_X, n_bootstraps=1000, min_cluster_size=min_cluster_size
)
post_bootstrap_scores = bootstrap_silhouette(
    post_X, n_bootstraps=1000, min_cluster_size=min_cluster_size
)

print("Before QC:", pre_bootstrap_scores.mean(), "+/-", pre_bootstrap_scores.std())
print("After QC:", post_bootstrap_scores.mean(), "+/-", post_bootstrap_scores.std())


# In[9]:


t_stat, p_value = stats.ttest_ind(
    pre_bootstrap_scores, post_bootstrap_scores, equal_var=False
)
print("T-statistic:", t_stat)
print("P-value:", p_value)

