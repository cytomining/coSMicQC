#!/usr/bin/env python
# coding: utf-8

# ## Generate UMAP embeddings from data after QC and plot

# In[1]:


from pathlib import Path

import pandas as pd
import umap
from plotnine import (
    aes,
    element_text,
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

# Load in no QC normalized dataframe for CFReT example plate
QC_df = pd.read_parquet(
    Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected.parquet"
    )
)

# Drop any NaN rows from columns that are not metadata (contain Metadata_ prefix)
QC_df = QC_df.dropna(
    subset=[col for col in QC_df.columns if not col.startswith("Metadata_")]
).reset_index(drop=True)

# Create new column for treatment cell type ID for each unique combo
QC_df["Metadata_Treatment_CellType_ID"] = (
    QC_df["Metadata_treatment"] + "_" + QC_df["Metadata_cell_type"]
)

# Print shape of the DataFrame
print(QC_df.shape)
QC_df.head()


# In[3]:


# Process cp_df to separate features and metadata
cp_features = infer_cp_features(QC_df)
meta_features = infer_cp_features(QC_df, metadata=True)

# Make sure to reinitialize UMAP instance per plate
umap_fit = umap.UMAP(
    random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
)

# Fit UMAP and convert to pandas DataFrame
embeddings = pd.DataFrame(
    umap_fit.fit_transform(QC_df.loc[:, cp_features]),
    columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
)
print(f"{embeddings.shape} UMAP embeddings generated")

# Combine with metadata
cp_umap_with_metadata_df = pd.concat([QC_df.loc[:, meta_features], embeddings], axis=1)

# Save UMAP with metadata DataFrame
cp_umap_with_metadata_df.to_parquet(output_dir / "post_QC_umap_embeddings.parquet")


# In[4]:


# Set the figure size
height = 8
width = 8
set_option("figure_size", (width, height))

# Plot with custom color palette
p = (
    ggplot(
        cp_umap_with_metadata_df,
        aes(x="UMAP0", y="UMAP1", color="Metadata_Treatment_CellType_ID"),
    )
    + labs(
        color="Cell type\nand treatment",
    )
    + geom_point(alpha=0.2, size=2)
    + theme_bw()
    + theme(
        axis_title=element_text(size=20),
        axis_text=element_text(size=16),
        legend_position="bottom",
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
    )
    + scale_color_brewer(type="qual", palette="Dark2")  # Change palette as needed
    + guides(
        color=guide_legend(
            override_aes={
                "alpha": 1,  # fully opaque in legend
                "size": 5,  # bigger points in legend
            }
        )
    )
)

# Save the plot
p.save(figure_dir / "facet_umap_with_QC_plot.png", dpi=400, width=width, height=height)

p.show()
