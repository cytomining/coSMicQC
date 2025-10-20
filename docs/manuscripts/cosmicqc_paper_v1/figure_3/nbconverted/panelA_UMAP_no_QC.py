#!/usr/bin/env python
# coding: utf-8

# ## Generate UMAP embeddings from no QC data and plot

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
    scale_color_manual,
    theme,
    theme_bw,
)
from plotnine.options import set_option
from pycytominer.cyto_utils import infer_cp_features

# In[2]:


# Set constants
umap_random_seed = 0
umap_n_components = 2

# Set output directory
figure_dir = Path("./figures")
output_dir = Path("./umap_embeddings")
output_dir.mkdir(parents=True, exist_ok=True)
figure_dir.mkdir(parents=True, exist_ok=True)

# Load in no QC normalized dataframe for CFReT example plate
no_QC_df = pd.read_parquet(
    Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected_no_QC.parquet"
    )
)

# Drop any NaN rows from columns that are not metadata (contain Metadata_ prefix)
no_QC_df = no_QC_df.dropna(
    subset=[col for col in no_QC_df.columns if not col.startswith("Metadata_")]
).reset_index(drop=True)

# Blind treatments for UMAP visualization
# (DMSO = treatment1, TGFRi = treatment2, drug_X = treatment3)
treatment_mapping = {
    "DMSO": "treatment1",
    "TGFRi": "treatment2",
    "drug_x": "treatment3",
}
no_QC_df["Metadata_treatment"] = no_QC_df["Metadata_treatment"].map(treatment_mapping)

# Create new column for treatment cell type ID for each unique combo
no_QC_df["Metadata_Treatment_CellType_ID"] = (
    no_QC_df["Metadata_treatment"] + "_" + no_QC_df["Metadata_cell_type"]
)

# Print shape of the DataFrame
print(no_QC_df.shape)
no_QC_df.head()


# In[3]:


# Check unique values for treatment and treatment-cell type ID
print("Unique blinded treatments:", no_QC_df["Metadata_treatment"].unique())
print(
    "Unique Treatment_CellType_IDs:",
    no_QC_df["Metadata_Treatment_CellType_ID"].unique(),
)


# In[4]:


# Process cp_df to separate features and metadata
cp_features = infer_cp_features(no_QC_df)
meta_features = infer_cp_features(no_QC_df, metadata=True)

# Initialize UMAP instance
umap_fit = umap.UMAP(
    random_state=umap_random_seed, n_components=umap_n_components, n_jobs=1
)

# Fit UMAP and convert to pandas DataFrame
embeddings = pd.DataFrame(
    umap_fit.fit_transform(no_QC_df.loc[:, cp_features]),
    columns=[f"UMAP{x}" for x in range(0, umap_n_components)],
)
print(f"{embeddings.shape} UMAP embeddings generated")

# Combine with metadata
cp_umap_with_metadata_df = pd.concat(
    [no_QC_df.loc[:, meta_features], embeddings], axis=1
)


# In[5]:


# Add QC_status column and set all to "failed"
cp_umap_with_metadata_df["Metadata_QC_status"] = "failed"

# Find matching rows and set QC_status to "passed"
match_cols = [
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]

# Load in no QC normalized dataframe for CFReT example plate
QC_df = pd.read_parquet(
    Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost230405150001_sc_feature_selected.parquet"
    )
)

# Create a MultiIndex for fast lookup
qc_index = QC_df.set_index(match_cols).index
mask = cp_umap_with_metadata_df.set_index(match_cols).index.isin(qc_index)
cp_umap_with_metadata_df.loc[mask, "Metadata_QC_status"] = "passed"

# Save UMAP with metadata DataFrame
cp_umap_with_metadata_df.to_parquet(output_dir / "pre_QC_umap_embeddings.parquet")

print(cp_umap_with_metadata_df.shape)
cp_umap_with_metadata_df.head()


# In[6]:


# Set the figure size
height = 8
width = 8
set_option("figure_size", (width, height))

# Plot UMAP of non-QC profiles labelled with QC status and
# faceted by treatment and cell type
p = (
    ggplot(
        cp_umap_with_metadata_df,
        aes(x="UMAP0", y="UMAP1", color="Metadata_QC_status"),
    )
    + labs(
        color="QC Status",
    )
    + geom_point(alpha=0.2, size=2)
    + facet_wrap(
        "Metadata_Treatment_CellType_ID",
        ncol=2,
        scales="fixed",
    )
    + theme_bw()
    + theme(
        axis_title=element_text(size=20),
        axis_text=element_text(size=16),
        legend_title=element_text(size=16),
        legend_text=element_text(size=14),
        legend_position="bottom",
        strip_text=element_text(size=16),  # Adjust facet label size
    )
    + scale_color_manual(
        values={"passed": "#0072B2", "failed": "#D55E00"}
    )  # Blue for passed, orange for failed (colorblind-friendly)
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
p.save(figure_dir / "facet_umap_no_QC_plot.png", dpi=600, width=width, height=height)

p.show()
