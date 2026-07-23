#!/usr/bin/env python
# coding: utf-8

# # Apply trained models onto respective failing cells.

# In[1]:


import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from joblib import load
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    geom_density,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    theme,
    theme_bw,
)
from plotnine.options import set_option

sys.path.append(str(pathlib.Path.cwd().resolve().parent / "figure_3"))
from figure3_utils import get_X_y_data


# ## Helper functions

# In[2]:


def per_class_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str
) -> list[dict]:
    """Compute accuracy separately for class 0 (failing) and class 1 (healthy).

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        model_name (str): Name of the model.

    Returns:
        list[dict]: List of dictionaries containing model name, class label, accuracy,
        and number of samples for each class.
    """
    results = []
    for cls, cls_label in [(0, "failing"), (1, "healthy")]:
        mask = y_true == cls
        acc = (y_pred[mask] == y_true[mask]).mean()
        results.append(
            {
                "model": model_name,
                "class": cls_label,
                "accuracy": acc,
                "n": mask.sum(),
            }
        )
    return results


# In[3]:


def apply_model_to_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    model: joblib.Parallel,
    group_label: str,
    model_label: str,
) -> pd.DataFrame:
    """Score a failing-cell group with a model and label the result.

    Args:
        df (pd.DataFrame): Failing-cell group (with "Metadata_cell_type" and
            `feature_cols`).
        feature_cols (list[str]): Feature columns `model` was trained on, in order.
        model (joblib.Parallel): Fitted classifier with `predict_proba`.
        group_label (str): Name of the failing-cell group (e.g. "Both").
        model_label (str): Name of the model used to score the group.

    Returns:
        pd.DataFrame: Predicted probability, true class, group, and model per cell.
    """
    filtered_df = df[metadata_cols_to_keep + feature_cols].dropna(subset=feature_cols)
    X, y = get_X_y_data(df=filtered_df, label="Metadata_cell_type")
    y_binary = le.transform(y)
    y_probs = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            "predicted_prob": y_probs,
            "true_class": np.where(y_binary == 0, "Diseased", "Healthy"),
            "group": group_label,
            "model": model_label,
        }
    )


# In[4]:


figure_path = pathlib.Path("./figures")
# make directory if it doesn't already exist
figure_path.mkdir(exist_ok=True)


# ## Load in label encoder

# In[5]:


# load in label encoder
le = load(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/5.machine_learning/0.train_logistic_regression/encoder_results/label_encoder_log_reg_fs_plate_4.joblib"
    )
)


# ## Load in model

# In[6]:


# Load the trained models
no_QC_model = joblib.load(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/5.machine_learning/0.train_logistic_regression/models/no_QC_models/log_reg_fs_plate_4_final_downsample_no_QC.joblib"
    )
)

coSMicQC_model = joblib.load(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/5.machine_learning/0.train_logistic_regression/models/log_reg_fs_plate_4_final_downsample.joblib"
    )
)

ECOD_model = joblib.load(
    pathlib.Path(
        "../figure_3/models/log_reg_fs_ecod_final_downsample.joblib"
    )
)


# ## Load in no-QC dataset and filter for failing cells

# In[ ]:


# Load the normalized QC plate 4 to be able to filter features
plate_4_no_QC = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_normalized_no_QC.parquet"
    )
)

# Load the failing cells metadata (contains per-cell QC failure booleans)
failing_cells_metadata = pd.read_parquet(
    pathlib.Path("../figure_3/failing_cells_metadata/failing_cells_metadata.parquet")
)

# Merge plate_4_no_QC with the QC failure metadata on shared cell identifiers
merge_cols = [
    "Metadata_Well",
    "Metadata_Site",
    "Metadata_Nuclei_Location_Center_X",
    "Metadata_Nuclei_Location_Center_Y",
]
plate_4_no_QC_merged = plate_4_no_QC.merge(
    failing_cells_metadata, on=merge_cols, how="inner"
)

# Filter into separate dataframes for each QC failure type
ecod_df = plate_4_no_QC_merged[plate_4_no_QC_merged["failed_pyod_ecod"]]

cosmicqc_df = plate_4_no_QC_merged[plate_4_no_QC_merged["failed_cosmicqc"]]

both_df = plate_4_no_QC_merged[plate_4_no_QC_merged["failed_both"]]

# Cells failing only one method (i.e. excluding the overlap with the other method)
cosmicqc_only_df = plate_4_no_QC_merged[
    plate_4_no_QC_merged["failed_cosmicqc"] & ~plate_4_no_QC_merged["failed_both"]
]
ecod_only_df = plate_4_no_QC_merged[
    plate_4_no_QC_merged["failed_pyod_ecod"] & ~plate_4_no_QC_merged["failed_both"]
]

print(f"ECOD (all failures, including overlap with coSMicQC): {ecod_df.shape[0]} cells")
print(f"coSMicQC (all failures, including overlap with ECOD): {cosmicqc_df.shape[0]} cells")
print(f"Both: {both_df.shape[0]} cells")
print(f"coSMicQC only (excluding overlap with ECOD): {cosmicqc_only_df.shape[0]} cells")
print(f"ECOD only (excluding overlap with coSMicQC): {ecod_only_df.shape[0]} cells")


# ## Load in feature selected profiles from coSMicQC and ECOD model training to collect features

# In[8]:


# Get feature columns (excluding any Metadata_ columns) used by each model's training data
coSMicQC_plate_4_features = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected.parquet"
    )
)
ECOD_plate_4_features = pd.read_parquet(
    pathlib.Path("../figure_3/models/idc_normalized_feature_selected.parquet")
)
no_QC_plate_4_features = pd.read_parquet(
    pathlib.Path(
        "/media/18tbdrive/1.Github_Repositories/cellpainting_predicts_cardiac_fibrosis/3.process_cfret_features/data/single_cell_profiles/localhost231120090001_sc_feature_selected_no_QC.parquet"
    )
)

cosmicqc_feature_cols = [
    col for col in coSMicQC_plate_4_features.columns if not col.startswith("Metadata_")
]
ecod_feature_cols = [
    col for col in ECOD_plate_4_features.columns if not col.startswith("Metadata_")
]
no_qc_feature_cols = [
    col for col in no_QC_plate_4_features.columns if not col.startswith("Metadata_")
]

print(f"coSMicQC feature columns: {len(cosmicqc_feature_cols)}")
print(f"ECOD feature columns: {len(ecod_feature_cols)}")
print(f"No QC feature columns: {len(no_qc_feature_cols)}")


# In[9]:


# Metadata columns needed downstream (label column for get_X_y_data)
metadata_cols_to_keep = ["Metadata_cell_type"]

# Filter each dataset down to only the columns its respective model was trained on
cosmicqc_filtered_df = cosmicqc_df[metadata_cols_to_keep + cosmicqc_feature_cols]
ecod_filtered_df = ecod_df[metadata_cols_to_keep + ecod_feature_cols]

# Drop rows with any NaN in the feature columns before predicting
cosmicqc_before = cosmicqc_filtered_df.shape[0]
ecod_before = ecod_filtered_df.shape[0]

cosmicqc_filtered_df = cosmicqc_filtered_df.dropna(subset=cosmicqc_feature_cols)
ecod_filtered_df = ecod_filtered_df.dropna(subset=ecod_feature_cols)

print(
    f"coSMicQC rows after dropping NaNs: {cosmicqc_filtered_df.shape[0]} (dropped {cosmicqc_before - cosmicqc_filtered_df.shape[0]})"
)
print(
    f"ECOD rows after dropping NaNs: {ecod_filtered_df.shape[0]} (dropped {ecod_before - ecod_filtered_df.shape[0]})"
)

# Load in X and y data for the coSMicQC-filtered dataset
X_cosmicqc, y_cosmicqc = get_X_y_data(
    df=cosmicqc_filtered_df, label="Metadata_cell_type"
)
y_binary_cosmicqc = le.transform(y_cosmicqc)
y_probs_cosmicqc = coSMicQC_model.predict_proba(X_cosmicqc)[:, 1]

# Load in X and y data for the ECOD-filtered dataset
X_ecod, y_ecod = get_X_y_data(df=ecod_filtered_df, label="Metadata_cell_type")
y_binary_ecod = le.transform(y_ecod)
y_probs_ecod = ECOD_model.predict_proba(X_ecod)[:, 1]


# In[10]:


print(f"coSMicQC positive class prevalence: {y_binary_cosmicqc.mean():.3f}")
print(f"ECOD positive class prevalence: {y_binary_ecod.mean():.3f}")


# In[11]:


# Convert probabilities to predicted class labels using a 0.5 threshold
threshold = 0.5
y_pred_cosmicqc = (y_probs_cosmicqc >= threshold).astype(int)
y_pred_ecod = (y_probs_ecod >= threshold).astype(int)

acc_records = (
    per_class_accuracy(y_binary_cosmicqc, y_pred_cosmicqc, "coSMicQC")
    + per_class_accuracy(y_binary_ecod, y_pred_ecod, "ECOD")
)
acc_df = pd.DataFrame(acc_records)
acc_df


# ## Distribution of predicted probabilities on failing cells, by QC model and true class
# 
# Apply all three models (no-QC, coSMicQC, ECOD) to each failing-cell group — cells
# caught by coSMicQC only, by ECOD only, and by both methods — and plot the resulting
# predicted probabilities, faceted by which model produced the prediction (columns)
# and the cell's true class (rows), with one density line per failing-cell group
# within each facet. This shows how well-separated the two classes are according to
# each model, within each QC method's failing-cell population.

# In[12]:


# Failing-cell groups (from each QC method's failures) to score
failing_cell_groups = {
    "coSMicQC only": cosmicqc_only_df,
    "PyOD ECOD only": ecod_only_df,
    "Both": both_df,
}

# Models to apply to each group, each with its own feature set
models_to_apply = {
    "No QC": (no_QC_model, no_qc_feature_cols),
    "coSMicQC": (coSMicQC_model, cosmicqc_feature_cols),
    "ECOD": (ECOD_model, ecod_feature_cols),
}

# Apply every model to every failing-cell group
df_failing_probs = pd.concat(
    [
        apply_model_to_group(
            df=group_df,
            feature_cols=feature_cols,
            model=model,
            group_label=group_label,
            model_label=model_label,
        )
        for group_label, group_df in failing_cell_groups.items()
        for model_label, (model, feature_cols) in models_to_apply.items()
    ]
)

# Fix facet column order (No QC, coSMicQC, ECOD) rather than alphabetical
df_failing_probs["model"] = pd.Categorical(
    df_failing_probs["model"], categories=["No QC", "coSMicQC", "ECOD"]
)

df_failing_probs.groupby(["model", "group", "true_class"])["predicted_prob"].describe()


# In[13]:


# Count how many samples of each true class are in each failing-cell group,
# per model (NaN-dropping is model-specific, so counts can vary slightly)
class_counts_by_group = (
    df_failing_probs.groupby(["model", "group", "true_class"], observed=True)
    .size()
    .unstack(fill_value=0)
)

class_counts_by_group


# In[14]:


# Set the figure size (update as needed) -- wider to fit 3 model columns
height = 14
width = 26
set_option("figure_size", (width, height))

# Plot: predicted probability distributions on failing cells, faceted by QC
# model (columns) and true class (rows), colored by failing-cell group
group_colors = {
    "coSMicQC only": "#CC79A7",
    "PyOD ECOD only": "#0072B2",
    "Both": "#D55E00",
}

# Add prefixed label columns so facet strips read "True class: X" / "Model: Y"
df_failing_probs = df_failing_probs.assign(
    true_class_label=lambda d: "True class: " + d["true_class"].astype(str),
    model_label=lambda d: "Model: " + d["model"].astype(str),
)

failing_probs_plot = (
    ggplot(df_failing_probs, aes(x="predicted_prob", color="group", fill="group"))
    + geom_density(size=1.2, alpha=0.3)
    + facet_wrap("~ true_class_label + model_label", scales="free_y")
    + scale_color_manual(values=group_colors)
    + scale_fill_manual(values=group_colors)
    + scale_x_continuous(labels=lambda breaks: [f"{b:g}" for b in breaks])
    + labs(
        x="Predicted probability (healthy class = 1)",
        y="Density",
        color="QC failure status",
        fill="QC failure status",
    )
    + theme_bw()
    + theme(
        legend_position="right",
        axis_title=element_text(size=38),
        axis_text=element_text(size=34),
        legend_title=element_text(size=36),
        legend_text=element_text(size=34),
        strip_text=element_text(size=36),
    )
)

failing_probs_plot.save(
    f"{figure_path}/predicted_probability_distributions_failing_cells.png",
    dpi=600,
    limitsize=False
)

