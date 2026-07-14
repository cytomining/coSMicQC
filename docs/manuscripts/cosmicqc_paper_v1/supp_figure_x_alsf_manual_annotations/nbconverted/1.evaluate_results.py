#!/usr/bin/env python
# coding: utf-8

# # Run evaluation on the manual annotations against coSMicQC

# In[1]:


import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    geom_hline,
    geom_point,
    geom_text,
    geom_tile,
    ggplot,
    labs,
    scale_fill_gradient,
    theme,
    theme_bw,
)
from plotnine.options import set_option
from sklearn.metrics import cohen_kappa_score, confusion_matrix


# In[2]:


# -----------------------------
# metrics function
# -----------------------------
def compute_metrics(df: pd.DataFrame, pred_col: str, true_col: str) -> dict:
    """
    Compute accuracy, precision, recall, and specificity from a confusion matrix.

    Args:
        df (pd.DataFrame): DataFrame containing the true and predicted labels.
        pred_col (str): Name of the column containing predicted labels.
        true_col (str): Name of the column containing true labels.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, specificity, 
            and number of cells.
    """

    cm = confusion_matrix(df[true_col], df[pred_col], labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "precision": tp / (tp + fp) if (tp + fp) else np.nan,
        "recall": tp / (tp + fn) if (tp + fn) else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "n_cells": len(df),
    }


# In[3]:


# Set figure directory
figure_dir = Path("./figures")
figure_dir.mkdir(exist_ok=True, parents=True)

# Set path to the stratified sample
sample_path = Path("./data/stratified_qc_sample.parquet")

# Load the stratified sample
qc_sample_df = pd.read_parquet(sample_path)
print("Stratified sample loaded from:", sample_path)
print(qc_sample_df.shape)

annotation_df = qc_sample_df.reset_index(names="sample_row_id").copy()
annotation_df = annotation_df.sample(frac=1, random_state=42).reset_index(drop=True)

preview_columns = [
    column
    for column in [
        "Metadata_cell_line",
        "Metadata_condition",
        "Metadata_Plate",
        "Metadata_Well",
        "Image_Metadata_Row",
        "Image_Metadata_Col",
    ]
    if column in annotation_df.columns
]
annotation_df[preview_columns].head()


# In[4]:


# Set path to manual annotation files from experts
annotation_paths = list(
    Path("./data").rglob("*manual_segmentation_annotations*.parquet")
)

if not annotation_paths:
    raise ValueError("No annotation parquet files found.")

# Concat annotations + add annotator from filename
annotations_df = pd.concat(
    [
        pd.read_parquet(p).assign(
            annotator=p.stem.split("manual_segmentation_annotations")[-1]
        )
        for p in annotation_paths
    ],
    ignore_index=True,
)

# Blind the annotators
annotations_df["annotator"] = annotations_df["annotator"].str.lstrip("_")

annotator_map = {
    annotator: f"Annotator {i + 1}"
    for i, annotator in enumerate(sorted(annotations_df["annotator"].unique()))
}

annotations_df["annotator"] = annotations_df["annotator"].map(annotator_map)


annotations_df.head()


# In[5]:


manual_eval_df = annotation_df.merge(
    annotations_df,
    on="sample_row_id",
    how="inner",
)

# keep only valid labels
manual_eval_df = manual_eval_df.loc[
    manual_eval_df["manual_segmentation_label"].isin(["good", "bad"])
].copy()

# encode manual labels
manual_eval_df["manual_segmentation_is_bad"] = (
    manual_eval_df["manual_segmentation_label"] == "bad"
).astype(int)

manual_eval_df["failed_qc"] = manual_eval_df["failed_qc"].astype(int)

# -----------------------------
# confusion matrix (split by annotator)
# -----------------------------
confusion_df = (
    manual_eval_df
    .groupby(["annotator", "failed_qc", "manual_segmentation_is_bad"])
    .size()
    .reset_index(name="count")
)

# -----------------------------
# save evaluation dataset
# -----------------------------
manual_eval_df.to_parquet(
    "./data/manual_segmentation_evaluation.parquet",
    index=False
)

print("Evaluation data saved to: ./data/manual_segmentation_evaluation.parquet")

display(confusion_df)


# In[6]:


annotations_df["manual_segmentation_label"].value_counts()


# ## Cohen's k plot per comparison across cell lines 

# In[7]:


results = []

cell_lines = manual_eval_df["Metadata_cell_line"].unique()

for cl in cell_lines:

    df_cl = manual_eval_df[manual_eval_df["Metadata_cell_line"] == cl]

    # split annotators
    a1 = df_cl[df_cl["annotator"] == "Annotator 1"][
        ["sample_row_id", "manual_segmentation_is_bad", "failed_qc"]
    ]

    a2 = df_cl[df_cl["annotator"] == "Annotator 2"][
        ["sample_row_id", "manual_segmentation_is_bad", "failed_qc"]
    ]

    # -----------------------------
    # align A1 vs A2 (human baseline)
    # -----------------------------
    human = a1.merge(
        a2,
        on="sample_row_id",
        suffixes=("_a1", "_a2")
    )

    if len(human) > 1:
        kappa_a1_a2 = cohen_kappa_score(
            human["manual_segmentation_is_bad_a1"],
            human["manual_segmentation_is_bad_a2"],
        )
    else:
        kappa_a1_a2 = np.nan

    # -----------------------------
    # model vs A1
    # -----------------------------
    if len(a1) > 0:
        kappa_model_a1 = cohen_kappa_score(
            a1["failed_qc"],
            a1["manual_segmentation_is_bad"],
        )
    else:
        kappa_model_a1 = np.nan

    # -----------------------------
    # model vs A2
    # -----------------------------
    if len(a2) > 0:
        kappa_model_a2 = cohen_kappa_score(
            a2["failed_qc"],
            a2["manual_segmentation_is_bad"],
        )
    else:
        kappa_model_a2 = np.nan

    results.append({
        "cell_line": cl,
        "kappa_A1_A2": kappa_a1_a2,
        "kappa_model_A1": kappa_model_a1,
        "kappa_model_A2": kappa_model_a2,
    })

kappa_df = pd.DataFrame(results)

print("Per cell line κ:")
print(kappa_df)

print("\nMean ± SD across cell lines:")
print(
    kappa_df[
        ["kappa_A1_A2", "kappa_model_A1", "kappa_model_A2"]
    ].agg(["mean", "std"])
)

# -----------------------------
# OVERALL κ (pooled across all cell lines)
# -----------------------------
human_all = manual_eval_df.merge(
    manual_eval_df,
    on="sample_row_id",
    suffixes=("_a1", "_a2")
)

human_all = human_all[
    (human_all["annotator_a1"] == "Annotator 1") &
    (human_all["annotator_a2"] == "Annotator 2")
]

overall_kappa = cohen_kappa_score(
    human_all["manual_segmentation_is_bad_a1"],
    human_all["manual_segmentation_is_bad_a2"],
)

print("\nOverall κ (pooled across all cell lines):")
print(overall_kappa)


# In[8]:


# -----------------------------
# reshape to long format
# -----------------------------
plot_df = kappa_df.melt(
    id_vars="cell_line",
    value_vars=["kappa_A1_A2", "kappa_model_A1", "kappa_model_A2"],
    var_name="comparison",
    value_name="kappa",
)

# -----------------------------
# label cleanup
# -----------------------------
label_map = {
    "kappa_A1_A2": "Human (A1 vs A2)",
    "kappa_model_A1": "coSMicQC vs A1",
    "kappa_model_A2": "coSMicQC vs A2",
}

plot_df["comparison"] = plot_df["comparison"].map(label_map)

# -----------------------------
# add overall mean summary row
# -----------------------------
overall = kappa_df[
    ["kappa_A1_A2", "kappa_model_A1", "kappa_model_A2"]
].mean()

overall_df = pd.DataFrame({
    "cell_line": ["Overall"] * 3,
    "comparison": [
        "Human (A1 vs A2)",
        "coSMicQC vs A1",
        "coSMicQC vs A2",
    ],
    "kappa": [
        overall["kappa_A1_A2"],
        overall["kappa_model_A1"],
        overall["kappa_model_A2"],
    ],
})

plot_df = pd.concat([plot_df, overall_df], ignore_index=True)

# enforce categorical alignment
cell_order = [*list(kappa_df["cell_line"].unique()), "Overall"]

plot_df["cell_line"] = pd.Categorical(
    plot_df["cell_line"],
    categories=cell_order,
    ordered=True
)

# Cell line mapping (blinded names)
cell_line_map = {
    "CHLA-10": "Cell line A",
    "CHLA-113": "Cell line B",
    "CHLA-218": "Cell line C",
    "CHLA-25": "Cell line D",
    "U2-OS": "Cell line E",
    "Overall": "Overall"
}

plot_df["cell_line"] = plot_df["cell_line"].astype(str).map(cell_line_map)

# -----------------------------
# plot
# -----------------------------
width = 7
height = 5
set_option("figure_size", (width, height))

p = (
    ggplot(
        plot_df,
        aes(
            x="cell_line",
            y="kappa",
            color="comparison"
        )
    )
    + geom_point(size=3)

    + geom_hline(
        yintercept=0,
        linetype="dashed",
        alpha=0.3
    )

    + labs(
        x="Cell line",
        y="Cohen's κ",
        color="Comparison",
    )

    + theme_bw(base_size=13)
    + theme(
        axis_text_x=element_text(rotation=45, hjust=1)
    )
)

p.save(
    figure_dir / "kappa_human_vs_model.png",
    width=width,
    height=height,
    dpi=600,
    verbose=False,
)

p.show()


# ## Generate confusion matrix data to compute performance metrics

# In[9]:


# --- column-normalized proportions (within annotator + manual decision) ---
confusion_df["col_sum"] = confusion_df.groupby(
    ["annotator", "manual_segmentation_is_bad"]
)["count"].transform("sum")

confusion_df["proportion"] = confusion_df["count"] / confusion_df["col_sum"]
confusion_df["proportion"] = confusion_df["proportion"].fillna(0)

# -----------------------------
# pretty labels for plotting
# -----------------------------
confusion_df["coSMicQC failed_qc"] = confusion_df["failed_qc"].map(
    {1: "Failed QC", 0: "Passed QC"}
)

confusion_df["Manual bad segmentation"] = confusion_df["manual_segmentation_is_bad"].map(
    {1: "Failed QC", 0: "Passed QC"}
)

# optional ordering for nicer plots
confusion_df["coSMicQC failed_qc"] = pd.Categorical(
    confusion_df["coSMicQC failed_qc"],
    categories=["Passed QC", "Failed QC"],
    ordered=True
)

confusion_df["Manual bad segmentation"] = pd.Categorical(
    confusion_df["Manual bad segmentation"],
    categories=["Passed QC", "Failed QC"],
    ordered=True
)

# -----------------------------
# save evaluation dataset (optional but useful)
# -----------------------------
manual_eval_df.to_parquet(
    "./data/manual_segmentation_evaluation.parquet",
    index=False
)

print("Evaluation data saved to: ./data/manual_segmentation_evaluation.parquet")

display(confusion_df)


# In[10]:


# -----------------------------
# MAIN LOOP
# -----------------------------
results = []

cell_lines = manual_eval_df["Metadata_cell_line"].unique()

for cl in cell_lines:

    df_cl = manual_eval_df[manual_eval_df["Metadata_cell_line"] == cl]

    # split annotators
    a1 = df_cl[df_cl["annotator"] == "Annotator 1"]
    a2 = df_cl[df_cl["annotator"] == "Annotator 2"]

    # -----------------------------
    # MODEL vs Annotator 1
    # -----------------------------
    if len(a1) > 0:
        row = compute_metrics(
            a1,
            pred_col="failed_qc",
            true_col="manual_segmentation_is_bad"
        )

        row["comparison"] = "coSMicQC vs Annotator 1"
        row["cell_line"] = cl
        results.append(row)

    # -----------------------------
    # MODEL vs Annotator 2
    # -----------------------------
    if len(a2) > 0:
        row = compute_metrics(
            a2,
            pred_col="failed_qc",
            true_col="manual_segmentation_is_bad"
        )

        row["comparison"] = "coSMicQC vs Annotator 2"
        row["cell_line"] = cl
        results.append(row)

    # -----------------------------
    # HUMAN vs HUMAN baseline
    # -----------------------------
    human = a1.merge(
        a2,
        on="sample_row_id",
        suffixes=("_a1", "_a2")
    )

    if len(human) > 0:
        row = compute_metrics(
            human,
            pred_col="manual_segmentation_is_bad_a2",
            true_col="manual_segmentation_is_bad_a1"
        )

        row["comparison"] = "Human (A1 vs A2)"
        row["cell_line"] = cl
        results.append(row)


metrics_df = pd.DataFrame(results)
metrics_df.head()


# In[11]:


# -----------------------------
# reshape to long format
# -----------------------------
plot_df = metrics_df.melt(
    id_vars=["cell_line", "comparison"],
    value_vars=[
        "accuracy",
        "precision",
        "recall",
        "specificity"
    ],
    var_name="metric",
    value_name="value"
)

# -----------------------------
# clean metric names
# -----------------------------
metric_map = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall (Sensitivity)",
    "specificity": "Specificity"
}

plot_df["metric"] = plot_df["metric"].map(metric_map)

# -----------------------------
# enforce ordering
# -----------------------------
metric_order = [
    "Accuracy",
    "Precision",
    "Recall (Sensitivity)",
    "Specificity"
]

plot_df["metric"] = pd.Categorical(
    plot_df["metric"],
    categories=metric_order,
    ordered=True
)

# -----------------------------
# plot
# -----------------------------
width = 7
height = 5

set_option("figure_size", (width, height))

p = (
    ggplot(
        plot_df,
        aes(
            x="comparison",
            y="value",
            color="comparison"
        )
    )

    # metric points
    + geom_point(size=3)

    # facet by metric
    + facet_wrap("~metric", scales="free_y")

    + geom_hline(
        yintercept=0,
        linetype="dashed",
        alpha=0.3
    )

    + labs(
        x="Comparison",
        y="Performance"
    )

    + theme_bw(base_size=12)
    + theme(
        axis_text_x=element_text(rotation=25, hjust=1),
        figure_size=(10, 6),
        legend_position="none"
    )
)

p.save(
    figure_dir / "model_performance_versus_annotators.png",
    dpi=600,
    width=width,
    height=height,
    verbose=False,
)

p.show()


# ## Generate confusion matrices comaparing coSMicQc to each annotator and the annotators to each other

# In[12]:


# --- long-format confusion matrix per cell line + annotator ---
cm_df_cell_line = (
    manual_eval_df
    .groupby([
        "annotator",
        "Metadata_cell_line",
        "failed_qc",
        "manual_segmentation_is_bad"
    ])
    .size()
    .reset_index(name="count")
)

# -----------------------------
# FORCE FULL 2x2 GRID PER FACET
# -----------------------------
full_index = pd.DataFrame(
    itertools.product(
        cm_df_cell_line["annotator"].unique(),
        cm_df_cell_line["Metadata_cell_line"].unique(),
        [0, 1],
        [0, 1],
    ),
    columns=[
        "annotator",
        "Metadata_cell_line",
        "failed_qc",
        "manual_segmentation_is_bad",
    ],
)

cm_df_cell_line = full_index.merge(
    cm_df_cell_line,
    on=[
        "annotator",
        "Metadata_cell_line",
        "failed_qc",
        "manual_segmentation_is_bad",
    ],
    how="left",
)

cm_df_cell_line["count"] = cm_df_cell_line["count"].fillna(0)

# -----------------------------
# Compute column normalized proportions
# -----------------------------
cm_df_cell_line["col_sum"] = cm_df_cell_line.groupby(
    ["annotator", "Metadata_cell_line", "manual_segmentation_is_bad"]
)["count"].transform("sum")

cm_df_cell_line["proportion"] = (
    cm_df_cell_line["count"] / cm_df_cell_line["col_sum"]
)

cm_df_cell_line["proportion"] = cm_df_cell_line["proportion"].fillna(0)

# -----------------------------
# pretty labels
# -----------------------------
cm_df_cell_line["coSMicQC failed_qc"] = cm_df_cell_line["failed_qc"].map(
    {1: "Failed QC", 0: "Passed QC"}
)

cm_df_cell_line["Manual bad segmentation"] = cm_df_cell_line[
    "manual_segmentation_is_bad"
].map(
    {1: "Failed QC", 0: "Passed QC"}
)

cm_df_cell_line["count"] = cm_df_cell_line["count"].fillna(0).astype(int)

cm_df_cell_line.head()


# In[13]:


# -----------------------------
# Cell line mapping (blinded names)
# -----------------------------
cell_line_map = {
    "CHLA-10": "Cell line A",
    "CHLA-113": "Cell line B",
    "CHLA-218": "Cell line C",
    "CHLA-25": "Cell line D",
    "U2-OS": "Cell line E",
}

cm_df_cell_line["Cell_line"] = cm_df_cell_line["Metadata_cell_line"].map(cell_line_map)

# Set the figure size
width = 14
height = 5.5
set_option("figure_size", (width, height))

p_cm = (
    ggplot(cm_df_cell_line)
    + aes(
        x="Manual bad segmentation",
        y="coSMicQC failed_qc",
        fill="proportion",
    )
    + geom_tile()
    + geom_text(aes(label="count"), size=10)
    + facet_grid("annotator ~ Cell_line")
    + scale_fill_gradient(low="white", high="steelblue", limits=(0, 1))
    + labs(
        x="Manual annotation",
        y="coSMicQC prediction",
        fill="Column-normalized\nproportion",
    )
    + theme_bw(base_size=12)
    + theme(legend_position="none")
)

p_cm.save(
    figure_dir / "manual_annotation_confusion_matrix_facet_celltype.png",
    dpi=600,
    width=width,
    height=height,
    verbose=False,
)
p_cm.show()


# In[14]:


# -----------------------------
# sample_row_id uniquely identifies a sample — use it directly
# -----------------------------
id_col = "sample_row_id"

annotators = sorted(manual_eval_df["annotator"].unique())
ann1, ann2 = annotators[0], annotators[1]

# -----------------------------
# Pull sample-level metadata (should be constant across annotators for the same sample_row_id)
# -----------------------------
sample_meta = (
    manual_eval_df
    .drop_duplicates(subset=[id_col])
    [[id_col, "Metadata_cell_line", "failed_qc"]]
)

# -----------------------------
# Pivot manual_segmentation_is_bad wide by annotator, per sample
# -----------------------------
wide_df = manual_eval_df.pivot_table(
    index=id_col,
    columns="annotator",
    values="manual_segmentation_is_bad",
    aggfunc="first",
).reset_index()

wide_df = wide_df.rename(columns={ann1: "annotator_1_seg", ann2: "annotator_2_seg"})
wide_df = wide_df.dropna(subset=["annotator_1_seg", "annotator_2_seg"])

# merge back in cell line + failed_qc
wide_df = wide_df.merge(sample_meta, on=id_col, how="left")
wide_df.head()


# In[15]:


# -----------------------------
# long-format confusion matrix: per cell line + failed_qc,
# annotator 1 (y) vs annotator 2 (x) segmentation labels
# -----------------------------
cm_df_cell_line = (
    wide_df
    .groupby(["Metadata_cell_line", "failed_qc", "annotator_1_seg", "annotator_2_seg"])
    .size()
    .reset_index(name="count")
)

# -----------------------------
# FORCE FULL 2x2 GRID PER FACET
# -----------------------------
full_index = pd.DataFrame(
    itertools.product(
        cm_df_cell_line["Metadata_cell_line"].unique(),
        [0, 1],  # failed_qc
        [0, 1],  # annotator_1_seg
        [0, 1],  # annotator_2_seg
    ),
    columns=["Metadata_cell_line", "failed_qc", "annotator_1_seg", "annotator_2_seg"],
)

cm_df_cell_line = full_index.merge(
    cm_df_cell_line,
    on=["Metadata_cell_line", "failed_qc", "annotator_1_seg", "annotator_2_seg"],
    how="left",
)
cm_df_cell_line["count"] = cm_df_cell_line["count"].fillna(0)

# -----------------------------
# pretty labels
# -----------------------------
label_map = {1: "Failed QC", 0: "Passed QC"}
cm_df_cell_line["coSMicQC failed_qc"] = cm_df_cell_line["failed_qc"].map(label_map)
cm_df_cell_line["Annotator 1"] = cm_df_cell_line["annotator_1_seg"].map(label_map)
cm_df_cell_line["Annotator 2"] = cm_df_cell_line["annotator_2_seg"].map(label_map)
cm_df_cell_line["count"] = cm_df_cell_line["count"].astype(int)

# -----------------------------
# Collapse across failed_qc so each cell line facet has one clean 2x2 grid
# -----------------------------
cm_df_cell_line_collapsed = (
    cm_df_cell_line
    .groupby(["Metadata_cell_line", "Annotator 1", "Annotator 2"])["count"]
    .sum()
    .reset_index()
)

cm_df_cell_line_collapsed["col_sum"] = cm_df_cell_line_collapsed.groupby(
    ["Metadata_cell_line", "Annotator 2"]
)["count"].transform("sum")

cm_df_cell_line_collapsed["proportion"] = (
    cm_df_cell_line_collapsed["count"] / cm_df_cell_line_collapsed["col_sum"]
).fillna(0)

cm_df_cell_line_collapsed.head()


# In[16]:


# -----------------------------
# Cell line mapping (blinded names)
# -----------------------------
cell_line_map = {
    "CHLA-10": "Cell line A",
    "CHLA-113": "Cell line B",
    "CHLA-218": "Cell line C",
    "CHLA-25": "Cell line D",
    "U2-OS": "Cell line E",
}

cm_df_cell_line_collapsed["Cell_line"] = cm_df_cell_line_collapsed[
    "Metadata_cell_line"
].map(cell_line_map)

width = 15
height = 5
set_option("figure_size", (width, height))

p_cm = (
    ggplot(cm_df_cell_line_collapsed)
    + aes(
        x="Annotator 2",
        y="Annotator 1",
        fill="proportion",
    )
    + geom_tile()
    + geom_text(aes(label="count"), size=10)
    + facet_wrap("~Cell_line", nrow=1)
    + scale_fill_gradient(low="white", high="steelblue", limits=(0, 1))
    + labs(
        x="Annotator 2 manual annotation",
        y="Annotator 1 manual annotation",
        fill="Column-normalized\nproportion",
    )
    + theme_bw(base_size=13)
    + theme(legend_position="bottom")
)

p_cm.save(
    figure_dir / "annotator_comparison_confusion_matrix_facet_celltype.png",
    dpi=600,
    width=width,
    height=height,
    verbose=False,
)
p_cm.show()

