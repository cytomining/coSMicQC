"""
Utility functions for evaluating neural network models in the context of MOA prediction.
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_abline,
    geom_line,
    geom_point,
    ggplot,
    labs,
    scale_color_manual,
    scale_size_continuous,
    scale_x_continuous,
    scale_y_continuous,
    theme,
    theme_bw,
)
from sklearn.base import BaseEstimator
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import MultiLabelBinarizer


def split_moas(series: pd.Series) -> pd.Series:
    """Split MOA strings into lists of stripped MOA components.

    Parameters
    ----------
    series : pd.Series
        Series containing MOA strings, where multiple MOAs are separated by '|'.

    Returns
    -------
    pd.Series
        Series of lists, where each list contains the individual MOAs for that row.
    """
    return series.fillna("").apply(
        lambda x: [moa.strip() for moa in x.split("|") if moa.strip()]
    )


def get_positive_class_scores(model: BaseEstimator, X: pd.DataFrame) -> pd.DataFrame:
    """Extract positive-class probabilities from a scikit-learn neural network model.

    This function assumes sklearn-compatible estimators that implement `predict_proba`,
    such as `MLPClassifier` or One-vs-Rest wrapped binary classifiers.

    Parameters
    ----------
    model : BaseEstimator
        A fitted scikit-learn estimator with a `predict_proba` method.
    X : pd.DataFrame
        Feature matrix for which to compute probabilities.

    Returns
    -------
    pd.DataFrame
        DataFrame of positive-class probabilities, with one column per class and rows
        aligned with X.
    """
    NDIM_THRESHOLD = 2  # Number of dimensions that indicates a single-label output
    probas = model.predict_proba(X)

    if isinstance(probas, list):
        positive_scores = [
            label_proba[:, 1] if label_proba.ndim == NDIM_THRESHOLD else label_proba
            for label_proba in probas
        ]
        return pd.DataFrame(positive_scores).T

    if probas.ndim == NDIM_THRESHOLD:
        return pd.DataFrame(probas)

    raise ValueError("Unsupported probability output shape returned by model.")


def build_test_frame(
    profiles_df: pd.DataFrame,
    test_compounds: set[str],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter profiles to the test compounds and prepare labels/features.

    Parameters
    ----------
    profiles_df : pd.DataFrame
        DataFrame containing all profiles, including metadata and features.
    test_compounds : set[str]
        Set of compound names to filter the profiles for testing.
    feature_columns : list[str]
        List of column names corresponding to the feature columns in profiles_df.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - test_df: DataFrame of profiles for the test compounds, including metadata and
            MOA as a list.
        - X_test: DataFrame of feature values for the test compounds, aligned with
            test_df.
    """
    test_df = (
        profiles_df[profiles_df["pert_iname"].isin(test_compounds)]
        .reset_index(drop=True)
        .copy()
    )
    test_df = test_df.drop(columns=["moa", "broad_id", "replicate_name"])
    test_df = test_df.rename(columns={"pert_iname": "Metadata_pert_iname"})
    test_df = test_df.assign(moa_list=split_moas(test_df["Metadata_moa"]))

    X_test = test_df[feature_columns].copy()
    return test_df, X_test


def build_selected_moa_pr_curve_df(
    selected_moas: list[str],
    mlb: MultiLabelBinarizer,
    model_scores: dict[str, dict[str, np.ndarray]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return long-form PR coordinates and AP summary for selected MOAs.

    Parameters
    ----------
    selected_moas : list[str]
        List of MOA names to include in the PR curve analysis.
    mlb : MultiLabelBinarizer
        The fitted MultiLabelBinarizer used to transform MOA lists into binary arrays.
    model_scores : dict[str, dict[str, np.ndarray]]
        Dictionary where keys are model labels and values are dictionaries with keys
        "y_true" and "y_score" containing the true binary labels and predicted scores
        for all classes.
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - curve_df: DataFrame with columns "moa", "recall", "precision", and "model" for
            plotting PR curves for each selected MOA.
        - summary_df: DataFrame with columns "moa", "model", "average_precision", and
            "test_positive_examples" summarizing the average precision and
            number of positives for each MOA and model.
    """
    missing_moas = [moa for moa in selected_moas if moa not in mlb.classes_]
    if missing_moas:
        raise ValueError(
            f"These MOAs were not found in the trained label space: {missing_moas}"
        )

    moa_to_index = {
        moa: next(iter(np.where(mlb.classes_ == moa)[0]), None) for moa in selected_moas
    }
    curve_rows = []
    summary_rows = []

    for moa in selected_moas:
        class_idx = moa_to_index[moa]

        for label, values in model_scores.items():
            y_true_class = values["y_true"][:, class_idx]
            y_score_class = values["y_score"][:, class_idx]
            precision, recall, _ = precision_recall_curve(y_true_class, y_score_class)
            ap = average_precision_score(y_true_class, y_score_class)

            curve_rows.append(
                pd.DataFrame(
                    {
                        "moa": moa,
                        "recall": recall,
                        "precision": precision,
                        "model": label,
                    }
                )
            )
            summary_rows.append(
                {
                    "moa": moa,
                    "model": label,
                    "average_precision": ap,
                    "test_positive_examples": int(y_true_class.sum()),
                }
            )

    return pd.concat(curve_rows, ignore_index=True), pd.DataFrame(summary_rows)


def build_moa_failed_fraction_summary(
    post_qc_df: pd.DataFrame,
    test_compounds: set[str],
) -> pd.DataFrame:
    """Summarize post-QC failed-cell fraction at the MOA level for test compounds."""
    metadata_cols = [
        "pert_iname",
        "Metadata_moa",
        "Metadata_sc_count_failed_qc",
        "Metadata_sc_count_passed_qc",
    ]
    test_post_qc_df = (
        post_qc_df.loc[post_qc_df["pert_iname"].isin(test_compounds), metadata_cols]
        .copy()
    )
    test_post_qc_df["Metadata_sc_count_failed_qc"] = test_post_qc_df[
        "Metadata_sc_count_failed_qc"
    ].fillna(0)
    test_post_qc_df["Metadata_sc_count_passed_qc"] = test_post_qc_df[
        "Metadata_sc_count_passed_qc"
    ].fillna(0)

    compound_failed_fraction = (
        test_post_qc_df.groupby(["pert_iname", "Metadata_moa"], dropna=False)
        .agg(
            total_failed_cells=("Metadata_sc_count_failed_qc", "sum"),
            total_passed_cells=("Metadata_sc_count_passed_qc", "sum"),
        )
        .reset_index()
    )
    compound_failed_fraction["total_cells"] = (
        compound_failed_fraction["total_failed_cells"]
        + compound_failed_fraction["total_passed_cells"]
    )
    compound_failed_fraction = compound_failed_fraction.loc[
        compound_failed_fraction["total_cells"] > 0
    ].copy()
    compound_failed_fraction["compound_failed_fraction"] = (
        compound_failed_fraction["total_failed_cells"]
        / compound_failed_fraction["total_cells"]
    )
    compound_failed_fraction["moa_list"] = split_moas(
        compound_failed_fraction["Metadata_moa"]
    )
    compound_failed_fraction = compound_failed_fraction.explode("moa_list")
    compound_failed_fraction = compound_failed_fraction.rename(
        columns={"moa_list": "moa", "pert_iname": "compound"}
    )

    return (
        compound_failed_fraction.groupby("moa", dropna=False)
        .agg(
            avg_compound_failed_fraction=("compound_failed_fraction", "mean"),
            median_compound_failed_fraction=("compound_failed_fraction", "median"),
            moa_compound_count=("compound", "nunique"),
            moa_total_failed_cells=("total_failed_cells", "sum"),
            moa_total_passed_cells=("total_passed_cells", "sum"),
        )
        .reset_index()
    )


def build_held_out_moa_compound_counts(test_targets_df: pd.DataFrame) -> pd.DataFrame:
    """Count unique test compounds contributing to each MOA in the test split."""
    held_out_moa_counts = (
        test_targets_df.loc[:, ["pert_iname", "moa_list"]]
        .explode("moa_list")
        .dropna(subset=["moa_list"])
        .groupby("moa_list", dropna=False)["pert_iname"]
        .nunique()
        .reset_index()
        .rename(
            columns={
                "moa_list": "moa",
                "pert_iname": "test_compound_count",
            }
        )
    )
    return held_out_moa_counts


def make_pr_curve_plot(
    curve_df: pd.DataFrame,
    config: dict,
    output_path: pathlib.Path,
    summary_df: Optional[pd.DataFrame] = None,
) -> ggplot:
    """
    Create and save a plotnine precision-recall curve plot.

    This function visualizes one-vs-rest PR curves for multiple models,
    optionally annotating curves with average precision (AP) values.

    Parameters
    ----------
    curve_df : pd.DataFrame
        DataFrame containing PR curve data with columns:
        ["recall", "precision", "model", ...optional facet column(s)]
    config : dict
        Plot configuration dictionary with keys:
        - "palette": dict mapping model names to colors
        - "facet_formula": optional facet expression (str), e.g. "~moa"
        - "figure_size": optional tuple (width, height)
    output_path : pathlib.Path
        Path where plot image will be saved.
    summary_df : Optional[pd.DataFrame]
        Optional summary table containing average precision per model.

    Returns
    -------
    ggplot
        The generated plotnine object.
    """
    # --- Configuration parsing ---
    palette = config["palette"]
    facet_formula = config.get("facet_formula")
    figure_size = config.get("figure_size", (7, 6))

    # Copy the curve_df to avoid modifying the original DataFrame
    plot_df = curve_df.copy()

    # Ensure consistent ordering of models
    plot_df["model"] = pd.Categorical(
        plot_df["model"], categories=list(palette.keys()), ordered=True
    )

    # --- Label + color handling ---
    if summary_df is not None:
        # Identify AP column safely
        ap_col = next(c for c in summary_df.columns if "average_precision" in c.lower())
        ap_lookup = summary_df.set_index("model")[ap_col]

        label_map = {
            model: f"{model} (AP={ap_lookup.get(model, float('nan')):.3f})"
            for model in palette
        }

        plot_df["model_label"] = plot_df["model"].astype(str).map(label_map)

        color_mapping = {
            label_map[model]: color
            for model, color in palette.items()
            if model in label_map
        }

        color_aes = "model_label"

    else:
        color_mapping = palette
        color_aes = "model"

    # --- Plot construction ---
    p = (
        ggplot(plot_df, aes(x="recall", y="precision", color=color_aes))
        + geom_line(size=1.3)
        + scale_color_manual(values=color_mapping)
        + scale_x_continuous(limits=(0, 1), expand=(0.01, 0.01))
        + scale_y_continuous(limits=(0, 1), expand=(0.01, 0.01))
        + labs(
            x="Recall",
            y="Precision",
            color="Model",
        )
        + theme_bw()
        + theme(
            figure_size=figure_size,
            text=element_text(size=14),
            axis_title=element_text(size=14),
            axis_text=element_text(size=12),
            legend_title=element_text(size=12),
            legend_text=element_text(size=11),
            panel_grid_minor=element_blank(),
        )
    )

    # --- Optional faceting ---
    if facet_formula is not None:
        p = p + facet_wrap(facet_formula, nrow=1)

    # --- Save ---
    p.save(output_path, dpi=600, verbose=False)

    return p


def make_aucpr_change_vs_failed_fraction_plot(
    per_class_aucpr_with_failures: pd.DataFrame,
    output_path: pathlib.Path,
) -> ggplot:
    """Plot per-MOA AUCPR change against mean test-set failed-cell fraction."""

    plot_df = per_class_aucpr_with_failures.dropna(
        subset=["avg_compound_failed_fraction", "delta_aucpr"]
    ).copy()

    delta_std = plot_df["delta_aucpr"].std()
    noise_threshold = 0.5 * delta_std
    print(f"Data-driven QC threshold (0.5 * std): {noise_threshold:.4f}")

    plot_df["improvement_status"] = np.where(
        plot_df["delta_aucpr"] > noise_threshold,
        "Improved after QC",
        np.where(
            plot_df["delta_aucpr"] < -noise_threshold,
            "Declined after QC",
            "No change",
        ),
    )
    plot_df["improvement_status"] = pd.Categorical(
        plot_df["improvement_status"],
        categories=["Improved after QC", "No change", "Declined after QC"],
        ordered=True,
    )

    # Print top movers
    top_inc = plot_df.nlargest(4, "delta_aucpr")[["moa", "delta_aucpr"]]
    top_dec = plot_df.nsmallest(4, "delta_aucpr")[["moa", "delta_aucpr"]]

    print("\nTop 4 increases:")
    print(top_inc.to_string(index=False))

    print("\nTop 4 decreases:")
    print(top_dec.to_string(index=False))

    p = (
        ggplot(
            plot_df,
            aes(
                x="avg_compound_failed_fraction",
                y="delta_aucpr",
                color="improvement_status",
                size="test_compound_count",
            ),
        )
        + geom_abline(slope=0, intercept=0, linetype="dashed", color="black", size=0.6)
        + geom_point(alpha=0.8)
        + scale_color_manual(
            values={
                "Improved after QC": "#06B6D4",
                "Declined after QC": "#F97316",
                "No change": "#7A7A7A",
            }
        )
        + scale_size_continuous(name="Number of\ntest compounds", range=(2.5, 7))
        + scale_x_continuous(limits=(0, 1), expand=(0.01, 0.01))
        + scale_y_continuous(expand=(0.01, 0.01))
        + labs(
            x="Average fraction of failed cells across test compounds in MOA",
            y="AUPR change after QC\n(post-QC - pre-QC)",
            color="Performance change",
        )
        + theme_bw()
        + theme(
            figure_size=(7, 7),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
            panel_grid_minor=element_blank(),
        )
    )

    p.save(output_path, dpi=600, verbose=False)
    return p


def make_aucpr_scatter_plot(
    per_class_aucpr_summary: pd.DataFrame,
    output_path: pathlib.Path,
) -> ggplot:
    """Create and save the pre-vs-post AUPR comparison as a plotnine scatter.

    Parameters
    ----------
    per_class_aucpr_summary : pd.DataFrame
        DataFrame containing "moa", "pre_qc_aucpr", "post_qc_aucpr", "delta_aucpr",
        and "test_compound_count".
    output_path : pathlib.Path
        Path to save the generated scatter plot.

    Returns
    -------
    ggplot
        The generated scatter plot comparing pre-QC and post-QC AUPR values for each
        MOA, colored by improvement status and sized by test compound count.
    """
    # Copy the summary DataFrame to avoid modifying the original
    plot_df = per_class_aucpr_summary.copy()

    # Define a noise threshold for defining meaningful improvement or decline.
    delta_std = plot_df["delta_aucpr"].std()
    # Use half a standard deviation as a conservative "noise floor"
    noise_threshold = 0.5 * delta_std
    print(f"Data-driven QC threshold (0.5 * std): {noise_threshold:.4f}")

    # Recompute delta to ensure consistency (in case upstream changes)
    plot_df["delta_aucpr"] = plot_df["post_qc_aucpr"] - plot_df["pre_qc_aucpr"]

    # Create a categorical column for improvement status based on delta_aucpr
    # with a minimum meaningful change threshold
    plot_df["improvement_status"] = np.where(
        plot_df["delta_aucpr"] > noise_threshold,
        "Improved after QC",
        np.where(
            plot_df["delta_aucpr"] < -noise_threshold,
            "Declined after QC",
            "No change",
        ),
    )

    plot_df["improvement_status"] = pd.Categorical(
        plot_df["improvement_status"],
        categories=[
            "Improved after QC",
            "No change",
            "Declined after QC",
        ],
        ordered=True,
    )

    # Print counts of MOAs in each improvement category
    improvement_counts = (
        plot_df["improvement_status"]
        .value_counts()
        .reindex(["Improved after QC", "No change", "Declined after QC"])
    )
    print("MOA improvement status counts:")
    for status, count in improvement_counts.items():
        print(f"{status}: {count} MOAs")

    # Print the top 6 MOAs by combined AUCPR, sorted by post-QC AUCPR
    top_right_moas = plot_df.nlargest(6, "combined_aucpr").copy()
    top_right_moas = top_right_moas.sort_values("post_qc_aucpr", ascending=False)
    print("Top 6 MOAs by combined AUCPR:")
    for _, row in top_right_moas.iterrows():
        status = row["improvement_status"]
        print(
            f"{row['moa']}: "
            f"x={row['pre_qc_aucpr']:.4f}, "
            f"y={row['post_qc_aucpr']:.4f}, "
            f"{status}"
        )

    # --- Plot construction ---
    p = (
        ggplot(
            plot_df,
            aes(
                x="pre_qc_aucpr",
                y="post_qc_aucpr",
                color="improvement_status",
                    size="test_compound_count",
            ),
        )
        + geom_abline(slope=1, intercept=0, linetype="dashed", color="black", size=0.6)
        + geom_point(alpha=0.8)
        + scale_color_manual(
            values={
                "Improved after QC": "#06B6D4",
                "No change": "#7A7A7A",
                "Declined after QC": "#F97316",
            }
        )
        + scale_size_continuous(name="Number of\ntest compounds", range=(2.5, 7))
        + scale_x_continuous(limits=(0, 1), expand=(0.01, 0.01))
        + scale_y_continuous(limits=(0, 1), expand=(0.01, 0.01))
        + labs(
            x="Pre-QC AUPR",
            y="Post-QC AUPR",
            color="Performance change",
        )
        + theme_bw()
        + theme(
            figure_size=(7, 7),
            legend_title=element_text(size=10),
            legend_text=element_text(size=9),
        )
    )

    p.save(output_path, dpi=600, verbose=False)
    return p
