"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
import re
from functools import reduce
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from cytodataframe.frame import CytoDataFrame
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)


def identify_outliers(  # noqa: C901, PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
    rule_name: Optional[str] = None,
) -> Union[pd.Series, CytoDataFrame]:
    """
    This function uses z-scoring to format the data for detecting outlier
    nuclei or cells using specific CellProfiler features. Thresholds are
    the number of standard deviations away from the mean, either above
    (positive) or below (negative). We recommend making sure to not use a
    threshold of 0 as that would represent the whole dataset.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file string-based filepath of a
            Parquet, CSV, or TSV file with CytoTable output or similar data.
        feature_thresholds: Dict[str, float]
            One of two options:
            A dictionary with the feature name(s) as the key(s) and their assigned
            threshold for identifying outliers. Positive int for the threshold
            will detect outliers "above" than the mean, negative int will detect
            outliers "below" the mean.
            Or a string which is a named key reference found within
            the feature_thresholds_file yaml file.
        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
            An optional feature thresholds file where thresholds may be
            defined within a file.
        rule_name: Optional[str]
            Optional explicit name of the threshold rule. This is used to
            construct output column names (e.g. `Metadata_cqc_<rule>_is_outlier`).
            If not provided and `feature_thresholds` is a string, the string
            value will be used. If neither is provided, a default name
            of `custom` will be used.
        include_threshold_scores: bool
            Whether to include the threshold scores in addition to whether
            the threshold set passes per row.
        export_path: Optional[str] = None
            An optional path to export the data using CytoDataFrame export
            capabilities. If None no export is performed.
            Note: compatible exports are CSV's, TSV's, and parquet.

    Returns:
        Union[pd.Series, CytoDataFrame]:
            Outlier series with booleans based on whether outliers were detected
            or not for use within other functions.
    """

    df = CytoDataFrame(data=df)
    outlier_df = df

    # --- naming ---
    # Preserve a caller-provided `rule_name` when possible. If the
    # caller passed a string for `feature_thresholds` (a named set in the
    # thresholds file) prefer that as the rule name unless `rule_name` was
    # explicitly provided by the caller. Resolve threshold sets from file
    # when a string is provided.
    if isinstance(feature_thresholds, str):
        if rule_name is None:
            rule_name = feature_thresholds

        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    if rule_name is None:
        rule_name = "custom"

    # Set name prefix for columns based on rule name
    name_prefix = f"Metadata_cqc_{rule_name}"

    zscore_columns = {}
    for feature in feature_thresholds:
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")

        zscore_col = f"{name_prefix}_{feature}_zscore"

        if zscore_col not in outlier_df:
            outlier_df[zscore_col] = scipy_zscore(df[feature])

        zscore_columns[feature] = zscore_col

    def create_condition(feature: str, threshold: float) -> pd.Series:
        if threshold > 0:
            return outlier_df[zscore_columns[feature]] > threshold
        return outlier_df[zscore_columns[feature]] < threshold

    conditions = [
        create_condition(feature, threshold)
        for feature, threshold in feature_thresholds.items()
    ]

    if include_threshold_scores:
        zscore_df = outlier_df[list(zscore_columns.values())]

        is_outlier_series = reduce(operator.and_, conditions).rename(
            f"{name_prefix}_is_outlier"
        )

        result = CytoDataFrame(
            data=pd.concat([zscore_df, is_outlier_series], axis=1),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )
    else:
        result = reduce(operator.and_, conditions)

    if export_path is not None:
        export_df = CytoDataFrame(result) if isinstance(result, pd.Series) else result
        export_df.export(file_path=export_path)

    return result


def find_outliers(
    df: Union[CytoDataFrame, pd.DataFrame, str],
    metadata_columns: List[str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    export_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    This function uses identify_outliers to return a dataframe
    with only the outliers and provided metadata columns.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file string-based filepath of a
            Parquet, CSV, or TSV file with CytoTable output or similar data.
        metadata_columns: List[str]
            List of metadata columns that should be outputted with the outlier data.
        feature_thresholds: Dict[str, float]
            One of two options:
            A dictionary with the feature name(s) as the key(s) and their assigned
            threshold for identifying outliers. Positive int for the threshold
            will detect outliers "above" than the mean, negative int will detect
            outliers "below" the mean.
            Or a string which is a named key reference found within
            the feature_thresholds_file yaml file.
        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
            An optional feature thresholds file where thresholds may be
            defined within a file.
        export_path: Optional[str] = None
            An optional path to export the data using CytoDataFrame export
            capabilities. If None no export is performed.
            Note: compatible exports are CSV's, TSV's, and parquet.

    Returns:
        pd.DataFrame:
            Outlier data frame for the given conditions.
    """

    # Resolve feature_thresholds if provided as a string and remember the name
    rule_name = feature_thresholds if isinstance(feature_thresholds, str) else None
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Determine the columns required for processing
    required_columns = list(feature_thresholds.keys()) + metadata_columns

    # Interpret the df as CytoDataFrame
    df = CytoDataFrame(data=df)[required_columns]

    # if we have nan's in our columns, emit a warning and drop them
    if any(df[list(feature_thresholds.keys())].isna().any()):
        print(
            "Warning: NaN values found in the DataFrame. "
            "These will be dropped before processing."
        )
        df = df.dropna(subset=list(feature_thresholds.keys()))

    # Filter DataFrame for outliers using identify_outliers
    outliers_mask = identify_outliers(
        # Select only the required columns from the DataFrame
        df=df,
        feature_thresholds=feature_thresholds,
        feature_thresholds_file=feature_thresholds_file,
        rule_name=rule_name,
    )
    outliers_df = df[outliers_mask]

    # Print outlier count and range for each feature
    print(
        "Number of outliers:",
        outliers_df.shape[0],
        f"({'{:.2f}'.format((outliers_df.shape[0] / df.shape[0]) * 100)}%)",
    )
    print("Outliers Range:")
    for feature in feature_thresholds:
        print(f"{feature} Min:", outliers_df[feature].min())
        print(f"{feature} Max:", outliers_df[feature].max())

    # Include metadata columns in the output DataFrame
    result = outliers_df[required_columns]

    # Export the file if specified
    if export_path is not None:
        result.export(file_path=export_path)

    # Return the resulting DataFrame
    return result


def label_outliers(  # noqa: C901, PLR0912, PLR0913, PLR0915
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Optional[Union[Dict, str]] = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
    export_mode: Optional[str] = None,
) -> CytoDataFrame:
    """
    Use identify_outliers to label the original dataset for
    where a cell passed or failed the quality control condition(s).

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file path (Parquet, CSV, or TSV).

        feature_thresholds: Union[
            Dict[str, float],
            Dict[str, Dict[str, float]],
            str,
            None,
        ]
            Defines one or more QC conditions.

            - Single condition:
                {"feature": threshold}

            - Multiple conditions:
                {
                    "weird_cells": {"feature1": -1, "feature2": -1},
                    "large_cells": {"feature3": 2},
                }

            - String:
                Named condition from the feature_thresholds_file.

            - None:
                Run all conditions defined in the thresholds file.

        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE
            YAML file containing named threshold conditions.

        include_threshold_scores: bool = False
            If True, include per-feature z-score columns.

        export_path: Optional[str] = None
            Path to export results.

        export_mode: str = "full"
            - "full": entire dataset
            - "annotation": metadata + QC columns

    Returns:
        CytoDataFrame:
            Input dataframe with added QC columns:

            - Metadata_cqc_<condition>_is_outlier
            - (optional) Metadata_cqc_<condition>_<feature>_zscore
    """

    # Ensure CytoDataFrame for consistent handling + metadata preservation
    if not isinstance(df, CytoDataFrame):
        df = CytoDataFrame(data=df)

    custom_attrs = dict(df._custom_attrs)

    # -------------------------
    # Normalize thresholds input
    # -------------------------
    if isinstance(feature_thresholds, dict):
        # Single condition: {"feature": threshold}
        if all(isinstance(v, (int, float)) for v in feature_thresholds.values()):
            thresholds_list = [("custom", feature_thresholds)]

        # Multiple conditions: {"rule_name": {...}}
        elif all(isinstance(v, dict) for v in feature_thresholds.values()):
            thresholds_list = list(feature_thresholds.items())

        else:
            raise ValueError("Invalid feature_thresholds format.")

    elif isinstance(feature_thresholds, str):
        thresholds = read_thresholds_set_from_file(
            feature_thresholds_file=feature_thresholds_file,
        )

        if feature_thresholds not in thresholds:
            raise KeyError(
                f"'{feature_thresholds}' not found in threshold file. "
                f"Available keys: {list(thresholds.keys())}"
            )

        thresholds_list = [(feature_thresholds, thresholds[feature_thresholds])]

    elif feature_thresholds is None:
        # Run all rules in the file
        thresholds_list = list(
            read_thresholds_set_from_file(
                feature_thresholds_file=feature_thresholds_file,
            ).items()
        )

    else:
        raise ValueError("Unsupported feature_thresholds type.")

    # -------------------------
    # Run outlier detection per rule
    # -------------------------
    results = [df]

    for rule_name, thresholds in thresholds_list:
        detected = identify_outliers(
            df=df,
            feature_thresholds=thresholds,
            feature_thresholds_file=feature_thresholds_file,
            include_threshold_scores=True,
            rule_name=rule_name,
        )

        # Identify columns by naming convention
        zscore_cols = [c for c in detected.columns if c.endswith("_zscore")]
        outlier_cols = [c for c in detected.columns if c.endswith("_is_outlier")]

        # Combine multiple feature flags into a single outlier flag
        combined_outlier = (
            detected[outlier_cols[0]]
            if len(outlier_cols) == 1
            else detected[outlier_cols].any(axis=1)
        )

        # Final column name
        condition_col = f"Metadata_cqc_{rule_name}_is_outlier"

        # Place z-score columns first when requested, then the combined
        # outlier annotation so output ordering matches historical output.
        if include_threshold_scores and zscore_cols:
            condition_df = pd.concat(
                [
                    detected[zscore_cols],
                    pd.DataFrame({condition_col: combined_outlier}),
                ],
                axis=1,
            )
        else:
            condition_df = pd.DataFrame({condition_col: combined_outlier})

        results.append(condition_df)

    # -------------------------
    # Merge all results
    # -------------------------
    result = CytoDataFrame(
        pd.concat(results, axis=1).loc[:, lambda x: ~x.columns.duplicated()],
        **custom_attrs,
    )

    # -------------------------
    # Optional export
    # -------------------------
    if export_path is not None:
        if export_mode is None:
            raise ValueError(
                "export_mode must be specified when export_path is provided."
            )

        if export_mode == "annotation":

            def _normalize_col(name: str) -> str:
                return re.sub(r"[^a-z0-9]", "_", name.lower())

            keywords = {"plate", "site", "well", "location_center", "locationcenter"}
            metadata_cols = []

            for col in result.columns:
                if col.startswith("Image_Metadata"):
                    metadata_cols.append(col)
                    continue

                norm = _normalize_col(col)

                if any(k in norm for k in keywords) or (
                    "location" in norm and "center" in norm
                ):
                    metadata_cols.append(col)

            metadata_cols = list(dict.fromkeys(metadata_cols))

            cqc_cols = [
                col for col in result.columns if col.startswith("Metadata_cqc_")
            ]

            export_df = result[metadata_cols + cqc_cols]

        elif export_mode == "full":
            export_df = result

        else:
            raise ValueError(f"Invalid export_mode: {export_mode}")

        export_df.export(file_path=export_path)

    return result


def read_thresholds_set_from_file(
    feature_thresholds_file: str, feature_thresholds: Optional[str] = None
) -> Union[Dict[str, int], Dict[str, Dict[str, int]]]:
    """
    Reads a set of feature thresholds from a specified file.

    This function takes the path to a feature thresholds file and a
    specific feature threshold string, reads the file, and returns
    the thresholds set from the file.

    Args:
        feature_thresholds_file (str):
            The path to the file containing feature thresholds.
        feature_thresholds (Optional str, default None):
            A string specifying the feature thresholds.
            If we have None, return all thresholds.

    Returns:
        dict: A dictionary containing the processed feature thresholds.

    Raises:
        LookupError: If the file does not contain the specified feature_thresholds key.
    """

    # open the yaml file
    with open(feature_thresholds_file, "r") as file:
        thresholds = yaml.safe_load(file)

    # if no feature thresholds name is specified, return all thresholds
    if feature_thresholds is None:
        return thresholds["thresholds"]

    if feature_thresholds not in thresholds["thresholds"]:
        raise LookupError(
            (
                f"Unable to find threshold set by name {feature_thresholds}"
                f" within {feature_thresholds_file}"
            )
        )

    return thresholds["thresholds"][feature_thresholds]
