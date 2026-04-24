"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
import warnings
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from cytodataframe.frame import CytoDataFrame
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)

# `label_outliers` and the helper accept raw caller input, including `None` to mean
# "load all named conditions from the thresholds file".
LabelOutliersFeatureThresholdInput = Optional[
    Union[Dict[str, float], Dict[str, Dict[str, float]], str]
]
# `identify_outliers` only accepts concrete threshold input.
IdentifyOutliersFeatureThresholdInput = Union[
    Dict[str, float], Dict[str, Dict[str, float]], str
]


def _warn_if_inline_thresholds_ignore_file(
    feature_thresholds_file: Optional[str],
    feature_thresholds: LabelOutliersFeatureThresholdInput,
) -> None:
    """Warn when inline thresholds override a non-default thresholds file."""
    if (
        feature_thresholds_file is not None
        and feature_thresholds_file != DEFAULT_QC_THRESHOLD_FILE
        and isinstance(feature_thresholds, dict)
    ):
        warnings.warn(
            (
                "Non-string `feature_thresholds` were provided, so "
                "`feature_thresholds_file` will be ignored."
            ),
            UserWarning,
            stacklevel=3,
        )


def _convert_feature_threshold_input_to_named_threshold_dicts(
    feature_thresholds_file: str,
    feature_thresholds: LabelOutliersFeatureThresholdInput = None,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Convert feature threshold input into named threshold dictionaries
    for processing.

    Args:

        feature_thresholds_file: str
            Path to the YAML file containing threshold definitions.
        feature_thresholds: LabelOutliersFeatureThresholdInput
            If you do not provide feature_thresholds, the function will default to using
            the default YAML file.
            - None: Use all thresholds from the file (only applicable to
            `label_outliers`).
            - str: Named threshold set from the file.
            - Dict[str, float]: Single unnamed threshold set.
            - Dict[str, Dict[str, float]]: Multiple named threshold sets.

    Returns:
        List[Tuple[str, Dict[str, float]]]: A list of (name, thresholds_dict) tuples.
        If a single unnamed threshold dictionary is provided as input, it will be
        returned with the name "custom".

    Raises:
        ValueError: If the input format is invalid.
    """
    # If feature_thresholds is None, return all thresholds from the YAML
    # file (default or specified) -> only used in label_outliers
    if feature_thresholds is None:
        return list(
            read_thresholds_set_from_file(
                feature_thresholds_file=feature_thresholds_file,
            ).items()
        )

    # If feature_thresholds is a string, treat it as a key to look up in the YAML file
    if isinstance(feature_thresholds, str):
        return [
            (
                feature_thresholds,
                read_thresholds_set_from_file(
                    feature_thresholds=feature_thresholds,
                    feature_thresholds_file=feature_thresholds_file,
                ),
            )
        ]

    # If feature_thresholds is a dict, determine if it's single or multiple conditions
    if isinstance(feature_thresholds, dict):
        # Warn user if they provided inline thresholds that will override the file
        _warn_if_inline_thresholds_ignore_file(
            feature_thresholds_file=feature_thresholds_file,
            feature_thresholds=feature_thresholds,
        )
        # Check that the dict is not empty to avoid silent failures later
        if feature_thresholds == {}:
            raise ValueError("feature_thresholds cannot be empty.")

        # If all values are dicts, treat as multiple conditions
        if all(isinstance(v, dict) for v in feature_thresholds.values()):
            return list(feature_thresholds.items())

        # If all values are numeric, treat as a single unnamed condition
        if all(isinstance(v, (int, float)) for v in feature_thresholds.values()):
            return [("custom", feature_thresholds)]

    # If we reach here, the input format is invalid or unexpected
    raise ValueError("Invalid feature_thresholds format.")


def _create_condition_map(
    df: CytoDataFrame,
    outlier_df: CytoDataFrame,
    thresholds: Dict[str, float],
    name_prefix: str,
) -> Tuple[List[pd.Series], Dict[str, str]]:
    """
    Create boolean outlier conditions and z-score column names for one threshold set.

    Args:
        df: CytoDataFrame
            Source data used to calculate z-scores.
        outlier_df: CytoDataFrame
            Working dataframe where z-score columns are stored.
        thresholds: Dict[str, float]
            Dictionary of feature thresholds.
        name_prefix: str
            Prefix to use when naming z-score columns.

    Raises:
        ValueError: If a feature in thresholds does not exist in the DataFrame.

    Returns:
        Tuple[List[pd.Series], Dict[str, str]]
            A list of boolean condition series and a mapping of features to their
            z-score column names.
    """
    # Keep track of z-score column names for each feature
    zscore_columns = {}

    # Loop through each feature and threshold to create z-score columns and conditions
    for feature in thresholds:
        # Check that the feature exists in the DataFrame before attempting to create
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")

        # Set the name for the z-score column for this feature and prefix
        zscore_col = f"{name_prefix}_{feature}_zscore"

        # Reuse any z-score column that has already been computed for this prefix.
        if zscore_col not in outlier_df.columns:
            outlier_df[zscore_col] = scipy_zscore(df[feature])

        # Store the z-score column name for this feature to use when creating conditions
        zscore_columns[feature] = zscore_col

    # Create a boolean condition for each feature based on the specified threshold.
    conditions = [
        (
            outlier_df[zscore_columns[feature]] > threshold
            if threshold > 0
            else outlier_df[zscore_columns[feature]] < threshold
        )
        for feature, threshold in thresholds.items()
    ]

    return conditions, zscore_columns


def _build_filter_display_options(
    threshold_sets: List[Dict[str, float]],
    source_df: pd.DataFrame,
    existing_display_options: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build display options for thresholded columns in CytoDataFrame plots.

    Thresholds are provided to coSMicQC as z-scores, but CytoDataFrame plot lines
    should use feature-scale values. We therefore convert each z-score threshold to
    the corresponding feature value using ``mean + z * std``.

    Args:
        threshold_sets: List[Dict[str, float]]
            One or more threshold mappings from feature name to z-score threshold.
        source_df: pd.DataFrame
            DataFrame containing feature columns used to compute feature-scale
            thresholds.
        existing_display_options: Optional[Dict[str, object]]
            Existing plot display options to preserve and extend.

    Returns:
        Dict[str, object]
            Display options including ``filter_columns`` and
            ``filter_plot_thresholds`` for threshold overlays.
    """
    filter_plot_thresholds: Dict[str, float] = {}
    for thresholds in threshold_sets:
        for feature, zscore_threshold in thresholds.items():
            feature_series = source_df[feature]
            feature_mean = feature_series.mean()
            feature_std = feature_series.std(ddof=0)
            if pd.isna(feature_mean):
                filter_plot_thresholds[feature] = float(zscore_threshold)
            elif pd.isna(feature_std) or feature_std == 0:
                filter_plot_thresholds[feature] = float(feature_mean)
            else:
                filter_plot_thresholds[feature] = float(
                    feature_mean + (zscore_threshold * feature_std)
                )

    return {
        **(existing_display_options or {}),
        "filter_columns": list(filter_plot_thresholds.keys()),
        "filter_plot_thresholds": filter_plot_thresholds,
    }


def identify_outliers(  # noqa: PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: IdentifyOutliersFeatureThresholdInput,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    condition_name: Optional[str] = None,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
) -> Union[
    pd.Series,
    CytoDataFrame,
]:
    """
    This function uses z-scoring to format the data for detecting outlier
    nuclei or cells using specific CellProfiler features.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            Input dataframe or file path.
        feature_thresholds: IdentifyOutliersFeatureThresholdInput
            Either:
            1. {feature: threshold}
            2. {condition_name: {feature: threshold}}
            3. string key from a YAML file
        include_threshold_scores: bool
            Whether to include z-score columns in the output.
        condition_name: Optional[str]
            Optional explicit name to use for CQC columns when
            `feature_thresholds` is a single dict (only features and thresholds).
            Default name is "custom" if not provided.
        export_path: Optional[str]
            If provided, export the result.

    Returns:
        Union[pd.Series, CytoDataFrame]
            Return shape depends on whether one or multiple conditions are provided:

            - Single condition:
                Returns a boolean `pd.Series` if `include_threshold_scores` is False.
                Returns a `CytoDataFrame` with z-score columns and the outlier column
                if `include_threshold_scores` is True.

            - Multiple conditions via `Dict[str, Dict[str, float]]`:
                Returns a `CytoDataFrame`.
                If `include_threshold_scores` is False, it contains one outlier
                column per condition.
                If `include_threshold_scores` is True, it contains the per-condition
                z-score columns plus one outlier column per condition.
    """
    # Convert input to CytoDataFrame if not already one
    df = CytoDataFrame(data=df)
    # Create a copy of the dataframe to avoid modifying the original
    outlier_df = df.copy()

    # Determine the key for naming if feature_thresholds is a string
    thresholds_key = feature_thresholds if isinstance(feature_thresholds, str) else None

    _warn_if_inline_thresholds_ignore_file(
        feature_thresholds_file=feature_thresholds_file,
        feature_thresholds=feature_thresholds,
    )

    # Read thresholds from file if a string key is provided
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Confirm that if a dict is provided, it is not empty to avoid silent failures later
    if isinstance(feature_thresholds, dict) and feature_thresholds == {}:
        raise ValueError("feature_thresholds cannot be empty.")

    # Handle multiple conditions if feature_thresholds is a dict of dicts
    if feature_thresholds and isinstance(next(iter(feature_thresholds.values())), dict):
        # We have multiple named conditions, so combine the per-condition results
        # into a single CytoDataFrame.
        result_frames = []

        # Loop through each condition set and create the corresponding outlier columns
        for name, thresholds in feature_thresholds.items():
            # Set name prefix for this condition
            name_prefix = f"Metadata_cqc_{name}"

            # Create the condition map for this set of thresholds
            conditions, zscore_columns = _create_condition_map(
                df=df,
                outlier_df=outlier_df,
                thresholds=thresholds,
                name_prefix=name_prefix,
            )

            # Combine conditions with AND logic to determine overall outlier status
            # for this condition set
            is_outlier_series = reduce(operator.and_, conditions).rename(
                f"{name_prefix}_is_outlier"
            )

            # If we want to include threshold scores, we either return just the boolean
            # series or a DataFrame with z-scores and the boolean column
            if include_threshold_scores:
                result_frames.append(
                    outlier_df[[*list(zscore_columns.values()), is_outlier_series.name]]
                    if is_outlier_series.name in outlier_df.columns
                    else pd.concat(
                        [outlier_df[list(zscore_columns.values())], is_outlier_series],
                        axis=1,
                    )
                )
            else:
                result_frames.append(is_outlier_series.to_frame())

        result = CytoDataFrame(
            data=pd.concat(result_frames, axis=1),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )

        if export_path is not None:
            result.export(file_path=export_path)

        return result

    # If we have a single condition (either from a dict or from a string key),
    # we create one set of columns
    name_prefix = f"Metadata_cqc_{condition_name or thresholds_key or 'custom'}"

    # Create the condition map for this set of thresholds
    conditions, zscore_columns = _create_condition_map(
        df=df,
        outlier_df=outlier_df,
        thresholds=feature_thresholds,
        name_prefix=name_prefix,
    )

    # Combine conditions with AND logic to determine overall outlier status
    # for this condition set
    is_outlier_series = reduce(operator.and_, conditions).rename(
        f"{name_prefix}_is_outlier"
    )

    # If we want to include threshold scores, we either return just the boolean series
    # or a DataFrame with z-scores and the boolean column
    if include_threshold_scores:
        zscore_df = outlier_df[list(zscore_columns.values())]
        result = CytoDataFrame(
            data=pd.concat([zscore_df, is_outlier_series], axis=1),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )
    else:
        result = is_outlier_series

    # Export if export_path is provided
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
) -> CytoDataFrame:
    """
    This function uses identify_outliers to return a dataframe
    with only the outliers and provided metadata columns.

    NOTE: This function can only be used with a single set of feature thresholds
    for finding what outliers look like in the data. For multiple sets of
    feature thresholds, use label_outliers instead and filter the results for
    the condition(s) of interest.

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
        CytoDataFrame:
            Outlier data frame for the given conditions.
    """
    _warn_if_inline_thresholds_ignore_file(
        feature_thresholds_file=feature_thresholds_file,
        feature_thresholds=feature_thresholds,
    )
    input_display_options = (
        dict(df._custom_attrs.get("display_options") or {})
        if isinstance(df, CytoDataFrame)
        else None
    )
    # Convert input to CytoDataFrame if it's a file path or a pandas DataFrame
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Determine the required columns for processing and output
    required_columns = list(feature_thresholds.keys()) + metadata_columns

    # Ensure the DataFrame contains the required columns and convert to CytoDataFrame.
    # Keep projected attrs so we can preserve them even if downstream ops
    # (e.g., dropna) return a pandas.DataFrame.
    projected_df = CytoDataFrame(data=df)[required_columns]
    projected_custom_attrs = dict(projected_df._custom_attrs)
    df = projected_df

    # Check for NaN values in the required feature columns and warn if any are found
    if any(df[list(feature_thresholds.keys())].isna().any()):
        print(
            "Warning: NaN values found in the DataFrame. "
            "These will be dropped before processing."
        )
        df = df.dropna(subset=list(feature_thresholds.keys()))

    # Use identify_outliers to get a boolean mask of outliers based on the
    # provided thresholds
    outliers_mask = identify_outliers(
        df=df,
        feature_thresholds=feature_thresholds,
        feature_thresholds_file=feature_thresholds_file,
    )
    # Filter the original DataFrame to return only the outliers along with the
    # specified metadata columns
    outliers_df = df[outliers_mask]

    # Print summary statistics about the outliers found
    print(
        "Number of outliers:",
        outliers_df.shape[0],
        f"({'{:.2f}'.format((outliers_df.shape[0] / df.shape[0]) * 100)}%)",
    )
    print("Outliers Range:")
    for feature in feature_thresholds:
        print(f"{feature} Min:", outliers_df[feature].min())
        print(f"{feature} Max:", outliers_df[feature].max())

    # Select only the required columns for output (metadata + features)
    result = outliers_df[required_columns]
    custom_attrs = (
        dict(df._custom_attrs)
        if isinstance(df, CytoDataFrame)
        else projected_custom_attrs
    )
    if input_display_options is not None:
        custom_attrs["display_options"] = input_display_options
    result = CytoDataFrame(data=result, **custom_attrs)

    # Export if export_path is provided
    if export_path is not None:
        result.export(file_path=export_path)

    return result


def label_outliers(  # noqa: PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: LabelOutliersFeatureThresholdInput = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
    export_as_annotations: bool = False,
    annotation_metadata_columns: Optional[List[str]] = None,
) -> CytoDataFrame:
    """
    This function labels outliers in the input dataframe based on specified
    feature thresholds and exports the whole dataframe or an annotations file with
    just metadata and outlier labels.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            DataFrame or file path (Parquet, CSV, or TSV).

        feature_thresholds: LabelOutliersFeatureThresholdInput
            Defines one or more QC conditions.

            - Single condition:
                {"feature": threshold}

            - Multiple conditions:
                {
                    "undersegmented_cells": {"feature1": -1, "feature2": -1},
                    "oversegmented_nuclei": {"feature3": 2},
                }

            - String:
                Named condition from the feature_thresholds_file.

            - None:
                Run all conditions defined in the thresholds file.

        feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE
            YAML file containing named threshold conditions.

        include_threshold_scores: bool = False
            If True, include per-feature z-score columns in the output.

        export_path: Optional[str] = None
            Path to export results.

        export_as_annotations: bool = False
            If True, export only metadata + QC columns (annotations file).
            If False, export the full dataset.
        annotation_metadata_columns: Optional[List[str]] = None
            Metadata columns to include when `export_as_annotations=True`.
            If annotation export is requested, these columns are required and
            will be written alongside the generated `Metadata_cqc_*` columns.

    Returns:
        CytoDataFrame:
            Either the full dataframe or only metadata with added QC columns:

            - Metadata_cqc_<condition>_is_outlier
            - (optional) Metadata_cqc_<condition>_<feature>_zscore if
            include_threshold_scores=True

            When ``export_as_annotations=True``, the exported file contains only
            `annotation_metadata_columns` plus QC-related columns.
    """
    # Convert input to CytoDataFrame if it's a file path or a pandas DataFrame
    if not isinstance(df, CytoDataFrame):
        df = CytoDataFrame(data=df)

    # Keep track of custom attributes to preserve them in the output
    custom_attrs = dict(df._custom_attrs)

    # Convert feature_thresholds input into a list of (name, thresholds) tuples
    # for consistent processing
    threshold_sets = _convert_feature_threshold_input_to_named_threshold_dicts(
        feature_thresholds=feature_thresholds,
        feature_thresholds_file=feature_thresholds_file,
    )

    # Start with the original dataframe and iteratively add outlier columns
    # for each condition
    base_df = df.drop(
        columns=[c for c in df.columns if c.startswith("Metadata_cqc_")],
        errors="ignore",
    )
    results = [base_df]

    # Loop through each set of thresholds and identify outliers
    for name, thresholds in threshold_sets:
        detected = identify_outliers(
            df=df,
            feature_thresholds=thresholds,
            feature_thresholds_file=feature_thresholds_file,
            condition_name=name,
            include_threshold_scores=include_threshold_scores,
        )

        # If the result is a Series, convert it to a DataFrame for
        # consistent concatenation
        if isinstance(detected, pd.Series):
            condition_df = detected.to_frame()
        else:
            condition_df = detected

        # Append the condition DataFrame to the results list for later concatenation
        results.append(condition_df)

    # Concatenate all results along the columns, ensuring we don't duplicate columns
    clean_results = [r for r in results if isinstance(r, (pd.Series, pd.DataFrame))]

    # Create the final result DataFrame by concatenating the original data
    # with the new outlier columns
    custom_attrs["display_options"] = _build_filter_display_options(
        threshold_sets=[thresholds for _, thresholds in threshold_sets],
        source_df=df,
        existing_display_options=custom_attrs.get("display_options"),
    )
    result = CytoDataFrame(
        pd.concat(clean_results, axis=1),
        **custom_attrs,
    )

    # Export if export_path is provided
    if export_path is not None:
        export_df = result.copy()

        # If exporting as annotations, filter to only metadata and CQC columns
        if export_as_annotations:
            if annotation_metadata_columns is None:
                raise ValueError(
                    "`annotation_metadata_columns` must be provided when "
                    "`export_as_annotations=True`."
                )

            # Check that all specified annotation metadata columns are present in the df
            missing_metadata_cols = [
                col
                for col in annotation_metadata_columns
                if col not in export_df.columns
            ]
            # Raise error if any specified metadata columns are missing
            if missing_metadata_cols:
                raise ValueError(
                    "The following annotation metadata columns were not found in the "
                    f"dataframe: {missing_metadata_cols}"
                )

            # Set CQC columns as those that start with "Metadata_cqc_" to capture
            # the outlier labels and z-score columns
            cqc_cols = [
                col for col in export_df.columns if col.startswith("Metadata_cqc_")
            ]

            # Filter the export_df to only include metadata and CQC columns
            export_df = export_df[annotation_metadata_columns + cqc_cols]

        # Use CytoDataFrame export capabilities to export the result
        CytoDataFrame(data=export_df, **custom_attrs).export(file_path=export_path)

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
    # Read the thresholds from the specified YAML file
    with open(feature_thresholds_file, "r") as file:
        thresholds = yaml.safe_load(file)

    # If feature_thresholds is None, return all thresholds from the file
    if feature_thresholds is None:
        return thresholds["thresholds"]

    # If feature_thresholds is a string, look it up in the thresholds and
    # return the corresponding set
    if feature_thresholds not in thresholds["thresholds"]:
        raise LookupError(
            (
                f"Unable to find threshold set by name {feature_thresholds}"
                f" within {feature_thresholds_file}"
            )
        )

    return thresholds["thresholds"][feature_thresholds]
