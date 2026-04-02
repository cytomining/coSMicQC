"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import yaml
from cytodataframe.frame import CytoDataFrame
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)


def _normalize_feature_thresholds(
    feature_thresholds: Optional[Union[Dict, str]],
    feature_thresholds_file: str,
) -> List[Tuple[str, Dict[str, float]]]:
    """
    Normalize threshold input into a list of named threshold dictionaries.

    Args:
        feature_thresholds: Optional[Union[Dict, str]]
            - None: Use all thresholds from the file.
            - str: Named threshold set from the file.
            - Dict[str, float]: Single unnamed threshold set.
            - Dict[str, Dict[str, float]]: Multiple named threshold sets.
        feature_thresholds_file: str
            Path to the YAML file containing threshold definitions.

    Returns:
        List[Tuple[str, Dict[str, float]]]: List of (name, thresholds) tuples.

    Raises:
        ValueError: If the input format is invalid.
    """
    # If feature_thresholds is None, return all thresholds from the YAML
    # file (default or specified)
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
        if not feature_thresholds:
            raise ValueError("feature_thresholds cannot be empty.")

        # If all values are dicts, treat as multiple conditions
        if all(isinstance(v, dict) for v in feature_thresholds.values()):
            return list(feature_thresholds.items())

        # If all values are numeric, treat as a single unnamed condition
        if all(isinstance(v, (int, float)) for v in feature_thresholds.values()):
            return [("custom", feature_thresholds)]

    # If we reach here, the input format is invalid or unexpected
    raise ValueError("Invalid feature_thresholds format.")


def identify_outliers(  # noqa: C901, PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Union[Dict[str, float], Dict[str, Dict[str, float]], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    condition_name: Optional[str] = None,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
) -> Union[pd.Series, CytoDataFrame]:
    """
    This function uses z-scoring to format the data for detecting outlier
    nuclei or cells using specific CellProfiler features.

    Args:
        df: Union[CytoDataFrame, pd.DataFrame, str]
            Input dataframe or file path.
        feature_thresholds: Union[Dict[str, float], Dict[str, Dict[str, float]], str]
            Either:
            1. {feature: threshold}
            2. {set_name: {feature: threshold}}
            3. string key from a YAML file
        include_threshold_scores: bool
            Whether to include z-score columns in the output.
        condition_name: Optional[str]
            Optional explicit name to use for CQC columns when
            `feature_thresholds` is a single dict.
        export_path: Optional[str]
            If provided, export the result.

    Returns:
        Union[pd.Series, CytoDataFrame]
            Boolean series or full dataframe with optional z-score columns.
    """
    # Convert input to CytoDataFrame if not already one
    df = CytoDataFrame(data=df)
    # Create a copy of the dataframe to avoid modifying the original
    outlier_df = df.copy()

    # Determine the key for naming if feature_thresholds is a string
    thresholds_key = feature_thresholds if isinstance(feature_thresholds, str) else None

    # Read thresholds from file if a string key is provided
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Validate that feature_thresholds is not empty if it's a dict
    if isinstance(feature_thresholds, dict) and not feature_thresholds:
        raise ValueError("feature_thresholds cannot be empty.")

    def create_condition_map(
        thresholds: Dict[str, float], name_prefix: str
    ) -> Tuple[List[pd.Series], Dict[str, str]]:
        """
        Create a list of boolean series representing outlier conditions and
        a mapping of features to their z-score column names.

        Args:
            thresholds (Dict[str, float]): Dictionary of feature thresholds.
            name_prefix (str): Prefix to use for naming z-score columns.

        Raises:
            ValueError: If a feature in thresholds does not exist in the DataFrame.
        Returns:
            Tuple[List[pd.Series], Dict[str, str]]: A tuple containing a list of boolean
            series representing conditions and a dictionary mapping features to their
            z-score column names.
        """
        # Initialize a dictionary to keep track of z-score column names for each feature
        zscore_columns = {}

        # Loop through each feature and threshold to create z-score cols and conditions
        for feature in thresholds:
            if feature not in df.columns:
                raise ValueError(
                    f"Feature '{feature}' does not exist in the DataFrame."
                )
            # Set the name of the column for zscores for this feature
            zscore_col = f"{name_prefix}_{feature}_zscore"

            # Only compute z-scores if we haven't already for this feature
            # (avoid duplication)
            if zscore_col not in outlier_df.columns:
                outlier_df[zscore_col] = scipy_zscore(df[feature])

            # Store the z-score column name for this feature
            zscore_columns[feature] = zscore_col

        # Create a boolean series for each condition based on the thresholds
        conditions = [
            (
                outlier_df[zscore_columns[feature]] > threshold
                if threshold > 0
                else outlier_df[zscore_columns[feature]] < threshold
            )
            for feature, threshold in thresholds.items()
        ]

        return conditions, zscore_columns

    # Handle multiple conditions if feature_thresholds is a dict of dicts
    if feature_thresholds and isinstance(next(iter(feature_thresholds.values())), dict):
        # We have multiple named conditions, so we will create separate columns for each
        results = {}

        # Loop through each condition set and create the corresponding outlier columns
        for name, thresholds in feature_thresholds.items():
            # Set name prefix for this condition
            name_prefix = f"Metadata_cqc_{name}"

            # Create the condition map for this set of thresholds
            conditions, zscore_columns = create_condition_map(thresholds, name_prefix)

            # Combine conditions with AND logic to determine overall outlier status
            # for this condition set
            is_outlier_series = reduce(operator.and_, conditions).rename(
                f"{name_prefix}_is_outlier"
            )

            # If we want to include threshold scores, we either return just the boolean
            # series or a DataFrame with z-scores and the boolean column
            if include_threshold_scores:
                zscore_df = outlier_df[list(zscore_columns.values())]
                result = CytoDataFrame(
                    data=pd.concat([zscore_df, is_outlier_series], axis=1),
                    data_context_dir=df._custom_attrs["data_context_dir"],
                    data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
                )
            else:
                result = is_outlier_series

            # Store the result for this condition set in the results dictionary
            results[name] = result

        return results

    # If we have a single condition (either from a dict or from a string key),
    # we create one set of columns
    name_prefix = f"Metadata_cqc_{condition_name or thresholds_key or 'custom'}"

    # Create the condition map for this set of thresholds
    conditions, zscore_columns = create_condition_map(feature_thresholds, name_prefix)

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
    # Convert input to CytoDataFrame if it's a file path or a pandas DataFrame
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Determine the required columns for processing and output
    required_columns = list(feature_thresholds.keys()) + metadata_columns

    # Ensure the DataFrame contains the required columns and convert to CytoDataFrame
    df = CytoDataFrame(data=df)[required_columns]

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

    # Export if export_path is provided
    if export_path is not None:
        result.export(file_path=export_path)

    return result


def label_outliers(  # noqa: PLR0913
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Optional[Union[Dict, str]] = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
    export_as_annotations: bool = False,
) -> CytoDataFrame:
    """
    This function labels outliers in the input dataframe based on specified
    feature thresholds and exports the whole dataframe or an annotations file with
    just metadata and outlier labels.

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

        export_as_annotations: bool = False
            If True, export only metadata + QC columns (annotations file).
            If False, export the full dataset.

    Returns:
        CytoDataFrame:
            Input dataframe with added QC columns:

            - Metadata_cqc_<condition>_is_outlier
            - (optional) Metadata_cqc_<condition>_<feature>_zscore
    """
    # Convert input to CytoDataFrame if it's a file path or a pandas DataFrame
    if not isinstance(df, CytoDataFrame):
        df = CytoDataFrame(data=df)

    # Keep track of custom attributes to preserve them in the output
    custom_attrs = dict(df._custom_attrs)

    # Normalize feature_thresholds into a list of (name, thresholds) tuples
    # for consistent processing
    thresholds_list = _normalize_feature_thresholds(
        feature_thresholds=feature_thresholds,
        feature_thresholds_file=feature_thresholds_file,
    )

    # Initialize results with the original dataframe to ensure we always have a base
    # to concatenate to
    results = [df]

    # Loop through each set of thresholds and identify outliers
    for name, thresholds in thresholds_list:
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

    # Create the final result DataFrame, ensuring we don't have duplicate columns
    # from multiple conditions
    result = CytoDataFrame(
        pd.concat(clean_results, axis=1).loc[:, lambda x: ~x.columns.duplicated()],
        **custom_attrs,
    )

    # Export if export_path is provided
    if export_path is not None:
        export_df = result.copy()

        # If exporting as annotations, filter to only metadata and CQC columns
        if export_as_annotations:
            # Set metadata columns as those that start with "Image_Metadata_" or
            # "Nuclei_Location" to capture common CellProfiler metadata patterns
            # for downstream annotation
            metadata_cols = [
                col
                for col in export_df.columns
                if col.startswith(("Image_Metadata_", "Nuclei_Location"))
            ]
            # Set CQC columns as those that start with "Metadata_cqc_" to capture
            # the outlier labels and z-score columns
            cqc_cols = [
                col for col in export_df.columns if col.startswith("Metadata_cqc_")
            ]

            # Filter the export_df to only include metadata and CQC columns
            export_df = export_df[metadata_cols + cqc_cols]

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
