"""
Module for detecting various quality control aspects from source data.
"""

import operator
import pathlib
from functools import reduce
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from cytodataframe.frame import CytoDataFrame
from scipy.stats import zscore as scipy_zscore

DEFAULT_QC_THRESHOLD_FILE = (
    f"{pathlib.Path(__file__).parent!s}/data/qc_nuclei_thresholds_default.yml"
)


def identify_outliers(
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Union[Dict[str, float], str],
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
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

    # Ensure the input is a CytoDataFrame, converting if necessary
    df = CytoDataFrame(data=df)

    # reference the df for a new outlier_df
    outlier_df = df

    # Define the naming scheme for z-score columns based on thresholds
    thresholds_name = (
        f"cqc.{feature_thresholds}"
        if isinstance(feature_thresholds, str)
        else "cqc.custom"
    )

    # If feature_thresholds is a string, load the thresholds from the specified file
    if isinstance(feature_thresholds, str):
        feature_thresholds = read_thresholds_set_from_file(
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
        )

    # Dictionary to store mappings of features to their z-score column names
    zscore_columns = {}
    for feature in feature_thresholds:
        # Ensure the feature exists in the DataFrame
        if feature not in df.columns:
            raise ValueError(f"Feature '{feature}' does not exist in the DataFrame.")

        # Construct the z-score column name
        zscore_col = f"{thresholds_name}.Z_Score.{feature}"

        # Calculate and store z-scores only if not already present
        if zscore_col not in outlier_df:
            outlier_df[zscore_col] = scipy_zscore(df[feature])

        # Add the column name to the zscore_columns dictionary
        zscore_columns[feature] = zscore_col

    # Helper function to create outlier detection conditions
    def create_condition(feature: str, threshold: float) -> pd.Series:
        # Positive threshold checks for outliers above the mean
        if threshold > 0:
            return outlier_df[zscore_columns[feature]] > threshold
        # Negative threshold checks for outliers below the mean
        return outlier_df[zscore_columns[feature]] < threshold

    # Generate outlier detection conditions for all features
    conditions = [
        create_condition(feature, threshold)
        for feature, threshold in feature_thresholds.items()
    ]

    # Construct the result based on whether threshold scores should be included
    if include_threshold_scores:
        # Extract z-score columns for each feature
        zscore_df = outlier_df[list(zscore_columns.values())]

        # Combine conditions into a single Series indicating outlier status
        is_outlier_series = reduce(operator.and_, conditions).rename(
            f"{thresholds_name}.is_outlier"
        )

        # Combine z-scores and outlier status into a single DataFrame
        result = CytoDataFrame(
            data=pd.concat([zscore_df, is_outlier_series], axis=1),
            data_context_dir=df._custom_attrs["data_context_dir"],
            data_mask_context_dir=df._custom_attrs["data_mask_context_dir"],
        )
    else:
        # Combine conditions into a single Series of boolean values
        result = reduce(operator.and_, conditions)

    # Export the result if an export path is specified
    if export_path is not None:
        export_df = CytoDataFrame(result) if isinstance(result, pd.Series) else result
        export_df.export(file_path=export_path)

    # Return the resulting Series or DataFrame
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

    # Resolve feature_thresholds if provided as a string
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


def label_outliers(
    df: Union[CytoDataFrame, pd.DataFrame, str],
    feature_thresholds: Optional[Union[Dict[str, float], str]] = None,
    feature_thresholds_file: Optional[str] = DEFAULT_QC_THRESHOLD_FILE,
    include_threshold_scores: bool = False,
    export_path: Optional[str] = None,
) -> CytoDataFrame:
    """
    Use identify_outliers to label the original dataset for
    where a cell passed or failed the quality control condition(s).

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
            include_threshold_scores: bool = False
                Whether to include the scores in addition to whether an outlier
                was detected or not.
            export_path: Optional[str] = None
                An optional path to export the data using CytoDataFrame export
                capabilities. If None no export is performed.
                Note: compatible exports are CSV's, TSV's, and parquet.

        Returns:
            CytoDataFrame:
                Full dataframe with optional scores and outlier boolean column.
    """

    # interpret the df as CytoDataFrame
    if not isinstance(df, CytoDataFrame):
        df = CytoDataFrame(data=df)

    # store the custom attributes
    custom_attrs = dict(df._custom_attrs)

    # for single outlier processing
    if isinstance(feature_thresholds, (str, dict)):
        # return the outlier dataframe for one threshold rule
        identified_outliers = identify_outliers(
            df=df,
            feature_thresholds=feature_thresholds,
            feature_thresholds_file=feature_thresholds_file,
            include_threshold_scores=include_threshold_scores,
        )

        result = CytoDataFrame(
            data=pd.concat(
                [
                    df,
                    (
                        identified_outliers
                        if isinstance(identified_outliers, pd.DataFrame)
                        else CytoDataFrame(
                            {
                                (
                                    f"cqc.{feature_thresholds}.is_outlier"
                                    if isinstance(feature_thresholds, str)
                                    else "cqc.custom.is_outlier"
                                ): identified_outliers
                            }
                        )
                    ),
                ],
                axis=1,
            ),
            # reuse the custom attributes
            **custom_attrs,
        )

    # for multiple outlier processing
    elif feature_thresholds is None:
        # return the outlier dataframe for all threshold rules
        labeled_df = pd.concat(
            [df]
            + [
                # identify outliers for each threshold rule
                identify_outliers(
                    df=df,
                    feature_thresholds=thresholds,
                    feature_thresholds_file=feature_thresholds_file,
                    include_threshold_scores=include_threshold_scores,
                )
                # loop through each threshold rule
                for thresholds in read_thresholds_set_from_file(
                    feature_thresholds_file=feature_thresholds_file,
                )
            ],
            axis=1,
        )

        # return a dataframe with deduplicated columns by name
        result = CytoDataFrame(
            labeled_df.loc[:, ~labeled_df.columns.duplicated()],
            # reuse the custom attributes
            **custom_attrs,
        )

    # export the file if specified
    if export_path is not None:
        result.export(file_path=export_path)

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
