"""
Tests cosmicqc contamination detector module
"""

from unittest.mock import patch

import matplotlib
import pandas as pd

from cosmicqc import detection as cd

matplotlib.use("Agg")  # non-GUI backend for headless environments


def test_skewness_cytoplasm_texture(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test skewness of cytoplasm texture
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Calculate skewness
    is_skewed = detector._skewness_test_cytoplasm_texture()

    # Check if the skewness is as expected
    assert is_skewed


def test_variability_formfactor(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test variability of cytoplasm texture
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Calculate variability
    is_variable = detector._variability_test_formfactor()

    # Check if the variability is as expected
    assert not is_variable


def test_check_skew_and_variable_basic(
    cytotable_NF1_contamination_data_df: pd.DataFrame,
):
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )
    detector.check_skew_and_variable()

    assert detector.is_skewed
    assert not detector.is_variable


def test_calculate_texture_mean(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test if there is whole plate or partial plate contamination based on texture mean.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Determine if whole plate is contaminated or partial
    whole_plate_contamination_texture = detector._calculate_texture_mean()

    # Assert the result is False as we expect partial plate contamination
    assert not whole_plate_contamination_texture


def test_calculate_formfactor_mean(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test if there is whole plate or partial plate contamination
    based on nuclei shape mean.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Determine if whole plate is contaminated or partial
    whole_plate_contamination_formfactor = detector._calculate_formfactor_mean()

    # Assert the result is False the data was not variable
    assert not whole_plate_contamination_formfactor


def test_check_feature_means(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test the behavior of step 2 in the contamination detection process.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Execute step 1 as it is required for step 2
    detector.check_skew_and_variable()

    # Execute step 2
    detector.check_feature_means()

    # Check if the results are as expected
    assert not detector.whole_plate_contamination_texture


def test_find_texture_outliers(
    cytotable_NF1_contamination_data_df: pd.DataFrame,
):
    # Instantiate the ContaminationDetector with the DataFrame
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )
    outliers_df = detector._find_texture_outliers()

    # Assert the outliers DataFrame is not empty and has the expected shape
    assert isinstance(outliers_df, pd.DataFrame)
    assert not outliers_df.empty
    assert outliers_df.shape[0] == 242


def test_get_outlier_proportion_per_well(
    cytotable_NF1_contamination_data_df: pd.DataFrame,
):
    """
    Test the get_outlier_proportion_per_well method.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Execute step 1 as it is required for step 2
    detector.check_skew_and_variable()

    # Execute step 2 prior to plotting (or it will yield an error)
    detector.check_feature_means()
    # Run the method
    outlier_proportions_df = detector._get_outlier_proportion_per_well()

    # Assert the returned DataFrame is not empty
    assert isinstance(outlier_proportions_df, pd.DataFrame)
    assert not outlier_proportions_df.empty
    assert outlier_proportions_df.shape[0] == 12

    # Assert the DataFrame contains the expected columns
    expected_columns = ["Well", "Proportion", "CellCount", "Image_Metadata_Well"]
    assert all(col in outlier_proportions_df.columns for col in expected_columns)

    # Assert the proportions are within the range 0 to 100
    assert outlier_proportions_df["Proportion"].between(0, 100).all()


def test_plot_outlier_proportions_runs_and_shows(
    cytotable_NF1_contamination_data_df: pd.DataFrame,
):
    """
    Test the plot_outlier_proportions function returns a plot.
    """
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    with patch("matplotlib.pyplot.show") as mock_show:
        outlier_proportions_df = detector._get_outlier_proportion_per_well()
        detector._plot_outlier_proportions(df=outlier_proportions_df)
        mock_show.assert_called_once()


def test_check_partial_contamination(
    cytotable_NF1_contamination_data_df: pd.DataFrame,
):
    """
    Test the behavior of step 3 in the contamination detection process.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Execute step 1 as it is required for step 2
    detector.check_skew_and_variable()

    # Execute step 2 as it is required for step 3
    detector.check_feature_means()

    # Mock the plotting functions to avoid rendering the plot
    with patch("matplotlib.pyplot.show"):
        # Execute step 3
        detector.check_partial_contamination()

    # Check if the results are as expected
    assert detector.partial_contamination_texture_detected


def test_run(cytotable_NF1_contamination_data_df: pd.DataFrame):
    """
    Test the run method using real data to make sure the stepwise method
    runs correctly.
    """
    # Create the ContaminationDetector object
    detector = cd.ContaminationDetector(
        dataframe=cytotable_NF1_contamination_data_df, nucleus_channel_naming="DAPI"
    )

    # Mock the plotting functions to avoid rendering the plot
    with patch("matplotlib.pyplot.show"):
        # Execute the run method
        detector.run()

    # Assertions for step 1
    assert hasattr(detector, "is_skewed"), "Step 1 did not set 'is_skewed'."
    assert hasattr(detector, "is_variable"), "Step 1 did not set 'is_variable'."

    # If no skewness or variability, ensure it exits early
    if not detector.is_skewed and not detector.is_variable:
        assert not hasattr(detector, "whole_plate_contamination_texture"), (
            "Step 2 should not have been executed if no skewness or "
            "variability was detected."
        )
        assert not hasattr(detector, "partial_contamination_texture_detected"), (
            "Step 3 should not have been executed if no skewness or "
            "variability was detected."
        )
        return

    # Assertions for step 2
    assert hasattr(detector, "whole_plate_contamination_texture"), (
        "Step 2 did not set 'whole_plate_contamination_texture'."
    )
    assert hasattr(detector, "whole_plate_contamination_formfactor"), (
        "Step 2 did not set 'whole_plate_contamination_formfactor'."
    )

    # If partial contamination is detected, ensure step 3 is executed
    if detector.partial_contamination_texture_detected:
        assert hasattr(detector, "partial_contamination_texture_detected"), (
            "Step 3 should have been executed if partial contamination was detected."
        )
    else:
        # If no partial contamination, ensure step 3 is skipped
        assert not hasattr(detector, "partial_contamination_texture_detected"), (
            "Step 3 should not have been executed if no partial contamination "
            "was detected."
        )
