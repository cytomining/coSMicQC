"""
Tests cosmicqc analyze module
"""

import pathlib

import numpy as np
import pandas as pd
import pytest
from cytodataframe import CytoDataFrame

from cosmicqc import analyze


def test_find_outliers_basic_dataframe(basic_outlier_dataframe: pd.DataFrame):
    """
    Testing find_outliers with basic/simulated data.
    """

    # add metadata to basic data
    metadata_column_name = "Image_Metadata_Plate"
    basic_outlier_dataframe[metadata_column_name] = "A"

    # assert that we have the output we expect
    assert analyze.find_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        metadata_columns=[metadata_column_name],
    ).to_dict(orient="dict") == {
        "example_feature": {8: 9, 9: 10},
        "Image_Metadata_Plate": {8: "A", 9: "A"},
    }


def test_find_outliers_nan_dataframe(basic_outlier_dataframe: pd.DataFrame):
    """
    Testing find_outliers with basic/simulated data.
    """

    # add metadata to basic data
    metadata_column_name = "Image_Metadata_Plate"
    basic_outlier_dataframe[metadata_column_name] = "A"

    # make the column include NaN's
    basic_outlier_dataframe.loc[
        basic_outlier_dataframe["example_feature"] > 6, "example_feature"
    ] = np.nan

    print(
        analyze.find_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            metadata_columns=[metadata_column_name],
        ).to_dict(orient="dict")
    )

    # assert that we have the output we expect
    assert analyze.find_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        metadata_columns=[metadata_column_name],
    ).to_dict(orient="dict") == {
        "example_feature": {5: 6.0},
        "Image_Metadata_Plate": {5: "A"},
    }


def test_find_outliers_basic_csv(basic_outlier_csv: str):
    """
    Testing find_outliers with csv data.
    """

    # assert that we have the output we expect
    assert analyze.find_outliers(
        df=basic_outlier_csv,
        feature_thresholds={"example_feature": 1},
        metadata_columns=[],
    ).to_dict(orient="dict") == {
        "example_feature": {8: 9, 9: 10},
    }


def test_find_outliers_cfret(cytotable_CFReT_data_df: pd.DataFrame):
    """
    Testing find_outliers with CytoTable CFReT data.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # Set a negative threshold to identify both outlier small nuclei
    # and low formfactor representing non-circular segmentations.
    feature_thresholds = {
        "Nuclei_AreaShape_Area": -1,
        "Nuclei_AreaShape_FormFactor": -1,
    }

    # run function to identify outliers given conditions
    small_area_formfactor_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we found the appropriate outliers
    assert small_area_formfactor_outliers_df.sort_values(
        list(feature_thresholds)
    ).to_dict(orient="dict") == {
        "Nuclei_AreaShape_Area": {
            7802: 752.0,
            11967: 788.0,
            5626: 796.0,
            9238: 798.0,
        },
        "Nuclei_AreaShape_FormFactor": {
            7802: 0.7981428715244236,
            11967: 0.5476435143650794,
            5626: 0.7348718757398186,
            9238: 0.8202563209583683,
        },
        "Image_Metadata_Plate": {
            7802: "localhost231120090001",
            11967: "localhost231120090001",
            5626: "localhost231120090001",
            9238: "localhost231120090001",
        },
        "Image_Metadata_Well": {
            7802: "D05",
            11967: "E04",
            5626: "C09",
            9238: "D07",
        },
        "Image_Metadata_Site": {
            7802: "f03",
            11967: "f10",
            5626: "f13",
            9238: "f04",
        },
    }

    # find very elongated nuclei segmentations (above mean)
    feature_thresholds = {
        "Nuclei_AreaShape_Eccentricity": 2,
    }

    # run function to identify outliers given conditions
    eccent_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    # test that we found the appropriate outliers
    assert eccent_outliers_df.sort_values(list(feature_thresholds)).to_dict(
        orient="dict"
    ) == {
        "Nuclei_AreaShape_Eccentricity": {
            7802: 0.8459531594205444,
            7884: 0.8528083158737935,
            5626: 0.8571246429020986,
            20609: 0.871038739898089,
            13920: 0.873104711473235,
            10061: 0.8755073203769763,
            10416: 0.876810524700015,
            4978: 0.8768628647595129,
            19420: 0.876866752777687,
            11967: 0.9109918316434343,
        },
        "Image_Metadata_Plate": {
            7802: "localhost231120090001",
            7884: "localhost231120090001",
            5626: "localhost231120090001",
            20609: "localhost231120090001",
            13920: "localhost231120090001",
            10061: "localhost231120090001",
            10416: "localhost231120090001",
            4978: "localhost231120090001",
            19420: "localhost231120090001",
            11967: "localhost231120090001",
        },
        "Image_Metadata_Well": {
            7802: "D05",
            7884: "D05",
            5626: "C09",
            20609: "G08",
            13920: "E08",
            10061: "D09",
            10416: "D11",
            4978: "C05",
            19420: "G07",
            11967: "E04",
        },
        "Image_Metadata_Site": {
            7802: "f03",
            7884: "f10",
            5626: "f13",
            20609: "f04",
            13920: "f07",
            10061: "f03",
            10416: "f11",
            4978: "f13",
            19420: "f00",
            11967: "f10",
        },
    }

    # find large nuclei segmentations (above mean) and low formfactor
    feature_thresholds = {
        "Nuclei_AreaShape_Area": 2,
        "Nuclei_AreaShape_FormFactor": -2,
    }

    # run function to identify outliers given conditions
    large_area_formfactor_outliers_df = analyze.find_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds=feature_thresholds,
        metadata_columns=metadata_columns,
    )

    assert large_area_formfactor_outliers_df.sort_values(
        list(feature_thresholds)
    ).to_dict(orient="dict") == {
        "Nuclei_AreaShape_Area": {
            20729: 1933.0,
            7796: 2098.0,
            14825: 2365.0,
            13920: 2664.0,
            9159: 2708.0,
            14178: 3498.0,
            10066: 3640.0,
            14811: 3751.0,
        },
        "Nuclei_AreaShape_FormFactor": {
            20729: 0.723128135340047,
            7796: 0.659333509302755,
            14825: 0.6794834892343651,
            13920: 0.5464078879768164,
            9159: 0.48231330364709524,
            14178: 0.41227682658167264,
            10066: 0.6545633283748163,
            14811: 0.6262476165070433,
        },
        "Image_Metadata_Plate": {
            20729: "localhost231120090001",
            7796: "localhost231120090001",
            14825: "localhost231120090001",
            13920: "localhost231120090001",
            9159: "localhost231120090001",
            14178: "localhost231120090001",
            10066: "localhost231120090001",
            14811: "localhost231120090001",
        },
        "Image_Metadata_Well": {
            20729: "G10",
            7796: "D05",
            14825: "F02",
            13920: "E08",
            9159: "D07",
            14178: "F02",
            10066: "D09",
            14811: "F02",
        },
        "Image_Metadata_Site": {
            20729: "f04",
            7796: "f03",
            14825: "f01",
            13920: "f07",
            9159: "f02",
            14178: "f03",
            10066: "f03",
            14811: "f01",
        },
    }


def test_read_thresholds_set_from_file():
    """
    Tests read_thresholds_set_from_file
    """

    # test that an exception is raised on receiving a bad
    # lookup value from the thresholds file.
    with pytest.raises(LookupError):
        analyze.read_thresholds_set_from_file(
            feature_thresholds="bad_lookup_value",
            feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
        )

    # test default threshold sets
    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="small_and_low_formfactor_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Area": -1, "Nuclei_AreaShape_FormFactor": -1}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="elongated_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Eccentricity": 2}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds="large_nuclei",
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {"Nuclei_AreaShape_Area": 2, "Nuclei_AreaShape_FormFactor": -2}

    assert analyze.read_thresholds_set_from_file(
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
    ) == {
        "small_and_low_formfactor_nuclei": {
            "Nuclei_AreaShape_Area": -1,
            "Nuclei_AreaShape_FormFactor": -1,
        },
        "elongated_nuclei": {"Nuclei_AreaShape_Eccentricity": 2},
        "large_nuclei": {"Nuclei_AreaShape_Area": 2, "Nuclei_AreaShape_FormFactor": -2},
    }


def test_convert_feature_threshold_input_to_named_threshold_dicts_none(
    tmp_path: pathlib.Path,
):
    """
    Test converting None into all named threshold sets from a YAML file.
    """

    thresholds_file = tmp_path / "thresholds.yml"
    thresholds_file.write_text(
        """
thresholds:
  low_feature:
    example_feature: -1
  high_feature:
    example_feature: 2
""".strip()
    )

    assert analyze._convert_feature_threshold_input_to_named_threshold_dicts(
        feature_thresholds_file=str(thresholds_file),
        feature_thresholds=None,
    ) == [
        ("low_feature", {"example_feature": -1}),
        ("high_feature", {"example_feature": 2}),
    ]


def test_convert_feature_threshold_input_to_named_threshold_dicts_string(
    tmp_path: pathlib.Path,
):
    """
    Test converting a named threshold key into one named threshold tuple.
    """

    thresholds_file = tmp_path / "thresholds.yml"
    thresholds_file.write_text(
        """
thresholds:
  oversegmented_cells:
    example_feature: 1
""".strip()
    )

    assert analyze._convert_feature_threshold_input_to_named_threshold_dicts(
        feature_thresholds_file=str(thresholds_file),
        feature_thresholds="oversegmented_cells",
    ) == [("oversegmented_cells", {"example_feature": 1})]


def test_convert_feature_threshold_input_to_named_threshold_dicts_inline_dict_warns():
    """
    Test inline thresholds become a custom named tuple and warn on file override.
    """

    with pytest.warns(UserWarning, match="feature_thresholds_file` will be ignored"):
        assert analyze._convert_feature_threshold_input_to_named_threshold_dicts(
            feature_thresholds_file="custom_thresholds.yml",
            feature_thresholds={"example_feature": 1},
        ) == [("custom", {"example_feature": 1})]


def test_convert_feature_threshold_input_to_named_threshold_dicts_multiple_dicts():
    """
    Test multiple inline named threshold dictionaries are returned unchanged.
    """

    feature_thresholds = {
        "oversegmented_cells": {"example_feature": 1},
        "missegmented_cells": {"example_feature": 2},
    }

    assert analyze._convert_feature_threshold_input_to_named_threshold_dicts(
        feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
        feature_thresholds=feature_thresholds,
    ) == list(feature_thresholds.items())


@pytest.mark.parametrize(
    "feature_thresholds",
    [
        {},
        {"example_feature": "bad"},
        {"oversegmented_cells": {"example_feature": 1}, "bad_feature": 2},
    ],
)
def test_convert_feature_threshold_input_to_named_threshold_dicts_invalid_input(
    feature_thresholds: object,
):
    """
    Test invalid threshold inputs raise a ValueError.
    """

    with pytest.raises(ValueError, match="feature_thresholds"):
        analyze._convert_feature_threshold_input_to_named_threshold_dicts(
            feature_thresholds_file=analyze.DEFAULT_QC_THRESHOLD_FILE,
            feature_thresholds=feature_thresholds,
        )


def test_create_condition_map_creates_conditions_and_reuses_zscores():
    """
    Test condition generation and z-score column reuse for one prefix.
    """

    df = CytoDataFrame(
        pd.DataFrame(
            {
                "high_feature": [1, 2, 3, 4, 5],
                "low_feature": [5, 4, 3, 2, 1],
            }
        )
    )
    outlier_df = df.copy()
    thresholds = {"high_feature": 1, "low_feature": -1}

    conditions, zscore_columns = analyze._create_condition_map(
        df=df,
        outlier_df=outlier_df,
        thresholds=thresholds,
        name_prefix="Metadata_cqc_custom",
    )

    assert zscore_columns == {
        "high_feature": "Metadata_cqc_custom_high_feature_zscore",
        "low_feature": "Metadata_cqc_custom_low_feature_zscore",
    }
    assert len(conditions) == 2
    assert conditions[0].tolist() == [False, False, False, False, True]
    assert conditions[1].tolist() == [False, False, False, False, True]

    original_high_zscore = outlier_df[zscore_columns["high_feature"]].copy()

    analyze._create_condition_map(
        df=df,
        outlier_df=outlier_df,
        thresholds={"high_feature": 1},
        name_prefix="Metadata_cqc_custom",
    )

    pd.testing.assert_series_equal(
        outlier_df[zscore_columns["high_feature"]],
        original_high_zscore,
        check_names=False,
    )


def test_create_condition_map_raises_for_missing_feature():
    """
    Test that missing features are rejected clearly.
    """

    df = CytoDataFrame(pd.DataFrame({"example_feature": [1, 2, 3]}))
    outlier_df = df.copy()

    with pytest.raises(ValueError, match="does not exist"):
        analyze._create_condition_map(
            df=df,
            outlier_df=outlier_df,
            thresholds={"missing_feature": 1},
            name_prefix="Metadata_cqc_custom",
        )


def test_find_outliers_dict_and_default_config_cfret(
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Testing find_outliers with dictionary vs yaml threshold sets
    using CytoTable CFReT data.
    """

    # metadata columns to include in output data frame
    metadata_columns = [
        "Image_Metadata_Plate",
        "Image_Metadata_Well",
        "Image_Metadata_Site",
    ]

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Area": -1,
                "Nuclei_AreaShape_FormFactor": -1,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="small_and_low_formfactor_nuclei",
            metadata_columns=metadata_columns,
        ),
    )

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Eccentricity": 2,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="elongated_nuclei",
            metadata_columns=metadata_columns,
        ),
    )

    # test that the output is the same from dict vs yaml
    pd.testing.assert_frame_equal(
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds={
                "Nuclei_AreaShape_Area": 2,
                "Nuclei_AreaShape_FormFactor": -2,
            },
            metadata_columns=metadata_columns,
        ),
        analyze.find_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="large_nuclei",
            metadata_columns=metadata_columns,
        ),
    )


def test_find_outliers_does_not_set_threshold_display_options(
    basic_outlier_dataframe: pd.DataFrame,
):
    """
    Ensure find_outliers does not add threshold-line display options.
    """

    cdf = CytoDataFrame(
        data=basic_outlier_dataframe.assign(Image_Metadata_Plate="A"),
    )
    cdf._custom_attrs["display_options"] = {"existing_setting": "keep"}
    result = analyze.find_outliers(
        df=cdf,
        feature_thresholds={"example_feature": 1},
        metadata_columns=["Image_Metadata_Plate"],
    )
    assert result._custom_attrs["display_options"]["existing_setting"] == "keep"
    assert "filter_columns" not in result._custom_attrs["display_options"]
    assert "filter_plot_thresholds" not in result._custom_attrs["display_options"]


def test_find_outliers_retains_custom_attrs_with_dropna_path(
    basic_outlier_dataframe: pd.DataFrame,
):
    """
    Ensure find_outliers preserves context attrs after projection and dropna.
    """

    data = basic_outlier_dataframe.assign(
        Image_Metadata_Plate="A",
        example_feature_two=basic_outlier_dataframe["example_feature"],
    )
    data.loc[data["example_feature"] > 6, "example_feature"] = np.nan
    cdf = CytoDataFrame(
        data=data,
        data_context_dir="example_context_dir",
        data_mask_context_dir="example_mask_dir",
    )

    result = analyze.find_outliers(
        df=cdf,
        feature_thresholds={"example_feature": 1},
        metadata_columns=["Image_Metadata_Plate"],
    )

    assert result._custom_attrs["data_context_dir"] == "example_context_dir"
    assert result._custom_attrs["data_mask_context_dir"] == "example_mask_dir"


def test_label_outliers(
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Tests label_outliers
    """

    # compare the dataframe vs csv output to make sure they are equivalent
    pd.testing.assert_frame_equal(
        analyze.label_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
        analyze.label_outliers(
            df=basic_outlier_csv,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
    )

    # test basic single-column result with zscores
    assert analyze.label_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=True,
    ).to_dict(orient="dict") == {
        "example_feature": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        },
        "Metadata_cqc_custom_example_feature_zscore": {
            0: -1.5666989036012806,
            1: -1.2185435916898848,
            2: -0.8703882797784892,
            3: -0.5222329678670935,
            4: -0.17407765595569785,
            5: 0.17407765595569785,
            6: 0.5222329678670935,
            7: 0.8703882797784892,
            8: 1.2185435916898848,
            9: 1.5666989036012806,
        },
        "Metadata_cqc_custom_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    # test for case when zscores are excluded
    assert analyze.label_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=False,
    ).to_dict(orient="dict") == {
        "example_feature": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        },
        "Metadata_cqc_custom_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    # test single-column result
    test_df = analyze.label_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds="large_nuclei",
        include_threshold_scores=True,
    )

    pd.testing.assert_frame_equal(
        test_df,
        pd.read_parquet(
            path="tests/data/coSMicQC/output_data/test_label_outliers_output.parquet",
        )[test_df.columns.tolist()],
    )

    # test full dataset
    pd.testing.assert_frame_equal(
        analyze.label_outliers(
            df=cytotable_CFReT_data_df, include_threshold_scores=True
        ),
        pd.read_parquet(
            path="tests/data/coSMicQC/output_data/test_label_outliers_output.parquet"
        ),
    )


def test_identify_outliers(
    basic_outlier_dataframe: pd.DataFrame,
    basic_outlier_csv: str,
    cytotable_CFReT_data_df: pd.DataFrame,
):
    """
    Tests identify_outliers helper function
    """

    # show that dataframe and csv output are the same
    pd.testing.assert_frame_equal(
        analyze.identify_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
        analyze.identify_outliers(
            df=basic_outlier_csv,
            feature_thresholds={"example_feature": 1},
            include_threshold_scores=True,
        ),
    )

    assert analyze.identify_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=True,
    ).to_dict(orient="dict") == {
        "Metadata_cqc_custom_example_feature_zscore": {
            0: -1.5666989036012806,
            1: -1.2185435916898848,
            2: -0.8703882797784892,
            3: -0.5222329678670935,
            4: -0.17407765595569785,
            5: 0.17407765595569785,
            6: 0.5222329678670935,
            7: 0.8703882797784892,
            8: 1.2185435916898848,
            9: 1.5666989036012806,
        },
        "Metadata_cqc_custom_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
    }

    pd.testing.assert_frame_equal(
        analyze.identify_outliers(
            df=cytotable_CFReT_data_df,
            feature_thresholds="large_nuclei",
            include_threshold_scores=True,
        ),
        pd.read_parquet(
            "tests/data/coSMicQC/output_data/test_identifier_outliers_output.parquet"
        ),
    )

    identified_df = analyze.identify_outliers(
        df=cytotable_CFReT_data_df,
        feature_thresholds="large_nuclei",
    )
    pd.testing.assert_series_equal(
        identified_df,
        pd.read_parquet(
            "tests/data/coSMicQC/output_data/test_identifier_outliers_output.parquet",
            columns=["Metadata_cqc_large_nuclei_is_outlier"],
        )["Metadata_cqc_large_nuclei_is_outlier"],
        check_names=False,
    )


def test_identify_outliers_multiple_conditions_returns_cytodataframe(
    basic_outlier_dataframe: pd.DataFrame,
):
    """
    Ensure multiple conditions return a combined CytoDataFrame.
    """

    result = analyze.identify_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={
            "oversegmented_cells": {"example_feature": 1},
            "missegmented_cells": {"example_feature": 2},
        },
        include_threshold_scores=False,
    )

    assert isinstance(result, CytoDataFrame)
    assert result.to_dict(orient="dict") == {
        "Metadata_cqc_oversegmented_cells_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
        "Metadata_cqc_missegmented_cells_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False,
        },
    }


def test_identify_outliers_multiple_conditions_with_scores_returns_cytodataframe(
    basic_outlier_dataframe: pd.DataFrame,
):
    """
    Ensure multiple conditions with scores return one combined CytoDataFrame.
    """

    result = analyze.identify_outliers(
        df=basic_outlier_dataframe,
        feature_thresholds={
            "oversegmented_cells": {"example_feature": 1},
            "missegmented_cells": {"example_feature": 2},
        },
        include_threshold_scores=True,
    )

    assert isinstance(result, CytoDataFrame)
    assert list(result.columns) == [
        "Metadata_cqc_oversegmented_cells_example_feature_zscore",
        "Metadata_cqc_oversegmented_cells_is_outlier",
        "Metadata_cqc_missegmented_cells_example_feature_zscore",
        "Metadata_cqc_missegmented_cells_is_outlier",
    ]


def test_label_outliers_retains_custom_attrs(basic_outlier_dataframe: pd.DataFrame):
    """
    Tests that label_outliers retains custom attributes
    """

    # create a CytoDataFrame with custom attributes
    cdf = CytoDataFrame(
        data=basic_outlier_dataframe,
        data_context_dir="example_context_dir",
        data_mask_context_dir="example_mask_dir",
        data_outline_context_dir="example_context_dir",
        segmentation_file_regex={"example": "example"},
    )

    # run the data through label_outliers
    df = analyze.label_outliers(
        df=cdf,
        feature_thresholds={"example_feature": 1},
    )

    assert isinstance(df, CytoDataFrame)


def test_label_outliers_sets_filter_display_options_multiple_conditions(
    basic_outlier_dataframe: pd.DataFrame,
):
    """
    Ensure label_outliers adds filter display options from threshold sets.
    """

    cdf = CytoDataFrame(
        data=basic_outlier_dataframe.assign(
            example_feature_two=basic_outlier_dataframe["example_feature"]
        ),
    )
    cdf._custom_attrs["display_options"] = {"existing_setting": "keep"}
    result = analyze.label_outliers(
        df=cdf,
        feature_thresholds={
            "oversegmented_cells": {"example_feature": 1},
            "small_cells": {"example_feature_two": -1},
        },
    )

    assert result._custom_attrs["display_options"]["existing_setting"] == "keep"
    assert result._custom_attrs["display_options"]["filter_columns"] == [
        "example_feature",
        "example_feature_two",
    ]
    assert result._custom_attrs["display_options"]["filter_plot_thresholds"][
        "example_feature"
    ] == pytest.approx(
        basic_outlier_dataframe["example_feature"].mean()
        + basic_outlier_dataframe["example_feature"].std(ddof=0)
    )
    assert result._custom_attrs["display_options"]["filter_plot_thresholds"][
        "example_feature_two"
    ] == pytest.approx(
        basic_outlier_dataframe["example_feature"].mean()
        - basic_outlier_dataframe["example_feature"].std(ddof=0)
    )


def test_label_outliers_multiple_conditions(basic_outlier_dataframe: pd.DataFrame):
    """
    Test `label_outliers` with a dict-of-dicts defining multiple
    conditions/rules to ensure multiple Metadata_cqc_<rule>_is_outlier
    columns are produced correctly when called directly.
    """

    feature_thresholds = {
        "oversegmented_cells": {"example_feature": 1},
        "missegmented_cells": {"example_feature": 2},
    }
    # run label_outliers with multiple named conditions
    result = analyze.label_outliers(
        df=basic_outlier_dataframe, feature_thresholds=feature_thresholds
    )

    # expected: last two values (9,10) are outliers for threshold 1,
    # and none exceed threshold 2 so missegmented_cells flags remain False
    assert result.to_dict(orient="dict") == {
        "example_feature": {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
        },
        "Metadata_cqc_oversegmented_cells_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: True,
            9: True,
        },
        "Metadata_cqc_missegmented_cells_is_outlier": {
            0: False,
            1: False,
            2: False,
            3: False,
            4: False,
            5: False,
            6: False,
            7: False,
            8: False,
            9: False,
        },
    }


def test_label_outliers_annotation_export(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Ensure `label_outliers` exports user-selected metadata + QC columns when
    `export_as_annotations=True` is used.
    """

    # prepare data with metadata columns used for downstream annotation
    df = basic_outlier_dataframe.copy()
    df["Image_Metadata_Plate"] = "plate1"
    df["Image_Metadata_Well"] = "A01"
    df["Image_Metadata_Site"] = "site1"

    export_path = tmp_path / "annotation_output.parquet"

    # run label_outliers and export in annotation mode
    analyze.label_outliers(
        df=df,
        feature_thresholds={"example_feature": 1},
        include_threshold_scores=False,
        export_path=str(export_path),
        export_as_annotations=True,
        annotation_metadata_columns=[
            "Image_Metadata_Plate",
            "Image_Metadata_Site",
        ],
    )

    # read exported parquet and assert columns are the requested metadata then CQC
    exported = pd.read_parquet(export_path)

    assert list(exported.columns) == [
        "Image_Metadata_Plate",
        "Image_Metadata_Site",
        "Metadata_cqc_custom_is_outlier",
    ]

    # check that the outlier flags match expectations (only last two are outliers)
    assert exported["Metadata_cqc_custom_is_outlier"].tolist() == [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_label_outliers_annotation_export_requires_metadata_columns(
    tmp_path: pathlib.Path, basic_outlier_dataframe: pd.DataFrame
):
    """
    Ensure annotation export requires explicit metadata columns.
    """

    export_path = tmp_path / "annotation_output_missing_metadata.parquet"

    with pytest.raises(ValueError, match="annotation_metadata_columns"):
        analyze.label_outliers(
            df=basic_outlier_dataframe,
            feature_thresholds={"example_feature": 1},
            export_path=str(export_path),
            export_as_annotations=True,
        )
