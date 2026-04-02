"""
Tests cosmicqc cli module
"""

import pathlib

from pyarrow import parquet

from .utils import run_cli_command


def test_cli_util():
    """
    Test the run_cli_command for successful output
    """

    command = """echo 'hello world'"""
    _, _, returncode = run_cli_command(command)

    assert returncode == 0


def test_cli_identify_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `identify_outliers` function of the CLI.
    """

    _, _, returncode = run_cli_command(
        (
            f"""cosmicqc identify_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/identify_outliers_output.parquet"
        )
    )

    assert returncode == 0

    assert parquet.read_table(
        f"{tmp_path}/identify_outliers_output.parquet"
    ).to_pydict() == {
        "Metadata_cqc_custom_example_feature_zscore": [
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
    }


def test_cli_find_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `find_outliers` function of the CLI.
    """

    stdout, _, returncode = run_cli_command(
        (
            f"""cosmicqc find_outliers --df {basic_outlier_csv}"""
            """ --metadata_columns [] --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/find_outliers_output.parquet"
        )
    )

    assert returncode == 0
    assert (
        stdout.strip()
        == """Number of outliers: 2 (20.00%)
Outliers Range:
example_feature Min: 9
example_feature Max: 10""".strip()
    )

    assert parquet.read_table(
        f"{tmp_path}/find_outliers_output.parquet"
    ).to_pydict() == {"example_feature": [9, 10], "__index_level_0__": [8, 9]}


def test_cli_label_outliers(tmp_path: pathlib.Path, basic_outlier_csv: str):
    """
    Test the `label_outliers` function of the CLI.
    """

    _, _, returncode = run_cli_command(
        (
            f"""cosmicqc label_outliers --df {basic_outlier_csv}"""
            """ --feature_thresholds {"example_feature":1.0}"""
            f" --export_path {tmp_path}/label_outliers_output.parquet"
            " --export_as_annotations False"
        )
    )

    assert returncode == 0
    assert parquet.read_table(
        f"{tmp_path}/label_outliers_output.parquet"
    ).to_pydict() == {
        "example_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Metadata_cqc_custom_is_outlier": [
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
        ],
    }


def test_cli_label_outliers_multiple_conditions(
    tmp_path: pathlib.Path, basic_outlier_csv: str
):
    """
    Test the `label_outliers` CLI with a dict-of-dicts defining multiple
    conditions/rules to ensure multiple Metadata_cqc_<rule>_is_outlier
    columns are produced correctly.
    """

    command = (
        "cosmicqc label_outliers --df "
        + str(basic_outlier_csv)
        + " --feature_thresholds "
        + (
            '{"weird_cells":{"example_feature":1.0},'
            '"large_cells":{"example_feature":2.0}}'
        )
        + " --export_path "
        + str(tmp_path / "label_outliers_output.parquet")
        + " --export_as_annotations False"
    )

    _, _, returncode = run_cli_command(command)

    assert returncode == 0
    assert parquet.read_table(
        f"{tmp_path}/label_outliers_output.parquet"
    ).to_pydict() == {
        "example_feature": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Metadata_cqc_weird_cells_is_outlier": [
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
        ],
        "Metadata_cqc_large_cells_is_outlier": [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
    }
