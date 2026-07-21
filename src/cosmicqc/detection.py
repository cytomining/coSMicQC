"""
Module for detecting abnormal perinuclear signal (e.g., mycoplasma, channel
bleedthrough) from the morphology profiles.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

from cosmicqc import find_outliers


def bool_to_emoji(val: bool) -> str:
    """Convert boolean flag to emoji for printing.

    Args:
        val (bool): True indicates an issue/failure was detected, False indicates none.

    Returns:
        str: emoji representing the flag status
    """
    return "🔴" if val else "🟢"


class PerinuclearSignalDetector:
    """
    This class implements a abnormal perinuclear signal detection
    process for high-content morphology data.
    Abnormal pernuclear can be classified as many things, including
    mycoplasma and channel bleedthough, which can impact the morphology
    of cells. For example, there are methods in the wet lab to detect
    mycoplasma, but it is not always full-proof.
    This class includes methods to detect abnormal perinuclear signal of
    any form based on nucleus morphology features.
    There are three steps to the process:

    1. Check for skewness in the cytoplasm texture and variability
    in the FormFactor features. If either is detected, the process
    moves on to the next step.
    2. Determine if the whole plate or partial plate is impacted by
    checking the mean values of the features.
    The process will move on to the last step only if partial
    abnormal signal is detected in the texture feature.
    3. If partial abnormal signal is detected for texture, find outliers
    and plot the proportion of outliers per well.

    Attributes:
        dataframe : pd.DataFrame
            The input DataFrame containing the features to be analyzed.
            Should be the output from the CytoTable convert function.
        nucleus_channel_naming : str
            The naming convention for the nucleus channel in the DataFrame.
        cyto_feature : str
            The feature name for cytoplasm texture.
        formfactor_feature : str
            The feature name for FormFactor.
        lower_skew_threshold : float
            Lower threshold for Bowley's skewness.
        upper_skew_threshold : float
            Upper threshold for Bowley's skewness.
        variability_threshold : float
            Threshold for variability in the FormFactor feature.
        texture_mean_threshold : float
            Threshold for mean value of cytoplasm texture.
        formfactor_mean_threshold : float
            Threshold for mean value of FormFactor.
        outlier_std_threshold : float
            Standard deviation threshold for outlier detection.
        is_skewed : bool
            Indicates if the distribution is skewed.
        is_variable : bool
            Indicates if the distribution is variable.
        whole_plate_abnormal_texture : bool
            Indicates if whole plate abnormal signal is detected in texture feature.
        whole_plate_abnormal_formfactor : bool
            Indicates if whole plate abnormal signal is detected in FormFactor feature.
        partial_abnormal_texture_detected : bool
            Indicates if partial abnormal signal is detected in the texture feature.
    """

    def __init__(  # noqa: PLR0913
        self,
        dataframe: pd.DataFrame,
        nucleus_channel_naming: str = "DNA",
        lower_skew_threshold: float = -0.15,
        upper_skew_threshold: float = 0.09,
        variability_threshold: float = 0.15,
        texture_mean_threshold: float = -0.25,
        formfactor_mean_threshold: float = 0.78,
        outlier_std_threshold: float = 1.0,
    ) -> None:
        """
        Initializer for the PerinuclearSignalDetector class.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing the features
                to be analyzed. Should be the output from CytoTable convert.
            nucleus_channel_naming (str): Naming convention for the nucleus
                channel in the DataFrame. Defaults to "DNA".
            lower_skew_threshold (float): Lower threshold for Bowley's
                skewness to flag whole-plate abnormal signal. Defaults to -0.15.
            upper_skew_threshold (float): Upper threshold for Bowley's
                skewness to flag partial-plate abnormal signal. Defaults to 0.09.
            variability_threshold (float): Threshold for acceptable
                coefficient of variation in selected features. Defaults to 0.15.
            texture_mean_threshold (float): Mean texture value threshold
                for detecting abnormalities. Defaults to -0.25.
            formfactor_mean_threshold (float): Mean FormFactor threshold
                for detecting nuclear shape issues. Defaults to 0.78.
            outlier_std_threshold (float): Number of standard deviations from
                the mean used to flag outliers. Defaults to 1.0.
        """
        self.dataframe = dataframe
        self.nucleus_channel_naming = nucleus_channel_naming
        self.lower_skew_threshold = lower_skew_threshold
        self.upper_skew_threshold = upper_skew_threshold
        self.variability_threshold = variability_threshold
        self.texture_mean_threshold = texture_mean_threshold
        self.formfactor_mean_threshold = formfactor_mean_threshold
        self.outlier_std_threshold = outlier_std_threshold

        # set the features to be used for abnormal perinuclear signal detection
        self.cyto_feature = (
            f"Cytoplasm_Texture_InfoMeas1_{self.nucleus_channel_naming}_3_02_256"
        )
        self.formfactor_feature = "Nuclei_AreaShape_FormFactor"

    def _skewness_test_cytoplasm_texture(self) -> bool:
        """
        Bowley's skewness score is calculated for the cytoplasm texture
        around the nucleus feature.
        Thresholds are assigned to determine if there is abnormal texture around
        the nucleus, either based on a serious negative skew
        (indicating a whole plate issue) or a positive skew
        (indicating a partial plate issue).
        Thresholds are set based on findings from multiple experiments
        (NF1 & CFReT, see citation file).

        Reference: Bowley, A. L. (1901): Elements of Statistics, P.S. King and Son, Laondon.

        Returns:
            boolean: True if the distribution is skewed, False otherwise.
        """  # noqa: E501
        # calculate the quartiles
        q1 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 25)
        q2 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 50)
        q3 = np.percentile(self.dataframe[self.cyto_feature].dropna(), 75)

        # calculate Bowley's skewness (to detect partial abnormal texture)
        bowley_skewness = (q3 + q1 - 2 * q2) / (q3 - q1)

        # check if the distribution is skewed
        # Bowley's skewness is negative for a whole plate issue
        # and positive for a partial plate issue
        return (
            bowley_skewness < self.lower_skew_threshold
            or bowley_skewness > self.upper_skew_threshold
        )

    def _variability_test_formfactor(self) -> bool:
        """
        Calculate Interquartile Range (IQR) score for the FormFactor feature
        to determine if there is abnormal variability in the shape of the nuclei.
        More variability in this feature indicates a higher proportion
        of non-circular nuclei.
        This can be an indicator of poor segmentations due to non-optimal
        segmentation parameters or abnormal perinuclear signal.
        Threshold is set based on findings from multiple experiments
        (NF1 & CFReT, see citation file).

        Returns:
            boolean: True if the distribution is variable, False otherwise.
        """
        # use IQR to measure variability of the feature
        vals = self.dataframe[self.formfactor_feature].dropna()
        iqr = np.percentile(vals, 75) - np.percentile(vals, 25)

        # determine if distribution is highly variable
        return iqr > self.variability_threshold

    def check_skew_and_variable(self) -> None:
        """
        Step 1: Check for skewness in the cytoplasm texture around the nucleus and
        variability in the FormFactor of the nucleus.
        """
        print("Running step 1...")

        # perform skewness and variability tests
        self.is_skewed = self._skewness_test_cytoplasm_texture()
        self.is_variable = self._variability_test_formfactor()

        # format results in a pretty table
        print("Summary:\n")
        print(
            tabulate(
                [
                    ["Texture skewed?", bool_to_emoji(self.is_skewed)],
                    ["Nucleus shape variable?", bool_to_emoji(self.is_variable)],
                ],
                headers=["Check", "Result"],
                tablefmt="simple",
            )
        )

        # set interpretation based on results
        if self.is_skewed and self.is_variable:
            interpretation = (
                "Abnormal perinuclear signal detected! 🚨\n"
                "Anomalous texture around nuclei detected and "
                "nuclei segmentation is impacted.\n"
                "Proceeding to step 2..."
            )
        elif self.is_skewed:
            interpretation = (
                "Abnormal perinuclear signal detected! 🚨\n"
                "Anomalous texture around nuclei detected but "
                "nuclei segmentation not clearly impacted.\n"
                "Proceeding to step 2..."
            )
        elif self.is_variable:
            interpretation = (
                "Potential abnormal perinuclear signal detected! 🛑\n"
                "This could indicate abnormal signal, segmentation issues, "
                "or an interesting phenotype.\n"
                "Proceeding to step 2..."
            )
        else:
            interpretation = (
                "No indication of abnormal perinuclear signal. Plate appears clean 🫧!"
            )

        print(f"Interpretation:\n{interpretation}")

    def _calculate_texture_mean(self) -> bool:
        """
        Check the mean value of cytoplasm texture around a nucleus feature to
        determine if whole plate or partial plate abnormal texture is present.
        Threshold is set based on findings from multiple experiments
        (NF1 & CFReT, see citation file).

        Normal raw values of this feature range below 0. In a clean plate,
        it is noted that the normal values range from -0.5 to -0.3.
        We set the threshold to -0.25 to detect whole plate abnormal texture.

        Returns:
            boolean: True if whole plate abnormal texture is detected, False otherwise
            indicates partial abnormal texture.
        """
        # check if the mean value is above the threshold
        # True = whole plate, False = partial plate
        return self.dataframe[self.cyto_feature].mean() >= self.texture_mean_threshold

    def _calculate_formfactor_mean(self) -> bool:
        """
        Check the mean value of the FormFactor of the nucleus to determine
        if whole plate or partial plate abnormal signal is present.
        Threshold is set based on findings from multiple experiments
        (NF1 & CFReT, see citation file).

        Normal raw values of this feature range from 0 to 1. In a clean plate,
        it is noted that the normal values range from 0.8 to 1 (heavily skewed).
        We set the threshold to 0.78 to detect whole plate abnormal signal.

        Returns:
            boolean: True if whole plate abnormal signal is detected, False otherwise
            indicates partial abnormal signal.
        """
        # check if the mean value is below the threshold
        # True = whole plate, False = partial plate
        return (
            self.dataframe[self.formfactor_feature].mean()
            <= self.formfactor_mean_threshold
        )

    def check_feature_means(self) -> None:
        """
        Step 2: If skewness and/or variability detected, determine if the whole plate or
        partial plate is impacted using the mean values of the features.

        Main interpretations:
        - If whole plate abnormal signal detected in both texture and formfactor
            → major warning.
        - If whole plate abnormal signal detected in texture only
            → check nucleus channel images for abnormal signal.
        - If whole plate abnormal signal detected in formfactor only
            → check segmentation parameters and/or images for abnormal signal.
        - If partial plate abnormal signal detected in texture only
            → proceed to step 3.
        - If partial plate abnormal signal detected in formfactor only
            → recommend performing 'find_outliers' with FormFactor.
        """
        # run sanity check to ensure step one was ran prior to step two
        if not hasattr(self, "is_skewed") or not hasattr(self, "is_variable"):
            raise RuntimeError(
                "You must run the skew and variable test before performing this test."
            )

        # catch case where steps are used individually and no abnormal signal is present
        if not self.is_skewed and not self.is_variable:
            print("Data is neither skewed nor variable based on step one.")
            print("No indication of abnormal signal. Plate appears clean 🫧.")
            self.partial_abnormal_texture_detected = False
            return

        print("Running step 2...")

        # Instantiate variables to check for whole plate abnormal signal
        self.whole_plate_abnormal_texture = (
            self._calculate_texture_mean() if self.is_skewed else False
        )
        self.whole_plate_abnormal_formfactor = (
            self._calculate_formfactor_mean() if self.is_variable else False
        )

        # Set lookup dictionary for all interpretations based on the boolean values
        interpretation_lookup = {
            (
                True,
                True,
                True,
                True,
            ): (
                "MAJOR WARNING! 💥\n"
                "Detected abnormal perinuclear signal across the entire plate. "
                "We see this flagged by both texture and nuclei shape features. "
                "We strongly suggest inspecting nucleus channel images for "
                "abnormal perinuclear signal."
            ),
            (
                True,
                False,
                True,
                False,
            ): (
                "Whole plate abnormal perinuclear texture detected. "
                "Please check your nucleus channel images."
            ),
            (
                False,
                True,
                False,
                True,
            ): (
                "Whole plate abnormal nuclear shape detected. "
                "Check segmentation parameters and/or images."
            ),
            (
                True,
                True,
                True,
                False,
            ): (
                "Whole plate abnormal texture + partial plate abnormal nuclear shape. "
                "Inspect nucleus channel and segmentation in a subset of FOVs."
            ),
            (
                True,
                True,
                False,
                False,
            ): (
                "Partial abnormal perinuclear signal detected in both texture and shape. "  # noqa: E501
                "Proceed to step 3 and inspect FormFactor outliers separately."
            ),
            (
                True,
                False,
                False,
                False,
            ): "Partial plate abnormal perinuclear signal in texture only. Proceed to step 3.",  # noqa: E501
            (
                False,
                True,
                False,
                False,
            ): (
                "Partial shape deviation detected. "
                "Recommend running 'find_outliers' with FormFactor."
            ),
        }

        # Set up key for lookup mapping to be applied
        key = (
            self.is_skewed,  # checks texture abnormal signal
            self.is_variable,  # checks shape abnormal signal
            self.whole_plate_abnormal_texture,
            self.whole_plate_abnormal_formfactor,
        )

        # Get the interpretation from the lookup dictionary
        interpretation = interpretation_lookup.get(
            key,
            (
                "Combination not explicitly handled. "
                "Consider inspecting both channels manually."
            ),
        )

        # Format interpretation as table
        print("Summary:\n")
        print(
            tabulate(
                [
                    ["Texture skewed?", bool_to_emoji(self.is_skewed)],
                    ["Nucleus shape variable?", bool_to_emoji(self.is_variable)],
                    [
                        "Whole plate has perinuclear signal due to abnormal texture?",
                        bool_to_emoji(self.whole_plate_abnormal_texture),
                    ],
                    [
                        "Whole plate has perinuclear signal due to abnormal nucleus shape?",  # noqa: E501
                        bool_to_emoji(self.whole_plate_abnormal_formfactor),
                    ],
                ],
                headers=["Check", "Result"],
                tablefmt="simple",
            )
        )
        print("Interpretation:\n" + interpretation)

        # Set flag used in step 3
        self.partial_abnormal_texture_detected = (
            self.is_skewed and not self.whole_plate_abnormal_texture
        )

    def _find_texture_outliers(self) -> pd.DataFrame:
        """
        Use coSMicQC find_outliers function to identify single-cells with abnormal
        perinuclear signal based on texture in the cytoplasm around the nucleus in
        partially abnormal plates (some abnormal wells).

        Returns:
            pd.DataFrame: DataFrame containing the outliers detected to use to plotting
            and calculating proportions.
        """
        # necessary metadata columns for processing
        metadata_columns = [
            "Image_Metadata_Plate",
            "Image_Metadata_Well",
            "Image_Metadata_Site",
            "Metadata_Nuclei_Location_Center_X",
            "Metadata_Nuclei_Location_Center_Y",
        ]

        # detect outliers based on original texture feature
        texture_feature_thresholds = {
            # outlier threshold for only cytoplasm texture in nuclei channel
            self.cyto_feature: self.outlier_std_threshold,
        }

        texture_nuclei_outliers = find_outliers(
            df=self.dataframe,
            metadata_columns=metadata_columns,
            feature_thresholds=texture_feature_thresholds,
        )

        # for added support, also check for outliers using granularity of the cytoplasm
        # in the nuclei channel
        granularity_feature_thresholds = {
            # outlier threshold for only cytoplasm granularity in nuclei channel
            f"Cytoplasm_Granularity_2_{self.nucleus_channel_naming}": self.outlier_std_threshold,  # noqa: E501
        }

        granularity_nuclei_outliers = find_outliers(
            df=self.dataframe,
            metadata_columns=metadata_columns,
            feature_thresholds=granularity_feature_thresholds,
        )

        # combine the two outlier dataframes
        combined_outliers = pd.concat(
            [texture_nuclei_outliers, granularity_nuclei_outliers]
        ).drop_duplicates(
            subset=[
                "Image_Metadata_Well",
                "Image_Metadata_Site",
                "Metadata_Nuclei_Location_Center_X",
                "Metadata_Nuclei_Location_Center_Y",
            ],
            keep="first",
        )

        # print total number of outliers detected
        print(f"Total number of outliers detected: {len(combined_outliers)}")

        # return the outliers dataframe
        return combined_outliers

    def _get_outlier_proportion_per_well(self) -> pd.DataFrame:
        """
        Calculate the proportion of outliers per well to be used for plotting.

        Returns:
            pd.DataFrame: Proportion per well dataframe
        """
        # Get the outliers detected
        combined_outliers = self._find_texture_outliers()

        # Calculate the proportion of outliers per well
        outlier_proportion_per_well = (
            combined_outliers.groupby("Image_Metadata_Well")
            .size()
            .div(self.dataframe.groupby("Image_Metadata_Well").size(), fill_value=0)
            .fillna(0)
            .reset_index()
        )
        outlier_proportion_per_well.columns = ["Well", "Proportion"]
        outlier_proportion_per_well["Proportion"] *= 100

        # Sort the wells in order of row and column (e.g., A01, A02, B01, B02, ...)
        outlier_proportion_per_well["Well"] = pd.Categorical(
            outlier_proportion_per_well["Well"],
            sorted(
                outlier_proportion_per_well["Well"], key=lambda x: (x[0], int(x[1:]))
            ),
        )
        outlier_proportion_per_well = outlier_proportion_per_well.sort_values("Well")

        # Calculate the cell counts per well
        cell_count_per_well = (
            self.dataframe.groupby("Image_Metadata_Well")
            .size()
            .reset_index(name="CellCount")
        )

        # Merge cell count with proportions
        outlier_proportion_per_well = outlier_proportion_per_well.merge(
            cell_count_per_well, left_on="Well", right_on="Image_Metadata_Well"
        )

        return outlier_proportion_per_well

    def _plot_outlier_proportions(self, df: pd.DataFrame) -> None:
        """
        Plot the proportion of outliers per well with a color gradient
        based on the total cell count per well.

        Args:
            df (pd.DataFrame): Proportion per well DataFrame
        """
        # Normalize the cell count for color mapping
        norm = plt.Normalize(df["CellCount"].min(), df["CellCount"].max())
        colors = plt.cm.viridis(norm(df["CellCount"]))

        # Create bar plot
        plt.figure(figsize=(20, 6))
        ax = sns.barplot(
            data=df,
            x="Well",
            y="Proportion",
            hue="Well",
            palette=list(colors),
            legend=False,
        )
        plt.xlabel("Well")
        plt.ylabel("Proportion")
        plt.xticks(rotation=45)
        plt.ylim(0, 100)

        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label="Total Cell Count")
        plt.show()

    def check_partial_abnormal_signal(self) -> None:
        """
        Step 3: If partial abnormal signal detected, find outliers and plot the
        proportion of outliers per well.
        """
        # run sanity check to ensure step two was ran prior to step three
        if not hasattr(self, "partial_abnormal_texture_detected"):
            raise RuntimeError(
                "You must run the feature mean test before performing this test."
            )

        print("Running step 3...")

        # check if partial abnormal signal was detected
        if self.partial_abnormal_texture_detected:
            print("Finding outlier cells with anomalous texture around the nucleus...")
            outliers = self._get_outlier_proportion_per_well()
            self._plot_outlier_proportions(outliers)

            # Calculate the 75th percentile (top 25%) of the outlier proportions
            top_25_percent_threshold = outliers["Proportion"].quantile(0.75)

            # Filter wells with proportions in the top 25%
            top_outlier_wells = outliers[
                outliers["Proportion"] >= top_25_percent_threshold
            ]
            # Print the wells in the top 25% horizontally
            print(f"\nNumber of wells in the top 25%: {len(top_outlier_wells)}")
            print("Wells in the top 25% of highest outlier proportions:")
            print(", ".join(top_outlier_wells["Well"]))
        else:
            print("No partial abnormal signal detected; no outliers to find.")

    def run(self) -> None:
        """
        Run all steps of the perinuclear signal detection process with conditional logic.
        """  # noqa: E501
        # Run the first step in the process to check for skewness and variability
        self.check_skew_and_variable()

        # Exit early if no skewness or variability detected
        if not self.is_skewed and not self.is_variable:
            print(
                "No skewness or variability detected. Exiting perinuclear signal detector."  # noqa: E501
            )
            return

        # Run the second step if skewness or variability was detected
        # Determines if the whole plate or partial plate is impacted
        self.check_feature_means()

        # Only continue to step 3 if partial abnormal texture was detected
        if (
            hasattr(self, "partial_abnormal_texture_detected")
            and self.partial_abnormal_texture_detected
        ):
            self.check_partial_abnormal_signal()
