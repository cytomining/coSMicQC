"""
This function were derived from the `cellpainting_predicts_cardiac_fibrois`
GitHub repository.
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

# set numpy seed to make random operations (shuffling data) reproducible
np.random.seed(0)


def get_X_y_data(
    df: pd.DataFrame, label: str, shuffle: bool = False
) -> Tuple[np.array, np.array]:
    """Get X (feature space) and labels (predicting class) from pandas Data frame

    Args:
        df (pd.DataFrame): Data frame containing morphology.
        label (str): Name of the Metadata column being used as the predicting class
        shuffle (bool, optional): Shuffle the feature columns to get a shuffled dataset.
            Defaults to False.

    Returns:
        Tuple[np.array, np.array]: Returns np.arrays for the feature space (X)
            and the predicting class (y)
    """
    # Remove "Metadata" columns from df, leaving only the feature space
    X = df.loc[:, ~df.columns.str.contains("Metadata")].values

    # Extract class label
    y = df.loc[:, [label]].values
    # Make labels as array for use in machine learning
    y = np.ravel(y)

    # If shuffle is True, shuffled the columns independently for the feature space
    if shuffle:
        for column in X.T:
            np.random.shuffle(column)

    return X, y


def bootstrap_roc_auc(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    n_bootstraps: int = 1000,
) -> np.ndarray:
    """
    Perform bootstrapping to compute the distribution of ROC AUC scores.

    This function generates a bootstrapped distribution of ROC AUC scores by
    resampling the provided true labels and predicted probabilities with
    replacement.

    Parameters:
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1) for the dataset.

    y_pred : array-like of shape (n_samples,)
        Predicted probabilities or scores for the positive class.

    n_bootstraps : int, optional, default=1000
        Number of bootstrap iterations to perform.

    Returns:
    -------
    bootstrapped_scores : np.ndarray
        An array of bootstrapped ROC AUC scores. Each element represents the
        ROC AUC computed for a resampled dataset.
    """
    # list for the scores to be appended to
    bootstrapped_scores = []

    MIN_CLASSES_REQUIRED = 2
    # loop through and create the amount of bootstrap samples and calculate ROC scores
    for i in range(n_bootstraps):
        indices = resample(np.arange(len(y_true)), replace=True)
        # evaluate if the subsample has both classes
        if len(np.unique(y_true[indices])) < MIN_CLASSES_REQUIRED:
            # skip this subsample if it doesn't have both classes
            continue
        # if there are both classes, then calculate the score
        else:
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

    return np.array(bootstrapped_scores)
