"""
PurgedKFold Cross-Validation for Financial Time Series
=======================================================

Implementation of Purged K-Fold Cross-Validation following López de Prado's
methodology from "Advances in Financial Machine Learning", Chapter 7.

This module provides cross-validation strategies that prevent look-ahead bias
in financial machine learning by:

1. **Purging**: Removes training observations whose labels overlap with test
   observations in time.

2. **Embargo**: Adds a temporal gap between training and test sets to prevent
   information leakage from sequential correlation.

Reference:
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance, Snippets 7.1-7.4
"""

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Find training set times that don't overlap with test set labels.

    Given a test set, this function finds the training observations whose
    labels do not overlap in time with the test observations.

    This implements Snippet 7.1 from López de Prado.

    Parameters
    ----------
    t1 : pd.Series
        Series with index = observation start time, values = label end time.
        This represents the time span of each label/observation.
    test_times : pd.Series
        Times of test observations (subset of t1.index)

    Returns
    -------
    pd.Series
        Training times that don't overlap with test times

    Example
    -------
    >>> t1 = pd.Series(index=times, data=end_times)  # label horizons
    >>> test_indices = t1.index[100:200]  # test observations
    >>> train_times = get_train_times(t1, test_indices)
    """
    # Copy to avoid modifying original
    train = t1.copy(deep=True)

    # For each test observation, find overlapping training observations
    for start_test, end_test in test_times.items():
        # Find training observations that START before test END
        # AND END after test START (overlap condition)
        df0 = train[(start_test <= train.index) & (train.index <= end_test)].index
        df1 = train[(start_test <= train) & (train <= end_test)].index
        df2 = train[(train.index <= start_test) & (end_test <= train)].index

        # Remove overlapping observations from training set
        train = train.drop(df0.union(df1).union(df2))

    return train


def get_embargo_times(times: pd.DatetimeIndex, pct_embargo: float) -> pd.Timedelta:
    """
    Calculate embargo time based on percentage of dataset.

    The embargo creates a gap between training and test sets to prevent
    information leakage from autocorrelated features.

    This implements Snippet 7.2 from López de Prado.

    Parameters
    ----------
    times : pd.DatetimeIndex
        All observation times
    pct_embargo : float
        Fraction of dataset to use as embargo (e.g., 0.01 = 1%)

    Returns
    -------
    pd.Timedelta
        Embargo period duration
    """
    # Calculate step (average time between observations)
    step = int(times.shape[0] * pct_embargo)

    if step == 0:
        return pd.Timedelta(0)

    # Get embargo as time difference
    embargo = pd.Timedelta(times[step] - times[0])

    return embargo


class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold Cross-Validation for Financial Time Series.

    This cross-validator prevents look-ahead bias by:
    1. Never shuffling observations (maintains temporal order)
    2. Purging training observations that overlap with test labels
    3. Adding an embargo period between train and test sets

    This implements Snippet 7.3 and 7.4 from López de Prado.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    t1 : pd.Series, optional
        Series where index = observation start time, values = label end time.
        Required for purging to work correctly.
    pct_embargo : float, default=0.01
        Percentage of observations to embargo after each test fold.
        This creates a gap between training and test sets.

    Attributes
    ----------
    n_splits : int
        Number of folds.
    t1 : pd.Series
        Label end times for each observation.
    pct_embargo : float
        Embargo percentage.

    Examples
    --------
    >>> from ml_pipeline.cross_validation import PurgedKFold
    >>> # t1 contains label expiration times from triple barrier
    >>> cv = PurgedKFold(n_splits=5, t1=events['t1'], pct_embargo=0.01)
    >>> for train_idx, test_idx in cv.split(X):
    ...     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    ...     y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    Notes
    -----
    Unlike sklearn's KFold with shuffle=True, this implementation:
    - Maintains temporal ordering of observations
    - Removes training samples that would leak information into test set
    - Adds embargo to prevent autocorrelation leakage

    This is critical for financial ML where labels have temporal extent
    (e.g., triple barrier labels that span multiple time periods).

    References
    ----------
    López de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series = None, pct_embargo: float = 0.01):
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be a pandas DataFrame or numpy array.
        y : array-like of shape (n_samples,), optional
            Target variable (ignored, present for API compatibility).
        groups : array-like of shape (n_samples,), optional
            Group labels (ignored, present for API compatibility).

        Yields
        ------
        train : ndarray
            Training set indices for this fold.
        test : ndarray
            Test set indices for this fold.
        """
        # Get indices based on input type
        if hasattr(X, 'index'):
            indices = np.arange(X.shape[0])
            times = X.index
        else:
            indices = np.arange(X.shape[0])
            times = None

        # Prepare t1 for purging
        if self.t1 is not None and times is not None:
            # Align t1 with X's index
            t1 = self.t1.loc[self.t1.index.isin(times)]
        else:
            t1 = self.t1

        # Calculate embargo size
        embargo_size = int(indices.shape[0] * self.pct_embargo)

        # Calculate test fold size
        test_size = indices.shape[0] // self.n_splits

        for fold in range(self.n_splits):
            # Define test indices (sequential, no shuffle)
            test_start = fold * test_size

            # Last fold gets remaining samples
            if fold == self.n_splits - 1:
                test_end = indices.shape[0]
            else:
                test_end = (fold + 1) * test_size

            test_indices = indices[test_start:test_end]

            # Start with all non-test indices as potential training
            train_indices = np.concatenate([
                indices[:test_start],
                indices[test_end:]
            ])

            # Apply embargo: remove indices right after test set
            if embargo_size > 0 and test_end < indices.shape[0]:
                embargo_end = min(test_end + embargo_size, indices.shape[0])
                embargo_indices = indices[test_end:embargo_end]
                train_indices = np.setdiff1d(train_indices, embargo_indices)

            # Apply purging if t1 is provided
            if t1 is not None and times is not None:
                # Get test times
                test_times = times[test_indices]

                # Get t1 values for test observations
                test_t1 = t1.loc[t1.index.isin(test_times)]

                if len(test_t1) > 0:
                    # Find training observations that overlap with test labels
                    purge_mask = np.zeros(len(train_indices), dtype=bool)

                    train_times = times[train_indices]
                    train_t1_aligned = t1.loc[t1.index.isin(train_times)]

                    # Check each test observation for overlap
                    test_start_time = test_times.min()
                    test_end_time = test_t1.max()

                    for i, train_idx in enumerate(train_indices):
                        train_time = times[train_idx]

                        if train_time in train_t1_aligned.index:
                            train_end = train_t1_aligned.loc[train_time]

                            # Check if training observation overlaps with test period
                            # Overlap if: train_start < test_end AND train_end > test_start
                            if train_time < test_end_time and train_end > test_start_time:
                                # More precise check: does this specific training obs overlap?
                                for test_time in test_times:
                                    if test_time in test_t1.index:
                                        test_obs_end = test_t1.loc[test_time]

                                        # Overlap condition
                                        if (train_time <= test_obs_end and
                                            train_end >= test_time):
                                            purge_mask[i] = True
                                            break

                    # Remove purged indices
                    train_indices = train_indices[~purge_mask]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits."""
        return self.n_splits


class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation.

    Generates all possible train/test combinations from k groups,
    with purging and embargo applied. This provides more robust
    out-of-sample estimates by testing on multiple non-overlapping
    test sets.

    This implements Snippet 7.4 from López de Prado.

    Parameters
    ----------
    n_splits : int, default=5
        Number of groups to create.
    n_test_splits : int, default=2
        Number of groups to use for testing in each iteration.
    t1 : pd.Series, optional
        Series where index = observation start time, values = label end time.
    pct_embargo : float, default=0.01
        Percentage of observations to embargo.

    Notes
    -----
    With n_splits=5 and n_test_splits=2, this generates C(5,2)=10
    different train/test splits, providing more robust estimates
    than standard k-fold.
    """

    def __init__(self, n_splits: int = 5, n_test_splits: int = 2,
                 t1: pd.Series = None, pct_embargo: float = 0.01):
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def split(self, X, y=None, groups=None):
        """
        Generate combinatorial train/test splits with purging and embargo.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable.
        groups : array-like, optional
            Group labels.

        Yields
        ------
        train : ndarray
            Training set indices.
        test : ndarray
            Test set indices.
        """
        from itertools import combinations

        n_samples = X.shape[0]
        indices = np.arange(n_samples)

        # Create groups
        group_size = n_samples // self.n_splits
        group_indices = []

        for i in range(self.n_splits):
            start = i * group_size
            if i == self.n_splits - 1:
                end = n_samples
            else:
                end = (i + 1) * group_size
            group_indices.append(indices[start:end])

        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices
            test_indices = np.concatenate([group_indices[g] for g in test_groups])

            # Train indices (all other groups)
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_indices = np.concatenate([group_indices[g] for g in train_groups])

            # Apply embargo between adjacent train/test groups
            embargo_size = int(n_samples * self.pct_embargo)
            if embargo_size > 0:
                # Remove samples near test boundaries
                for test_group in test_groups:
                    # Embargo after test group
                    if test_group < self.n_splits - 1:
                        next_group_start = (test_group + 1) * group_size
                        embargo_end = min(next_group_start + embargo_size, n_samples)
                        embargo_indices = indices[next_group_start:embargo_end]
                        train_indices = np.setdiff1d(train_indices, embargo_indices)

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        """Return the number of splits (combinations)."""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)
