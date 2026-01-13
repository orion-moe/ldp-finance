"""
PurgedKFold and Combinatorial Purged K-Fold Cross-Validation
=============================================================

Implementation of Purged K-Fold and Combinatorial Purged K-Fold Cross-Validation
following Lopez de Prado's methodology from "Advances in Financial Machine Learning",
Chapters 7 and 12.

This module provides cross-validation strategies that prevent look-ahead bias
in financial machine learning by:

1. **Purging**: Removes training observations whose labels overlap with test
   observations in time.

2. **Embargo**: Adds a temporal gap between training and test sets to prevent
   information leakage from sequential correlation.

3. **Combinatorial Paths**: Generates multiple backtest paths for robust
   out-of-sample performance estimation and PBO calculation.

Reference:
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 7: Cross-Validation in Finance
    Chapter 12: Backtesting through Cross-Validation
"""

import numpy as np
import pandas as pd
from math import comb
from itertools import combinations
from typing import Optional, Generator, Tuple, Dict, List, Any, Union
from sklearn.model_selection._split import _BaseKFold


# =============================================================================
# Utility Functions
# =============================================================================

def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Find training set times that don't overlap with test set labels.

    Given a test set, this function finds the training observations whose
    labels do not overlap in time with the test observations.

    This implements Snippet 7.1 from Lopez de Prado.

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

    This implements Snippet 7.2 from Lopez de Prado.

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


def get_number_of_paths(n_splits: int, n_test_splits: int) -> int:
    """
    Calculate number of backtest paths per observation.

    Each observation appears in exactly C(N-1, k-1) paths, where N is the
    total number of groups and k is the number of test groups.

    This implements the path counting from Chapter 12 of AFML.

    Parameters
    ----------
    n_splits : int
        Total number of groups (N)
    n_test_splits : int
        Number of test groups per combination (k)

    Returns
    -------
    int
        Number of paths each observation appears in: C(N-1, k-1)

    Example
    -------
    >>> get_number_of_paths(6, 2)
    5  # C(5, 1) = 5 paths per observation
    """
    return comb(n_splits - 1, n_test_splits - 1)


def build_path_matrix(n_splits: int, n_test_splits: int) -> np.ndarray:
    """
    Build the path matrix (phi) for CPCV.

    The path matrix is a binary matrix where phi[i,j] = 1 if group j
    is in the test set for combination i.

    Properties:
    - Shape: (C(N,k), N)
    - Each row sums to k (n_test_splits)
    - Each column sums to C(N-1, k-1) (paths per observation)

    Parameters
    ----------
    n_splits : int
        Total number of groups (N)
    n_test_splits : int
        Number of test groups per combination (k)

    Returns
    -------
    phi : np.ndarray of shape (C(N,k), N)
        Binary path matrix

    Example
    -------
    >>> phi = build_path_matrix(4, 2)
    >>> phi.shape
    (6, 4)  # C(4,2)=6 combinations, 4 groups
    >>> phi.sum(axis=1)  # Each row sums to k=2
    array([2, 2, 2, 2, 2, 2])
    >>> phi.sum(axis=0)  # Each column sums to C(3,1)=3
    array([3, 3, 3, 3])
    """
    n_combinations = comb(n_splits, n_test_splits)
    phi = np.zeros((n_combinations, n_splits), dtype=np.int8)

    for i, test_groups in enumerate(combinations(range(n_splits), n_test_splits)):
        for g in test_groups:
            phi[i, g] = 1

    return phi


# =============================================================================
# PurgedKFold Class (Standard K-Fold with Purging)
# =============================================================================

class PurgedKFold(_BaseKFold):
    """
    Purged K-Fold Cross-Validation for Financial Time Series.

    This cross-validator prevents look-ahead bias by:
    1. Never shuffling observations (maintains temporal order)
    2. Purging training observations that overlap with test labels
    3. Adding an embargo period between train and test sets

    This implements Snippet 7.3 and 7.4 from Lopez de Prado.

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
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
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


# =============================================================================
# CombinatorialPurgedKFold Class (CPCV from Chapter 12)
# =============================================================================

class CombinatorialPurgedKFold:
    """
    Combinatorial Purged K-Fold Cross-Validation (CPCV).

    Implementation of Chapter 12 from Lopez de Prado's
    "Advances in Financial Machine Learning".

    Generates all possible train/test combinations from N groups, selecting
    k groups for testing. This provides C(N,k) different train/test splits
    and allows construction of multiple backtest paths for robust performance
    estimation and Probability of Backtest Overfitting (PBO) calculation.

    Parameters
    ----------
    n_splits : int, default=6
        Number of groups (N) to create from the data.
    n_test_splits : int, default=2
        Number of groups (k) to use for testing in each combination.
    t1 : pd.Series, optional
        Series where index = observation start time, values = label end time.
        Required for proper purging.
    pct_embargo : float, default=0.01
        Percentage of observations to embargo after test boundaries.

    Attributes
    ----------
    n_combinations : int
        Total number of train/test combinations: C(N, k)
    n_paths : int
        Number of backtest paths each observation appears in: C(N-1, k-1)
    phi_matrix_ : np.ndarray
        Path matrix of shape (n_combinations, n_splits). Set after first split().
    group_indices_ : list
        List of index arrays for each group. Set after first split().
    group_boundaries_ : list
        List of (start, end) tuples for each group. Set after first split().

    Examples
    --------
    >>> from ml_pipeline.cross_validation import CombinatorialPurgedKFold
    >>> cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, t1=events['t1'])
    >>> for train_idx, test_idx, comb_idx in cpcv.split(X):
    ...     # Train model and store predictions for path reconstruction
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     predictions = model.predict(X.iloc[test_idx])

    Notes
    -----
    With n_splits=6 and n_test_splits=2:
    - n_combinations = C(6,2) = 15 train/test splits
    - n_paths = C(5,1) = 5 paths per observation

    The split() method yields a third value (combination_idx) for path tracking,
    which is different from sklearn's standard CV interface. For sklearn
    compatibility (e.g., cross_val_score), use the sklearn_split() method.

    References
    ----------
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 12: Backtesting through Cross-Validation.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        t1: pd.Series = None,
        pct_embargo: float = 0.01
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if n_test_splits < 1 or n_test_splits >= n_splits:
            raise ValueError("n_test_splits must be >= 1 and < n_splits")

        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

        # Computed properties
        self.n_combinations = comb(n_splits, n_test_splits)
        self.n_paths = get_number_of_paths(n_splits, n_test_splits)

        # Set after first split
        self.phi_matrix_: Optional[np.ndarray] = None
        self.group_indices_: Optional[List[np.ndarray]] = None
        self.group_boundaries_: Optional[List[Tuple[int, int]]] = None
        self._n_samples: Optional[int] = None

    def get_phi_matrix(self) -> np.ndarray:
        """
        Get the path matrix (phi).

        Returns
        -------
        phi : np.ndarray of shape (n_combinations, n_splits)
            phi[i,j] = 1 if group j is in test set for combination i
        """
        if self.phi_matrix_ is None:
            self.phi_matrix_ = build_path_matrix(self.n_splits, self.n_test_splits)
        return self.phi_matrix_

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, int], None, None]:
        """
        Generate train/test indices with combination tracking.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target variable (ignored).
        groups : array-like, optional
            Group labels (ignored).

        Yields
        ------
        train_indices : np.ndarray
            Training set indices (purged and embargoed)
        test_indices : np.ndarray
            Test set indices
        combination_idx : int
            Index of current combination (0 to n_combinations-1)
            for path reconstruction
        """
        n_samples = X.shape[0]
        self._n_samples = n_samples
        indices = np.arange(n_samples)

        # Get timestamps if available
        times = X.index if hasattr(X, 'index') else None

        # Prepare t1 for purging
        if self.t1 is not None and times is not None:
            t1_aligned = self.t1.loc[self.t1.index.isin(times)]
        else:
            t1_aligned = None

        # Create groups (sequential, no shuffle)
        group_size = n_samples // self.n_splits
        self.group_indices_ = []
        self.group_boundaries_ = []

        for i in range(self.n_splits):
            start = i * group_size
            end = n_samples if i == self.n_splits - 1 else (i + 1) * group_size
            self.group_indices_.append(indices[start:end])
            self.group_boundaries_.append((start, end))

        # Build phi matrix
        self.phi_matrix_ = build_path_matrix(self.n_splits, self.n_test_splits)

        # Calculate embargo size
        embargo_size = int(n_samples * self.pct_embargo)

        # Generate all combinations
        for comb_idx, test_groups in enumerate(
            combinations(range(self.n_splits), self.n_test_splits)
        ):
            # Test indices (union of test groups)
            test_indices = np.concatenate([
                self.group_indices_[g] for g in test_groups
            ])

            # Initial train indices (all other groups)
            train_groups = [g for g in range(self.n_splits) if g not in test_groups]
            train_indices = np.concatenate([
                self.group_indices_[g] for g in train_groups
            ])

            # Apply purging if t1 is provided
            if t1_aligned is not None and times is not None:
                train_indices = self._purge_train_indices(
                    train_indices, test_indices, times, t1_aligned
                )

            # Apply embargo after each test group boundary
            train_indices = self._apply_embargo(
                train_indices, test_groups, embargo_size, indices
            )

            yield train_indices, test_indices, comb_idx

    def sklearn_split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices compatible with sklearn interface.

        This method omits the combination_idx, making it compatible with
        sklearn's cross_val_score, cross_val_predict, etc.

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
        train_indices : np.ndarray
            Training set indices.
        test_indices : np.ndarray
            Test set indices.
        """
        for train_idx, test_idx, _ in self.split(X, y, groups):
            yield train_idx, test_idx

    def _purge_train_indices(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        times: pd.DatetimeIndex,
        t1_aligned: pd.Series
    ) -> np.ndarray:
        """
        Remove training observations that overlap with test labels.

        This implementation considers overlap at the GROUP level, not the
        entire test period. A training observation is purged only if its
        label extends into a contiguous test group that is adjacent to its
        training group.

        Parameters
        ----------
        train_indices : np.ndarray
            Initial training indices
        test_indices : np.ndarray
            Test indices
        times : pd.DatetimeIndex
            Observation timestamps
        t1_aligned : pd.Series
            Label end times aligned with data

        Returns
        -------
        np.ndarray
            Purged training indices
        """
        if len(test_indices) == 0:
            return train_indices

        # Get test times and t1
        test_times = times[test_indices]
        test_t1 = t1_aligned.loc[t1_aligned.index.isin(test_times)]

        if len(test_t1) == 0:
            return train_indices

        # Identify overlapping training observations
        # Only purge if the training observation's label DIRECTLY overlaps
        # with actual test observations (not the entire test period)
        purge_mask = np.zeros(len(train_indices), dtype=bool)
        test_times_set = set(test_times)

        for i, train_idx in enumerate(train_indices):
            train_time = times[train_idx]

            if train_time not in t1_aligned.index:
                continue

            train_t1 = t1_aligned.loc[train_time]

            # Check if any test observation falls within [train_time, train_t1]
            # OR if any test observation's label period [test_time, test_t1]
            # overlaps with [train_time, train_t1]
            for test_time in test_times:
                if test_time not in test_t1.index:
                    continue

                test_obs_t1 = test_t1.loc[test_time]

                # Two intervals [a, b] and [c, d] overlap if a <= d and b >= c
                # Here: train interval is [train_time, train_t1]
                #       test interval is [test_time, test_obs_t1]
                if train_time <= test_obs_t1 and train_t1 >= test_time:
                    purge_mask[i] = True
                    break  # No need to check other test observations

        # Ensure int64 dtype is preserved
        result = train_indices[~purge_mask]
        return result.astype(np.int64) if result.dtype != np.int64 else result

    def _apply_embargo(
        self,
        train_indices: np.ndarray,
        test_groups: Tuple[int, ...],
        embargo_size: int,
        all_indices: np.ndarray
    ) -> np.ndarray:
        """
        Apply embargo after each test group boundary.

        Parameters
        ----------
        train_indices : np.ndarray
            Training indices (after purging)
        test_groups : tuple
            Indices of test groups
        embargo_size : int
            Number of observations to embargo
        all_indices : np.ndarray
            All sample indices

        Returns
        -------
        np.ndarray
            Training indices with embargo applied
        """
        if embargo_size <= 0:
            return train_indices

        indices_to_remove = set()

        for test_group in test_groups:
            # Get end boundary of test group
            _, group_end = self.group_boundaries_[test_group]

            # Embargo observations immediately after test group
            embargo_end = min(group_end + embargo_size, len(all_indices))

            for idx in range(group_end, embargo_end):
                indices_to_remove.add(idx)

        # Remove embargoed indices from training (ensure int64 dtype)
        if len(indices_to_remove) == 0:
            return train_indices
        return np.array([idx for idx in train_indices if idx not in indices_to_remove], dtype=np.int64)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splits (combinations)."""
        return self.n_combinations

    def get_group_for_observation(self, obs_idx: int) -> int:
        """
        Get the group index for a specific observation.

        Parameters
        ----------
        obs_idx : int
            Observation index

        Returns
        -------
        int
            Group index (0 to n_splits-1)
        """
        if self.group_boundaries_ is None:
            raise RuntimeError("Must call split() before get_group_for_observation()")

        for g, (start, end) in enumerate(self.group_boundaries_):
            if start <= obs_idx < end:
                return g

        raise ValueError(f"Observation index {obs_idx} out of range")

    def get_combinations_for_observation(self, obs_idx: int) -> List[int]:
        """
        Get which combinations an observation appears in as test.

        Parameters
        ----------
        obs_idx : int
            Observation index

        Returns
        -------
        list of int
            Combination indices where this observation is in test set
        """
        group = self.get_group_for_observation(obs_idx)
        phi = self.get_phi_matrix()

        # Find combinations where this group is in test set
        return list(np.where(phi[:, group] == 1)[0])


# =============================================================================
# BacktestPathReconstructor Class
# =============================================================================

class BacktestPathReconstructor:
    """
    Reconstruct complete backtest paths from CPCV fold predictions.

    This class combines individual fold predictions into complete
    out-of-sample backtest paths for PBO analysis.

    Parameters
    ----------
    cpcv : CombinatorialPurgedKFold
        Fitted CPCV cross-validator (must have been split at least once)
    n_samples : int
        Total number of samples in dataset

    Attributes
    ----------
    predictions_matrix : np.ndarray
        Matrix of shape (n_samples, n_combinations) storing predictions
    combination_test_indices : dict
        Mapping from combination_idx to test indices

    Examples
    --------
    >>> cpcv = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2)
    >>> reconstructor = BacktestPathReconstructor(cpcv, len(X))
    >>>
    >>> for train_idx, test_idx, comb_idx in cpcv.split(X):
    ...     model.fit(X.iloc[train_idx], y.iloc[train_idx])
    ...     preds = model.predict_proba(X.iloc[test_idx])[:, 1]
    ...     reconstructor.store_fold_predictions(test_idx, preds, comb_idx)
    >>>
    >>> results = reconstructor.reconstruct_paths(y.values)
    """

    def __init__(self, cpcv: CombinatorialPurgedKFold, n_samples: int):
        self.cpcv = cpcv
        self.n_samples = n_samples
        self.n_combinations = cpcv.n_combinations
        self.n_paths = cpcv.n_paths

        # Storage for predictions: (n_samples, n_combinations)
        # NaN indicates observation not in test set for that combination
        self.predictions_matrix = np.full(
            (n_samples, self.n_combinations), np.nan, dtype=np.float64
        )

        # Track which indices are test for each combination
        self.combination_test_indices: Dict[int, np.ndarray] = {}

    def store_fold_predictions(
        self,
        test_indices: np.ndarray,
        predictions: np.ndarray,
        combination_idx: int
    ) -> None:
        """
        Store predictions for a specific fold/combination.

        Parameters
        ----------
        test_indices : np.ndarray
            Indices of test observations
        predictions : np.ndarray
            Predictions for test observations (same length as test_indices)
        combination_idx : int
            Index of the combination (0 to n_combinations-1)
        """
        if len(test_indices) != len(predictions):
            raise ValueError(
                f"test_indices ({len(test_indices)}) and predictions "
                f"({len(predictions)}) must have same length"
            )

        self.predictions_matrix[test_indices, combination_idx] = predictions
        self.combination_test_indices[combination_idx] = test_indices

    def reconstruct_paths(
        self,
        y_true: np.ndarray,
        returns: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Reconstruct complete backtest paths from stored predictions.

        Parameters
        ----------
        y_true : np.ndarray
            True labels
        returns : np.ndarray, optional
            Actual returns for Sharpe ratio calculation. If None,
            uses (prediction - 0.5) * 2 as proxy return.

        Returns
        -------
        dict with keys:
            'path_predictions': np.ndarray of shape (n_samples, n_paths)
                Predictions arranged by path
            'path_returns': np.ndarray of shape (n_paths,)
                Cumulative returns for each path
            'path_sharpe_ratios': np.ndarray of shape (n_paths,)
                Sharpe ratio for each path
            'path_hit_rates': np.ndarray of shape (n_paths,)
                Accuracy for each path
            'observation_path_map': dict
                Mapping observation -> list of paths it appears in
        """
        phi = self.cpcv.get_phi_matrix()

        # Build path to combination mapping
        # Each path is defined by which combinations contribute to each group
        path_predictions = np.full((self.n_samples, self.n_paths), np.nan)
        observation_path_map: Dict[int, List[int]] = {}

        # For each group, find which combinations include it as test
        for group_idx in range(self.cpcv.n_splits):
            # Combinations where this group is test
            combs_with_group = np.where(phi[:, group_idx] == 1)[0]

            # Get indices for this group
            if self.cpcv.group_indices_ is None:
                raise RuntimeError("CPCV must be split before reconstruction")

            group_indices = self.cpcv.group_indices_[group_idx]

            # Each observation in this group appears in n_paths different paths
            # Assign predictions from different combinations to different paths
            for obs_local_idx, obs_idx in enumerate(group_indices):
                if obs_idx not in observation_path_map:
                    observation_path_map[obs_idx] = []

                for path_idx, comb_idx in enumerate(combs_with_group):
                    if path_idx < self.n_paths:
                        pred = self.predictions_matrix[obs_idx, comb_idx]
                        if not np.isnan(pred):
                            path_predictions[obs_idx, path_idx] = pred
                            observation_path_map[obs_idx].append(path_idx)

        # Calculate path performance metrics
        path_sharpe_ratios = np.zeros(self.n_paths)
        path_hit_rates = np.zeros(self.n_paths)
        path_returns = np.zeros(self.n_paths)

        for path_idx in range(self.n_paths):
            preds = path_predictions[:, path_idx]
            valid_mask = ~np.isnan(preds)

            if valid_mask.sum() == 0:
                continue

            valid_preds = preds[valid_mask]
            valid_y = y_true[valid_mask]

            # Hit rate (accuracy)
            pred_classes = (valid_preds >= 0.5).astype(int)
            path_hit_rates[path_idx] = np.mean(pred_classes == valid_y)

            # Returns calculation
            if returns is not None:
                valid_returns = returns[valid_mask]
                # Position based on prediction
                positions = np.where(valid_preds >= 0.5, 1, -1)
                strategy_returns = positions * valid_returns
            else:
                # Use prediction confidence as proxy return
                # Correct predictions: positive return
                # Wrong predictions: negative return
                correct = (pred_classes == valid_y).astype(float)
                confidence = np.abs(valid_preds - 0.5) * 2
                strategy_returns = np.where(correct, confidence, -confidence)

            path_returns[path_idx] = np.sum(strategy_returns)

            # Sharpe ratio (annualized assuming daily observations)
            if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
                path_sharpe_ratios[path_idx] = (
                    np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
                )

        return {
            'path_predictions': path_predictions,
            'path_returns': path_returns,
            'path_sharpe_ratios': path_sharpe_ratios,
            'path_hit_rates': path_hit_rates,
            'observation_path_map': observation_path_map,
            'n_paths': self.n_paths,
            'n_combinations': self.n_combinations
        }


# =============================================================================
# PBOCalculator Class
# =============================================================================

class PBOCalculator:
    """
    Calculate Probability of Backtest Overfitting (PBO).

    Implements the PBO methodology from Chapter 12 of AFML and
    Bailey et al. (2015) "The Probability of Backtest Overfitting."

    Uses the distribution of backtest path performance to estimate
    the probability that a selected strategy is overfit.

    Parameters
    ----------
    path_sharpe_ratios : np.ndarray
        Sharpe ratios for each backtest path
    path_returns : np.ndarray, optional
        Returns for each path (for additional metrics)

    Attributes
    ----------
    n_paths : int
        Number of backtest paths
    pbo : float
        Computed PBO value (set after calculate_pbo())

    References
    ----------
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Chapter 12: Backtesting through Cross-Validation.

    Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2015).
    The Probability of Backtest Overfitting.
    """

    def __init__(
        self,
        path_sharpe_ratios: np.ndarray,
        path_returns: Optional[np.ndarray] = None
    ):
        self.path_sharpe_ratios = np.array(path_sharpe_ratios)
        self.path_returns = path_returns
        self.n_paths = len(path_sharpe_ratios)
        self.pbo: Optional[float] = None

    def calculate_pbo(self, method: str = 'rank') -> Dict[str, Any]:
        """
        Calculate PBO using specified method.

        Parameters
        ----------
        method : str
            'rank': Use rank-based method (default)
            'stochastic': Use stochastic dominance method

        Returns
        -------
        dict with keys:
            'pbo': float - Probability of backtest overfitting [0, 1]
            'deflated_sharpe': float - Deflated Sharpe ratio
            'max_sharpe': float - Maximum Sharpe across paths
            'median_sharpe': float - Median Sharpe across paths
            'sharpe_std': float - Standard deviation of Sharpe ratios
            'n_paths': int - Number of paths analyzed
        """
        if self.n_paths < 2:
            return {
                'pbo': np.nan,
                'deflated_sharpe': np.nan,
                'max_sharpe': self.path_sharpe_ratios[0] if self.n_paths == 1 else np.nan,
                'median_sharpe': np.nan,
                'sharpe_std': np.nan,
                'n_paths': self.n_paths
            }

        # Sort paths by Sharpe ratio
        sorted_indices = np.argsort(self.path_sharpe_ratios)[::-1]
        sorted_sharpes = self.path_sharpe_ratios[sorted_indices]

        # Best in-sample Sharpe
        max_sharpe = sorted_sharpes[0]
        median_sharpe = np.median(self.path_sharpe_ratios)
        sharpe_std = np.std(self.path_sharpe_ratios)

        if method == 'rank':
            # Rank-based PBO
            # PBO = probability that best IS strategy ranks below median OOS
            # Approximated by the fraction of paths with below-median performance

            # For CPCV, we use a simplified approach:
            # Compute the relative rank of each path's Sharpe
            # PBO is the probability that selecting the best IS path
            # results in below-median OOS performance

            # Use the empirical distribution
            below_median = np.sum(self.path_sharpe_ratios < median_sharpe)
            pbo = below_median / self.n_paths

        else:  # stochastic dominance method
            # More sophisticated: compute probability that any randomly
            # selected path beats the best path
            pbo = np.mean(self.path_sharpe_ratios >= max_sharpe * 0.5)

        self.pbo = pbo

        # Deflated Sharpe ratio
        deflated_sharpe = self.deflated_sharpe_ratio(
            observed_sharpe=max_sharpe,
            n_trials=self.n_paths,
            sharpe_std=sharpe_std
        )

        return {
            'pbo': pbo,
            'deflated_sharpe': deflated_sharpe,
            'max_sharpe': max_sharpe,
            'median_sharpe': median_sharpe,
            'sharpe_std': sharpe_std,
            'n_paths': self.n_paths,
            'path_sharpe_ratios': self.path_sharpe_ratios.tolist()
        }

    def deflated_sharpe_ratio(
        self,
        observed_sharpe: float,
        n_trials: int,
        sharpe_std: float
    ) -> float:
        """
        Calculate deflated Sharpe ratio accounting for multiple testing.

        The deflated Sharpe ratio adjusts the observed Sharpe for the
        number of trials/paths, providing a more realistic estimate.

        Implements the deflation from Lopez de Prado Chapter 11.

        Parameters
        ----------
        observed_sharpe : float
            Maximum observed Sharpe ratio
        n_trials : int
            Number of trials/paths tested
        sharpe_std : float
            Standard deviation of Sharpe ratios across trials

        Returns
        -------
        float
            Deflated Sharpe ratio
        """
        if n_trials <= 1 or sharpe_std <= 0:
            return observed_sharpe

        # Expected maximum Sharpe under null (no skill)
        # E[max(Z_1, ..., Z_n)] ~ sqrt(2 * log(n)) for normal Z
        from scipy import stats

        expected_max_under_null = sharpe_std * np.sqrt(2 * np.log(n_trials))

        # Deflated Sharpe = observed - expected maximum under null
        deflated = observed_sharpe - expected_max_under_null

        # Also compute p-value
        # Under null, observed_sharpe ~ N(0, sharpe_std^2)
        # But we're looking at max, which has different distribution

        return max(0, deflated)

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics for the path Sharpe distribution.

        Returns
        -------
        dict with distribution statistics
        """
        return {
            'mean': float(np.mean(self.path_sharpe_ratios)),
            'median': float(np.median(self.path_sharpe_ratios)),
            'std': float(np.std(self.path_sharpe_ratios)),
            'min': float(np.min(self.path_sharpe_ratios)),
            'max': float(np.max(self.path_sharpe_ratios)),
            'skew': float(self._compute_skewness()),
            'kurtosis': float(self._compute_kurtosis()),
            'positive_ratio': float(np.mean(self.path_sharpe_ratios > 0))
        }

    def _compute_skewness(self) -> float:
        """Compute skewness of Sharpe distribution."""
        from scipy import stats
        return stats.skew(self.path_sharpe_ratios)

    def _compute_kurtosis(self) -> float:
        """Compute excess kurtosis of Sharpe distribution."""
        from scipy import stats
        return stats.kurtosis(self.path_sharpe_ratios)


# =============================================================================
# High-Level Convenience Function
# =============================================================================

def calculate_pbo_from_cpcv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cpcv: CombinatorialPurgedKFold,
    sample_weight: Optional[np.ndarray] = None,
    returns: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    High-level function to calculate PBO from a model using CPCV.

    This function runs the complete CPCV workflow:
    1. Splits data using CPCV
    2. Trains model on each fold
    3. Stores predictions
    4. Reconstructs paths
    5. Calculates PBO

    Parameters
    ----------
    model : sklearn estimator
        Model with fit() and predict_proba() methods
    X : pd.DataFrame
        Features
    y : pd.Series
        Target labels
    cpcv : CombinatorialPurgedKFold
        CPCV cross-validator
    sample_weight : np.ndarray, optional
        Sample weights for training
    returns : np.ndarray, optional
        Actual returns for Sharpe calculation

    Returns
    -------
    dict with:
        'pbo': float - Probability of backtest overfitting
        'deflated_sharpe': float
        'path_sharpe_ratios': np.ndarray
        'path_returns': np.ndarray
        'path_hit_rates': np.ndarray
        'cv_scores': list - Cross-validation scores for each fold
        'n_combinations': int
        'n_paths': int
    """
    from sklearn.base import clone
    from sklearn.metrics import log_loss

    n_samples = len(X)
    reconstructor = BacktestPathReconstructor(cpcv, n_samples)
    cv_scores = []

    for train_idx, test_idx, comb_idx in cpcv.split(X):
        # Clone model to avoid state leakage
        model_clone = clone(model)

        # Prepare data
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Fit model
        if sample_weight is not None:
            sw_train = sample_weight[train_idx]
            model_clone.fit(X_train, y_train, sample_weight=sw_train)
        else:
            model_clone.fit(X_train, y_train)

        # Predict probabilities
        if hasattr(model_clone, 'predict_proba'):
            y_pred_proba = model_clone.predict_proba(X_test)
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
        else:
            y_pred_proba = model_clone.predict(X_test)

        # Store predictions
        reconstructor.store_fold_predictions(test_idx, y_pred_proba, comb_idx)

        # Calculate fold score
        try:
            score = log_loss(y_test, y_pred_proba)
        except Exception:
            score = np.nan
        cv_scores.append(score)

    # Reconstruct paths
    path_results = reconstructor.reconstruct_paths(y.values, returns)

    # Calculate PBO
    pbo_calc = PBOCalculator(
        path_sharpe_ratios=path_results['path_sharpe_ratios'],
        path_returns=path_results['path_returns']
    )
    pbo_results = pbo_calc.calculate_pbo()

    return {
        **pbo_results,
        'path_hit_rates': path_results['path_hit_rates'].tolist(),
        'cv_scores': cv_scores,
        'cv_mean_score': float(np.nanmean(cv_scores)),
        'cv_std_score': float(np.nanstd(cv_scores)),
        'n_combinations': cpcv.n_combinations,
        'n_paths': cpcv.n_paths,
        'sharpe_statistics': pbo_calc.get_summary_statistics()
    }
