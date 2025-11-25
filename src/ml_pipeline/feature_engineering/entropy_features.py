"""
Entropy Features - Optimized entropy calculations for financial time series
"""

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import entropy


@njit
def calculate_entropy_numba(data, bins):
    """
    Calculate Shannon entropy using Numba for speed

    Parameters:
    -----------
    data : np.array
        Data array
    bins : int
        Number of bins for histogram

    Returns:
    --------
    float : Entropy value
    """
    if len(data) == 0:
        return np.nan

    # Calculate histogram
    hist, _ = np.histogram(data, bins=bins)

    # Normalize
    hist = hist[hist > 0]
    prob = hist / np.sum(hist)

    # Calculate entropy
    ent = -np.sum(prob * np.log2(prob))

    return ent


class EntropyFeatures:
    """Entropy feature calculations"""

    @staticmethod
    def calculate_entropy_features_batch(series, window_sizes, bins_list):
        """
        Calculate entropy features for multiple windows and bin sizes

        Parameters:
        -----------
        series : pd.Series
            Time series data
        window_sizes : list
            List of window sizes
        bins_list : list
            List of bin counts

        Returns:
        --------
        pd.DataFrame : Entropy features
        """
        results = pd.DataFrame(index=range(len(series)))

        series_array = np.array(series)

        for window in window_sizes:
            for bins in bins_list:
                col_name = f'entropy_w{window}_b{bins}'
                entropy_values = []

                for i in range(len(series)):
                    if i < window - 1:
                        entropy_values.append(np.nan)
                    else:
                        window_data = series_array[i - window + 1:i + 1]
                        ent = calculate_entropy_numba(window_data, bins)
                        entropy_values.append(ent)

                results[col_name] = entropy_values

        return results

    @staticmethod
    def move_nans_to_front(series):
        """
        Move NaN values to the beginning of the series

        Parameters:
        -----------
        series : pd.Series
            Series with NaN values

        Returns:
        --------
        pd.Series : Series with NaN values at the beginning
        """
        if series.isna().sum() == 0:
            return series

        # Get non-NaN values
        valid_values = series.dropna()

        # Create new series with NaNs at the beginning
        n_nans = len(series) - len(valid_values)
        new_series = pd.Series([np.nan] * n_nans + list(valid_values.values))

        return new_series

    @staticmethod
    def calculate_shannon_entropy(data, bins=50):
        """
        Calculate Shannon entropy

        Parameters:
        -----------
        data : array-like
            Input data
        bins : int
            Number of bins for histogram

        Returns:
        --------
        float : Entropy value
        """
        if len(data) == 0:
            return np.nan

        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]

        if len(hist) == 0:
            return 0.0

        prob = hist / np.sum(hist)
        return entropy(prob, base=2)

    @staticmethod
    def calculate_rolling_entropy(series, window, bins=50):
        """
        Calculate rolling entropy

        Parameters:
        -----------
        series : pd.Series
            Time series
        window : int
            Window size
        bins : int
            Number of bins

        Returns:
        --------
        pd.Series : Rolling entropy
        """
        return series.rolling(window).apply(
            lambda x: EntropyFeatures.calculate_shannon_entropy(x, bins),
            raw=True
        )
