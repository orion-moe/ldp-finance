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


@njit
def lempel_ziv_complexity_numba(binary_sequence):
    """
    Calculate Lempel-Ziv complexity (LZ76) for binary sequence.

    Returns normalized complexity: n_patterns / (n / log2(n))
    Higher values = more complex/random sequence

    Parameters:
    -----------
    binary_sequence : np.array
        Binary (0/1) sequence

    Returns:
    --------
    float : Normalized LZ complexity
    """
    n = len(binary_sequence)
    if n <= 1:
        return np.nan

    # LZ76 algorithm
    i = 0
    n_patterns = 1

    while i < n:
        # Find longest match in prefix
        max_match_len = 0
        for j in range(i):
            match_len = 0
            while (i + match_len < n and
                   j + match_len < i and
                   binary_sequence[j + match_len] == binary_sequence[i + match_len]):
                match_len += 1
            if match_len > max_match_len:
                max_match_len = match_len

        # Move past the match + 1 new symbol
        i += max_match_len + 1
        if i < n:
            n_patterns += 1

    # Normalize by theoretical maximum
    if n > 1:
        normalized = n_patterns / (n / np.log2(n))
    else:
        normalized = np.nan

    return normalized


@njit
def kontoyiannis_entropy_numba(sequence):
    """
    Calculate Kontoyiannis entropy estimate based on match lengths.

    H ≈ log2(n) / mean(match_lengths)

    Lower mean match length = higher entropy = more random

    Parameters:
    -----------
    sequence : np.array
        Discretized sequence (integers)

    Returns:
    --------
    float : Entropy estimate
    """
    n = len(sequence)
    if n < 4:
        return np.nan

    # Pre-allocate array for match lengths
    match_lengths = np.zeros(n - 1)
    count = 0

    for i in range(1, n):
        # Find longest match in previous subsequence
        max_match = 0
        for j in range(i):
            match_len = 0
            while (i + match_len < n and
                   j + match_len < i and
                   sequence[j + match_len] == sequence[i + match_len]):
                match_len += 1
            if match_len > max_match:
                max_match = match_len

        # Add 1 to avoid log(0), convention from Kontoyiannis
        match_lengths[count] = max_match + 1
        count += 1

    if count == 0:
        return np.nan

    mean_match = np.mean(match_lengths[:count])

    # Entropy estimate
    entropy = np.log2(n) / mean_match

    return entropy


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

    @staticmethod
    def calculate_lempel_ziv_batch(series, window_sizes):
        """
        Calculate Lempel-Ziv complexity for multiple window sizes.

        Parameters:
        -----------
        series : pd.Series
            Time series data
        window_sizes : list
            List of window sizes

        Returns:
        --------
        pd.DataFrame : LZ complexity features with columns lz_{window}
        """
        series_array = series.values if hasattr(series, 'values') else np.array(series)
        results = {}

        for window in window_sizes:
            col_name = f'lz_{window}'
            lz_values = []

            for i in range(len(series_array)):
                if i < window - 1:
                    lz_values.append(np.nan)
                else:
                    window_data = series_array[i - window + 1:i + 1]
                    # Discretize to binary (above/below median)
                    median = np.nanmedian(window_data)
                    binary_seq = (window_data > median).astype(np.int32)
                    lz = lempel_ziv_complexity_numba(binary_seq)
                    lz_values.append(lz)

            results[col_name] = lz_values

        index = series.index if hasattr(series, 'index') else None
        return pd.DataFrame(results, index=index)

    @staticmethod
    def calculate_kontoyiannis_batch(series, window_sizes):
        """
        Calculate Kontoyiannis entropy for multiple window sizes.

        Parameters:
        -----------
        series : pd.Series
            Time series data
        window_sizes : list
            List of window sizes

        Returns:
        --------
        pd.DataFrame : Kontoyiannis entropy features with columns kont_{window}
        """
        series_array = series.values if hasattr(series, 'values') else np.array(series)
        results = {}

        for window in window_sizes:
            col_name = f'kont_{window}'
            kont_values = []

            for i in range(len(series_array)):
                if i < window - 1:
                    kont_values.append(np.nan)
                else:
                    window_data = series_array[i - window + 1:i + 1]
                    # Discretize to 4 levels (quartiles)
                    bins = np.nanpercentile(window_data, [25, 50, 75])
                    discrete_seq = np.digitize(window_data, bins).astype(np.int32)
                    kont = kontoyiannis_entropy_numba(discrete_seq)
                    kont_values.append(kont)

            results[col_name] = kont_values

        index = series.index if hasattr(series, 'index') else None
        return pd.DataFrame(results, index=index)
