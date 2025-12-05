"""
Fast Microstructure Features - Numba-optimized calculations
"""

import numpy as np
import pandas as pd
from numba import njit


@njit
def rolling_std_numba(data, window):
    """Calculate rolling standard deviation with Numba"""
    result = np.empty(len(data))
    result[:] = np.nan

    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        result[i] = np.std(window_data)

    return result


@njit
def rolling_mean_numba(data, window):
    """Calculate rolling mean with Numba"""
    result = np.empty(len(data))
    result[:] = np.nan

    for i in range(window - 1, len(data)):
        window_data = data[i - window + 1:i + 1]
        result[i] = np.mean(window_data)

    return result


class FastMicrostructureFeatures:
    """Fast microstructure calculations using Numba"""

    @staticmethod
    def calculate_fast_features(prices, volumes, windows=[50, 100]):
        """
        Calculate microstructure features quickly

        Parameters:
        -----------
        prices : pd.Series
            Price series
        volumes : pd.Series
            Volume series
        windows : list
            Window sizes

        Returns:
        --------
        pd.DataFrame : Microstructure features
        """
        result = pd.DataFrame(index=prices.index)

        price_array = prices.values
        volume_array = volumes.values

        for window in windows:
            # Volatility
            result[f'volatility_{window}'] = rolling_std_numba(price_array, window)

            # Volume mean
            result[f'volume_mean_{window}'] = rolling_mean_numba(volume_array, window)

        return result
