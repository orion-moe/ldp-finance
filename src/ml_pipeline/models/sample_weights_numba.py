"""
Sample Weights with Numba - Optimized sample weight calculations
"""

import numpy as np
import pandas as pd
from numba import njit
import time


@njit
def calculate_uniqueness_numba(t1_array, timestamps):
    """
    Calculate uniqueness weights using Numba

    Parameters:
    -----------
    t1_array : np.array
        End times for each event
    timestamps : np.array
        Event timestamps

    Returns:
    --------
    np.array : Uniqueness weights
    """
    n = len(timestamps)
    uniqueness = np.zeros(n)

    for i in range(n):
        # Count concurrent events
        t_start = timestamps[i]
        t_end = t1_array[i]

        if np.isnan(t_end):
            uniqueness[i] = 1.0
            continue

        concurrent_count = 0

        for j in range(n):
            if i == j:
                concurrent_count += 1
                continue

            t_start_j = timestamps[j]
            t_end_j = t1_array[j]

            if np.isnan(t_end_j):
                continue

            # Check if events overlap
            if t_start_j <= t_end and t_end_j >= t_start:
                concurrent_count += 1

        if concurrent_count > 0:
            uniqueness[i] = 1.0 / concurrent_count
        else:
            uniqueness[i] = 1.0

    return uniqueness


# Note: This function is kept for backward compatibility but NOT used
# per López de Prado methodology - no normalization after time decay
@njit
def normalize_weights_numba(weights):
    """
    [DEPRECATED - Not used in López de Prado methodology]

    Previously normalized weights to have mean = 1.0
    Now kept only for backward compatibility.

    López de Prado approach: NO normalization after time decay
    Final weights = uniqueness × magnitude × time_decay

    Parameters:
    -----------
    weights : np.array
        Raw weights

    Returns:
    --------
    np.array : Same weights (no normalization)
    """
    return weights  # Return as-is, no normalization


class SampleWeightsCalculator:
    """
    Calculate sample weights for machine learning

    Combines:
    - Uniqueness (based on event overlap)
    - Magnitude (based on target values)
    - Returns (based on actual returns)
    """

    def __init__(self):
        self.weights = None
        self.diagnostics = {}

    def calculate_sample_weights(self, triple_barrier_events, series_prim, events,
                                apply_time_decay=True, decay_rate=0.98):
        """
        Calculate sample weights using Numba-optimized functions with time decay

        Parameters:
        -----------
        triple_barrier_events : pd.DataFrame
            Triple barrier results
        series_prim : pd.DataFrame
            Primary series data
        events : pd.DataFrame
            Detected events
        apply_time_decay : bool, default=True
            Whether to apply time decay to weights
        decay_rate : float, default=0.98
            Time decay rate (between 0 and 1)

        Returns:
        --------
        dict : Weights and diagnostics
        """
        start_time = time.time()

        # Extract timestamps and t1
        if isinstance(triple_barrier_events.index, pd.DatetimeIndex):
            timestamps = triple_barrier_events.index.values.astype('int64') / 10**9  # Convert to seconds
        else:
            # Ensure it's a numpy array, not pandas Index
            timestamps = np.arange(len(triple_barrier_events), dtype=np.float64)

        # Extract t1 values
        if 't1' in triple_barrier_events.columns:
            t1_series = triple_barrier_events['t1']

            # Check if series is not empty and first value is Timestamp
            if len(t1_series) > 0 and isinstance(t1_series.iloc[0], pd.Timestamp):
                t1_array = t1_series.values.astype('int64') / 10**9
            else:
                t1_array = t1_series.values.astype('float64')

            # Replace NaT with nan
            t1_array = np.where(pd.isna(t1_series), np.nan, t1_array)
        else:
            # If no t1, use simple weights
            t1_array = np.full(len(triple_barrier_events), np.nan)

        # Calculate uniqueness weights
        # Ensure both arrays are numpy arrays with correct dtype
        t1_array = np.asarray(t1_array, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        uniqueness_weights = calculate_uniqueness_numba(t1_array, timestamps)

        # Note: Magnitude weights REMOVED
        # Following pure López de Prado methodology
        # Weight = Uniqueness × Time Decay only

        # Apply time decay if requested
        if apply_time_decay:
            # Calculate time decay weights
            if isinstance(triple_barrier_events.index, pd.DatetimeIndex):
                # For datetime index, calculate days from last event
                time_diffs = (triple_barrier_events.index[-1] - triple_barrier_events.index).days
                # Convert to numpy array
                time_decay_weights = np.array(decay_rate ** time_diffs, dtype=np.float64)
            else:
                # For numeric index, use positions
                positions = np.arange(len(triple_barrier_events))
                time_decay_weights = decay_rate ** (len(triple_barrier_events) - 1 - positions)
        else:
            time_decay_weights = np.ones(len(triple_barrier_events))

        # López de Prado pure methodology:
        # weight = uniqueness × time_decay
        #
        # 1. Uniqueness weights ∈ [0, 1] - information uniqueness based on event overlap
        # 2. Time decay - exponential decay for recency bias
        # NO magnitude weights (avoids data leakage)
        # NO final normalization - weights represent true relative importance

        # Ensure all weights are numpy arrays
        uniqueness_weights = np.asarray(uniqueness_weights, dtype=np.float64)
        time_decay_weights = np.asarray(time_decay_weights, dtype=np.float64)

        # Pure López de Prado: weight = uniqueness × time_decay
        final_weights = uniqueness_weights * time_decay_weights

        # NO normalization - use weights as-is per López de Prado
        normalized_weights = final_weights  # Keep name for compatibility

        # Calculate diagnostics
        calc_time = time.time() - start_time

        valid_events = np.sum(~np.isnan(uniqueness_weights))
        avg_uniqueness = np.mean(uniqueness_weights[~np.isnan(uniqueness_weights)])

        self.diagnostics = {
            'calculation_time': calc_time,
            'total_events': len(triple_barrier_events),
            'valid_events': int(valid_events),
            'avg_uniqueness': float(avg_uniqueness),
            'time_decay_applied': apply_time_decay,
            'decay_rate': decay_rate if apply_time_decay else None,
            'methodology': 'Pure López de Prado: weight = uniqueness × time_decay',
            'uniqueness_stats': {
                'min': float(np.min(uniqueness_weights)),
                'max': float(np.max(uniqueness_weights)),
                'mean': float(np.mean(uniqueness_weights)),
                'std': float(np.std(uniqueness_weights)),
                'info': 'Based on event overlap, range [0,1]'
            },
            'final_weight_stats': {
                'min': float(np.min(normalized_weights)),
                'max': float(np.max(normalized_weights)),
                'mean': float(np.mean(normalized_weights)),
                'std': float(np.std(normalized_weights)),
                'info': 'uniqueness × time_decay, no normalization'
            },
            'time_decay_stats': {
                'min': float(np.min(time_decay_weights)),
                'max': float(np.max(time_decay_weights)),
                'mean': float(np.mean(time_decay_weights))
            } if apply_time_decay else None
        }

        return {
            'normalized_weights': normalized_weights,
            'uniqueness_weights': uniqueness_weights,
            'time_decay_weights': time_decay_weights,
            'diagnostics': self.diagnostics
        }

    def calculate_time_decay_weights(self, timestamps, half_life=100):
        """
        Calculate time decay weights

        NOTE: This is a legacy method. Time decay is now integrated in
        calculate_sample_weights() with the apply_time_decay parameter.

        Parameters:
        -----------
        timestamps : array-like
            Event timestamps
        half_life : float
            Half-life for exponential decay

        Returns:
        --------
        np.array : Time decay weights
        """
        if isinstance(timestamps, pd.DatetimeIndex):
            time_diff = (timestamps[-1] - timestamps).total_seconds()
        else:
            time_diff = timestamps[-1] - timestamps

        decay_weights = np.exp(-np.log(2) * time_diff / half_life)

        return normalize_weights_numba(decay_weights)
