"""
Time Decay Functions for Financial Time Series Analysis

This module provides functions for applying time decay to financial data,
useful for giving more weight to recent observations in modeling and analysis.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Literal


def calculate_time_decay(
    timestamps: Union[pd.DatetimeIndex, pd.Series],
    decay_rate: float = 0.98,
    time_unit: Literal['days', 'hours', 'months'] = 'days',
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None
) -> pd.Series:
    """
    Calculate time decay factors for a series of timestamps.

    Parameters
    ----------
    timestamps : pd.DatetimeIndex or pd.Series
        The timestamps to calculate decay factors for
    decay_rate : float, default=0.98
        The decay rate (between 0 and 1). Lower values = faster decay
    time_unit : str, default='days'
        Time unit for decay calculation: 'days', 'hours', or 'months'
    reference_date : datetime-like, optional
        Reference date for decay calculation. If None, uses the most recent timestamp

    Returns
    -------
    pd.Series
        Decay factors for each timestamp (values between 0 and 1)

    Examples
    --------
    >>> dates = pd.date_range('2024-01-01', periods=10, freq='D')
    >>> decay = calculate_time_decay(dates, decay_rate=0.95)
    """
    # Convert to DatetimeIndex if needed
    if isinstance(timestamps, pd.Series):
        timestamps = pd.to_datetime(timestamps)
        index = timestamps.index
    elif isinstance(timestamps, pd.DatetimeIndex):
        index = timestamps
    else:
        timestamps = pd.to_datetime(timestamps)
        index = range(len(timestamps))

    # Handle no decay case
    if decay_rate >= 1.0:
        return pd.Series(1.0, index=index)

    # Set reference date if not provided
    if reference_date is None:
        reference_date = timestamps.max()
    else:
        reference_date = pd.to_datetime(reference_date)

    # Calculate time differences
    time_diff = reference_date - timestamps

    # Convert to appropriate time unit
    if time_unit == 'days':
        time_distance = time_diff.total_seconds() / 86400
    elif time_unit == 'hours':
        time_distance = time_diff.total_seconds() / 3600
    elif time_unit == 'months':
        time_distance = time_diff.total_seconds() / (86400 * 30.44)
    else:
        raise ValueError(f"Invalid time_unit: {time_unit}. Must be 'days', 'hours', or 'months'")

    # Convert to Series if needed
    if not isinstance(time_distance, pd.Series):
        time_distance = pd.Series(time_distance, index=index)

    # Calculate decay factors
    decay_factors = decay_rate ** time_distance

    # Clip to ensure values are between 0 and 1
    decay_factors = decay_factors.clip(lower=0.0, upper=1.0)

    return decay_factors


def exponential_decay_weights(
    n_samples: int,
    halflife: float = 10,
    reverse: bool = False
) -> np.ndarray:
    """
    Generate exponential decay weights based on halflife.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate weights for
    halflife : float, default=10
        Number of periods for weight to decay to half
    reverse : bool, default=False
        If True, decay from oldest to newest (instead of newest to oldest)

    Returns
    -------
    np.ndarray
        Array of exponential decay weights

    Examples
    --------
    >>> weights = exponential_decay_weights(100, halflife=20)
    """
    positions = np.arange(n_samples)
    if not reverse:
        positions = positions[::-1]

    # Calculate decay constant from halflife
    decay_constant = np.log(2) / halflife

    # Calculate weights
    weights = np.exp(-decay_constant * positions)

    return weights


def linear_decay_weights(
    n_samples: int,
    min_weight: float = 0.1,
    reverse: bool = False
) -> np.ndarray:
    """
    Generate linear decay weights.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate weights for
    min_weight : float, default=0.1
        Minimum weight for the oldest observation
    reverse : bool, default=False
        If True, decay from oldest to newest

    Returns
    -------
    np.ndarray
        Array of linear decay weights

    Examples
    --------
    >>> weights = linear_decay_weights(100, min_weight=0.2)
    """
    if n_samples == 1:
        return np.array([1.0])

    # Generate linear weights
    weights = np.linspace(min_weight, 1.0, n_samples)

    if not reverse:
        weights = weights[::-1]

    return weights


def apply_time_decay_to_series(
    data: pd.Series,
    decay_rate: float = 0.98,
    time_unit: Literal['days', 'hours', 'months'] = 'days',
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None
) -> pd.Series:
    """
    Apply time decay weights directly to a pandas Series with datetime index.

    Parameters
    ----------
    data : pd.Series
        Series with datetime index
    decay_rate : float, default=0.98
        The decay rate (between 0 and 1)
    time_unit : str, default='days'
        Time unit for decay calculation
    reference_date : datetime-like, optional
        Reference date for decay calculation

    Returns
    -------
    pd.Series
        The original series multiplied by decay factors

    Examples
    --------
    >>> prices = pd.Series([100, 101, 102], index=pd.date_range('2024-01-01', periods=3))
    >>> decayed = apply_time_decay_to_series(prices, decay_rate=0.95)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("Series must have a DatetimeIndex")

    # Calculate decay factors
    decay_factors = calculate_time_decay(
        data.index,
        decay_rate=decay_rate,
        time_unit=time_unit,
        reference_date=reference_date
    )

    # Apply decay to data
    return data * decay_factors


def calculate_decay_statistics(
    decay_factors: Union[pd.Series, np.ndarray]
) -> dict:
    """
    Calculate statistics for decay factors.

    Parameters
    ----------
    decay_factors : pd.Series or np.ndarray
        Array of decay factors

    Returns
    -------
    dict
        Dictionary with decay statistics
    """
    return {
        'max': np.max(decay_factors),
        'min': np.min(decay_factors),
        'mean': np.mean(decay_factors),
        'median': np.median(decay_factors),
        'std': np.std(decay_factors),
        'effective_samples': np.sum(decay_factors),  # Effective number of samples
        'decay_ratio': np.min(decay_factors) / np.max(decay_factors) if np.max(decay_factors) > 0 else 0
    }


def adaptive_decay_rate(
    volatility: float,
    base_decay: float = 0.98,
    volatility_scaling: float = 0.1
) -> float:
    """
    Calculate adaptive decay rate based on market volatility.
    Higher volatility → faster decay (more weight on recent data).

    Parameters
    ----------
    volatility : float
        Current market volatility (e.g., standard deviation of returns)
    base_decay : float, default=0.98
        Base decay rate in normal conditions
    volatility_scaling : float, default=0.1
        Scaling factor for volatility adjustment

    Returns
    -------
    float
        Adjusted decay rate

    Examples
    --------
    >>> vol = 0.02  # 2% volatility
    >>> decay = adaptive_decay_rate(vol)
    """
    # Higher volatility → lower decay rate (faster decay)
    adjustment = volatility * volatility_scaling
    adjusted_decay = base_decay * (1 - adjustment)

    # Ensure decay rate stays in valid range
    return np.clip(adjusted_decay, 0.5, 0.999)


if __name__ == "__main__":
    # Example usage
    print("Time Decay Module - Example Usage\n")

    # Create sample timestamps
    dates = pd.date_range('2024-01-01', periods=30, freq='D')

    # Calculate decay factors
    decay_factors = calculate_time_decay(dates, decay_rate=0.95, time_unit='days')

    print("First 5 decay factors:")
    print(decay_factors.head())

    print("\nDecay statistics:")
    stats = calculate_decay_statistics(decay_factors)
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")

    # Example with exponential decay
    exp_weights = exponential_decay_weights(30, halflife=10)
    print(f"\nExponential decay weights (first 5): {exp_weights[:5]}")

    # Example with adaptive decay
    volatility = 0.03  # 3% volatility
    adaptive_rate = adaptive_decay_rate(volatility)
    print(f"\nAdaptive decay rate for {volatility:.1%} volatility: {adaptive_rate:.4f}")