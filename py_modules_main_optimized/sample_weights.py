"""
Sample Weights Functions for Financial Machine Learning

This module provides functions for calculating sample weights for ML models,
including uniqueness, return-based weights, and combined weighting schemes.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Union, Optional, Literal, Tuple


def calculate_uniqueness_weights(
    events: pd.DataFrame,
    start_col: Optional[str] = None,
    end_col: str = 'exit_time'
) -> pd.Series:
    """
    Calculate uniqueness weights based on overlapping events.
    Events that overlap with many others get lower weights.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame with events (index = start time)
    start_col : str, optional
        Column with start times. If None, uses index
    end_col : str, default='exit_time'
        Column with end times

    Returns
    -------
    pd.Series
        Uniqueness weights for each event

    Examples
    --------
    >>> events = pd.DataFrame({'exit_time': end_times}, index=start_times)
    >>> uniqueness = calculate_uniqueness_weights(events)
    """
    n_events = len(events)
    uniqueness = pd.Series(index=events.index, dtype=np.float64)

    # Get start and end times
    if start_col is None:
        start_times = events.index.values.astype(np.int64)
    else:
        start_times = pd.to_datetime(events[start_col]).values.astype(np.int64)

    end_times = pd.to_datetime(events[end_col]).values.astype(np.int64)

    # Calculate overlaps
    for i in range(n_events):
        # Count concurrent events
        concurrent = (
            (start_times <= end_times[i]) &
            (end_times >= start_times[i])
        ).sum()
        uniqueness.iloc[i] = 1.0 / max(concurrent, 1)

    return uniqueness


def calculate_return_weights(
    returns: pd.Series,
    weight_scheme: Literal['linear', 'sqrt', 'log', 'squared'] = 'sqrt'
) -> pd.Series:
    """
    Calculate weights based on return magnitudes.

    Parameters
    ----------
    returns : pd.Series
        Series of returns
    weight_scheme : str, default='sqrt'
        Weighting scheme:
        - 'linear': Use absolute returns directly
        - 'sqrt': Square root of absolute returns
        - 'log': Log of (1 + absolute returns)
        - 'squared': Squared returns (emphasize large moves)

    Returns
    -------
    pd.Series
        Return-based weights

    Examples
    --------
    >>> returns = pd.Series([0.01, -0.02, 0.03])
    >>> weights = calculate_return_weights(returns, weight_scheme='sqrt')
    """
    abs_returns = returns.abs()

    if weight_scheme == 'linear':
        weights = abs_returns
    elif weight_scheme == 'sqrt':
        weights = np.sqrt(abs_returns)
    elif weight_scheme == 'log':
        weights = np.log1p(abs_returns)
    elif weight_scheme == 'squared':
        weights = abs_returns ** 2
    else:
        raise ValueError(f"Invalid weight_scheme: {weight_scheme}")

    # Handle any NaN or inf values
    weights = weights.replace([np.inf, -np.inf], 0)
    weights = weights.fillna(0)

    return weights


def calculate_sample_weights(
    events: pd.DataFrame,
    return_col: str = 'retorno',
    return_weight: float = 0.5,
    time_decay: float = 0.98,
    time_unit: Literal['days', 'hours', 'months'] = 'days',
    reference_date: Optional[Union[str, datetime, pd.Timestamp]] = None,
    normalization: Literal['max', 'mean', 'sum'] = 'max',
    include_uniqueness: bool = True,
    weight_scheme: Literal['linear', 'sqrt', 'log', 'squared'] = 'sqrt'
) -> Tuple[pd.Series, dict]:
    """
    Calculate combined sample weights for meta-labeling.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame with events and returns
    return_col : str, default='retorno'
        Column containing returns
    return_weight : float, default=0.5
        Weight for return component (0 to 1)
    time_decay : float, default=0.98
        Time decay factor
    time_unit : str, default='days'
        Time unit for decay calculation
    reference_date : datetime-like, optional
        Reference date for decay
    normalization : str, default='max'
        Normalization method: 'max', 'mean', or 'sum'
    include_uniqueness : bool, default=True
        Whether to include uniqueness weights
    weight_scheme : str, default='sqrt'
        Return weighting scheme

    Returns
    -------
    pd.Series
        Final sample weights
    dict
        Information about weight calculation

    Examples
    --------
    >>> events = pd.DataFrame({'retorno': returns, 'exit_time': exit_times}, index=start_times)
    >>> weights, info = calculate_sample_weights(events)
    """
    from time_decay import calculate_time_decay

    # 1. UNIQUENESS WEIGHTS
    if include_uniqueness and 'exit_time' in events.columns:
        uniqueness = calculate_uniqueness_weights(events)
    else:
        uniqueness = pd.Series(1.0, index=events.index)

    # 2. RETURN WEIGHTS
    if return_col in events.columns:
        return_weights = calculate_return_weights(
            events[return_col],
            weight_scheme=weight_scheme
        )
    else:
        return_weights = pd.Series(1.0, index=events.index)

    # 3. TIME DECAY
    if time_decay < 1.0:
        decay_factors = calculate_time_decay(
            events.index,
            decay_rate=time_decay,
            time_unit=time_unit,
            reference_date=reference_date
        )
    else:
        decay_factors = pd.Series(1.0, index=events.index)

    # 4. COMBINE WEIGHTS
    # Normalize components
    if uniqueness.sum() > 0:
        u_norm = uniqueness / uniqueness.sum()
    else:
        u_norm = uniqueness

    if return_weights.sum() > 0:
        r_norm = return_weights / return_weights.sum()
    else:
        r_norm = return_weights

    # Weighted combination
    if include_uniqueness:
        combined = (
            (1 - return_weight) * u_norm +
            return_weight * r_norm
        ) * decay_factors
    else:
        combined = r_norm * decay_factors

    # 5. FINAL NORMALIZATION
    if normalization == 'max' and combined.max() > 0:
        final_weights = combined / combined.max()
    elif normalization == 'mean' and combined.mean() > 0:
        final_weights = combined / combined.mean()
    elif normalization == 'sum' and combined.sum() > 0:
        final_weights = combined / combined.sum()
    else:
        final_weights = combined

    # 6. CALCULATE METRICS
    info = {
        'n_samples': len(final_weights),
        'max': float(final_weights.max()),
        'mean': float(final_weights.mean()),
        'min': float(final_weights.min()),
        'std': float(final_weights.std()),
        'effective_samples': float(final_weights.sum()),
        'uniqueness_mean': float(uniqueness.mean()),
        'return_weight': return_weight,
        'time_decay': time_decay,
        'time_unit': time_unit,
        'normalization': normalization,
        'weight_scheme': weight_scheme
    }

    if reference_date is not None:
        info['reference_date'] = pd.to_datetime(reference_date).strftime('%Y-%m-%d')

    return final_weights, info


def calculate_inverse_volatility_weights(
    returns: pd.DataFrame,
    lookback: int = 20,
    min_periods: int = 10
) -> pd.Series:
    """
    Calculate weights inversely proportional to volatility.

    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame of returns (columns = assets)
    lookback : int, default=20
        Lookback period for volatility calculation
    min_periods : int, default=10
        Minimum periods required for calculation

    Returns
    -------
    pd.Series
        Inverse volatility weights for each asset

    Examples
    --------
    >>> returns = pd.DataFrame({'A': [...], 'B': [...]})
    >>> weights = calculate_inverse_volatility_weights(returns)
    """
    # Calculate rolling volatility
    volatility = returns.rolling(window=lookback, min_periods=min_periods).std()

    # Take the last available volatility
    last_vol = volatility.iloc[-1]

    # Calculate inverse volatility weights
    inv_vol = 1 / last_vol
    weights = inv_vol / inv_vol.sum()

    return weights


def calculate_classification_weights(
    labels: pd.Series,
    balance_method: Literal['inverse', 'sqrt', 'log'] = 'inverse'
) -> pd.Series:
    """
    Calculate weights for imbalanced classification problems.

    Parameters
    ----------
    labels : pd.Series
        Classification labels
    balance_method : str, default='inverse'
        Method for balancing:
        - 'inverse': Inverse frequency
        - 'sqrt': Square root of inverse frequency
        - 'log': Log of inverse frequency

    Returns
    -------
    pd.Series
        Weights for each sample

    Examples
    --------
    >>> labels = pd.Series([0, 0, 0, 1, 1, 2])
    >>> weights = calculate_classification_weights(labels)
    """
    # Calculate class frequencies
    class_counts = labels.value_counts()
    total_count = len(labels)

    # Calculate class weights
    if balance_method == 'inverse':
        class_weights = total_count / (len(class_counts) * class_counts)
    elif balance_method == 'sqrt':
        class_weights = np.sqrt(total_count / (len(class_counts) * class_counts))
    elif balance_method == 'log':
        class_weights = np.log1p(total_count / (len(class_counts) * class_counts))
    else:
        raise ValueError(f"Invalid balance_method: {balance_method}")

    # Map to sample weights
    sample_weights = labels.map(class_weights)

    return sample_weights


def validate_weights(
    weights: pd.Series,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    check_sum: Optional[float] = None
) -> Tuple[bool, dict]:
    """
    Validate sample weights.

    Parameters
    ----------
    weights : pd.Series
        Weights to validate
    min_weight : float, default=0.0
        Minimum allowed weight
    max_weight : float, default=1.0
        Maximum allowed weight
    check_sum : float, optional
        If provided, checks if weights sum to this value

    Returns
    -------
    bool
        Whether weights are valid
    dict
        Validation results and statistics

    Examples
    --------
    >>> weights = pd.Series([0.1, 0.2, 0.3, 0.4])
    >>> is_valid, info = validate_weights(weights, check_sum=1.0)
    """
    results = {
        'valid': True,
        'n_samples': len(weights),
        'min': float(weights.min()),
        'max': float(weights.max()),
        'sum': float(weights.sum()),
        'mean': float(weights.mean()),
        'std': float(weights.std()),
        'n_zero': int((weights == 0).sum()),
        'n_nan': int(weights.isna().sum())
    }

    # Check for NaN
    if results['n_nan'] > 0:
        results['valid'] = False
        results['error'] = f"Found {results['n_nan']} NaN values"

    # Check range
    elif results['min'] < min_weight:
        results['valid'] = False
        results['error'] = f"Minimum weight {results['min']:.4f} < {min_weight}"

    elif results['max'] > max_weight:
        results['valid'] = False
        results['error'] = f"Maximum weight {results['max']:.4f} > {max_weight}"

    # Check sum if specified
    elif check_sum is not None:
        tolerance = 1e-6
        if abs(results['sum'] - check_sum) > tolerance:
            results['valid'] = False
            results['error'] = f"Sum {results['sum']:.4f} != {check_sum}"

    return results['valid'], results


if __name__ == "__main__":
    # Example usage
    print("Sample Weights Module - Example Usage\n")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    returns = pd.Series(np.random.randn(100) * 0.02, index=dates)

    # Create events DataFrame
    events = pd.DataFrame({
        'retorno': returns,
        'exit_time': dates + pd.Timedelta(days=5)
    }, index=dates)

    # Calculate sample weights
    weights, info = calculate_sample_weights(
        events,
        return_col='retorno',
        return_weight=0.5,
        time_decay=0.95,
        time_unit='days'
    )

    print("Sample Weights Statistics:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    # Validate weights
    is_valid, validation_info = validate_weights(weights)
    print(f"\nWeights valid: {is_valid}")
    print("Validation info:")
    for key, value in validation_info.items():
        if key != 'valid' and isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")