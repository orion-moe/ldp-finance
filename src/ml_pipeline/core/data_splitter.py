"""
Data splitting module for train/validation/test sets.
"""

import logging
import pandas as pd
from typing import Tuple


def split_timeseries_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train, validation, and test sets.

    Args:
        df: Input DataFrame (assumed to be time-ordered)
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger = logging.getLogger(__name__)

    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total_ratio <= 1.01):  # Allow small floating point errors
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    n_samples = len(df)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # Split data
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(f"\n📊 Data Split:")
    logger.info(f"   Total samples: {n_samples:,}")
    logger.info(f"   Train: {len(train_df):,} ({len(train_df)/n_samples*100:.1f}%)")
    logger.info(f"   Validation: {len(val_df):,} ({len(val_df)/n_samples*100:.1f}%)")
    logger.info(f"   Test: {len(test_df):,} ({len(test_df)/n_samples*100:.1f}%)")

    return train_df, val_df, test_df
