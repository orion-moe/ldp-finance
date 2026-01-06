"""
Cross-Validation for Financial Time Series
==========================================

This module provides specialized cross-validation strategies for financial
machine learning that prevent look-ahead bias.

Classes
-------
PurgedKFold
    K-Fold cross-validation with purging and embargo for financial time series.
CombinatorialPurgedKFold
    Combinatorial K-Fold with purging for more robust out-of-sample estimates.

Functions
---------
get_train_times
    Find training observations that don't overlap with test labels.
get_embargo_times
    Calculate embargo duration based on dataset size.

Reference
---------
López de Prado, M. (2018). Advances in Financial Machine Learning.
Chapter 7: Cross-Validation in Finance.
"""

from .purged_kfold import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    get_train_times,
    get_embargo_times,
)

__all__ = [
    'PurgedKFold',
    'CombinatorialPurgedKFold',
    'get_train_times',
    'get_embargo_times',
]
