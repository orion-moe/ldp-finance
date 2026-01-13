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
    Combinatorial Purged K-Fold (CPCV) from Chapter 12 of AFML.
    Generates multiple backtest paths for robust out-of-sample estimates.
BacktestPathReconstructor
    Reconstructs complete backtest paths from CPCV fold predictions.
PBOCalculator
    Calculates Probability of Backtest Overfitting (PBO).

Functions
---------
get_train_times
    Find training observations that don't overlap with test labels.
get_embargo_times
    Calculate embargo duration based on dataset size.
get_number_of_paths
    Calculate number of backtest paths per observation: C(N-1, k-1).
build_path_matrix
    Build the path matrix (phi) for CPCV.
calculate_pbo_from_cpcv
    High-level function to calculate PBO from a model using CPCV.

Reference
---------
Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
Chapter 7: Cross-Validation in Finance.
Chapter 12: Backtesting through Cross-Validation.
"""

from .purged_kfold import (
    # Classes
    PurgedKFold,
    CombinatorialPurgedKFold,
    BacktestPathReconstructor,
    PBOCalculator,
    # Functions
    get_train_times,
    get_embargo_times,
    get_number_of_paths,
    build_path_matrix,
    calculate_pbo_from_cpcv,
)

__all__ = [
    # Classes
    'PurgedKFold',
    'CombinatorialPurgedKFold',
    'BacktestPathReconstructor',
    'PBOCalculator',
    # Functions
    'get_train_times',
    'get_embargo_times',
    'get_number_of_paths',
    'build_path_matrix',
    'calculate_pbo_from_cpcv',
]
