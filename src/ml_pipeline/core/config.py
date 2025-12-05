"""
Configuration module for main_optimized pipeline.
Centralizes all hyperparameters and settings.
"""

from dataclasses import dataclass
from typing import Dict, Any
import os


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""

    # Data paths
    data_path: str = 'data/btcusdt-futures-um/output/standard'
    base_output_path: str = 'data/btcusdt-futures-um/output/standard'
    file_name: str = '20251123-003308-standard-futures-volume200000000.parquet'

    # Data splitting ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Fractional differentiation
    frac_diff_d: float = 0.4
    frac_diff_threshold: float = 1e-5

    # AR Model
    ar_max_lag: int = 5
    ar_vif_threshold: float = 10.0

    # CUSUM event detection
    cusum_threshold: float = 0.02

    # Triple Barrier Method
    barrier_num_days: int = 5
    barrier_pt_sl: list = None  # Will be [1, 1] by default
    barrier_num_threads: int = 4

    # Feature engineering
    entropy_window: int = 100
    microstructure_windows: list = None  # Will be [10, 20, 50] by default

    # Sample weights
    sample_weights_decay: float = 1.0

    # Random Forest hyperparameters
    rf_n_estimators_grid: list = None  # [100, 200]
    rf_max_depth_grid: list = None  # [10, 20, None]
    rf_min_samples_split_grid: list = None  # [2, 5]
    rf_min_samples_leaf_grid: list = None  # [1, 2]
    rf_max_features_grid: list = None  # ['sqrt', 'log2']
    rf_random_state: int = 42
    rf_n_jobs: int = -1

    # GridSearch CV
    cv_n_splits: int = 5
    cv_scoring: str = 'f1'
    cv_verbose: int = 1

    # Visualization
    viz_dpi: int = 300
    viz_style: str = 'seaborn-v0_8-darkgrid'
    viz_top_n_features: int = 20

    # Logging
    log_level: str = 'INFO'
    log_max_bytes: int = 50 * 1024 * 1024  # 50MB
    log_backup_count: int = 10

    def __post_init__(self):
        """Initialize default lists after dataclass creation."""
        if self.barrier_pt_sl is None:
            self.barrier_pt_sl = [1, 1]

        if self.microstructure_windows is None:
            self.microstructure_windows = [10, 20, 50]

        if self.rf_n_estimators_grid is None:
            self.rf_n_estimators_grid = [100, 200]

        if self.rf_max_depth_grid is None:
            self.rf_max_depth_grid = [10, 20, None]

        if self.rf_min_samples_split_grid is None:
            self.rf_min_samples_split_grid = [2, 5]

        if self.rf_min_samples_leaf_grid is None:
            self.rf_min_samples_leaf_grid = [1, 2]

        if self.rf_max_features_grid is None:
            self.rf_max_features_grid = ['sqrt', 'log2']

    def get_rf_param_grid(self, reduced: bool = True) -> Dict[str, Any]:
        """
        Get Random Forest parameter grid for GridSearchCV.

        Args:
            reduced: If True, use reduced grid for faster search

        Returns:
            Parameter grid dictionary
        """
        if reduced:
            return {
                'n_estimators': [200],
                'max_depth': [20],
                'min_samples_split': [5],
                'min_samples_leaf': [2],
                'max_features': ['sqrt'],
                'bootstrap': [True],
                'class_weight': ['balanced']
            }
        else:
            return {
                'n_estimators': self.rf_n_estimators_grid,
                'max_depth': self.rf_max_depth_grid,
                'min_samples_split': self.rf_min_samples_split_grid,
                'min_samples_leaf': self.rf_min_samples_leaf_grid,
                'max_features': self.rf_max_features_grid,
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample']
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'data_path': self.data_path,
            'base_output_path': self.base_output_path,
            'file_name': self.file_name,
            'train_ratio': self.train_ratio,
            'val_ratio': self.val_ratio,
            'test_ratio': self.test_ratio,
            'frac_diff_d': self.frac_diff_d,
            'cusum_threshold': self.cusum_threshold,
            'barrier_num_days': self.barrier_num_days,
            'cv_n_splits': self.cv_n_splits,
            'cv_scoring': self.cv_scoring,
        }


# Default configuration instance
DEFAULT_CONFIG = PipelineConfig()
