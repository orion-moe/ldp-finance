"""
Fractional Differentiation module with two-stage optimization.
Implements López de Prado's fractional differentiation for stationarity
while preserving memory.
"""

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from py_modules_main_optimized.ml_framework import FractionalDifferentiation, StationarityTester


class FractionalDifferentiationOptimizer:
    """
    Optimizes fractional differentiation parameter d using two-stage search.

    Stage 1: Coarse search over range [0.1, 1.0] with step 0.1
    Stage 2: Fine search around optimal coarse d with step 0.02

    Finds minimum d that achieves stationarity (ADF test).
    """

    def __init__(
        self,
        conf: float = 0.05,
        thresh: float = 1e-3,
        max_len_adf: int = 100_000,
        coarse_step: float = 0.1,
        fine_step: float = 0.02
    ):
        """
        Initialize optimizer.

        Args:
            conf: Confidence level for ADF test (default: 0.05)
            thresh: Threshold for differentiation (default: 1e-3)
            max_len_adf: Maximum series length for ADF test (default: 100,000)
            coarse_step: Step size for coarse search (default: 0.1)
            fine_step: Step size for fine search (default: 0.02)
        """
        self.conf = conf
        self.thresh = thresh
        self.max_len_adf = max_len_adf
        self.coarse_step = coarse_step
        self.fine_step = fine_step
        self.logger = logging.getLogger(__name__)

        self.d_optimal = None
        self.adf_results = {}

    def _test_stationarity(
        self,
        series: pd.Series,
        d: float,
        stage: str = "coarse"
    ) -> bool:
        """
        Test if series is stationary after fractional differentiation.

        Args:
            series: Input series
            d: Fractional differentiation parameter
            stage: "coarse" or "fine" (for logging)

        Returns:
            True if stationary, False otherwise
        """
        self.logger.debug(f'Testing d={d} ({stage})')

        # Apply fractional differentiation
        series_diff = FractionalDifferentiation.frac_diff_optimized(
            np.array(series), d
        ).dropna().reset_index(drop=True)

        # Limit size for ADF test
        if len(series_diff) > self.max_len_adf:
            series_diff_reduced = series_diff[-self.max_len_adf:]
        else:
            series_diff_reduced = series_diff

        # ADF test
        try:
            adf_result = StationarityTester.adf_test(
                series_diff_reduced,
                title=f'Fractional Differentiation d={d}'
            )

            critical_value = adf_result['critical_values'][f"{self.conf * 100:.0f}%"]
            adf_statistic = adf_result['statistic']
            p_value = adf_result['pvalue']

            # Log results
            self.logger.debug(
                f'd={d}: ADF stat={adf_statistic:.4f}, '
                f'p-value={p_value:.4f}, critical={critical_value:.4f}'
            )

            # Check stationarity
            is_stationary = (p_value < self.conf) and (adf_statistic < critical_value)

            # Store results
            self.adf_results[d] = {
                'statistic': adf_statistic,
                'pvalue': p_value,
                'critical_value': critical_value,
                'is_stationary': is_stationary
            }

            if is_stationary:
                self.logger.debug(f'd={d}: Stationary ✓')
            else:
                self.logger.debug(f'd={d}: Not stationary (p-value={p_value:.4f})')

            return is_stationary

        except Exception as e:
            self.logger.warning(f'Error testing d={d}: {e}')
            return False

    def _coarse_search(self, series: pd.Series) -> Optional[float]:
        """
        Stage 1: Coarse search for optimal d.

        Args:
            series: Input series

        Returns:
            Optimal d from coarse search, or None if not found
        """
        self.logger.info("Stage 1: Coarse search...")

        d_values_coarse = np.round(
            np.arange(0.1, 1.01, self.coarse_step), 2
        )

        self.logger.info(f"Coarse search range: {d_values_coarse.tolist()}")

        for d in d_values_coarse:
            if self._test_stationarity(series, d, stage="coarse"):
                self.logger.info(f'✅ Coarse search found: d = {d}')
                return d

        return None

    def _fine_search(
        self,
        series: pd.Series,
        d_coarse: float
    ) -> Optional[float]:
        """
        Stage 2: Fine search around coarse d.

        Args:
            series: Input series
            d_coarse: Result from coarse search

        Returns:
            Optimal d from fine search, or None if not found
        """
        self.logger.info(f"\nStage 2: Fine search around d={d_coarse}")

        # Define fine search range
        if d_coarse > 0.1:
            d_min = max(0.01, d_coarse - 0.1)
            d_max = d_coarse
        else:
            d_min = 0.01
            d_max = d_coarse

        d_values_fine = np.round(
            np.arange(d_min, d_max + 0.01, self.fine_step), 3
        )

        self.logger.info(
            f"Fine search range: {d_min:.2f} to {d_max:.2f} "
            f"(step={self.fine_step})"
        )

        for d in d_values_fine:
            if self._test_stationarity(series, d, stage="fine"):
                adf_info = self.adf_results[d]
                self.logger.info(f'✅ OPTIMAL d FOUND (refined): d = {d:.3f}')
                self.logger.info(f'   ADF statistic: {adf_info["statistic"]:.4f}')
                self.logger.info(f'   p-value: {adf_info["pvalue"]:.6f}')
                self.logger.info(f'   Critical value (5%): {adf_info["critical_value"]:.4f}')
                self.logger.info(f'   Improvement from coarse: {d_coarse - d:.3f}')
                return d

        return None

    def find_optimal_d(
        self,
        series: pd.Series,
        column_name: str = 'log_close'
    ) -> float:
        """
        Find optimal d using two-stage search.

        Args:
            series: Input series to test
            column_name: Name of column (for logging)

        Returns:
            Optimal d value
        """
        self.logger.info("\n🔍 Two-stage search for optimal d:")
        self.logger.info(f"Confidence level: {self.conf}, Threshold: {self.thresh}")

        # Stage 1: Coarse search
        d_coarse = self._coarse_search(series)

        if d_coarse is None:
            self.logger.error("❌ No suitable d found in coarse search!")
            self.logger.info("Using d=1.0 as fallback (full differentiation)")
            self.d_optimal = 1.0
            return self.d_optimal

        # Stage 2: Fine search
        d_fine = self._fine_search(series, d_coarse)

        if d_fine is not None:
            self.d_optimal = d_fine
        else:
            self.logger.info(f"Using coarse result: d = {d_coarse}")
            self.d_optimal = d_coarse

        return self.d_optimal

    def apply_to_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        d: float,
        column: str = 'log_close',
        output_column: str = 'fraq_close'
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Apply fractional differentiation to all datasets.

        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            d: Fractional differentiation parameter
            column: Column to differentiate
            output_column: Name for output column

        Returns:
            Tuple of (train_diff, val_diff, test_diff) series
        """
        self.logger.info(f"\nApplying fractional differentiation (d={d:.3f})...")

        # Train
        series_train = np.array(train_df[column])
        train_diff = FractionalDifferentiation.frac_diff_optimized(
            series_train, d
        ).dropna().reset_index(drop=True)

        # Validation
        series_val = np.array(val_df[column])
        val_diff = FractionalDifferentiation.frac_diff_optimized(
            series_val, d
        ).dropna().reset_index(drop=True)

        # Test
        series_test = np.array(test_df[column])
        test_diff = FractionalDifferentiation.frac_diff_optimized(
            series_test, d
        ).dropna().reset_index(drop=True)

        self.logger.info(f"✅ Fractional differentiation applied:")
        self.logger.info(f"   Train: {len(train_diff):,} samples")
        self.logger.info(f"   Val: {len(val_diff):,} samples")
        self.logger.info(f"   Test: {len(test_diff):,} samples")

        return train_diff, val_diff, test_diff

    def add_to_datasets(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_diff: pd.Series,
        val_diff: pd.Series,
        test_diff: pd.Series,
        column_name: str = 'fraq_close'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Add fractional differentiation columns to datasets.

        Args:
            train_df: Training dataset
            val_df: Validation dataset
            test_df: Test dataset
            train_diff: Differentiated training series
            val_diff: Differentiated validation series
            test_diff: Differentiated test series
            column_name: Column name to add

        Returns:
            Tuple of (train_df, val_df, test_df) with added column
        """
        from py_modules_main_optimized.utils import log_memory

        # Make copies to avoid modifying originals
        train_result = train_df.copy()
        val_result = val_df.copy()
        test_result = test_df.copy()

        # Align and add fraq_close column
        # Use helper function from main_optimized.py
        def add_feature(dataset, series_fraq, column):
            """Add fractional differentiated feature to dataset."""
            min_len = min(len(dataset), len(series_fraq))
            dataset_aligned = dataset.iloc[:min_len].copy()
            dataset_aligned[column] = series_fraq.iloc[:min_len].values
            return dataset_aligned

        train_result = add_feature(train_result, train_diff, column_name)
        val_result = add_feature(val_result, val_diff, column_name)
        test_result = add_feature(test_result, test_diff, column_name)

        log_memory("fractional differentiation")

        return train_result, val_result, test_result

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of optimization results.

        Returns:
            Dictionary with optimization summary
        """
        return {
            'd_optimal': self.d_optimal,
            'conf_level': self.conf,
            'adf_results': self.adf_results.get(self.d_optimal, {}) if self.d_optimal else {},
            'search_parameters': {
                'coarse_step': self.coarse_step,
                'fine_step': self.fine_step,
                'max_len_adf': self.max_len_adf
            }
        }
