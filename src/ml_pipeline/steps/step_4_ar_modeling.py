"""
Autoregressive (AR) Modeling module with multicollinearity treatment.
Wraps ImprovedAutoRegressiveModel for pipeline integration.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from py_modules_main_optimized.improved_ar_model import ImprovedAutoRegressiveModel
from py_modules_main_optimized.ar_multicollinearity_solutions import AutoRegressiveModel


class ARModelWrapper:
    """
    Wrapper for AR modeling with automatic order selection and
    multicollinearity treatment.
    """

    def __init__(
        self,
        p_max_limit: int = 200,
        criterion: str = 'other',
        threshold: float = 0.01,
        pvalue_threshold: float = 0.05,
        min_reduction: float = 0.01,
        treatment_method: str = 'auto'
    ):
        """
        Initialize AR model wrapper.

        Args:
            p_max_limit: Maximum AR order limit (default: 200)
            criterion: Criterion for order selection (default: 'other')
            threshold: Threshold for multicollinearity detection
            pvalue_threshold: P-value threshold for coefficients
            min_reduction: Minimum AIC reduction threshold
            treatment_method: Method for multicollinearity treatment ('auto', 'ridge', 'lasso', etc.)
        """
        self.p_max_limit = p_max_limit
        self.criterion = criterion
        self.threshold = threshold
        self.pvalue_threshold = pvalue_threshold
        self.min_reduction = min_reduction
        self.treatment_method = treatment_method

        self.logger = logging.getLogger(__name__)
        self.model = None
        self.ar_results = None
        self.p_optimal = None

    def _calculate_p_max(self, series_length: int) -> int:
        """
        Calculate p_max based on series length with progressive limits.

        Args:
            series_length: Length of time series

        Returns:
            Calculated p_max value
        """
        p_max_calculated = round(series_length ** (1/2))

        # Progressive limits based on data size
        if p_max_calculated > 200:
            p_max = 50  # Very large datasets: conservative limit
            self.logger.warning(
                f"⚠️ Very large dataset detected (p_max would be {p_max_calculated})"
            )
            self.logger.info(f"Using conservative limit: p_max = {p_max}")
        elif p_max_calculated > 100:
            p_max = 75  # Large datasets: moderate limit
            self.logger.info(f"Large dataset: using p_max = {p_max} (calculated: {p_max_calculated})")
        else:
            p_max = p_max_calculated  # Small datasets: use calculated value
            self.logger.info(f"Using calculated p_max = {p_max}")

        # Apply absolute limit
        p_max = min(p_max, self.p_max_limit)

        return p_max

    def select_order(
        self,
        series: pd.Series,
        p_max: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select optimal AR order using AIC/BIC criteria.

        Args:
            series: Input time series (fractionally differentiated)
            p_max: Maximum order to test (auto-calculated if None)

        Returns:
            Tuple of (optimal_order, metrics)
        """
        if p_max is None:
            p_max = self._calculate_p_max(len(series))
        else:
            self.logger.info(f"Using provided p_max = {p_max}")

        self.logger.info(f"\n🔍 Selecting AR order (p_max={p_max})...")

        # Select order
        p_optimal, metrics = AutoRegressiveModel.select_ar_order(
            series,
            p_max=p_max,
            criterio=self.criterion,
            limiar=self.threshold,
            limiar_pvalor=self.pvalue_threshold,
            min_reducao_absoluta=self.min_reduction
        )

        self.p_optimal = p_optimal
        self.logger.info(f"✅ Optimal order selected: p = {p_optimal}")

        return p_optimal, metrics

    def fit(
        self,
        series: pd.Series,
        p: Optional[int] = None,
        treatment_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fit AR model with multicollinearity treatment.

        Args:
            series: Input time series
            p: AR order (uses self.p_optimal if None)
            treatment_method: Treatment method (uses self.treatment_method if None)

        Returns:
            Dictionary with model results
        """
        if p is None:
            if self.p_optimal is None:
                raise ValueError("AR order not selected. Run select_order() first.")
            p = self.p_optimal

        if treatment_method is None:
            treatment_method = self.treatment_method

        self.logger.info(f"\n🔧 Fitting AR({p}) model with {treatment_method} treatment...")

        # Initialize improved AR model
        self.model = ImprovedAutoRegressiveModel(series)

        # Fit with automatic multicollinearity treatment
        ar_results = self.model.fit_with_multicollinearity_treatment(
            p, treatment_method=treatment_method
        )

        self.ar_results = ar_results

        # Log results
        self._log_fit_results(ar_results)

        return ar_results

    def _log_fit_results(self, ar_results: Dict[str, Any]):
        """Log AR model fitting results."""
        self.logger.info(f"\n✅ Model fitted: {ar_results['method'].upper()}")
        self.logger.info(f"✅ Final order: AR({ar_results['p']})")
        self.logger.info(
            f"✅ In-sample metrics: "
            f"MAE={ar_results['metrics']['mae']:.6f}, "
            f"RMSE={ar_results['metrics']['rmse']:.6f}, "
            f"R²={ar_results['metrics']['r2']:.4f}"
        )

    def predict_out_of_sample(
        self,
        series: pd.Series,
        ar_results: Optional[Dict[str, Any]] = None
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Make out-of-sample predictions using Monte Carlo simulation.

        Args:
            series: Out-of-sample series
            ar_results: AR model results (uses self.ar_results if None)

        Returns:
            Tuple of (predictions, residuals)
        """
        if ar_results is None:
            if self.ar_results is None:
                raise ValueError("Model not fitted. Run fit() first.")
            ar_results = self.ar_results

        if self.model is None:
            raise ValueError("Model not initialized. Run fit() first.")

        # Monte Carlo predictions
        predictions_df = self.model.monteCarlo_improved(series, ar_results)
        predictions = predictions_df['fraqdiff_pred']
        residuals = series - predictions

        return predictions, residuals

    def run_full_modeling(
        self,
        train_series: pd.Series,
        val_series: pd.Series,
        test_series: pd.Series,
        p_max: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run complete AR modeling pipeline.

        Args:
            train_series: Training series (fractionally differentiated)
            val_series: Validation series
            test_series: Test series
            p_max: Maximum AR order

        Returns:
            Dictionary with all results
        """
        from py_modules_main_optimized.utils import log_memory

        self.logger.info("\n🔍 AR MODEL WITH MULTICOLLINEARITY TREATMENT")
        self.logger.info("="*65)

        # Step 1: Select optimal order
        p_optimal, selection_metrics = self.select_order(train_series, p_max)

        # Step 2: Fit model
        ar_results = self.fit(train_series, p_optimal)

        # Step 3: Out-of-sample predictions
        self.logger.info("\n📊 Generating out-of-sample predictions...")

        val_pred, val_residuals = self.predict_out_of_sample(val_series, ar_results)
        test_pred, test_residuals = self.predict_out_of_sample(test_series, ar_results)

        log_memory("AR model optimization")

        # Compile results
        results = {
            'p_optimal': p_optimal,
            'ar_results': ar_results,
            'train': {
                'predictions': ar_results['y_pred'],
                'residuals': ar_results['residuals'],
                'metrics': ar_results['metrics']
            },
            'validation': {
                'predictions': val_pred,
                'residuals': val_residuals
            },
            'test': {
                'predictions': test_pred,
                'residuals': test_residuals
            }
        }

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of AR modeling results.

        Returns:
            Dictionary with model summary
        """
        if self.ar_results is None:
            return {
                'fitted': False,
                'configuration': {
                    'p_max_limit': self.p_max_limit,
                    'criterion': self.criterion,
                    'treatment_method': self.treatment_method
                }
            }

        return {
            'fitted': True,
            'p_optimal': self.p_optimal,
            'method': self.ar_results['method'],
            'final_order': self.ar_results['p'],
            'metrics': self.ar_results['metrics'],
            'configuration': {
                'p_max_limit': self.p_max_limit,
                'criterion': self.criterion,
                'treatment_method': self.treatment_method
            }
        }
