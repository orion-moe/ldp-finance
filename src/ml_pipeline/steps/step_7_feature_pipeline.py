"""
Feature Engineering Pipeline module.
Orchestrates calculation of microstructure and entropy features.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from py_modules_main_optimized.unified_microstructure_features import UnifiedMicrostructureFeatures
from py_modules_main_optimized.entropy_features import EntropyFeatures


class FeatureEngineeringPipeline:
    """
    Orchestrates feature engineering pipeline.

    Combines:
    - Microstructure features (Corwin-Schultz, VPIN, OIR, etc.)
    - Entropy features (multiple windows and bins)
    """

    def __init__(
        self,
        microstructure_windows: List[int] = None,
        corwin_schultz_range: Tuple[int, int, int] = (10, 200, 10),
        entropy_windows: List[int] = None,
        entropy_bins: List[int] = None,
        skip_stationarity: bool = True
    ):
        """
        Initialize feature pipeline.

        Args:
            microstructure_windows: Windows for microstructure features (default: [50, 100, ..., 450])
            corwin_schultz_range: (start, stop, step) for Corwin-Schultz spread estimator
            entropy_windows: Windows for entropy calculation (default: [50, 100, ..., 450])
            entropy_bins: Bin sizes for entropy calculation (default: [25, 50, ..., 225])
            skip_stationarity: Skip stationarity tests for speed
        """
        self.microstructure_windows = microstructure_windows or list(range(50, 500, 50))
        self.corwin_schultz_range = corwin_schultz_range
        self.entropy_windows = entropy_windows or list(range(50, 500, 50))
        self.entropy_bins = entropy_bins or list(range(25, 250, 25))
        self.skip_stationarity = skip_stationarity

        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.entropy_feature_names = []
        self.microstructure_feature_names = []

    def calculate_microstructure_features(
        self,
        dataset: pd.DataFrame,
        windows: Optional[List[int]] = None,
        sl_range: Optional[Tuple[int, int, int]] = None,
        skip_stationarity: Optional[bool] = None
    ) -> pd.DataFrame:
        """
        Calculate all microstructure features using UnifiedMicrostructureFeatures.

        Args:
            dataset: Input dataset
            windows: Windows for base features
            sl_range: Range for Corwin-Schultz spread estimator
            skip_stationarity: Skip stationarity tests

        Returns:
            Dataset with microstructure features added
        """
        if windows is None:
            windows = self.microstructure_windows

        if sl_range is None:
            sl_range = self.corwin_schultz_range

        if skip_stationarity is None:
            skip_stationarity = self.skip_stationarity

        self.logger.info("🔄 Calculating microstructure features...")
        self.logger.info("   ✅ Corwin-Schultz spread estimator")
        self.logger.info("   ✅ VPIN using actual buy/sell volumes")
        self.logger.info("   ✅ OIR (Order Imbalance Ratio)")

        # Calculate features
        calculator = UnifiedMicrostructureFeatures()
        dataset_with_features = calculator.calculate_all_features(
            dataset,
            windows=windows,
            sl_range=sl_range,
            skip_stationarity=skip_stationarity
        )

        # Log results
        self._log_microstructure_results(dataset_with_features, calculator)

        self.microstructure_feature_names = calculator.feature_names

        return dataset_with_features

    def _log_microstructure_results(
        self,
        dataset: pd.DataFrame,
        calculator: UnifiedMicrostructureFeatures
    ):
        """Log microstructure feature calculation results."""
        self.logger.info("\n📊 Microstructure features calculated:")

        # Corwin-Schultz spread
        spread_cols = [col for col in dataset.columns if 'corwin_schultz' in col]
        self.logger.info(f"   - Corwin-Schultz spread: {len(spread_cols)}")

        # VPIN
        vpin_cols = [col for col in dataset.columns if 'vpin_fixed' in col]
        self.logger.info(f"   - VPIN fixed: {len(vpin_cols)}")

        # OIR
        oir_cols = [col for col in dataset.columns if 'oir' in col]
        self.logger.info(f"   - OIR: {len(oir_cols)}")

        # Verify Corwin-Schultz values
        if spread_cols:
            self.logger.info("\n📊 Corwin-Schultz spread verification (first 3):")
            for col in spread_cols[:3]:
                valid_data = dataset[col].dropna()
                if len(valid_data) > 0:
                    zeros = (valid_data == 0).sum()
                    self.logger.info(
                        f"   {col}: mean={valid_data.mean():.6f}, "
                        f"zeros={zeros/len(valid_data)*100:.1f}%"
                    )

        self.logger.info(f"\n✅ Total microstructure features: {len(calculator.feature_names)}")

    def calculate_entropy_features(
        self,
        series_frac_diff: pd.Series,
        window_sizes: Optional[List[int]] = None,
        bins_list: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Calculate entropy features in batch mode.

        Args:
            series_frac_diff: Fractionally differentiated series
            window_sizes: Window sizes for entropy calculation
            bins_list: Bin sizes for entropy calculation

        Returns:
            DataFrame with entropy features
        """
        if window_sizes is None:
            window_sizes = self.entropy_windows

        if bins_list is None:
            bins_list = self.entropy_bins

        self.logger.info("\n📊 Calculating entropy features...")
        self.logger.info(f"   Windows: {len(window_sizes)}")
        self.logger.info(f"   Bins: {len(bins_list)}")

        # Calculate entropy features
        entropy_results = EntropyFeatures.calculate_entropy_features_batch(
            series_frac_diff, window_sizes, bins_list
        )

        self.logger.info(
            f"✅ Entropy features: {len(entropy_results.columns)} combinations"
        )

        self.entropy_feature_names = list(entropy_results.columns)

        return entropy_results

    def add_entropy_to_dataset(
        self,
        dataset: pd.DataFrame,
        entropy_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add entropy features to dataset and move NaNs to front.

        Args:
            dataset: Base dataset
            entropy_results: Calculated entropy features

        Returns:
            Dataset with entropy features added
        """
        dataset_with_entropy = dataset.copy()

        # Add entropy features
        for col in entropy_results.columns:
            dataset_with_entropy[col] = entropy_results[col]

        # Move NaNs to front
        for col in entropy_results.columns:
            dataset_with_entropy[col] = EntropyFeatures.move_nans_to_front(
                dataset_with_entropy[col]
            )

        return dataset_with_entropy

    def merge_with_events(
        self,
        dataset: pd.DataFrame,
        events: pd.DataFrame,
        time_column: str = 'end_time'
    ) -> pd.DataFrame:
        """
        Merge features with events (triple barrier results).

        Args:
            dataset: Dataset with features
            events: Events DataFrame (triple barrier results)
            time_column: Time column name

        Returns:
            Merged dataset
        """
        # Clean and prepare
        original_size = len(dataset)
        dataset_clean = dataset.dropna().reset_index(drop=True)
        final_size = len(dataset_clean)

        self.logger.info(f"\n🧹 Cleanup: {original_size:,} → {final_size:,} samples")

        # Set index
        dataset_indexed = dataset_clean.set_index(time_column)

        # Merge with events
        temp_merged = pd.merge(
            events,
            dataset_indexed,
            how='left',
            left_index=True,
            right_on=time_column
        )

        temp_merged.index = temp_merged[time_column]
        temp_merged_filtered = temp_merged[temp_merged['close'].notna()]
        remove_count = len(temp_merged) - len(temp_merged_filtered)

        # Final dataset
        final_dataset = pd.concat([events, dataset_indexed], axis=1)
        final_dataset = final_dataset[final_dataset['meta_label'].notna()]
        final_dataset = final_dataset[remove_count:]

        return final_dataset

    def prepare_ml_dataset(
        self,
        final_dataset: pd.DataFrame,
        columns_to_drop: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare final ML dataset (X, y).

        Args:
            final_dataset: Merged dataset with features and labels
            columns_to_drop: Columns to exclude from features

        Returns:
            Tuple of (X, y)
        """
        if columns_to_drop is None:
            columns_to_drop = [
                't1', 'side', 'sl', 'pt', 'retorno', 'max_drawdown_in_trade',
                'label', 'meta_label', 'open', 'high', 'low', 'close',
                'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd',
                'total_volume', 'params', 'time_trial', 'log_close', 'fraq_close'
            ]

        # Extract y
        y = final_dataset[['meta_label']].copy()

        # Log class distribution
        self._log_class_distribution(y)

        # Prepare X
        columns_to_drop_existing = [col for col in columns_to_drop if col in final_dataset.columns]
        X = final_dataset.drop(columns=columns_to_drop_existing)

        self.logger.info(f"\n✅ Total features: {len(X.columns)}")
        self.logger.info(f"✅ Samples: {len(X):,}")

        # Feature breakdown
        self._log_feature_breakdown(X)

        self.feature_names = list(X.columns)

        return X, y

    def _log_class_distribution(self, y: pd.Series):
        """Log class distribution statistics."""
        class_distribution = y['meta_label'].value_counts()
        class_ratio = (
            class_distribution[0] / class_distribution[1]
            if 1 in class_distribution.index else float('inf')
        )

        self.logger.info("\n📊 Target Class Distribution:")
        self.logger.info(
            f"   Class 0 (Negative): {class_distribution.get(0, 0)} samples "
            f"({class_distribution.get(0, 0)/len(y)*100:.1f}%)"
        )
        self.logger.info(
            f"   Class 1 (Positive): {class_distribution.get(1, 0)} samples "
            f"({class_distribution.get(1, 0)/len(y)*100:.1f}%)"
        )
        self.logger.info(f"   Imbalance Ratio: {class_ratio:.2f}:1")

        if class_ratio > 3 or class_ratio < 0.33:
            self.logger.warning(
                "   ⚠️ Classes are HIGHLY IMBALANCED - "
                "custom sample_weight will be applied"
            )
        elif class_ratio > 1.5 or class_ratio < 0.67:
            self.logger.info(
                "   ⚠️ Classes are MODERATELY IMBALANCED - "
                "custom sample_weight will be applied"
            )
        else:
            self.logger.info(
                "   ✅ Classes are RELATIVELY BALANCED - "
                "custom sample_weight will still be applied"
            )

    def _log_feature_breakdown(self, X: pd.DataFrame):
        """Log feature breakdown by type."""
        entropy_features = [col for col in X.columns if 'entropy' in col.lower()]
        micro_features = [
            col for col in X.columns
            if any(x in col for x in [
                'corwin_schultz', 'becker', 'roll', 'amihud',
                'vpin', 'oir', 'kyle'
            ])
        ]
        other_features = [
            col for col in X.columns
            if col not in entropy_features + micro_features
        ]

        self.logger.info(f"\n📋 Feature breakdown:")
        self.logger.info(f"   🧠 Entropy: {len(entropy_features)}")
        self.logger.info(f"   🏗️ Microstructure: {len(micro_features)}")
        self.logger.info(f"   📊 Others: {len(other_features)}")

    def run_full_pipeline(
        self,
        dataset: pd.DataFrame,
        series_frac_diff: pd.Series,
        events: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Run complete feature engineering pipeline.

        Args:
            dataset: Base dataset (with fraq_close, etc.)
            series_frac_diff: Fractionally differentiated series for entropy
            events: Triple barrier events

        Returns:
            Tuple of (X, y) ready for ML
        """
        from py_modules_main_optimized.utils import log_memory

        self.logger.info("\n🔬 FEATURE ENGINEERING PIPELINE")
        self.logger.info("="*60)

        # Step 1: Microstructure features
        dataset_with_micro = self.calculate_microstructure_features(dataset)

        # Step 2: Entropy features
        entropy_results = self.calculate_entropy_features(series_frac_diff)

        # Step 3: Add entropy to dataset
        dataset_with_all_features = self.add_entropy_to_dataset(
            dataset_with_micro, entropy_results
        )

        # Step 4: Merge with events
        final_dataset = self.merge_with_events(dataset_with_all_features, events)

        # Step 5: Prepare ML dataset
        X, y = self.prepare_ml_dataset(final_dataset)

        log_memory("feature engineering")

        return X, y

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature engineering pipeline.

        Returns:
            Dictionary with pipeline summary
        """
        return {
            'total_features': len(self.feature_names),
            'microstructure_features': len(self.microstructure_feature_names),
            'entropy_features': len(self.entropy_feature_names),
            'configuration': {
                'microstructure_windows': self.microstructure_windows,
                'corwin_schultz_range': self.corwin_schultz_range,
                'entropy_windows': self.entropy_windows,
                'entropy_bins': self.entropy_bins
            }
        }
