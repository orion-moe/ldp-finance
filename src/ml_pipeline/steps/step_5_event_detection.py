"""
Event Detection module using CUSUM filter.
Implements López de Prado's event detection methodology.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from py_modules_main_optimized.ml_framework import EventAnalyzer


class CUSUMEventDetector:
    """
    CUSUM-based event detector for financial time series.

    Detects significant directional moves in residuals using
    cumulative sum (CUSUM) filter with volatility scaling.
    """

    def __init__(
        self,
        tau1: float = 2.0,
        tau2: float = 1.0,
        volatility_span: int = 200,
        time_window_minutes: int = 240
    ):
        """
        Initialize event detector.

        Args:
            tau1: CUSUM threshold multiplier (default: 2.0)
            tau2: Target scaling factor (default: 1.0)
            volatility_span: Span for exponential volatility (default: 200)
            time_window_minutes: Time window for event horizon (default: 240)
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.volatility_span = volatility_span
        self.time_window_minutes = time_window_minutes
        self.logger = logging.getLogger(__name__)

        self.events = None
        self.volatility = None

    def calculate_volatility(
        self,
        series: pd.Series,
        span: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate exponential weighted volatility.

        Args:
            series: Input series (typically fraq_close)
            span: Span for EWM (uses self.volatility_span if None)

        Returns:
            Volatility series
        """
        if span is None:
            span = self.volatility_span

        vol = EventAnalyzer.getVol(series, span0=span)
        self.volatility = vol

        self.logger.info(f"Volatility calculated (span={span})")
        return vol

    def align_residuals_to_series(
        self,
        dataset: pd.DataFrame,
        residuals: np.ndarray,
        n_lost_fracdiff: int,
        ar_order: int
    ) -> pd.DataFrame:
        """
        Align residuals with dataset accounting for losses from frac diff and AR.

        Args:
            dataset: Dataset with fraq_close and vol columns
            residuals: AR model residuals
            n_lost_fracdiff: Number of samples lost in fractional diff
            ar_order: AR model order (p)

        Returns:
            Aligned DataFrame with residuals column
        """
        # Calculate alignment offset
        start_idx = n_lost_fracdiff + ar_order

        # Align
        aligned_df = dataset.iloc[start_idx:start_idx + len(residuals)].copy()

        # Add residuals
        if isinstance(residuals, np.ndarray):
            aligned_df['residuals'] = residuals
        else:
            aligned_df['residuals'] = residuals.values

        # Drop NaNs
        aligned_df = aligned_df.dropna(subset=['vol', 'residuals'])

        self.logger.info(f"Data after alignment: {len(aligned_df):,} records")

        return aligned_df

    def detect_events(
        self,
        residuals: pd.Series,
        volatility: pd.Series,
        tau1: Optional[float] = None,
        auto_adjust: bool = True
    ) -> pd.DataFrame:
        """
        Detect events using CUSUM filter.

        Args:
            residuals: Residuals series (indexed by datetime)
            volatility: Volatility series (indexed by datetime)
            tau1: CUSUM threshold (uses self.tau1 if None)
            auto_adjust: If True, reduce tau1 if no events detected

        Returns:
            DataFrame with detected events
        """
        if tau1 is None:
            tau1 = self.tau1

        self.logger.info(f"Detecting events with tau1={tau1}")

        # Detect events
        events = EventAnalyzer.getTEvents(
            residuals,
            volatility,
            tau1
        )

        # Auto-adjust if no events detected
        if len(events) == 0 and auto_adjust:
            self.logger.warning("⚠️ No events detected. Adjusting threshold...")
            tau1_adjusted = tau1 / 10
            self.logger.info(f"Reducing tau1 from {tau1:.2f} to {tau1_adjusted:.2f}")

            events = EventAnalyzer.getTEvents(
                residuals,
                volatility,
                tau1_adjusted
            )

            if len(events) > 0:
                self.logger.info(f"✅ {len(events)} events detected after adjustment")
                tau1 = tau1_adjusted
            else:
                self.logger.warning("Still no events detected after adjustment")

        # Format events
        if len(events) > 0:
            events.columns = ['time', 'trgt', 'side']
            events.set_index('time', inplace=True)

            # Add event horizon (t1)
            events['t1'] = events.index + pd.Timedelta(minutes=self.time_window_minutes)

            # Scale target
            events['trgt'] = events['trgt'] / tau1 * self.tau2

        self.events = events
        self.logger.info(f"\n✅ Events detected: {len(events):,}")

        return events

    def prepare_event_data(
        self,
        dataset: pd.DataFrame,
        volatility_column: str = 'vol'
    ) -> pd.DataFrame:
        """
        Prepare dataset for event detection.

        Args:
            dataset: Input dataset
            volatility_column: Name of volatility column (default: 'vol')

        Returns:
            Dataset indexed by time, ready for event detection
        """
        # Set index to end_time if not already
        if 'end_time' in dataset.columns:
            dataset_indexed = dataset.set_index('end_time')
        else:
            dataset_indexed = dataset.copy()

        return dataset_indexed

    def run_full_detection(
        self,
        dataset: pd.DataFrame,
        residuals: np.ndarray,
        n_lost_fracdiff: int,
        ar_order: int,
        fraq_close_column: str = 'fraq_close'
    ) -> pd.DataFrame:
        """
        Run complete event detection pipeline.

        Convenience method that:
        1. Calculates volatility
        2. Aligns residuals
        3. Detects events

        Args:
            dataset: Dataset with fraq_close column
            residuals: AR model residuals
            n_lost_fracdiff: Samples lost in fractional differentiation
            ar_order: AR model order
            fraq_close_column: Column name for fractionally differentiated series

        Returns:
            DataFrame with detected events
        """
        self.logger.info("\n🎯 EVENT DETECTION (CUSUM)")

        # Step 1: Calculate volatility
        vol = self.calculate_volatility(dataset[fraq_close_column])
        dataset_with_vol = dataset.copy()
        dataset_with_vol['vol'] = vol

        # Step 2: Align residuals
        aligned_df = self.align_residuals_to_series(
            dataset_with_vol,
            residuals,
            n_lost_fracdiff,
            ar_order
        )

        # Step 3: Prepare for detection
        series_for_events = self.prepare_event_data(aligned_df)

        # Step 4: Detect events
        events = self.detect_events(
            series_for_events['residuals'],
            series_for_events['vol'],
            auto_adjust=True
        )

        return events

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of event detection results.

        Returns:
            Dictionary with summary statistics
        """
        if self.events is None or len(self.events) == 0:
            return {
                'num_events': 0,
                'detection_parameters': {
                    'tau1': self.tau1,
                    'tau2': self.tau2,
                    'volatility_span': self.volatility_span,
                    'time_window_minutes': self.time_window_minutes
                }
            }

        return {
            'num_events': len(self.events),
            'event_sides': self.events['side'].value_counts().to_dict(),
            'avg_target': float(self.events['trgt'].mean()),
            'target_std': float(self.events['trgt'].std()),
            'detection_parameters': {
                'tau1': self.tau1,
                'tau2': self.tau2,
                'volatility_span': self.volatility_span,
                'time_window_minutes': self.time_window_minutes
            }
        }
