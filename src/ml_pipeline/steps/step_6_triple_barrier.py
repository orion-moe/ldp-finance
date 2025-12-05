"""
Triple Barrier Method module.
Implements López de Prado's triple barrier labeling for meta-labeling.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from joblib import Parallel, delayed
from py_modules_main_optimized.ml_framework import TripleBarrierMethod


class TripleBarrierLabeler:
    """
    Triple Barrier labeling for financial ML.

    Implements vertical (time) and horizontal (profit/loss) barriers
    to generate labels for supervised learning.
    """

    def __init__(
        self,
        pt_sl: List[float] = None,
        n_jobs: int = 1
    ):
        """
        Initialize triple barrier labeler.

        Args:
            pt_sl: [profit_target, stop_loss] multipliers (default: [1.0, 1.0])
            n_jobs: Number of parallel jobs (default: 1, use -1 for all cores)
        """
        self.pt_sl = pt_sl if pt_sl is not None else [1.0, 1.0]
        self.n_jobs = n_jobs
        self.logger = logging.getLogger(__name__)

        self.barrier_events = None

    def clip_event_horizon(
        self,
        events: pd.DataFrame,
        close_series: pd.Series
    ) -> pd.DataFrame:
        """
        Clip event horizons (t1) to not exceed available data.

        Args:
            events: Events DataFrame with 't1' column
            close_series: Close price series

        Returns:
            Events with clipped t1
        """
        events_clipped = events.copy()

        # Clip t1 to last available timestamp
        events_clipped['t1'] = events_clipped['t1'].where(
            events_clipped['t1'] <= close_series.index[-1],
            pd.NaT
        )

        return events_clipped

    def apply_barriers(
        self,
        events: pd.DataFrame,
        close_series: pd.Series,
        pt_sl: Optional[List[float]] = None,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply triple barrier method to events.

        Args:
            events: Events DataFrame (must have 't1', 'trgt', 'side' columns)
            close_series: Close price series (indexed by datetime)
            pt_sl: Profit target and stop loss multipliers
            n_jobs: Number of parallel jobs

        Returns:
            DataFrame with barrier results
        """
        if pt_sl is None:
            pt_sl = self.pt_sl

        if n_jobs is None:
            n_jobs = self.n_jobs

        self.logger.info(f"\nApplying triple barrier method...")
        self.logger.info(f"   Profit Target: {pt_sl[0]}x")
        self.logger.info(f"   Stop Loss: {pt_sl[1]}x")
        self.logger.info(f"   Parallel jobs: {n_jobs}")

        # Clip event horizons
        events_clipped = self.clip_event_horizon(events, close_series)

        # Get molecules (event timestamps)
        molecules = events_clipped.index.tolist()

        # Apply barriers in parallel
        barrier_results = Parallel(n_jobs=n_jobs)(
            delayed(TripleBarrierMethod.applyPtSlOnT1)(
                close_series,
                events_clipped.loc[[molecule]],
                pt_sl,
                [molecule]
            ) for molecule in molecules
        )

        # Concatenate results
        barrier_events = pd.concat(barrier_results)

        self.barrier_events = barrier_events
        self.logger.info(f"✅ Barriers applied to {len(barrier_events):,} events")

        return barrier_events

    def generate_labels(
        self,
        barrier_events: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate labels from barrier events.

        Labels:
        - 1: Profit target hit first
        - -1: Stop loss hit first
        - 0: Time barrier hit first (no conclusion)

        Args:
            barrier_events: Barrier events (uses self.barrier_events if None)

        Returns:
            DataFrame with 'label' and 'meta_label' columns added
        """
        if barrier_events is None:
            if self.barrier_events is None:
                raise ValueError("No barrier events available. Run apply_barriers() first.")
            barrier_events = self.barrier_events
        else:
            self.barrier_events = barrier_events

        # Generate labels
        labeled_events = barrier_events.copy()
        labeled_events['label'] = labeled_events.apply(
            TripleBarrierMethod.get_label, axis=1
        )

        # Generate meta-labels (binary: profitable = 1, not profitable = 0)
        labeled_events['meta_label'] = np.where(
            labeled_events['label'] == 1, 1, 0
        )

        self.logger.info(f"\n✅ Labels generated:")
        self.logger.info(f"   Total: {len(labeled_events):,}")
        self.logger.info(
            f"   Profit Targets (1): "
            f"{(labeled_events['label'] == 1).sum():,} "
            f"({(labeled_events['label'] == 1).sum() / len(labeled_events) * 100:.1f}%)"
        )
        self.logger.info(
            f"   Stop Losses (-1): "
            f"{(labeled_events['label'] == -1).sum():,} "
            f"({(labeled_events['label'] == -1).sum() / len(labeled_events) * 100:.1f}%)"
        )
        self.logger.info(
            f"   Time Exits (0): "
            f"{(labeled_events['label'] == 0).sum():,} "
            f"({(labeled_events['label'] == 0).sum() / len(labeled_events) * 100:.1f}%)"
        )

        return labeled_events

    def run_full_labeling(
        self,
        events: pd.DataFrame,
        close_series: pd.Series,
        pt_sl: Optional[List[float]] = None,
        n_jobs: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Run complete triple barrier labeling pipeline.

        Convenience method that:
        1. Applies barriers
        2. Generates labels

        Args:
            events: Events DataFrame
            close_series: Close price series
            pt_sl: Profit target and stop loss multipliers
            n_jobs: Number of parallel jobs

        Returns:
            DataFrame with barriers and labels
        """
        self.logger.info("\n🎯 TRIPLE BARRIER METHOD")

        # Check if events is empty
        if len(events) == 0:
            self.logger.warning("⚠️ No events provided. Skipping triple barrier.")
            return pd.DataFrame()

        # Apply barriers
        barrier_events = self.apply_barriers(events, close_series, pt_sl, n_jobs)

        # Generate labels
        labeled_events = self.generate_labels(barrier_events)

        return labeled_events

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of triple barrier results.

        Returns:
            Dictionary with summary statistics
        """
        if self.barrier_events is None or len(self.barrier_events) == 0:
            return {
                'num_events': 0,
                'parameters': {
                    'pt_sl': self.pt_sl,
                    'n_jobs': self.n_jobs
                }
            }

        label_counts = self.barrier_events['label'].value_counts().to_dict()

        return {
            'num_events': len(self.barrier_events),
            'label_distribution': {
                'profit_targets': label_counts.get(1, 0),
                'stop_losses': label_counts.get(-1, 0),
                'time_exits': label_counts.get(0, 0)
            },
            'meta_label_ratio': float(self.barrier_events['meta_label'].mean()),
            'parameters': {
                'pt_sl': self.pt_sl,
                'n_jobs': self.n_jobs
            }
        }
