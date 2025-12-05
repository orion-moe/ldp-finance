"""
Feature Importance Analysis module.
Analyzes and categorizes feature importance from trained models.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import RandomForestClassifier


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance from trained models.

    Provides categorization and detailed breakdown of feature contributions.
    """

    def __init__(self):
        """Initialize feature importance analyzer."""
        self.logger = logging.getLogger(__name__)
        self.feature_importance_df = None
        self.categories = {}

    def extract_importance(
        self,
        model: RandomForestClassifier,
        feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained RandomForestClassifier
            feature_names: List of feature names

        Returns:
            DataFrame with features sorted by importance
        """
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        self.feature_importance_df = feature_importance

        return feature_importance

    def log_top_features(
        self,
        feature_importance: Optional[pd.DataFrame] = None,
        n: int = 20
    ):
        """
        Log top N most important features.

        Args:
            feature_importance: Feature importance DataFrame
            n: Number of top features to display
        """
        if feature_importance is None:
            if self.feature_importance_df is None:
                raise ValueError("No feature importance data. Run extract_importance() first.")
            feature_importance = self.feature_importance_df

        self.logger.info(f"\n🏆 Top {n} Most Important Features:")
        self.logger.info("-"*50)

        for idx, row in feature_importance.head(n).iterrows():
            self.logger.info(f"{row['feature']:<40} {row['importance']:.6f}")

    def categorize_features(
        self,
        feature_names: List[str],
        entropy_features: Optional[List[str]] = None,
        micro_features: Optional[List[str]] = None,
        other_features: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Categorize features into groups.

        Args:
            feature_names: All feature names
            entropy_features: List of entropy feature names
            micro_features: List of microstructure feature names
            other_features: List of other feature names

        Returns:
            Dictionary mapping category names to feature lists
        """
        # Auto-detect if not provided
        if entropy_features is None:
            entropy_features = [f for f in feature_names if 'entropy' in f.lower()]

        if micro_features is None:
            micro_features = [
                f for f in feature_names
                if any(x in f for x in [
                    'corwin_schultz', 'becker', 'roll', 'amihud',
                    'vpin', 'oir', 'kyle'
                ])
            ]

        if other_features is None:
            other_features = [
                f for f in feature_names
                if f not in entropy_features + micro_features
            ]

        # Additional volatility category
        volatility_features = [
            f for f in feature_names
            if 'vol' in f.lower() and f not in micro_features
        ]

        # Update 'others' to exclude volatility
        other_features = [
            f for f in other_features
            if f not in volatility_features
        ]

        categories = {
            'Entropy': entropy_features,
            'Microstructure': micro_features,
            'Volatility': volatility_features,
            'Others': other_features
        }

        self.categories = categories

        return categories

    def analyze_by_category(
        self,
        feature_importance: Optional[pd.DataFrame] = None,
        categories: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, float]:
        """
        Analyze feature importance by category.

        Args:
            feature_importance: Feature importance DataFrame
            categories: Dictionary of feature categories

        Returns:
            Dictionary mapping category names to total importance
        """
        if feature_importance is None:
            if self.feature_importance_df is None:
                raise ValueError("No feature importance data. Run extract_importance() first.")
            feature_importance = self.feature_importance_df

        if categories is None:
            if not self.categories:
                raise ValueError("No categories defined. Run categorize_features() first.")
            categories = self.categories

        self.logger.info("\n📊 Feature Importance by Category:")
        self.logger.info("-"*50)

        category_importance = {}

        for cat_name, cat_features in categories.items():
            cat_imp = feature_importance[
                feature_importance['feature'].isin(cat_features)
            ]['importance'].sum()

            category_importance[cat_name] = cat_imp

            self.logger.info(
                f"{cat_name:<20} {cat_imp:.4f} ({cat_imp*100:.2f}%)"
            )

        return category_importance

    def get_bottom_features(
        self,
        feature_importance: Optional[pd.DataFrame] = None,
        n: int = 20
    ) -> pd.DataFrame:
        """
        Get N least important features.

        Args:
            feature_importance: Feature importance DataFrame
            n: Number of bottom features

        Returns:
            DataFrame with bottom N features
        """
        if feature_importance is None:
            if self.feature_importance_df is None:
                raise ValueError("No feature importance data. Run extract_importance() first.")
            feature_importance = self.feature_importance_df

        return feature_importance.tail(n)

    def run_full_analysis(
        self,
        model: RandomForestClassifier,
        feature_names: List[str],
        entropy_features: Optional[List[str]] = None,
        micro_features: Optional[List[str]] = None,
        other_features: Optional[List[str]] = None,
        n_top: int = 20
    ) -> Dict[str, Any]:
        """
        Run complete feature importance analysis.

        Args:
            model: Trained RandomForestClassifier
            feature_names: List of feature names
            entropy_features: List of entropy feature names
            micro_features: List of microstructure feature names
            other_features: List of other feature names
            n_top: Number of top features to display

        Returns:
            Dictionary with analysis results
        """
        self.logger.info("\n📊 FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("="*60)

        # Step 1: Extract importance
        feature_importance = self.extract_importance(model, feature_names)

        # Step 2: Log top features
        self.log_top_features(feature_importance, n=n_top)

        # Step 3: Categorize features
        categories = self.categorize_features(
            feature_names,
            entropy_features,
            micro_features,
            other_features
        )

        # Step 4: Analyze by category
        category_importance = self.analyze_by_category(feature_importance, categories)

        # Compile results
        results = {
            'feature_importance': feature_importance,
            'categories': categories,
            'category_importance': category_importance,
            'top_features': feature_importance.head(n_top),
            'bottom_features': feature_importance.tail(n_top)
        }

        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature importance analysis.

        Returns:
            Dictionary with analysis summary
        """
        if self.feature_importance_df is None:
            return {
                'analyzed': False
            }

        return {
            'analyzed': True,
            'total_features': len(self.feature_importance_df),
            'top_feature': {
                'name': self.feature_importance_df.iloc[0]['feature'],
                'importance': float(self.feature_importance_df.iloc[0]['importance'])
            },
            'num_categories': len(self.categories),
            'category_names': list(self.categories.keys())
        }
