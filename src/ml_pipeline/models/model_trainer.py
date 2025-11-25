"""
Model training module with hyperparameter optimization.
Implements Random Forest with GridSearchCV.
"""

import time
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from typing import Tuple, Dict, Any, Optional


class RandomForestTrainer:
    """
    Random Forest model trainer with hyperparameter optimization.
    """

    def __init__(
        self,
        param_grid: Optional[Dict[str, Any]] = None,
        cv_n_splits: int = 5,
        cv_scoring: str = 'f1',
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 1
    ):
        """
        Initialize trainer.

        Args:
            param_grid: Parameter grid for GridSearchCV
            cv_n_splits: Number of CV folds
            cv_scoring: Scoring metric for CV
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs
            verbose: Verbosity level
        """
        self.param_grid = param_grid or self._get_default_param_grid()
        self.cv_n_splits = cv_n_splits
        self.cv_scoring = cv_scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

        self.model = None
        self.grid_search = None
        self.train_time = 0.0
        self.logger = logging.getLogger(__name__)

    def _get_default_param_grid(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Get default parameter grid.

        Args:
            quick_mode: If True, use reduced grid for faster search

        Returns:
            Parameter grid dictionary
        """
        if quick_mode:
            return {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [20, 50],
                'min_samples_leaf': [10, 20],
                'max_features': ['sqrt'],
                'bootstrap': [True]
            }
        else:
            return {
                'n_estimators': [50, 100, 150],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [20, 50, 100],
                'min_samples_leaf': [10, 20, 40],
                'max_features': ['sqrt', 'log2', 0.3],
                'bootstrap': [True]
            }

    def _log_param_grid(self):
        """Log parameter grid information."""
        self.logger.info("📊 Parameter Grid for optimization:")
        for key, value in self.param_grid.items():
            if isinstance(value, list):
                self.logger.info(f"   {key}: {value}")
            else:
                self.logger.info(f"   {key}: [{value}]")

        # Calculate total combinations
        total_combos = 1
        for values in self.param_grid.values():
            if isinstance(values, list):
                total_combos *= len(values)
        self.logger.info(f"\n   Total parameter combinations: {total_combos}")

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        sample_weight: Optional[np.ndarray] = None
    ) -> RandomForestClassifier:
        """
        Train Random Forest model with GridSearchCV.

        Args:
            X_train: Training features
            y_train: Training labels
            sample_weight: Optional sample weights

        Returns:
            Trained RandomForestClassifier
        """
        self.logger.info("\n🎯 RANDOM FOREST WITH GRID SEARCH OPTIMIZATION")
        self.logger.info("="*60)

        # Log parameter grid
        self._log_param_grid()

        # Base Random Forest model
        base_rf = RandomForestClassifier(
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        # Configure cross-validation strategy
        cv_strategy = StratifiedKFold(
            n_splits=self.cv_n_splits,
            shuffle=True,
            random_state=self.random_state
        )

        # Configure GridSearchCV
        self.grid_search = GridSearchCV(
            estimator=base_rf,
            param_grid=self.param_grid,
            cv=cv_strategy,
            scoring=self.cv_scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            refit=True,
            return_train_score=True
        )

        # Train with Grid Search
        self.logger.info("\n🔍 Starting Grid Search optimization...")
        self.logger.info(f"   Cross-validation: {self.cv_n_splits}-fold")
        self.logger.info(f"   Scoring metric: {self.cv_scoring.upper()}")
        self.logger.info("   This may take several minutes...")

        start_time = time.time()

        # Fit with sample weights
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight

        self.grid_search.fit(
            X_train,
            y_train.values.ravel() if hasattr(y_train, 'values') else y_train,
            **fit_params
        )

        self.train_time = time.time() - start_time

        # Get best model
        self.model = self.grid_search.best_estimator_

        self.logger.info(f"\n✅ Grid Search completed in {self.train_time:.2f} seconds")
        self.logger.info(f"   ({self.train_time/60:.2f} minutes)")

        # Display best results
        self._log_best_results()

        return self.model

    def _log_best_results(self):
        """Log best model results from GridSearchCV."""
        self.logger.info(f"\n🏆 Best Cross-Validation Score:")
        self.logger.info(f"   {self.cv_scoring.upper()} Score: {self.grid_search.best_score_:.4f}")

        self.logger.info("\n🏆 Best Hyperparameters found:")
        for key, value in self.grid_search.best_params_.items():
            self.logger.info(f"   {key}: {value}")

        # Show top 3 models from grid search
        self.logger.info("\n📊 Top 3 Models from Grid Search:")
        cv_results = pd.DataFrame(self.grid_search.cv_results_)
        top_3 = cv_results.nlargest(3, 'mean_test_score')[
            ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
        ]

        for idx, row in top_3.iterrows():
            rank = int(row['rank_test_score'])
            score = row['mean_test_score']
            std = row['std_test_score']
            self.logger.info(f"\n   Rank #{rank}:")
            self.logger.info(f"   {self.cv_scoring.upper()} Score: {score:.4f} (+/- {std:.4f})")

            # Show only key parameters that differ from best
            if rank > 1:
                for param_key, param_val in row['params'].items():
                    if param_val != self.grid_search.best_params_.get(param_key):
                        self.logger.info(f"     {param_key}: {param_val}")

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "Training"
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.

        Args:
            X: Features
            y: True labels
            dataset_name: Name of dataset for logging

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        # Make predictions
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'auc_roc': roc_auc_score(y, y_pred_proba)
        }

        # Log metrics
        self.logger.info(f"\n📊 {dataset_name} Metrics:")
        self.logger.info(f"   Accuracy:  {metrics['accuracy']:.4f}")
        self.logger.info(f"   Precision: {metrics['precision']:.4f}")
        self.logger.info(f"   Recall:    {metrics['recall']:.4f}")
        self.logger.info(f"   F1-Score:  {metrics['f1_score']:.4f}")
        self.logger.info(f"   AUC-ROC:   {metrics['auc_roc']:.4f}")

        return metrics

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance DataFrame.

        Args:
            feature_names: List of feature names

        Returns:
            DataFrame with features sorted by importance
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return feature_importance

    def get_cv_results(self) -> pd.DataFrame:
        """
        Get GridSearch CV results as DataFrame.

        Returns:
            DataFrame with CV results
        """
        if self.grid_search is None:
            raise ValueError("GridSearch not run yet. Call train() first.")

        return pd.DataFrame(self.grid_search.cv_results_)
