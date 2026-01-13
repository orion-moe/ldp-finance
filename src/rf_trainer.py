#!/usr/bin/env python
# coding: utf-8

"""
Random Forest Trainer for Trading Signals
==========================================

Loads pre-generated features and trains Random Forest classifier
with Optuna hyperparameter optimization.

This script performs:
- Load pre-generated features (Step 9)
- Optuna hyperparameter optimization (Step 10)
- Feature importance analysis (Step 10B - optional feature selection)
- Visualization generation (Step 11)

Usage:
    python rf_trainer.py --features /path/to/experiment_xxx/features/

Input Structure:
    experiment_{timestamp}/
    └── features/
        ├── X_train.parquet
        ├── y_train.parquet
        ├── sample_weights.parquet
        ├── feature_names.json
        └── triple_barrier_events.parquet

Output Structure (added to same experiment folder):
    experiment_{timestamp}/
    ├── features/                    (existing)
    ├── report/                      (existing)
    ├── model/                       (NEW)
    │   ├── random_forest_model.pkl
    │   ├── feature_importances.csv
    │   ├── optuna_trials.csv
    │   └── analysis_results.json
    └── plot/                        (NEW)
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── top_20_best_features.png
        ├── top_20_worst_features.png
        └── optuna_*.png/html
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import time
import gc
import psutil

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score, log_loss
)
from sklearn.model_selection import cross_val_predict

# Optuna for hyperparameter optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

# Cross-validation for financial time series
from ml_pipeline.cross_validation import (
    PurgedKFold,
    CombinatorialPurgedKFold,
    BacktestPathReconstructor,
    PBOCalculator,
    calculate_pbo_from_cpcv
)


def load_config(config_path=None):
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Set up logging configuration"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"rf_trainer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    logger = logging.getLogger('rf_trainer')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50*1024*1024,
        backupCount=10
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"="*70)
    logger.info(f"Random Forest Trainer Started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"="*70)

    return logger, log_path


logger, log_file_path = setup_logging()

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

logger.info("All imports loaded successfully!")


# ============================================================================
# 2. UTILITY FUNCTIONS
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory(operation_name):
    """Log memory usage for an operation"""
    memory_mb = get_memory_usage()
    logger.info(f"Memory after {operation_name}: {memory_mb:.1f} MB")


def find_latest_experiment(output_path=None):
    """
    Find the most recent experiment folder.

    Parameters:
    -----------
    output_path : str, optional
        Base path to search for experiments.
        Defaults to data/btcusdt-futures-um/output

    Returns:
    --------
    str : Path to the latest experiment's features directory
    """
    import glob

    if output_path is None:
        output_path = "data/btcusdt-futures-um/output"

    pattern = os.path.join(output_path, "experiment_*")
    experiments = glob.glob(pattern)

    if not experiments:
        raise FileNotFoundError(f"No experiments found in {output_path}")

    # Sort by name (timestamp format ensures chronological order)
    latest = sorted(experiments)[-1]
    features_dir = os.path.join(latest, 'features')

    if not os.path.exists(features_dir):
        raise FileNotFoundError(f"Features directory not found: {features_dir}")

    logger.info(f"Auto-detected latest experiment: {os.path.basename(latest)}")

    return features_dir


def generate_lagged_features(X_train, n_lags, feature_patterns=None):
    """
    Generate lagged features on-the-fly.

    Parameters:
    -----------
    X_train : pd.DataFrame
        Base features (without lags)
    n_lags : int
        Number of lags to create
    feature_patterns : list, optional
        Patterns to match for lagging.
        Defaults to all microstructure and entropy features.

    Returns:
    --------
    pd.DataFrame : Features with lags added
    """
    if feature_patterns is None:
        feature_patterns = [
            'entropy', 'vpin', 'oir', 'volatility',
            'becker_parkinson', 'amihud', 'kyle_lambda',
            'roll_spread', 'corwin_schultz', 'lz_', 'kont_'
        ]

    logger.info(f"Generating {n_lags} lags for features...")

    # Select columns to lag
    cols_to_lag = [col for col in X_train.columns
                   if any(p in col.lower() for p in feature_patterns)]

    logger.info(f"   Columns to lag: {len(cols_to_lag)}")

    # Generate lags
    lagged_dfs = [X_train]
    for lag in range(1, n_lags + 1):
        lagged = X_train[cols_to_lag].shift(lag)
        lagged.columns = [f"{col}_lag_{lag}" for col in cols_to_lag]
        lagged_dfs.append(lagged)

    X_with_lags = pd.concat(lagged_dfs, axis=1)

    logger.info(f"   Features after lags: {X_with_lags.shape[1]}")
    logger.info(f"   Lagged features added: {X_with_lags.shape[1] - X_train.shape[1]}")

    return X_with_lags


def load_features(features_dir):
    """Load pre-generated features from directory."""
    logger.info(f"Loading features from: {features_dir}")

    # Load X_train
    X_train_path = os.path.join(features_dir, 'X_train.parquet')
    X_train = pd.read_parquet(X_train_path)
    logger.info(f"   X_train loaded: {X_train.shape}")

    # Load y_train
    y_train_path = os.path.join(features_dir, 'y_train.parquet')
    y_train = pd.read_parquet(y_train_path)
    logger.info(f"   y_train loaded: {y_train.shape}")

    # Load sample weights
    weights_path = os.path.join(features_dir, 'sample_weights.parquet')
    weights_df = pd.read_parquet(weights_path)
    sample_weights = weights_df['weight'].values
    logger.info(f"   Sample weights loaded: {len(sample_weights)}")

    # Load feature names
    feature_names_path = os.path.join(features_dir, 'feature_names.json')
    with open(feature_names_path, 'r') as f:
        feature_names = json.load(f)
    logger.info(f"   Feature names loaded: {feature_names['n_features']} features")

    # Load triple barrier events (for PurgedKFold)
    triple_barrier_path = os.path.join(features_dir, 'triple_barrier_events.parquet')
    triple_barrier_events = pd.read_parquet(triple_barrier_path)
    logger.info(f"   Triple barrier events loaded: {len(triple_barrier_events)}")

    return X_train, y_train, sample_weights, feature_names, triple_barrier_events


# ============================================================================
# 3. MAIN TRAINING PIPELINE
# ============================================================================

def main(features_dir=None, config_path=None):
    """Main training pipeline

    Parameters:
    -----------
    features_dir : str, optional
        Path to features directory. If None, auto-detects latest experiment.
    config_path : str, optional
        Path to config.yaml. If None, uses default.
    """

    pipeline_start_time = time.time()

    # Load configuration
    config = load_config(config_path)

    logger.info("\n" + "="*70)
    logger.info("STARTING RANDOM FOREST TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Process PID: {os.getpid()}")
    logger.info(f"Initial memory: {get_memory_usage():.1f} MB")

    try:
        # ========================================================================
        # STEP 0: AUTO-DETECT EXPERIMENT (if not provided)
        # ========================================================================

        if features_dir is None:
            logger.info("\nSTEP 0: AUTO-DETECTING LATEST EXPERIMENT")
            features_dir = find_latest_experiment(config['data'].get('output_path'))
            logger.info(f"   Using: {features_dir}")

        # ========================================================================
        # STEP 1: LOAD FEATURES
        # ========================================================================

        logger.info("\nSTEP 1: LOADING FEATURES")
        step_start = time.time()

        X_train, y_train, sample_weights, feature_names, triple_barrier_events = load_features(features_dir)

        # Get experiment folder (parent of features folder)
        experiment_dir = os.path.dirname(features_dir.rstrip('/'))
        logger.info(f"Experiment directory: {experiment_dir}")

        # Create output directories
        model_dir = os.path.join(experiment_dir, 'model')
        plot_dir = os.path.join(experiment_dir, 'plot')
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)

        log_memory("load features")
        logger.info(f"Step 1 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 1B: GENERATE LAGGED FEATURES
        # ========================================================================

        logger.info("\nSTEP 1B: GENERATING LAGGED FEATURES")
        step_start = time.time()

        n_lags = config['features'].get('n_lags', 15)
        logger.info(f"   Number of lags to generate: {n_lags}")

        # Check if features already have lags (backward compatibility)
        has_lags = any('_lag_' in col for col in X_train.columns)

        if has_lags:
            logger.info("   Features already have lags, skipping lag generation")
        else:
            # Generate lags
            original_shape = X_train.shape
            X_train = generate_lagged_features(X_train, n_lags)

            # Dropna after adding lags (lags create NaNs at the beginning)
            logger.info(f"   Dropping NaN rows created by lags...")
            n_before = len(X_train)
            valid_idx = X_train.dropna().index

            X_train = X_train.loc[valid_idx]
            y_train = y_train.loc[valid_idx]
            triple_barrier_events = triple_barrier_events.loc[valid_idx]

            # Handle sample_weights - convert to series with index for alignment
            weights_series = pd.Series(sample_weights, index=feature_names.get('original_index', range(len(sample_weights))))
            if len(weights_series) == n_before:
                # Rebuild weights series with X_train's original index before lag generation
                weights_df = pd.read_parquet(os.path.join(features_dir, 'sample_weights.parquet'))
                sample_weights = weights_df.loc[valid_idx, 'weight'].values
            else:
                sample_weights = sample_weights[:len(X_train)]

            n_after = len(X_train)
            logger.info(f"   Rows before dropna: {n_before}")
            logger.info(f"   Rows after dropna: {n_after}")
            logger.info(f"   Rows dropped: {n_before - n_after}")
            logger.info(f"   Final feature matrix: {X_train.shape}")

        log_memory("generate lags")
        logger.info(f"Step 1B completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 2: MODEL TRAINING WITH OPTUNA
        # ========================================================================

        logger.info("\nSTEP 2: RANDOM FOREST WITH OPTUNA OPTIMIZATION")
        logger.info("="*60)
        step_start = time.time()

        # Configure cross-validation strategy
        cv_folds = config['optuna']['cv_folds']
        pct_embargo = config['optuna']['pct_embargo']

        # Align t1 with X_train index
        t1_aligned = triple_barrier_events['t1'].reindex(X_train.index)

        # Check if CPCV is enabled
        cpcv_config = config.get('cpcv', {})
        use_cpcv = cpcv_config.get('enabled', False)

        if use_cpcv:
            # Use Combinatorial Purged K-Fold Cross-Validation (Chapter 12 AFML)
            n_splits = cpcv_config.get('n_splits', 6)
            n_test_splits = cpcv_config.get('n_test_splits', 2)
            cpcv_embargo = cpcv_config.get('pct_embargo', pct_embargo)

            cv_strategy = CombinatorialPurgedKFold(
                n_splits=n_splits,
                n_test_splits=n_test_splits,
                t1=t1_aligned,
                pct_embargo=cpcv_embargo
            )
            cv_name = f"CPCV (N={n_splits}, k={n_test_splits})"
            n_cv_splits = cv_strategy.n_combinations
        else:
            # Use standard PurgedKFold
            cv_strategy = PurgedKFold(
                n_splits=cv_folds,
                t1=t1_aligned,
                pct_embargo=pct_embargo
            )
            cv_name = f"{cv_folds}-fold PurgedKFold"
            n_cv_splits = cv_folds

        # Optuna configuration
        N_TRIALS = config['optuna']['n_trials']
        optuna_seed = config['optuna']['seed']
        rf_config = config['random_forest']

        logger.info(f"Optuna Configuration:")
        logger.info(f"   - Sampler: TPE (Tree-structured Parzen Estimator)")
        logger.info(f"   - Pruner: MedianPruner")
        logger.info(f"   - Trials: {N_TRIALS}")
        logger.info(f"   - Cross-validation: {cv_name}")
        logger.info(f"   - CV Splits: {n_cv_splits}")
        logger.info(f"   - Embargo: {pct_embargo*100:.1f}%")
        if use_cpcv:
            logger.info(f"   - Paths per observation: {cv_strategy.n_paths}")

        # Suppress Optuna's verbose output
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Define objective function
        def optuna_objective(trial):
            """Objective function for Optuna hyperparameter optimization."""
            params = {
                'n_estimators': trial.suggest_int('n_estimators',
                    rf_config['n_estimators_min'], rf_config['n_estimators_max']),
                'max_depth': trial.suggest_int('max_depth',
                    rf_config['max_depth_min'], rf_config['max_depth_max']),
                'min_samples_split': trial.suggest_int('min_samples_split',
                    rf_config['min_samples_split_min'], rf_config['min_samples_split_max']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                    rf_config['min_samples_leaf_min'], rf_config['min_samples_leaf_max']),
                'max_features': trial.suggest_categorical('max_features',
                    rf_config['max_features_options']),
                'bootstrap': True,
                'random_state': optuna_seed,
                'n_jobs': -1
            }

            rf = RandomForestClassifier(**params)

            cv_scores = []
            # Use sklearn_split for CPCV (returns 2 values), regular split for PurgedKFold
            if use_cpcv:
                cv_iterator = cv_strategy.sklearn_split(X_train)
            else:
                cv_iterator = cv_strategy.split(X_train)

            for fold_idx, (train_idx, val_idx) in enumerate(cv_iterator):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.values.ravel()[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.values.ravel()[val_idx]
                weights_fold = sample_weights[train_idx]

                rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
                y_pred_proba = rf.predict_proba(X_fold_val)[:, 1]
                fold_score = log_loss(y_fold_val, y_pred_proba)
                cv_scores.append(fold_score)

                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        # Create study
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=optuna_seed),
            pruner=MedianPruner(n_warmup_steps=2)
        )

        # Progress tracking
        trials_completed = [0]
        best_score_so_far = [float('inf')]

        def optuna_callback(study, trial):
            """Callback to log progress."""
            trials_completed[0] += 1
            if trial.value is not None and trial.value < best_score_so_far[0]:
                best_score_so_far[0] = trial.value
                logger.info(f"   Trial {trials_completed[0]}/{N_TRIALS}: New best Log Loss = {trial.value:.4f}")
            elif trials_completed[0] % 10 == 0:
                logger.info(f"   Trial {trials_completed[0]}/{N_TRIALS}: Current best = {best_score_so_far[0]:.4f}")

        logger.info("\nStarting Optuna optimization...")

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=N_TRIALS,
            callbacks=[optuna_callback],
            show_progress_bar=False
        )

        train_time = time.time() - step_start

        # Get best parameters and create final model
        best_params = study.best_params.copy()
        best_params['bootstrap'] = True
        best_params['random_state'] = optuna_seed
        best_params['n_jobs'] = -1

        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(X_train, y_train.values.ravel(), sample_weight=sample_weights[:len(X_train)])

        logger.info(f"\nOptuna optimization completed in {train_time:.2f} seconds")

        # Display best results
        logger.info(f"\nBest Cross-Validation Score:")
        logger.info(f"   Log Loss: {study.best_value:.4f}")

        logger.info("\nBest Hyperparameters found:")
        for key, value in study.best_params.items():
            logger.info(f"   {key}: {value}")

        # Show optimization statistics
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        logger.info(f"\nOptimization Statistics:")
        logger.info(f"   - Total trials: {len(study.trials)}")
        logger.info(f"   - Completed trials: {len(completed_trials)}")
        logger.info(f"   - Pruned trials: {len(pruned_trials)}")

        # Training predictions
        y_pred_train = rf_model.predict(X_train)
        y_pred_proba_train = rf_model.predict_proba(X_train)[:, 1]

        # CV predictions - create sklearn-compatible CV iterator
        if use_cpcv:
            # For CPCV, use sklearn_split which returns (train, test) only
            cv_for_sklearn = list(cv_strategy.sklearn_split(X_train))
        else:
            cv_for_sklearn = cv_strategy

        y_pred_cv = cross_val_predict(
            rf_model, X_train, y_train.values.ravel(),
            cv=cv_for_sklearn, method='predict', n_jobs=-1
        )
        y_pred_proba_cv = cross_val_predict(
            rf_model, X_train, y_train.values.ravel(),
            cv=cv_for_sklearn, method='predict_proba', n_jobs=-1
        )[:, 1]

        # Metrics
        logger.info("\nTraining Metrics (In-Sample):")
        logger.info(f"   Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Precision: {precision_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Recall: {recall_score(y_train, y_pred_train):.4f}")
        logger.info(f"   F1-Score: {f1_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Log Loss: {log_loss(y_train, y_pred_proba_train):.4f}")
        logger.info(f"   AUC-ROC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

        logger.info(f"\nValidation Metrics (Out-of-Sample via {cv_name}):")
        logger.info(f"   Accuracy: {accuracy_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Precision: {precision_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Recall: {recall_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   F1-Score: {f1_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Log Loss: {log_loss(y_train, y_pred_proba_cv):.4f}")
        logger.info(f"   AUC-ROC: {roc_auc_score(y_train, y_pred_proba_cv):.4f}")

        # Overfitting analysis
        train_auc = roc_auc_score(y_train, y_pred_proba_train)
        val_auc = roc_auc_score(y_train, y_pred_proba_cv)
        logger.info(f"\nOverfitting Analysis:")
        logger.info(f"   Train AUC: {train_auc:.4f}  |  Val AUC: {val_auc:.4f}")
        logger.info(f"   AUC Gap: {train_auc - val_auc:.4f}")

        # Calculate PBO if CPCV is enabled
        pbo_results = None
        if use_cpcv and cpcv_config.get('calculate_pbo', True):
            logger.info("\nCalculating Probability of Backtest Overfitting (PBO)...")

            # Reinitialize cv_strategy to get fresh group indices
            cv_strategy_pbo = CombinatorialPurgedKFold(
                n_splits=n_splits,
                n_test_splits=n_test_splits,
                t1=t1_aligned,
                pct_embargo=cpcv_embargo
            )

            # Calculate PBO using the trained model
            pbo_results = calculate_pbo_from_cpcv(
                model=rf_model,
                X=X_train,
                y=y_train.iloc[:, 0] if hasattr(y_train, 'iloc') else pd.Series(y_train.values.ravel(), index=X_train.index),
                cpcv=cv_strategy_pbo,
                sample_weight=sample_weights[:len(X_train)]
            )

            logger.info(f"\nPBO Results:")
            logger.info(f"   Probability of Backtest Overfitting: {pbo_results['pbo']:.4f}")
            logger.info(f"   Deflated Sharpe Ratio: {pbo_results['deflated_sharpe']:.4f}")
            logger.info(f"   Max Path Sharpe: {pbo_results['max_sharpe']:.4f}")
            logger.info(f"   Median Path Sharpe: {pbo_results['median_sharpe']:.4f}")
            logger.info(f"   Path Sharpe Std: {pbo_results['sharpe_std']:.4f}")
            logger.info(f"   Number of Paths: {pbo_results['n_paths']}")
            logger.info(f"   Number of Combinations: {pbo_results['n_combinations']}")

            # Warn if PBO exceeds threshold
            pbo_threshold = cpcv_config.get('pbo_warning_threshold', 0.5)
            if pbo_results['pbo'] > pbo_threshold:
                logger.warning(f"   WARNING: PBO ({pbo_results['pbo']:.4f}) exceeds threshold ({pbo_threshold})")
                logger.warning(f"   The model may be overfit to the backtest data!")

        log_memory("model training")

        # ========================================================================
        # STEP 3: FEATURE IMPORTANCE ANALYSIS
        # ========================================================================

        logger.info("\nSTEP 3: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        step_start = time.time()

        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        logger.info("Top 20 Most Important Features:")
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"   {row['feature']:<40} {row['importance']:.6f}")

        # Feature selection tracking
        feature_selection_info = {
            'enabled': False,
            'used_selected_model': False,
            'original_features': len(X_train.columns),
            'selected_features': len(X_train.columns),
            'auc_improvement': 0.0,
            'log_loss_improvement': 0.0
        }

        # ========================================================================
        # STEP 3B: FEATURE SELECTION AND RETRAINING (Optional)
        # ========================================================================

        if config.get('feature_selection', {}).get('enabled', False):
            logger.info("\nSTEP 3B: FEATURE SELECTION AND RETRAINING")
            logger.info("="*60)

            feature_selection_info['enabled'] = True
            top_n = config['feature_selection']['top_n_features']
            min_importance = config['feature_selection'].get('min_importance', 0)

            selected_features = feature_importance[
                (feature_importance['importance'] > min_importance)
            ].head(top_n)['feature'].tolist()

            logger.info(f"Feature Selection:")
            logger.info(f"   - Original features: {len(X_train.columns)}")
            logger.info(f"   - Selected features: {len(selected_features)}")

            X_train_selected = X_train[selected_features]

            # Retrain with selected features
            def optuna_objective_selected(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators',
                        rf_config['n_estimators_min'], rf_config['n_estimators_max']),
                    'max_depth': trial.suggest_int('max_depth',
                        rf_config['max_depth_min'], rf_config['max_depth_max']),
                    'min_samples_split': trial.suggest_int('min_samples_split',
                        rf_config['min_samples_split_min'], rf_config['min_samples_split_max']),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf',
                        rf_config['min_samples_leaf_min'], rf_config['min_samples_leaf_max']),
                    'max_features': trial.suggest_categorical('max_features',
                        rf_config['max_features_options']),
                    'bootstrap': True,
                    'random_state': optuna_seed,
                    'n_jobs': -1
                }

                rf = RandomForestClassifier(**params)

                cv_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train_selected)):
                    X_fold_train = X_train_selected.iloc[train_idx]
                    y_fold_train = y_train.values.ravel()[train_idx]
                    X_fold_val = X_train_selected.iloc[val_idx]
                    y_fold_val = y_train.values.ravel()[val_idx]
                    weights_fold = sample_weights[train_idx]

                    rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
                    y_pred_proba = rf.predict_proba(X_fold_val)[:, 1]
                    fold_score = log_loss(y_fold_val, y_pred_proba)
                    cv_scores.append(fold_score)

                    trial.report(np.mean(cv_scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return np.mean(cv_scores)

            study_selected = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=optuna_seed + 1),
                pruner=MedianPruner(n_warmup_steps=2)
            )

            trials_sel = [0]
            best_sel = [float('inf')]

            def callback_sel(study, trial):
                trials_sel[0] += 1
                if trial.value is not None and trial.value < best_sel[0]:
                    best_sel[0] = trial.value
                    logger.info(f"   Trial {trials_sel[0]}/{N_TRIALS}: New best = {trial.value:.4f}")
                elif trials_sel[0] % 20 == 0:
                    logger.info(f"   Trial {trials_sel[0]}/{N_TRIALS}: Best = {best_sel[0]:.4f}")

            logger.info(f"\nRetraining with {len(selected_features)} features...")

            study_selected.optimize(
                optuna_objective_selected,
                n_trials=N_TRIALS,
                callbacks=[callback_sel],
                show_progress_bar=False
            )

            best_params_sel = study_selected.best_params.copy()
            best_params_sel['bootstrap'] = True
            best_params_sel['random_state'] = optuna_seed
            best_params_sel['n_jobs'] = -1

            rf_model_selected = RandomForestClassifier(**best_params_sel)
            rf_model_selected.fit(X_train_selected, y_train.values.ravel(),
                                  sample_weight=sample_weights[:len(X_train_selected)])

            # Compare results - use sklearn-compatible CV iterator
            if use_cpcv:
                cv_for_sklearn_sel = list(cv_strategy.sklearn_split(X_train_selected))
            else:
                cv_for_sklearn_sel = cv_strategy

            y_pred_cv_sel = cross_val_predict(
                rf_model_selected, X_train_selected, y_train.values.ravel(),
                cv=cv_for_sklearn_sel, method='predict', n_jobs=-1
            )
            y_pred_proba_cv_sel = cross_val_predict(
                rf_model_selected, X_train_selected, y_train.values.ravel(),
                cv=cv_for_sklearn_sel, method='predict_proba', n_jobs=-1
            )[:, 1]

            auc_all = roc_auc_score(y_train, y_pred_proba_cv)
            ll_all = log_loss(y_train, y_pred_proba_cv)
            auc_sel = roc_auc_score(y_train, y_pred_proba_cv_sel)
            ll_sel = log_loss(y_train, y_pred_proba_cv_sel)

            logger.info(f"\nCOMPARISON: All Features vs Selected")
            logger.info(f"   AUC: {auc_all:.4f} vs {auc_sel:.4f} (diff: {auc_sel - auc_all:+.4f})")
            logger.info(f"   Log Loss: {ll_all:.4f} vs {ll_sel:.4f} (diff: {ll_all - ll_sel:+.4f})")

            feature_selection_info['selected_features'] = len(selected_features)
            feature_selection_info['auc_improvement'] = float(auc_sel - auc_all)
            feature_selection_info['log_loss_improvement'] = float(ll_all - ll_sel)
            feature_selection_info['auc_all_features'] = float(auc_all)
            feature_selection_info['auc_selected_features'] = float(auc_sel)

            if auc_sel > auc_all:
                logger.info("\nSelected features model is BETTER! Using it.")
                feature_selection_info['used_selected_model'] = True
                rf_model = rf_model_selected
                y_pred_train = rf_model_selected.predict(X_train_selected)
                y_pred_proba_train = rf_model_selected.predict_proba(X_train_selected)[:, 1]
                y_pred_cv = y_pred_cv_sel
                y_pred_proba_cv = y_pred_proba_cv_sel
                X_train = X_train_selected
                study = study_selected

                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                logger.info("\nOriginal model is better. Keeping all features.")

            # Save feature selection results
            selection_results_path = os.path.join(model_dir, 'feature_selection_results.json')
            with open(selection_results_path, 'w') as f:
                json.dump({
                    'selected_features': selected_features,
                    'feature_selection_info': feature_selection_info
                }, f, indent=2)
            logger.info(f"   Saved: {selection_results_path}")

        logger.info(f"Step 3 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 4: GENERATE VISUALIZATION PLOTS
        # ========================================================================

        logger.info("\nSTEP 4: GENERATING VISUALIZATION PLOTS")
        logger.info("="*60)
        step_start = time.time()

        # Set plot style
        sns.set_palette("husl")

        # 1. CONFUSION MATRIX
        logger.info("1. Generating Confusion Matrix...")
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        cm_train = confusion_matrix(y_train, y_pred_train)
        tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()
        prec_tr = tp_tr / (tp_tr + fp_tr) if (tp_tr + fp_tr) > 0 else 0
        rec_tr = tp_tr / (tp_tr + fn_tr) if (tp_tr + fn_tr) > 0 else 0
        f1_tr = 2 * prec_tr * rec_tr / (prec_tr + rec_tr) if (prec_tr + rec_tr) > 0 else 0

        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0])
        axes[0].set_title(f'TRAINING (In-Sample)\nPrec={prec_tr:.3f} | Rec={rec_tr:.3f} | F1={f1_tr:.3f}',
                          fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        cm_val = confusion_matrix(y_train, y_pred_cv)
        tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
        prec_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        rec_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        f1_val = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0

        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges', square=True,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[1])
        axes[1].set_title(f'VALIDATION (Out-of-Sample)\nPrec={prec_val:.3f} | Rec={rec_val:.3f} | F1={f1_val:.3f}',
                          fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        gap = f1_tr - f1_val
        status = "OVERFITTING" if gap > 0.1 else "OK" if gap < 0.05 else "SLIGHT OVERFIT"
        plt.suptitle(f'Confusion Matrix Comparison | F1 Gap: {gap:.3f} ({status})',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {cm_path}")

        # 2. ROC CURVE
        logger.info("2. Generating ROC Curve...")
        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
        auc_train = roc_auc_score(y_train, y_pred_proba_train)
        fpr_val, tpr_val, _ = roc_curve(y_train, y_pred_proba_cv)
        auc_val = roc_auc_score(y_train, y_pred_proba_cv)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr_train, tpr_train, color='blue', lw=2,
                 label=f'Training (AUC = {auc_train:.4f})')
        plt.plot(fpr_val, tpr_val, color='darkorange', lw=2,
                 label=f'Validation (AUC = {auc_val:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--',
                 label='Random Classifier (AUC = 0.5000)')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve Comparison\nOverfitting Gap: {auc_train - auc_val:.4f}',
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        roc_path = os.path.join(plot_dir, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {roc_path}")

        # 3. TOP 20 BEST FEATURES
        logger.info("3. Generating Top 20 Best Features...")
        top_20 = feature_importance.head(20)

        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(top_20)), top_20['importance'], color='forestgreen')
        plt.yticks(range(len(top_20)), top_20['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, top_20['importance'])):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        top_path = os.path.join(plot_dir, 'top_20_best_features.png')
        plt.savefig(top_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {top_path}")

        # 4. TOP 20 WORST FEATURES
        logger.info("4. Generating Top 20 Worst Features...")
        bottom_20 = feature_importance.tail(20).sort_values('importance', ascending=True)

        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(bottom_20)), bottom_20['importance'], color='crimson')
        plt.yticks(range(len(bottom_20)), bottom_20['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Least Important Features', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        for i, (bar, val) in enumerate(zip(bars, bottom_20['importance'])):
            plt.text(val, i, f' {val:.6f}', va='center', fontsize=9)

        plt.tight_layout()
        bottom_path = os.path.join(plot_dir, 'top_20_worst_features.png')
        plt.savefig(bottom_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {bottom_path}")

        # OPTUNA PLOTS
        logger.info("5. Generating Optuna plots...")

        try:
            fig = plot_optimization_history(study)
            fig.update_layout(title="Optimization History - Log Loss Over Trials", template="plotly_white")
            fig.write_html(os.path.join(plot_dir, 'optuna_optimization_history.html'))
            fig.write_image(os.path.join(plot_dir, 'optuna_optimization_history.png'), width=1200, height=600, scale=2)
            logger.info(f"   Saved: optuna_optimization_history.png/html")
        except Exception as e:
            logger.warning(f"   Could not generate optimization history: {e}")

        try:
            fig = plot_param_importances(study)
            fig.update_layout(title="Hyperparameter Importances", template="plotly_white")
            fig.write_html(os.path.join(plot_dir, 'optuna_param_importances.html'))
            fig.write_image(os.path.join(plot_dir, 'optuna_param_importances.png'), width=1000, height=600, scale=2)
            logger.info(f"   Saved: optuna_param_importances.png/html")
        except Exception as e:
            logger.warning(f"   Could not generate parameter importances: {e}")

        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(title="Parallel Coordinate Plot", template="plotly_white")
            fig.write_html(os.path.join(plot_dir, 'optuna_parallel_coordinate.html'))
            fig.write_image(os.path.join(plot_dir, 'optuna_parallel_coordinate.png'), width=1400, height=700, scale=2)
            logger.info(f"   Saved: optuna_parallel_coordinate.png/html")
        except Exception as e:
            logger.warning(f"   Could not generate parallel coordinate: {e}")

        try:
            fig = plot_slice(study)
            fig.update_layout(title="Slice Plot - Parameter vs Objective", template="plotly_white")
            fig.write_html(os.path.join(plot_dir, 'optuna_slice_plot.html'))
            fig.write_image(os.path.join(plot_dir, 'optuna_slice_plot.png'), width=1400, height=800, scale=2)
            logger.info(f"   Saved: optuna_slice_plot.png/html")
        except Exception as e:
            logger.warning(f"   Could not generate slice plot: {e}")

        logger.info(f"Step 4 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 5: SAVE MODEL AND RESULTS
        # ========================================================================

        logger.info("\nSTEP 5: SAVING MODEL AND RESULTS")
        logger.info("="*60)
        step_start = time.time()

        # Save model
        import joblib
        model_path = os.path.join(model_dir, 'random_forest_model.pkl')
        joblib.dump(rf_model, model_path)
        logger.info(f"   Saved: {model_path}")

        # Save feature importances
        fi_path = os.path.join(model_dir, 'feature_importances.csv')
        feature_importance.to_csv(fi_path, index=False)
        logger.info(f"   Saved: {fi_path}")

        # Save Optuna trials
        trials_path = os.path.join(model_dir, 'optuna_trials.csv')
        study.trials_dataframe().to_csv(trials_path, index=False)
        logger.info(f"   Saved: {trials_path}")

        # Save analysis results
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'data_stats': {
                'samples': len(X_train),
                'features': len(X_train.columns)
            },
            'cross_validation': {
                'method': cv_name,
                'use_cpcv': use_cpcv,
                'n_splits': n_cv_splits,
                'embargo_pct': pct_embargo
            },
            'hyperparameter_optimization': {
                'method': 'Optuna',
                'n_trials': N_TRIALS,
                'cv_folds': cv_folds,
                'best_cv_score_log_loss': float(study.best_value),
                'best_params': study.best_params,
                'training_time_seconds': train_time
            },
            'model_performance': {
                'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
                'train_f1_score': float(f1_score(y_train, y_pred_train)),
                'train_log_loss': float(log_loss(y_train, y_pred_proba_train)),
                'train_auc_roc': float(roc_auc_score(y_train, y_pred_proba_train)),
                'val_accuracy': float(accuracy_score(y_train, y_pred_cv)),
                'val_f1_score': float(f1_score(y_train, y_pred_cv)),
                'val_log_loss': float(log_loss(y_train, y_pred_proba_cv)),
                'val_auc_roc': float(roc_auc_score(y_train, y_pred_proba_cv))
            },
            'feature_selection': feature_selection_info
        }

        # Add PBO results if available
        if pbo_results is not None:
            results_summary['pbo_analysis'] = {
                'pbo': pbo_results['pbo'],
                'deflated_sharpe': pbo_results['deflated_sharpe'],
                'max_sharpe': pbo_results['max_sharpe'],
                'median_sharpe': pbo_results['median_sharpe'],
                'sharpe_std': pbo_results['sharpe_std'],
                'n_paths': pbo_results['n_paths'],
                'n_combinations': pbo_results['n_combinations']
            }

            # Save full PBO results separately
            pbo_path = os.path.join(model_dir, 'pbo_results.json')
            with open(pbo_path, 'w') as f:
                json.dump(pbo_results, f, indent=2)
            logger.info(f"   Saved: {pbo_path}")

        results_path = os.path.join(model_dir, 'analysis_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logger.info(f"   Saved: {results_path}")

        logger.info(f"Step 5 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        pipeline_elapsed = time.time() - pipeline_start_time

        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETE")
        logger.info("="*70)

        logger.info(f"\nModel Performance Summary:")
        logger.info(f"   Training AUC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")
        logger.info(f"   Validation AUC: {roc_auc_score(y_train, y_pred_proba_cv):.4f}")
        logger.info(f"   Best Log Loss (CV): {study.best_value:.4f}")

        logger.info(f"\nOutput directory: {experiment_dir}")
        logger.info(f"\nFiles generated:")
        logger.info(f"  model/")
        logger.info(f"    random_forest_model.pkl     - Trained model")
        logger.info(f"    feature_importances.csv     - Feature ranking")
        logger.info(f"    optuna_trials.csv           - All trials")
        logger.info(f"    analysis_results.json       - Complete results")
        if pbo_results is not None:
            logger.info(f"    pbo_results.json            - PBO analysis (CPCV)")
        logger.info(f"  plot/")
        logger.info(f"    confusion_matrix.png        - Confusion matrix")
        logger.info(f"    roc_curve.png               - ROC curve")
        logger.info(f"    top_20_best_features.png    - Best features")
        logger.info(f"    top_20_worst_features.png   - Worst features")
        logger.info(f"    optuna_*.png/html           - Optuna plots")

        logger.info(f"\nTotal time: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
        logger.info(f"Final memory: {get_memory_usage():.1f} MB")

        return {
            'model': rf_model,
            'feature_importance': feature_importance,
            'results_summary': results_summary,
            'model_dir': model_dir,
            'plot_dir': plot_dir
        }

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        raise

    finally:
        logger.info("\n" + "="*70)
        logger.info("Pipeline execution finished")
        logger.info(f"Log file: {log_file_path}")
        logger.info("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Random Forest on pre-generated features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect latest experiment
  python rf_trainer.py

  # Specify experiment path
  python rf_trainer.py --features data/btcusdt-futures-um/output/experiment_20260108_221845/features/

  # Use custom config
  python rf_trainer.py --config my_config.yaml
        """
    )
    parser.add_argument('--features', '-f', type=str, default=None,
                        help='Path to features directory. If not provided, auto-detects latest experiment.')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to config.yaml (optional)')

    args = parser.parse_args()

    try:
        logger.info("Starting Random Forest Trainer execution")
        if args.features:
            logger.info(f"Using provided features path: {args.features}")
        else:
            logger.info("No features path provided, will auto-detect latest experiment")

        results = main(args.features, args.config)
        logger.info("\nTraining completed successfully!")
        logger.info(f"Model saved to: {results['model_dir']}")
        logger.info(f"Plots saved to: {results['plot_dir']}")
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}")
        sys.exit(1)
