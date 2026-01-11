#!/usr/bin/env python
# coding: utf-8

"""
Random Forest Classifier Search for Trading Signals
===================================================

Optimized pipeline for hyperparameter search and training of Random Forest
classifiers for financial market prediction using López de Prado's methodology.

This script performs:
- Data loading and preprocessing
- Fractional differentiation for stationarity
- AR modeling with multicollinearity treatment
- CUSUM event detection
- Triple barrier labeling
- Feature engineering (microstructure + entropy)
- Random Forest training with Optuna hyperparameter optimization
- Feature importance analysis
- Visualization generation
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================

import os
import sys
import gc
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import yaml

# Add src directory to Python path for module imports
sys.path.insert(0, os.path.dirname(__file__))

def load_config(config_path=None):
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import json
import time
from matplotlib.backends.backend_pdf import PdfPages
import psutil
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, accuracy_score, log_loss
)
from sklearn.model_selection import cross_val_predict

# Optuna for hyperparameter optimization (faster than GridSearchCV)
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice
)

# PurgedKFold for financial time series cross-validation (López de Prado)
from ml_pipeline.cross_validation import PurgedKFold

from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ML Framework imports
from ml_pipeline.models.ml_framework import (
    FractionalDifferentiation, StationarityTester, AutoRegressiveModel,
    TripleBarrierMethod, EventAnalyzer, PerformanceAnalyzer, ResidualAnalyzer,
    DataDownloader, DollarBarsProcessor, TripleBarrierExecutor,
    zscore_normalize, av_error
)

# Feature engineering modules
from ml_pipeline.feature_engineering.entropy_features import EntropyFeatures
from ml_pipeline.feature_engineering.microstructure_features import (
    MicrostructureFeatures, CryptoMicrostructureAnalysis
)
from ml_pipeline.feature_engineering.fast_microstructure import FastMicrostructureFeatures
from ml_pipeline.feature_engineering.unified_microstructure_features import (
    UnifiedMicrostructureFeatures, micro_features_unified
)

# Model modules
from ml_pipeline.models.improved_ar_model import ImprovedAutoRegressiveModel
from ml_pipeline.models.ar_multicollinearity_solutions import ARMulticollinearityTreatment

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Set up logging configuration for the optimized pipeline"""
    # Create logs directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file with timestamp
    log_filename = f"search_rf_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    # Configure logger for this module
    logger = logging.getLogger('search_rf_classifier')
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear existing handlers to prevent duplicates
    logger.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Less verbose on console

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log initial message
    logger.info(f"="*70)
    logger.info(f"Main Optimized Pipeline Started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"="*70)

    return logger, log_path

# Initialize logger
logger, log_file_path = setup_logging()

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Memory settings
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 8000  # 8GB limit

logger.info("✅ All imports loaded successfully!")
logger.info("   - Corwin-Schultz spread estimator integrated in UnifiedMicrostructureFeatures")


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
    logger.info(f"💾 Memory after {operation_name}: {memory_mb:.1f} MB")
    logger.debug(f"Detailed memory usage - RSS: {memory_mb:.2f} MB, Operation: {operation_name}")

    if memory_mb > MAX_MEMORY_MB:
        logger.warning(f"⚠️ WARNING: Memory usage exceeds limit ({MAX_MEMORY_MB} MB)")
        gc.collect()
        new_memory = get_memory_usage()
        logger.info(f"   After garbage collection: {new_memory:.1f} MB")
        logger.debug(f"Memory freed by GC: {memory_mb - new_memory:.2f} MB")


def add_feature(data_set, series_fraq, column):
    """Add fractional feature aligned to dataset"""
    data_set[column] = series_fraq
    shift_data = len(data_set) - len(series_fraq)
    data_set[column] = data_set[column].shift(shift_data)
    return data_set


def featureEngineering(series):
    """Calculate microstructure features using optimized implementations."""
    logger.info("🔬 OPTIMIZED MICROSTRUCTURE ANALYSIS")
    start_time = time.time()

    # Prepare data
    data_dict = {
        'BTCUSDT': series[['fraq_close', 'total_volume', 'total_volume_usd']].copy()
    }
    logger.debug(f"Data prepared for BTCUSDT: {len(series)} records")

    # Rename columns
    data_dict['BTCUSDT'].rename(columns={
        'total_volume': 'volume',
        'total_volume_usd': 'quoteVolume'
    }, inplace=True)

    # Create analyzer
    micro_analyzer = CryptoMicrostructureAnalysis(
        symbols=['BTCUSDT'],
        windows=[50, 100]
    )
    logger.debug("CryptoMicrostructureAnalysis initialized with windows=[50, 100]")

    # Calculate features
    micro_analyzer.load_data(data_dict)
    logger.info("Calculating optimized microstructure features...")
    micro_analyzer.calculate_all_features()

    # Get features
    micro_features = micro_analyzer.features['BTCUSDT']
    logger.info(f"✅ Features calculated: {len(micro_features.columns)} features")
    logger.debug(f"Feature names: {list(micro_features.columns)}")

    # Concatenate
    series = pd.concat([series, micro_features], axis=1)

    elapsed_time = time.time() - start_time
    logger.info(f"Microstructure analysis completed in {elapsed_time:.2f} seconds")

    return series


# ============================================================================
# 3. MAIN PIPELINE
# ============================================================================

def main(config_path=None):
    """Main analysis pipeline with all optimizations"""

    pipeline_start_time = time.time()

    # Load configuration
    config = load_config(config_path)

    logger.info("\n" + "="*70)
    logger.info("🚀 STARTING OPTIMIZED TRADING ANALYSIS PIPELINE")
    logger.info("="*70)
    logger.info(f"Process PID: {os.getpid()}")
    logger.info(f"Initial memory: {get_memory_usage():.1f} MB")

    try:
        # ========================================================================
        # STEP 1: DATA LOADING AND PREPARATION
        # ========================================================================

        logger.info("\n📁 STEP 1: DATA LOADING AND PREPARATION")
        step_start = time.time()

        # Load configuration from YAML
        DATA_PATH = config['data']['data_path']
        BASE_OUTPUT_PATH = config['data']['output_path']
        FILE_NAME = config['data']['file_name']

        logger.info(f"📋 Configuration loaded from config.yaml")
        logger.info(f"   Data path: {DATA_PATH}")
        logger.info(f"   File: {FILE_NAME}")

        # Create experiment folder with timestamp
        experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_folder = f"experiment_{experiment_timestamp}"
        OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, experiment_folder)

        logger.info(f"Experiment folder: {experiment_folder}")
        logger.debug(f"Configuration: DATA_PATH={DATA_PATH}, OUTPUT_PATH={OUTPUT_PATH}")

        # Create output directory for this specific sampling
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        logger.info(f"Output directory created for this sampling: {OUTPUT_PATH}")

        # Save experiment metadata (including full config)
        experiment_metadata = {
            'experiment_id': experiment_folder,
            'experiment_timestamp': experiment_timestamp,
            'source_file': FILE_NAME,
            'source_path': DATA_PATH,
            'created_at': datetime.now().isoformat(),
            'pipeline_version': '2.0_optimized',
            'config': config  # Save full configuration for reproducibility
        }

        metadata_path = os.path.join(OUTPUT_PATH, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        logger.info(f"Experiment metadata saved to: {metadata_path}")

        # Load data - check if it's in a folder or directly in the path
        # First try to find the file in a folder with the same name (new structure)
        file_base_name = FILE_NAME.replace('.parquet', '')
        folder_path = os.path.join(DATA_PATH, file_base_name)
        direct_path = os.path.join(DATA_PATH, FILE_NAME)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # New structure: file is inside a folder
            filepath = os.path.join(folder_path, FILE_NAME)
            logger.info(f"Found file in folder structure: {folder_path}/")
        elif os.path.exists(direct_path):
            # Old structure: file is directly in the output folder
            filepath = direct_path
            logger.info(f"Found file in direct path (legacy structure)")
        else:
            logger.error(f"File not found in either structure!")
            logger.error(f"  Tried folder: {folder_path}")
            logger.error(f"  Tried direct: {direct_path}")
            raise FileNotFoundError(f"Could not find {FILE_NAME}")

        logger.info(f"Loading data from {filepath}...")

        try:
            series = pd.read_parquet(filepath)
            logger.info(f"✅ Data loaded: {len(series):,} records")
            logger.debug(f"Data columns: {list(series.columns)}")
            logger.debug(f"Data shape: {series.shape}")
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {str(e)}")
            raise

        log_memory("data loading")
        logger.info(f"Step 1 completed in {time.time() - step_start:.2f} seconds")

        # Aggregation and initial processing
        logger.info("Performing data aggregation...")
        agg_rules = {
            'end_time': 'first',
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'total_volume': 'sum',
            'total_volume_usd': 'sum',
            'total_volume_buy_usd': 'sum',
            'imbalance_col': 'sum',
            'params': 'first',
            'time_trial': 'first'
        }

        # Filter existing columns
        existing_columns = series.columns
        agg_rules_filtered = {col: func for col, func in agg_rules.items() if col in existing_columns}
        logger.debug(f"Aggregation rules applied: {list(agg_rules_filtered.keys())}")

        # Aggregate
        series = series.groupby('end_time').agg(agg_rules_filtered)
        series = series[:-1]  # Remove incomplete last row

        # Add log-prices
        series['log_close'] = np.log(series['close'])

        logger.info(f"✅ Data aggregated: {len(series):,} records")
        log_memory("data aggregation")

        # ========================================================================
        # STEP 2: DATA SPLITTING
        # ========================================================================

        logger.info("\n📊 STEP 2: DATA SPLITTING")
        step_start = time.time()

        # Split train/validation/test
        train_pct = 0.6
        val_pct = 0.25
        test_pct = 1 - train_pct - val_pct

        n_samples = len(series)
        train_end = int(n_samples * train_pct)
        val_end = int(n_samples * (train_pct + val_pct))

        train_dataset = series.iloc[:train_end]
        val_dataset = series.iloc[train_end:val_end]
        test_dataset = series.iloc[val_end:]

        logger.info(f"Train: {len(train_dataset):,} ({train_pct*100:.0f}%)")
        logger.info(f"Validation: {len(val_dataset):,} ({val_pct*100:.0f}%)")
        logger.info(f"Test: {len(test_dataset):,} ({test_pct*100:.0f}%)")

        # Create copies
        train_dataset_copy = train_dataset.copy().reset_index(drop=True)
        val_dataset_copy = val_dataset.copy().reset_index(drop=True)
        test_dataset_copy = test_dataset.copy().reset_index(drop=True)

        log_memory("data splitting")
        logger.info(f"Step 2 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 3: FRACTIONAL DIFFERENTIATION
        # ========================================================================

        logger.info("\n🔄 STEP 3: FRACTIONAL DIFFERENTIATION")
        step_start = time.time()

        # Grid of values - Two-stage search for optimal d
        # Stage 1: Coarse search
        d_values_coarse = np.round(np.arange(0.1, 1.01, 0.1), 2)  # Coarse grid: 0.1, 0.2, ..., 1.0
        conf = 0.05
        thresh = 1e-3
        column = 'log_close'

        logger.info("🔍 Two-stage search for optimal d:")
        logger.info(f"Stage 1 - Coarse search: {d_values_coarse.tolist()}")
        logger.info(f"Confidence level: {conf}, Threshold: {thresh}")

        # Find optimal d - looking for the MINIMUM d that achieves stationarity
        d_found = None
        series_frac_diff_IS = None
        series_frac_diff_OOS = None
        series_frac_diff_OOB = None

        # Stage 1: Coarse search
        logger.info("Stage 1: Coarse search...")
        d_coarse = None

        for d in d_values_coarse:
            logger.debug(f'Testing d={d} (coarse)')
            series_train = np.array(train_dataset_copy[column])
            series_frac_diff_temp = FractionalDifferentiation.frac_diff_optimized(
                series_train, d).dropna().reset_index(drop=True)

            # Limit size for ADF test - REDUCED for speed
            len_max = 100_000  # Reduced from 500k to 100k for faster testing
            if len(series_frac_diff_temp) > len_max:
                series_frac_diff_reduced = series_frac_diff_temp[-len_max:]
            else:
                series_frac_diff_reduced = series_frac_diff_temp

            # ADF test
            try:
                train = StationarityTester.adf_test(series_frac_diff_reduced,
                                                   title=f'Fractional Differentiation d={d}')

                critical_value = train['critical_values'][f"{conf * 100:.0f}%"]
                adf_statistic = train['statistic']

                # Log test results for debugging
                logger.debug(f'd={d}: ADF stat={adf_statistic:.4f}, p-value={train["pvalue"]:.4f}, critical={critical_value:.4f}')

                # Check if series is stationary
                is_stationary = (train['pvalue'] < conf) and (adf_statistic < critical_value)

                if is_stationary:
                    # Found a d that achieves stationarity in coarse search
                    d_coarse = d
                    logger.info(f'✅ Coarse search found: d = {d}')
                    break  # Stop at the first (minimum) d that works
                else:
                    logger.debug(f'd={d}: Not stationary (p-value={train["pvalue"]:.4f})')

            except Exception as e:
                logger.warning(f'Error testing d={d}: {e}')
                continue

        # Stage 2: Fine search around the coarse value
        if d_coarse is not None:
            logger.info(f"\nStage 2: Fine search around d={d_coarse}")

            # Define fine search range
            if d_coarse > 0.1:
                # Search from (d_coarse - 0.1) to d_coarse in small steps
                d_min = max(0.01, d_coarse - 0.1)
                d_max = d_coarse
                d_values_fine = np.round(np.arange(d_min, d_max + 0.01, 0.02), 3)
                logger.info(f"Fine search range: {d_min:.2f} to {d_max:.2f} (step=0.02)")
            else:
                # If d_coarse is 0.1, search from 0.01 to 0.1
                d_values_fine = np.round(np.arange(0.01, d_coarse + 0.01, 0.02), 3)
                logger.info(f"Fine search range: 0.01 to {d_coarse:.2f} (step=0.02)")

            # Fine search
            for d in d_values_fine:
                logger.debug(f'Testing d={d} (fine)')
                series_train = np.array(train_dataset_copy[column])
                series_frac_diff_temp = FractionalDifferentiation.frac_diff_optimized(
                    series_train, d).dropna().reset_index(drop=True)

                # Limit size for ADF test
                len_max = 100_000
                if len(series_frac_diff_temp) > len_max:
                    series_frac_diff_reduced = series_frac_diff_temp[-len_max:]
                else:
                    series_frac_diff_reduced = series_frac_diff_temp

                # ADF test
                try:
                    train = StationarityTester.adf_test(series_frac_diff_reduced,
                                                       title=f'Fractional Differentiation d={d}')

                    critical_value = train['critical_values'][f"{conf * 100:.0f}%"]
                    adf_statistic = train['statistic']

                    # Log test results for debugging
                    logger.debug(f'd={d}: ADF stat={adf_statistic:.4f}, p-value={train["pvalue"]:.4f}, critical={critical_value:.4f}')

                    # Check if series is stationary
                    is_stationary = (train['pvalue'] < conf) and (adf_statistic < critical_value)

                    if is_stationary:
                        # Found the optimal d in fine search
                        series_frac_diff_IS = series_frac_diff_temp

                        # Apply to other sets
                        series_val = np.array(val_dataset_copy[column])
                        series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                            series_val, d).dropna().reset_index(drop=True)

                        series_test = np.array(test_dataset_copy[column])
                        series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                            series_test, d).dropna().reset_index(drop=True)

                        logger.info(f'✅ OPTIMAL d FOUND (refined): {column} - d = {d:.3f} - Stationary')
                        logger.info(f'   ADF statistic: {adf_statistic:.4f}')
                        logger.info(f'   p-value: {train["pvalue"]:.6f}')
                        logger.info(f'   Critical value (5%): {critical_value:.4f}')
                        logger.info(f'   Improvement from coarse: {d_coarse - d:.3f}')
                        d_found = d
                        break  # Stop at the first (minimum) d that works

                except Exception as e:
                    logger.warning(f'Error testing d={d} (fine): {e}')
                    continue

            # If fine search didn't find anything, use coarse result
            if d_found is None:
                d_found = d_coarse
                logger.info(f"Using coarse result: d = {d_found}")

                # Apply differentiation with coarse d
                series_train = np.array(train_dataset_copy[column])
                series_frac_diff_IS = FractionalDifferentiation.frac_diff_optimized(
                    series_train, d_found).dropna().reset_index(drop=True)

                series_val = np.array(val_dataset_copy[column])
                series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                    series_val, d_found).dropna().reset_index(drop=True)

                series_test = np.array(test_dataset_copy[column])
                series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                    series_test, d_found).dropna().reset_index(drop=True)

        # Check if we found a suitable d
        if d_found is None:
            logger.error("❌ No suitable d found for stationarity!")
            logger.info("Using d=1.0 as fallback (full differentiation)")
            d_found = 1.0

            # Apply full differentiation as fallback
            series_train = np.array(train_dataset_copy[column])
            series_frac_diff_IS = FractionalDifferentiation.frac_diff_optimized(
                series_train, d_found).dropna().reset_index(drop=True)

            series_val = np.array(val_dataset_copy[column])
            series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                series_val, d_found).dropna().reset_index(drop=True)

            series_test = np.array(test_dataset_copy[column])
            series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                series_test, d_found).dropna().reset_index(drop=True)

        # Add frac_diff to datasets
        column = 'fraq_close'
        train_dataset_diff = add_feature(train_dataset_copy.copy(), series_frac_diff_IS, column)
        val_dataset_diff = add_feature(val_dataset_copy.copy(), series_frac_diff_OOS, column)
        test_dataset_diff = add_feature(test_dataset_copy.copy(), series_frac_diff_OOB, column)

        log_memory("fractional differentiation")
        logger.info(f"Step 3 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 4: AR MODEL WITH MULTICOLLINEARITY TREATMENT
        # ========================================================================

        logger.info("\n🔍 STEP 4: AR MODEL WITH MULTICOLLINEARITY TREATMENT")
        logger.info("="*65)
        step_start = time.time()

        # Use improved AR model
        improved_ar = ImprovedAutoRegressiveModel(series_frac_diff_IS)

        # Select order with multicollinearity check
        # IMPORTANT: Limit p_max to avoid computational explosion
        # Testing AR models with order > 100 is computationally intensive and rarely useful
        p_max_calculated = round(len(series_frac_diff_IS)**(1/2))

        # Progressive limits based on data size
        if p_max_calculated > 200:
            p_max = 50  # Very large datasets: use conservative limit
            logger.warning(f"⚠️ Very large dataset detected (p_max would be {p_max_calculated})")
        elif p_max_calculated > 100:
            p_max = 75  # Large datasets: moderate limit
        else:
            p_max = p_max_calculated  # Small datasets: use calculated value

        logger.info(f"p_max calculated: {p_max_calculated}, using: {p_max} for performance")

        p_otimo, metrics = AutoRegressiveModel.select_ar_order(
        series_frac_diff_IS,
        p_max=p_max,
        criterio='other',
        limiar=0.01,
        limiar_pvalor=conf,
        min_reducao_absoluta=0.01
        )

        logger.info(f"\n✅ Optimal order selected: p = {p_otimo}")

        # Fit model with automatic treatment
        ar_results = improved_ar.fit_with_multicollinearity_treatment(p_otimo, treatment_method='auto')

        # Extract results
        Y_pred_IS = ar_results['y_pred']
        constant = ar_results['constant']
        params_ = ar_results['params']
        residuals_IS = ar_results['residuals']
        mae_IS = ar_results['metrics']['mae']
        rmse_IS = ar_results['metrics']['rmse']
        r2_IS = ar_results['metrics']['r2']

        logger.info(f"\n✅ Model fitted: {ar_results['method'].upper()}")
        logger.info(f"✅ Final order: AR({ar_results['p']})")
        logger.info(f"✅ In-sample metrics: MAE={mae_IS:.6f}, RMSE={rmse_IS:.6f}, R²={r2_IS:.4f}")

        # Out-of-sample predictions
        Y_pred_OOS = improved_ar.monteCarlo_improved(series_frac_diff_OOS, ar_results)
        residuals_OOS = series_frac_diff_OOS - Y_pred_OOS['fraqdiff_pred']

        Y_pred_OOB = improved_ar.monteCarlo_improved(series_frac_diff_OOB, ar_results)
        residuals_OOB = series_frac_diff_OOB - Y_pred_OOB['fraqdiff_pred']

        log_memory("AR model optimization")

        # ========================================================================
        # STEP 5: EVENT DETECTION (CUSUM)
        # ========================================================================

        logger.info("\n🎯 STEP 5: EVENT DETECTION (CUSUM)")

        # Parameters from config
        tau1 = config['cusum']['tau1']
        tau2 = config['cusum']['tau2']
        sTime = config['triple_barrier']['time_horizon']
        ptSl = [config['triple_barrier']['profit_take'], config['triple_barrier']['stop_loss']]

        logger.info(f"   Triple Barrier: ptSl={ptSl}, sTime={sTime} min")
        logger.info(f"   CUSUM: tau1={tau1}, tau2={tau2}")

        # Prepare data
        series_prim = train_dataset_diff.copy()
        n_lost_fracdiff = len(series_prim) - len(series_frac_diff_IS)

        # Volatility
        span0 = config['volatility']['span']
        vol = EventAnalyzer.getVol(series_prim['fraq_close'], span0=span0)
        series_prim['vol'] = vol

        # Align residuals
        start_idx = n_lost_fracdiff + ar_results['p']
        aligned_df = series_prim.iloc[start_idx:start_idx+len(residuals_IS)].copy()
        # residuals_IS is already a numpy array, no need for .values
        aligned_df['residuals'] = residuals_IS if isinstance(residuals_IS, np.ndarray) else residuals_IS.values
        aligned_df = aligned_df.dropna(subset=['vol', 'residuals'])

        logger.info(f"Data after alignment: {len(aligned_df)} records")

        # Detect events
        series_prim_events = aligned_df.set_index('end_time')
        events = EventAnalyzer.getTEvents(
        series_prim_events['residuals'],
        series_prim_events['vol'],
        tau1
        )

        if len(events) == 0:
            logger.info("⚠️ No events detected. Adjusting threshold...")
            tau1 = tau1 / 10
            events = EventAnalyzer.getTEvents(
                series_prim_events['residuals'],
                series_prim_events['vol'],
                tau1
            )

        events.columns = ['time', 'trgt', 'side']
        events.set_index('time', inplace=True)
        events['t1'] = events.index + pd.Timedelta(minutes=sTime)
        events['trgt'] = events['trgt'] / tau1 * tau2

        logger.info(f"\n✅ Events detected: {len(events)}")
        events_IS = events.copy()

        log_memory("event detection")

        # ========================================================================
        # STEP 6: TRIPLE BARRIER METHOD
        # ========================================================================

        logger.info("\n🎯 STEP 6: TRIPLE BARRIER METHOD")

        if len(events) > 0:
            close = series_prim_events['close']
            events['t1'] = events['t1'].where(events['t1'] <= close.index[-1], pd.NaT)

            molecules = events.index.tolist()

            # Process with parallelization
            triple_barrier_events = Parallel(n_jobs=1)(
                delayed(TripleBarrierMethod.applyPtSlOnT1)(
                    close, events.loc[[molecule]], ptSl, [molecule]
                ) for molecule in molecules
            )

            triple_barrier_events = pd.concat(triple_barrier_events)
            triple_barrier_events['label'] = triple_barrier_events.apply(
                TripleBarrierMethod.get_label, axis=1)
            triple_barrier_events['meta_label'] = np.where(
                triple_barrier_events['label'] == 1, 1, 0)

            logger.info(f"\n✅ Results:")
            logger.info(f"   Total: {len(triple_barrier_events)}")
            logger.info(f"   Take Profits: {(triple_barrier_events['label'] == 1).sum()}")
            logger.info(f"   Stop Losses: {(triple_barrier_events['label'] == -1).sum()}")
            logger.info(f"   Time exits: {(triple_barrier_events['label'] == 0).sum()}")

        triple_barrier_events_IS = triple_barrier_events.copy()
        log_memory("triple barrier")

        # ========================================================================
        # STEP 7: FEATURE ENGINEERING (WITH CORWIN-SCHULTZ)
        # ========================================================================

        logger.info("\n🔬 STEP 7: ULTRA-OPTIMIZED FEATURE ENGINEERING")
        logger.info("="*60)

        # Configuration
        windows = list(range(50, 500, 50))
        sl_range = (10, 200, 10)  # Range for Corwin-Schultz spread estimator

        train_dataset_builded = train_dataset_diff.copy()

        logger.info("🔄 Calculating microstructure features with Corwin-Schultz spread estimator...")
        logger.info("   ✅ VPIN using actual buy/sell volumes")
        logger.info("   ✅ OIR (Order Imbalance Ratio) included")

        # Calculate ALL microstructure features
        calculator = UnifiedMicrostructureFeatures()
        train_dataset_builded = calculator.calculate_all_features(
        train_dataset_builded,
        windows=windows,           # Windows for base features
        sl_range=sl_range,        # Range for Corwin-Schultz spread estimator
        skip_stationarity=True    # Skip tests for maximum speed
        )

        # Verify results
        logger.info("\n📊 Features calculated:")

        # Show Corwin-Schultz spread features
        spread_cols = [col for col in train_dataset_builded.columns if 'corwin_schultz' in col]
        logger.info(f"   - Corwin-Schultz spread features: {len(spread_cols)}")

        # Show VPIN features
        vpin_cols = [col for col in train_dataset_builded.columns if 'vpin_fixed' in col]
        logger.info(f"   - VPIN fixed features: {len(vpin_cols)}")

        # Show OIR features
        oir_cols = [col for col in train_dataset_builded.columns if 'oir' in col]
        logger.info(f"   - OIR features: {len(oir_cols)}")

        # Verify Corwin-Schultz spread values
        if spread_cols:
            logger.info("\n📊 Corwin-Schultz spread verification (first 3):")
            for col in spread_cols[:3]:
                valid_data = train_dataset_builded[col].dropna()
                if len(valid_data) > 0:
                    zeros = (valid_data == 0).sum()
                    logger.info(f"   {col}: mean={valid_data.mean():.6f}, zeros={zeros/len(valid_data)*100:.1f}%")

        logger.info(f"\n✅ Total microstructure features: {len(calculator.feature_names)}")

        # Entropy features
        logger.info("\n📊 Calculating entropy features...")
        window_sizes = list(range(50, 500, 50))
        bins_list = list(range(25, 250, 25))

        entropy_results = EntropyFeatures.calculate_entropy_features_batch(
        series_frac_diff_IS, window_sizes, bins_list
        )
        logger.info(f"✅ Entropy features: {len(entropy_results.columns)} combinations")

        # Add entropy features
        for col in entropy_results.columns:
            train_dataset_builded[col] = entropy_results[col]

        # Move NaNs
        for col in entropy_results.columns:
            train_dataset_builded[col] = EntropyFeatures.move_nans_to_front(train_dataset_builded[col])

        # ========================================================================
        # STEP 7B: LAGGED FEATURES
        # ========================================================================

        logger.info("\n📊 STEP 7B: GENERATING LAGGED FEATURES")

        # Load from config
        N_LAGS = config['features']['n_lags']

        # Identificar colunas para criar lags (features de microestrutura e entropia)
        feature_columns = [col for col in train_dataset_builded.columns
                           if any(x in col.lower() for x in [
                               'corwin_schultz', 'vpin', 'oir', 'volatility',
                               'becker_parkinson', 'amihud', 'kyle_lambda',
                               'roll_spread', 'entropy'
                           ])]

        logger.info(f"   Features base: {len(feature_columns)}")
        logger.info(f"   Lags a criar: {N_LAGS}")

        # Criar features defasadas
        lag_count = 0
        for col in feature_columns:
            for lag in range(1, N_LAGS + 1):
                train_dataset_builded[f'{col}_lag_{lag}'] = train_dataset_builded[col].shift(lag)
                lag_count += 1

        logger.info(f"   ✅ Features com lag criadas: {lag_count}")
        logger.info(f"   ✅ Total de features: {len(train_dataset_builded.columns)}")

        log_memory("lagged features")

        # Clean and prepare final dataset
        original_size = len(train_dataset_builded)
        train_dataset_builded = train_dataset_builded.dropna().reset_index(drop=True)
        final_size = len(train_dataset_builded)
        logger.info(f"\n🧹 Cleanup: {original_size:,} → {final_size:,} samples")

        train_dataset_builded = train_dataset_builded.set_index('end_time')

        # Merge with events
        temp_merged_df = pd.merge(
        triple_barrier_events_IS,
        train_dataset_builded,
        how='left',
        left_index=True,
        right_on='end_time'
        )

        temp_merged_df.index = temp_merged_df['end_time']
        temp_merged_df_filt = temp_merged_df[temp_merged_df['close'].notna()]
        remove_IS = len(temp_merged_df) - len(temp_merged_df_filt)

        # Final dataset
        final_dataset_IS = pd.concat([triple_barrier_events_IS, train_dataset_builded], axis=1)
        final_dataset_IS = final_dataset_IS[final_dataset_IS['meta_label'].notna()]
        final_dataset_IS = final_dataset_IS[remove_IS:]

        y_train = final_dataset_IS[['meta_label']]

        # Check class distribution
        class_distribution = y_train['meta_label'].value_counts()
        class_ratio = class_distribution[0] / class_distribution[1] if 1 in class_distribution.index else float('inf')

        logger.info("\n📊 Target Class Distribution:")
        logger.info(f"   Class 0 (Negative): {class_distribution.get(0, 0)} samples ({class_distribution.get(0, 0)/len(y_train)*100:.1f}%)")
        logger.info(f"   Class 1 (Positive): {class_distribution.get(1, 0)} samples ({class_distribution.get(1, 0)/len(y_train)*100:.1f}%)")
        logger.info(f"   Imbalance Ratio: {class_ratio:.2f}:1")

        if class_ratio > 3 or class_ratio < 0.33:
            logger.warning("   ⚠️ Classes are HIGHLY IMBALANCED - custom sample_weight will be applied")
        elif class_ratio > 1.5 or class_ratio < 0.67:
            logger.info("   ⚠️ Classes are MODERATELY IMBALANCED - custom sample_weight will be applied")
        else:
            logger.info("   ✅ Classes are RELATIVELY BALANCED - custom sample_weight will still be applied")

        # Clean columns
        columns_to_drop = ['t1', 'side', 'sl', 'pt', 'retorno', 'max_drawdown_in_trade',
                       'label', 'meta_label', 'open', 'high', 'low', 'close',
                       'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd',
                       'total_volume', 'params', 'time_trial', 'log_close', 'fraq_close']

        # Filter only existing columns
        columns_to_drop = [col for col in columns_to_drop if col in final_dataset_IS.columns]
        final_dataset_IS.drop(columns=columns_to_drop, inplace=True)
        X_train = final_dataset_IS.copy()

        logger.info(f"\n✅ Total features: {len(X_train.columns)}")
        logger.info(f"✅ Samples: {len(X_train):,}")

        # Feature breakdown
        entropy_features = [col for col in X_train.columns if 'entropy' in col.lower()]
        micro_features = [col for col in X_train.columns if any(x in col for x in ['corwin_schultz', 'becker', 'roll', 'amihud', 'vpin', 'oir', 'kyle'])]
        other_features = [col for col in X_train.columns if col not in entropy_features + micro_features]

        logger.info(f"\n📋 Feature breakdown:")
        logger.info(f"   🧠 Entropy: {len(entropy_features)}")
        logger.info(f"   🏗️ Microstructure: {len(micro_features)}")
        logger.info(f"   📊 Others: {len(other_features)}")

        log_memory("feature engineering")

        # ========================================================================
        # STEP 8: SAMPLE WEIGHTS (NUMBA OPTIMIZED)
        # ========================================================================

        logger.info("\n📊 STEP 8: CALCULATING SAMPLE WEIGHTS (NUMBA OPTIMIZED)")

        # Import the optimized sample weights calculator
        from ml_pipeline.models.sample_weights_numba import SampleWeightsCalculator

        # Create calculator instance
        weights_calculator = SampleWeightsCalculator()

        # Calculate weights using Numba-optimized functions with time decay
        weights_results = weights_calculator.calculate_sample_weights(
        triple_barrier_events_IS,
        series_prim,
        events_IS,
        apply_time_decay=True,
        decay_rate=0.999  # Suave: 1 ano=69%, 3 anos=33%, 5 anos=16%
        )

        # Extract results
        # Pure López de Prado: weight = uniqueness × time_decay
        normalized_weights_pure = weights_results['normalized_weights']
        uniqueness_weights = weights_results['uniqueness_weights']
        time_decay_weights = weights_results['time_decay_weights']

        # Display diagnostics
        diagnostics = weights_results['diagnostics']
        logger.info(f"\n📊 Sample Weights Statistics:")
        logger.info(f"   - Calculation time: {diagnostics['calculation_time']:.3f}s")
        logger.info(f"   - Valid events: {diagnostics['valid_events']}/{diagnostics['total_events']}")
        logger.info(f"   - Time decay applied: {diagnostics['time_decay_applied']}")
        if diagnostics['time_decay_applied']:
            logger.info(f"   - Decay rate: {diagnostics['decay_rate']:.3f}")
        logger.info(f"   - Average uniqueness: {diagnostics['avg_uniqueness']:.4f}")

        # Show uniqueness weights - should be in [0,1] per López de Prado
        logger.info(f"\n   Uniqueness weights (information overlap):")
        if 'uniqueness_stats' in diagnostics:
            logger.info(f"   - Range: [{diagnostics['uniqueness_stats']['min']:.4f}, {diagnostics['uniqueness_stats']['max']:.4f}]")
            logger.info(f"   - Mean: {diagnostics['uniqueness_stats']['mean']:.4f}")
            logger.info(f"   - Std: {diagnostics['uniqueness_stats']['std']:.4f}")

            if diagnostics['uniqueness_stats']['max'] > 1.0:
                logger.error("   ❌ Uniqueness weights exceed 1.0 - should be in [0,1]")
            else:
                logger.info("   ✅ Uniqueness weights correctly in [0,1] range")

        # Show final weights (uniqueness × time_decay, NO normalization per López de Prado)
        logger.info(f"\n   Final weights (uniqueness × time_decay):")
        logger.info(f"   - Range: [{diagnostics['final_weight_stats']['min']:.4f}, {diagnostics['final_weight_stats']['max']:.4f}]")
        logger.info(f"   - Mean: {diagnostics['final_weight_stats']['mean']:.4f}")
        logger.info(f"   - Std: {diagnostics['final_weight_stats']['std']:.4f}")
        logger.info("   ✅ Pure López de Prado: weight = uniqueness × time_decay")

        log_memory("sample weights (numba)")

        # ========================================================================
        # STEP 9: MODEL TRAINING WITH OPTUNA HYPERPARAMETER OPTIMIZATION
        # ========================================================================

        logger.info("\n🎯 STEP 9: RANDOM FOREST WITH OPTUNA OPTIMIZATION")
        logger.info("="*60)

        # Configure cross-validation strategy
        # Using PurgedKFold (López de Prado) instead of StratifiedKFold
        # This prevents look-ahead bias by:
        # 1. Not shuffling data (maintains temporal order)
        # 2. Purging training samples that overlap with test labels
        # 3. Adding embargo between train and test sets
        cv_folds = config['optuna']['cv_folds']
        pct_embargo = config['optuna']['pct_embargo']
        cv_strategy = PurgedKFold(
            n_splits=cv_folds,
            t1=triple_barrier_events_IS['t1'],  # Label expiration times
            pct_embargo=pct_embargo
        )

        # Optuna configuration from config
        N_TRIALS = config['optuna']['n_trials']
        optuna_seed = config['optuna']['seed']

        # Random Forest hyperparameter ranges from config
        rf_config = config['random_forest']

        logger.info(f"📊 Optuna Configuration:")
        logger.info(f"   - Sampler: TPE (Tree-structured Parzen Estimator)")
        logger.info(f"   - Pruner: MedianPruner (stops unpromising trials early)")
        logger.info(f"   - Trials: {N_TRIALS}")
        logger.info(f"   - Cross-validation: {cv_folds}-fold PurgedKFold")
        logger.info(f"   - Embargo: {pct_embargo*100:.1f}%")
        logger.info(f"   - Scoring metric: Log Loss (minimize)")

        # Suppress Optuna's verbose output (we'll log our own)
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # Define the objective function for Optuna
        def optuna_objective(trial):
            """Objective function for Optuna hyperparameter optimization."""
            # Define hyperparameters to optimize (ranges from config)
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

            # Create model with suggested parameters
            rf = RandomForestClassifier(**params)

            # Cross-validation with PurgedKFold (López de Prado methodology)
            cv_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.values.ravel()[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.values.ravel()[val_idx]
                weights_fold = normalized_weights_pure[train_idx]

                rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
                y_pred_proba = rf.predict_proba(X_fold_val)[:, 1]
                fold_score = log_loss(y_fold_val, y_pred_proba)
                cv_scores.append(fold_score)

                # Report intermediate score for pruning
                trial.report(np.mean(cv_scores), fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(cv_scores)

        # Create Optuna study
        study = optuna.create_study(
            direction='minimize',  # minimize log_loss
            sampler=TPESampler(seed=optuna_seed),
            pruner=MedianPruner(n_warmup_steps=2)
        )

        # Track progress
        trials_completed = [0]
        best_score_so_far = [float('inf')]

        def optuna_callback(study, trial):
            """Callback to log progress during optimization."""
            trials_completed[0] += 1
            if trial.value is not None and trial.value < best_score_so_far[0]:
                best_score_so_far[0] = trial.value
                logger.info(f"   Trial {trials_completed[0]}/{N_TRIALS}: New best Log Loss = {trial.value:.4f}")
            elif trials_completed[0] % 10 == 0:
                logger.info(f"   Trial {trials_completed[0]}/{N_TRIALS}: Current best = {best_score_so_far[0]:.4f}")

        logger.info("\n🔍 Starting Optuna optimization...")
        logger.info("   (Bayesian search - much faster than GridSearch)")

        start_time = time.time()

        # Run optimization
        study.optimize(
            optuna_objective,
            n_trials=N_TRIALS,
            callbacks=[optuna_callback],
            show_progress_bar=False
        )

        train_time = time.time() - start_time

        # Get best parameters and create final model
        best_params = study.best_params.copy()
        best_params['bootstrap'] = True
        best_params['random_state'] = 42
        best_params['n_jobs'] = -1

        rf_model = RandomForestClassifier(**best_params)
        rf_model.fit(X_train, y_train.values.ravel(), sample_weight=normalized_weights_pure[:len(X_train)])

        logger.info(f"\n✅ Optuna optimization completed in {train_time:.2f} seconds")
        logger.info(f"   ({train_time/60:.2f} minutes)")

        # Display best results
        logger.info(f"\n🏆 Best Cross-Validation Score:")
        logger.info(f"   Log Loss: {study.best_value:.4f}")

        logger.info("\n🏆 Best Hyperparameters found:")
        for key, value in study.best_params.items():
            logger.info(f"   {key}: {value}")

        # Show optimization statistics
        logger.info(f"\n📊 Optimization Statistics:")
        logger.info(f"   - Total trials: {len(study.trials)}")
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
        logger.info(f"   - Completed trials: {len(completed_trials)}")
        logger.info(f"   - Pruned trials: {len(pruned_trials)} (saved time by stopping early)")

        # Show top 3 trials
        logger.info("\n📊 Top 3 Trials:")
        sorted_trials = sorted(completed_trials, key=lambda t: t.value)[:3]
        for rank, trial in enumerate(sorted_trials, 1):
            logger.info(f"\n   Rank #{rank}:")
            logger.info(f"   Log Loss: {trial.value:.4f}")
            if rank > 1:
                for param_key, param_val in trial.params.items():
                    if param_val != study.best_params.get(param_key):
                        logger.info(f"     {param_key}: {param_val}")

        # Store total combinations for compatibility with results summary
        total_combos = N_TRIALS

        # Training predictions (in-sample)
        y_pred_train = rf_model.predict(X_train)
        y_pred_proba_train = rf_model.predict_proba(X_train)[:, 1]

        # Training metrics (in-sample)
        logger.info("\n📊 Training Metrics (In-Sample):")
        logger.info(f"   Accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Precision: {precision_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Recall: {recall_score(y_train, y_pred_train):.4f}")
        logger.info(f"   F1-Score: {f1_score(y_train, y_pred_train):.4f}")
        logger.info(f"   Log Loss: {log_loss(y_train, y_pred_proba_train):.4f}")
        logger.info(f"   AUC-ROC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

        # Cross-validation predictions (out-of-sample)
        logger.info("\n📊 Generating Cross-Validation Predictions (Out-of-Sample)...")

        # Use cross_val_predict to get out-of-sample predictions for each fold
        y_pred_cv = cross_val_predict(
            rf_model, X_train, y_train.values.ravel(),
            cv=cv_strategy, method='predict', n_jobs=-1
        )
        y_pred_proba_cv = cross_val_predict(
            rf_model, X_train, y_train.values.ravel(),
            cv=cv_strategy, method='predict_proba', n_jobs=-1
        )[:, 1]

        # Validation metrics (out-of-sample via CV)
        logger.info("\n📊 Validation Metrics (Out-of-Sample via PurgedKFold):")
        logger.info(f"   Accuracy: {accuracy_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Precision: {precision_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Recall: {recall_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   F1-Score: {f1_score(y_train, y_pred_cv):.4f}")
        logger.info(f"   Log Loss: {log_loss(y_train, y_pred_proba_cv):.4f}")
        logger.info(f"   AUC-ROC: {roc_auc_score(y_train, y_pred_proba_cv):.4f}")

        # Compare train vs validation
        train_auc = roc_auc_score(y_train, y_pred_proba_train)
        val_auc = roc_auc_score(y_train, y_pred_proba_cv)
        train_ll = log_loss(y_train, y_pred_proba_train)
        val_ll = log_loss(y_train, y_pred_proba_cv)
        logger.info(f"\n📊 Overfitting Analysis:")
        logger.info(f"   Train AUC: {train_auc:.4f}  |  Train Log Loss: {train_ll:.4f}")
        logger.info(f"   Val AUC:   {val_auc:.4f}  |  Val Log Loss:   {val_ll:.4f}")
        logger.info(f"   AUC Gap:   {train_auc - val_auc:.4f} ({(train_auc - val_auc)/val_auc*100:.1f}%)")
        logger.info(f"   Log Loss Gap: {val_ll - train_ll:.4f} (lower is better, val should be close to train)")

        log_memory("model training")

        # ========================================================================
        # STEP 10: FEATURE IMPORTANCE ANALYSIS
        # ========================================================================

        logger.info("\n📊 STEP 10: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)

        # Get feature importances
        feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # Top 20 features
        logger.info("🏆 Top 20 Most Important Features:")
        logger.info("-"*50)
        for idx, row in feature_importance.head(20).iterrows():
            logger.info(f"{row['feature']:<40} {row['importance']:.6f}")

        # Analysis by category
        logger.info("\n📊 Feature Importance by Category:")
        logger.info("-"*50)

        # Categorize features
        categories = {
        'Entropy': entropy_features,
        'Microstructure': micro_features,
        'Volatility': [f for f in X_train.columns if 'vol' in f.lower() and f not in micro_features],
        'Others': other_features
        }

        for cat_name, cat_features in categories.items():
            cat_importance = feature_importance[feature_importance['feature'].isin(cat_features)]['importance'].sum()
        logger.info(f"{cat_name:<20} {cat_importance:.4f} ({cat_importance*100:.2f}%)")

        # ========================================================================
        # STEP 10B: FEATURE SELECTION AND RETRAINING (Optional)
        # ========================================================================

        # Track feature selection results
        feature_selection_info = {
            'enabled': False,
            'used_selected_model': False,
            'original_features': len(X_train.columns),
            'selected_features': len(X_train.columns),
            'auc_improvement': 0.0,
            'log_loss_improvement': 0.0
        }

        if config.get('feature_selection', {}).get('enabled', False):
            logger.info("\n🎯 STEP 10B: FEATURE SELECTION AND RETRAINING")
            logger.info("="*60)

            feature_selection_info['enabled'] = True
            top_n = config['feature_selection']['top_n_features']
            min_importance = config['feature_selection'].get('min_importance', 0)

            # Select top features
            selected_features = feature_importance[
                (feature_importance['importance'] > min_importance)
            ].head(top_n)['feature'].tolist()

            logger.info(f"📊 Feature Selection:")
            logger.info(f"   - Original features: {len(X_train.columns)}")
            logger.info(f"   - Selected features: {len(selected_features)} (top {top_n})")
            logger.info(f"   - Min importance threshold: {min_importance}")

            # Show selected features
            logger.info(f"\n🏆 Top {min(10, len(selected_features))} Selected Features:")
            for i, feat in enumerate(selected_features[:10]):
                imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
                logger.info(f"   {i+1}. {feat}: {imp:.6f}")

            # Create reduced datasets
            X_train_selected = X_train[selected_features]

            logger.info(f"\n🔄 Retraining model with {len(selected_features)} features...")

            # Retrain Optuna with selected features
            def optuna_objective_selected(trial):
                """Objective function with selected features."""
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
                    weights_fold = normalized_weights_pure[train_idx]

                    rf.fit(X_fold_train, y_fold_train, sample_weight=weights_fold)
                    y_pred_proba = rf.predict_proba(X_fold_val)[:, 1]
                    fold_score = log_loss(y_fold_val, y_pred_proba)
                    cv_scores.append(fold_score)

                    trial.report(np.mean(cv_scores), fold_idx)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                return np.mean(cv_scores)

            # Create new study for selected features
            study_selected = optuna.create_study(
                direction='minimize',
                sampler=TPESampler(seed=optuna_seed + 1),  # Different seed
                pruner=MedianPruner(n_warmup_steps=2)
            )

            # Track progress
            trials_completed_sel = [0]
            best_score_sel = [float('inf')]

            def callback_selected(study, trial):
                trials_completed_sel[0] += 1
                if trial.value is not None and trial.value < best_score_sel[0]:
                    best_score_sel[0] = trial.value
                    logger.info(f"   Trial {trials_completed_sel[0]}/{N_TRIALS}: New best = {trial.value:.4f}")
                elif trials_completed_sel[0] % 20 == 0:
                    logger.info(f"   Trial {trials_completed_sel[0]}/{N_TRIALS}: Best so far = {best_score_sel[0]:.4f}")

            start_time_sel = time.time()
            study_selected.optimize(
                optuna_objective_selected,
                n_trials=N_TRIALS,
                callbacks=[callback_selected],
                show_progress_bar=False
            )
            train_time_sel = time.time() - start_time_sel

            # Get best model with selected features
            best_params_sel = study_selected.best_params.copy()
            best_params_sel['bootstrap'] = True
            best_params_sel['random_state'] = optuna_seed
            best_params_sel['n_jobs'] = -1

            rf_model_selected = RandomForestClassifier(**best_params_sel)
            rf_model_selected.fit(X_train_selected, y_train.values.ravel(),
                                  sample_weight=normalized_weights_pure[:len(X_train_selected)])

            # Predictions with selected features
            y_pred_train_sel = rf_model_selected.predict(X_train_selected)
            y_pred_proba_train_sel = rf_model_selected.predict_proba(X_train_selected)[:, 1]

            # CV predictions
            y_pred_cv_sel = cross_val_predict(
                rf_model_selected, X_train_selected, y_train.values.ravel(),
                cv=cv_strategy, method='predict', n_jobs=-1
            )
            y_pred_proba_cv_sel = cross_val_predict(
                rf_model_selected, X_train_selected, y_train.values.ravel(),
                cv=cv_strategy, method='predict_proba', n_jobs=-1
            )[:, 1]

            # Compare results
            logger.info(f"\n✅ Retraining completed in {train_time_sel:.2f}s ({train_time_sel/60:.2f} min)")
            logger.info(f"\n📊 COMPARISON: All Features vs Selected Features")
            logger.info("="*60)
            logger.info(f"{'Metric':<25} {'All ({})'.format(len(X_train.columns)):<20} {'Selected ({})'.format(len(selected_features)):<20}")
            logger.info("-"*60)

            # All features metrics
            auc_all = roc_auc_score(y_train, y_pred_proba_cv)
            ll_all = log_loss(y_train, y_pred_proba_cv)
            f1_all = f1_score(y_train, y_pred_cv)
            acc_all = accuracy_score(y_train, y_pred_cv)

            # Selected features metrics
            auc_sel = roc_auc_score(y_train, y_pred_proba_cv_sel)
            ll_sel = log_loss(y_train, y_pred_proba_cv_sel)
            f1_sel = f1_score(y_train, y_pred_cv_sel)
            acc_sel = accuracy_score(y_train, y_pred_cv_sel)

            logger.info(f"{'AUC-ROC (CV)':<25} {auc_all:<20.4f} {auc_sel:<20.4f}")
            logger.info(f"{'Log Loss (CV)':<25} {ll_all:<20.4f} {ll_sel:<20.4f}")
            logger.info(f"{'F1-Score (CV)':<25} {f1_all:<20.4f} {f1_sel:<20.4f}")
            logger.info(f"{'Accuracy (CV)':<25} {acc_all:<20.4f} {acc_sel:<20.4f}")

            # Highlight improvement
            auc_diff = auc_sel - auc_all
            ll_diff = ll_all - ll_sel  # Lower is better
            logger.info("-"*60)
            logger.info(f"{'AUC Improvement:':<25} {auc_diff:+.4f} ({'✅ Better' if auc_diff > 0 else '❌ Worse'})")
            logger.info(f"{'Log Loss Improvement:':<25} {ll_diff:+.4f} ({'✅ Better' if ll_diff > 0 else '❌ Worse'})")

            # Update feature selection info
            feature_selection_info['selected_features'] = len(selected_features)
            feature_selection_info['auc_improvement'] = float(auc_diff)
            feature_selection_info['log_loss_improvement'] = float(ll_diff)
            feature_selection_info['auc_all_features'] = float(auc_all)
            feature_selection_info['auc_selected_features'] = float(auc_sel)
            feature_selection_info['log_loss_all_features'] = float(ll_all)
            feature_selection_info['log_loss_selected_features'] = float(ll_sel)

            # Update model and predictions for plots if selected is better
            if auc_sel > auc_all:
                logger.info("\n🎉 Selected features model is BETTER! Using it for final results.")
                feature_selection_info['used_selected_model'] = True
                rf_model = rf_model_selected
                y_pred_train = y_pred_train_sel
                y_pred_proba_train = y_pred_proba_train_sel
                y_pred_cv = y_pred_cv_sel
                y_pred_proba_cv = y_pred_proba_cv_sel
                X_train = X_train_selected
                study = study_selected

                # Update feature importance for the selected model
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                logger.info("\n⚠️ Original model is better or equal. Keeping all features.")
                feature_selection_info['used_selected_model'] = False

            # Save selected features list and comparison
            selected_features_path = os.path.join(OUTPUT_PATH, 'feature_selection_results.json')
            with open(selected_features_path, 'w') as f:
                json.dump({
                    'selected_features': selected_features,
                    'feature_selection_info': feature_selection_info,
                    'comparison': {
                        'all_features': {
                            'n_features': feature_selection_info['original_features'],
                            'auc_cv': float(auc_all),
                            'log_loss_cv': float(ll_all),
                            'f1_cv': float(f1_all),
                            'accuracy_cv': float(acc_all)
                        },
                        'selected_features': {
                            'n_features': len(selected_features),
                            'auc_cv': float(auc_sel),
                            'log_loss_cv': float(ll_sel),
                            'f1_cv': float(f1_sel),
                            'accuracy_cv': float(acc_sel)
                        },
                        'winner': 'selected' if auc_sel > auc_all else 'all'
                    }
                }, f, indent=2)
            logger.info(f"💾 Feature selection results saved to: {selected_features_path}")

            log_memory("feature selection")

        # ========================================================================
        # STEP 11: GENERATE VISUALIZATION PLOTS
        # ========================================================================

        logger.info("\n📊 STEP 11: GENERATING VISUALIZATION PLOTS")
        logger.info("="*60)

        # Create plot directory
        plot_dir = os.path.join(OUTPUT_PATH, 'plot')
        os.makedirs(plot_dir, exist_ok=True)
        logger.info(f"Plot directory created: {plot_dir}")

        # Set plot style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. CONFUSION MATRIX (Train vs Validation - Side by Side)
        logger.info("\n1️⃣ Generating Confusion Matrix (Train vs Validation)...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # --- TRAINING Confusion Matrix ---
        cm_train = confusion_matrix(y_train, y_pred_train)
        tn_tr, fp_tr, fn_tr, tp_tr = cm_train.ravel()
        prec_tr = tp_tr / (tp_tr + fp_tr) if (tp_tr + fp_tr) > 0 else 0
        rec_tr = tp_tr / (tp_tr + fn_tr) if (tp_tr + fn_tr) > 0 else 0
        f1_tr = 2 * prec_tr * rec_tr / (prec_tr + rec_tr) if (prec_tr + rec_tr) > 0 else 0

        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', square=True,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[0])
        axes[0].set_title(f'TRAINING (In-Sample)\n'
                          f'Prec={prec_tr:.3f} | Rec={rec_tr:.3f} | F1={f1_tr:.3f}',
                          fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Label', fontsize=11)
        axes[0].set_xlabel('Predicted Label', fontsize=11)

        # --- VALIDATION Confusion Matrix ---
        cm_val = confusion_matrix(y_train, y_pred_cv)
        tn_val, fp_val, fn_val, tp_val = cm_val.ravel()
        prec_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0
        rec_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0
        f1_val = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0

        sns.heatmap(cm_val, annot=True, fmt='d', cmap='Oranges', square=True,
                    xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'], ax=axes[1])
        axes[1].set_title(f'VALIDATION (Out-of-Sample via CV)\n'
                          f'Prec={prec_val:.3f} | Rec={rec_val:.3f} | F1={f1_val:.3f}',
                          fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Label', fontsize=11)
        axes[1].set_xlabel('Predicted Label', fontsize=11)

        # Suptitle com análise de overfitting
        gap = f1_tr - f1_val
        status = "⚠️ OVERFITTING" if gap > 0.1 else "✅ OK" if gap < 0.05 else "⚡ LEVE OVERFIT"
        plt.suptitle(f'Confusion Matrix Comparison | F1 Gap: {gap:.3f} ({status})',
                     fontsize=14, fontweight='bold', y=1.02)

        plt.tight_layout()
        confusion_matrix_path = os.path.join(plot_dir, 'confusion_matrix.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ Saved: {confusion_matrix_path}")
        logger.info(f"   📊 Train: TP={tp_tr}, TN={tn_tr}, FP={fp_tr}, FN={fn_tr} | F1={f1_tr:.4f}")
        logger.info(f"   📊 Val:   TP={tp_val}, TN={tn_val}, FP={fp_val}, FN={fn_val} | F1={f1_val:.4f}")
        logger.info(f"   📊 Gap: {gap:.4f} ({status})")

        # 2. ROC CURVE (Train vs Validation overlaid)
        logger.info("\n2️⃣ Generating ROC Curve (Train vs Validation)...")

        # Training ROC
        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
        auc_train = roc_auc_score(y_train, y_pred_proba_train)

        # Validation ROC (CV out-of-sample)
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
        plt.title(f'ROC Curve Comparison - Random Forest\nOverfitting Gap: {auc_train - auc_val:.4f}',
                  fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add annotation for gap
        plt.annotate(f'Gap: {(auc_train - auc_val)*100:.1f}%',
                     xy=(0.6, 0.3), fontsize=12, color='red',
                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

        plt.tight_layout()
        roc_curve_path = os.path.join(plot_dir, 'roc_curve.png')
        plt.savefig(roc_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ Saved: {roc_curve_path}")

        # 3. TOP 20 BEST FEATURES
        logger.info("\n3️⃣ Generating Top 20 Best Features...")
        top_20_features = feature_importance.head(20)

        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(top_20_features)), top_20_features['importance'], color='forestgreen')
        plt.yticks(range(len(top_20_features)), top_20_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Most Important Features\nRandom Forest', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, top_20_features['importance'])):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        top_20_path = os.path.join(plot_dir, 'top_20_best_features.png')
        plt.savefig(top_20_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ Saved: {top_20_path}")

        # 4. TOP 20 WORST FEATURES
        logger.info("\n4️⃣ Generating Top 20 Worst Features...")
        bottom_20_features = feature_importance.tail(20).sort_values('importance', ascending=True)

        plt.figure(figsize=(12, 10))
        bars = plt.barh(range(len(bottom_20_features)), bottom_20_features['importance'], color='crimson')
        plt.yticks(range(len(bottom_20_features)), bottom_20_features['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title('Top 20 Least Important Features\nRandom Forest', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, bottom_20_features['importance'])):
            plt.text(val, i, f' {val:.6f}', va='center', fontsize=9)

        plt.tight_layout()
        bottom_20_path = os.path.join(plot_dir, 'top_20_worst_features.png')
        plt.savefig(bottom_20_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"   ✅ Saved: {bottom_20_path}")

        # ========================================================================
        # OPTUNA VISUALIZATION PLOTS
        # ========================================================================

        logger.info("\n📊 GENERATING OPTUNA OPTIMIZATION PLOTS...")

        # 5. OPTIMIZATION HISTORY - Shows how the best value improved over trials
        logger.info("\n5️⃣ Generating Optuna Optimization History...")
        try:
            fig = plot_optimization_history(study)
            fig.update_layout(
                title="Optimization History - Log Loss Over Trials",
                xaxis_title="Trial Number",
                yaxis_title="Log Loss (lower is better)",
                template="plotly_white"
            )
            optuna_history_path = os.path.join(plot_dir, 'optuna_optimization_history.html')
            fig.write_html(optuna_history_path)
            # Also save as PNG
            optuna_history_png = os.path.join(plot_dir, 'optuna_optimization_history.png')
            fig.write_image(optuna_history_png, width=1200, height=600, scale=2)
            logger.info(f"   ✅ Saved: {optuna_history_path}")
            logger.info(f"   ✅ Saved: {optuna_history_png}")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not generate optimization history: {e}")

        # 6. PARAMETER IMPORTANCES - Which hyperparameters matter most
        logger.info("\n6️⃣ Generating Optuna Parameter Importances...")
        try:
            fig = plot_param_importances(study)
            fig.update_layout(
                title="Hyperparameter Importances",
                xaxis_title="Importance",
                template="plotly_white"
            )
            optuna_importance_path = os.path.join(plot_dir, 'optuna_param_importances.html')
            fig.write_html(optuna_importance_path)
            optuna_importance_png = os.path.join(plot_dir, 'optuna_param_importances.png')
            fig.write_image(optuna_importance_png, width=1000, height=600, scale=2)
            logger.info(f"   ✅ Saved: {optuna_importance_path}")
            logger.info(f"   ✅ Saved: {optuna_importance_png}")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not generate parameter importances: {e}")

        # 7. PARALLEL COORDINATE PLOT - Relationships between parameters
        logger.info("\n7️⃣ Generating Optuna Parallel Coordinate Plot...")
        try:
            fig = plot_parallel_coordinate(study)
            fig.update_layout(
                title="Parallel Coordinate Plot - Parameter Relationships",
                template="plotly_white"
            )
            optuna_parallel_path = os.path.join(plot_dir, 'optuna_parallel_coordinate.html')
            fig.write_html(optuna_parallel_path)
            optuna_parallel_png = os.path.join(plot_dir, 'optuna_parallel_coordinate.png')
            fig.write_image(optuna_parallel_png, width=1400, height=700, scale=2)
            logger.info(f"   ✅ Saved: {optuna_parallel_path}")
            logger.info(f"   ✅ Saved: {optuna_parallel_png}")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not generate parallel coordinate plot: {e}")

        # 8. SLICE PLOT - How each parameter affects the objective
        logger.info("\n8️⃣ Generating Optuna Slice Plot...")
        try:
            fig = plot_slice(study)
            fig.update_layout(
                title="Slice Plot - Parameter vs Objective Value",
                template="plotly_white"
            )
            optuna_slice_path = os.path.join(plot_dir, 'optuna_slice_plot.html')
            fig.write_html(optuna_slice_path)
            optuna_slice_png = os.path.join(plot_dir, 'optuna_slice_plot.png')
            fig.write_image(optuna_slice_png, width=1400, height=800, scale=2)
            logger.info(f"   ✅ Saved: {optuna_slice_path}")
            logger.info(f"   ✅ Saved: {optuna_slice_png}")
        except Exception as e:
            logger.warning(f"   ⚠️ Could not generate slice plot: {e}")

        logger.info(f"\n✅ All plots saved successfully to: {plot_dir}")

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        logger.info("\n" + "="*70)
        logger.info("🎯 FINAL ANALYSIS SUMMARY")
        logger.info("="*70)

        logger.info("\n📊 DATA PROCESSED:")
        logger.info(f"   - Original dataset: 4,946,183 records")
        logger.info(f"   - Events detected: {len(events_IS):,}")
        logger.info(f"   - Final ML samples: {len(X_train):,}")


        logger.info("\n🎯 MODEL RESULTS:")
        logger.info(f"   - Total features: {len(X_train.columns)}")
        logger.info(f"   - Training accuracy: {accuracy_score(y_train, y_pred_train):.4f}")
        logger.info(f"   - Training F1-Score: {f1_score(y_train, y_pred_train):.4f}")
        logger.info(f"   - Training AUC-ROC: {roc_auc_score(y_train, y_pred_proba_train):.4f}")

        logger.info("\n💡 NEXT STEPS:")
        logger.info("   1. Out-of-sample validation (val and test datasets)")
        logger.info("   2. Feature selection and engineering refinement")
        logger.info("   3. Complete meta-labeling implementation")
        logger.info("   4. Backtesting with transaction costs")
        logger.info("   5. Model deployment to production")

        logger.info("\n⏱️ PERFORMANCE:")
        logger.info(f"   - Training time: {train_time:.2f}s")
        logger.info(f"   - Current memory: {get_memory_usage():.1f} MB")
        logger.info(f"   - Estimated speedup: ~30-50x vs original version")

        logger.info("\n✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("="*70)

        # Save results
        results_summary = {
        'timestamp': datetime.now().isoformat(),
        'data_stats': {
            'original_records': 4946183,
            'events_detected': len(events_IS),
            'final_samples': len(X_train),
            'features': len(X_train.columns)
        },
        'hyperparameter_optimization': {
            'method': 'Optuna',
            'total_combinations_tested': total_combos,
            'cv_folds': 5,
            'scoring_metric': 'neg_log_loss',
            'best_cv_score_log_loss': float(study.best_value),
            'best_params': study.best_params,
            'optimization_time_seconds': train_time
        },
        'model_performance': {
            'accuracy': float(accuracy_score(y_train, y_pred_train)),
            'f1_score': float(f1_score(y_train, y_pred_train)),
            'log_loss': float(log_loss(y_train, y_pred_proba_train)),
            'auc_roc': float(roc_auc_score(y_train, y_pred_proba_train))
        },
        'corwin_schultz_info': {
            'total_spread_features': len(spread_cols),
            'example_features': spread_cols[:3] if spread_cols else [],
            'integrated_in': 'UnifiedMicrostructureFeatures.calculate_all_features()'
        },
        'feature_selection': feature_selection_info
        }

        # Save JSON with results
        with open(os.path.join(OUTPUT_PATH, 'analysis_results_optimized.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)

        logger.info(f"\n💾 Results saved to: {OUTPUT_PATH}/analysis_results_optimized.json")

        # Save the trained model
        import joblib
        model_path = os.path.join(OUTPUT_PATH, 'random_forest_model_optimized.pkl')
        joblib.dump(rf_model, model_path)
        logger.info(f"💾 Model saved to: {model_path}")

        # Save feature importances
        feature_importance_path = os.path.join(OUTPUT_PATH, 'feature_importances.csv')
        feature_importance.to_csv(feature_importance_path, index=False)
        logger.info(f"💾 Feature importances saved to: {feature_importance_path}")

        # Save Optuna study results
        optuna_results_path = os.path.join(OUTPUT_PATH, 'optuna_trials_results.csv')
        trials_df = study.trials_dataframe()
        trials_df.to_csv(optuna_results_path, index=False)
        logger.info(f"💾 Optuna trials results saved to: {optuna_results_path}")

        # Log pipeline completion
        pipeline_elapsed = time.time() - pipeline_start_time
        logger.info(f"\n⏱️ TOTAL PIPELINE TIME: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
        logger.info(f"Final memory usage: {get_memory_usage():.1f} MB")

        # Create README with execution summary
        readme_content = f"""# Experiment Results

## Experiment Information
- **ID:** {experiment_folder}
- **Timestamp:** {experiment_timestamp}
- **Source File:** {FILE_NAME}

## Execution Summary
- **Pipeline Version:** 2.0 Optimized
- **Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Total Time:** {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)
- **Peak Memory:** {get_memory_usage():.1f} MB

## Model Performance
- **Method:** Random Forest with Optuna
- **Best CV Score (Log Loss):** {study.best_value:.4f}
- **Training Accuracy:** {accuracy_score(y_train, y_pred_train):.4f}
- **Training F1-Score:** {f1_score(y_train, y_pred_train):.4f}
- **Training Log Loss:** {log_loss(y_train, y_pred_proba_train):.4f}

## Best Hyperparameters
```python
{json.dumps(study.best_params, indent=2)}
```

## Dataset Statistics
- **Total Events:** {len(events_IS)}
- **Training Samples:** {len(X_train)}
- **Features:** {len(X_train.columns)}
- **Class Distribution:** {class_distribution.to_dict()}

## Files Generated
- `experiment_metadata.json` - Experiment configuration
- `analysis_results_optimized.json` - Complete results (includes feature_selection info)
- `random_forest_model_optimized.pkl` - Trained model (best model: all or selected features)
- `feature_importances.csv` - Feature importance ranking (from final model)
- `optuna_trials_results.csv` - All Optuna trials results
- `feature_selection_results.json` - Comparison: all features vs selected features

## Notes
- López de Prado methodology applied for sample weights
- Corwin-Schultz spread estimator used
- Bootstrap enabled for Random Forest
- Class weights balanced due to imbalance
"""

        readme_path = os.path.join(OUTPUT_PATH, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        logger.info(f"📝 README created at: {readme_path}")

        # Final summary
        logger.info("\n" + "="*70)
        logger.info("📂 ALL RESULTS SAVED TO:")
        logger.info(f"   {OUTPUT_PATH}/")
        logger.info("")
        logger.info("Files in this folder:")
        logger.info("  ├── experiment_metadata.json       - Experiment configuration")
        logger.info("  ├── analysis_results_optimized.json - Complete results + feature selection")
        logger.info("  ├── random_forest_model_optimized.pkl - Best model (all or selected)")
        logger.info("  ├── feature_importances.csv        - Feature ranking (final model)")
        logger.info("  ├── optuna_trials_results.csv      - Optuna trials")
        logger.info("  ├── feature_selection_results.json - All vs Selected comparison")
        logger.info("  └── README.md                      - Execution summary")
        logger.info("="*70)

        return {
            'model': rf_model,
            'X_train': X_train,
            'y_train': y_train,
            'feature_importance': feature_importance,
            'results_summary': results_summary
        }

    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Pipeline execution time before failure: {time.time() - pipeline_start_time:.2f} seconds")
        logger.error(f"Memory at failure: {get_memory_usage():.1f} MB")

        # Save partial results if possible
        try:
            error_summary = {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'error_type': type(e).__name__,
                'execution_time': time.time() - pipeline_start_time,
                'memory_usage_mb': get_memory_usage()
            }

            error_path = os.path.join(OUTPUT_PATH, 'pipeline_error.json')
            with open(error_path, 'w') as f:
                json.dump(error_summary, f, indent=2)
            logger.info(f"Error details saved to: {error_path}")
        except Exception as save_error:
            logger.error(f"Failed to save error details: {save_error}")

        raise  # Re-raise the exception after logging

    finally:
        # Always log pipeline closure
        logger.info("\n" + "="*70)
        logger.info("Pipeline execution finished")
        logger.info(f"Log file saved at: {log_file_path}")
        logger.info("="*70)


if __name__ == "__main__":
    try:
        # Run the main pipeline
        logger.info("Starting Random Forest Classifier Search execution")
        results = main()

        logger.info("\n🎉 Pipeline execution completed successfully!")
        logger.info(f"   Model saved in results['model']")
        logger.info(f"   Feature importance saved in results['feature_importance']")
        logger.info(f"   Full results summary saved in results['results_summary']")

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Pipeline interrupted by user (Ctrl+C)")
        logger.info("Shutting down gracefully...")
        sys.exit(1)

    except Exception as e:
        logger.error("\n❌ Pipeline failed with unexpected error")
        logger.error(f"Error: {str(e)}")
        sys.exit(1)