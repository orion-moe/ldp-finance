#!/usr/bin/env python
# coding: utf-8

"""
Feature Generator for Trading Signals
======================================

Generates features and produces comprehensive EDA (Exploratory Data Analysis)
report for feature exploration before ML training.

This script performs:
- Data loading and preprocessing (Step 1)
- Data splitting (Step 2)
- Fractional differentiation for stationarity (Step 3)
- AR modeling with multicollinearity treatment (Step 4)
- CUSUM event detection (Step 5)
- Triple barrier labeling (Step 6)
- Feature engineering (microstructure + entropy) (Step 7)
- Sample weights calculation (Step 8)
- EDA Report Generation (New)

Output Structure:
    experiment_{timestamp}/
    ├── features/
    │   ├── X_train.parquet
    │   ├── y_train.parquet
    │   ├── sample_weights.parquet
    │   └── feature_names.json
    ├── report/
    │   ├── feature_statistics.csv
    │   ├── correlation_matrix.png
    │   ├── target_distribution.png
    │   ├── class_balance.png
    │   ├── missing_values.png
    │   ├── feature_distributions/
    │   └── feature_eda.html
    └── experiment_metadata.json
"""

# ============================================================================
# 1. IMPORTS AND CONFIGURATION
# ============================================================================

import os
import sys
import gc
import re
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
import psutil
from joblib import Parallel, delayed

# Add src directory to Python path for module imports
sys.path.insert(0, os.path.dirname(__file__))

def load_config(config_path=None):
    """Load experiment configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config

# ML Framework imports
from ml_pipeline.models.ml_framework import (
    FractionalDifferentiation, StationarityTester, AutoRegressiveModel,
    TripleBarrierMethod, EventAnalyzer, zscore_normalize
)

# Feature engineering modules
from ml_pipeline.feature_engineering.entropy_features import EntropyFeatures
from ml_pipeline.feature_engineering.unified_microstructure_features import UnifiedMicrostructureFeatures

# Model modules
from ml_pipeline.models.improved_ar_model import ImprovedAutoRegressiveModel

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging():
    """Set up logging configuration for the feature generator"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_filename = f"feature_generator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    logger = logging.getLogger('feature_generator')
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
    logger.info(f"Feature Generator Started")
    logger.info(f"Log file: {log_path}")
    logger.info(f"="*70)

    return logger, log_path


logger, log_file_path = setup_logging()

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

# Memory settings
ENABLE_MEMORY_OPTIMIZATION = True
MAX_MEMORY_MB = 8000

logger.info("All imports loaded successfully!")


def parse_downsampling_info(file_name: str) -> dict:
    """
    Parse the parquet file name to extract downsampling information.

    Expected format: {date}-{type}-{market}-volume{amount}.parquet
    Example: 20251123-003308-standard-futures-volume200000000.parquet

    Returns dict with:
        - bar_type: "Standard" or "Imbalance"
        - dollar_volume: formatted string like "200M USD"
        - full_description: "Standard Dollar Bars 200M USD"
    """
    file_name_lower = file_name.lower()

    # Detect bar type
    if 'imbalance' in file_name_lower:
        bar_type = 'Imbalance'
    elif 'standard' in file_name_lower:
        bar_type = 'Standard'
    else:
        bar_type = 'Unknown'

    # Extract volume
    volume_match = re.search(r'volume(\d+)', file_name_lower)
    if volume_match:
        volume_raw = int(volume_match.group(1))
        if volume_raw >= 1_000_000_000:
            dollar_volume = f"{volume_raw / 1_000_000_000:.0f}B USD"
        elif volume_raw >= 1_000_000:
            dollar_volume = f"{volume_raw / 1_000_000:.0f}M USD"
        elif volume_raw >= 1_000:
            dollar_volume = f"{volume_raw / 1_000:.0f}K USD"
        else:
            dollar_volume = f"{volume_raw} USD"
    else:
        dollar_volume = 'N/A'

    # Build full description
    if bar_type != 'Unknown' and dollar_volume != 'N/A':
        full_description = f"{bar_type} Dollar Bars {dollar_volume}"
    elif bar_type != 'Unknown':
        full_description = f"{bar_type} Dollar Bars"
    else:
        full_description = 'Unknown Downsampling'

    return {
        'bar_type': bar_type,
        'dollar_volume': dollar_volume,
        'full_description': full_description
    }


# ============================================================================
# FEATURE TAXONOMY - Academic classification with descriptions
# ============================================================================

FEATURE_TAXONOMY = {
    'microstructure': {
        'name': '1. Microstructure Features',
        'description': 'Features derived from market microstructure theory, measuring liquidity, price impact, and information asymmetry.',
        'subcategories': {
            'first_generation': {
                'name': '1.1 First Generation (Liquidity/Price Impact)',
                'description': 'Measures of market liquidity and price impact from early microstructure literature.',
                'features': {
                    'kyle_lambda': {
                        'name': 'Kyle Lambda',
                        'description': 'Price impact coefficient measuring how much prices move per unit of net order flow. Higher values indicate lower liquidity.',
                        'formula': 'λ = Cov(ΔP, V) / Var(V)',
                        'reference': 'Kyle (1985)',
                        'pattern': 'kyle_lambda',
                        'interpretation': 'Higher λ = Lower liquidity, higher price impact'
                    },
                    'amihud': {
                        'name': 'Amihud Illiquidity',
                        'description': 'Ratio of absolute return to dollar volume, capturing the price impact per dollar traded.',
                        'formula': 'ILLIQ = |r| / Volume_USD',
                        'reference': 'Amihud (2002)',
                        'pattern': 'amihud',
                        'interpretation': 'Higher ILLIQ = Less liquid market'
                    },
                    'volatility': {
                        'name': 'Realized Volatility',
                        'description': 'Standard deviation of returns over a rolling window.',
                        'formula': 'σ = √(Σ(r - r̄)² / (n-1))',
                        'reference': 'Standard',
                        'pattern': 'volatility',
                        'interpretation': 'Measures price uncertainty/risk'
                    }
                }
            },
            'second_generation': {
                'name': '1.2 Second Generation (Volatility Estimators)',
                'description': 'Volatility and spread estimators using high-low price ranges.',
                'features': {
                    'becker_parkinson': {
                        'name': 'Becker-Parkinson Volatility',
                        'description': 'High-low range volatility estimator, more efficient than close-to-close estimators.',
                        'formula': 'σ² = (ln(H/L))² / (4·ln(2))',
                        'reference': 'Parkinson (1980)',
                        'pattern': 'becker_parkinson',
                        'interpretation': 'Measures intraday price variation'
                    },
                    'corwin_schultz': {
                        'name': 'Corwin-Schultz Spread',
                        'description': 'Bid-ask spread estimator derived from daily high and low prices.',
                        'formula': 'S = 2(e^α - 1) / (1 + e^α)',
                        'reference': 'Corwin & Schultz (2012)',
                        'pattern': 'corwin_schultz',
                        'interpretation': 'Estimates effective bid-ask spread from OHLC data'
                    },
                    'roll_spread': {
                        'name': 'Roll Spread',
                        'description': 'Implicit spread estimator from negative serial covariance of price changes.',
                        'formula': 'S = 2·√(-Cov(r_t, r_{t-1}))',
                        'reference': 'Roll (1984)',
                        'pattern': 'roll_spread',
                        'interpretation': 'Larger spread = Higher transaction costs'
                    }
                }
            },
            'third_generation': {
                'name': '1.3 Third Generation (Information-based)',
                'description': 'Measures of informed trading and order flow toxicity.',
                'features': {
                    'vpin': {
                        'name': 'VPIN (Volume-Synchronized PIN)',
                        'description': 'Estimates probability of informed trading using volume buckets. High VPIN indicates toxic order flow.',
                        'formula': 'VPIN = Σ|V_buy - V_sell| / (n·V_bucket)',
                        'reference': 'Easley, López de Prado & O\'Hara (2012)',
                        'pattern': 'vpin',
                        'interpretation': 'High VPIN = High probability of informed trading'
                    },
                    'oir': {
                        'name': 'Order Imbalance Ratio',
                        'description': 'Net order flow direction normalized by total volume. Measures buying/selling pressure.',
                        'formula': 'OIR = (V_buy - V_sell) / (V_buy + V_sell)',
                        'reference': 'Chordia, Roll & Subrahmanyam (2002)',
                        'pattern': 'oir',
                        'interpretation': 'Positive = Buy pressure, Negative = Sell pressure'
                    }
                }
            }
        }
    },
    'entropy': {
        'name': '2. Entropy Features',
        'description': 'Information-theoretic measures of market uncertainty and complexity.',
        'subcategories': {
            'shannon': {
                'name': '2.1 Shannon Entropy',
                'description': 'Measures uncertainty in the distribution of price changes.',
                'features': {
                    'entropy': {
                        'name': 'Shannon Entropy',
                        'description': 'Information entropy of price changes distribution. Higher entropy indicates more unpredictable markets.',
                        'formula': 'H = -Σ p(x)·log₂(p(x))',
                        'reference': 'Shannon (1948)',
                        'pattern': 'entropy',
                        'interpretation': 'Higher entropy = More random/unpredictable'
                    }
                }
            },
            'lempel_ziv': {
                'name': '2.2 Lempel-Ziv Complexity',
                'description': 'Measures pattern complexity in price sequences using compression theory.',
                'features': {
                    'lz': {
                        'name': 'Lempel-Ziv Complexity',
                        'description': 'Counts unique patterns in discretized price sequence. Higher values indicate more complex/random behavior, lower values suggest repetitive patterns.',
                        'formula': 'LZ = n_patterns / (n / log₂(n))',
                        'reference': 'Lempel & Ziv (1976)',
                        'pattern': 'lz_',
                        'interpretation': 'Higher LZ = More random/less compressible'
                    }
                }
            },
            'kontoyiannis': {
                'name': '2.3 Kontoyiannis Entropy',
                'description': 'Entropy estimate based on longest match lengths with past subsequences.',
                'features': {
                    'kont': {
                        'name': 'Kontoyiannis Entropy',
                        'description': 'Estimates entropy using longest match lengths with past subsequences. Based on data compression theory and provides consistent entropy estimates.',
                        'formula': 'H ≈ log₂(n) / mean(match_lengths)',
                        'reference': 'Kontoyiannis et al. (1998)',
                        'pattern': 'kont_',
                        'interpretation': 'Higher entropy = More unpredictable'
                    }
                }
            }
        }
    }
}

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

    if memory_mb > MAX_MEMORY_MB:
        logger.warning(f"WARNING: Memory usage exceeds limit ({MAX_MEMORY_MB} MB)")
        gc.collect()
        new_memory = get_memory_usage()
        logger.info(f"   After garbage collection: {new_memory:.1f} MB")


def add_feature(data_set, series_fraq, column):
    """Add fractional feature aligned to dataset"""
    data_set[column] = series_fraq
    shift_data = len(data_set) - len(series_fraq)
    data_set[column] = data_set[column].shift(shift_data)
    return data_set


# ============================================================================
# 3. EDA REPORT GENERATION FUNCTIONS
# ============================================================================

def count_features_by_taxonomy(X_train, taxonomy):
    """Count features matching each taxonomy pattern.

    Returns a nested dict with feature counts for each category, subcategory, and feature.
    """
    counts = {}
    for main_cat_key, main_cat in taxonomy.items():
        main_count = 0
        counts[main_cat_key] = {'subcategories': {}}
        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            subcat_count = 0
            counts[main_cat_key]['subcategories'][subcat_key] = {'features': {}}
            for feat_key, feat_info in subcat.get('features', {}).items():
                pattern = feat_info['pattern']
                matching = len([c for c in X_train.columns if pattern in c.lower()])
                counts[main_cat_key]['subcategories'][subcat_key]['features'][feat_key] = matching
                subcat_count += matching
            counts[main_cat_key]['subcategories'][subcat_key]['total'] = subcat_count
            main_count += subcat_count
        counts[main_cat_key]['total'] = main_count
    return counts


def generate_feature_statistics(X_train, report_dir):
    """Generate and save descriptive statistics for all features."""
    logger.info("Generating feature statistics...")

    stats = X_train.describe().T
    stats['missing'] = X_train.isnull().sum()
    stats['missing_pct'] = (X_train.isnull().sum() / len(X_train) * 100).round(2)
    stats['zeros'] = (X_train == 0).sum()
    stats['zeros_pct'] = ((X_train == 0).sum() / len(X_train) * 100).round(2)
    stats['skewness'] = X_train.skew()
    stats['kurtosis'] = X_train.kurtosis()

    stats_path = os.path.join(report_dir, 'feature_statistics.csv')
    stats.to_csv(stats_path)
    logger.info(f"   Saved: {stats_path}")

    return stats


def generate_correlation_analysis(X_train, y_train, report_dir, top_n=50):
    """Generate correlation matrix and correlation with target."""
    logger.info("Generating correlation analysis...")

    # Correlation with target
    target_corr = X_train.corrwith(y_train['meta_label']).sort_values(ascending=False)

    # Save target correlations
    target_corr_df = pd.DataFrame({
        'feature': target_corr.index,
        'correlation_with_target': target_corr.values
    })
    target_corr_path = os.path.join(report_dir, 'correlation_with_target.csv')
    target_corr_df.to_csv(target_corr_path, index=False)
    logger.info(f"   Saved: {target_corr_path}")

    # Top features correlation matrix (top N by variance)
    top_features = X_train.var().nlargest(top_n).index.tolist()
    corr_matrix = X_train[top_features].corr()

    # Plot correlation matrix
    plt.figure(figsize=(20, 16))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
                annot=False, square=True, linewidths=0.5,
                cbar_kws={'shrink': 0.5})
    plt.title(f'Feature Correlation Matrix (Top {top_n} by Variance)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    corr_matrix_path = os.path.join(report_dir, 'correlation_matrix.png')
    plt.savefig(corr_matrix_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {corr_matrix_path}")

    # Top 20 correlations with target
    plt.figure(figsize=(12, 10))
    top_20_corr = target_corr.head(20)
    bottom_20_corr = target_corr.tail(20)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Positive correlations
    axes[0].barh(range(len(top_20_corr)), top_20_corr.values, color='forestgreen')
    axes[0].set_yticks(range(len(top_20_corr)))
    axes[0].set_yticklabels(top_20_corr.index, fontsize=8)
    axes[0].set_xlabel('Correlation')
    axes[0].set_title('Top 20 Positive Correlations with Target', fontweight='bold')
    axes[0].invert_yaxis()

    # Negative correlations
    axes[1].barh(range(len(bottom_20_corr)), bottom_20_corr.values, color='crimson')
    axes[1].set_yticks(range(len(bottom_20_corr)))
    axes[1].set_yticklabels(bottom_20_corr.index, fontsize=8)
    axes[1].set_xlabel('Correlation')
    axes[1].set_title('Top 20 Negative Correlations with Target', fontweight='bold')
    axes[1].invert_yaxis()

    plt.tight_layout()
    corr_target_path = os.path.join(report_dir, 'correlation_with_target.png')
    plt.savefig(corr_target_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {corr_target_path}")

    return target_corr


def generate_target_distribution(y_train, report_dir):
    """Generate target distribution plots."""
    logger.info("Generating target distribution plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Class balance pie chart
    class_counts = y_train['meta_label'].value_counts()
    labels = ['Down (0)', 'Up (1)']
    colors = ['#ff6b6b', '#4ecdc4']
    axes[0].pie(class_counts, labels=labels, autopct='%1.1f%%', colors=colors,
                explode=(0.02, 0.02), shadow=True)
    axes[0].set_title('Class Balance', fontsize=12, fontweight='bold')

    # Class bar chart
    axes[1].bar(labels, class_counts.values, color=colors, edgecolor='black')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Class Distribution', fontsize=12, fontweight='bold')
    for i, v in enumerate(class_counts.values):
        axes[1].text(i, v + 50, str(v), ha='center', fontweight='bold')

    plt.tight_layout()
    target_dist_path = os.path.join(report_dir, 'target_distribution.png')
    plt.savefig(target_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {target_dist_path}")


def generate_missing_values_analysis(X_train, report_dir):
    """Generate missing values analysis."""
    logger.info("Generating missing values analysis...")

    missing = X_train.isnull().sum()
    missing_pct = (missing / len(X_train) * 100).round(2)

    # Features with missing values
    missing_df = pd.DataFrame({
        'feature': missing.index,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)

    if len(missing_df) > 0:
        plt.figure(figsize=(14, 8))
        top_missing = missing_df.head(30)
        plt.barh(range(len(top_missing)), top_missing['missing_pct'], color='salmon')
        plt.yticks(range(len(top_missing)), top_missing['feature'], fontsize=8)
        plt.xlabel('Missing %')
        plt.title('Top 30 Features with Missing Values', fontsize=12, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        missing_path = os.path.join(report_dir, 'missing_values.png')
        plt.savefig(missing_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {missing_path}")
    else:
        logger.info("   No missing values found!")


def generate_feature_histograms(X_train, report_dir, n_features=50):
    """Generate histograms for top features by variance."""
    logger.info(f"Generating histograms for top {n_features} features...")

    hist_dir = os.path.join(report_dir, 'feature_distributions')
    os.makedirs(hist_dir, exist_ok=True)

    # Select top features by variance
    top_features = X_train.var().nlargest(n_features).index.tolist()

    # Categorize features
    categories = {
        'entropy': [f for f in top_features if 'entropy' in f.lower()],
        'vpin': [f for f in top_features if 'vpin' in f.lower()],
        'microstructure': [f for f in top_features if any(x in f.lower() for x in
                          ['corwin', 'becker', 'amihud', 'kyle', 'roll', 'oir', 'volatility'])]
    }

    for category, features in categories.items():
        if not features:
            continue

        n_cols = 4
        n_rows = (len(features) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3*n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_rows == 1 and n_cols == 1 else axes

        for idx, feature in enumerate(features):
            if idx < len(axes):
                data = X_train[feature].dropna()
                axes[idx].hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.7)
                axes[idx].set_title(feature[:30], fontsize=8)
                axes[idx].tick_params(labelsize=7)

        # Hide unused axes
        for idx in range(len(features), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'{category.upper()} Features Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()

        hist_path = os.path.join(hist_dir, f'{category}_histograms.png')
        plt.savefig(hist_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"   Saved: {hist_path}")


def generate_outliers_analysis(X_train, report_dir, n_features=20):
    """Generate outliers analysis using IQR method."""
    logger.info("Generating outliers analysis...")

    outlier_counts = {}

    for col in X_train.columns:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((X_train[col] < lower) | (X_train[col] > upper)).sum()
        outlier_counts[col] = outliers

    outlier_df = pd.DataFrame({
        'feature': list(outlier_counts.keys()),
        'outlier_count': list(outlier_counts.values()),
        'outlier_pct': [c / len(X_train) * 100 for c in outlier_counts.values()]
    }).sort_values('outlier_count', ascending=False)

    # Save outliers summary
    outlier_path = os.path.join(report_dir, 'outliers_summary.csv')
    outlier_df.to_csv(outlier_path, index=False)
    logger.info(f"   Saved: {outlier_path}")

    # Plot top features with outliers
    top_outliers = outlier_df.head(n_features)

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_outliers)), top_outliers['outlier_pct'], color='orange')
    plt.yticks(range(len(top_outliers)), top_outliers['feature'], fontsize=8)
    plt.xlabel('Outlier %')
    plt.title(f'Top {n_features} Features with Most Outliers (IQR Method)', fontsize=12, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    outlier_plot_path = os.path.join(report_dir, 'outliers_analysis.png')
    plt.savefig(outlier_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {outlier_plot_path}")


def generate_feature_summary_by_category(X_train, report_dir):
    """Generate summary statistics by feature category."""
    logger.info("Generating feature summary by category...")

    categories = {
        'Entropy': [f for f in X_train.columns if 'entropy' in f.lower()],
        'VPIN': [f for f in X_train.columns if 'vpin' in f.lower()],
        'Corwin-Schultz': [f for f in X_train.columns if 'corwin' in f.lower()],
        'Becker-Parkinson': [f for f in X_train.columns if 'becker' in f.lower()],
        'Amihud': [f for f in X_train.columns if 'amihud' in f.lower()],
        'Kyle Lambda': [f for f in X_train.columns if 'kyle' in f.lower()],
        'Roll Spread': [f for f in X_train.columns if 'roll' in f.lower()],
        'OIR': [f for f in X_train.columns if 'oir' in f.lower()],
        'Volatility': [f for f in X_train.columns if 'volatility' in f.lower()],
    }

    summary = []
    for cat_name, features in categories.items():
        if features:
            summary.append({
                'Category': cat_name,
                'Count': len(features),
                'Avg_Mean': X_train[features].mean().mean(),
                'Avg_Std': X_train[features].std().mean(),
                'Total_Missing': X_train[features].isnull().sum().sum(),
                'Avg_Variance': X_train[features].var().mean()
            })

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(report_dir, 'feature_summary_by_category.csv')
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"   Saved: {summary_path}")

    # Plot feature counts by category
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(summary_df)))
    bars = plt.bar(summary_df['Category'], summary_df['Count'], color=colors, edgecolor='black')
    plt.ylabel('Number of Features')
    plt.title('Feature Count by Category', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    for bar, count in zip(bars, summary_df['Count']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 str(count), ha='center', fontweight='bold')

    plt.tight_layout()
    category_plot_path = os.path.join(report_dir, 'feature_count_by_category.png')
    plt.savefig(category_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"   Saved: {category_plot_path}")

    return summary_df


def generate_hierarchical_feature_plots(X_train, y_train, report_dir):
    """Generate hierarchical plots organized by feature taxonomy."""
    logger.info("Generating hierarchical feature plots...")

    # Debug: Check if volatility columns exist in X_train
    vol_cols = [c for c in X_train.columns if 'volatility' in c.lower()]
    logger.info(f"DEBUG: X_train has {len(X_train.columns)} columns, {len(vol_cols)} are volatility")
    if vol_cols:
        logger.info(f"DEBUG: volatility columns = {vol_cols}")

    # Create plots directory structure
    plots_dir = os.path.join(report_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Calculate target correlation for all features
    target_corr = X_train.corrwith(y_train['meta_label'])

    feature_stats = {}

    for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
        main_cat_dir = os.path.join(plots_dir, main_cat_key)
        os.makedirs(main_cat_dir, exist_ok=True)

        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            subcat_dir = os.path.join(main_cat_dir, subcat_key)
            os.makedirs(subcat_dir, exist_ok=True)

            for feat_key, feat_info in subcat.get('features', {}).items():
                pattern = feat_info['pattern']

                # Find matching columns
                matching_cols = [c for c in X_train.columns if pattern in c.lower()]

                # Debug: Log volatility pattern matching
                if feat_key == 'volatility':
                    logger.info(f"DEBUG volatility: pattern='{pattern}', matched {len(matching_cols)} columns")
                    if matching_cols:
                        logger.info(f"DEBUG volatility: first 5 cols = {matching_cols[:5]}")

                if not matching_cols:
                    continue

                # Get base features (without lags) for the main plot
                base_cols = [c for c in matching_cols if '_lag_' not in c]
                if not base_cols:
                    base_cols = matching_cols[:5]  # Use first 5 if all have lags

                # Generate histogram plot for this feature type
                fig, axes = plt.subplots(1, 2, figsize=(14, 5))

                # Left: Histogram of one representative feature
                rep_col = base_cols[0]
                data = X_train[rep_col].dropna()
                axes[0].hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
                axes[0].set_xlabel('Value')
                axes[0].set_ylabel('Frequency')
                axes[0].set_title(f'Distribution: {rep_col}', fontsize=10)
                axes[0].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.4f}')
                axes[0].legend()

                # Right: Box plot of all base features
                if len(base_cols) <= 10:
                    plot_data = X_train[base_cols].dropna()
                    axes[1].boxplot([plot_data[c].dropna() for c in base_cols], labels=[c.split('_')[-1] for c in base_cols])
                    axes[1].set_title(f'{feat_info["name"]} by Window Size', fontsize=10)
                    axes[1].tick_params(axis='x', rotation=45)
                else:
                    # If too many, show correlation with target
                    corrs = [target_corr.get(c, 0) for c in base_cols[:10]]
                    axes[1].barh(range(len(corrs)), corrs, color='teal')
                    axes[1].set_yticks(range(len(corrs)))
                    axes[1].set_yticklabels([c[:30] for c in base_cols[:10]], fontsize=8)
                    axes[1].set_xlabel('Correlation with Target')
                    axes[1].set_title('Top 10 Feature Variants', fontsize=10)

                plt.suptitle(f'{feat_info["name"]}', fontsize=12, fontweight='bold')
                plt.tight_layout()

                plot_path = os.path.join(subcat_dir, f'{feat_key}.png')
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()

                # Calculate statistics using only base columns (without lags)
                base_data = X_train[base_cols].dropna()
                feature_stats[feat_key] = {
                    'n_variants': len(base_cols),
                    'mean': base_data.mean().mean(),
                    'std': base_data.std().mean(),
                    'min': base_data.min().min(),
                    'max': base_data.max().max(),
                    'q25': base_data.quantile(0.25).mean(),
                    'q50': base_data.quantile(0.50).mean(),
                    'q75': base_data.quantile(0.75).mean(),
                    'corr_target_mean': target_corr[base_cols].mean(),
                    'corr_target_max': target_corr[base_cols].abs().max(),
                    'plot_path': f'plots/{main_cat_key}/{subcat_key}/{feat_key}.png'
                }

    logger.info(f"   Generated plots for {len(feature_stats)} feature types")
    return feature_stats


def generate_html_report(X_train, y_train, report_dir, experiment_metadata, stats_df, feature_stats):
    """Generate comprehensive hierarchical HTML report."""
    logger.info("Generating hierarchical HTML report...")

    n_samples = len(X_train)
    n_features = len(X_train.columns)
    class_dist = y_train['meta_label'].value_counts()

    # Extract downsampling info from source file name
    source_file = experiment_metadata.get('source_file', '')
    downsampling_info = parse_downsampling_info(source_file)

    # Calculate feature counts for TOC
    feature_counts = count_features_by_taxonomy(X_train, FEATURE_TAXONOMY)

    # Build table of contents with feature counts
    toc_html = '<nav class="toc"><h3>Table of Contents</h3><ul>'
    for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
        main_count = feature_counts[main_cat_key]['total']
        toc_html += f'<li><a href="#{main_cat_key}">{main_cat["name"]} ({main_count})</a><ul>'
        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            subcat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['total']
            toc_html += f'<li><a href="#{main_cat_key}_{subcat_key}">{subcat["name"]} ({subcat_count})</a><ul>'
            # Add individual feature names
            for feat_key, feat_info in subcat.get('features', {}).items():
                feat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['features'][feat_key]
                if feat_count > 0:
                    toc_html += f'<li><a href="#{feat_key}">{feat_info["name"]} ({feat_count})</a></li>'
            toc_html += '</ul></li>'
        toc_html += '</ul></li>'
    toc_html += '<li><a href="#target">3. Target Analysis</a></li>'
    toc_html += '<li><a href="#quality">4. Data Quality</a></li>'
    toc_html += '</ul></nav>'

    # Build feature sections
    feature_sections_html = ''
    feat_counter = 0

    for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
        feature_sections_html += f'''
        <div class="container category-section" id="{main_cat_key}">
            <h2>{main_cat["name"]}</h2>
            <p class="category-desc">{main_cat["description"]}</p>
        '''

        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            feature_sections_html += f'''
            <div class="subcategory" id="{main_cat_key}_{subcat_key}">
                <h3>{subcat["name"]}</h3>
                <p class="subcategory-desc">{subcat["description"]}</p>
            '''

            for feat_key, feat_info in subcat.get('features', {}).items():
                feat_counter += 1
                stats = feature_stats.get(feat_key, {})

                if not stats:
                    continue

                feature_sections_html += f'''
                <div class="feature-card" id="{feat_key}">
                    <h4>{feat_info["name"]}</h4>
                    <p class="description">{feat_info["description"]}</p>
                    <p class="formula"><strong>Formula:</strong> <code>{feat_info["formula"]}</code></p>
                    <p class="reference"><em>Reference: {feat_info["reference"]}</em></p>
                    <p class="interpretation"><strong>Interpretation:</strong> {feat_info.get("interpretation", "")}</p>

                    <div class="feature-content">
                        <div class="feature-plot">
                            <img src="{stats.get('plot_path', '')}" alt="{feat_info['name']} Distribution">
                        </div>
                        <div class="feature-stats">
                            <table class="stats-table">
                                <tr><th colspan="2">Statistics ({stats.get('n_variants', 0)} variants)</th></tr>
                                <tr><td>Mean</td><td>{stats.get('mean', 0):.6f}</td></tr>
                                <tr><td>Std</td><td>{stats.get('std', 0):.6f}</td></tr>
                                <tr><td>Min</td><td>{stats.get('min', 0):.6f}</td></tr>
                                <tr><td>25%</td><td>{stats.get('q25', 0):.6f}</td></tr>
                                <tr><td>50%</td><td>{stats.get('q50', 0):.6f}</td></tr>
                                <tr><td>75%</td><td>{stats.get('q75', 0):.6f}</td></tr>
                                <tr><td>Max</td><td>{stats.get('max', 0):.6f}</td></tr>
                                <tr class="highlight"><td>Corr w/ Target (avg)</td><td>{stats.get('corr_target_mean', 0):.4f}</td></tr>
                                <tr class="highlight"><td>Corr w/ Target (max)</td><td>{stats.get('corr_target_max', 0):.4f}</td></tr>
                            </table>
                        </div>
                    </div>
                </div>
                '''

            feature_sections_html += '</div>'  # Close subcategory

        feature_sections_html += '</div>'  # Close category

    html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <title>Exploratory Data Analysis Report</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f7fa; color: #333; }}

        .main-content {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{ color: #1a365d; border-bottom: 3px solid #3182ce; padding-bottom: 15px; margin-bottom: 30px; }}
        h2 {{ color: #2c5282; font-size: 1.8em; margin-top: 40px; padding: 15px; background: linear-gradient(90deg, #ebf8ff, transparent); border-left: 4px solid #3182ce; }}
        h3 {{ color: #2b6cb0; font-size: 1.4em; margin-top: 25px; padding: 10px 15px; background-color: #e2e8f0; border-radius: 5px; }}
        h4 {{ color: #2d3748; font-size: 1.2em; margin-bottom: 10px; }}

        .toc {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 25px; }}
        .toc h3 {{ margin-top: 0; color: #2c5282; font-size: 1.1em; }}
        .toc ul {{ padding-left: 15px; margin: 5px 0; }}
        .toc li {{ margin: 5px 0; font-size: 0.9em; }}
        .toc a {{ color: #4a5568; text-decoration: none; }}
        .toc a:hover {{ color: #3182ce; text-decoration: underline; }}

        .container {{ background-color: white; padding: 25px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 25px; }}

        .metric-row {{ display: flex; flex-wrap: wrap; gap: 15px; margin: 20px 0; }}
        .metric {{ padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; min-width: 150px; color: white; }}
        .metric-value {{ font-size: 28px; font-weight: bold; }}
        .metric-label {{ font-size: 12px; opacity: 0.9; margin-top: 5px; }}

        .category-desc, .subcategory-desc {{ color: #718096; font-style: italic; margin: 10px 0 20px 0; padding: 10px; background-color: #f7fafc; border-radius: 5px; }}

        .feature-card {{ background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 20px; margin: 20px 0; transition: box-shadow 0.3s; }}
        .feature-card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
        .feature-card .description {{ color: #4a5568; margin: 10px 0; }}
        .feature-card .formula {{ background-color: #edf2f7; padding: 10px; border-radius: 5px; font-family: 'Courier New', monospace; }}
        .feature-card .reference {{ color: #718096; font-size: 0.9em; }}
        .feature-card .interpretation {{ color: #2f855a; background-color: #f0fff4; padding: 8px; border-radius: 5px; margin-top: 10px; }}

        .feature-content {{ display: flex; gap: 20px; margin-top: 20px; flex-wrap: wrap; }}
        .feature-plot {{ flex: 2; min-width: 400px; }}
        .feature-plot img {{ width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .feature-stats {{ flex: 1; min-width: 200px; }}

        .stats-table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
        .stats-table th {{ background-color: #4a5568; color: white; padding: 10px; text-align: left; }}
        .stats-table td {{ padding: 8px; border-bottom: 1px solid #e2e8f0; }}
        .stats-table tr:hover {{ background-color: #f7fafc; }}
        .stats-table tr.highlight td {{ background-color: #ebf8ff; font-weight: bold; }}

        img {{ max-width: 100%; height: auto; }}

        .data-quality {{ margin-top: 20px; }}
        .data-quality img {{ margin: 10px 0; border-radius: 8px; }}

        footer {{ margin-top: 40px; padding: 20px; text-align: center; color: #718096; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="main-content">
        {toc_html}

        <h1>Exploratory Data Analysis Report</h1>

        <div class="container" id="overview">
            <h2>Experiment Overview</h2>
            <div class="metric-row">
                <div class="metric">
                    <div class="metric-value">{downsampling_info['bar_type']}</div>
                    <div class="metric-label">Dollar Bars {downsampling_info['dollar_volume']}</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{experiment_metadata.get('experiment_id', 'N/A').replace('experiment_', '')}</div>
                    <div class="metric-label">Experiment ID</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{n_samples:,}</div>
                    <div class="metric-label">Total Samples</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{n_features:,}</div>
                    <div class="metric-label">Total Features</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{class_dist.get(0, 0):,}</div>
                    <div class="metric-label">Class 0 (Down)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{class_dist.get(1, 0):,}</div>
                    <div class="metric-label">Class 1 (Up)</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{class_dist.get(0, 0) / max(class_dist.get(1, 1), 1):.2f}:1</div>
                    <div class="metric-label">Class Ratio</div>
                </div>
            </div>
        </div>

        {feature_sections_html}

        <div class="container" id="target">
            <h2>3. Target Analysis</h2>
            <h3>3.1 Class Distribution</h3>
            <img src="target_distribution.png" alt="Target Distribution">
        </div>

        <div class="container data-quality" id="quality">
            <h2>4. Data Quality</h2>

            <h3>4.1 Missing Values</h3>
            <img src="missing_values.png" alt="Missing Values">

            <h3>4.2 Outliers Analysis</h3>
            <img src="outliers_analysis.png" alt="Outliers">

            <h3>4.3 Correlation Matrix</h3>
            <img src="correlation_matrix.png" alt="Correlation Matrix">

            <h3>4.4 Top Correlations with Target</h3>
            <img src="correlation_with_target.png" alt="Correlations with Target">
        </div>

        <footer>
            Generated by Feature Generator | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </footer>
    </div>
</body>
</html>
'''

    html_path = os.path.join(report_dir, 'feature_eda.html')
    with open(html_path, 'w') as f:
        f.write(html_content)
    logger.info(f"   Saved: {html_path}")


def generate_pdf_report(X_train, y_train, report_dir, experiment_metadata, feature_stats):
    """Generate PDF report with all plots embedded."""
    from matplotlib.backends.backend_pdf import PdfPages
    import math

    logger.info("Generating PDF report...")

    pdf_path = os.path.join(report_dir, 'feature_eda.pdf')

    # Extract asset and period info from metadata
    config = experiment_metadata.get('config', {})
    data_config = config.get('data', {})
    data_path = data_config.get('data_path', '')

    # Extract asset from path
    path_parts = data_path.split('/')
    asset = path_parts[1].upper().replace('-', ' ') if len(path_parts) > 1 else 'N/A'

    # Extract downsampling info from source file name
    source_file = experiment_metadata.get('source_file', '')
    downsampling_info = parse_downsampling_info(source_file)

    # Extract training period from X_train index
    if hasattr(X_train, 'index') and len(X_train) > 0:
        try:
            train_start = pd.to_datetime(X_train.index.min()).strftime('%Y-%m-%d %H:%M')
            train_end = pd.to_datetime(X_train.index.max()).strftime('%Y-%m-%d %H:%M')
            period_str = f"{train_start} to {train_end}"
        except:
            period_str = f"{X_train.index.min()} to {X_train.index.max()}"
    else:
        period_str = "N/A"

    with PdfPages(pdf_path) as pdf:
        # Title page with asset and period
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.75, 'Exploratory Data Analysis Report', ha='center', fontsize=28, fontweight='bold')
        fig.text(0.5, 0.65, f'Asset: {asset}', ha='center', fontsize=18)
        fig.text(0.5, 0.58, f'Downsampling: {downsampling_info["full_description"]}', ha='center', fontsize=14, color='#2c5282')
        fig.text(0.5, 0.50, f'Training Period: {period_str}', ha='center', fontsize=14)
        exp_id = experiment_metadata.get("experiment_id", "N/A").replace("experiment_", "")
        fig.text(0.5, 0.42, f'Experiment: {exp_id}', ha='center', fontsize=12)
        fig.text(0.5, 0.35, f'Samples: {len(X_train):,}  |  Features: {len(X_train.columns):,}', ha='center', fontsize=12)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Calculate feature counts for TOC
        feature_counts = count_features_by_taxonomy(X_train, FEATURE_TAXONOMY)

        # Table of Contents page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.92, 'Table of Contents', ha='center', fontsize=24, fontweight='bold')

        y_pos = 0.82
        for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
            main_count = feature_counts[main_cat_key]['total']
            # Main category header
            fig.text(0.08, y_pos, f"{main_cat['name']} ({main_count} features)",
                    fontsize=14, fontweight='bold', color='#2c3e50')
            y_pos -= 0.04

            for subcat_key, subcat in main_cat.get('subcategories', {}).items():
                subcat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['total']
                # Subcategory
                fig.text(0.12, y_pos, f"{subcat['name']} - {subcat_count} features",
                        fontsize=11, color='#34495e')
                y_pos -= 0.025

                # Individual features
                for feat_key, feat_info in subcat.get('features', {}).items():
                    feat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['features'][feat_key]
                    if feat_count > 0:
                        fig.text(0.18, y_pos, f"- {feat_info['name']} ({feat_count})",
                                fontsize=9, color='#7f8c8d')
                        y_pos -= 0.02

                y_pos -= 0.01  # Extra space after subcategory

            y_pos -= 0.02  # Extra space after main category

        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

        # Generate pages for each feature category
        for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
            # Category divider page
            main_count = feature_counts[main_cat_key]['total']
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.5, 0.60, main_cat['name'], ha='center', fontsize=28, fontweight='bold', color='#2c3e50')
            fig.text(0.5, 0.50, f'{main_count} features', ha='center', fontsize=18, color='#7f8c8d')
            fig.text(0.5, 0.38, main_cat['description'], ha='center', fontsize=11,
                    wrap=True, color='#34495e', style='italic')

            # List subcategories
            y_pos = 0.28
            for subcat_key, subcat in main_cat.get('subcategories', {}).items():
                subcat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['total']
                fig.text(0.5, y_pos, f"{subcat['name']} ({subcat_count} features)",
                        ha='center', fontsize=10, color='#34495e')
                y_pos -= 0.04

            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

            for subcat_key, subcat in main_cat.get('subcategories', {}).items():
                for feat_key, feat_info in subcat.get('features', {}).items():
                    pattern = feat_info['pattern']
                    matching_cols = [c for c in X_train.columns if pattern in c.lower()]

                    if not matching_cols:
                        continue

                    # Get base features (without lags)
                    base_cols = [c for c in matching_cols if '_lag_' not in c]
                    if not base_cols:
                        continue

                    # Pages with description header + histograms grid
                    n_cols_grid = 3
                    max_per_page = 6  # Reduced to fit description

                    for page_idx, page_start in enumerate(range(0, len(base_cols), max_per_page)):
                        page_cols = base_cols[page_start:page_start + max_per_page]
                        n_on_page = len(page_cols)
                        n_hist_rows = math.ceil(n_on_page / n_cols_grid)

                        # Calculate figure height
                        if page_idx == 0:
                            fig_height = 2.5 * n_hist_rows + 2.5  # Extra space for description
                        else:
                            fig_height = 2.5 * n_hist_rows + 1

                        fig = plt.figure(figsize=(11, fig_height))

                        if page_idx == 0:
                            # Add description at the top of the first page
                            ax_desc = fig.add_axes([0.05, 0.78, 0.9, 0.18])
                            ax_desc.axis('off')

                            stats = feature_stats.get(feat_key, {})
                            desc_text = f"""{feat_info['name']}

{feat_info['description']}

Interpretation: {feat_info.get('interpretation', 'N/A')}  |  Reference: {feat_info['reference']}

Statistics (across {len(base_cols)} window sizes):  Mean: {stats.get('mean', 0):.2e}  |  Std: {stats.get('std', 0):.2e}  |  Range: [{stats.get('min', 0):.2e}, {stats.get('max', 0):.2e}]"""

                            ax_desc.text(0, 1, desc_text, transform=ax_desc.transAxes, fontsize=9,
                                        verticalalignment='top', wrap=True,
                                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

                            grid_bottom = 0.05
                            grid_height = 0.68
                        else:
                            fig.suptitle(f'{feat_info["name"]} (continued)', fontsize=12, y=0.98)
                            grid_bottom = 0.08
                            grid_height = 0.85

                        # Create histogram grid
                        for idx, col in enumerate(page_cols):
                            row = idx // n_cols_grid
                            col_idx = idx % n_cols_grid

                            left = 0.06 + col_idx * 0.31
                            bottom = grid_bottom + (n_hist_rows - 1 - row) * (grid_height / n_hist_rows)
                            width = 0.27
                            height = (grid_height / n_hist_rows) * 0.85

                            ax = fig.add_axes([left, bottom, width, height])
                            data = X_train[col].dropna()
                            ax.hist(data, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
                            ax.axvline(data.mean(), color='red', linestyle='--', linewidth=1)

                            # Simplified label (just window)
                            col_label = col.replace(pattern + '_', '').replace('_fixed', '').replace('fixed_', '')
                            ax.set_title(f"w={col_label}", fontsize=9)
                            ax.tick_params(labelsize=7)

                            # Add stats inside histogram
                            ax.text(0.95, 0.95, f"μ={data.mean():.2e}\nσ={data.std():.2e}",
                                   transform=ax.transAxes, fontsize=6, ha='right', va='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()

    logger.info(f"   Saved: {pdf_path}")
    return pdf_path


def generate_markdown_report(X_train, y_train, report_dir, experiment_metadata, feature_stats):
    """Generate Markdown report with embedded base64 images."""
    import base64
    from io import BytesIO

    logger.info("Generating Markdown report...")

    # Extract downsampling info from source file name
    source_file = experiment_metadata.get('source_file', '')
    downsampling_info = parse_downsampling_info(source_file)

    def fig_to_base64(fig):
        """Convert matplotlib figure to base64 string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    md_content = f"""# Exploratory Data Analysis Report

**Downsampling:** {downsampling_info['full_description']}
**Experiment ID:** {experiment_metadata.get('experiment_id', 'N/A').replace('experiment_', '')}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Samples:** {len(X_train):,}
**Features:** {len(X_train.columns):,}

---

## Table of Contents

"""
    # Calculate feature counts for TOC
    feature_counts = count_features_by_taxonomy(X_train, FEATURE_TAXONOMY)

    # Build TOC with feature counts
    for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
        main_count = feature_counts[main_cat_key]['total']
        md_content += f"- [{main_cat['name']} ({main_count} features)](#{main_cat_key})\n"
        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            subcat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['total']
            md_content += f"  - [{subcat['name']} ({subcat_count})](#{main_cat_key}-{subcat_key})\n"
            # Add individual feature names
            for feat_key, feat_info in subcat.get('features', {}).items():
                feat_count = feature_counts[main_cat_key]['subcategories'][subcat_key]['features'][feat_key]
                if feat_count > 0:
                    md_content += f"    - {feat_info['name']} ({feat_count})\n"
    md_content += "- [Target Analysis](#target-analysis)\n"
    md_content += "- [Data Quality](#data-quality)\n\n---\n\n"

    target_corr = X_train.corrwith(y_train['meta_label'])

    # Generate feature sections
    for main_cat_key, main_cat in FEATURE_TAXONOMY.items():
        md_content += f"## {main_cat['name']} {{#{main_cat_key}}}\n\n"
        md_content += f"*{main_cat['description']}*\n\n"

        for subcat_key, subcat in main_cat.get('subcategories', {}).items():
            md_content += f"### {subcat['name']} {{#{main_cat_key}-{subcat_key}}}\n\n"
            md_content += f"*{subcat['description']}*\n\n"

            for feat_key, feat_info in subcat.get('features', {}).items():
                pattern = feat_info['pattern']
                matching_cols = [c for c in X_train.columns if pattern in c.lower()]

                if not matching_cols:
                    continue

                base_cols = [c for c in matching_cols if '_lag_' not in c]
                if not base_cols:
                    base_cols = matching_cols[:5]

                stats = feature_stats.get(feat_key, {})

                md_content += f"#### {feat_info['name']}\n\n"
                md_content += f"**Description:** {feat_info['description']}\n\n"
                md_content += f"**Formula:** `{feat_info['formula']}`\n\n"
                md_content += f"**Reference:** {feat_info['reference']}\n\n"
                md_content += f"**Interpretation:** {feat_info.get('interpretation', 'N/A')}\n\n"

                # Generate plot
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                rep_col = base_cols[0]
                data = X_train[rep_col].dropna()
                axes[0].hist(data, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
                axes[0].axvline(data.mean(), color='red', linestyle='--')
                axes[0].set_title(f'Distribution: {rep_col[:35]}', fontsize=9)

                if len(base_cols) <= 10 and len(base_cols) > 1:
                    axes[1].boxplot([X_train[c].dropna() for c in base_cols],
                                   labels=[c.split('_')[-1] for c in base_cols])
                    axes[1].tick_params(axis='x', rotation=45)
                else:
                    corrs = [target_corr.get(c, 0) for c in base_cols[:8]]
                    axes[1].barh(range(len(corrs)), corrs, color='teal')
                    axes[1].set_yticks(range(len(corrs)))
                    axes[1].set_yticklabels([c[:20] for c in base_cols[:8]], fontsize=8)
                axes[1].set_title('Variants', fontsize=9)
                plt.tight_layout()

                img_base64 = fig_to_base64(fig)
                md_content += f"![{feat_info['name']}](data:image/png;base64,{img_base64})\n\n"

                # Statistics table
                md_content += "| Statistic | Value |\n|-----------|-------|\n"
                md_content += f"| Variants | {stats.get('n_variants', 0)} |\n"
                md_content += f"| Mean | {stats.get('mean', 0):.6f} |\n"
                md_content += f"| Std | {stats.get('std', 0):.6f} |\n"
                md_content += f"| Min | {stats.get('min', 0):.6f} |\n"
                md_content += f"| Max | {stats.get('max', 0):.6f} |\n"
                md_content += f"| Corr w/ Target (avg) | {stats.get('corr_target_mean', 0):.4f} |\n"
                md_content += f"| Corr w/ Target (max) | {stats.get('corr_target_max', 0):.4f} |\n\n"
                md_content += "---\n\n"

    # Target Analysis
    md_content += "## Target Analysis {#target-analysis}\n\n"
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    class_counts = y_train['meta_label'].value_counts()
    labels = ['Down (0)', 'Up (1)']
    colors = ['#ff6b6b', '#4ecdc4']
    axes[0].pie(class_counts, labels=labels, autopct='%1.1f%%', colors=colors)
    axes[0].set_title('Class Balance')
    axes[1].bar(labels, class_counts.values, color=colors)
    axes[1].set_title('Class Distribution')
    plt.tight_layout()
    img_base64 = fig_to_base64(fig)
    md_content += f"![Target Distribution](data:image/png;base64,{img_base64})\n\n"

    # Data Quality
    md_content += "## Data Quality {#data-quality}\n\n"

    # Correlation with target
    md_content += "### Top Correlations with Target\n\n"
    top_corr = target_corr.abs().nlargest(20)
    md_content += "| Feature | Correlation |\n|---------|-------------|\n"
    for feat, corr in top_corr.items():
        md_content += f"| {feat[:40]} | {target_corr[feat]:.4f} |\n"
    md_content += "\n"

    # Save markdown
    md_path = os.path.join(report_dir, 'feature_eda.md')
    with open(md_path, 'w') as f:
        f.write(md_content)

    logger.info(f"   Saved: {md_path}")
    return md_path


def generate_eda_report(X_train, y_train, report_dir, experiment_metadata):
    """Generate complete hierarchical EDA report."""
    logger.info("\n" + "="*70)
    logger.info("GENERATING HIERARCHICAL EDA REPORT")
    logger.info("="*70)

    os.makedirs(report_dir, exist_ok=True)

    # 1. Feature Statistics
    stats_df = generate_feature_statistics(X_train, report_dir)

    # 2. Correlation Analysis
    generate_correlation_analysis(X_train, y_train, report_dir)

    # 3. Target Distribution
    generate_target_distribution(y_train, report_dir)

    # 4. Missing Values
    generate_missing_values_analysis(X_train, report_dir)

    # 5. Outliers Analysis
    generate_outliers_analysis(X_train, report_dir)

    # 6. Feature Summary by Category
    generate_feature_summary_by_category(X_train, report_dir)

    # 7. Hierarchical Feature Plots (NEW)
    feature_stats = generate_hierarchical_feature_plots(X_train, y_train, report_dir)

    # 8. Hierarchical HTML Report
    generate_html_report(X_train, y_train, report_dir, experiment_metadata, stats_df, feature_stats)

    # 9. PDF Report (self-contained)
    generate_pdf_report(X_train, y_train, report_dir, experiment_metadata, feature_stats)

    # 10. Markdown Report (with embedded images)
    generate_markdown_report(X_train, y_train, report_dir, experiment_metadata, feature_stats)

    logger.info("\nHierarchical EDA Report generation completed!")
    logger.info(f"Report directory: {report_dir}")
    logger.info(f"\nFiles generated:")
    logger.info(f"   feature_eda.html - Interactive HTML (local use)")
    logger.info(f"   feature_eda.pdf  - PDF (shareable)")
    logger.info(f"   feature_eda.md   - Markdown (shareable)")


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================

def main(config_path=None):
    """Main feature generation pipeline"""

    pipeline_start_time = time.time()

    # Load configuration
    config = load_config(config_path)

    logger.info("\n" + "="*70)
    logger.info("STARTING FEATURE GENERATION PIPELINE")
    logger.info("="*70)
    logger.info(f"Process PID: {os.getpid()}")
    logger.info(f"Initial memory: {get_memory_usage():.1f} MB")

    try:
        # ========================================================================
        # STEP 1: DATA LOADING AND PREPARATION
        # ========================================================================

        logger.info("\nSTEP 1: DATA LOADING AND PREPARATION")
        step_start = time.time()

        DATA_PATH = config['data']['data_path']
        BASE_OUTPUT_PATH = config['data']['output_path']
        FILE_NAME = config['data']['file_name']

        logger.info(f"Configuration loaded from config.yaml")
        logger.info(f"   Data path: {DATA_PATH}")
        logger.info(f"   File: {FILE_NAME}")

        # Create experiment folder
        experiment_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_folder = f"experiment_{experiment_timestamp}"
        OUTPUT_PATH = os.path.join(BASE_OUTPUT_PATH, experiment_folder)

        logger.info(f"Experiment folder: {experiment_folder}")

        # Create directories
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        features_dir = os.path.join(OUTPUT_PATH, 'features')
        report_dir = os.path.join(OUTPUT_PATH, 'report')
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(report_dir, exist_ok=True)

        # Save experiment metadata
        experiment_metadata = {
            'experiment_id': experiment_folder,
            'experiment_timestamp': experiment_timestamp,
            'source_file': FILE_NAME,
            'source_path': DATA_PATH,
            'created_at': datetime.now().isoformat(),
            'pipeline_type': 'feature_generator',
            'config': config
        }

        metadata_path = os.path.join(OUTPUT_PATH, 'experiment_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        logger.info(f"Experiment metadata saved to: {metadata_path}")

        # Load data
        file_base_name = FILE_NAME.replace('.parquet', '')
        folder_path = os.path.join(DATA_PATH, file_base_name)
        direct_path = os.path.join(DATA_PATH, FILE_NAME)

        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            filepath = os.path.join(folder_path, FILE_NAME)
            logger.info(f"Found file in folder structure: {folder_path}/")
        elif os.path.exists(direct_path):
            filepath = direct_path
            logger.info(f"Found file in direct path")
        else:
            logger.error(f"File not found!")
            raise FileNotFoundError(f"Could not find {FILE_NAME}")

        logger.info(f"Loading data from {filepath}...")
        series = pd.read_parquet(filepath)
        logger.info(f"Data loaded: {len(series):,} records")

        log_memory("data loading")
        logger.info(f"Step 1 completed in {time.time() - step_start:.2f} seconds")

        # Aggregation
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

        existing_columns = series.columns
        agg_rules_filtered = {col: func for col, func in agg_rules.items() if col in existing_columns}

        series = series.groupby('end_time').agg(agg_rules_filtered)
        series = series[:-1]
        series['log_close'] = np.log(series['close'])

        logger.info(f"Data aggregated: {len(series):,} records")
        log_memory("data aggregation")

        # ========================================================================
        # STEP 2: DATA SPLITTING
        # ========================================================================

        logger.info("\nSTEP 2: DATA SPLITTING")
        step_start = time.time()

        train_pct = config['data_split']['train_pct']
        val_pct = config['data_split']['val_pct']
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

        train_dataset_copy = train_dataset.copy().reset_index(drop=True)
        val_dataset_copy = val_dataset.copy().reset_index(drop=True)
        test_dataset_copy = test_dataset.copy().reset_index(drop=True)

        log_memory("data splitting")
        logger.info(f"Step 2 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 3: FRACTIONAL DIFFERENTIATION
        # ========================================================================

        logger.info("\nSTEP 3: FRACTIONAL DIFFERENTIATION")
        step_start = time.time()

        d_values_coarse = np.round(np.arange(0.1, 1.01, 0.1), 2)
        conf = 0.05
        thresh = 1e-3
        column = 'log_close'

        logger.info("Two-stage search for optimal d:")
        logger.info(f"Stage 1 - Coarse search: {d_values_coarse.tolist()}")

        d_found = None
        series_frac_diff_IS = None
        series_frac_diff_OOS = None
        series_frac_diff_OOB = None

        # Stage 1: Coarse search
        logger.info("Stage 1: Coarse search...")
        d_coarse = None

        for d in d_values_coarse:
            series_train = np.array(train_dataset_copy[column])
            series_frac_diff_temp = FractionalDifferentiation.frac_diff_optimized(
                series_train, d).dropna().reset_index(drop=True)

            len_max = 100_000
            if len(series_frac_diff_temp) > len_max:
                series_frac_diff_reduced = series_frac_diff_temp[-len_max:]
            else:
                series_frac_diff_reduced = series_frac_diff_temp

            try:
                train = StationarityTester.adf_test(series_frac_diff_reduced,
                                                   title=f'Fractional Differentiation d={d}')

                critical_value = train['critical_values'][f"{conf * 100:.0f}%"]
                adf_statistic = train['statistic']
                is_stationary = (train['pvalue'] < conf) and (adf_statistic < critical_value)

                if is_stationary:
                    d_coarse = d
                    logger.info(f'Coarse search found: d = {d}')
                    break
            except Exception as e:
                logger.warning(f'Error testing d={d}: {e}')
                continue

        # Stage 2: Fine search
        if d_coarse is not None:
            logger.info(f"\nStage 2: Fine search around d={d_coarse}")

            if d_coarse > 0.1:
                d_min = max(0.01, d_coarse - 0.1)
                d_max = d_coarse
                d_values_fine = np.round(np.arange(d_min, d_max + 0.01, 0.02), 3)
            else:
                d_values_fine = np.round(np.arange(0.01, d_coarse + 0.01, 0.02), 3)

            for d in d_values_fine:
                series_train = np.array(train_dataset_copy[column])
                series_frac_diff_temp = FractionalDifferentiation.frac_diff_optimized(
                    series_train, d).dropna().reset_index(drop=True)

                len_max = 100_000
                if len(series_frac_diff_temp) > len_max:
                    series_frac_diff_reduced = series_frac_diff_temp[-len_max:]
                else:
                    series_frac_diff_reduced = series_frac_diff_temp

                try:
                    train = StationarityTester.adf_test(series_frac_diff_reduced,
                                                       title=f'Fractional Differentiation d={d}')

                    critical_value = train['critical_values'][f"{conf * 100:.0f}%"]
                    adf_statistic = train['statistic']
                    is_stationary = (train['pvalue'] < conf) and (adf_statistic < critical_value)

                    if is_stationary:
                        series_frac_diff_IS = series_frac_diff_temp

                        series_val = np.array(val_dataset_copy[column])
                        series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                            series_val, d).dropna().reset_index(drop=True)

                        series_test = np.array(test_dataset_copy[column])
                        series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                            series_test, d).dropna().reset_index(drop=True)

                        logger.info(f'OPTIMAL d FOUND (refined): {column} - d = {d:.3f} - Stationary')
                        d_found = d
                        break
                except Exception as e:
                    logger.warning(f'Error testing d={d} (fine): {e}')
                    continue

            if d_found is None:
                d_found = d_coarse
                logger.info(f"Using coarse result: d = {d_found}")

                series_train = np.array(train_dataset_copy[column])
                series_frac_diff_IS = FractionalDifferentiation.frac_diff_optimized(
                    series_train, d_found).dropna().reset_index(drop=True)

                series_val = np.array(val_dataset_copy[column])
                series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                    series_val, d_found).dropna().reset_index(drop=True)

                series_test = np.array(test_dataset_copy[column])
                series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                    series_test, d_found).dropna().reset_index(drop=True)

        if d_found is None:
            logger.error("No suitable d found for stationarity!")
            logger.info("Using d=1.0 as fallback")
            d_found = 1.0

            series_train = np.array(train_dataset_copy[column])
            series_frac_diff_IS = FractionalDifferentiation.frac_diff_optimized(
                series_train, d_found).dropna().reset_index(drop=True)

            series_val = np.array(val_dataset_copy[column])
            series_frac_diff_OOS = FractionalDifferentiation.frac_diff_optimized(
                series_val, d_found).dropna().reset_index(drop=True)

            series_test = np.array(test_dataset_copy[column])
            series_frac_diff_OOB = FractionalDifferentiation.frac_diff_optimized(
                series_test, d_found).dropna().reset_index(drop=True)

        column = 'fraq_close'
        train_dataset_diff = add_feature(train_dataset_copy.copy(), series_frac_diff_IS, column)
        val_dataset_diff = add_feature(val_dataset_copy.copy(), series_frac_diff_OOS, column)
        test_dataset_diff = add_feature(test_dataset_copy.copy(), series_frac_diff_OOB, column)

        log_memory("fractional differentiation")
        logger.info(f"Step 3 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 4: AR MODEL WITH MULTICOLLINEARITY TREATMENT
        # ========================================================================

        logger.info("\nSTEP 4: AR MODEL WITH MULTICOLLINEARITY TREATMENT")
        step_start = time.time()

        improved_ar = ImprovedAutoRegressiveModel(series_frac_diff_IS)

        p_max_calculated = round(len(series_frac_diff_IS)**(1/2))

        if p_max_calculated > 200:
            p_max = 50
            logger.warning(f"Very large dataset detected (p_max would be {p_max_calculated})")
        elif p_max_calculated > 100:
            p_max = 75
        else:
            p_max = p_max_calculated

        logger.info(f"p_max calculated: {p_max_calculated}, using: {p_max}")

        p_otimo, metrics = AutoRegressiveModel.select_ar_order(
            series_frac_diff_IS,
            p_max=p_max,
            criterio='other',
            limiar=0.01,
            limiar_pvalor=conf,
            min_reducao_absoluta=0.01
        )

        logger.info(f"Optimal order selected: p = {p_otimo}")

        ar_results = improved_ar.fit_with_multicollinearity_treatment(p_otimo, treatment_method='auto')

        residuals_IS = ar_results['residuals']

        logger.info(f"Model fitted: {ar_results['method'].upper()}")
        logger.info(f"Final order: AR({ar_results['p']})")

        Y_pred_OOS = improved_ar.monteCarlo_improved(series_frac_diff_OOS, ar_results)
        residuals_OOS = series_frac_diff_OOS - Y_pred_OOS['fraqdiff_pred']

        Y_pred_OOB = improved_ar.monteCarlo_improved(series_frac_diff_OOB, ar_results)
        residuals_OOB = series_frac_diff_OOB - Y_pred_OOB['fraqdiff_pred']

        log_memory("AR model")
        logger.info(f"Step 4 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 5: EVENT DETECTION (CUSUM)
        # ========================================================================

        logger.info("\nSTEP 5: EVENT DETECTION (CUSUM)")
        step_start = time.time()

        tau1 = config['cusum']['tau1']
        tau2 = config['cusum']['tau2']
        sTime = config['triple_barrier']['time_horizon']
        ptSl = [config['triple_barrier']['profit_take'], config['triple_barrier']['stop_loss']]

        logger.info(f"   Triple Barrier: ptSl={ptSl}, sTime={sTime} min")
        logger.info(f"   CUSUM: tau1={tau1}, tau2={tau2}")

        series_prim = train_dataset_diff.copy()
        n_lost_fracdiff = len(series_prim) - len(series_frac_diff_IS)

        span0 = config['volatility']['span']
        vol = EventAnalyzer.getVol(series_prim['fraq_close'], span0=span0)
        series_prim['vol'] = vol

        start_idx = n_lost_fracdiff + ar_results['p']
        aligned_df = series_prim.iloc[start_idx:start_idx+len(residuals_IS)].copy()
        aligned_df['residuals'] = residuals_IS if isinstance(residuals_IS, np.ndarray) else residuals_IS.values
        aligned_df = aligned_df.dropna(subset=['vol', 'residuals'])

        logger.info(f"Data after alignment: {len(aligned_df)} records")

        series_prim_events = aligned_df.set_index('end_time')
        events = EventAnalyzer.getTEvents(
            series_prim_events['residuals'],
            series_prim_events['vol'],
            tau1
        )

        if len(events) == 0:
            logger.info("No events detected. Adjusting threshold...")
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

        logger.info(f"Events detected: {len(events)}")
        events_IS = events.copy()

        log_memory("event detection")
        logger.info(f"Step 5 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 6: TRIPLE BARRIER METHOD
        # ========================================================================

        logger.info("\nSTEP 6: TRIPLE BARRIER METHOD")
        step_start = time.time()

        if len(events) > 0:
            close = series_prim_events['close']
            events['t1'] = events['t1'].where(events['t1'] <= close.index[-1], pd.NaT)

            molecules = events.index.tolist()

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

            logger.info(f"Results:")
            logger.info(f"   Total: {len(triple_barrier_events)}")
            logger.info(f"   Take Profits: {(triple_barrier_events['label'] == 1).sum()}")
            logger.info(f"   Stop Losses: {(triple_barrier_events['label'] == -1).sum()}")
            logger.info(f"   Time exits: {(triple_barrier_events['label'] == 0).sum()}")

        triple_barrier_events_IS = triple_barrier_events.copy()
        log_memory("triple barrier")
        logger.info(f"Step 6 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 7: FEATURE ENGINEERING
        # ========================================================================

        logger.info("\nSTEP 7: FEATURE ENGINEERING")
        step_start = time.time()

        windows = list(range(50, 500, 50))
        sl_range = (10, 200, 10)

        train_dataset_builded = train_dataset_diff.copy()

        logger.info("Calculating microstructure features...")

        calculator = UnifiedMicrostructureFeatures()
        train_dataset_builded = calculator.calculate_all_features(
            train_dataset_builded,
            windows=windows,
            sl_range=sl_range,
            skip_stationarity=True
        )

        logger.info(f"Total microstructure features: {len(calculator.feature_names)}")

        # Entropy features
        logger.info("Calculating entropy features...")
        window_sizes = list(range(50, 500, 50))
        bins_list = list(range(25, 250, 25))
        all_entropy_bins = [25, 50, 75, 100, 150, 200]

        # Helper functions to filter valid bin/window combinations
        def get_valid_bins_kont(window: int, max_ratio: int = 4) -> list:
            """Kontoyiannis: require b <= w/4 (4+ obs per bin)"""
            return [b for b in all_entropy_bins if window >= b * max_ratio]

        def get_valid_bins_lz(window: int, max_ratio: int = 2) -> list:
            """Lempel-Ziv: require b <= w/2 (2+ obs per bin)"""
            return [b for b in all_entropy_bins if window >= b * max_ratio]

        def filter_low_variance_features(df: pd.DataFrame, min_variance: float = 1e-10):
            """Remove features with variance below threshold."""
            variances = df.var()
            low_var_cols = variances[variances < min_variance].index.tolist()
            if low_var_cols:
                logger.warning(f"Removing {len(low_var_cols)} features with variance < {min_variance}:")
                for col in low_var_cols[:10]:  # Show max 10
                    logger.warning(f"   - {col}: var={variances[col]:.2e}")
                if len(low_var_cols) > 10:
                    logger.warning(f"   ... and {len(low_var_cols) - 10} more")
            return df.drop(columns=low_var_cols), low_var_cols

        entropy_results = EntropyFeatures.calculate_entropy_features_batch(
            series_frac_diff_IS, window_sizes, bins_list
        )
        logger.info(f"Entropy features: {len(entropy_results.columns)} combinations")

        for col in entropy_results.columns:
            train_dataset_builded[col] = entropy_results[col]

        for col in entropy_results.columns:
            train_dataset_builded[col] = EntropyFeatures.move_nans_to_front(train_dataset_builded[col])

        # Lempel-Ziv Complexity (filtered: b <= w/2)
        logger.info("Calculating Lempel-Ziv complexity (filtered b <= w/2)...")
        lz_results_list = []
        for window in window_sizes:
            valid_bins = get_valid_bins_lz(window, max_ratio=2)
            if valid_bins:
                result = EntropyFeatures.calculate_lempel_ziv_batch(
                    series_frac_diff_IS, [window], valid_bins
                )
                lz_results_list.append(result)
        lz_results = pd.concat(lz_results_list, axis=1) if lz_results_list else pd.DataFrame()
        logger.info(f"Lempel-Ziv features: {len(lz_results.columns)} combinations (filtered from 54)")

        for col in lz_results.columns:
            train_dataset_builded[col] = lz_results[col]

        for col in lz_results.columns:
            train_dataset_builded[col] = EntropyFeatures.move_nans_to_front(train_dataset_builded[col])

        # Kontoyiannis Entropy (filtered: b <= w/4)
        logger.info("Calculating Kontoyiannis entropy (filtered b <= w/4)...")
        kont_results_list = []
        for window in window_sizes:
            valid_bins = get_valid_bins_kont(window, max_ratio=4)
            if valid_bins:
                result = EntropyFeatures.calculate_kontoyiannis_batch(
                    series_frac_diff_IS, [window], valid_bins
                )
                kont_results_list.append(result)
        kont_results = pd.concat(kont_results_list, axis=1) if kont_results_list else pd.DataFrame()
        logger.info(f"Kontoyiannis features: {len(kont_results.columns)} combinations (filtered from 54)")

        for col in kont_results.columns:
            train_dataset_builded[col] = kont_results[col]

        for col in kont_results.columns:
            train_dataset_builded[col] = EntropyFeatures.move_nans_to_front(train_dataset_builded[col])

        # Filter low-variance entropy features (safety net)
        entropy_cols = [c for c in train_dataset_builded.columns
                       if any(x in c for x in ['entropy_', 'lz_', 'kont_'])]
        if entropy_cols:
            entropy_subset = train_dataset_builded[entropy_cols]
            _, removed_cols = filter_low_variance_features(entropy_subset, min_variance=1e-10)
            if removed_cols:
                train_dataset_builded = train_dataset_builded.drop(columns=removed_cols)
                logger.info(f"Removed {len(removed_cols)} low-variance entropy features")
            else:
                logger.info("No low-variance entropy features found (all valid)")

        # ========================================================================
        # STEP 7B: CLEANUP (Lags are now applied in rf_trainer.py)
        # ========================================================================

        logger.info("\nSTEP 7B: CLEANUP (lags will be applied in rf_trainer.py)")
        logger.info(f"   Base features: {len(train_dataset_builded.columns)}")

        # Prepare final dataset (NaN rows kept - dropna applied in rf_trainer.py)
        original_size = len(train_dataset_builded)
        n_nan_rows = train_dataset_builded.isna().any(axis=1).sum()
        # Keep end_time as column (drop=False) since it's needed for merge
        train_dataset_builded = train_dataset_builded.reset_index(drop=False)
        logger.info(f"   Samples: {original_size:,} (includes {n_nan_rows:,} rows with NaN)")

        train_dataset_builded = train_dataset_builded.set_index('end_time')

        # Merge with events (both use index)
        temp_merged_df = pd.merge(
            triple_barrier_events_IS,
            train_dataset_builded,
            how='left',
            left_index=True,
            right_index=True
        )
        # Index is already end_time after merge
        temp_merged_df_filt = temp_merged_df[temp_merged_df['close'].notna()]
        remove_IS = len(temp_merged_df) - len(temp_merged_df_filt)

        final_dataset_IS = pd.concat([triple_barrier_events_IS, train_dataset_builded], axis=1)
        final_dataset_IS = final_dataset_IS[final_dataset_IS['meta_label'].notna()]
        final_dataset_IS = final_dataset_IS[remove_IS:]

        y_train = final_dataset_IS[['meta_label']]

        # Class distribution
        class_distribution = y_train['meta_label'].value_counts()
        class_ratio = class_distribution[0] / class_distribution[1] if 1 in class_distribution.index else float('inf')

        logger.info("\nTarget Class Distribution:")
        logger.info(f"   Class 0 (Negative): {class_distribution.get(0, 0)} samples ({class_distribution.get(0, 0)/len(y_train)*100:.1f}%)")
        logger.info(f"   Class 1 (Positive): {class_distribution.get(1, 0)} samples ({class_distribution.get(1, 0)/len(y_train)*100:.1f}%)")
        logger.info(f"   Imbalance Ratio: {class_ratio:.2f}:1")

        # Clean columns
        columns_to_drop = ['t1', 'side', 'sl', 'pt', 'retorno', 'max_drawdown_in_trade',
                          'label', 'meta_label', 'open', 'high', 'low', 'close',
                          'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd',
                          'total_volume', 'params', 'time_trial', 'log_close', 'fraq_close']

        columns_to_drop = [col for col in columns_to_drop if col in final_dataset_IS.columns]
        final_dataset_IS.drop(columns=columns_to_drop, inplace=True)
        X_train = final_dataset_IS.copy()

        logger.info(f"\nTotal features: {len(X_train.columns)}")
        logger.info(f"Samples: {len(X_train):,}")

        # Feature breakdown
        entropy_features = [col for col in X_train.columns if 'entropy' in col.lower()]
        micro_features = [col for col in X_train.columns if any(x in col for x in ['corwin_schultz', 'becker', 'roll', 'amihud', 'vpin', 'oir', 'kyle'])]
        other_features = [col for col in X_train.columns if col not in entropy_features + micro_features]

        logger.info(f"\nFeature breakdown:")
        logger.info(f"   Entropy: {len(entropy_features)}")
        logger.info(f"   Microstructure: {len(micro_features)}")
        logger.info(f"   Others: {len(other_features)}")

        log_memory("feature engineering")
        logger.info(f"Step 7 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 8: SAMPLE WEIGHTS
        # ========================================================================

        logger.info("\nSTEP 8: CALCULATING SAMPLE WEIGHTS")
        step_start = time.time()

        from ml_pipeline.models.sample_weights_numba import SampleWeightsCalculator

        weights_calculator = SampleWeightsCalculator()

        weights_results = weights_calculator.calculate_sample_weights(
            triple_barrier_events_IS,
            series_prim,
            events_IS,
            apply_time_decay=True,
            decay_rate=0.999
        )

        normalized_weights_pure = weights_results['normalized_weights']

        diagnostics = weights_results['diagnostics']
        logger.info(f"Sample Weights Statistics:")
        logger.info(f"   - Calculation time: {diagnostics['calculation_time']:.3f}s")
        logger.info(f"   - Valid events: {diagnostics['valid_events']}/{diagnostics['total_events']}")
        logger.info(f"   - Average uniqueness: {diagnostics['avg_uniqueness']:.4f}")

        log_memory("sample weights")
        logger.info(f"Step 8 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 9: GENERATE EDA REPORT
        # ========================================================================

        logger.info("\nSTEP 9: GENERATING EDA REPORT")
        step_start = time.time()

        generate_eda_report(X_train, y_train, report_dir, experiment_metadata)

        logger.info(f"Step 9 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # STEP 10: SAVE FEATURES
        # ========================================================================

        logger.info("\nSTEP 10: SAVING FEATURES")
        step_start = time.time()

        # Save X_train
        X_train_path = os.path.join(features_dir, 'X_train.parquet')
        X_train.to_parquet(X_train_path)
        logger.info(f"   Saved: {X_train_path}")

        # Save y_train
        y_train_path = os.path.join(features_dir, 'y_train.parquet')
        y_train.to_parquet(y_train_path)
        logger.info(f"   Saved: {y_train_path}")

        # Save sample weights
        weights_df = pd.DataFrame({
            'weight': normalized_weights_pure[:len(X_train)]
        }, index=X_train.index)
        weights_path = os.path.join(features_dir, 'sample_weights.parquet')
        weights_df.to_parquet(weights_path)
        logger.info(f"   Saved: {weights_path}")

        # Save feature names
        feature_names = {
            'all_features': X_train.columns.tolist(),
            'entropy_features': entropy_features,
            'microstructure_features': micro_features,
            'other_features': other_features,
            'n_features': len(X_train.columns)
        }
        feature_names_path = os.path.join(features_dir, 'feature_names.json')
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        logger.info(f"   Saved: {feature_names_path}")

        # Save triple barrier events (needed for PurgedKFold)
        triple_barrier_path = os.path.join(features_dir, 'triple_barrier_events.parquet')
        triple_barrier_events_IS.to_parquet(triple_barrier_path)
        logger.info(f"   Saved: {triple_barrier_path}")

        log_memory("save features")
        logger.info(f"Step 10 completed in {time.time() - step_start:.2f} seconds")

        # ========================================================================
        # FINAL SUMMARY
        # ========================================================================

        pipeline_elapsed = time.time() - pipeline_start_time

        logger.info("\n" + "="*70)
        logger.info("FEATURE GENERATION COMPLETE")
        logger.info("="*70)

        logger.info(f"\nOutput directory: {OUTPUT_PATH}")
        logger.info(f"\nFiles generated:")
        logger.info(f"  features/")
        logger.info(f"    X_train.parquet           - {len(X_train):,} samples, {len(X_train.columns)} features")
        logger.info(f"    y_train.parquet           - Target labels")
        logger.info(f"    sample_weights.parquet    - Sample weights")
        logger.info(f"    feature_names.json        - Feature metadata")
        logger.info(f"    triple_barrier_events.parquet - For PurgedKFold")
        logger.info(f"  report/")
        logger.info(f"    feature_eda.html       - Interactive HTML report")
        logger.info(f"    feature_statistics.csv    - Descriptive statistics")
        logger.info(f"    correlation_*.csv/png     - Correlation analysis")
        logger.info(f"    target_distribution.png   - Target distribution")
        logger.info(f"    missing_values.png        - Missing values analysis")
        logger.info(f"    outliers_*.csv/png        - Outliers analysis")
        logger.info(f"    feature_distributions/    - Histograms by category")

        logger.info(f"\nTotal time: {pipeline_elapsed:.2f} seconds ({pipeline_elapsed/60:.2f} minutes)")
        logger.info(f"Final memory: {get_memory_usage():.1f} MB")

        logger.info("\nNext step: Run rf_trainer.py with this experiment folder")
        logger.info(f"  python src/rf_trainer.py --features {OUTPUT_PATH}/features/")

        return {
            'output_path': OUTPUT_PATH,
            'features_dir': features_dir,
            'report_dir': report_dir,
            'X_train': X_train,
            'y_train': y_train,
            'sample_weights': normalized_weights_pure[:len(X_train)]
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
    try:
        logger.info("Starting Feature Generator execution")
        results = main()
        logger.info("\nFeature generation completed successfully!")
        logger.info(f"Features saved to: {results['features_dir']}")
        logger.info(f"EDA Report: {results['report_dir']}/feature_eda.html")
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}")
        sys.exit(1)
