"""
Visualization module for model analysis plots.
Generates confusion matrix, ROC curve, and feature importance plots.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from typing import Optional


def setup_plot_style(style: str = 'seaborn-v0_8-darkgrid'):
    """
    Configure matplotlib and seaborn style.

    Args:
        style: Plot style name
    """
    plt.style.use(style)
    sns.set_palette("husl")


def create_plot_directory(output_path: str) -> str:
    """
    Create plot subdirectory in output path.

    Args:
        output_path: Base output directory

    Returns:
        Path to plot directory
    """
    plot_dir = os.path.join(output_path, 'plot')
    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plot_dir: str,
    filename: str = 'confusion_matrix.png',
    dpi: int = 300,
    labels: list = None
) -> str:
    """
    Generate and save confusion matrix plot.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        plot_dir: Directory to save plot
        filename: Output filename
        dpi: Resolution in dots per inch
        labels: Class labels (default: ['Down', 'Up'])

    Returns:
        Path to saved plot
    """
    logger = logging.getLogger(__name__)

    if labels is None:
        labels = ['Down', 'Up']

    logger.info(f"\n1️⃣ Generating Confusion Matrix...")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True,
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix - Random Forest\nTraining Set',
              fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()

    output_path = os.path.join(plot_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"   ✅ Saved: {output_path}")
    return output_path


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    plot_dir: str,
    filename: str = 'roc_curve.png',
    dpi: int = 300
) -> str:
    """
    Generate and save ROC curve plot.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        plot_dir: Directory to save plot
        filename: Output filename
        dpi: Resolution in dots per inch

    Returns:
        Path to saved plot
    """
    logger = logging.getLogger(__name__)

    logger.info(f"\n2️⃣ Generating ROC Curve...")

    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Random Forest\nTraining Set',
              fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(plot_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"   ✅ Saved: {output_path}")
    return output_path


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    plot_dir: str,
    n_features: int = 20,
    ascending: bool = False,
    color: str = 'forestgreen',
    filename: str = 'top_20_best_features.png',
    title: str = 'Top 20 Most Important Features',
    dpi: int = 300
) -> str:
    """
    Generate and save feature importance plot.

    Args:
        feature_importance_df: DataFrame with 'feature' and 'importance' columns
        plot_dir: Directory to save plot
        n_features: Number of features to plot
        ascending: If True, plot least important features
        color: Bar color
        filename: Output filename
        title: Plot title
        dpi: Resolution in dots per inch

    Returns:
        Path to saved plot
    """
    logger = logging.getLogger(__name__)

    step = "3️⃣" if not ascending else "4️⃣"
    logger.info(f"\n{step} Generating {title}...")

    # Select top/bottom features
    if ascending:
        features = feature_importance_df.tail(n_features).sort_values('importance', ascending=True)
    else:
        features = feature_importance_df.head(n_features)

    plt.figure(figsize=(12, 10))
    bars = plt.barh(range(len(features)), features['importance'], color=color)
    plt.yticks(range(len(features)), features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'{title}\nRandom Forest', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, features['importance'])):
        format_str = f' {val:.6f}' if ascending else f' {val:.4f}'
        plt.text(val, i, format_str, va='center', fontsize=9)

    plt.tight_layout()

    output_path = os.path.join(plot_dir, filename)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"   ✅ Saved: {output_path}")
    return output_path


def generate_all_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    feature_importance_df: pd.DataFrame,
    output_path: str,
    dpi: int = 300,
    n_features: int = 20
) -> dict:
    """
    Generate all visualization plots.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        feature_importance_df: Feature importance DataFrame
        output_path: Base output directory
        dpi: Resolution in dots per inch
        n_features: Number of features to plot

    Returns:
        Dictionary with paths to all generated plots
    """
    logger = logging.getLogger(__name__)

    logger.info("\n" + "="*60)
    logger.info("📊 GENERATING VISUALIZATION PLOTS")
    logger.info("="*60)

    # Setup style
    setup_plot_style()

    # Create plot directory
    plot_dir = create_plot_directory(output_path)
    logger.info(f"Plot directory created: {plot_dir}")

    # Generate plots
    plots = {}

    # 1. Confusion Matrix
    plots['confusion_matrix'] = plot_confusion_matrix(
        y_true, y_pred, plot_dir, dpi=dpi
    )

    # 2. ROC Curve
    plots['roc_curve'] = plot_roc_curve(
        y_true, y_pred_proba, plot_dir, dpi=dpi
    )

    # 3. Top N Best Features
    plots['best_features'] = plot_feature_importance(
        feature_importance_df, plot_dir,
        n_features=n_features,
        ascending=False,
        color='forestgreen',
        filename='top_20_best_features.png',
        title=f'Top {n_features} Most Important Features',
        dpi=dpi
    )

    # 4. Top N Worst Features
    plots['worst_features'] = plot_feature_importance(
        feature_importance_df, plot_dir,
        n_features=n_features,
        ascending=True,
        color='crimson',
        filename='top_20_worst_features.png',
        title=f'Top {n_features} Least Important Features',
        dpi=dpi
    )

    logger.info(f"\n✅ All plots saved successfully to: {plot_dir}")

    return plots
