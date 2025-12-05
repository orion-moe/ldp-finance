"""
Data loading module for the optimized pipeline.
Handles loading and initial processing of parquet files.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple


def find_data_file(data_path: str, file_name: str) -> str:
    """
    Find data file in folder structure or direct path.

    Args:
        data_path: Base data directory path
        file_name: Name of the file to find

    Returns:
        Full path to the file

    Raises:
        FileNotFoundError: If file cannot be found
    """
    logger = logging.getLogger(__name__)

    # Try folder structure first (new structure)
    file_base_name = file_name.replace('.parquet', '')
    folder_path = os.path.join(data_path, file_base_name)
    direct_path = os.path.join(data_path, file_name)

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # New structure: file is inside a folder
        filepath = os.path.join(folder_path, file_name)
        logger.info(f"Found file in folder structure: {folder_path}/")
        return filepath

    elif os.path.exists(direct_path):
        # Old structure: file is directly in the output folder
        logger.info(f"Found file in direct path (legacy structure)")
        return direct_path

    else:
        logger.error(f"File not found in either structure!")
        logger.error(f"  Tried folder: {folder_path}")
        logger.error(f"  Tried direct: {direct_path}")
        raise FileNotFoundError(f"Could not find {file_name}")


def load_parquet_data(filepath: str) -> pd.DataFrame:
    """
    Load parquet file and return as DataFrame.

    Args:
        filepath: Path to parquet file

    Returns:
        DataFrame with loaded data

    Raises:
        Exception: If loading fails
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Loading data from {filepath}...")

    try:
        df = pd.read_parquet(filepath)
        logger.info(f"✅ Data loaded: {len(df):,} records")
        logger.debug(f"Data columns: {list(df.columns)}")
        logger.debug(f"Data shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Failed to load data from {filepath}: {str(e)}")
        raise


def aggregate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by end_time using predefined rules.

    Args:
        df: Input DataFrame

    Returns:
        Aggregated DataFrame
    """
    logger = logging.getLogger(__name__)

    logger.info("Performing data aggregation...")

    # Define aggregation rules
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
    existing_columns = df.columns
    agg_rules_filtered = {col: func for col, func in agg_rules.items() if col in existing_columns}
    logger.debug(f"Aggregation rules applied: {list(agg_rules_filtered.keys())}")

    # Aggregate
    df_agg = df.groupby('end_time').agg(agg_rules_filtered)
    df_agg = df_agg[:-1]  # Remove incomplete last row

    # Add log-prices
    df_agg['log_close'] = np.log(df_agg['close'])

    logger.info(f"✅ Data aggregated: {len(df_agg):,} records")

    return df_agg


def load_and_prepare_data(data_path: str, file_name: str) -> pd.DataFrame:
    """
    Main function to load and prepare data for pipeline.

    Args:
        data_path: Base data directory path
        file_name: Name of the file to load

    Returns:
        Prepared DataFrame ready for processing

    Raises:
        FileNotFoundError: If file cannot be found
        Exception: If loading or processing fails
    """
    logger = logging.getLogger(__name__)

    # Find file
    filepath = find_data_file(data_path, file_name)

    # Load data
    df = load_parquet_data(filepath)

    # Aggregate data
    df_prepared = aggregate_data(df)

    return df_prepared


def add_fractional_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fractional price representation (for feature engineering).

    Args:
        df: DataFrame with 'close' column

    Returns:
        DataFrame with added 'fraq_close' column
    """
    df = df.copy()

    # Simple fractional representation (can be enhanced)
    # This extracts the decimal part of the price
    df['fraq_close'] = df['close'] % 1.0

    return df


def prepare_for_feature_engineering(df: pd.DataFrame) -> dict:
    """
    Prepare data structure for feature engineering pipeline.

    Args:
        df: Prepared DataFrame

    Returns:
        Dictionary with data structures needed for feature engineering
    """
    # Create dictionary structure similar to original code
    data_dict = {
        'BTCUSDT': df[['close', 'total_volume', 'total_volume_usd']].copy()
    }

    # Rename columns to match expected format
    if 'fraq_close' in df.columns:
        data_dict['BTCUSDT']['fraq_close'] = df['fraq_close']

    return data_dict
