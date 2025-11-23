#!/usr/bin/env python3
"""
Imbalance Dollar Bars Generator - PyArrow Version

This module generates "imbalance dollar bars" from Bitcoin trade data.
Uses PyArrow chunked reading and Numba optimizations for memory-efficient processing.
"""

import os
import datetime
import logging
import time
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from numba import njit, types
from numba.typed import List

from sqlalchemy import create_engine

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================

host = 'localhost'
port = '5432'
dbname = 'superset'
user = 'superset'
password = 'superset'

db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(db_url)

def setup_logging():
    """Configure logging system."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def get_data_path(data_type='futures', futures_type='um', granularity='daily', symbol='BTCUSDT'):
    """
    Builds the data path in a portable way, relative to the script.
    Uses the new ticker-based directory structure.

    Args:
        data_type: Type of market data ('spot' or 'futures')
        futures_type: Type of futures ('um' or 'cm')
        granularity: Data granularity ('daily' or 'monthly')
        symbol: Trading pair symbol (e.g., 'BTCUSDT')

    Returns:
        Path to the data directory containing parquet files
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = project_root / 'data'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Build ticker-based path: data/btcusdt-spot/raw-parquet-merged-daily/
    ticker_name = f"{symbol.lower()}-{data_type}"
    if data_type == 'futures':
        ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

    return data_dir / ticker_name / f'raw-parquet-merged-{granularity}'


def read_parquet_chunked(file_path, chunk_size=500_000):
    """
    Reads a Parquet file in memory-efficient chunks using PyArrow.

    Args:
        file_path: Path to the parquet file
        chunk_size: Number of rows per chunk

    Yields:
        pandas DataFrames with the required columns
    """
    parquet_file = pq.ParquetFile(file_path)

    columns = ['price', 'qty', 'quoteQty', 'time', 'isBuyerMaker']

    for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
        df = batch.to_pandas()

        # Convert types for memory efficiency
        df['price'] = df['price'].astype('float32')
        df['qty'] = df['qty'].astype('float32')
        df['quoteQty'] = df['quoteQty'].astype('float32')

        yield df


def process_chunk_data(df):
    """
    Assigns 'side' based on price change and calculates 'net_volumes'
    for a single data chunk (Pandas DataFrame).
    """
    # 1. Assign trade side
    df['side'] = np.where(df['price'].shift() > df['price'], 1,
                          np.where(df['price'].shift() < df['price'], -1, np.nan))

    # Fill null values
    df['side'] = df['side'].ffill().fillna(1).astype('int8')

    # 2. Calculate net volumes
    df['net_volumes'] = df['quoteQty'] * df['side']

    return df


# ==============================================================================
# BAR GENERATION ALGORITHM (NUMBA)
# ==============================================================================

@njit(
    types.Tuple((
        types.ListType(types.Tuple((
            types.int64, types.int64, types.float64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64
        ))),
        types.Tuple((
            types.float64, types.float64, types.float64, types.float64, types.int64,
            types.int64, types.float64, types.float64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64, types.boolean
        )),
    ))(
        types.float64[:],
        types.int64[:],
        types.float64[:],
        types.int8[:],
        types.float64[:],
        types.float64,
        types.float64,
        types.Tuple((
            types.float64, types.float64, types.float64, types.float64, types.int64,
            types.int64, types.float64, types.float64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64, types.boolean
        )),
        types.float64,
        types.float64
    )
)
def process_chunk_imbalance_numba(
    prices, times, net_volumes, sides, qtys, alpha_ticks, alpha_imbalance, system_state, init_ticks, time_reset
):
    """Processes a data chunk with Numba to generate bars."""
    bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time, \
    current_imbalance, buy_volume_usd, total_volume_usd, total_volume, \
    ticks, ticks_buy, ewma_T, ewma_imbalance, warm = system_state

    if warm:
        threshold = init_ticks
    else:
        threshold = ewma_T * ewma_imbalance

    time_reset_ms = 3_600_000.0 * time_reset

    bars = List()

    for i in range(len(prices)):
        ticks += 1

        if np.isnan(bar_open):
            bar_open = prices[i]
            bar_start_time = times[i]

        trade_price = prices[i]
        bar_high = max(bar_high, trade_price)
        bar_low = min(bar_low, trade_price)
        bar_close = trade_price

        if sides[i] > 0:
            buy_volume_usd += net_volumes[i]
            ticks_buy += 1

        total_volume += qtys[i]
        total_volume_usd += abs(net_volumes[i])
        current_imbalance += net_volumes[i]

        time_since_bar_start = times[i] - bar_start_time

        if warm:
            trigger_var = ticks
        else:
            trigger_var = abs(current_imbalance)

        if trigger_var >= threshold:
            bar_end_time = times[i]

            if warm:
                ewma_T = ticks
                ewma_imbalance = abs(current_imbalance) / ticks
                warm = False
            else:
                ewma_T += alpha_ticks * (ticks - ewma_T)
                ewma_imbalance += alpha_imbalance * (trigger_var / ticks - ewma_imbalance)

            bars.append((
                bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                current_imbalance, buy_volume_usd, total_volume_usd, total_volume,
                ticks, ticks_buy, ewma_T, ewma_imbalance
            ))

            bar_open, bar_high, bar_low, bar_close = np.nan, -np.inf, np.inf, np.nan
            bar_start_time, bar_end_time = 0, 0
            current_imbalance, buy_volume_usd, total_volume_usd, total_volume, ticks, ticks_buy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            if time_since_bar_start > time_reset_ms:
                warm = True
                threshold = init_ticks
            else:
                threshold = ewma_T * ewma_imbalance

    final_state = (
        bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time,
        current_imbalance, buy_volume_usd, total_volume_usd, total_volume,
        ticks, ticks_buy, ewma_T, ewma_imbalance, warm
    )
    return bars, final_state


def create_imbalance_dollar_bars_numba(chunk, system_state, alpha_ticks, alpha_imbalance, init_ticks, time_reset):
    """Wrapper function to process a chunk with Numba."""
    prices = chunk['price'].values.astype(np.float64)
    times = chunk['time'].values.astype('datetime64[ms]').astype(np.int64)
    net_volumes = chunk['net_volumes'].values.astype(np.float64)
    sides = chunk['side'].values.astype(np.int8)
    qtys = chunk['qty'].values.astype(np.float64)

    bars, system_state = process_chunk_imbalance_numba(
        prices, times, net_volumes, sides, qtys,
        alpha_ticks, alpha_imbalance, system_state, init_ticks, time_reset
    )

    if bars:
        df_bars = pd.DataFrame(bars, columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'theta_k', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume',
            'ticks', 'ticks_buy', 'ewma_T', 'ewma_imbalance'
        ])
    else:
        df_bars = pd.DataFrame()
    return df_bars, system_state


def process_file_chunks(file_path, system_state, alpha_ticks, alpha_imbalance, init_ticks, time_reset, chunk_size=500_000):
    """Process a single file in chunks."""
    results = []
    chunk_num = 0

    for chunk in read_parquet_chunked(file_path, chunk_size):
        chunk_num += 1

        # Process chunk data
        chunk = process_chunk_data(chunk)

        # Generate bars
        bars, system_state = create_imbalance_dollar_bars_numba(
            chunk, system_state, alpha_ticks, alpha_imbalance, init_ticks, time_reset
        )

        if not bars.empty:
            results.append(bars)

        # Clean up
        del chunk
        gc.collect()

    if results:
        return pd.concat(results, ignore_index=True), system_state
    else:
        return pd.DataFrame(), system_state


def process_files_and_generate_bars(
    data_type='futures', futures_type='um', granularity='daily',
    init_ticks=1_000, alpha_ticks=0.9,
    alpha_imbalance=0.9, output_dir=None, time_reset=5.0, db_engine=None,
    symbol='BTCUSDT'
):
    """Main function that orchestrates the entire process."""
    # Build ticker-based output directory inside data/
    ticker_name = f"{symbol.lower()}-{data_type}"
    if data_type == 'futures':
        ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

    project_root = Path(__file__).resolve().parent.parent.parent.parent

    if output_dir is None:
        output_dir = project_root / 'data' / ticker_name / 'output' / 'imbalance'
    else:
        output_dir = Path(output_dir) / ticker_name / 'output' / 'imbalance'

    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    raw_dataset_path = get_data_path(data_type, futures_type, granularity, symbol)

    logging.info(f"Data path: {raw_dataset_path}")

    if not raw_dataset_path.exists():
        logging.error(f"Data directory not found: {raw_dataset_path}")
        return

    files = sorted([f for f in os.listdir(raw_dataset_path) if f.endswith('.parquet')])
    if not files:
        logging.error("No Parquet files found!")
        return
    logging.info(f"Found {len(files)} Parquet files")

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file_prefix = f'{data_type}-ticks{init_ticks}-aticks{alpha_ticks}-aimb{alpha_imbalance}-treset{time_reset}'
    output_file_prefix_parquet = f'{timestamp}-imbalance-{data_type}-ticks{init_ticks}-aticks{alpha_ticks}-aimb{alpha_imbalance}-treset{time_reset}'

    # Initial state (15 elements)
    system_state = (
        np.nan, -np.inf, np.inf, np.nan, 0, 0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0,
        float(init_ticks), 0.0,
        True
    )

    results = []

    for i, file in enumerate(files):
        logging.info(f"Processing file {i + 1}/{len(files)}: {file}")
        start_time_file = time.time()

        file_path = raw_dataset_path / file

        bars, system_state = process_file_chunks(
            file_path, system_state, alpha_ticks, alpha_imbalance, init_ticks, time_reset
        )

        if not bars.empty:
            results.append(bars)
            if db_engine:
                bars['end_time'] = pd.to_datetime(bars['end_time'], unit='ms')
                bars.drop(columns=['start_time'], inplace=True)
                bars['type'] = 'Imbalance'
                bars['sample_date'] = timestamp
                bars['sample'] = output_file_prefix
                with db_engine.connect() as conn:
                    bars.to_sql(
                        name='dollar_bars',
                        con=conn,
                        if_exists='append',
                        index=False
                    )
                    print("Data inserted successfully!")

            elapsed_time = (time.time() - start_time_file) / 60
            logging.info(f"{len(bars)} bars generated. Time: {elapsed_time:.2f} min.")
        else:
            logging.warning(f"No bars generated for file {file}")

        gc.collect()

    if results:
        all_bars = pd.concat(results, ignore_index=True)
        all_bars['end_time'] = pd.to_datetime(all_bars['end_time'], unit='ms')
        all_bars['start_time'] = pd.to_datetime(all_bars['start_time'], unit='ms')
        all_bars.drop(columns=['start_time'], inplace=True)

        # Create a folder with the same name as the output file
        folder_name = output_file_prefix_parquet
        output_folder = output_dir / folder_name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Save the parquet file inside the folder
        final_path = output_folder / f'{output_file_prefix_parquet}.parquet'
        all_bars.to_parquet(final_path, index=False)

        logging.info(f"\nProcessing complete!")
        logging.info(f"📁 Output folder: {output_folder}")
        logging.info(f"📊 Bars file: {final_path}")
        logging.info(f"Total bars in final file: {len(all_bars)}")

        # Ask user if they want to run global_analysis.py
        run_analysis = input("\nDo you want to run global_analysis.py to see the sampling results? (y/n): ").strip().lower()
        if run_analysis == 'y':
            import subprocess
            import re
            analysis_script = Path(__file__).resolve().parent.parent.parent.parent / 'notebooks' / 'global_analysis.py'

            # Update the dataset path in global_analysis.py
            with open(analysis_script, 'r') as f:
                content = f.read()

            # Replace the dataset_path line
            content = re.sub(
                r'dataset_path = "[^"]*"',
                f'dataset_path = "{final_path}"',
                content
            )

            # Change to analyze_imbalance_bars
            content = re.sub(
                r'analyze_standard_bars\(dataset_path\)',
                'analyze_imbalance_bars(dataset_path)',
                content
            )

            with open(analysis_script, 'w') as f:
                f.write(content)

            logging.info(f"Running analysis script: {analysis_script}")
            subprocess.run(['python', str(analysis_script)])

if __name__ == '__main__':
    data_type = 'futures'
    futures_type = 'um'
    granularity = 'daily'
    output_dir = './output/'

    # Single test - reasonable parameters
    params = [[1000, 0.5, 0.5, 5.0]]

    setup_logging()

    engine = None

    try:
        for init_ticks, alpha_ticks, alpha_imbalance, time_reset in params:
            print(f'# SAMPLE Ticks0 = {init_ticks} - Alpha ticks = {alpha_ticks} - Alpha imbalance = {alpha_imbalance} - Time reset = {time_reset}')

            start_time_sample = time.time()

            process_files_and_generate_bars(
                data_type=data_type, futures_type=futures_type, granularity=granularity,
                init_ticks=init_ticks, alpha_ticks=alpha_ticks, alpha_imbalance=alpha_imbalance,
                output_dir=output_dir, time_reset=time_reset, db_engine=engine
            )

            end_time_sample = time.time()
            logging.info(f"Total sample time: {(end_time_sample - start_time_sample) / 60:.2f} minutes.")
            gc.collect()
    finally:
        if engine is not None:
            logging.info("Disposing database engine.")
            engine.dispose()
