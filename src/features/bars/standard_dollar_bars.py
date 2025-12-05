#!/usr/bin/env python3
"""
Standard Dollar Bars Generator - Optimized Version

This module generates "standard dollar bars" from Bitcoin trade data.
Uses PyArrow chunked reading and Numba optimizations for memory-efficient processing.

ARCHITECTURE:
============

Mode 1: Sequential (use_pipeline=False) - Simple and Stable
------------------------------------------------------------
  Read Chunk → Pre-process → Generate Bars → Next Chunk
  (all sequential, single thread)

Mode 2: Hybrid Pipeline (use_pipeline=True) - ~1.5-2x Faster
--------------------------------------------------------------
  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
  │ Thread 1    │───▶│  Thread 2    │───▶│  Main Thread    │
  │ I/O: Read   │    │  Pre-process │    │  Generate Bars  │
  │ Chunk N+2   │    │  Chunk N+1   │    │  Chunk N        │
  └─────────────┘    └──────────────┘    └─────────────────┘

  Stage 1 (Thread 1): Read chunks from disk (PyArrow)
  Stage 2 (Thread 2): Calculate side & net_volumes
  Stage 3 (Main):     Generate bars using Numba (stateful)

  All stages run in parallel, processing different chunks simultaneously.

MEMORY USAGE:
============
- Sequential: ~1-2GB per chunk
- Pipeline: ~3-4GB (2-3 chunks in flight)
- Much better than Dask (which needed 30-50GB and crashed)
"""

import os
import datetime
import logging
import time
from pathlib import Path
import gc

import numpy as np
import pandas as pd

from numba import njit, types
from numba.typed import List

from sqlalchemy import create_engine

# ==============================================================================
# SEÇÃO DE FUNÇÕES UTILITÁRIAS INTEGRADAS
# ==============================================================================

# Pegando as credenciais do ambiente (ou defina diretamente para testar)
host = 'localhost'
port = '5432'
dbname = 'superset'
user = 'superset'
password = 'superset'

# Criar a URL de conexão para o SQLAlchemy
db_url = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

def setup_logging():
    """Configura o sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# DEPRECATED: This function is no longer used
# We now use PyArrow's chunked reading instead of Dask for better memory efficiency
#
# def setup_dask_client(n_workers=None, threads_per_worker=None, memory_limit=None):
#     """Configura o cliente Dask para processamento distribuído com otimização automática de CPU."""
#     ...

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
    # Encontra a raiz do projeto subindo três níveis (src/features/bars -> raiz)
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    # Usa o diretório 'data' na raiz do projeto
    data_dir = project_root / 'data'

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Build ticker-based path: data/btcusdt-spot/raw-parquet-merged-daily/
    ticker_name = f"{symbol.lower()}-{data_type}"
    if data_type == 'futures':
        ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

    return data_dir / ticker_name / f'raw-parquet-merged-{granularity}'


def read_parquet_in_chunks(file_path, chunk_size=50_000_000, use_async=False, prefetch_size=2):
    """
    Reads a Parquet file in memory-efficient chunks using PyArrow.
    Returns an iterator of pandas DataFrames.

    Args:
        file_path: Path to the Parquet file
        chunk_size: Number of rows per chunk (default: 50M rows ≈ 1-2GB RAM)
        use_async: Enable async prefetching (default: False)
        prefetch_size: Number of chunks to prefetch (default: 2)

    Yields:
        pandas DataFrame chunks
    """
    import pyarrow.parquet as pq

    if not use_async:
        # Synchronous version (original)
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(
            batch_size=chunk_size,
            columns=['price', 'qty', 'quoteQty', 'time', 'isBuyerMaker']
        ):
            df = batch.to_pandas()
            df['price'] = df['price'].astype('float32')
            df['qty'] = df['qty'].astype('float32')
            df['quoteQty'] = df['quoteQty'].astype('float32')
            df['isBuyerMaker'] = df['isBuyerMaker'].astype('bool')
            yield df
    else:
        # Async version with prefetching
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading

        chunk_queue = Queue(maxsize=prefetch_size)
        stop_event = threading.Event()

        def read_chunks():
            """Background thread: reads chunks from disk."""
            try:
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(
                    batch_size=chunk_size,
                    columns=['price', 'qty', 'quoteQty', 'time', 'isBuyerMaker']
                ):
                    if stop_event.is_set():
                        break

                    # Convert and optimize types
                    df = batch.to_pandas()
                    df['price'] = df['price'].astype('float32')
                    df['qty'] = df['qty'].astype('float32')
                    df['quoteQty'] = df['quoteQty'].astype('float32')
                    df['isBuyerMaker'] = df['isBuyerMaker'].astype('bool')

                    chunk_queue.put(df)

                chunk_queue.put(None)  # Sentinel to signal completion
            except Exception as e:
                chunk_queue.put(e)

        # Start background reader thread
        reader_thread = threading.Thread(target=read_chunks, daemon=True)
        reader_thread.start()

        try:
            while True:
                chunk = chunk_queue.get()

                if chunk is None:  # End of file
                    break

                if isinstance(chunk, Exception):  # Error occurred
                    raise chunk

                yield chunk
        finally:
            stop_event.set()
            reader_thread.join(timeout=5)

def process_partition_data(df):
    """
    Atribui 'side' com base na mudança de preço e calcula 'net_volumes'
    for a single data partition (Pandas DataFrame).
    """
    df['side'] = np.where(df['price'].shift() > df['price'], 1,
                          np.where(df['price'].shift() < df['price'], -1, np.nan))

    df['side'] = df['side'].ffill().fillna(1).astype('int8')
    df['net_volumes'] = df['quoteQty'] * df['side']
    return df

# ==============================================================================
# SEÇÃO DO ALGORITMO DE GERAÇÃO DE BARRAS (NUMBA)
# ==============================================================================

@njit(
    # Assinatura de Retorno: (Lista de Barras, Estado Final do Sistema)
    types.Tuple((
        # 1. Lista de Barras: cada barra é uma tupla de 12 elementos
        # start_time(int64), end_time(int64), open, high, low, close, theta_k, buy_vol, total_vol_usd, total_vol, ticks, ticks_buy
        types.ListType(types.Tuple((
            types.int64, types.int64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64
        ))),
        # 2. Estado Final do Sistema: uma tupla com 12 elementos
        # bar_open, high, low, close, start_time(int64), end_time(int64), imbalance, buy_vol, total_vol_usd, total_vol, ticks, ticks_buy
        types.Tuple((
            types.float64, types.float64, types.float64, types.float64,
            types.int64, types.int64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64
        )),
    ))(
        # Assinaturas dos Argumentos de Entrada
        types.float64[:],       # prices
        types.int64[:],         # times (agora int64 - timestamps em milissegundos)
        types.float64[:],       # net_volumes
        types.int8[:],          # sides
        types.float64[:],       # qtys
        # Estado do sistema (12 elementos)
        types.Tuple((
            types.float64, types.float64, types.float64, types.float64,
            types.int64, types.int64, types.float64, types.float64,
            types.float64, types.float64, types.float64, types.float64
        )),
        types.float64           # init_vol (limiar fixo)
    )
)
def process_partition_numba(
    prices, times, net_volumes, sides, qtys, system_state, init_vol
):
    """Processes a data partition with Numba to generate bars."""
    bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time, \
    current_imbalance, buy_volume_usd, total_volume_usd, total_volume, \
    ticks, ticks_buy = system_state

    threshold = init_vol
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

        if total_volume_usd >= threshold:
            bar_end_time = times[i]

            bars.append((
                bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                current_imbalance, buy_volume_usd, total_volume_usd, total_volume,
                ticks, ticks_buy
            ))

            # Reseta o estado para a próxima barra
            bar_open, bar_high, bar_low, bar_close = np.nan, -np.inf, np.inf, np.nan
            bar_start_time, bar_end_time = 0, 0  # int64: 0 indica sem valor
            current_imbalance, buy_volume_usd, total_volume_usd, total_volume, ticks, ticks_buy = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    final_state = (
        bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time,
        current_imbalance, buy_volume_usd, total_volume_usd, total_volume,
        ticks, ticks_buy
    )
    return bars, final_state

# ==============================================================================
# SEÇÃO DE ORQUESTRAÇÃO E PROCESSAMENTO PRINCIPAL
# ==============================================================================

def create_dollar_bars_numba(partition, system_state, init_vol):
    """Função wrapper para processar uma partição com Numba."""
    prices = partition['price'].values.astype(np.float64)
    # Converte datetime64[ms] para timestamp em milissegundos (int64)
    times = partition['time'].values.astype('datetime64[ms]').astype(np.int64)
    net_volumes = partition['net_volumes'].values.astype(np.float64)
    sides = partition['side'].values.astype(np.int8)
    qtys = partition['qty'].values.astype(np.float64)

    bars, system_state = process_partition_numba(
        prices, times, net_volumes, sides, qtys,
        system_state, init_vol
    )

    if bars:
        # A lista de colunas agora tem 12 itens para corresponder à tupla da barra
        df_bars = pd.DataFrame(bars, columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'theta_k', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume',
            'ticks', 'ticks_buy'
        ])
    else:
        df_bars = pd.DataFrame()
    return df_bars, system_state


def process_file_in_chunks(file_path, system_state, init_vol, chunk_size=50_000_000, use_pipeline=False):
    """
    Processes a single Parquet file in memory-efficient chunks.

    Args:
        file_path: Path to the Parquet file
        system_state: Current state of the bar generation system
        init_vol: Volume threshold for bar generation
        chunk_size: Rows per chunk (default: 50M rows ≈ 1-2GB RAM)
        use_pipeline: Enable 3-stage pipeline (I/O → Pre-process → Generate bars)

    Returns:
        Tuple of (bars_dataframe, final_system_state)
    """
    if not use_pipeline:
        # Original synchronous version
        results = []
        chunk_num = 0

        for chunk in read_parquet_in_chunks(file_path, chunk_size, use_async=False):
            chunk_num += 1
            logging.info(f'  └─ Processing chunk {chunk_num} ({len(chunk):,} rows)')

            try:
                # Process data: add side and net_volumes columns
                chunk = process_partition_data(chunk)

                # Generate bars using Numba
                bars, system_state = create_dollar_bars_numba(
                    chunk, system_state, init_vol
                )

                if not bars.empty:
                    results.append(bars)
                    logging.info(f'     Generated {len(bars)} bars from chunk {chunk_num}')

                # Force garbage collection after each chunk
                del chunk
                gc.collect()

            except Exception as e:
                logging.error(f'❌ Error processing chunk {chunk_num}: {e}')
                import traceback
                logging.error(traceback.format_exc())
                continue

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), system_state

    else:
        # Hybrid version: 3-stage pipeline
        from concurrent.futures import ThreadPoolExecutor
        from queue import Queue
        import threading

        results = []
        chunk_num = 0

        # Pipeline queues
        preprocessed_queue = Queue(maxsize=2)
        stop_event = threading.Event()

        def preprocess_worker():
            """Stage 2: Pre-processes chunks in background thread."""
            try:
                for chunk in read_parquet_in_chunks(file_path, chunk_size, use_async=True, prefetch_size=2):
                    if stop_event.is_set():
                        break

                    # Add side and net_volumes columns
                    chunk = process_partition_data(chunk)
                    preprocessed_queue.put(chunk)

                preprocessed_queue.put(None)  # Sentinel
            except Exception as e:
                preprocessed_queue.put(e)

        # Start preprocessing thread
        preprocess_thread = threading.Thread(target=preprocess_worker, daemon=True)
        preprocess_thread.start()

        try:
            while True:
                chunk = preprocessed_queue.get()

                if chunk is None:  # End of file
                    break

                if isinstance(chunk, Exception):  # Error occurred
                    raise chunk

                chunk_num += 1
                logging.info(f'  └─ Processing chunk {chunk_num} ({len(chunk):,} rows)')

                try:
                    # Stage 3: Generate bars (sequential, stateful)
                    bars, system_state = create_dollar_bars_numba(
                        chunk, system_state, init_vol
                    )

                    if not bars.empty:
                        results.append(bars)
                        logging.info(f'     Generated {len(bars)} bars from chunk {chunk_num}')

                    # Force garbage collection
                    del chunk
                    gc.collect()

                except Exception as e:
                    logging.error(f'❌ Error processing chunk {chunk_num}: {e}')
                    import traceback
                    logging.error(traceback.format_exc())
                    continue

        finally:
            stop_event.set()
            preprocess_thread.join(timeout=5)

        return pd.concat(results, ignore_index=True) if results else pd.DataFrame(), system_state


def process_files_and_generate_bars(
    data_type='futures', futures_type='um', granularity='daily',
    init_vol=40_000_000, output_dir=None, db_engine=None, use_pipeline=False,
    symbol='BTCUSDT'
):
    """
    Função principal que orquestra todo o processo.

    Args:
        data_type: Type of market data ('spot' or 'futures')
        futures_type: Type of futures ('um' or 'cm')
        granularity: Data granularity ('daily' or 'monthly')
        init_vol: Volume threshold for bar generation
        output_dir: Base output directory (will create ticker subdirectory)
        db_engine: SQLAlchemy engine for database writing (optional)
        use_pipeline: Enable 3-stage pipeline for better performance (default: False)
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    """
    # Build ticker-based output directory inside data/
    # Example: data/btcusdt-futures-um/output/standard/
    ticker_name = f"{symbol.lower()}-{data_type}"
    if data_type == 'futures':
        ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

    # Find project root
    project_root = Path(__file__).resolve().parent.parent.parent.parent

    if output_dir is None:
        output_dir = project_root / 'data' / ticker_name / 'output' / 'standard'
    else:
        output_dir = Path(output_dir) / ticker_name / 'output' / 'standard'

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
    output_file_prefix = f'{data_type}-volume{init_vol}'
    output_file_prefix_parquet = f'{timestamp}-standard-{data_type}-volume{init_vol}'

    # O estado inicial agora tem 12 elementos, sem ewma_T, ewma_imbalance e warm
    # start_time e end_time agora são int64 (0 indica sem valor)
    system_state = (
        np.nan, -np.inf, np.inf, np.nan, 0, 0,  # bar_open, high, low, close, start_time(int64), end_time(int64) (6)
        0.0, 0.0, 0.0, 0.0,                     # current_imbalance, buy_volume_usd, total_volume_usd, total_volume (4)
        0.0, 0.0                                # ticks, ticks_buy (2)
    )

    results = []

    for i, file in enumerate(files):
        logging.info(f"Processing file {i + 1}/{len(files)}: {file}")
        start_time_file = time.time()

        file_path = raw_dataset_path / file

        # Process file in memory-efficient chunks (no Dask needed)
        bars, system_state = process_file_in_chunks(
            file_path, system_state, init_vol, chunk_size=50_000_000, use_pipeline=use_pipeline
        )

        if not bars.empty:
            results.append(bars)
            if db_engine:
                bars_to_db = bars.copy()
                # Converte timestamps de milissegundos para datetime
                bars_to_db['end_time'] = pd.to_datetime(bars_to_db['end_time'], unit='ms')
                bars_to_db['start_time'] = pd.to_datetime(bars_to_db['start_time'], unit='ms')
                bars_to_db.drop(columns=['start_time'], inplace=True)
                bars_to_db['type'] = 'Standard'
                bars_to_db['sample_date'] = timestamp
                bars_to_db['sample'] = output_file_prefix
                with db_engine.connect() as conn:
                    bars_to_db.to_sql(
                        name='dollar_bars',
                        con=conn,
                        if_exists='append',
                        index=False
                    )
                    conn.commit()
                print("✅ Dados inseridos com sucesso no banco!")

            elapsed_time = (time.time() - start_time_file) / 60
            logging.info(f"{len(bars)} bars generated. Tempo: {elapsed_time:.2f} min.")
        else:
            logging.warning(f"No bars generated for file {file}")

    if results:
        all_bars = pd.concat(results, ignore_index=True)
        # Converte timestamps de milissegundos para datetime
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
            analysis_script = Path(__file__).resolve().parent.parent.parent.parent / 'notebooks' / 'global_analysis.py'
            # Update the dataset path in global_analysis.py
            with open(analysis_script, 'r') as f:
                content = f.read()

            # Replace the dataset_path line
            import re
            content = re.sub(
                r'dataset_path = "[^"]*"',
                f'dataset_path = "{final_path}"',
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
    output_dir = './output/standard/'

    # PERFORMANCE OPTIONS:
    # use_pipeline=False: Simple sequential processing (default, stable)
    # use_pipeline=True: 3-stage pipeline (I/O → Pre-process → Generate bars)
    #                    Expected speedup: 1.5-2x
    USE_PIPELINE = True  # ← Change to True to enable hybrid pipeline

    setup_logging()

    if USE_PIPELINE:
        logging.info("🚀 Using HYBRID PIPELINE mode (3-stage: I/O → Pre-process → Generate)")
    else:
        logging.info("📝 Using SEQUENTIAL mode (simple, stable)")

    # NOTE: Dask is NO LONGER NEEDED for file reading
    # We now use PyArrow's chunked reading which is more memory-efficient
    # and doesn't require distributed workers

    # Uncomment the line below to enable database writing
    # engine = create_engine(db_url)
    engine = None

    try:
        # Loop para testar diferentes limiares de volume em USD
        for volume_usd_trig in range(40_000_000, 45_000_000, 5_000_000):
            print(f'\n# INICIANDO AMOSTRA COM VOLUME USD: {volume_usd_trig:,}')
            start_time_sample = time.time()

            process_files_and_generate_bars(
                data_type=data_type, futures_type=futures_type, granularity=granularity,
                init_vol=volume_usd_trig, output_dir=output_dir, db_engine=engine,
                use_pipeline=USE_PIPELINE
            )

            end_time_sample = time.time()
            logging.info(f"Tempo total da amostra: {(end_time_sample - start_time_sample) / 60:.2f} minutos.")
            logging.info("Forçando a coleta de lixo antes da próxima amostra...")
            gc.collect()
    finally:
        if engine is not None:
            logging.info("Disposing database engine.")
            engine.dispose()