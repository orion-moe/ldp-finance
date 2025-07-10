#!/usr/bin/env python3
"""
Imbalance Dollar Bars Generator
Converts imbalance_dollar_barsv3.ipynb notebook to a standalone Python script
"""

import os
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
import datetime
import logging
import time
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.distributed import Client
from numba import njit, types
from numba.typed import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_dask_client(n_workers=10, threads_per_worker=1, memory_limit='6.4GB'):
    """Setup Dask distributed client"""
    client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    logger.info(client)
    return client

def read_parquet_files_optimized(raw_dataset_path, file):
    """Lê arquivos Parquet de forma otimizada."""
    parquet_pattern = os.path.join(raw_dataset_path, file)
    df_dask = dd.read_parquet(
        parquet_pattern,
        columns=['price', 'qty', 'quoteQty', 'time'],
        engine='pyarrow',
        dtype={'price': 'float32', 'qty': 'float32', 'quoteQty': 'float32'}
    )
    return df_dask

def assign_side_optimized(df):
    """Atribui o lado da negociação com base na mudança de preço."""
    df['side'] = np.where(df['price'].shift() > df['price'], 1,
                          np.where(df['price'].shift() < df['price'], -1, np.nan))
    df['side'] = df['side'].ffill().fillna(1).astype('int8')
    return df

def apply_operations_optimized(df_dask, meta):
    """Aplica operações otimizadas no DataFrame."""
    df_dask = df_dask.map_partitions(assign_side_optimized, meta=meta)
    df_dask['dollar_imbalance'] = df_dask['quoteQty'] * df_dask['side']
    return df_dask

# Função compilada com numba
@njit(
    types.Tuple((
        types.ListType(types.Tuple((
            types.float64,  # start_time
            types.float64,  # end_time
            types.float64,  # open
            types.float64,  # high
            types.float64,  # low
            types.float64,  # close
            types.float64,  # imbalance_col
            types.float64,  # total_volume_buy_usd
            types.float64,  # total_volume_usd
            types.float64   # total_volume
        ))),
        types.float64,  # exp_T
        types.float64,  # exp_dif
        types.Tuple((
            types.float64,  # bar_open
            types.float64,  # bar_high
            types.float64,  # bar_low
            types.float64,  # bar_close
            types.float64,  # bar_start_time
            types.float64,  # bar_end_time
            types.float64,  # current_imbalance
            types.float64,  # buy_volume_usd
            types.float64,  # total_volume_usd
            types.float64   # total_volume
        )),
        types.ListType(types.Tuple((
            types.float64,  # exp_T
            types.float64,  # exp_dif
            types.float64   # thres
        )))
    ))(
        types.float64[:],  # prices
        types.float64[:],  # times
        types.float64[:],  # imbalances
        types.int8[:],     # sides
        types.float64[:],  # qtys
        types.float64,     # init_T
        types.float64,     # init_dif
        types.float64,     # alpha_volume
        types.float64,     # alpha_imbalance
        types.Tuple((
            types.float64,  # bar_open
            types.float64,  # bar_high
            types.float64,  # bar_low
            types.float64,  # bar_close
            types.float64,  # bar_start_time
            types.float64,  # bar_end_time
            types.float64,  # current_imbalance
            types.float64,  # buy_volume_usd
            types.float64,  # total_volume_usd
            types.float64   # total_volume
        ))
    )
)
def process_partition_imbalance_numba(
    prices, times, imbalances, sides, qtys,
    init_T, init_dif, alpha_volume, alpha_imbalance, res_init
):
    """Processa uma partição usando numba para aceleração."""
    exp_T = init_T
    exp_dif = init_dif
    threshold = exp_T * abs(exp_dif)

    bars = List()  # Lista tipada para armazenar as barras formadas
    params = List()

    # Desempacota res_init
    bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time, \
    current_imbalance, buy_volume_usd, total_volume_usd, total_volume = res_init

    # Verifica se res_init está inicializado (usando -1.0 como sentinela para não inicializado)
    if bar_open == -1.0:
        # Reseta as variáveis de agregação
        bar_open = np.nan
        bar_high = -np.inf
        bar_low = np.inf
        bar_close = np.nan
        bar_start_time = np.nan
        bar_end_time = np.nan
        current_imbalance = 0.0
        buy_volume_usd = 0.0
        total_volume_usd = 0.0
        total_volume = 0.0

    for i in range(len(prices)):
        if np.isnan(bar_open):
            bar_open = prices[i]
            bar_start_time = times[i]

        trade_price = prices[i]
        bar_high = max(bar_high, trade_price)
        bar_low = min(bar_low, trade_price)
        bar_close = trade_price

        trade_imbalance = imbalances[i]

        if sides[i] > 0:
            buy_volume_usd += trade_imbalance

        total_volume += qtys[i]
        total_volume_usd += abs(trade_imbalance)
        current_imbalance += trade_imbalance
        imbalance = abs(current_imbalance)

        if imbalance >= threshold:
            bar_end_time = times[i]

            # Salva a barra formada
            bars.append((
                bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                current_imbalance, buy_volume_usd, total_volume_usd, total_volume
            ))

            # Atualiza os valores exponenciais
            if exp_dif == 1.0:
                exp_T = total_volume_usd
                exp_dif = abs(2 * buy_volume_usd / total_volume_usd - 1)
            else:
                exp_T += alpha_volume * (total_volume_usd - exp_T)
                exp_dif += alpha_imbalance * (abs(2 * buy_volume_usd / total_volume_usd - 1) - exp_dif)

            threshold = exp_T * abs(exp_dif)

            params.append((
                exp_T, exp_dif, threshold
            ))

            # Reseta as variáveis de agregação
            bar_open = np.nan
            bar_high = -np.inf
            bar_low = np.inf
            bar_close = np.nan
            bar_start_time = np.nan
            bar_end_time = np.nan
            current_imbalance = 0.0
            buy_volume_usd = 0.0
            total_volume_usd = 0.0
            total_volume = 0.0

    # Prepara o estado final para a próxima partição
    final_state = (
        bar_open, bar_high, bar_low, bar_close,
        bar_start_time, bar_end_time, current_imbalance,
        buy_volume_usd, total_volume_usd, total_volume
    )

    return bars, exp_T, exp_dif, final_state, params

def create_imbalance_dollar_bars_numba(partition, init_T, init_dif, res_init, alpha_volume, alpha_imbalance):
    """Função wrapper para processar uma partição com numba."""
    # Converte a partição para arrays numpy
    prices = partition['price'].values.astype(np.float64)
    times = partition['time'].values.astype(np.float64)
    imbalances = partition['dollar_imbalance'].values.astype(np.float64)
    sides = partition['side'].values.astype(np.int8)
    qtys = partition['qty'].values.astype(np.float64)

    # Inicializa res_init se vazio ou inválido
    if res_init is None or len(res_init) != 10:
        res_init = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0)

    # Processa a partição usando a função compilada com numba
    bars, exp_T, exp_dif, res_init, params = process_partition_imbalance_numba(
        prices, times, imbalances, sides, qtys,
        init_T, init_dif, alpha_volume, alpha_imbalance, res_init
    )

    # Converte as barras para um DataFrame
    if len(bars) > 0:
        bars_df = pd.DataFrame(bars, columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])
        params_df = pd.DataFrame(params, columns=['ewma_volume', 'ewma_dif', 'thres'])
    else:
        # Retorna um DataFrame vazio com as colunas apropriadas
        bars_df = pd.DataFrame(columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])
        params_df = pd.DataFrame(columns=['ewma_volume', 'ewma_dif', 'thres'])

    return bars_df, exp_T, exp_dif, res_init, params_df

def batch_create_imbalance_dollar_bars_optimized(df_dask, init_T, init_dif, res_init, alpha_volume, alpha_imbalance):
    """Processa partições em lote para criar barras de desequilíbrio em dólares."""
    results = []
    params_save = []
    for partition in range(df_dask.npartitions):
        logger.info(f'Processando partição {partition+1} de {df_dask.npartitions}')
        part = df_dask.get_partition(partition).compute()

        bars, init_T, init_dif, res_init, params = create_imbalance_dollar_bars_numba(
            part, init_T, init_dif, res_init, alpha_volume, alpha_imbalance
        )
        results.append(bars)
        params_save.append(params)
    
    # Filtra DataFrames vazios
    non_empty_results = [df for df in results if not df.empty]
    non_empty_params = [df for df in params_save if not df.empty]
    
    if non_empty_results:
        results_df = pd.concat(non_empty_results, ignore_index=True)
    else:
        results_df = pd.DataFrame(columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])
    
    if non_empty_params:
        params_df = pd.concat(non_empty_params, ignore_index=True)
    else:
        params_df = pd.DataFrame(columns=['ewma_volume', 'ewma_dif', 'thres'])
    
    return results_df, init_T, init_dif, res_init, params_df

def generate_initial_states(file_count):
    """Generate initial parameter states for processing"""
    initial_state = [[init_T0, alpha_volume/100, alpha_imbalance/100, number]
                      for init_T0 in range(1_000_000, 40_000_000, 5_000_000)  # Reduced range for efficiency
                      for alpha_volume in range(10, 100, 30)  # Reduced iterations
                      for alpha_imbalance in range(10, 100, 30)  # Reduced iterations
                      for number in range(1, min(file_count, 13))]  # Limit to available files
    
    return initial_state[:min(len(initial_state), file_count-1)]

def process_imbalance_bars(raw_dataset_path, output_path, initial_state, timestamp):
    """Main processing function for imbalance dollar bars"""
    
    # Meta DataFrame for map_partitions
    meta = pd.DataFrame({
        'price': pd.Series(dtype='float32'),
        'qty': pd.Series(dtype='float32'),
        'quoteQty': pd.Series(dtype='float32'),
        'time': pd.Series(dtype='float64'),
        'side': pd.Series(dtype='int8')
    })
    
    # List files
    files = [f for f in os.listdir(raw_dataset_path) if os.path.isfile(os.path.join(raw_dataset_path, f))]
    file_count = len(files)
    
    processing_times = {}
    
    for init_T0, alpha_volume, alpha_imbalance, number in initial_state:
        if number == 1:
            start_time = time.time()
            output_file = f'imbalance_dolar_{init_T0}-{alpha_volume}-{alpha_imbalance}'
            params = pd.DataFrame()
            init_dif = 1.0
            res_init = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
            init_T = init_T0
            logger.info(f"params_{output_file}")

        number_str = str(number).zfill(3)
        file = f'BTCUSDT-Trades-Optimized-{number_str}.parquet'

        logger.info(f"Dask n{number_str} de {file_count-1}")

        if not os.path.exists(os.path.join(raw_dataset_path, file)):
            logger.warning(f"Arquivo {file} não encontrado. Pulando para o próximo.")
            continue

        df_dask = read_parquet_files_optimized(raw_dataset_path, file)
        df_dask = apply_operations_optimized(df_dask, meta)

        bars, init_T, init_dif, res_init, params_df = batch_create_imbalance_dollar_bars_optimized(
            df_dask, init_T, init_dif, res_init, alpha_volume, alpha_imbalance
        )
        
        # Save individual part files to avoid memory accumulation
        if not bars.empty:
            # Add params and time_trial columns to bars before saving
            bars_with_params = bars.copy()
            bars_with_params['params'] = output_file
            bars_with_params['time_trial'] = timestamp
            
            # Create dedicated folder for this output file's parts
            parts_folder = f'{output_path}/{output_file}'
            os.makedirs(parts_folder, exist_ok=True)
            
            part_file_path = f'{parts_folder}/part_{number_str}.parquet'
            bars_with_params.to_parquet(part_file_path, index=False)
            logger.info(f"Saved part file: {part_file_path}")
        
        # Save params for this part
        if not params_df.empty:
            parts_folder = f'{output_path}/{output_file}'
            os.makedirs(parts_folder, exist_ok=True)
            params_part_file_path = f'{parts_folder}/params_part_{number_str}.parquet'
            params_df.to_parquet(params_part_file_path, index=False)
        
        # Keep track of accumulated params (small memory footprint)
        non_empty_params = [params, params_df]
        non_empty_params = [df for df in non_empty_params if not df.empty]
        if non_empty_params:
            params = pd.concat(non_empty_params, ignore_index=True)

        if number == file_count - 1:
            # Add the final incomplete bar if exists
            bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time, \
            current_imbalance, buy_volume_usd, total_volume_usd, total_volume = res_init

            if not np.isnan(bar_open):  # Only add if there's an incomplete bar
                bar_end_time = df_dask['time'].tail().iloc[-1]

                lastbar = [[bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                                current_imbalance, buy_volume_usd, total_volume_usd, total_volume]]

                lastbar = pd.DataFrame(lastbar, columns=['start_time', 'end_time', 'open', 'high', 'low', 'close', 
                                                       'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'])
                
                # Add params and time_trial columns to lastbar
                lastbar['params'] = output_file
                lastbar['time_trial'] = timestamp
                
                # Save the last bar as a separate part file
                parts_folder = f'{output_path}/{output_file}'
                lastbar_file_path = f'{parts_folder}/lastbar.parquet'
                lastbar.to_parquet(lastbar_file_path, index=False)
                logger.info(f"Saved last bar file: {lastbar_file_path}")

            # Now combine all part files into final result
            logger.info(f"Combining all part files for {output_file}")
            parts_folder = f'{output_path}/{output_file}'
            
            # Read all part files
            part_files = []
            for part_file in sorted(os.listdir(parts_folder)):
                if part_file.startswith('part_') and part_file.endswith('.parquet'):
                    part_path = f'{parts_folder}/{part_file}'
                    part_df = pd.read_parquet(part_path)
                    part_files.append(part_df)
            
            # Add lastbar if it exists
            lastbar_path = f'{parts_folder}/lastbar.parquet'
            if os.path.exists(lastbar_path):
                lastbar_df = pd.read_parquet(lastbar_path)
                part_files.append(lastbar_df)
            
            # Combine all parts
            if part_files:
                results_ = pd.concat(part_files, ignore_index=True)
            else:
                results_ = pd.DataFrame(columns=['start_time', 'end_time', 'open', 'high', 'low', 'close', 
                                              'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume',
                                              'params', 'time_trial'])

            # Convert datetime and drop start_time column
            results_['start_time'] = pd.to_datetime(results_['start_time'])
            results_['end_time'] = pd.to_datetime(results_['end_time'])
            results_.drop(columns=['start_time'], inplace=True)
            
            # Save to binance/futures-um folder structure
            output_file_path = f'{output_path}/binance/futures-um/{output_file}.xlsx'
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            results_.to_excel(output_file_path, index=False)

            # Note: Part files are kept in the dedicated folder for future reference
            logger.info(f"Part files preserved in: {parts_folder}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time_minutes = elapsed_time / 60
            processing_times[file] = elapsed_time_minutes
            logger.info(f"Tempo de processamento para {file}: {elapsed_time_minutes:.2f} minutos")
    
    return processing_times

def main(symbol='BTCUSDT', data_type='futures', futures_type='um', granularity='daily'):
    """Main function to run imbalance dollar bars generation"""
    # Setup paths based on parameters
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    if data_type == 'spot':
        raw_dataset_path = os.path.join(base_path, 'datasets', f'dataset-raw-{granularity}-compressed-optimized', 'spot')
    else:
        raw_dataset_path = os.path.join(base_path, 'datasets', f'dataset-raw-{granularity}-compressed-optimized', f'futures-{futures_type}')
    
    output_base_path = os.path.join(base_path, 'output')
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_base_path, exist_ok=True)
    
    logger.info(f"Processing imbalance bars for {symbol} {data_type} {granularity}")
    logger.info(f"Input path: {raw_dataset_path}")
    logger.info(f"Output path: {output_base_path}")
    
    # Setup Dask client
    client = setup_dask_client()
    
    try:
        # Check if input directory exists
        if not os.path.exists(raw_dataset_path):
            raise FileNotFoundError(f"Input directory not found: {raw_dataset_path}")
        
        # List files to determine count
        files = [f for f in os.listdir(raw_dataset_path) if os.path.isfile(os.path.join(raw_dataset_path, f)) and f.endswith('.parquet')]
        file_count = len(files)
        
        # Generate initial states
        initial_state = generate_initial_states(file_count)
        
        # Process imbalance bars
        processing_times = process_imbalance_bars(raw_dataset_path, output_base_path, initial_state, timestamp)
        
        logger.info("Processing completed successfully!")
        logger.info(f"Processing times: {processing_times}")
        
    finally:
        # Close Dask client
        client.close()

if __name__ == "__main__":
    main()