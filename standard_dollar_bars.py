#!/usr/bin/env python3
"""
Standard Dollar Bars Generator

Este módulo gera "standard dollar bars" a partir de dados de trades do Bitcoin.
Usa processamento distribuído com Dask e otimizações com Numba para processar
grandes volumes de dados eficientemente.

Baseado no notebook: dollar_barsv2.ipynb
"""

import os
import argparse
import datetime
import logging
import time
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client
from numba import njit, types
from numba.typed import List


def setup_logging():
    """Configura o sistema de logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def setup_dask_client(n_workers=10, threads_per_worker=1, memory_limit='6.4GB'):
    """Configura o cliente Dask para processamento distribuído."""
    client = Client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit
    )
    logging.info(f"Dask client inicializado: {client}")
    return client


def get_data_path(data_type='futures', futures_type='um', granularity='daily'):
    """
    Constrói o caminho para os dados baseado nos parâmetros.

    Args:
        data_type: 'spot' ou 'futures'
        futures_type: 'um' ou 'cm' (apenas para futures)
        granularity: 'daily' ou 'monthly'

    Returns:
        Path para os dados
    """
    project_root = Path(__file__).parent

    if data_type == 'spot':
        return project_root / 'datasets' / f'dataset-raw-{granularity}-compressed-optimized' / 'spot'
    else:
        return project_root / 'datasets' / f'dataset-raw-{granularity}-compressed-optimized' / f'futures-{futures_type}'


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


@njit(
    types.Tuple((
        # Primeiro elemento: lista de barras (cada barra é uma tupla)
        types.ListType(types.Tuple((types.float64,)*10)),
        # Segundo elemento: estado final da partição atual (tupla com 10 valores float64)
        types.Tuple((types.float64,) * 10),
    ))(
        # Parâmetros de entrada
        types.float64[:],   # prices - Preços das operações
        types.float64[:],   # times - Horários (em milissegundos/seconds)
        types.float64[:],   # imbalances - Dados de desequilíbrio em USD
        types.int8[:],      # sides - Lado da negociação (-1, 0 ou +1)
        types.float64[:],   # qtys - Quantidade negociada (em ordens)
        types.Tuple((types.float64,) * 10),    # res_init: estado inicial
        types.float64       # dollar_volume: volume em dólar para formar a barra
    )
)
def process_partition_dollar_numba(
    prices, times, imbalances, sides, qtys, res_init, dollar_volume
):
    """Processa uma partição usando numba para aceleração."""
    threshold = dollar_volume

    bars = List()  # Lista tipada para armazenar as barras formadas

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

        if total_volume_usd >= threshold:
            bar_end_time = times[i]

            # Salva a barra formada
            bars.append((
                bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                current_imbalance, buy_volume_usd, total_volume_usd, total_volume
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

    return bars, final_state


def create_dollar_bars_numba(partition, res_init, dollar_volume):
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
    bars, res_init = process_partition_dollar_numba(
        prices, times, imbalances, sides, qtys, res_init, dollar_volume
    )

    # Converte as barras para um DataFrame
    if len(bars) > 0:
        bars_df = pd.DataFrame(bars, columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])
    else:
        # Retorna um DataFrame vazio com as colunas apropriadas
        bars_df = pd.DataFrame(columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])

    return bars_df, res_init


def batch_create_dollar_bars_optimized(df_dask, res_init, dollar_volume):
    """Processa partições em lote para criar barras de desequilíbrio em dólares."""
    results = []
    for partition in range(df_dask.npartitions):
        logging.info(f'Processando partição {partition+1} de {df_dask.npartitions}')
        part = df_dask.get_partition(partition).compute()

        bars, res_init = create_dollar_bars_numba(
            part, res_init, dollar_volume
        )
        results.append(bars)

    # Filtra DataFrames vazios
    results = [df for df in results if not df.empty]
    if results:
        results_df = pd.concat(results, ignore_index=True)
    else:
        # Retorna um DataFrame vazio com as colunas apropriadas se não houver resultados
        results_df = pd.DataFrame(columns=[
            'start_time', 'end_time', 'open', 'high', 'low', 'close',
            'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
        ])
    return results_df, res_init


def process_standard_dollar_bars(
    data_type='futures',
    futures_type='um',
    granularity='daily',
    dollar_volume=10_000_000,
    output_dir=None
):
    """
    Função principal para processar standard dollar bars.

    Args:
        data_type: 'spot' ou 'futures'
        futures_type: 'um' ou 'cm' (apenas para futures)
        granularity: 'daily' ou 'monthly'
        dollar_volume: Volume em dólares para formar cada barra
        output_dir: Diretório de saída (opcional)
    """
    # Configuração inicial
    setup_logging()
    client = setup_dask_client()

    # Caminhos
    raw_dataset_path = get_data_path(data_type, futures_type, granularity)

    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    logging.info(f"Caminho dos dados: {raw_dataset_path}")
    logging.info(f"Diretório de saída: {output_dir}")
    logging.info(f"Dollar volume threshold: ${dollar_volume:,.0f}")

    # Verifica se o diretório existe
    if not raw_dataset_path.exists():
        logging.error(f"Diretório de dados não encontrado: {raw_dataset_path}")
        return

    # Lista arquivos
    files = [f for f in os.listdir(raw_dataset_path) if f.endswith('.parquet')]
    file_count = len(files)

    if file_count == 0:
        logging.error("Nenhum arquivo Parquet encontrado!")
        return

    logging.info(f"Encontrados {file_count} arquivos Parquet")

    # Meta DataFrame para map_partitions
    meta = pd.DataFrame({
        'price': pd.Series(dtype='float32'),
        'qty': pd.Series(dtype='float32'),
        'quoteQty': pd.Series(dtype='float32'),
        'time': pd.Series(dtype='float64'),
        'side': pd.Series(dtype='int8')
    })

    # Processamento
    start_time = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f'standard_dollar_bars_v{dollar_volume}'

    results = pd.DataFrame()
    res_init = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0)

    logging.info(f"Iniciando processamento: {output_file}")

    for number in range(1, file_count + 1):
        number_ = str(number).zfill(3)
        file = f'BTCUSDT-Trades-Optimized-{number_}.parquet'

        logging.info(f"Processando arquivo {number} de {file_count}: {file}")

        file_path = raw_dataset_path / file
        if not file_path.exists():
            logging.warning(f"Arquivo não encontrado: {file}")
            continue

        df_dask = read_parquet_files_optimized(str(raw_dataset_path), file)
        df_dask = apply_operations_optimized(df_dask, meta)

        bars, res_init = batch_create_dollar_bars_optimized(
            df_dask, res_init, dollar_volume
        )

        results = pd.concat([results, bars], ignore_index=True)

        # Processa última barra se for o último arquivo
        if number == file_count:
            bar_open, bar_high, bar_low, bar_close, bar_start_time, bar_end_time, \
            current_imbalance, buy_volume_usd, total_volume_usd, total_volume = res_init

            if not np.isnan(bar_open):
                bar_end_time = df_dask['time'].tail().iloc[-1]

                lastbar = pd.DataFrame([[
                    bar_start_time, bar_end_time, bar_open, bar_high, bar_low, bar_close,
                    current_imbalance, buy_volume_usd, total_volume_usd, total_volume
                ]], columns=[
                    'start_time', 'end_time', 'open', 'high', 'low', 'close',
                    'imbalance_col', 'total_volume_buy_usd', 'total_volume_usd', 'total_volume'
                ])

                results = pd.concat([results, lastbar], ignore_index=True)

    # Finaliza processamento
    if not results.empty:
        results_final = results.copy()
        results_final['start_time'] = pd.to_datetime(results_final['start_time'])
        results_final['end_time'] = pd.to_datetime(results_final['end_time'])
        results_final.drop(columns=['start_time'], inplace=True)

        # Adiciona metadados
        results_final['dollar_volume_threshold'] = dollar_volume
        results_final['data_type'] = data_type
        results_final['futures_type'] = futures_type if data_type == 'futures' else None
        results_final['granularity'] = granularity

        # Salva arquivo
        output_path = output_dir / f'{output_file}_{timestamp}.parquet'
        results_final.to_parquet(output_path, index=False)

        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60

        logging.info(f"Processamento concluído em {elapsed_time:.2f} minutos")
        logging.info(f"Arquivo salvo: {output_path}")
        logging.info(f"Total de barras geradas: {len(results_final)}")
        logging.info(f"Colunas no arquivo: {list(results_final.columns)}")
    else:
        logging.warning("Nenhuma barra foi gerada!")

    # Fecha cliente Dask
    client.close()


def main():
    """Função principal para execução via linha de comando."""
    parser = argparse.ArgumentParser(description='Gerador de Standard Dollar Bars')

    parser.add_argument('--data-type', choices=['spot', 'futures'], default='futures',
                        help='Tipo de dados (padrão: futures)')
    parser.add_argument('--futures-type', choices=['um', 'cm'], default='um',
                        help='Tipo de futures (padrão: um)')
    parser.add_argument('--granularity', choices=['daily', 'monthly'], default='daily',
                        help='Granularidade (padrão: daily)')
    parser.add_argument('--dollar-volume', type=float, default=10_000_000,
                        help='Volume em dólares para formar cada barra (padrão: 10000000)')
    parser.add_argument('--output-dir', type=str,
                        help='Diretório de saída (padrão: ./output)')

    args = parser.parse_args()

    print("=== STANDARD DOLLAR BARS GENERATOR ===")
    print(f"Parâmetros em uso:")
    print(f"  Data type: {args.data_type}")
    if args.data_type == 'futures':
        print(f"  Futures type: {args.futures_type}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Dollar volume threshold: ${args.dollar_volume:,.0f}")
    print("=" * 50)

    process_standard_dollar_bars(
        data_type=args.data_type,
        futures_type=args.futures_type,
        granularity=args.granularity,
        dollar_volume=args.dollar_volume,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()