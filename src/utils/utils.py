from loguru import logger
import pandas as pd
from dask import delayed
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import os
import traceback
import numpy as np

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

def calculate_dollar_directions(df):
    try:
        logger.info("Calculando 'trade_dollar', 'price_change' e 'dollar_direction'")
        df['trade_dollar'] = df['price'] * df['qty']
        df['price_change'] = df['price'].diff().fillna(0)
        df['dollar_direction'] = df['price_change'].apply(
            lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
        ).astype(int)
        logger.info("'dollar_directions' calculadas com sucesso")
        return df
    except KeyError as e:
        logger.error(f"Erro ao calcular 'dollar_directions': {e}")
        return df
    except Exception as e:
        logger.error(f"Erro inesperado ao calcular 'dollar_directions': {e}")
        return df

def calculate_dollar_imbalance(df):
    try:
        logger.info("Calculando 'trade_dollar', 'dollar_side' e 'dollar_imbalance'")
        df['trade_dollar'] = df['price'] * df['qty']
        df['dollar_side'] = df['isBuyerMaker'].map({True: -1, False: 1})
        df['dollar_imbalance'] = df['trade_dollar'] * df['dollar_side']
        logger.info("'dollar_imbalance' calculadas com sucesso")
        return df
    except KeyError as e:
        logger.error(f"Erro ao calcular 'dollar_imbalance': {e}")
        return df
    except Exception as e:
        logger.error(f"Erro inesperado ao calcular 'dollar_imbalance': {e}")
        return df

def check_column_types(df, expected_types, file):
    for column, expected in expected_types.items():
        if column not in df.columns:
            logger.error(f"A coluna '{column}' está ausente no arquivo {file}.")
            return False

        actual_dtype = df[column].dtype

        if expected == int:
            if not np.issubdtype(actual_dtype, np.integer):
                logger.error(f"A coluna '{column}' possui tipo de dado inesperado: {actual_dtype}, esperado: int")
                return False
        elif expected == float:
            if not np.issubdtype(actual_dtype, np.floating):
                logger.error(f"A coluna '{column}' possui tipo de dado inesperado: {actual_dtype}, esperado: float")
                return False
        elif expected == str:
            if not (np.issubdtype(actual_dtype, np.object_) or np.issubdtype(actual_dtype, np.str_)):
                logger.error(f"A coluna '{column}' possui tipo de dado inesperado: {actual_dtype}, esperado: str")
                return False
        elif expected == bool:
            if not np.issubdtype(actual_dtype, np.bool_):
                logger.error(f"A coluna '{column}' possui tipo de dado inesperado: {actual_dtype}, esperado: bool")
                return False
        elif expected == pd.Timestamp:
            if not np.issubdtype(actual_dtype, np.datetime64):
                logger.error(f"A coluna '{column}' possui tipo de dado inesperado: {actual_dtype}, esperado: datetime64")
                return False
        else:
            logger.error(f"Tipo esperado não reconhecido para a coluna '{column}': {expected}")
            return False
    return True

def create_dollar_bars(df, dollar_threshold, output_path):
    if os.path.exists(output_path):
        logger.info(f"'dollar_bars' já existe. Carregando de {output_path}...")
        dollar_bars = dd.read_parquet(output_path).compute()
        logger.info("'dollar_bars' carregado com sucesso.")
    else:
        logger.info("Criando 'dollar_bars'...")
        with ProgressBar():
            # Calcular o valor em dólares de cada trade
            df = df.assign(trade_dollar=df['price'] * df['qty'])

            # Calcular o acumulado
            df = df.assign(cumulative_dollar=df['trade_dollar'].cumsum())

            # Definir o número do bar
            df = df.assign(bar_number=(df['cumulative_dollar'] // dollar_threshold).astype(int))

            # Agrupar por bar_number e realizar todas as agregações em uma única chamada
            grouped = df.groupby('bar_number').agg(
                trade_count=('trade_id', 'count'),
                price_open=('price', 'first'),
                price_high=('price', 'max'),
                price_low=('price', 'min'),
                price_close=('price', 'last'),
                qty_sum=('qty', 'sum'),
                quoteQty_sum=('quoteQty', 'sum'),
                time=('time', 'max'),
                isBuyerMaker_avg=('isBuyerMaker', 'mean'),
                isBestMatch_avg=('isBestMatch', 'mean')
            )

            # Salvar diretamente como Parquet usando Dask
            grouped.to_parquet(output_path, engine='pyarrow', compression='snappy')

        logger.info("'dollar_bars' criado e salvo com sucesso.")
        # Carregar os dados para retornar como pandas DataFrame
        dollar_bars = dd.read_parquet(output_path).compute()

    return dollar_bars

@delayed
def process_file_run_bars(file, dollar_threshold, max_records):
    try:
        df = pd.read_parquet(file)
        logger.info(f"Processando arquivo: {file}")
        logger.info(f"Colunas disponíveis: {df.columns.tolist()}")

        # Definir os tipos esperados
        expected_types = {
            'price': float,       # Assumindo que 'price' é float
            'qty': float,         # Assumindo que 'qty' é float
            'trade_id': int,      # 'trade_id' como inteiro
            'isBuyerMaker': bool,
            'isBestMatch': bool,
            'time': pd.Timestamp  # Assumindo que 'time' é datetime
        }

        # Verificar tipos de colunas
        if not check_column_types(df, expected_types, file):
            logger.error(f"Tipos de colunas inválidos no arquivo {file}. Pulando processamento.")
            return pd.DataFrame()

        # Converter 'time' para datetime, caso ainda não esteja
        if not np.issubdtype(df['time'].dtype, np.datetime64):
            logger.info(f"Convertendo coluna 'time' para datetime no arquivo {file}")
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            if df['time'].isnull().any():
                logger.error(f"Falha na conversão da coluna 'time' para datetime no arquivo {file}.")
                return pd.DataFrame()

        # Verificar se o DataFrame possui dados suficientes
        if len(df) < 2:
            logger.warning(f"O arquivo {file} possui menos de duas linhas. Pulando processamento.")
            return pd.DataFrame()

        # Calcular direções
        df = calculate_dollar_directions(df)

        # Verificar se 'price_change' foi criada
        if 'price_change' not in df.columns:
            logger.error(f"A coluna 'price_change' não foi criada no arquivo {file}.")
            return pd.DataFrame()

        # Continuar com o processamento original
        df['direction_change'] = (df['dollar_direction'] != df['dollar_direction'].shift()).cumsum()
        df['cumulative_dollar'] = df.groupby('direction_change')['trade_dollar'].cumsum()
        df['bar_number'] = (df['cumulative_dollar'] // dollar_threshold).astype(int) + (df['direction_change'] * 1e6).astype(int)

        # Limitar o DataFrame ao número máximo de registros
        df = df.head(max_records)

        # Agrupar por bar_number e realizar as agregações
        grouped = df.groupby('bar_number').agg(
            trade_count=('trade_id', 'count'),
            price_open=('price', 'first'),
            price_high=('price', 'max'),
            price_low=('price', 'min'),
            price_close=('price', 'last'),
            qty_sum=('qty', 'sum'),
            quoteQty_sum=('quoteQty', 'sum'),
            time=('time', 'max'),
            isBuyerMaker_avg=('isBuyerMaker', 'mean'),
            isBestMatch_avg=('isBestMatch', 'mean')
        ).reset_index(drop=True)

        logger.info(f"Arquivo {file} processado com sucesso.")
        return grouped
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo {file}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def create_dollar_run_bars(dataset_path, dollar_threshold, output_path, max_records):
    if os.path.exists(output_path):
        logger.info(f"'dollar_run_bars' já existe. Carregando de {output_path}...")
        dollar_run_bars = dd.read_parquet(output_path).compute()
        logger.info("'dollar_run_bars' carregado com sucesso.")
    else:
        logger.info("Criando 'dollar_run_bars'...")
        # Obter a lista de arquivos Parquet no dataset_path
        parquet_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]
        parquet_files.sort()

        # Inicializar contador de registros
        total_records = 0
        limited_files = []
        records_per_file = {}

        # Iterar sobre os arquivos para selecionar apenas os necessários para alcançar max_records
        for file in parquet_files:
            df_temp = pd.read_parquet(file)
            num_records = len(df_temp)
            if total_records + num_records <= max_records:
                limited_files.append(file)
                records_per_file[file] = num_records
                total_records += num_records
            else:
                remaining = max_records - total_records
                if remaining > 0:
                    limited_files.append(file)
                    records_per_file[file] = remaining
                    total_records += remaining
                break
            if total_records >= max_records:
                break

        if not limited_files:
            logger.warning("Nenhum arquivo para processar dentro do limite de registros.")
            return pd.DataFrame()

        logger.info(f"Total de registros a serem processados: {total_records}")

        # Aplicar a função a todos os arquivos limitados
        delayed_dfs = [process_file_run_bars(file, dollar_threshold, records_per_file[file]) for file in limited_files]

        # Computar em paralelo
        with ProgressBar():
            run_bars_dask = dd.from_delayed(delayed_dfs)
            dollar_run_bars = run_bars_dask.compute()

        # Verificar se o DataFrame resultante não está vazio
        if not dollar_run_bars.empty:
            # Salvar como Parquet
            dollar_run_bars.to_parquet(output_path, engine='pyarrow', compression='snappy')
            logger.info("'dollar_run_bars' criado e salvo com sucesso.")
        else:
            logger.warning("'dollar_run_bars' está vazio. Nenhum dado foi salvo.")

    return dollar_run_bars

@delayed
def process_file_imbalance_bars(file, dollar_threshold, max_records):
    try:
        df = pd.read_parquet(file)
        logger.info(f"Processando arquivo: {file}")
        logger.info(f"Colunas disponíveis: {df.columns.tolist()}")

        # Definir os tipos esperados
        expected_types = {
            'price': float,       # Assumindo que 'price' é float
            'qty': float,         # Assumindo que 'qty' é float
            'trade_id': int,      # 'trade_id' como inteiro
            'isBuyerMaker': bool,
            'isBestMatch': bool,
            'time': pd.Timestamp  # Assumindo que 'time' é datetime
        }

        # Verificar tipos de colunas
        if not check_column_types(df, expected_types, file):
            logger.error(f"Tipos de colunas inválidos no arquivo {file}. Pulando processamento.")
            return pd.DataFrame()

        # Converter 'time' para datetime, caso ainda não esteja
        if not np.issubdtype(df['time'].dtype, np.datetime64):
            logger.info(f"Convertendo coluna 'time' para datetime no arquivo {file}")
            df['time'] = pd.to_datetime(df['time'], unit='s', errors='coerce')
            if df['time'].isnull().any():
                logger.error(f"Falha na conversão da coluna 'time' para datetime no arquivo {file}.")
                return pd.DataFrame()

        # Verificar se o DataFrame possui dados suficientes
        if len(df) < 1:
            logger.warning(f"O arquivo {file} está vazio. Pulando processamento.")
            return pd.DataFrame()

        # Calcular desequilíbrio
        df = calculate_dollar_imbalance(df)

        # Calcular o acumulado de desequilíbrio
        df['cumulative_imbalance'] = df['dollar_imbalance'].cumsum()

        # Definir o número do bar baseado no limiar
        df['bar_number'] = (df['cumulative_imbalance'].abs() // dollar_threshold).astype(int)

        # Limitar o DataFrame ao número máximo de registros
        df = df.head(max_records)

        # Agrupar por bar_number e realizar as agregações
        grouped = df.groupby('bar_number').agg(
            trade_count=('trade_id', 'count'),
            price_open=('price', 'first'),
            price_high=('price', 'max'),
            price_low=('price', 'min'),
            price_close=('price', 'last'),
            qty_sum=('qty', 'sum'),
            quoteQty_sum=('quoteQty', 'sum'),
            time=('time', 'max'),
            isBuyerMaker_avg=('isBuyerMaker', 'mean'),
            isBestMatch_avg=('isBestMatch', 'mean')
        ).reset_index(drop=True)

        logger.info(f"Arquivo {file} processado com sucesso.")
        return grouped
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo {file}: {e}")
        logger.error(traceback.format_exc())
        return pd.DataFrame()

def create_dollar_imbalance_bars(dataset_path, dollar_threshold, output_path, max_records):
    if os.path.exists(output_path):
        logger.info(f"'dollar_imbalance_bars' já existe. Carregando de {output_path}...")
        dollar_imbalance_bars = dd.read_parquet(output_path).compute()
        logger.info("'dollar_imbalance_bars' carregado com sucesso.")
    else:
        logger.info("Criando 'dollar_imbalance_bars'...")
        # Obter a lista de arquivos Parquet no dataset_path
        parquet_files = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.parquet')]
        parquet_files.sort()

        # Inicializar contador de registros
        total_records = 0
        limited_files = []
        records_per_file = {}

        # Iterar sobre os arquivos para selecionar apenas os necessários para alcançar max_records
        for file in parquet_files:
            df_temp = pd.read_parquet(file)
            num_records = len(df_temp)
            if total_records + num_records <= max_records:
                limited_files.append(file)
                records_per_file[file] = num_records
                total_records += num_records
            else:
                remaining = max_records - total_records
                if remaining > 0:
                    limited_files.append(file)
                    records_per_file[file] = remaining
                    total_records += remaining
                break
            if total_records >= max_records:
                break

        if not limited_files:
            logger.warning("Nenhum arquivo para processar dentro do limite de registros.")
            return pd.DataFrame()

        logger.info(f"Total de registros a serem processados: {total_records}")

        # Aplicar a função a todos os arquivos limitados
        delayed_dfs = [process_file_imbalance_bars(file, dollar_threshold, records_per_file[file]) for file in limited_files]

        # Computar em paralelo
        with ProgressBar():
            imbalance_bars_dask = dd.from_delayed(delayed_dfs)
            dollar_imbalance_bars = imbalance_bars_dask.compute()

        # Verificar se o DataFrame resultante não está vazio
        if not dollar_imbalance_bars.empty:
            # Salvar como Parquet
            dollar_imbalance_bars.to_parquet(output_path, engine='pyarrow', compression='snappy')
            logger.info("'dollar_imbalance_bars' criado e salvo com sucesso.")
        else:
            logger.warning("'dollar_imbalance_bars' está vazio. Nenhum dado foi salvo.")

    return dollar_imbalance_bars
