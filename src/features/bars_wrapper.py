"""
Wrapper functions for different bar generation methods
"""

import logging
from pathlib import Path
import time
import gc
from dask.distributed import Client

def generate_standard_bars(symbol='BTCUSDT', data_type='spot', futures_type='um',
                          granularity='daily', volume_threshold=40_000_000):
    """Generate standard dollar bars"""
    from standard_dollar_bars import process_files_and_generate_bars, setup_logging, setup_dask_client

    # Setup
    setup_logging()
    output_dir = Path('./output/standard/')

    # Start Dask client
    client = setup_dask_client(n_workers=10, threads_per_worker=1, memory_limit='6GB')

    try:
        logging.info(f"Generating Standard Dollar Bars for {symbol}")
        logging.info(f"Data type: {data_type}, Futures type: {futures_type}, Granularity: {granularity}")
        logging.info(f"Volume threshold: {volume_threshold:,} USD")

        # Generate bars
        process_files_and_generate_bars(
            data_type=data_type,
            futures_type=futures_type,
            granularity=granularity,
            init_vol=volume_threshold,
            output_dir=output_dir,
            db_engine=None
        )

        logging.info("Standard Dollar Bars generation completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Error generating Standard Dollar Bars: {e}")
        raise
    finally:
        client.close()
        gc.collect()


def generate_imbalance_bars(symbol='BTCUSDT', data_type='spot', futures_type='um',
                           granularity='daily'):
    """Generate imbalance dollar bars"""
    # Import the existing imbalance bars main function
    from imbalance_bars import main as imbalance_main

    try:
        logging.info(f"Generating Imbalance Dollar Bars for {symbol}")
        logging.info(f"Data type: {data_type}, Futures type: {futures_type}, Granularity: {granularity}")

        # Call the existing imbalance main function
        imbalance_main(
            symbol=symbol,
            data_type=data_type,
            futures_type=futures_type,
            granularity=granularity
        )

        logging.info("Imbalance Dollar Bars generation completed successfully!")
        return True

    except Exception as e:
        logging.error(f"Error generating Imbalance Dollar Bars: {e}")
        raise


def generate_run_bars(symbol='BTCUSDT', data_type='spot', futures_type='um',
                     granularity='daily'):
    """Generate run dollar bars"""
    import sys
    import subprocess
    from pathlib import Path

    try:
        logging.info(f"Generating Run Dollar Bars for {symbol}")
        logging.info(f"Data type: {data_type}, Futures type: {futures_type}, Granularity: {granularity}")

        # Build command line arguments
        script_path = Path(__file__).parent.parent / "scripts" / "run_dollar_bars.py"

        cmd = [
            sys.executable,
            str(script_path),
            '--data-type', data_type,
            '--futures-type', futures_type,
            '--symbol', symbol,
            '--granularity', granularity
        ]

        # Run the script
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logging.info("Run Dollar Bars generation completed successfully!")
            return True
        else:
            logging.error(f"Run Dollar Bars generation failed: {result.stderr}")
            raise Exception(f"Script failed with return code {result.returncode}")

    except Exception as e:
        logging.error(f"Error generating Run Dollar Bars: {e}")
        raise