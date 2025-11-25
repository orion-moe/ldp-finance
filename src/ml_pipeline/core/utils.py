"""
Utility functions for the optimized pipeline.
Includes logging setup, memory monitoring, and helper functions.
"""

import os
import logging
import psutil
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = 'data/logs', log_level: str = 'INFO',
                  max_bytes: int = 50*1024*1024, backup_count: int = 10) -> str:
    """
    Configure logging with both console and file handlers.

    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep

    Returns:
        Path to the log file
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    log_filename = f"main_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(log_dir, log_filename)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info("="*70)
    logger.info("LOGGING CONFIGURED")
    logger.info(f"Log file: {log_file_path}")
    logger.info(f"Log level: {log_level}")
    logger.info("="*70)

    return log_file_path


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in megabytes
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB


def log_memory(operation_name: str = "", logger: logging.Logger = None):
    """
    Log current memory usage.

    Args:
        operation_name: Name of the operation (for context)
        logger: Logger instance. If None, uses root logger
    """
    if logger is None:
        logger = logging.getLogger()

    memory_mb = get_memory_usage()

    if operation_name:
        logger.info(f"💾 Memory after {operation_name}: {memory_mb:.1f} MB")
    else:
        logger.info(f"💾 Current memory usage: {memory_mb:.1f} MB")


def create_output_directory(base_path: str, file_name: str) -> tuple:
    """
    Create output directory structure based on filename.

    Args:
        base_path: Base output path
        file_name: Input filename (e.g., '20251123-003308-standard-futures-volume200000000.parquet')

    Returns:
        Tuple of (output_path, sampling_folder, metadata_dict)
    """
    # Extract sampling info from filename
    # Format: YYYYMMDD-HHMMSS-type-market-volumeXXX.parquet
    file_parts = file_name.replace('.parquet', '').split('-')

    sampling_date = file_parts[0] if len(file_parts) > 0 else datetime.now().strftime('%Y%m%d')
    sampling_time = file_parts[1] if len(file_parts) > 1 else datetime.now().strftime('%H%M%S')
    sampling_type = file_parts[2] if len(file_parts) > 2 else 'standard'
    market_type = file_parts[3] if len(file_parts) > 3 else 'futures'

    # Extract volume info if present
    volume_info = ''
    for part in file_parts:
        if 'volume' in part:
            volume_info = f"-{part}"
            break

    # Create folder name with sampling info (using hyphens)
    sampling_folder = f"{sampling_date}-{sampling_time}-{sampling_type}-{market_type}{volume_info}"
    output_path = os.path.join(base_path, sampling_folder)

    # Create directory
    os.makedirs(output_path, exist_ok=True)

    # Metadata
    metadata = {
        'sampling_id': sampling_folder,
        'date': sampling_date,
        'time': sampling_time,
        'type': sampling_type,
        'market': market_type,
        'volume': volume_info.replace('-volume', '') if volume_info else '',
        'source_file': file_name,
        'created_at': datetime.now().isoformat(),
        'pipeline_version': '2.0_refactored'
    }

    return output_path, sampling_folder, metadata


def save_metadata(output_path: str, metadata: dict):
    """
    Save sampling metadata to JSON file.

    Args:
        output_path: Output directory path
        metadata: Metadata dictionary
    """
    import json

    metadata_path = os.path.join(output_path, 'sampling_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logging.info(f"Sampling metadata saved to: {metadata_path}")


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "2m 30s" or "1h 5m 30s")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {mins}m {secs:.1f}s"


def print_step_header(step_number: int, step_name: str, logger: logging.Logger = None):
    """
    Print a formatted step header.

    Args:
        step_number: Step number
        step_name: Step name/description
        logger: Logger instance. If None, uses root logger
    """
    if logger is None:
        logger = logging.getLogger()

    logger.info(f"\n{'='*70}")
    logger.info(f"STEP {step_number}: {step_name}")
    logger.info(f"{'='*70}")
