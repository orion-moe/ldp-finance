"""
ZIP to Parquet Streamer - Direct conversion from ZIP to Parquet without intermediate CSV
Optimized for memory efficiency and speed
"""

import sys
import zipfile
import hashlib
import io
import csv
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Iterator
from datetime import datetime
from loguru import logger
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Column names for CSV files - SPOT (7 columns)
COLUMN_NAMES_SPOT = [
    'trade_id',
    'price',
    'qty',
    'quoteQty',
    'time',
    'isBuyerMaker',
    'isBestMatch'
]

# Column names for CSV files - FUTURES (6 columns, no isBestMatch)
COLUMN_NAMES_FUTURES = [
    'trade_id',
    'price',
    'qty',
    'quoteQty',
    'time',
    'isBuyerMaker'
]

# Optimized data types for Arrow schema - SPOT
ARROW_SCHEMA_SPOT = pa.schema([
    ('trade_id', pa.int64()),
    ('price', pa.float32()),
    ('qty', pa.float32()),
    ('quoteQty', pa.float32()),
    ('time', pa.timestamp('ms')),
    ('isBuyerMaker', pa.bool_()),
    ('isBestMatch', pa.bool_())
])

# Optimized data types for Arrow schema - FUTURES
ARROW_SCHEMA_FUTURES = pa.schema([
    ('trade_id', pa.int64()),
    ('price', pa.float32()),
    ('qty', pa.float32()),
    ('quoteQty', pa.float32()),
    ('time', pa.timestamp('ms')),
    ('isBuyerMaker', pa.bool_()),
    ('isBestMatch', pa.bool_())  # Always included in schema, but set to False for futures
])

class ZipToParquetStreamer:
    def __init__(self, symbol: str = "BTCUSDT", data_type: str = "spot",
                 futures_type: str = "um", granularity: str = "monthly",
                 base_dir: Path = Path("."), compression: str = "snappy",
                 chunk_size: int = 5_000_000):
        """
        Initialize ZIP to Parquet Streamer

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            data_type: 'spot' or 'futures'
            futures_type: 'um' (USD-M) or 'cm' (COIN-M) - only used if data_type is 'futures'
            granularity: 'daily' or 'monthly'
            base_dir: Base directory for data storage
            compression: Compression algorithm ('snappy', 'zstd', 'lz4', 'brotli')
            chunk_size: Number of rows to process at once (default 5M for ~100MB memory)
        """
        self.symbol = symbol
        self.data_type = data_type
        self.futures_type = futures_type if data_type == "futures" else None
        self.granularity = granularity
        self.compression = compression
        self.chunk_size = chunk_size

        # Set schema based on data type
        # Note: Both schemas have 7 fields (isBestMatch always included, but False for futures)
        if data_type == "spot":
            self.expected_columns = 7
            self.arrow_schema = ARROW_SCHEMA_SPOT
        else:  # futures
            self.expected_columns = 6  # CSV only has 6 columns
            self.arrow_schema = ARROW_SCHEMA_FUTURES

        # Always use full column names (7) for the schema
        self.column_names = COLUMN_NAMES_SPOT

        # Ensure base_dir points to data folder
        if base_dir == Path("."):
            self.base_dir = Path.cwd() / "data"
        else:
            self.base_dir = base_dir

        # Set up directories using modern ticker-based structure
        ticker_name = f"{symbol.lower()}-{data_type}"
        if data_type == "futures":
            ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

        self.ticker_dir = self.base_dir / ticker_name
        self.raw_dir = self.ticker_dir / f"raw-zip-{granularity}"
        self.compressed_dir = self.ticker_dir / f"raw-parquet-{granularity}"

        self.compressed_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        log_file = self.base_dir / "logs" / f"zip_to_parquet_{self.symbol}_{self.data_type}_{self.granularity}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        logger.remove()
        logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        logger.add(log_file, rotation="500 MB", retention="10 days")

        self.logger = logger

    def verify_checksum(self, zip_path: Path) -> bool:
        """Verify ZIP file checksum if exists"""
        checksum_path = Path(str(zip_path) + ".CHECKSUM")
        if not checksum_path.exists():
            self.logger.warning(f"No checksum file found for {zip_path.name}")
            return True  # Continue without checksum

        # Calculate actual checksum
        sha256_hash = hashlib.sha256()
        with open(zip_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        actual_checksum = sha256_hash.hexdigest()

        # Read expected checksum
        with open(checksum_path, 'r') as f:
            expected_checksum = f.read().strip().split()[0]

        if actual_checksum != expected_checksum:
            self.logger.error(f"Checksum mismatch for {zip_path.name}")
            return False

        return True

    def stream_csv_from_zip(self, zip_path: Path) -> Iterator[Dict]:
        """
        Stream CSV data from ZIP file without extracting to disk

        Yields:
            Dict with parsed row data
        """
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            # Find the CSV file in the ZIP
            csv_filename = None
            for name in zip_file.namelist():
                if name.endswith('.csv'):
                    csv_filename = name
                    break

            if not csv_filename:
                raise ValueError(f"No CSV file found in {zip_path.name}")

            # Stream CSV content
            with zip_file.open(csv_filename) as csv_file:
                # Read as text with proper encoding
                text_wrapper = io.TextIOWrapper(csv_file, encoding='utf-8')
                csv_reader = csv.reader(text_wrapper)

                row_count = 0
                header_skipped = False

                for row in csv_reader:
                    # Skip header row if present (2025+ files have headers)
                    if not header_skipped:
                        # Check if first row is a header by looking for common header keywords
                        if any(val.lower() in ['id', 'price', 'qty', 'time', 'is_buyer_maker'] for val in row):
                            self.logger.debug(f"Detected and skipping CSV header: {row}")
                            header_skipped = True
                            continue
                        header_skipped = True  # Mark as checked even if no header

                    if len(row) != self.expected_columns:
                        self.logger.warning(f"Skipping malformed row {row_count}: expected {self.expected_columns} columns, got {len(row)}")
                        continue

                    try:
                        # Parse timestamp and normalize to milliseconds
                        timestamp_raw = int(row[4])

                        # Detect unit based on digit count and normalize to milliseconds
                        # This handles both old (ms) and new (us) Binance CSV formats
                        if len(str(timestamp_raw)) >= 16:  # Microseconds (16 digits)
                            timestamp_ms = timestamp_raw // 1000
                        elif len(str(timestamp_raw)) >= 13:  # Milliseconds (13 digits)
                            timestamp_ms = timestamp_raw
                        else:  # Seconds (10 digits)
                            timestamp_ms = timestamp_raw * 1000

                        # Parse and convert data types
                        row_data = {
                            'trade_id': int(row[0]),
                            'price': float(row[1]),
                            'qty': float(row[2]),
                            'quoteQty': float(row[3]),
                            'time': timestamp_ms,  # Normalized to milliseconds
                            'isBuyerMaker': row[5].lower() == 'true',
                        }

                        # Add isBestMatch based on data type
                        if self.data_type == "spot":
                            row_data['isBestMatch'] = row[6].lower() == 'true'
                        else:  # futures - set to False (not available in futures data)
                            row_data['isBestMatch'] = False

                        yield row_data
                        row_count += 1

                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing row {row_count}: {e}")
                        continue

    def convert_zip_to_parquet(self, zip_path: Path) -> Optional[Path]:
        """
        Convert a single ZIP file directly to Parquet

        Args:
            zip_path: Path to ZIP file

        Returns:
            Path to created Parquet file or None if failed
        """
        # Generate output filename
        zip_name = zip_path.stem  # Remove .zip extension
        if self.granularity == "daily":
            # Format: BTCUSDT-Trades-2024-01-01.zip -> BTCUSDT-Trades-2024-01-01.parquet
            parquet_name = f"{zip_name}.parquet"
        else:
            # Format: BTCUSDT-Trades-2024-01.zip -> BTCUSDT-Trades-2024-01.parquet
            parquet_name = f"{zip_name}.parquet"

        output_path = self.compressed_dir / parquet_name

        # Skip if already converted
        if output_path.exists():
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(f"Already exists: {parquet_name} ({file_size_mb:.2f} MB)")
            return output_path

        try:
            # Verify checksum
            if not self.verify_checksum(zip_path):
                self.logger.error(f"Checksum verification failed for {zip_path.name}")
                return None

            self.logger.info(f"Converting {zip_path.name} to Parquet...")

            # Process in chunks
            chunk_data = []
            chunks_to_write = []
            total_rows = 0

            # Stream data from ZIP
            for row in self.stream_csv_from_zip(zip_path):
                chunk_data.append(row)

                # Process chunk when it reaches the size limit
                if len(chunk_data) >= self.chunk_size:
                    # Convert to Arrow table
                    batch = self._create_arrow_batch(chunk_data)
                    chunks_to_write.append(batch)
                    total_rows += len(chunk_data)

                    self.logger.debug(f"Processed {total_rows:,} rows...")
                    chunk_data = []

            # Process remaining data
            if chunk_data:
                batch = self._create_arrow_batch(chunk_data)
                chunks_to_write.append(batch)
                total_rows += len(chunk_data)

            # Combine all chunks and write to Parquet
            if chunks_to_write:
                table = pa.Table.from_batches(chunks_to_write, schema=self.arrow_schema)

                # Write with optimization settings
                pq.write_table(
                    table,
                    output_path,
                    compression=self.compression,
                    use_dictionary=True,
                    compression_level=None,  # Use default for the algorithm
                    row_group_size=100_000  # Optimal for query performance
                )

                # Verify written file integrity
                try:
                    # First check if file can be opened
                    test_file = pq.ParquetFile(output_path)
                    # Then verify row count
                    written_table = pq.read_table(output_path)
                    if len(written_table) != total_rows:
                        raise ValueError(f"Row count mismatch: expected {total_rows}, got {len(written_table)}")
                except Exception as e:
                    # Delete corrupted file
                    if output_path.exists():
                        output_path.unlink()
                    raise ValueError(f"Corrupted parquet file detected and deleted: {e}")

                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                self.logger.success(
                    f"✓ Converted {zip_path.name} → {parquet_name} "
                    f"({total_rows:,} rows, {file_size_mb:.2f} MB)"
                )

                return output_path

            else:
                self.logger.warning(f"No data found in {zip_path.name}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to convert {zip_path.name}: {str(e)}")
            return None

    def _create_arrow_batch(self, data: List[Dict]) -> pa.RecordBatch:
        """
        Create Arrow RecordBatch from parsed data

        Args:
            data: List of parsed row dictionaries

        Returns:
            Arrow RecordBatch
        """
        # Prepare columnar data
        arrays = {
            'trade_id': [],
            'price': [],
            'qty': [],
            'quoteQty': [],
            'time': [],
            'isBuyerMaker': [],
            'isBestMatch': []
        }

        for row in data:
            arrays['trade_id'].append(row['trade_id'])
            arrays['price'].append(row['price'])
            arrays['qty'].append(row['qty'])
            arrays['quoteQty'].append(row['quoteQty'])
            arrays['time'].append(row['time'])
            arrays['isBuyerMaker'].append(row['isBuyerMaker'])
            arrays['isBestMatch'].append(row['isBestMatch'])

        # Convert to Arrow arrays with proper types
        arrow_arrays = [
            pa.array(arrays['trade_id'], type=pa.int64()),
            pa.array(arrays['price'], type=pa.float32()),
            pa.array(arrays['qty'], type=pa.float32()),
            pa.array(arrays['quoteQty'], type=pa.float32()),
            pa.array(arrays['time'], type=pa.timestamp('ms')),
            pa.array(arrays['isBuyerMaker'], type=pa.bool_()),
            pa.array(arrays['isBestMatch'], type=pa.bool_())
        ]

        return pa.RecordBatch.from_arrays(arrow_arrays, names=self.column_names)

    def _process_single_zip(self, zip_path: Path, delete_after: bool = False) -> Tuple[bool, float]:
        """
        Process a single ZIP file (helper for parallel processing)

        Args:
            zip_path: Path to ZIP file
            delete_after: Delete ZIP after successful conversion

        Returns:
            Tuple of (success, freed_space_mb)
        """
        freed_space_mb = 0.0

        try:
            # Convert to Parquet
            parquet_path = self.convert_zip_to_parquet(zip_path)

            if parquet_path:
                # Optionally delete ZIP after successful conversion
                if delete_after:
                    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                    zip_path.unlink()

                    # Also delete checksum file if exists
                    checksum_path = Path(str(zip_path) + ".CHECKSUM")
                    if checksum_path.exists():
                        checksum_path.unlink()

                    freed_space_mb = zip_size_mb
                    self.logger.info(f"Deleted {zip_path.name} (freed {zip_size_mb:.2f} MB)")

                return True, freed_space_mb
            else:
                return False, 0.0

        except Exception as e:
            self.logger.error(f"Failed to process {zip_path.name}: {e}")
            return False, 0.0

    def process_all_zips(self, skip_existing: bool = True, delete_zip_after: bool = False,
                        parallel: bool = False, max_workers: Optional[int] = None) -> Tuple[int, int, int]:
        """
        Process all ZIP files in the raw directory

        Args:
            skip_existing: Skip if Parquet already exists
            delete_zip_after: Delete ZIP file after successful conversion
            parallel: Enable parallel processing
            max_workers: Number of parallel workers (None for auto-detect)

        Returns:
            Tuple of (success_count, skip_count, fail_count)
        """
        # Find all ZIP files
        zip_pattern = f"{self.symbol}-*trades*.zip"
        zip_files = sorted(self.raw_dir.glob(zip_pattern))

        if not zip_files:
            self.logger.warning(f"No ZIP files found matching pattern: {zip_pattern}")
            return 0, 0, 0

        self.logger.info(f"Found {len(zip_files)} ZIP files to process")

        # Filter files to process (check if output parquet exists)
        files_to_process = []
        skip_count = 0

        for zip_path in zip_files:
            # Check if output parquet already exists
            zip_name = zip_path.stem
            parquet_name = f"{zip_name}.parquet"
            output_path = self.compressed_dir / parquet_name

            if output_path.exists() and skip_existing:
                skip_count += 1
                continue
            files_to_process.append(zip_path)

        if not files_to_process:
            self.logger.info(f"All files already processed (skipped {skip_count})")
            return 0, skip_count, 0

        success_count = 0
        fail_count = 0
        total_freed_space = 0

        # Process files (parallel or sequential)
        if parallel:
            # Determine optimal worker count
            if max_workers is None:
                cpu_count = mp.cpu_count()
                # Use fewer workers for I/O bound tasks
                max_workers = min(cpu_count - 1, 4)

            self.logger.info(f"Processing {len(files_to_process)} files in parallel with {max_workers} workers")

            # Process in parallel using ThreadPoolExecutor (better for I/O)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all conversion tasks
                future_to_zip = {
                    executor.submit(self._process_single_zip, zip_path, delete_zip_after): zip_path
                    for zip_path in files_to_process
                }

                # Process results with progress bar
                with tqdm(total=len(files_to_process), desc="Converting ZIPs to Parquet") as pbar:
                    for future in as_completed(future_to_zip):
                        zip_path = future_to_zip[future]

                        try:
                            success, freed_space = future.result()
                            if success:
                                success_count += 1
                                total_freed_space += freed_space
                            else:
                                fail_count += 1
                        except Exception as e:
                            self.logger.error(f"Error processing {zip_path.name}: {e}")
                            fail_count += 1

                        pbar.update(1)

        else:
            # Sequential processing (original implementation)
            with tqdm(total=len(files_to_process), desc="Converting ZIPs to Parquet") as pbar:
                for zip_path in files_to_process:
                    # Convert to Parquet
                    parquet_path = self.convert_zip_to_parquet(zip_path)

                    if parquet_path:
                        success_count += 1

                        # Optionally delete ZIP after successful conversion
                        if delete_zip_after:
                            zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
                            zip_path.unlink()

                            # Also delete checksum file if exists
                            checksum_path = Path(str(zip_path) + ".CHECKSUM")
                            if checksum_path.exists():
                                checksum_path.unlink()

                            total_freed_space += zip_size_mb
                            self.logger.info(f"Deleted {zip_path.name} (freed {zip_size_mb:.2f} MB)")
                    else:
                        fail_count += 1

                    pbar.update(1)

        # Summary
        self.logger.info("=" * 50)
        self.logger.success(f"✓ Successfully converted: {success_count} files")
        if skip_count > 0:
            self.logger.info(f"⊘ Skipped (already exist): {skip_count} files")
        if fail_count > 0:
            self.logger.error(f"✗ Failed: {fail_count} files")
        if delete_zip_after and total_freed_space > 0:
            self.logger.info(f"💾 Freed disk space: {total_freed_space:.2f} MB")

        return success_count, skip_count, fail_count

    def get_statistics(self) -> Dict:
        """Get conversion statistics"""
        # Count actual parquet files
        parquet_files = list(self.compressed_dir.glob(f"{self.symbol}-*.parquet"))
        total_parquet_size = sum(pf.stat().st_size for pf in parquet_files)

        stats = {
            "total_parquet_files": len(parquet_files),
            "total_parquet_size_gb": total_parquet_size / (1024**3)
        }

        return stats


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description="Convert ZIP files directly to Parquet")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--type", default="spot", choices=["spot", "futures"],
                       help="Data type")
    parser.add_argument("--futures-type", default="um", choices=["um", "cm"],
                       help="Futures type (if applicable)")
    parser.add_argument("--granularity", default="monthly", choices=["daily", "monthly"],
                       help="Data granularity")
    parser.add_argument("--compression", default="snappy",
                       choices=["snappy", "zstd", "lz4", "brotli"],
                       help="Compression algorithm")
    parser.add_argument("--chunk-size", type=int, default=5_000_000,
                       help="Rows to process per chunk")
    parser.add_argument("--delete-zip", action="store_true",
                       help="Delete ZIP files after successful conversion")

    args = parser.parse_args()

    # Initialize converter
    converter = ZipToParquetStreamer(
        symbol=args.symbol,
        data_type=args.type,
        futures_type=args.futures_type,
        granularity=args.granularity,
        compression=args.compression,
        chunk_size=args.chunk_size
    )

    # Process all ZIPs
    success, skip, fail = converter.process_all_zips(
        skip_existing=True,
        delete_zip_after=args.delete_zip
    )

    # Show statistics
    stats = converter.get_statistics()
    print(f"\n📊 Conversion Statistics:")
    print(f"  Total Parquet Files: {stats['total_parquet_files']}")
    print(f"  Total Parquet Size: {stats['total_parquet_size_gb']:.2f} GB")


if __name__ == "__main__":
    main()