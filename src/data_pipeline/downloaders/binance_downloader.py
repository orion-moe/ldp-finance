"""
Binance Data Downloader - Interactive Main Script
Downloads and processes Binance trading data (spot/futures) with daily/monthly granularity
"""

import sys
import requests
import concurrent.futures
import hashlib
import zipfile
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import argparse
import logging
import json
import time

# Column names for CSV files
COLUMN_NAMES = [
    'trade_id',
    'price',
    'qty',
    'quoteQty',
    'time',
    'isBuyerMaker',
    'isBestMatch'
]

# Alternative column names for CSV files with headers
ALT_COLUMN_NAMES = {
    'id': 'trade_id',
    'price': 'price',
    'qty': 'qty',
    'quote_qty': 'quoteQty',
    'time': 'time',
    'is_buyer_maker': 'isBuyerMaker'
}

class BinanceDataDownloader:
    def __init__(self, symbol: str = "BTCUSDT", data_type: str = "spot",
                 futures_type: str = "um", granularity: str = "daily",
                 base_dir: Path = Path("."), stop_on_error: bool = True):
        """
        Initialize Binance Data Downloader

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            data_type: 'spot' or 'futures'
            futures_type: 'um' (USD-M) or 'cm' (COIN-M) - only used if data_type is 'futures'
            granularity: 'daily' or 'monthly'
            base_dir: Base directory for data storage
            stop_on_error: Stop processing on errors (default: True)
        """
        self.symbol = symbol
        self.data_type = data_type
        self.futures_type = futures_type if data_type == "futures" else None
        self.granularity = granularity

        # Ensure base_dir always points to the datasets folder
        if base_dir == Path("."):
            # If running from project root, add datasets folder
            current_file = Path(__file__).resolve()
            datasets_dir = current_file.parent
            if datasets_dir.name == "datasets":
                self.base_dir = datasets_dir
            else:
                # If somehow not in datasets folder, use current directory
                self.base_dir = Path.cwd()
                if "datasets" not in str(self.base_dir):
                    # If not in datasets folder, append it
                    self.base_dir = self.base_dir / "datasets"
        else:
            self.base_dir = base_dir
        self.stop_on_error = stop_on_error

        # Set up logging
        self.setup_logging()

        # Progress tracking file
        self.progress_file = self.base_dir / f"download_progress_{symbol}_{data_type}_{granularity}.json"
        self.progress = self.load_progress()

        # Set up directories based on data type
        if self.data_type == "spot":
            self.raw_dir = self.base_dir / f"dataset-raw-{granularity}" / "spot"
            self.compressed_dir = self.base_dir / f"dataset-raw-{granularity}-compressed" / "spot"
        else:  # futures
            self.raw_dir = self.base_dir / f"dataset-raw-{granularity}" / f"futures-{futures_type}"
            self.compressed_dir = self.base_dir / f"dataset-raw-{granularity}-compressed" / f"futures-{futures_type}"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.compressed_dir.mkdir(parents=True, exist_ok=True)

        # Base URLs for different data types
        self.base_urls = {
            "spot": {
                "daily": f"https://data.binance.vision/data/spot/daily/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}-{{day:02d}}.zip",
                "monthly": f"https://data.binance.vision/data/spot/monthly/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}.zip"
            },
            "futures-um": {
                "daily": f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}-{{day:02d}}.zip",
                "monthly": f"https://data.binance.vision/data/futures/um/monthly/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}.zip"
            },
            "futures-cm": {
                "daily": f"https://data.binance.vision/data/futures/cm/daily/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}-{{day:02d}}.zip",
                "monthly": f"https://data.binance.vision/data/futures/cm/monthly/trades/{symbol}/{symbol}-trades-{{year}}-{{month:02d}}.zip"
            }
        }

    def setup_logging(self):
        """Set up logging configuration"""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'

        # Clear any existing handlers to avoid duplicates
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Create logs directory if it doesn't exist
        logs_dir = self.base_dir / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Create log filename with timestamp
        log_filename = f'download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        log_path = logs_dir / log_filename

        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(str(log_path)),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    def load_progress(self) -> Dict:
        """Load progress from file if exists"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                    # Use print if logger not available yet, otherwise use logger
                    if hasattr(self, 'logger'):
                        self.logger.info(f"Loaded progress from {self.progress_file}")
                    else:
                        print(f"Loaded progress from {self.progress_file}")
                    return progress
            except Exception as e:
                if hasattr(self, 'logger'):
                    self.logger.warning(f"Could not load progress file: {e}")
                else:
                    print(f"Warning: Could not load progress file: {e}")
        return {
            'downloaded': [],
            'processed': [],
            'failed': [],
            'processing_failed': [],
            'last_update': None
        }

    def save_progress(self):
        """Save current progress to file"""
        self.progress['last_update'] = datetime.now().isoformat()
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Could not save progress: {e}")
            else:
                print(f"Error: Could not save progress: {e}")

    def generate_dates(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """Generate dates based on granularity"""
        dates = []
        current = start_date

        if self.granularity == "daily":
            while current <= end_date:
                dates.append(current)
                current += timedelta(days=1)
        else:  # monthly
            while current <= end_date:
                dates.append(current)
                if current.month == 12:
                    current = datetime(current.year + 1, 1, 1)
                else:
                    current = datetime(current.year, current.month + 1, 1)

        return dates

    def detect_csv_format(self, csv_file: Path) -> tuple:
        """Detect CSV format and return (has_header, column_mapping)"""
        try:
            # Read first line to detect format
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()

            # Check if first line looks like a header
            first_parts = first_line.split(',')

            # Check if first line contains clear header indicators
            has_text_headers = any(
                part.lower().strip() in ['id', 'trade_id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
                for part in first_parts
            )

            # Also check if first part looks like a header (contains letters)
            first_part_clean = first_parts[0].lower().strip()
            looks_like_header = (
                first_part_clean in ['id', 'trade_id'] or
                any(c.isalpha() for c in first_part_clean)
            )

            if has_text_headers or looks_like_header:
                self.logger.info(f"📋 Detected CSV with header: {first_line}")

                # Map headers to standard column names
                column_mapping = {}
                for i, header in enumerate(first_parts):
                    clean_header = header.lower().strip()
                    if clean_header in ALT_COLUMN_NAMES:
                        column_mapping[i] = ALT_COLUMN_NAMES[clean_header]
                    elif clean_header in ['trade_id', 'price', 'qty', 'quotqty', 'time', 'isbuyermaker', 'isbestmatch']:
                        column_mapping[i] = clean_header

                return True, column_mapping
            else:
                self.logger.info(f"📋 Detected CSV without header, using standard format")
                return False, None

        except Exception as e:
            self.logger.warning(f"⚠️  Could not detect CSV format for {csv_file.name}: {e}")
            return False, None

    def get_file_processing_state(self, date: datetime) -> str:
        """Determine the current processing state of a file for given date"""
        if self.granularity == "daily":
            file_suffix = f"{date.year}-{date.month:02d}-{date.day:02d}"
        else:
            file_suffix = f"{date.year}-{date.month:02d}"

        zip_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip"
        csv_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.csv"

        # Check if already processed to parquet
        date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
        if date_str in self.progress.get('processed', []):
            # Double-check that parquet actually exists
            parquet_files = self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet")
            for pf in parquet_files:
                try:
                    # Quick check if this parquet contains the date
                    import pyarrow.parquet as pq
                    pf_handle = pq.ParquetFile(pf)
                    first_time = pf_handle.read_row_group(0, columns=['time']).to_pandas()['time'].iloc[0]
                    last_time = pf_handle.read_row_group(pf_handle.num_row_groups - 1, columns=['time']).to_pandas()['time'].iloc[-1]

                    if first_time.strftime('%Y-%m') <= date_str <= last_time.strftime('%Y-%m'):
                        return 'completed'
                except:
                    continue

            # If we couldn't verify parquet, just continue checking other states
            # Don't log warning here as it creates too much noise
            pass

        # Check if CSV exists
        if csv_file.exists():
            return 'csv_ready'

        # Check if ZIP exists
        if zip_file.exists():
            return 'zip_ready'

        return 'not_downloaded'

    def calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()

    def verify_checksum(self, file_path: Path, checksum_path: Path) -> bool:
        """Verify file integrity using checksum"""
        try:
            calculated_hash = self.calculate_file_hash(file_path)

            with open(checksum_path, "r") as f:
                checksum_data = f.read().strip()
                parts = checksum_data.split()
                expected_hash = parts[0] if parts else ""

            return calculated_hash == expected_hash
        except Exception as e:
            self.logger.error(f"Error verifying checksum: {e}")
            return False

    def download_file(self, url: str, dest_path: Path) -> bool:
        """Download a file with progress bar"""
        try:
            # Use longer timeout for large files (30 minutes total)
            response = requests.get(url, stream=True, timeout=(30, 1800))
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192

            # Log download start
            self.logger.info(f"📥 Downloading {dest_path.name} ({total_size / (1024*1024):.1f}MB)")

            with open(dest_path, 'wb') as f:
                downloaded = 0
                last_log = 0
                for data in response.iter_content(block_size):
                    downloaded += len(data)
                    f.write(data)
                    # Log progress every 100MB
                    if downloaded - last_log >= (100 * 1024 * 1024):
                        progress_pct = (downloaded / total_size * 100) if total_size > 0 else 0
                        self.logger.info(f"  → {progress_pct:.1f}% ({downloaded / (1024*1024):.0f}MB / {total_size / (1024*1024):.0f}MB)")
                        last_log = downloaded

            return True
        except Exception as e:
            self.logger.error(f"Error downloading {url}: {e}")
            return False

    def download_with_checksum(self, date: datetime) -> Tuple[Optional[Path], Optional[Path]]:
        """Download data file and its checksum - but only if not already processed"""
        # First check if we need to download anything at all
        state = self.get_file_processing_state(date)

        if state == 'completed':
            self.logger.info(f"⏭️  Skipping {date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')} - already processed to Parquet")
            return None, None

        if self.granularity == "daily":
            file_suffix = f"{date.year}-{date.month:02d}-{date.day:02d}"
        else:
            file_suffix = f"{date.year}-{date.month:02d}"

        zip_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip"
        checksum_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip.CHECKSUM"

        if state == 'csv_ready':
            self.logger.info(f"⏭️  Skipping download for {file_suffix} - CSV already exists")
            # Don't try to clean up ZIP files here - they might not exist
            return None, None

        if state == 'zip_ready':
            # ZIP exists, verify it and return
            if checksum_file.exists() and self.verify_checksum(zip_file, checksum_file):
                self.logger.info(f"✅ {zip_file.name} already exists and is valid")
                return zip_file, checksum_file
            else:
                self.logger.warning(f"❌ {zip_file.name} is corrupted, re-downloading...")
                zip_file.unlink(missing_ok=True)
                checksum_file.unlink(missing_ok=True)

        # Need to download - proceed with normal download logic
        url_key = f"futures-{self.futures_type}" if self.data_type == "futures" else "spot"
        base_url = self.base_urls[url_key][self.granularity]

        if self.granularity == "daily":
            url = base_url.format(year=date.year, month=date.month, day=date.day)
        else:
            url = base_url.format(year=date.year, month=date.month)

        # Download files
        self.logger.info(f"Downloading {file_suffix}...")

        # Download checksum first
        checksum_url = f"{url}.CHECKSUM"
        if not self.download_file(checksum_url, checksum_file):
            return None, None

        # Download data file
        if not self.download_file(url, zip_file):
            checksum_file.unlink(missing_ok=True)
            return None, None

        # Verify integrity
        if not self.verify_checksum(zip_file, checksum_file):
            self.logger.error(f"❌ Downloaded file {zip_file.name} failed integrity check")
            zip_file.unlink(missing_ok=True)
            checksum_file.unlink(missing_ok=True)
            return None, None

        self.logger.info(f"✅ Successfully downloaded and verified {zip_file.name}")

        # Mark as downloaded
        date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
        if date_str not in self.progress.get('downloaded', []):
            self.progress['downloaded'].append(date_str)
            self.save_progress()
        return zip_file, checksum_file

    def extract_zip(self, zip_path: Path) -> Optional[Path]:
        """Extract zip file, return CSV path, and delete ZIP after successful extraction"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                csv_filename = None
                for name in zf.namelist():
                    if name.endswith('.csv'):
                        csv_filename = name
                        break

                if csv_filename:
                    csv_path = self.raw_dir / csv_filename
                    # Check if CSV already exists
                    if csv_path.exists():
                        self.logger.info(f"✅ CSV already exists: {csv_filename}, skipping extraction")
                        return csv_path

                    # Extract CSV
                    zf.extract(csv_filename, self.raw_dir)
                    self.logger.info(f"✅ Extracted: {csv_filename}")

                    # Keep ZIP file - don't delete in step 1
                    return csv_path

            return None
        except Exception as e:
            self.logger.error(f"Error extracting {zip_path}: {e}")
            if self.stop_on_error:
                raise
            return None

    def process_csv_to_parquet(self, csv_files: List[Path], max_size_gb: int = 10) -> bool:
        """Process CSV files and save as Parquet with memory-efficient chunked processing."""
        if not csv_files:
            self.logger.info("No CSV files to process")
            return True

        self.logger.info(f"Processing {len(csv_files)} CSV files...")

        # Verify which files actually need processing
        files_to_process = []
        for csv_file in csv_files:
            # Extract date from filename
            if self.granularity == "daily":
                # Pattern: BTCUSDT-trades-YYYY-MM-DD.csv
                date_match = csv_file.stem.split('-trades-')[-1]
                date_str = date_match
            else:
                # Pattern: BTCUSDT-trades-YYYY-MM.csv
                date_match = csv_file.stem.split('-trades-')[-1]
                date_str = date_match

            # Check if already processed
            if date_str in self.progress.get('processed', []):
                self.logger.info(f"⏭️  Skipping {csv_file.name} - already processed to Parquet")
                continue

            files_to_process.append(csv_file)

        if not files_to_process:
            self.logger.info("✅ All CSV files have already been processed to Parquet")
            return True

        self.logger.info(f"📊 Processing {len(files_to_process)} CSV files (skipped {len(csv_files) - len(files_to_process)} already processed)...")

        # Find existing parquet files to determine file count
        existing_parquets = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))
        file_count = len(existing_parquets) + 1 if existing_parquets else 1

        max_size_bytes = max_size_gb * 1024**3
        writer = None
        output_path = None
        processed_files = []

        total_files = len(files_to_process)
        for idx, csv_file in enumerate(files_to_process, 1):
            self.logger.info(f"Processing file {idx}/{total_files}: {csv_file.name}")

            try:
                # Detect CSV format first
                has_header, column_mapping = self.detect_csv_format(csv_file)

                # Process CSV in optimized chunks (~100MB each)
                # Estimated: ~49 bytes per row, 100MB = ~2.1M rows
                chunk_size = 2_100_000  # Optimized chunk size for ~100MB memory usage
                chunk_count = 0

                # Set up pandas read parameters optimized for large chunks
                read_params = {
                    'chunksize': chunk_size,
                    'low_memory': False,  # Allow pandas to infer dtypes for better performance
                    'engine': 'c',  # Use faster C engine
                    'dtype': {
                        'trade_id': 'int64',
                        'price': 'float64',
                        'qty': 'float64',
                        'quoteQty': 'float64',
                        'time': 'int64',  # Will convert to datetime later
                        'isBuyerMaker': 'object',  # Handle as object first, convert later
                        'isBestMatch': 'object'     # Handle as object first, convert later
                    }
                }

                if has_header:
                    # CSV has header, use it
                    read_params['header'] = 0
                    # Remove dtype for header files since pandas will infer from column names
                    read_params.pop('dtype', None)
                else:
                    # No header, use standard column names
                    read_params['header'] = None
                    read_params['names'] = COLUMN_NAMES
                    # dtype is already set above for better performance

                for chunk in pd.read_csv(csv_file, **read_params):
                    chunk_count += 1

                    # Standardize column names if needed
                    if has_header and column_mapping:
                        # Rename columns to standard names
                        rename_map = {}
                        for old_name, new_name in column_mapping.items():
                            if old_name < len(chunk.columns):
                                rename_map[chunk.columns[old_name]] = new_name
                        if rename_map:
                            chunk = chunk.rename(columns=rename_map)

                    # Convert boolean columns from object to bool
                    if 'isBuyerMaker' in chunk.columns and chunk['isBuyerMaker'].dtype == 'object':
                        chunk['isBuyerMaker'] = chunk['isBuyerMaker'].map({'True': True, 'False': False, True: True, False: False}).fillna(False).astype(bool)
                    if 'isBestMatch' in chunk.columns and chunk['isBestMatch'].dtype == 'object':
                        chunk['isBestMatch'] = chunk['isBestMatch'].map({'True': True, 'False': False, True: True, False: False}).fillna(False).astype(bool)

                    # Ensure we have the time column
                    if 'time' not in chunk.columns:
                        self.logger.error(f"❌ No 'time' column found in {csv_file.name}")
                        self.logger.info(f"Available columns: {list(chunk.columns)}")
                        raise ValueError(f"Missing 'time' column in {csv_file.name}")

                    # Convert time from microseconds to datetime (Bitcoin data uses microseconds)
                    try:
                        # First try microseconds conversion (Bitcoin futures/spot data format)
                        chunk['time'] = pd.to_datetime(chunk['time'], unit='us')
                    except (ValueError, OverflowError, OSError) as e:
                        self.logger.warning(f"⚠️  Microseconds time conversion failed, trying alternative methods: {e}")
                        try:
                            # Get timestamp statistics for intelligent detection
                            max_timestamp = chunk['time'].max()
                            min_timestamp = chunk['time'].min()
                            sample_value = chunk['time'].iloc[0]

                            self.logger.info(f"🔍 Timestamp analysis - Sample: {sample_value}, Min: {min_timestamp}, Max: {max_timestamp}")

                            # Intelligent timestamp unit detection based on value ranges
                            # 2025 reference timestamps:
                            # Seconds: ~1735689600 (10 digits)
                            # Milliseconds: ~1735689600000 (13 digits)
                            # Microseconds: ~1735689600000000 (16 digits) - Bitcoin data format
                            # Nanoseconds: ~1735689600000000000 (19 digits)

                            # Prioritize microseconds for Bitcoin trading data
                            if max_timestamp > 1e15 and max_timestamp < 1e18:  # 16-18 digits = microseconds (Bitcoin format)
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='us')
                                self.logger.info(f"✅ Successfully converted time using microseconds unit (Bitcoin format)")
                            elif max_timestamp > 1e18:  # 19+ digits = nanoseconds
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='ns')
                                self.logger.info(f"✅ Successfully converted time using nanoseconds unit")
                            elif max_timestamp > 1e12:  # 13-15 digits = milliseconds
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='ms', errors='coerce')
                                self.logger.info(f"✅ Successfully converted time using milliseconds unit (with coercion)")
                            else:  # 10+ digits = seconds
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='s')
                                self.logger.info(f"✅ Successfully converted time using seconds unit")
                        except Exception as e2:
                            self.logger.warning(f"⚠️  Smart detection failed, trying direct conversion: {e2}")
                            try:
                                # Try direct conversion as last resort
                                chunk['time'] = pd.to_datetime(chunk['time'], errors='coerce')
                                self.logger.info(f"✅ Successfully converted time using direct conversion with coercion")
                            except Exception as e3:
                                self.logger.error(f"❌ All time conversion methods failed: {e3}")
                                self.logger.info(f"Sample time values: {chunk['time'].head()}")
                                try:
                                    self.logger.info(f"Value analysis - digits: {len(str(int(sample_value)))}")
                                except:
                                    self.logger.info(f"Could not analyze sample value: {sample_value}")
                                raise

                    # Convert to PyArrow table
                    table = pa.Table.from_pandas(chunk, preserve_index=False)

                    # Check if we need a new parquet file
                    if writer is None or (output_path and output_path.exists() and
                                        output_path.stat().st_size >= max_size_bytes):
                        if writer:
                            writer.close()
                            writer = None
                            self.logger.info(f"✅ Completed parquet file: {output_path.name} "
                                           f"({output_path.stat().st_size / (1024**3):.2f} GB)")

                        # Create new parquet file
                        # Use date from CSV filename for parquet naming
                        csv_date = csv_file.stem.split('-trades-')[-1]  # Extract date from CSV name
                        output_path = self.compressed_dir / f"{self.symbol}-Trades-{csv_date}.parquet"
                        writer = pq.ParquetWriter(output_path, table.schema, compression='snappy')
                        file_count += 1
                        self.logger.info(f"📄 Creating new parquet file: {output_path.name}")

                    # Write chunk to parquet
                    writer.write_table(table)

                    # Log progress every 2 chunks (since chunks are now ~100MB each)
                    if chunk_count % 2 == 0:
                        current_size_mb = output_path.stat().st_size / (1024**2) if output_path.exists() else 0
                        processed_mb = chunk_count * 100  # Approximate MB processed
                        self.logger.info(f"  Processed {chunk_count} chunks (~{processed_mb} MB) from {csv_file.name} "
                                       f"(current file: {current_size_mb:.1f} MB)")

                    # Clear chunk from memory
                    del chunk, table

                processed_files.append(csv_file)
                self.logger.info(f"✅ Successfully processed {csv_file.name} ({chunk_count} chunks)")

                # Mark this date as processed in progress tracking
                if self.granularity == "daily":
                    date_str = csv_file.stem.split('-trades-')[-1]
                else:
                    date_str = csv_file.stem.split('-trades-')[-1]

                if date_str not in self.progress.get('processed', []):
                    self.progress['processed'].append(date_str)
                    self.save_progress()
                    self.logger.info(f"📝 Marked {date_str} as processed in progress tracking")

            except Exception as e:
                self.logger.error(f"❌ Error processing {csv_file}: {e}")
                if writer:
                    writer.close()
                    writer = None
                if self.stop_on_error:
                    self.logger.error(f"⚠️  Stopping due to error. Files processed so far have been saved.")
                    # Clean up successfully processed CSV files before stopping
                    for processed_csv in processed_files:
                        try:
                            processed_csv.unlink()
                            self.logger.info(f"🗑️  Cleaned up: {processed_csv.name}")
                        except Exception as cleanup_e:
                            self.logger.warning(f"⚠️  Could not cleanup {processed_csv.name}: {cleanup_e}")
                    return False
                continue

        # Close final writer
        if writer:
            writer.close()
            if output_path and output_path.exists():
                self.logger.info(f"✅ Completed final parquet file: {output_path.name} "
                               f"({output_path.stat().st_size / (1024**3):.2f} GB)")

        # Clean up successfully processed CSV files
        for csv_file in processed_files:
            try:
                csv_file.unlink()
                self.logger.info(f"🗑️  Cleaned up CSV: {csv_file.name}")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not cleanup {csv_file.name}: {e}")

        self.logger.info(f"✅ Processing completed - {len(processed_files)}/{total_files} files processed successfully")
        return len(processed_files) == total_files

    def process_file_with_retry(self, date: datetime, zip_file: Path, max_retries: int = 3) -> bool:
        """Process a single file with retry logic for parquet conversion"""
        date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')

        # Check if already processed
        if date_str in self.progress.get('processed', []):
            self.logger.info(f"Skipping {zip_file.name} - already processed in previous run")
            return True

        for attempt in range(1, max_retries + 1):
            try:
                self.logger.info(f"🔄 Processing attempt {attempt}/{max_retries} for {zip_file.name}")

                # Step 1: Extract ZIP file
                csv_file = self.extract_zip(zip_file)
                if not csv_file:
                    self.logger.error(f"❌ Failed to extract {zip_file.name}")
                    if attempt < max_retries:
                        wait_time = 2 ** attempt  # Exponential backoff
                        self.logger.info(f"⏳ Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    else:
                        return False

                self.logger.info(f"✅ Extracted: {csv_file.name}")

                # Step 2: Convert CSV to Parquet with retry
                success = False
                for csv_attempt in range(1, max_retries + 1):
                    try:
                        self.logger.info(f"🔄 Converting {csv_file.name} to Parquet (attempt {csv_attempt}/{max_retries})...")
                        success = self.process_csv_to_parquet([csv_file])
                        if success:
                            self.logger.info(f"✅ Successfully converted to Parquet")
                            break
                        else:
                            self.logger.warning(f"⚠️  Conversion failed, attempt {csv_attempt}/{max_retries}")
                    except Exception as csv_e:
                        self.logger.error(f"❌ Error in CSV conversion attempt {csv_attempt}: {csv_e}")
                        if csv_attempt < max_retries:
                            wait_time = 2 ** csv_attempt
                            self.logger.info(f"⏳ Retrying CSV conversion in {wait_time} seconds...")
                            time.sleep(wait_time)
                        else:
                            # If CSV still exists, remove it before final failure
                            if csv_file.exists():
                                csv_file.unlink()
                                self.logger.info(f"🗑️  Removed failed CSV: {csv_file.name}")
                            raise

                if success:
                    # Mark as processed
                    if date_str not in self.progress.get('processed', []):
                        self.progress['processed'].append(date_str)
                        self.save_progress()
                    return True
                else:
                    self.logger.error(f"❌ Failed to convert {csv_file.name} to Parquet after {max_retries} attempts")
                    return False

            except Exception as e:
                self.logger.error(f"❌ Error in processing attempt {attempt} for {zip_file.name}: {e}")
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    self.logger.info(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    # Mark as processing failed
                    if date_str not in self.progress.get('processing_failed', []):
                        self.progress['processing_failed'].append(date_str)
                        self.save_progress()
                    return False

        return False

    def cleanup_orphaned_zips(self) -> int:
        """Clean up ZIP files where CSV already exists"""
        cleaned_count = 0
        
        # Find all ZIP files
        zip_files = list(self.raw_dir.glob(f"{self.symbol}-trades-*.zip"))
        
        for zip_file in zip_files:
            # Check if corresponding CSV exists
            csv_file = zip_file.with_suffix('.csv')
            if csv_file.exists():
                try:
                    zip_file.unlink()
                    checksum_file = zip_file.with_suffix('.zip.CHECKSUM')
                    checksum_file.unlink(missing_ok=True)
                    self.logger.info(f"🗑️  Cleaned up orphaned ZIP: {zip_file.name}")
                    cleaned_count += 1
                except Exception as e:
                    self.logger.warning(f"⚠️  Could not cleanup {zip_file.name}: {e}")
        
        if cleaned_count > 0:
            self.logger.info(f"✅ Cleaned up {cleaned_count} orphaned ZIP files")
        
        return cleaned_count

    def print_download_summary(self, total_dates: int, successful: int, failed: int):
        """Print a summary of the download operation"""
        processed_count = len(self.progress.get('processed', []))
        processing_failed_count = len(self.progress.get('processing_failed', []))

        summary = f"\n{'='*60}\n DOWNLOAD & PROCESSING SUMMARY \n{'='*60}\n"
        summary += f"Total dates requested: {total_dates}\n"
        summary += f"✅ Successfully downloaded: {successful}\n"
        summary += f"✅ Successfully processed to Parquet: {processed_count}\n"
        summary += f"❌ Download failed/missing: {failed}\n"
        summary += f"❌ Processing failed (after retries): {processing_failed_count}\n"
        summary += f"Download success rate: {(successful/total_dates*100):.1f}%\n"
        if successful > 0:
            summary += f"Processing success rate: {(processed_count/successful*100):.1f}%\n"

        if failed > 0:
            summary += f"\n💡 Tips for missing downloads:\n"
            summary += "  1. Run 'python check_missing_data.py --fix' to identify recoverable dates\n"
            summary += "  2. Some dates may be weekends or holidays with no trading\n"
            summary += f"  3. Check {self.base_dir}/failed_downloads.txt for the complete list\n"

        if processing_failed_count > 0:
            summary += f"\n⚠️  Processing failures:\n"
            summary += f"  - {processing_failed_count} files failed to convert to Parquet after 3 retry attempts\n"
            summary += f"  - These ZIP files are preserved for manual inspection or retry\n"
            summary += f"  - Re-run the script to retry failed processing\n"

        summary += "="*60
        self.logger.info(summary)

    def download_monthly_data(self, start_month: str, end_month: str, 
                            specific_months: List[Tuple[int, int]] = None,
                            max_workers: int = 5) -> Tuple[int, int]:
        """
        Download monthly data with support for specific months
        
        Args:
            start_month: Start month in YYYY-MM format
            end_month: End month in YYYY-MM format
            specific_months: List of (year, month) tuples to download (overrides range)
            max_workers: Number of parallel workers
            
        Returns:
            Tuple of (successful_count, failed_count)
        """
        try:
            # Parse dates
            start_date = datetime.strptime(start_month, '%Y-%m')
            end_date = datetime.strptime(end_month, '%Y-%m')
            
            # If specific months provided, filter dates
            if specific_months:
                dates = []
                for year, month in specific_months:
                    dates.append(datetime(year, month, 1))
            else:
                dates = self.generate_dates(start_date, end_date)
                
            # Download files
            successful = 0
            failed = 0
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_date = {
                    executor.submit(self.download_with_checksum, date): date
                    for date in dates
                }
                
                for future in concurrent.futures.as_completed(future_to_date):
                    date = future_to_date[future]
                    try:
                        zip_file, checksum_file = future.result()
                        if zip_file and checksum_file:
                            successful += 1
                            self.logger.info(f"✅ Downloaded: {zip_file.name}")
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        self.logger.error(f"❌ Failed to download {date}: {e}")
                        
            return successful, failed
            
        except Exception as e:
            self.logger.error(f"❌ Error in download_monthly_data: {e}")
            return 0, len(specific_months) if specific_months else 0
    
    def download_date_range(self, start_date: datetime, end_date: datetime,
                          max_workers: int = 5) -> None:
        """Download data for a date range"""
        # Exclude current day if end_date is today (data might not be complete)
        current_date = datetime.now()
        if end_date.date() == current_date.date() and self.granularity == 'daily':
            end_date = end_date - timedelta(days=1)
            self.logger.info(f"📅 Excluding current day {current_date.date()} from download (incomplete data)")

        dates = self.generate_dates(start_date, end_date)

        self.logger.info(f"Downloading {self.data_type} {self.granularity} data from "
                        f"{start_date.date()} to {end_date.date()}")
        self.logger.info(f"Total periods to download: {len(dates)}")

        # Pre-check what's already processed
        already_processed = []
        need_download = []
        for date in dates:
            date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
            if date_str in self.progress.get('processed', []):
                already_processed.append(date_str)
            else:
                need_download.append(date)

        # Always show pre-check summary, even if nothing is processed yet
        self.logger.info(f"📊 Pre-check summary:")
        self.logger.info(f"  - Already processed to Parquet: {len(already_processed)} dates")
        self.logger.info(f"  - Need to download/process: {len(need_download)} dates")

        if already_processed:
            if len(already_processed) <= 10:
                for date_str in already_processed:
                    self.logger.info(f"    ✅ {date_str} - already in Parquet")
            else:
                self.logger.info(f"    ✅ {already_processed[0]} ... {already_processed[-1]} ({len(already_processed)} dates)")

        # Download files
        downloaded_files = []
        failed_downloads = []
        skipped_processed = 0  # Count files skipped because already processed

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_date = {
                executor.submit(self.download_with_checksum, date): date
                for date in dates
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    zip_file, checksum_file = future.result()
                    if zip_file and checksum_file:
                        downloaded_files.append((date, zip_file))
                    else:
                        # Check if it was skipped because already processed
                        date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
                        if date_str in self.progress.get('processed', []):
                            skipped_processed += 1
                        else:
                            # Only add to failed if not already processed
                            failed_downloads.append(date)
                except Exception as e:
                    self.logger.error(f"Exception for {date}: {e}")
                    failed_downloads.append(date)

        if failed_downloads:
            self.logger.warning(f"⚠️  Failed to download {len(failed_downloads)} files")
            
            # For monthly data, weekend logic doesn't apply (monthly data is always available)
            if self.granularity == "daily":
                # Group by reason if possible
                weekends = [d for d in failed_downloads if d.weekday() in [5, 6]]
                others = [d for d in failed_downloads if d.weekday() not in [5, 6]]

                if weekends:
                    self.logger.info(f"  Weekends (no trading): {len(weekends)} days")
                    for date in weekends[:5]:
                        self.logger.info(f"    - {date.strftime('%Y-%m-%d (%A)')}")
                    if len(weekends) > 5:
                        self.logger.info(f"    ... and {len(weekends) - 5} more")

                if others:
                    self.logger.info(f"  Other missing dates: {len(others)} days")
                    for date in others[:10]:
                        self.logger.info(f"    - {date.strftime('%Y-%m-%d (%A)')}")
                    if len(others) > 10:
                        self.logger.info(f"    ... and {len(others) - 10} more")
            else:
                # For monthly data, all failures are "other" (not weekend-related)
                self.logger.info(f"  Failed monthly downloads: {len(failed_downloads)} months")
                for date in failed_downloads[:10]:
                    self.logger.info(f"    - {date.strftime('%Y-%m')}")
                if len(failed_downloads) > 10:
                    self.logger.info(f"    ... and {len(failed_downloads) - 10} more")

            # Save failed downloads to file
            failed_downloads_path = self.base_dir / 'failed_downloads.txt'
            with open(failed_downloads_path, 'w') as f:
                f.write(f"Failed downloads for {self.symbol} {self.data_type} {self.granularity}\n")
                f.write(f"Generated on {datetime.now()}\n\n")
                for date in sorted(failed_downloads):
                    f.write(f"{date.strftime('%Y-%m-%d')}\n")
            self.logger.info(f"💾 Failed downloads saved to: {failed_downloads_path}")

            # Update progress with failed downloads
            for date in failed_downloads:
                date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
                if date_str not in self.progress.get('failed', []):
                    self.progress['failed'].append(date_str)
            self.save_progress()

        # Note: CSV file detection is now handled in the unified state checking above

        # Ensure all files are downloaded before proceeding to extraction
        if skipped_processed > 0:
            self.logger.info(f"✅ Download phase complete: {len(downloaded_files)} new files, {skipped_processed} already processed, {len(failed_downloads)} failed")
        else:
            self.logger.info(f"✅ Download phase complete: {len(downloaded_files)} files downloaded, {len(failed_downloads)} failed")

        # Collect all files that need processing from different states
        files_to_process = []

        # Check all requested dates for files that need processing
        for date in dates:
            state = self.get_file_processing_state(date)

            if state == 'completed':
                # Already processed to Parquet, skip completely
                self.logger.info(f"⏭️  Skipping {date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')} - already processed to Parquet")
                continue
            elif state == 'csv_ready':
                # CSV exists, needs conversion to Parquet
                if self.granularity == "daily":
                    file_suffix = f"{date.year}-{date.month:02d}-{date.day:02d}"
                else:
                    file_suffix = f"{date.year}-{date.month:02d}"
                csv_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.csv"
                files_to_process.append((date, csv_file, 'csv_ready'))
            elif state == 'zip_ready':
                # ZIP exists, needs extraction then conversion
                if self.granularity == "daily":
                    file_suffix = f"{date.year}-{date.month:02d}-{date.day:02d}"
                else:
                    file_suffix = f"{date.year}-{date.month:02d}"
                zip_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip"
                files_to_process.append((date, zip_file, 'zip_ready'))
            # Note: 'not_downloaded' files are handled by the download phase above

        # Add any files that were just downloaded
        for date, zip_file in downloaded_files:
            # Double-check they're not already in the list
            date_key = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
            already_listed = any(d.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m') == date_key
                               for d, _, _ in files_to_process)
            if not already_listed:
                files_to_process.append((date, zip_file, 'zip_ready'))

        if files_to_process:
            self.logger.info(f"Processing {len(files_to_process)} files from various states...")

            total_files = len(files_to_process)
            for idx, (date, file_path, state) in enumerate(files_to_process, 1):
                self.logger.info(f"📁 Processing file {idx}/{total_files}: {file_path.name} (state: {state})...")

                success = False
                if state == 'zip_ready':
                    # Process ZIP file with retry logic
                    success = self.process_file_with_retry(date, file_path, max_retries=3)
                elif state == 'csv_ready':
                    # Process CSV file directly
                    try:
                        success = self.process_csv_to_parquet([file_path])
                        if success:
                            date_str = date.strftime('%Y-%m-%d' if self.granularity == 'daily' else '%Y-%m')
                            if date_str not in self.progress.get('processed', []):
                                self.progress['processed'].append(date_str)
                                self.save_progress()
                    except Exception as e:
                        self.logger.error(f"❌ Error processing CSV {file_path.name}: {e}")
                        success = False

                if not success:
                    self.logger.error(f"❌ Failed to process {file_path.name} after all retry attempts")
                    if self.stop_on_error:
                        self.logger.error(f"⚠️  Stopping due to error. File {file_path.name} preserved for retry.")
                        raise RuntimeError(f"Failed to process {file_path.name} after all retry attempts")
                    else:
                        self.logger.warning(f"⚠️  Continuing with next file. File {file_path.name} preserved for retry.")

        # Note: existing CSV files are now handled in the unified processing above

        self.logger.info(f"📊 Summary: Found {len(files_to_process)} files that need processing")
        if files_to_process:
            csv_count = sum(1 for _, _, state in files_to_process if state == 'csv_ready')
            zip_count = sum(1 for _, _, state in files_to_process if state == 'zip_ready')
            self.logger.info(f"  - {csv_count} CSV files ready for Parquet conversion")
            self.logger.info(f"  - {zip_count} ZIP files ready for extraction and conversion")

        # Print summary
        self.print_download_summary(len(dates), len(downloaded_files), len(failed_downloads))

        # Verify all downloads with checksums
        if downloaded_files:
            self.verify_all_downloads(dates)

        # Verify data integrity
        self.verify_data_integrity()

        # Optional: Optimize parquet files after processing (only if no files were processed in this run)
        if skipped_processed > 0 and len(downloaded_files) == 0 and len(files_to_process) == 0:
            self.logger.info("\n💡 All files are already processed to Parquet.")
            response = input("\nWould you like to optimize the Parquet files (combine into 10GB files)? (y/n): ").strip().lower()
            if response == 'y':
                self.optimize_parquet_files()

    def verify_data_integrity(self):
        """Verify that all processed dates are actually in the Parquet files"""
        self.logger.info("\n🔍 Verifying data integrity...")

        # Get all dates marked as processed
        processed_dates = self.progress.get('processed', [])
        if not processed_dates:
            self.logger.info("No processed dates to verify")
            return

        # Check Parquet files
        parquet_files = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))
        if not parquet_files:
            self.logger.warning("⚠️  No Parquet files found but progress shows processed dates!")
            return

        # Build a map of dates actually in Parquet files
        dates_in_parquet = set()
        for pf in parquet_files:
            try:
                pf_handle = pq.ParquetFile(pf)
                # Read first and last timestamps
                first_time = pf_handle.read_row_group(0, columns=['time']).to_pandas()['time'].iloc[0]
                last_time = pf_handle.read_row_group(pf_handle.num_row_groups - 1, columns=['time']).to_pandas()['time'].iloc[-1]

                # Add all dates in this range
                if self.granularity == 'daily':
                    current_date = first_time.date()
                    while current_date <= last_time.date():
                        dates_in_parquet.add(current_date.strftime('%Y-%m-%d'))
                        current_date += timedelta(days=1)
                else:  # monthly
                    current_date = first_time.date().replace(day=1)
                    while current_date <= last_time.date():
                        dates_in_parquet.add(current_date.strftime('%Y-%m'))
                        if current_date.month == 12:
                            current_date = current_date.replace(year=current_date.year + 1, month=1)
                        else:
                            current_date = current_date.replace(month=current_date.month + 1)
            except Exception as e:
                self.logger.warning(f"⚠️  Could not verify {pf.name}: {e}")

        # Compare with progress tracking
        missing_from_parquet = []
        for date_str in processed_dates:
            if date_str not in dates_in_parquet:
                missing_from_parquet.append(date_str)

        if missing_from_parquet:
            self.logger.warning(f"⚠️  Found {len(missing_from_parquet)} dates marked as processed but not in Parquet files:")
            for date_str in missing_from_parquet[:10]:
                self.logger.warning(f"    - {date_str}")
            if len(missing_from_parquet) > 10:
                self.logger.warning(f"    ... and {len(missing_from_parquet) - 10} more")

            # Clean up progress tracking
            self.logger.info("🧹 Cleaning up progress tracking...")
            for date_str in missing_from_parquet:
                self.progress['processed'].remove(date_str)
            self.save_progress()
            self.logger.info(f"✅ Removed {len(missing_from_parquet)} invalid entries from progress tracking")
        else:
            self.logger.info(f"✅ Data integrity verified: All {len(processed_dates)} processed dates are in Parquet files")
            
    def verify_all_downloads(self, expected_dates: List[datetime]) -> bool:
        """
        Final verification of all downloaded files with hash validation
        
        Returns:
            True if all files are valid, False otherwise
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(" 🔍 Final Download Verification ")
        self.logger.info("="*60)
        
        all_valid = True
        verified_count = 0
        missing_count = 0
        invalid_count = 0
        
        for date in expected_dates:
            if self.granularity == "daily":
                file_suffix = f"{date.year}-{date.month:02d}-{date.day:02d}"
            else:
                file_suffix = f"{date.year}-{date.month:02d}"
                
            zip_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip"
            checksum_file = self.raw_dir / f"{self.symbol}-trades-{file_suffix}.zip.CHECKSUM"
            
            # Check if files exist
            if not zip_file.exists():
                self.logger.warning(f"❌ Missing ZIP: {zip_file.name}")
                missing_count += 1
                all_valid = False
                continue
                
            if not checksum_file.exists():
                self.logger.warning(f"❌ Missing checksum: {checksum_file.name}")
                missing_count += 1
                all_valid = False
                continue
                
            # Verify checksum
            if self.verify_checksum(zip_file, checksum_file):
                verified_count += 1
                self.logger.info(f"✅ Verified: {zip_file.name}")
            else:
                self.logger.error(f"❌ Invalid checksum: {zip_file.name}")
                invalid_count += 1
                all_valid = False
                
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(" 📊 Verification Summary ")
        self.logger.info("="*60)
        self.logger.info(f"Total expected files: {len(expected_dates)}")
        self.logger.info(f"✅ Verified with valid checksums: {verified_count}")
        self.logger.info(f"❌ Missing files: {missing_count}")
        self.logger.info(f"❌ Invalid checksums: {invalid_count}")
        
        if all_valid:
            self.logger.info("\n✅ All downloads verified successfully!")
        else:
            self.logger.error("\n❌ Download verification failed!")
            self.logger.info("💡 Run the download again to retry failed files")
            
        return all_valid

    def verify_optimized_integrity(self, cleanup_old_files: bool = False):
        """Verify integrity of optimized parquet files and optionally cleanup old files"""
        self.logger.info("\n🔍 Verifying optimized parquet files integrity...")

        # Find optimized directory
        optimized_dir = self.base_dir / f"dataset-raw-{self.granularity}-compressed-optimized" / (f"futures-{self.futures_type}" if self.data_type == "futures" else "spot")

        if not optimized_dir.exists():
            self.logger.error(f"❌ No optimized directory found at: {optimized_dir}")
            return False

        # Find all optimized parquet files
        optimized_files = sorted(optimized_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))

        if not optimized_files:
            self.logger.error("❌ No optimized parquet files found")
            return False

        self.logger.info(f"📊 Found {len(optimized_files)} optimized parquet files")

        # Collect all dates from optimized files
        all_dates = set()
        file_date_ranges = []
        total_rows = 0
        total_size_gb = 0

        for opt_file in optimized_files:
            try:
                pf = pq.ParquetFile(opt_file)
                file_size_gb = opt_file.stat().st_size / (1024**3)
                total_size_gb += file_size_gb

                # Get metadata
                num_rows = pf.metadata.num_rows
                total_rows += num_rows

                # Read first and last timestamps
                first_time = pf.read_row_group(0, columns=['time']).to_pandas()['time'].iloc[0]
                last_time = pf.read_row_group(pf.num_row_groups - 1, columns=['time']).to_pandas()['time'].iloc[-1]

                file_date_ranges.append({
                    'file': opt_file.name,
                    'first_date': first_time,
                    'last_date': last_time,
                    'rows': num_rows,
                    'size_gb': file_size_gb
                })

                # Collect all dates in this file
                current_date = first_time.date()
                while current_date <= last_time.date():
                    all_dates.add(current_date)
                    current_date += timedelta(days=1)

                self.logger.info(f"  ✅ {opt_file.name}: {first_time.strftime('%Y-%m-%d')} to {last_time.strftime('%Y-%m-%d')} ({num_rows:,} rows, {file_size_gb:.2f} GB)")

            except Exception as e:
                self.logger.error(f"  ❌ Error reading {opt_file.name}: {e}")
                return False

        # Check for missing dates
        sorted_dates = sorted(all_dates)
        if sorted_dates:
            expected_start = sorted_dates[0]
            expected_end = sorted_dates[-1]

            # Generate all expected dates
            expected_dates = set()
            current = expected_start
            while current <= expected_end:
                expected_dates.add(current)
                current += timedelta(days=1)

            # Find missing dates
            missing_dates = expected_dates - all_dates

            # Filter out weekends and known holidays
            missing_trading_days = []
            for date in missing_dates:
                # Skip weekends
                if date.weekday() in [5, 6]:  # Saturday, Sunday
                    continue
                # Add to missing trading days
                missing_trading_days.append(date)

            if missing_trading_days:
                self.logger.warning(f"\n⚠️  Found {len(missing_trading_days)} missing trading days:")
                for date in sorted(missing_trading_days)[:10]:
                    self.logger.warning(f"    - {date.strftime('%Y-%m-%d (%A)')}")
                if len(missing_trading_days) > 10:
                    self.logger.warning(f"    ... and {len(missing_trading_days) - 10} more")
            else:
                self.logger.info(f"\n✅ All trading days from {expected_start} to {expected_end} are included")

        # Summary
        self.logger.info(f"\n📊 Optimized Dataset Summary:")
        self.logger.info(f"  - Total files: {len(optimized_files)}")
        self.logger.info(f"  - Total size: {total_size_gb:.2f} GB")
        self.logger.info(f"  - Total rows: {total_rows:,}")
        self.logger.info(f"  - Date range: {sorted_dates[0]} to {sorted_dates[-1]}")
        self.logger.info(f"  - Total days: {len(all_dates):,}")

        # Cleanup old files if requested
        if cleanup_old_files:
            self.logger.info("\n🧹 Cleaning up old non-optimized parquet files...")

            # Find old compressed files
            old_files = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))

            if old_files:
                total_old_size = sum(f.stat().st_size for f in old_files) / (1024**3)
                self.logger.info(f"Found {len(old_files)} old parquet files ({total_old_size:.2f} GB)")

                confirm = input(f"\nAre you sure you want to delete {len(old_files)} old parquet files? This will free up {total_old_size:.2f} GB. (yes/no): ").strip().lower()

                if confirm == 'yes':
                    deleted_count = 0
                    for old_file in old_files:
                        try:
                            old_file.unlink()
                            deleted_count += 1
                        except Exception as e:
                            self.logger.error(f"❌ Could not delete {old_file.name}: {e}")

                    self.logger.info(f"✅ Deleted {deleted_count} old parquet files, freed {total_old_size:.2f} GB")
                else:
                    self.logger.info("Cleanup cancelled")
            else:
                self.logger.info("No old parquet files found to cleanup")

        return True

    def optimize_parquet_files(self):
        """Optimize Parquet files by combining them into larger files"""
        try:
            from optimize_parquet_files import ParquetOptimizer

            source_dir = self.compressed_dir
            target_dir = self.base_dir / f"dataset-raw-{self.granularity}-compressed-optimized" / (f"futures-{self.futures_type}" if self.data_type == "futures" else "spot")

            self.logger.info(f"\n🚀 Starting Parquet optimization...")
            self.logger.info(f"Source: {source_dir.absolute()}")
            self.logger.info(f"Target: {target_dir.absolute()}")

            optimizer = ParquetOptimizer(source_dir, target_dir, max_size_gb=10)
            optimizer.optimize_parquet_files()

        except ImportError:
            self.logger.error("❌ Could not import ParquetOptimizer. Make sure optimize_parquet_files.py is in the same directory.")
        except Exception as e:
            self.logger.error(f"❌ Error during optimization: {e}")

    def update_dataset(self) -> None:
        """Update dataset with missing days since last download"""
        # Find the latest date in existing parquet files
        parquet_files = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))

        if not parquet_files:
            self.logger.error("No existing parquet files found. Use download command instead.")
            return

        # Read the last parquet file to get the latest date
        last_parquet = parquet_files[-1]
        df = pd.read_parquet(last_parquet, columns=['time'])
        last_date = df['time'].max()

        self.logger.info(f"Last date in dataset: {last_date}")

        # Download from the next day to today
        start_date = last_date + timedelta(days=1)
        end_date = datetime.now()

        if start_date > end_date:
            self.logger.info("Dataset is already up to date!")
            return

        self.download_date_range(start_date, end_date)

    def update_optimized_dataset(self) -> None:
        """Update optimized dataset with missing days since last timestamp"""
        # Check for optimized dataset in monthly directory (where it was created)
        optimized_dir = self.base_dir / "dataset-raw-monthly-compressed-optimized" / (f"futures-{self.futures_type}" if self.data_type == "futures" else "spot")

        self.logger.info(f"🔍 Looking for optimized dataset at: {optimized_dir}")
        self.logger.info(f"📁 Base directory: {self.base_dir.absolute()}")
        self.logger.info(f"📍 Absolute path: {optimized_dir.absolute()}")
        self.logger.info(f"🗂️  Directory exists: {optimized_dir.exists()}")

        if not optimized_dir.exists():
            self.logger.error("❌ No optimized dataset found. Please run optimization first.")
            self.logger.info(f"💡 Expected location: {optimized_dir.absolute()}")
            return

        # Find all optimized parquet files
        optimized_files = sorted(optimized_dir.glob("*-Optimized-*.parquet"))

        if not optimized_files:
            self.logger.error("No optimized parquet files found in the optimized directory.")
            return

        # Find the latest timestamp across all optimized files
        latest_timestamp = None
        self.logger.info("🔍 Finding latest timestamp in optimized dataset...")
        self.logger.info(f"📁 Checking directory: {optimized_dir}")
        self.logger.info(f"📊 Found {len(optimized_files)} optimized files")

        for file_path in optimized_files:
            try:
                # Read last row group to get the latest timestamp
                pf = pq.ParquetFile(file_path)
                last_rg = pf.read_row_group(pf.num_row_groups - 1, columns=['time'])
                file_latest = last_rg.to_pandas()['time'].max()

                if latest_timestamp is None or file_latest > latest_timestamp:
                    latest_timestamp = file_latest

                self.logger.info(f"  {file_path.name}: Latest date {file_latest.strftime('%Y-%m-%d %H:%M:%S')}")
            except Exception as e:
                self.logger.warning(f"⚠️  Could not read {file_path.name}: {e}")

        if latest_timestamp is None:
            self.logger.error("Could not determine latest timestamp from optimized files.")
            return

        self.logger.info(f"\n📅 Latest timestamp in optimized dataset: {latest_timestamp}")

        # Calculate date range for update
        start_date = latest_timestamp.date() + timedelta(days=1)
        # Exclude current day since it's not complete yet
        end_date = datetime.now().date() - timedelta(days=1)

        if start_date > end_date:
            self.logger.info("✅ Optimized dataset is already up to date!")
            return

        # Calculate number of days to download
        days_to_download = (end_date - start_date).days + 1
        self.logger.info(f"📥 Need to download {days_to_download} days of data from {start_date} to {end_date}")

        # Force daily granularity for updates
        original_granularity = self.granularity
        self.granularity = "daily"

        # Update directory paths for daily data
        if self.data_type == "spot":
            self.raw_dir = self.base_dir / "dataset-raw-daily" / "spot"
            self.compressed_dir = self.base_dir / "dataset-raw-daily-compressed" / "spot"
        else:  # futures
            self.raw_dir = self.base_dir / "dataset-raw-daily" / f"futures-{self.futures_type}"
            self.compressed_dir = self.base_dir / "dataset-raw-daily-compressed" / f"futures-{self.futures_type}"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.compressed_dir.mkdir(parents=True, exist_ok=True)

        # Download the missing daily data
        self.logger.info("\n🚀 Starting daily data download...")
        self.download_date_range(datetime.combine(start_date, datetime.min.time()),
                               datetime.combine(end_date, datetime.min.time()))

        # Find new parquet files that were created
        new_parquet_files = []
        for date in pd.date_range(start_date, end_date):
            # Check if parquet file exists for this date
            potential_files = self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet")
            for pf in potential_files:
                try:
                    # Quick check if this parquet contains the date
                    pf_handle = pq.ParquetFile(pf)
                    first_time = pf_handle.read_row_group(0, columns=['time']).to_pandas()['time'].iloc[0]
                    last_time = pf_handle.read_row_group(pf_handle.num_row_groups - 1, columns=['time']).to_pandas()['time'].iloc[-1]

                    if first_time.date() <= date.date() <= last_time.date():
                        if pf not in new_parquet_files:
                            new_parquet_files.append(pf)
                except:
                    continue

        if not new_parquet_files:
            self.logger.info("No new data was downloaded.")
            self.granularity = original_granularity
            return

        self.logger.info(f"\n📦 Found {len(new_parquet_files)} new parquet files to add to optimized dataset")

        # Add new data to optimized dataset
        self.logger.info("\n🔄 Adding new data to optimized dataset...")

        try:
            # Find the last optimized file (to append to it)
            last_optimized = sorted(optimized_files)[-1]
            last_number = int(last_optimized.stem.split('-')[-1])

            # Get current size of last optimized file
            current_size = last_optimized.stat().st_size
            max_size_bytes = 10 * 1024**3  # 10GB

            # Define schema for all cases
            schema = pa.schema([
                ('trade_id', pa.int64()),
                ('price', pa.float64()),
                ('qty', pa.float64()),
                ('quoteQty', pa.float64()),
                ('time', pa.timestamp('ns')),
                ('isBuyerMaker', pa.bool_()),
                ('isBestMatch', pa.float64())
            ])

            # Open writer for the last file if it's not full
            if current_size < max_size_bytes * 0.95:  # Leave 5% buffer
                self.logger.info(f"📝 Appending to {last_optimized.name} (current size: {current_size / (1024**3):.2f} GB)")
                # Read existing data and create new file with combined data
                existing_table = pq.read_table(last_optimized)
                writer = pq.ParquetWriter(last_optimized, schema, compression='snappy')
                writer.write_table(existing_table)
                current_output = last_optimized
            else:
                # Create new optimized file
                last_number += 1
                current_output = optimized_dir / f"{self.symbol}-Trades-Optimized-{last_number:03d}.parquet"
                self.logger.info(f"📝 Creating new optimized file: {current_output.name}")
                writer = pq.ParquetWriter(current_output, schema, compression='snappy')
                current_size = 0

            # Process each new parquet file
            for idx, new_file in enumerate(new_parquet_files, 1):
                self.logger.info(f"Processing {idx}/{len(new_parquet_files)}: {new_file.name}")

                # Read the new data
                table = pq.read_table(new_file)

                # Ensure schema compatibility - add missing columns if needed
                if 'isBestMatch' not in table.column_names:
                    # Add isBestMatch column with default values (0.0)
                    num_rows = table.num_rows
                    isBestMatch_col = pa.array([0.0] * num_rows, type=pa.float64())
                    table = table.append_column('isBestMatch', isBestMatch_col)
                    self.logger.info(f"  Added missing isBestMatch column with default values")
                else:
                    # Check if isBestMatch needs type conversion (bool to float64)
                    current_type = table.schema.field('isBestMatch').type
                    if current_type != pa.float64():
                        # Convert boolean to float64 (True->1.0, False->0.0)
                        if current_type == pa.bool_():
                            bool_column = table.column('isBestMatch')
                            float_column = pa.compute.cast(bool_column, pa.float64())
                            table = table.set_column(table.schema.get_field_index('isBestMatch'), 'isBestMatch', float_column)
                            self.logger.info(f"  Converted isBestMatch from bool to float64")
                        else:
                            # Handle other potential type mismatches
                            column = table.column('isBestMatch')
                            float_column = pa.compute.cast(column, pa.float64())
                            table = table.set_column(table.schema.get_field_index('isBestMatch'), 'isBestMatch', float_column)
                            self.logger.info(f"  Converted isBestMatch from {current_type} to float64")

                # Reorder columns to match expected schema
                expected_columns = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
                table = table.select(expected_columns)

                # Check if we need to create a new optimized file
                table_size = new_file.stat().st_size  # Use file size as estimate
                if current_size + table_size > max_size_bytes:
                    writer.close()
                    self.logger.info(f"✅ Completed: {current_output.name} ({current_output.stat().st_size / (1024**3):.2f} GB)")

                    # Create new file
                    last_number += 1
                    current_output = optimized_dir / f"{self.symbol}-Trades-Optimized-{last_number:03d}.parquet"
                    writer = pq.ParquetWriter(current_output, schema, compression='snappy')
                    current_size = 0
                    self.logger.info(f"📝 Creating new optimized file: {current_output.name}")

                # Write data
                writer.write_table(table)
                current_size += table_size

            # Close final writer
            writer.close()
            self.logger.info(f"✅ Completed: {current_output.name} ({current_output.stat().st_size / (1024**3):.2f} GB)")

            self.logger.info("\n✅ Successfully updated optimized dataset!")

        except Exception as e:
            self.logger.error(f"❌ Error updating optimized dataset: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore original granularity
            self.granularity = original_granularity


def interactive_menu():
    """Interactive menu for selecting download options"""
    print("\n" + "="*60)
    print(" Binance Data Downloader - Interactive Mode ")
    print("="*60)

    # Get symbol first
    print("\nEnter trading pair symbol:")
    print("1. BTCUSDT (default)")
    print("2. Other symbol")

    symbol_choice = input("\nEnter your choice (1-2) or press Enter for BTCUSDT: ").strip()

    if symbol_choice == "2":
        symbol = input("Enter symbol: ").strip().upper()
        if not symbol:
            symbol = "BTCUSDT"
    else:
        symbol = "BTCUSDT"

    # Select data type
    print("\nSelect data type:")
    print("1. Spot")
    print("2. Futures USD-M (USDT-margined)")
    print("3. Futures COIN-M (Coin-margined)")

    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        if choice == "1":
            data_type = "spot"
            futures_type = None
            break
        elif choice == "2":
            data_type = "futures"
            futures_type = "um"
            break
        elif choice == "3":
            data_type = "futures"
            futures_type = "cm"
            # For COIN-M futures, adjust symbol if needed
            if symbol == "BTCUSDT":
                symbol = "BTCUSD_PERP"
                print(f"Note: For COIN-M futures, using symbol: {symbol}")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Select action
    print("\nSelect action:")
    print("1. Download new data")
    print("2. Optimize existing Parquet files")
    print("3. Update optimized dataset (add new daily data)")
    print("4. Verify optimized dataset integrity (check all days included)")

    while True:
        action = input("\nEnter your choice (1-4): ").strip()
        if action in ["1", "2", "3", "4"]:
            if action == "1":
                action = "download"
            elif action == "2":
                action = "optimize"
            elif action == "3":
                action = "update_optimized"
            else:
                action = "verify_integrity"
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")

    # Select granularity only if not update_optimized or verify_integrity
    if action in ["update_optimized", "verify_integrity"]:
        # For update_optimized, always use daily
        # For verify_integrity, we need to ask since it depends on which dataset to verify
        if action == "update_optimized":
            granularity = "daily"
        else:  # verify_integrity
            print("\nSelect dataset granularity to verify:")
            print("1. Daily")
            print("2. Monthly")

            while True:
                gran_choice = input("\nEnter your choice (1-2): ").strip()
                if gran_choice == "1":
                    granularity = "daily"
                    break
                elif gran_choice == "2":
                    granularity = "monthly"
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
    else:
        print("\nSelect data granularity:")
        print("1. Daily")
        print("2. Monthly")

        while True:
            gran_choice = input("\nEnter your choice (1-2): ").strip()
            if gran_choice == "1":
                granularity = "daily"
                break
            elif gran_choice == "2":
                granularity = "monthly"
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")


    # Get dates if downloading
    if action == "download":
        print("\nEnter date range:")

        # Provide helpful date suggestions based on data type
        print("\n📅 Available data ranges:")
        if data_type == "spot":
            print("  - Spot data: Available from 2017-08-17")
            if granularity == "daily":
                date_format = "YYYY-MM-DD"
                example_start = "2024-01-01"
                example_end = "2024-01-31"
            else:
                date_format = "YYYY-MM"
                example_start = "2024-01"
                example_end = "2024-12"
        else:  # futures
            if futures_type == "um":
                print("  - Futures USD-M: Available from 2019-09-08")
                if granularity == "daily":
                    date_format = "YYYY-MM-DD"
                    example_start = "2019-09-08"
                    example_end = "2019-09-30"
                else:
                    date_format = "YYYY-MM"
                    example_start = "2019-09"
                    example_end = "2019-12"
            else:  # cm
                print("  - Futures COIN-M: Available from 2020-02-10")
                if granularity == "daily":
                    date_format = "YYYY-MM-DD"
                    example_start = "2020-02-10"
                    example_end = "2020-02-29"
                else:
                    date_format = "YYYY-MM"
                    example_start = "2020-02"
                    example_end = "2020-12"

        while True:
            start_str = input(f"Start date ({date_format}, e.g., {example_start}): ").strip()
            try:
                if granularity == "daily":
                    start_date = datetime.strptime(start_str, '%Y-%m-%d')
                else:
                    start_date = datetime.strptime(start_str, '%Y-%m')
                break
            except ValueError:
                print(f"Invalid date format. Please use {date_format}")

        while True:
            end_str = input(f"End date ({date_format}, e.g., {example_end}): ").strip()
            try:
                if granularity == "daily":
                    end_date = datetime.strptime(end_str, '%Y-%m-%d')
                else:
                    end_date = datetime.strptime(end_str, '%Y-%m')
                break
            except ValueError:
                print(f"Invalid date format. Please use {date_format}")

        # Get number of workers
        workers_str = input("\nNumber of concurrent downloads (default: 5): ").strip()
        max_workers = int(workers_str) if workers_str.isdigit() else 5
    elif action in ["optimize", "update_optimized", "verify_integrity"]:
        # For optimization, update_optimized, and verify_integrity, we don't need date ranges or workers
        start_str = end_str = "N/A"
        max_workers = 1

    # Confirm settings
    print("\n" + "="*60)
    print(" Summary of your selections:")
    print("="*60)
    print(f"Data Type: {data_type.upper()}{f' ({futures_type.upper()})' if futures_type else ''}")
    print(f"Action: {action.upper()}")
    print(f"Granularity: {granularity.upper()}")
    print(f"Symbol: {symbol}")
    if action == "download":
        print(f"Date Range: {start_str} to {end_str}")
        print(f"Concurrent Downloads: {max_workers}")
    elif action == "optimize":
        print("Target: Combine existing Parquet files into 10GB files")
    elif action == "update_optimized":
        print("Target: Update optimized dataset with new daily data")
    elif action == "verify_integrity":
        print("Target: Verify all days are included in optimized dataset")
        print("Option: Clean up old non-optimized files after verification")
    print("="*60)

    if action == "optimize":
        confirm = input("\nProceed with optimization? (y/n): ").strip().lower()
        action_name = "optimization"
    elif action == "update_optimized":
        confirm = input("\nProceed with updating optimized dataset? (y/n): ").strip().lower()
        action_name = "update"
    elif action == "verify_integrity":
        confirm = input("\nProceed with integrity verification? (y/n): ").strip().lower()
        action_name = "verification"
    else:
        confirm = input("\nProceed with download? (y/n): ").strip().lower()
        action_name = "download"

    if confirm != 'y':
        print(f"{action_name.capitalize()} cancelled.")
        return

    # Initialize downloader
    downloader = BinanceDataDownloader(
        symbol=symbol,
        data_type=data_type,
        futures_type=futures_type,
        granularity=granularity,
        stop_on_error=True
    )

    # Execute action
    if action == "download":
        downloader.download_date_range(start_date, end_date, max_workers=max_workers)
    elif action == "optimize":
        downloader.optimize_parquet_files()
    elif action == "update_optimized":
        downloader.update_optimized_dataset()
    else:  # verify_integrity
        # Ask about cleanup option
        cleanup = input("\nDo you want to delete old non-optimized parquet files after verification? (y/n): ").strip().lower()
        downloader.verify_optimized_integrity(cleanup_old_files=(cleanup == 'y'))

    print("\n✅ Operation completed!")


def main():
    # Check if running with command-line arguments
    if len(sys.argv) > 1:
        # Legacy command-line mode
        parser = argparse.ArgumentParser(
            description="Download and process Binance trading data",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Interactive mode (recommended)
  python main.py

  # Command-line mode for automation
  python main.py download --symbol BTCUSDT --type spot --granularity daily \\
                         --start 2024-01-01 --end 2024-01-31
            """
        )

        subparsers = parser.add_subparsers(dest='command', help='Commands')

        # Download command
        download_parser = subparsers.add_parser('download', help='Download data for date range')
        download_parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
        download_parser.add_argument('--type', choices=['spot', 'futures'], required=True,
                                   help='Data type: spot or futures')
        download_parser.add_argument('--futures-type', choices=['um', 'cm'], default='um',
                                   help='Futures type: um (USD-M) or cm (COIN-M)')
        download_parser.add_argument('--granularity', choices=['daily', 'monthly'], required=True,
                                   help='Data granularity: daily or monthly')
        download_parser.add_argument('--start', required=True,
                                   help='Start date (YYYY-MM-DD for daily, YYYY-MM for monthly)')
        download_parser.add_argument('--end', required=True,
                                   help='End date (YYYY-MM-DD for daily, YYYY-MM for monthly)')
        download_parser.add_argument('--workers', type=int, default=5,
                                   help='Number of concurrent downloads (default: 5)')

        # Update command
        update_parser = subparsers.add_parser('update', help='Update existing dataset')
        update_parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
        update_parser.add_argument('--type', choices=['spot', 'futures'], required=True,
                                 help='Data type: spot or futures')
        update_parser.add_argument('--futures-type', choices=['um', 'cm'], default='um',
                                 help='Futures type: um (USD-M) or cm (COIN-M)')
        update_parser.add_argument('--granularity', choices=['daily', 'monthly'], required=True,
                                 help='Data granularity: daily or monthly')

        args = parser.parse_args()

        if not args.command:
            parser.print_help()
            sys.exit(1)

        # Initialize downloader
        futures_type = args.futures_type if args.type == 'futures' else None
        downloader = BinanceDataDownloader(
            symbol=args.symbol,
            data_type=args.type,
            futures_type=futures_type,
            granularity=args.granularity,
            stop_on_error=True
        )

        if args.command == 'download':
            # Parse dates based on granularity
            if args.granularity == 'daily':
                start_date = datetime.strptime(args.start, '%Y-%m-%d')
                end_date = datetime.strptime(args.end, '%Y-%m-%d')
            else:  # monthly
                start_date = datetime.strptime(args.start, '%Y-%m')
                end_date = datetime.strptime(args.end, '%Y-%m')

            downloader.download_date_range(start_date, end_date, max_workers=args.workers)

        elif args.command == 'update':
            downloader.update_dataset()
    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()