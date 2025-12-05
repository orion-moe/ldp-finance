"""
CSV to Parquet Converter - Convert CSV files to Parquet with proper naming
Preserves month-based naming convention and optimizes data types
"""

import sys
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import json
from datetime import datetime
from loguru import logger
import numpy as np

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

# Optimized data types
COLUMN_DTYPES = {
    'trade_id': 'int64',
    'price': 'float32',
    'qty': 'float32',
    'quoteQty': 'float32',
    'time': 'datetime64[ms]',
    'isBuyerMaker': 'bool',
    'isBestMatch': 'bool'
}

class CSVToParquetConverter:
    def __init__(self, symbol: str = "BTCUSDT", data_type: str = "spot",
                 futures_type: str = "um", granularity: str = "monthly",
                 base_dir: Path = Path("."), compression: str = "snappy"):
        """
        Initialize CSV to Parquet Converter

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            data_type: 'spot' or 'futures'
            futures_type: 'um' (USD-M) or 'cm' (COIN-M) - only used if data_type is 'futures'
            granularity: 'daily' or 'monthly'
            base_dir: Base directory for data storage
            compression: Compression algorithm ('snappy', 'zstd', 'lz4', 'brotli', 'gzip', 'none')
        """
        self.symbol = symbol
        self.data_type = data_type
        self.futures_type = futures_type if data_type == "futures" else None
        self.granularity = granularity
        self.compression = compression
        
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
        
        # Progress tracking
        self.progress_file = self.base_dir / f"conversion_progress_{symbol}_{data_type}_{granularity}.json"
        self.progress = self.load_progress()
        
        # Configure logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        log_file = self.base_dir / "logs" / f"csv_to_parquet_{self.symbol}_{self.data_type}_{self.granularity}.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.remove()
        logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        logger.add(log_file, rotation="500 MB", retention="10 days")
        
        self.logger = logger
        
    def load_progress(self) -> dict:
        """Load progress from tracking file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {
            "converted": [],
            "failed": []
        }
        
    def save_progress(self):
        """Save progress to tracking file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
            
    def detect_csv_format(self, csv_file: Path) -> Tuple[bool, Optional[Dict[int, str]]]:
        """Detect if CSV has header and return column mapping"""
        try:
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
                
            first_parts = first_line.split(',')
            
            # Check if first line contains headers
            has_text_headers = any(
                part.lower().strip() in ['id', 'trade_id', 'price', 'qty', 'quote_qty', 'time', 'is_buyer_maker', 'is_best_match']
                for part in first_parts
            )
            
            if has_text_headers:
                self.logger.info(f"📋 Detected CSV with header: {first_line}")
                return True, None
            else:
                self.logger.info(f"📋 Detected CSV without header, using standard format")
                return False, None
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not detect CSV format for {csv_file.name}: {e}")
            return False, None
            
    def convert_single_csv(self, csv_file: Path) -> Optional[Path]:
        """Convert a single CSV file to Parquet"""
        try:
            # Extract date from filename
            date_part = csv_file.stem.split('-trades-')[-1]
            output_file = self.compressed_dir / f"{self.symbol}-Trades-{date_part}.parquet"
            
            # Check if already converted
            if output_file.exists():
                self.logger.info(f"⏭️  Skipping {csv_file.name} - already converted")
                if date_part not in self.progress['converted']:
                    self.progress['converted'].append(date_part)
                    self.save_progress()
                return output_file
                
            self.logger.info(f"🔄 Converting {csv_file.name} to Parquet...")
            
            # Detect CSV format
            has_header, _ = self.detect_csv_format(csv_file)
            
            # Read CSV with optimized settings
            chunks = []
            chunk_size = 5_000_000  # 5M rows per chunk
            
            # Check how many columns the CSV has
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
                num_columns = len(first_line.split(','))
            
            # Define dtype based on whether we have headers
            column_names = COLUMN_NAMES  # Default
            if has_header:
                # For CSV with headers, we need to use the actual column names
                dtype_spec = {
                    'id': 'int64',
                    'price': 'float32',
                    'qty': 'float32',
                    'quote_qty': 'float32',
                    'is_buyer_maker': 'bool'
                }
            else:
                # For CSV without headers, use the expected column names
                # But only for columns that exist
                if num_columns == 7:
                    dtype_spec = {
                        'trade_id': 'int64',
                        'price': 'float32',
                        'qty': 'float32',
                        'quoteQty': 'float32',
                        'isBuyerMaker': 'bool',
                        'isBestMatch': 'bool'
                    }
                    column_names = COLUMN_NAMES
                else:
                    # Files with 6 columns don't have isBestMatch
                    dtype_spec = {
                        'trade_id': 'int64',
                        'price': 'float32',
                        'qty': 'float32',
                        'quoteQty': 'float32',
                        'isBuyerMaker': 'bool'
                    }
                    column_names = COLUMN_NAMES[:-1]  # Exclude last column
            
            for chunk_idx, chunk in enumerate(pd.read_csv(
                csv_file,
                names=None if has_header else (column_names if not has_header else COLUMN_NAMES),
                header=0 if has_header else None,
                chunksize=chunk_size,
                dtype=dtype_spec
            )):
                # Rename columns to standard names if we have headers
                if has_header:
                    column_mapping = {
                        'id': 'trade_id',
                        'quote_qty': 'quoteQty',
                        'is_buyer_maker': 'isBuyerMaker'
                    }
                    chunk = chunk.rename(columns=column_mapping)
                    
                    # Add missing columns with default values
                    if 'isBestMatch' not in chunk.columns:
                        chunk['isBestMatch'] = False  # Not available in futures data
                else:
                    # For files without headers, also check if isBestMatch is missing
                    if 'isBestMatch' not in chunk.columns:
                        chunk['isBestMatch'] = False  # Not available in futures data
                
                # Convert time column
                if 'time' in chunk.columns:
                    # Try different time formats
                    if chunk['time'].dtype == 'object' or chunk['time'].dtype == 'int64':
                        # First check if it's already in milliseconds
                        sample_value = chunk['time'].iloc[0]
                        
                        if isinstance(sample_value, (int, np.integer)) or (isinstance(sample_value, str) and sample_value.isdigit()):
                            sample_int = int(sample_value)

                            # Normalize all timestamps to milliseconds for consistency
                            # This prevents schema mismatches in parquet files
                            if len(str(sample_int)) >= 16:  # Microseconds (16 digits)
                                # Convert microseconds to milliseconds first
                                chunk['time'] = chunk['time'].astype('int64') // 1000
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='ms')
                            elif len(str(sample_int)) >= 13:  # Milliseconds (13 digits)
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='ms')
                            else:  # Seconds (10 digits)
                                # Convert seconds to milliseconds
                                chunk['time'] = chunk['time'].astype('int64') * 1000
                                chunk['time'] = pd.to_datetime(chunk['time'], unit='ms')
                        else:
                            # String format, use pandas auto-detection
                            chunk['time'] = pd.to_datetime(chunk['time'])
                            
                chunks.append(chunk)
                
                if chunk_idx % 5 == 0:
                    self.logger.info(f"  Processed {(chunk_idx + 1) * chunk_size:,} rows...")
                    
            # Combine all chunks
            df = pd.concat(chunks, ignore_index=True)
            self.logger.info(f"  Total rows: {len(df):,}")
            
            # Convert to PyArrow Table
            table = pa.Table.from_pandas(df, preserve_index=False)
            
            # Write to Parquet with compression
            pq.write_table(
                table,
                output_file,
                compression=self.compression,
                use_dictionary=True,
                row_group_size=100_000
            )
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            self.logger.info(f"✅ Created {output_file.name} ({file_size_mb:.1f} MB)")
            
            # Verify Parquet file integrity
            if self.verify_single_parquet(output_file, len(df)):
                # Update progress
                if date_part not in self.progress['converted']:
                    self.progress['converted'].append(date_part)
                    self.save_progress()
                return output_file
            else:
                # If verification fails, remove the corrupted Parquet file
                self.logger.error(f"❌ Parquet verification failed for {output_file.name}")
                if output_file.exists():
                    output_file.unlink()
                    self.logger.info(f"🗑️ Removed corrupted Parquet: {output_file.name}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Failed to convert {csv_file.name}: {e}")
            date_part = csv_file.stem.split('-trades-')[-1]
            if date_part not in self.progress['failed']:
                self.progress['failed'].append(date_part)
                self.save_progress()
            return None
            
    def convert_all_csv_files(self, cleanup_csv: bool = False) -> Tuple[int, int]:
        """
        Convert all CSV files to Parquet with automatic verification and cleanup
        
        Args:
            cleanup_csv: Deprecated - CSV cleanup now happens automatically after verification
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(" 🔄 CSV to Parquet Conversion with Auto-Cleanup ")
        self.logger.info("="*60)
        
        # Find all CSV files
        csv_files = sorted(self.raw_dir.glob(f"{self.symbol}-trades-*.csv"))
        self.logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Find existing Parquet files
        existing_parquet = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))
        self.logger.info(f"Found {len(existing_parquet)} existing Parquet files")
        
        successful = 0
        failed = 0
        total_freed_space = 0.0
        
        for i, csv_file in enumerate(csv_files, 1):
            self.logger.info(f"\n[{i}/{len(csv_files)}] Processing {csv_file.name}...")
            
            output_file = self.convert_single_csv(csv_file)
            
            if output_file:
                successful += 1
                
                # Always cleanup CSV after successful conversion and verification
                # This saves disk space since Parquet has been verified
                try:
                    csv_size_mb = csv_file.stat().st_size / (1024 * 1024)
                    csv_file.unlink()
                    total_freed_space += csv_size_mb
                    self.logger.info(f"🗑️ Removed CSV: {csv_file.name} (freed {csv_size_mb:.1f} MB)")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not remove CSV {csv_file.name}: {e}")
            else:
                failed += 1
                
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(" 📊 Conversion Summary ")
        self.logger.info("="*60)
        self.logger.info(f"Total CSV files: {len(csv_files)}")
        self.logger.info(f"✅ Successfully converted: {successful}")
        self.logger.info(f"❌ Failed conversions: {failed}")
        self.logger.info(f"📁 Total Parquet files: {len(list(self.compressed_dir.glob(f'{self.symbol}-Trades-*.parquet')))}")
        if total_freed_space > 0:
            if total_freed_space > 1024:
                self.logger.info(f"💾 Total space freed: {total_freed_space / 1024:.2f} GB")
            else:
                self.logger.info(f"💾 Total space freed: {total_freed_space:.1f} MB")
        
        return successful, failed
        
    def verify_single_parquet(self, parquet_file: Path, expected_rows: int = None) -> bool:
        """Verify a single Parquet file integrity"""
        try:
            # Try reading the file
            table = pq.read_table(parquet_file)
            num_rows = table.num_rows
            
            # Basic integrity checks
            if num_rows == 0:
                self.logger.error(f"❌ Parquet file {parquet_file.name} has no data")
                return False
                
            # Check if row count matches expected (if provided)
            if expected_rows is not None and abs(num_rows - expected_rows) > 100:
                self.logger.error(f"❌ Row count mismatch in {parquet_file.name}: expected ~{expected_rows:,}, got {num_rows:,}")
                return False
                
            # Try reading timestamp column (critical for time series data)
            if 'time' in table.column_names:
                time_col = table.column('time').to_pandas()
                min_date = time_col.min()
                max_date = time_col.max()
                
                # Check for null timestamps
                if time_col.isna().any():
                    self.logger.error(f"❌ Found null timestamps in {parquet_file.name}")
                    return False
                    
                self.logger.info(f"🔍 Verified {parquet_file.name}: {num_rows:,} rows, {min_date} to {max_date}")
            else:
                self.logger.info(f"🔍 Verified {parquet_file.name}: {num_rows:,} rows")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to verify Parquet file {parquet_file.name}: {e}")
            return False

    def verify_parquet_files(self) -> bool:
        """Verify all Parquet files are readable and have correct schema"""
        self.logger.info("\n🔍 Verifying Parquet files...")
        
        parquet_files = sorted(self.compressed_dir.glob(f"{self.symbol}-Trades-*.parquet"))
        all_valid = True
        
        for pf in parquet_files:
            if not self.verify_single_parquet(pf):
                all_valid = False
                
        return all_valid
        

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert Binance CSV files to Parquet format")
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--type', choices=['spot', 'futures'], default='spot', help='Data type')
    parser.add_argument('--futures-type', choices=['um', 'cm'], default='um', help='Futures type')
    parser.add_argument('--granularity', choices=['daily', 'monthly'], default='monthly', help='Data granularity')
    parser.add_argument('--cleanup', action='store_true', help='Remove CSV files after conversion')
    parser.add_argument('--verify', action='store_true', help='Verify Parquet files after conversion')
    parser.add_argument('--base-dir', type=Path, default=Path('.'), help='Base directory')
    
    args = parser.parse_args()
    
    converter = CSVToParquetConverter(
        symbol=args.symbol,
        data_type=args.type,
        futures_type=args.futures_type,
        granularity=args.granularity,
        base_dir=args.base_dir
    )
    
    # Convert all files
    successful, failed = converter.convert_all_csv_files(cleanup_csv=args.cleanup)
    
    # Optional verification
    if args.verify:
        converter.verify_parquet_files()
        
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())