"""
CSV Extractor - Extract and verify CSV files from downloaded ZIPs
Handles extraction, verification, and cleanup of trading data
"""

import sys
import zipfile
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import json
from datetime import datetime, timedelta
from loguru import logger
import calendar

class CSVExtractor:
    def __init__(self, symbol: str = "BTCUSDT", data_type: str = "spot",
                 futures_type: str = "um", granularity: str = "monthly",
                 base_dir: Path = Path(".")):
        """
        Initialize CSV Extractor
        
        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT)
            data_type: 'spot' or 'futures'
            futures_type: 'um' (USD-M) or 'cm' (COIN-M) - only used if data_type is 'futures'
            granularity: 'daily' or 'monthly'
            base_dir: Base directory for data storage
        """
        self.symbol = symbol
        self.data_type = data_type
        self.futures_type = futures_type if data_type == "futures" else None
        self.granularity = granularity
        
        # Ensure base_dir points to data folder
        if base_dir == Path("."):
            self.base_dir = Path.cwd() / "data"
        else:
            self.base_dir = base_dir
            
        # Set up directories
        if self.data_type == "spot":
            self.raw_dir = self.base_dir / f"dataset-raw-{granularity}" / "spot"
        else:
            self.raw_dir = self.base_dir / f"dataset-raw-{granularity}" / f"futures-{futures_type}"
            
        # Progress tracking
        self.progress_file = self.base_dir / f"extraction_progress_{symbol}_{data_type}_{granularity}.json"
        self.progress = self.load_progress()
        
        # Configure logging
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging"""
        log_file = self.base_dir / "logs" / f"csv_extraction_{self.symbol}_{self.data_type}_{self.granularity}.log"
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
            "extracted": [],
            "verified": [],
            "failed": []
        }
        
    def save_progress(self):
        """Save progress to tracking file"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)
            
    def calculate_file_hash(self, file_path: Path, algorithm: str = "sha256") -> str:
        """Calculate file hash"""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def detect_missing_days(self, csv_path: Path) -> Optional[Dict]:
        """
        Detect missing days in a monthly CSV file
        
        Returns:
            Dict with missing days info or None if complete
        """
        try:
            # Extract date from filename
            filename_parts = csv_path.stem.split('-trades-')
            if len(filename_parts) != 2:
                return None
                
            date_str = filename_parts[1]  # e.g., "2023-03"
            
            if self.granularity != "monthly":
                return None
                
            # Parse year and month
            try:
                year, month = map(int, date_str.split('-'))
            except:
                return None
                
            # Get expected days in month
            days_in_month = calendar.monthrange(year, month)[1]
            expected_start = datetime(year, month, 1)
            expected_end = datetime(year, month, days_in_month, 23, 59, 59)
            
            # Read CSV to get actual date range
            import pandas as pd
            COLUMN_NAMES = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
            
            # Check if file has header
            with open(csv_path, 'r') as f:
                first_line = f.readline().strip()
            has_header = any(keyword in first_line.lower() for keyword in ['time', 'price', 'qty', 'id'])
            
            # Read first timestamp
            if has_header:
                df_head = pd.read_csv(csv_path, nrows=1)
                # Rename columns if needed
                column_mapping = {
                    'id': 'trade_id',
                    'price': 'price',
                    'qty': 'qty',
                    'quote_qty': 'quoteQty',
                    'time': 'time',
                    'is_buyer_maker': 'isBuyerMaker',
                    'is_best_match': 'isBestMatch'
                }
                df_head.rename(columns=column_mapping, inplace=True)
            else:
                df_head = pd.read_csv(csv_path, names=COLUMN_NAMES, header=None, nrows=1)
            
            # Read last row efficiently
            with open(csv_path, 'rb') as f:
                f.seek(0, 2)
                file_size = f.tell()
                chunk_size = min(1024 * 1024, file_size)
                f.seek(max(0, file_size - chunk_size))
                tail_data = f.read()
                lines = tail_data.decode('utf-8', errors='ignore').strip().split('\n')
                last_line = [line for line in lines if line.strip()][-1]
                
            # Parse last row
            import io
            if has_header:
                df_tail = pd.read_csv(io.StringIO(last_line), names=list(df_head.columns), header=None)
            else:
                df_tail = pd.read_csv(io.StringIO(last_line), names=COLUMN_NAMES, header=None)
            
            # Convert timestamps
            first_time = df_head['time'].iloc[0]
            last_time = df_tail['time'].iloc[0]
            
            # Detect timestamp format
            time_str = str(int(first_time))
            if len(time_str) == 16:  # Microseconds
                first_dt = pd.to_datetime(first_time, unit='us')
                last_dt = pd.to_datetime(last_time, unit='us')
            elif len(time_str) == 13:  # Milliseconds
                first_dt = pd.to_datetime(first_time, unit='ms')
                last_dt = pd.to_datetime(last_time, unit='ms')
            elif len(time_str) == 10:  # Seconds
                first_dt = pd.to_datetime(first_time, unit='s')
                last_dt = pd.to_datetime(last_time, unit='s')
            else:
                return None
                
            # Check for missing periods
            missing_info = {
                'year': year,
                'month': month,
                'expected_start': expected_start,
                'expected_end': expected_end,
                'actual_start': first_dt,
                'actual_end': last_dt,
                'days_covered': (last_dt - first_dt).days + 1,
                'expected_days': days_in_month,
                'missing_periods': []
            }
            
            # Check for missing start
            if first_dt.date() > expected_start.date():
                missing_info['missing_periods'].append({
                    'type': 'start',
                    'from': expected_start,
                    'to': first_dt - timedelta(seconds=1),
                    'days': (first_dt.date() - expected_start.date()).days
                })
                
            # Check for missing end
            if last_dt.date() < expected_end.date():
                missing_info['missing_periods'].append({
                    'type': 'end',
                    'from': last_dt + timedelta(seconds=1),
                    'to': expected_end,
                    'days': (expected_end.date() - last_dt.date()).days
                })
                
            # Only return if there are missing periods
            if missing_info['missing_periods']:
                return missing_info
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error detecting missing days in {csv_path.name}: {e}")
            return None
        
    def verify_zip_integrity(self, zip_path: Path) -> bool:
        """Verify ZIP file integrity"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Test the ZIP file
                result = zf.testzip()
                if result is not None:
                    self.logger.error(f"❌ ZIP file {zip_path.name} is corrupted: {result}")
                    return False
                return True
        except Exception as e:
            self.logger.error(f"❌ Failed to verify ZIP {zip_path.name}: {e}")
            return False
            
    def extract_single_zip(self, zip_path: Path) -> Optional[Path]:
        """Extract a single ZIP file and return CSV path"""
        try:
            # Verify ZIP integrity first
            if not self.verify_zip_integrity(zip_path):
                return None
                
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # Get the CSV filename(s)
                all_csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                
                # Filter to get only root-level CSV files (ignore subdirectories)
                root_csv_files = [f for f in all_csv_files if '/' not in f]
                
                # If no root files, try to find the main CSV in subdirectories
                if not root_csv_files and all_csv_files:
                    # Look for the expected filename pattern in any subdirectory
                    expected_pattern = f"{self.symbol}-trades-"
                    for csv_file in all_csv_files:
                        if expected_pattern in csv_file.lower():
                            root_csv_files = [csv_file]
                            break
                
                csv_files = root_csv_files
                
                if len(csv_files) == 0:
                    self.logger.error(f"❌ No CSV files found in {zip_path.name}")
                    return None
                elif len(all_csv_files) > 1 and len(csv_files) == 1:
                    # Found duplicates in subdirectories, but need to check if they're truly duplicates
                    self.logger.warning(f"⚠️ Found {len(all_csv_files)} CSV files in ZIP. Files: {all_csv_files}")
                    
                    # Extract all files to check their content
                    temp_dir = self.raw_dir / f"temp_{zip_path.stem}"
                    temp_dir.mkdir(exist_ok=True)
                    
                    file_info = []
                    for csv_file in all_csv_files:
                        zf.extract(csv_file, temp_dir)
                        csv_path = temp_dir / csv_file
                        
                        try:
                            # Quick check of file size and row count
                            import pandas as pd
                            file_size = csv_path.stat().st_size
                            df_sample = pd.read_csv(csv_path, nrows=100)
                            total_rows = sum(1 for _ in open(csv_path)) - (1 if len(df_sample.columns) > 7 else 0)
                            
                            file_info.append({
                                'path': csv_file,
                                'size': file_size,
                                'rows': total_rows,
                                'full_path': csv_path
                            })
                            
                            self.logger.info(f"  📊 {csv_file}: {file_size/(1024*1024):.1f} MB, ~{total_rows:,} rows")
                        except Exception as e:
                            self.logger.error(f"  ❌ Error checking {csv_file}: {e}")
                    
                    # Decide which file(s) to use
                    if len(file_info) == 2:
                        # Check if files are similar size (likely duplicates)
                        size_diff = abs(file_info[0]['size'] - file_info[1]['size']) / max(file_info[0]['size'], file_info[1]['size'])
                        
                        if size_diff < 0.1:  # Less than 10% difference
                            # Likely duplicates, use the one in root or with simpler path
                            root_file = next((f for f in file_info if '/' not in f['path']), file_info[0])
                            self.logger.info(f"📂 Files appear to be duplicates (size diff: {size_diff*100:.1f}%), using: {root_file['path']}")
                            
                            # Copy the chosen file
                            import shutil
                            expected_name = zip_path.stem + ".csv"
                            expected_path = self.raw_dir / expected_name
                            shutil.copy2(root_file['full_path'], expected_path)
                        else:
                            # Files are different sizes, need to merge
                            self.logger.warning(f"📊 Files have different sizes (diff: {size_diff*100:.1f}%), merging required")
                            
                            # Read and merge both files
                            all_dfs = []
                            COLUMN_NAMES = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
                            
                            for f_info in file_info:
                                try:
                                    # Check if file has header
                                    with open(f_info['full_path'], 'r') as f:
                                        first_line = f.readline().strip()
                                    has_header_merge = any(keyword in first_line.lower() for keyword in ['time', 'price', 'qty', 'id'])
                                    
                                    if has_header_merge:
                                        df = pd.read_csv(f_info['full_path'])
                                        # Rename columns to match expected names
                                        column_mapping = {
                                            'id': 'trade_id',
                                            'price': 'price', 
                                            'qty': 'qty',
                                            'quote_qty': 'quoteQty',
                                            'time': 'time',
                                            'is_buyer_maker': 'isBuyerMaker',
                                            'is_best_match': 'isBestMatch'
                                        }
                                        df.rename(columns=column_mapping, inplace=True)
                                    else:
                                        df = pd.read_csv(f_info['full_path'], names=COLUMN_NAMES, header=None)
                                    all_dfs.append(df)
                                    self.logger.info(f"  ✅ Read {len(df):,} rows from {f_info['path']}")
                                except Exception as e:
                                    self.logger.error(f"  ❌ Failed to read {f_info['path']}: {e}")
                            
                            if all_dfs:
                                # Merge dataframes
                                merged_df = pd.concat(all_dfs, ignore_index=True)
                                self.logger.info(f"📊 Total rows before deduplication: {len(merged_df):,}")
                                
                                # Remove duplicates
                                if 'trade_id' in merged_df.columns:
                                    merged_df = merged_df.drop_duplicates(subset=['trade_id'], keep='first')
                                else:
                                    merged_df = merged_df.drop_duplicates(keep='first')
                                
                                # Sort by time
                                if 'time' in merged_df.columns:
                                    merged_df = merged_df.sort_values('time')
                                
                                self.logger.info(f"📊 Total rows after deduplication: {len(merged_df):,}")
                                
                                # Save merged file
                                expected_name = zip_path.stem + ".csv"
                                expected_path = self.raw_dir / expected_name
                                merged_df.to_csv(expected_path, index=False, header=False)
                                self.logger.info(f"💾 Saved merged CSV: {expected_path.name}")
                    
                    # Cleanup temp directory
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
                    expected_path = self.raw_dir / (zip_path.stem + ".csv")
                    return expected_path if expected_path.exists() else None
                elif len(csv_files) > 1:
                    # Multiple root-level CSV files (shouldn't happen with Binance data)
                    self.logger.warning(f"⚠️ Found {len(csv_files)} root-level CSV files in {zip_path.name}: {csv_files}")
                    # Use the first one that matches the expected pattern
                    expected_pattern = f"{self.symbol}-trades-"
                    matching_file = None
                    for csv_file in csv_files:
                        if expected_pattern in csv_file.lower():
                            matching_file = csv_file
                            break
                    
                    if matching_file:
                        csv_files = [matching_file]
                        self.logger.info(f"📂 Using matching file: {matching_file}")
                    else:
                        # Use the first file as fallback
                        csv_files = [csv_files[0]]
                        self.logger.warning(f"⚠️ No matching pattern found, using first file: {csv_files[0]}")
                    
                # Now we have exactly one CSV file to extract
                csv_filename = csv_files[0]
                
                # Extract to raw directory
                zf.extract(csv_filename, self.raw_dir)
                csv_path = self.raw_dir / csv_filename
                
                # If the file was in a subdirectory, move it to the root
                if '/' in csv_filename:
                    # File is in a subdirectory, move to root
                    expected_name = zip_path.stem + ".csv"
                    expected_path = self.raw_dir / expected_name
                    
                    # Move file from subdirectory to root
                    import shutil
                    shutil.move(str(csv_path), str(expected_path))
                    
                    # Clean up empty subdirectories
                    subdirs = csv_filename.split('/')[:-1]
                    if subdirs:
                        subdir_path = self.raw_dir / '/'.join(subdirs)
                        try:
                            shutil.rmtree(subdir_path)
                        except:
                            pass
                    
                    csv_path = expected_path
                else:
                    # File is already in root, just rename if needed
                    expected_name = zip_path.stem + ".csv"
                    expected_path = self.raw_dir / expected_name
                    
                    if csv_path.name != expected_name:
                        csv_path.rename(expected_path)
                        csv_path = expected_path
                    
                self.logger.info(f"✅ Extracted: {csv_path.name}")
                return csv_path
                
        except Exception as e:
            self.logger.error(f"❌ Failed to extract {zip_path.name}: {e}")
            return None
            
    def verify_csv_integrity(self, csv_path: Path) -> bool:
        """Verify CSV file integrity with enhanced validation"""
        try:
            # Check file exists and has size
            if not csv_path.exists():
                self.logger.error(f"❌ CSV file does not exist: {csv_path.name}")
                return False
                
            file_size = csv_path.stat().st_size
            if file_size == 0:
                self.logger.error(f"❌ CSV file is empty: {csv_path.name}")
                return False
                
            # Enhanced validation with pandas
            import pandas as pd
            import numpy as np
            from datetime import datetime
            
            COLUMN_NAMES = ['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch']
            
            # Initialize has_header variable
            has_header = False
            
            # Try reading the CSV
            try:
                # First, check if the file has a header by reading the first line
                with open(csv_path, 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if first line contains header keywords
                has_header = any(keyword in first_line.lower() for keyword in ['time', 'price', 'qty', 'id'])
                
                if has_header:
                    # Read with header
                    df_head = pd.read_csv(csv_path, nrows=10000)
                    # Rename columns to match expected names if needed
                    column_mapping = {
                        'id': 'trade_id',
                        'price': 'price',
                        'qty': 'qty',
                        'quote_qty': 'quoteQty',
                        'time': 'time',
                        'is_buyer_maker': 'isBuyerMaker',
                        'is_best_match': 'isBestMatch'
                    }
                    df_head.rename(columns=column_mapping, inplace=True)
                    
                    # Add missing columns with default values if they don't exist
                    if 'isBestMatch' not in df_head.columns:
                        df_head['isBestMatch'] = True
                else:
                    # Try without header (original Binance format)
                    df_head = pd.read_csv(csv_path, names=COLUMN_NAMES, header=None, nrows=10000)
                
                # Also read the last rows to get accurate time range
                # Count total lines first (excluding potential header)
                with open(csv_path, 'rb') as f:
                    # Move to end of file
                    f.seek(0, 2)
                    file_size = f.tell()
                    
                    # Read last ~1MB to get tail data
                    chunk_size = min(1024 * 1024, file_size)
                    f.seek(max(0, file_size - chunk_size))
                    tail_data = f.read()
                    
                    # Find last complete lines
                    lines = tail_data.decode('utf-8', errors='ignore').strip().split('\n')
                    # Get last 1000 complete lines
                    tail_lines = [line for line in lines if line.strip()][-1000:]
                
                # Parse tail data
                import io
                try:
                    if has_header:
                        # Parse with awareness that these are data rows, not header
                        df_tail = pd.read_csv(io.StringIO('\n'.join(tail_lines)), 
                                             names=list(df_head.columns), header=None)
                    else:
                        df_tail = pd.read_csv(io.StringIO('\n'.join(tail_lines)), 
                                             names=COLUMN_NAMES, header=None)
                except:
                    # If tail parsing fails, use head data
                    df_tail = df_head
                
                # Use head for validation, but merge with tail for time range
                df = df_head
                
            except Exception as e:
                # Try with header
                try:
                    df = pd.read_csv(csv_path, nrows=10000)
                    df_tail = df  # Use same data if reading fails
                except:
                    self.logger.error(f"❌ Failed to read CSV {csv_path.name}: {e}")
                    return False
                
            if len(df) == 0:
                self.logger.error(f"❌ CSV file has no data: {csv_path.name}")
                return False
                
            # Validate timestamp column
            if 'time' in df.columns:
                # Check if timestamps are valid
                sample_time = df['time'].iloc[0]
                
                # Handle different timestamp formats
                try:
                    # Check if it's already a datetime string
                    if isinstance(sample_time, str):
                        # Try parsing as datetime string
                        first_dt = pd.to_datetime(df['time'].iloc[0])
                        # Use tail data for accurate last timestamp
                        last_dt = pd.to_datetime(df_tail['time'].iloc[-1]) if 'df_tail' in locals() and len(df_tail) > 0 else pd.to_datetime(df['time'].iloc[-1])
                    elif isinstance(sample_time, (int, float, np.integer, np.floating)):
                        # Handle numpy types and regular int/float
                        # Check timestamp format by digit count
                        time_str = str(int(sample_time))
                        if len(time_str) == 16:
                            # Microseconds (new format from 2025)
                            first_dt = pd.to_datetime(df['time'].iloc[0], unit='us')
                            # Use tail data for accurate last timestamp
                            last_dt = pd.to_datetime(df_tail['time'].iloc[-1], unit='us') if 'df_tail' in locals() and len(df_tail) > 0 else pd.to_datetime(df['time'].iloc[-1], unit='us')
                        elif len(time_str) == 13:
                            # Milliseconds
                            first_dt = pd.to_datetime(df['time'].iloc[0], unit='ms')
                            # Use tail data for accurate last timestamp
                            last_dt = pd.to_datetime(df_tail['time'].iloc[-1], unit='ms') if 'df_tail' in locals() and len(df_tail) > 0 else pd.to_datetime(df['time'].iloc[-1], unit='ms')
                        elif len(time_str) == 10:
                            # Seconds
                            first_dt = pd.to_datetime(df['time'].iloc[0], unit='s')
                            # Use tail data for accurate last timestamp
                            last_dt = pd.to_datetime(df_tail['time'].iloc[-1], unit='s') if 'df_tail' in locals() and len(df_tail) > 0 else pd.to_datetime(df['time'].iloc[-1], unit='s')
                        else:
                            # Unknown numeric format
                            self.logger.error(f"❌ Unknown timestamp format in {csv_path.name}: {sample_time} (length: {len(time_str)})")
                            return False
                    else:
                        self.logger.error(f"❌ Invalid timestamp type in {csv_path.name}: {type(sample_time)}")
                        return False
                    
                    # Check if data spans the full expected period
                    data_duration = (last_dt - first_dt).total_seconds() / 3600  # hours
                    
                    if self.granularity == "monthly":
                        if data_duration < 1:  # Less than 1 hour for monthly data
                            self.logger.warning(f"⚠️ Data severely truncated - only {data_duration:.1f} hours of data for monthly file")
                            self.logger.warning(f"⚠️ This may indicate the file needs to be merged with others in the same ZIP")
                            # Don't fail validation - let the merge process handle it
                        elif data_duration < 24:  # Less than 1 day
                            self.logger.warning(f"⚠️ Data appears incomplete - only {data_duration:.1f} hours of data for monthly file")
                        elif data_duration < 168:  # Less than 1 week
                            self.logger.info(f"ℹ️ Partial monthly data - {data_duration:.1f} hours ({data_duration/24:.1f} days)")
                    elif self.granularity == "daily":
                        if data_duration < 1:  # Less than 1 hour for daily data
                            self.logger.warning(f"⚠️ Data appears truncated - only {data_duration:.1f} hours of data")
                    
                    self.logger.info(f"📅 Time range: {first_dt} to {last_dt} UTC ({data_duration:.1f} hours)")
                    
                    # Extract date from filename for validation
                    filename_parts = csv_path.stem.split('-trades-')
                    if len(filename_parts) == 2:
                        date_str = filename_parts[1]  # e.g., "2021-05"
                        
                        # Check if data matches the expected date
                        if self.granularity == "monthly":
                            expected_year_month = date_str  # YYYY-MM format
                            actual_year_month = first_dt.strftime('%Y-%m')
                            
                            if actual_year_month != expected_year_month:
                                self.logger.warning(f"⚠️ Date mismatch: filename says {expected_year_month}, data starts at {actual_year_month}")
                            
                            # Skip checking for missing days based on sample
                            # The sample only contains first 10k rows which may not represent all days
                            # The time range check above is sufficient to validate data completeness
                            
                            # Calculate actual days covered based on time range
                            days_covered = (last_dt - first_dt).days + 1
                            if days_covered > 25:
                                self.logger.info(f"✅ Good coverage for {expected_year_month}: ~{days_covered} days of data")
                            elif days_covered > 15:
                                self.logger.info(f"ℹ️ Partial coverage for {expected_year_month}: ~{days_covered} days of data")
                            else:
                                self.logger.warning(f"⚠️ Limited coverage for {expected_year_month}: only ~{days_covered} days of data")
                                
                            # Check for missing days
                            missing_info = self.detect_missing_days(csv_path)
                            if missing_info:
                                self.logger.warning(f"🔍 Missing data detected for {expected_year_month}:")
                                for period in missing_info['missing_periods']:
                                    self.logger.warning(f"   - Missing {period['type']}: {period['days']} days from {period['from'].strftime('%Y-%m-%d')} to {period['to'].strftime('%Y-%m-%d')}")
                                    
                                # Mark file as having missing data
                                if date_str not in self.progress.get('missing_data', {}):
                                    if 'missing_data' not in self.progress:
                                        self.progress['missing_data'] = {}
                                    # Convert datetime objects to strings for JSON serialization
                                    missing_info_serializable = {
                                        'year': missing_info['year'],
                                        'month': missing_info['month'],
                                        'expected_start': missing_info['expected_start'].isoformat(),
                                        'expected_end': missing_info['expected_end'].isoformat(),
                                        'actual_start': missing_info['actual_start'].isoformat(),
                                        'actual_end': missing_info['actual_end'].isoformat(),
                                        'days_covered': missing_info['days_covered'],
                                        'expected_days': missing_info['expected_days'],
                                        'missing_periods': [
                                            {
                                                'type': p['type'],
                                                'from': p['from'].isoformat(),
                                                'to': p['to'].isoformat(),
                                                'days': p['days']
                                            }
                                            for p in missing_info['missing_periods']
                                        ]
                                    }
                                    self.progress['missing_data'][date_str] = missing_info_serializable
                                    self.save_progress()
                                
                        elif self.granularity == "daily":
                            expected_date = date_str  # YYYY-MM-DD format
                            actual_date = first_dt.strftime('%Y-%m-%d')
                            
                            if actual_date != expected_date:
                                self.logger.warning(f"⚠️ Date mismatch: filename says {expected_date}, data is for {actual_date}")
                                    
                except Exception as e:
                    self.logger.error(f"❌ Invalid timestamps in {csv_path.name}: {e}")
                    return False
            else:
                self.logger.error(f"❌ No 'time' column found in {csv_path.name}")
                return False
                
            # Additional data quality checks
            if 'price' in df.columns:
                if (df['price'] <= 0).any():
                    self.logger.warning(f"⚠️ Found non-positive prices in {csv_path.name}")
                    
            if 'qty' in df.columns:
                if (df['qty'] <= 0).any():
                    self.logger.warning(f"⚠️ Found non-positive quantities in {csv_path.name}")
                    
            self.logger.info(f"✅ Verified CSV integrity: {csv_path.name} ({file_size / (1024*1024):.1f} MB, {len(df)} rows sampled)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to verify CSV {csv_path.name}: {e}")
            return False
            
    def find_all_zip_files(self) -> List[Path]:
        """Find all ZIP files in raw directory"""
        zip_files = sorted(self.raw_dir.glob(f"{self.symbol}-trades-*.zip"))
        return zip_files
        
    def find_all_csv_files(self) -> List[Path]:
        """Find all CSV files in raw directory"""
        csv_files = sorted(self.raw_dir.glob(f"{self.symbol}-trades-*.csv"))
        return csv_files
        
    def extract_and_verify_all(self, retry_failed: bool = True, force_reextract: bool = False) -> Tuple[int, int]:
        """
        Extract all ZIP files and verify CSV files
        
        Returns:
            Tuple of (successful_count, failed_count)
        """
        self.logger.info("\n" + "="*60)
        self.logger.info(" 📦 CSV Extraction and Verification ")
        self.logger.info("="*60)
        
        # Find all ZIP files
        zip_files = self.find_all_zip_files()
        self.logger.info(f"Found {len(zip_files)} ZIP files")
        
        # Find existing CSV files
        existing_csv = self.find_all_csv_files()
        self.logger.info(f"Found {len(existing_csv)} existing CSV files")
        
        # Determine which ZIPs need extraction
        if force_reextract:
            # Force re-extraction of all ZIP files
            zips_to_extract = zip_files.copy()
            self.logger.info("🔄 Force re-extraction enabled - will re-extract all ZIP files")
            
            # Remove existing CSV files if force re-extract
            for csv_file in existing_csv:
                try:
                    csv_file.unlink()
                    self.logger.info(f"🗑️ Removed existing CSV: {csv_file.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not remove {csv_file.name}: {e}")
        else:
            # Normal mode - only extract if CSV doesn't exist
            extracted_stems = {csv.stem.replace('-trades-', '-trades-') for csv in existing_csv}
            zips_to_extract = []
            
            for zip_file in zip_files:
                zip_stem = zip_file.stem
                if zip_stem not in extracted_stems:
                    zips_to_extract.append(zip_file)
                else:
                    # Check if already marked as extracted
                    date_str = zip_stem.split('-trades-')[-1]
                    if date_str not in self.progress.get('extracted', []):
                        self.progress['extracted'].append(date_str)
                    
        self.logger.info(f"Need to extract: {len(zips_to_extract)} ZIP files")
        
        # Extract missing ZIPs
        successful = 0
        failed = 0
        
        for i, zip_file in enumerate(zips_to_extract, 1):
            self.logger.info(f"\n[{i}/{len(zips_to_extract)}] Processing {zip_file.name}...")
            
            # Extract ZIP
            csv_path = self.extract_single_zip(zip_file)
            
            if csv_path:
                # Verify CSV
                if self.verify_csv_integrity(csv_path):
                    successful += 1
                    date_str = zip_file.stem.split('-trades-')[-1]
                    if date_str not in self.progress['extracted']:
                        self.progress['extracted'].append(date_str)
                    if date_str not in self.progress['verified']:
                        self.progress['verified'].append(date_str)
                    self.save_progress()
                else:
                    failed += 1
                    # Remove corrupted CSV
                    csv_path.unlink()
                    self.logger.warning(f"⚠️  Removed corrupted CSV: {csv_path.name}")
            else:
                failed += 1
                date_str = zip_file.stem.split('-trades-')[-1]
                if date_str not in self.progress['failed']:
                    self.progress['failed'].append(date_str)
                self.save_progress()
                
        # Verify all existing CSV files
        self.logger.info("\n🔍 Verifying all CSV files...")
        
        all_csv_files = self.find_all_csv_files()
        verified_count = 0
        corrupted_count = 0
        
        for csv_file in all_csv_files:
            if self.verify_csv_integrity(csv_file):
                verified_count += 1
                date_str = csv_file.stem.split('-trades-')[-1]
                if date_str not in self.progress['verified']:
                    self.progress['verified'].append(date_str)
            else:
                corrupted_count += 1
                self.logger.warning(f"⚠️  Found corrupted CSV: {csv_file.name}")
                
        self.save_progress()
        
        # Summary
        self.logger.info("\n" + "="*60)
        self.logger.info(" 📊 Extraction Summary ")
        self.logger.info("="*60)
        self.logger.info(f"ZIP files found: {len(zip_files)}")
        self.logger.info(f"✅ Successfully extracted: {successful}")
        self.logger.info(f"❌ Failed extractions: {failed}")
        self.logger.info(f"✅ Verified CSV files: {verified_count}")
        self.logger.info(f"❌ Corrupted CSV files: {corrupted_count}")
        
        # Check if all expected files are present
        expected_count = len(zip_files)
        actual_count = len(all_csv_files)
        
        if actual_count < expected_count:
            missing_count = expected_count - actual_count
            self.logger.warning(f"\n⚠️  Missing {missing_count} CSV files!")
            
            if retry_failed and self.progress.get('failed'):
                self.logger.info("\n🔄 Retrying failed extractions...")
                # Clear failed list and retry
                failed_dates = self.progress['failed'].copy()
                self.progress['failed'] = []
                self.save_progress()
                
                # This would trigger re-download of failed files
                return successful, failed + missing_count
        else:
            self.logger.info(f"\n✅ All expected CSV files are present!")
            
        # Check for files with missing data
        if self.progress.get('missing_data'):
            self.logger.warning(f"\n⚠️ Found {len(self.progress['missing_data'])} files with incomplete data")
            
            # Show details of missing data
            for date_str, info in self.progress['missing_data'].items():
                # Convert ISO strings back to datetime for display
                expected_start = datetime.fromisoformat(info['expected_start']).strftime('%Y-%m-%d')
                expected_end = datetime.fromisoformat(info['expected_end']).strftime('%Y-%m-%d')
                actual_start = datetime.fromisoformat(info['actual_start']).strftime('%Y-%m-%d')
                actual_end = datetime.fromisoformat(info['actual_end']).strftime('%Y-%m-%d')
                
                self.logger.warning(f"\n📅 {date_str}:")
                self.logger.warning(f"   - Expected: {expected_start} to {expected_end}")
                self.logger.warning(f"   - Actual: {actual_start} to {actual_end}")
                self.logger.warning(f"   - Missing: {info['expected_days'] - info['days_covered']} days")
            
            # Ask user if they want to re-download
            response = input("\n🔄 Would you like to re-download files with missing data? (yes/no): ").strip().lower()
            if response == 'yes':
                if self.handle_missing_data_redownload():
                    # Re-run extraction for newly downloaded files
                    self.logger.info("\n🔄 Re-running extraction for newly downloaded files...")
                    new_successful, new_failed = self.extract_and_verify_all(retry_failed=False, force_reextract=False)
                    successful += new_successful
                    failed = max(0, failed - new_successful)  # Adjust failed count
                    
        return successful, failed
    
    def request_missing_data_download(self) -> List[str]:
        """
        Request re-download of files with missing data
        
        Returns:
            List of date strings that need re-downloading
        """
        if 'missing_data' not in self.progress or not self.progress['missing_data']:
            self.logger.info("✅ No missing data detected")
            return []
            
        missing_dates = list(self.progress['missing_data'].keys())
        self.logger.warning(f"\n⚠️ Found {len(missing_dates)} files with missing data:")
        
        for date_str, info in self.progress['missing_data'].items():
            # Convert ISO strings back to datetime for display
            expected_start = datetime.fromisoformat(info['expected_start']).strftime('%Y-%m-%d')
            expected_end = datetime.fromisoformat(info['expected_end']).strftime('%Y-%m-%d')
            actual_start = datetime.fromisoformat(info['actual_start']).strftime('%Y-%m-%d')
            actual_end = datetime.fromisoformat(info['actual_end']).strftime('%Y-%m-%d')
            
            self.logger.warning(f"\n📅 {date_str}:")
            self.logger.warning(f"   - Expected: {expected_start} to {expected_end}")
            self.logger.warning(f"   - Actual: {actual_start} to {actual_end}")
            self.logger.warning(f"   - Coverage: {info['days_covered']}/{info['expected_days']} days")
            
            for period in info['missing_periods']:
                self.logger.warning(f"   - Missing {period['type']}: {period['days']} days")
                
        return missing_dates
    
    def handle_missing_data_redownload(self) -> bool:
        """
        Handle re-download of files with missing data
        
        Returns:
            True if re-download was successful
        """
        missing_dates = self.request_missing_data_download()
        
        if not missing_dates:
            return True
            
        self.logger.info(f"\n🔄 Preparing to re-download {len(missing_dates)} files with missing data")
        
        # Import downloader
        try:
            from ..downloaders.binance_downloader import BinanceDataDownloader
            
            # Create downloader instance with same parameters
            downloader = BinanceDataDownloader(
                symbol=self.symbol,
                data_type=self.data_type,
                futures_type=self.futures_type,
                granularity=self.granularity,
                base_dir=self.base_dir.parent  # Go up one level from raw_dir
            )
            
            # Remove existing files for re-download
            for date_str in missing_dates:
                # Remove CSV file
                csv_pattern = f"{self.symbol}-trades-{date_str}.csv"
                csv_files = list(self.raw_dir.glob(csv_pattern))
                for csv_file in csv_files:
                    try:
                        csv_file.unlink()
                        self.logger.info(f"🗑️ Removed incomplete CSV: {csv_file.name}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to remove {csv_file.name}: {e}")
                        
                # Remove ZIP file to force re-download
                zip_pattern = f"{self.symbol}-trades-{date_str}.zip*"
                zip_files = list(self.raw_dir.glob(zip_pattern))
                for zip_file in zip_files:
                    try:
                        zip_file.unlink()
                        self.logger.info(f"🗑️ Removed ZIP for re-download: {zip_file.name}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to remove {zip_file.name}: {e}")
                        
                # Clear from progress tracking
                if date_str in self.progress.get('extracted', []):
                    self.progress['extracted'].remove(date_str)
                if date_str in self.progress.get('verified', []):
                    self.progress['verified'].remove(date_str)
                if date_str in self.progress.get('missing_data', {}):
                    del self.progress['missing_data'][date_str]
                    
            self.save_progress()
            
            # Parse dates for downloader
            dates_to_download = []
            for date_str in missing_dates:
                try:
                    if self.granularity == "monthly":
                        # Parse YYYY-MM format
                        year, month = map(int, date_str.split('-'))
                        dates_to_download.append((year, month))
                    elif self.granularity == "daily":
                        # Parse YYYY-MM-DD format
                        dates_to_download.append(datetime.strptime(date_str, '%Y-%m-%d').date())
                except:
                    self.logger.error(f"❌ Failed to parse date: {date_str}")
                    
            if not dates_to_download:
                return False
                
            # Download missing data
            self.logger.info(f"\n📥 Starting re-download of {len(dates_to_download)} files...")
            
            # Sort dates for proper download order
            dates_to_download.sort()
            
            # Call downloader
            if self.granularity == "monthly":
                # Convert to date ranges for monthly
                if dates_to_download:
                    start_year, start_month = dates_to_download[0]
                    end_year, end_month = dates_to_download[-1]
                    
                    success, failed = downloader.download_monthly_data(
                        f"{start_year}-{start_month:02d}",
                        f"{end_year}-{end_month:02d}",
                        specific_months=dates_to_download
                    )
            else:
                # Daily download
                success, failed = downloader.download_daily_data(
                    dates_to_download[0],
                    dates_to_download[-1],
                    specific_dates=dates_to_download
                )
                
            if failed == 0:
                self.logger.info(f"✅ Successfully re-downloaded all missing data")
                return True
            else:
                self.logger.error(f"❌ Failed to re-download {failed} files")
                return False
                
        except ImportError:
            self.logger.error("❌ Cannot import BinanceDataDownloader - manual re-download required")
            self.logger.info("\nTo re-download missing data manually, run:")
            for date_str in missing_dates:
                self.logger.info(f"python main.py download --start {date_str} --end {date_str} --symbol {self.symbol} --type {self.data_type} --granularity {self.granularity}")
            return False
        except Exception as e:
            self.logger.error(f"❌ Re-download failed: {e}")
            return False
        
    def cleanup_zip_files(self, force: bool = False):
        """Remove ZIP files after successful extraction"""
        if not force:
            response = input("\n⚠️  Remove all successfully extracted ZIP files? (yes/no): ").strip().lower()
            if response != 'yes':
                self.logger.info("Cleanup cancelled")
                return
                
        zip_files = self.find_all_zip_files()
        csv_files = {csv.stem for csv in self.find_all_csv_files()}
        
        removed_count = 0
        freed_space = 0
        
        for zip_file in zip_files:
            # Check if corresponding CSV exists and is verified
            csv_stem = zip_file.stem
            if csv_stem in csv_files:
                date_str = csv_stem.split('-trades-')[-1]
                if date_str in self.progress.get('verified', []):
                    try:
                        file_size = zip_file.stat().st_size
                        zip_file.unlink()
                        removed_count += 1
                        freed_space += file_size
                        self.logger.info(f"🗑️  Removed: {zip_file.name}")
                    except Exception as e:
                        self.logger.error(f"❌ Failed to remove {zip_file.name}: {e}")
                        
        self.logger.info(f"\n✅ Cleanup complete: Removed {removed_count} ZIP files, freed {freed_space / (1024**3):.2f} GB")
        

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and verify CSV files from Binance ZIP downloads")
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--type', choices=['spot', 'futures'], default='spot', help='Data type')
    parser.add_argument('--futures-type', choices=['um', 'cm'], default='um', help='Futures type')
    parser.add_argument('--granularity', choices=['daily', 'monthly'], default='monthly', help='Data granularity')
    parser.add_argument('--cleanup', action='store_true', help='Remove ZIP files after extraction')
    parser.add_argument('--base-dir', type=Path, default=Path('.'), help='Base directory')
    
    args = parser.parse_args()
    
    extractor = CSVExtractor(
        symbol=args.symbol,
        data_type=args.type,
        futures_type=args.futures_type,
        granularity=args.granularity,
        base_dir=args.base_dir
    )
    
    # Extract and verify
    successful, failed = extractor.extract_and_verify_all()
    
    # Optional cleanup
    if args.cleanup and successful > 0:
        extractor.cleanup_zip_files()
        
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())