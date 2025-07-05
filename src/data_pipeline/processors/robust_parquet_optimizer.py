#!/usr/bin/env python3
"""
Robust Parquet Optimizer with Data Corruption Prevention
Implements fail-safe mechanisms to prevent data corruption during optimization
"""

import os
import sys
import json
import shutil
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from validators.missing_dates_validator import MissingDatesValidator


@dataclass
class FileIntegrityInfo:
    """Information about a parquet file's integrity"""
    path: str
    size_bytes: int
    rows: int
    checksum: str
    min_time: int
    max_time: int
    price_range: tuple
    volume_sum: float
    is_corrupted: bool
    error_msg: Optional[str] = None


@dataclass
class OptimizationConfig:
    """Configuration for parquet optimization"""
    max_file_size_gb: int = 10
    row_group_size: int = 100_000
    compression: str = 'snappy'
    use_dictionary: bool = True
    verify_checksum: bool = True
    keep_backup: bool = True
    temp_dir: Optional[str] = None
    max_workers: int = 1  # Conservative parallelism


@njit
def calculate_checksum_array(data: np.ndarray) -> str:
    """Calculate checksum for numpy array using Numba"""
    return str(hash(data.tobytes()))


class RobustParquetOptimizer:
    """
    Robust Parquet file optimizer with comprehensive data integrity protection
    """
    
    def __init__(self, source_dir: str, target_dir: str, config: OptimizationConfig = None):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.config = config or OptimizationConfig()
        
        # Create directories
        self.target_dir.mkdir(parents=True, exist_ok=True)
        self.backup_dir = self.target_dir.parent / f"{self.target_dir.name}_backup"
        
        # Setup temp directory
        if self.config.temp_dir:
            self.temp_dir = Path(self.config.temp_dir)
        else:
            self.temp_dir = self.target_dir / "temp"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Track optimization state
        self.state_file = self.target_dir / "optimization_state.json"
        self.state = self.load_state()
        
        # Detect schema from source files (will be set in collect_source_files)
        self.standard_schema = None
        
        self.logger.info(f"Initialized RobustParquetOptimizer")
        self.logger.info(f"Source: {self.source_dir}")
        self.logger.info(f"Target: {self.target_dir}")
        self.logger.info(f"Config: {asdict(self.config)}")
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_file = self.target_dir / "robust_optimization.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Logger
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def load_state(self) -> dict:
        """Load optimization state from file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load state: {e}")
        
        return {
            'processed_files': [],
            'integrity_info': {},
            'optimization_start': None,
            'optimization_complete': False,
            'verified_files': []
        }
    
    def save_state(self):
        """Save optimization state to file"""
        self.state['last_update'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
    
    def calculate_file_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum for a file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def get_file_integrity_info(self, file_path: Path) -> FileIntegrityInfo:
        """Get comprehensive integrity information for a parquet file"""
        try:
            # Basic file info
            size_bytes = file_path.stat().st_size
            
            # Try to read parquet file
            pf = pq.ParquetFile(file_path)
            metadata = pf.metadata
            rows = metadata.num_rows
            
            # Sample data for integrity checks
            sample_table = pf.read(columns=['time', 'price', 'qty'])
            sample_df = sample_table.to_pandas()
            
            if len(sample_df) == 0:
                return FileIntegrityInfo(
                    path=str(file_path),
                    size_bytes=size_bytes,
                    rows=0,
                    checksum="",
                    min_time=0,
                    max_time=0,
                    price_range=(0, 0),
                    volume_sum=0.0,
                    is_corrupted=True,
                    error_msg="Empty file"
                )
            
            # Calculate checksum
            checksum = self.calculate_file_checksum(file_path)
            
            # Data integrity metrics
            min_time = int(sample_df['time'].min().timestamp() * 1000)
            max_time = int(sample_df['time'].max().timestamp() * 1000)
            price_min = float(sample_df['price'].min())
            price_max = float(sample_df['price'].max())
            volume_sum = float(sample_df['qty'].sum())
            
            return FileIntegrityInfo(
                path=str(file_path),
                size_bytes=size_bytes,
                rows=rows,
                checksum=checksum,
                min_time=min_time,
                max_time=max_time,
                price_range=(price_min, price_max),
                volume_sum=volume_sum,
                is_corrupted=False
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze {file_path}: {e}")
            return FileIntegrityInfo(
                path=str(file_path),
                size_bytes=file_path.stat().st_size if file_path.exists() else 0,
                rows=0,
                checksum="",
                min_time=0,
                max_time=0,
                price_range=(0, 0),
                volume_sum=0.0,
                is_corrupted=True,
                error_msg=str(e)
            )
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify file integrity with comprehensive checks"""
        try:
            # Basic existence check
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return False
            
            # Size check
            size = file_path.stat().st_size
            if size < 1000:  # Less than 1KB
                self.logger.error(f"File too small ({size} bytes): {file_path}")
                return False
            
            # Parquet structure check
            pf = pq.ParquetFile(file_path)
            metadata = pf.metadata
            
            if metadata.num_rows == 0:
                self.logger.error(f"File has no rows: {file_path}")
                return False
            
            # Schema validation
            schema = pf.schema
            required_columns = {'time', 'price', 'qty'}
            actual_columns = {field.name for field in schema}
            
            if not required_columns.issubset(actual_columns):
                missing = required_columns - actual_columns
                self.logger.error(f"Missing required columns {missing}: {file_path}")
                return False
            
            # Data sample test - read entire table and sample in pandas
            sample_table = pf.read(columns=['time', 'price', 'qty'])
            sample_df = sample_table.to_pandas()
            
            # Sample first 1000 rows if file is large
            if len(sample_df) > 1000:
                sample_df = sample_df.head(1000)
            
            # Check for null values in critical columns
            if sample_df[['time', 'price', 'qty']].isnull().any().any():
                self.logger.error(f"Null values found in critical columns: {file_path}")
                return False
            
            # Check for reasonable data ranges
            if (sample_df['price'] <= 0).any():
                self.logger.error(f"Invalid price values (≤0): {file_path}")
                return False
            
            if (sample_df['qty'] <= 0).any():
                self.logger.error(f"Invalid quantity values (≤0): {file_path}")
                return False
            
            self.logger.debug(f"File integrity verified: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Integrity verification failed for {file_path}: {e}")
            return False
    
    def safe_read_parquet(self, file_path: Path) -> Optional[pa.Table]:
        """Safely read parquet file with error handling"""
        try:
            # First verify integrity
            if not self.verify_file_integrity(file_path):
                return None
            
            # Read the file
            table = pq.read_table(file_path)
            
            # Additional validation
            if table.num_rows == 0:
                self.logger.error(f"Empty table: {file_path}")
                return None
            
            self.logger.debug(f"Successfully read {file_path}: {table.num_rows} rows")
            return table
            
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return None
    
    def safe_write_parquet(self, table: pa.Table, output_path: Path) -> bool:
        """Safely write parquet file with verification"""
        try:
            # Write to temporary file first
            temp_path = self.temp_dir / f"temp_{output_path.name}"
            
            # Write with robust settings (schema normalization is done before calling this)
            pq.write_table(
                table,
                temp_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size,
                use_dictionary=self.config.use_dictionary,
                write_statistics=True,
                use_deprecated_int96_timestamps=False
            )
            
            # Verify the written file
            if not self.verify_file_integrity(temp_path):
                self.logger.error(f"Written file failed integrity check: {temp_path}")
                temp_path.unlink()
                return False
            
            # Move to final location
            shutil.move(str(temp_path), str(output_path))
            
            # Final verification
            if not self.verify_file_integrity(output_path):
                self.logger.error(f"Final file failed integrity check: {output_path}")
                output_path.unlink()
                return False
            
            self.logger.info(f"Successfully written: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write {output_path}: {e}")
            # Cleanup temp file
            temp_path = self.temp_dir / f"temp_{output_path.name}"
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def collect_source_files(self) -> List[Path]:
        """Collect and validate source files"""
        self.logger.info("Collecting source files...")
        
        source_files = sorted(self.source_dir.glob("*.parquet"))
        
        if not source_files:
            self.logger.error("No parquet files found in source directory")
            return []
        
        self.logger.info(f"Found {len(source_files)} parquet files")
        
        # Verify all source files and detect schema
        valid_files = []
        for file_path in source_files:
            if self.verify_file_integrity(file_path):
                valid_files.append(file_path)
                self.logger.debug(f"Valid source file: {file_path.name}")
                
                # Set standard schema from first valid file
                if self.standard_schema is None:
                    try:
                        table = pq.read_table(file_path)
                        self.standard_schema = table.schema
                        self.logger.info(f"Detected standard schema from {file_path.name}:")
                        self.logger.info(f"  {self.standard_schema}")
                    except Exception as e:
                        self.logger.warning(f"Failed to read schema from {file_path.name}: {e}")
            else:
                self.logger.warning(f"Invalid source file: {file_path.name}")
        
        if self.standard_schema is None:
            self.logger.error("Could not detect schema from any valid file")
            return []
        
        self.logger.info(f"Valid source files: {len(valid_files)}")
        return valid_files
    
    def analyze_source_files(self, source_files: List[Path]) -> Dict[str, Any]:
        """Analyze source files for optimization planning"""
        self.logger.info("Analyzing source files...")
        
        analysis = {
            'total_files': len(source_files),
            'total_size': 0,
            'total_rows': 0,
            'time_range': (None, None),
            'integrity_info': {}
        }
        
        min_time = None
        max_time = None
        
        for file_path in source_files:
            integrity_info = self.get_file_integrity_info(file_path)
            analysis['integrity_info'][str(file_path)] = asdict(integrity_info)
            
            if not integrity_info.is_corrupted:
                analysis['total_size'] += integrity_info.size_bytes
                analysis['total_rows'] += integrity_info.rows
                
                if min_time is None or integrity_info.min_time < min_time:
                    min_time = integrity_info.min_time
                if max_time is None or integrity_info.max_time > max_time:
                    max_time = integrity_info.max_time
        
        analysis['time_range'] = (min_time, max_time)
        
        self.logger.info(f"Analysis complete:")
        self.logger.info(f"  Total size: {analysis['total_size'] / (1024**3):.2f} GB")
        self.logger.info(f"  Total rows: {analysis['total_rows']:,}")
        self.logger.info(f"  Time range: {min_time} to {max_time}")
        
        return analysis
    
    def optimize_files(self, source_files: List[Path]) -> bool:
        """Main optimization process with fail-safe mechanisms"""
        self.logger.info("Starting robust parquet optimization...")
        
        # Analyze source files
        analysis = self.analyze_source_files(source_files)
        
        # Calculate target file count
        max_size_bytes = self.config.max_file_size_gb * 1024**3
        target_files = max(1, analysis['total_size'] // max_size_bytes + 1)
        
        self.logger.info(f"Target files: {target_files}")
        
        # Group files for optimization
        current_batch = []
        current_size = 0
        output_file_count = 1
        
        # Keep track of successful operations
        successful_files = []
        
        try:
            for file_path in source_files:
                # Read file safely
                table = self.safe_read_parquet(file_path)
                if table is None:
                    self.logger.error(f"Skipping corrupted file: {file_path}")
                    continue
                
                file_size = file_path.stat().st_size
                
                # Check if we need to write current batch
                if current_batch and (current_size + file_size > max_size_bytes):
                    # Process current batch
                    if self.write_optimized_batch(current_batch, output_file_count):
                        successful_files.extend(current_batch)
                        output_file_count += 1
                    else:
                        self.logger.error(f"Failed to write batch {output_file_count}")
                        return False
                    
                    # Reset batch
                    current_batch = []
                    current_size = 0
                
                # Add to current batch
                current_batch.append((file_path, table))
                current_size += file_size
                
                self.logger.debug(f"Added to batch: {file_path.name}")
            
            # Process final batch
            if current_batch:
                if self.write_optimized_batch(current_batch, output_file_count):
                    successful_files.extend([fp for fp, _ in current_batch])
                else:
                    self.logger.error(f"Failed to write final batch {output_file_count}")
                    return False
            
            # Verify all output files
            if not self.verify_optimization_results():
                self.logger.error("Optimization verification failed")
                return False
            
            self.logger.info(f"Optimization completed successfully")
            self.logger.info(f"Processed {len(successful_files)} source files")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Optimization failed: {e}")
            return False
    
    def write_optimized_batch(self, batch: List[Tuple[Path, pa.Table]], batch_number: int) -> bool:
        """Write a batch of tables to optimized parquet file"""
        try:
            # Check if all tables have the same schema
            schemas = [table.schema for _, table in batch]
            if not all(schema.equals(self.standard_schema) for schema in schemas):
                self.logger.info("Schema normalization required for batch")
                
            # Normalize all tables to ensure consistent schema
            normalized_tables = []
            for file_path, table in batch:
                if table.schema.equals(self.standard_schema):
                    normalized_tables.append(table)
                    self.logger.debug(f"Schema already matches for {file_path.name}")
                else:
                    try:
                        # Cast to standard schema
                        normalized_table = table.cast(self.standard_schema)
                        normalized_tables.append(normalized_table)
                        self.logger.debug(f"Normalized schema for {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"Schema normalization failed for {file_path.name}: {e}")
                        self.logger.warning(f"  Source schema: {table.schema}")
                        self.logger.warning(f"  Target schema: {self.standard_schema}")
                        # Try to use the original table if it's compatible
                        normalized_tables.append(table)
            
            # Combine all normalized tables
            combined_table = pa.concat_tables(normalized_tables)
            
            # Create output path
            output_path = self.target_dir / f"BTCUSDT-Trades-Optimized-{batch_number:03d}.parquet"
            
            # Write safely
            if self.safe_write_parquet(combined_table, output_path):
                self.logger.info(f"Successfully wrote batch {batch_number}: {output_path.name}")
                self.logger.info(f"  Rows: {combined_table.num_rows:,}")
                self.logger.info(f"  Size: {output_path.stat().st_size / (1024**2):.1f} MB")
                return True
            else:
                self.logger.error(f"Failed to write batch {batch_number}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error writing batch {batch_number}: {e}")
            return False
    
    def verify_optimization_results(self) -> bool:
        """Verify optimization results comprehensively"""
        self.logger.info("Verifying optimization results...")
        
        # Find all output files
        output_files = sorted(self.target_dir.glob("*-Optimized-*.parquet"))
        
        if not output_files:
            self.logger.error("No optimized files found")
            return False
        
        # Verify each file
        all_valid = True
        total_rows = 0
        
        for output_file in output_files:
            if self.verify_file_integrity(output_file):
                # Get row count
                pf = pq.ParquetFile(output_file)
                file_rows = pf.metadata.num_rows
                total_rows += file_rows
                
                self.logger.info(f"✅ Verified: {output_file.name} ({file_rows:,} rows)")
            else:
                self.logger.error(f"❌ Invalid: {output_file.name}")
                all_valid = False
        
        if all_valid:
            self.logger.info(f"✅ All optimized files verified successfully")
            self.logger.info(f"Total rows in optimized files: {total_rows:,}")
        else:
            self.logger.error("❌ Some optimized files failed verification")
        
        return all_valid
    
    def run_optimization(self) -> bool:
        """Run the complete optimization process"""
        try:
            self.logger.info("="*60)
            self.logger.info("STARTING ROBUST PARQUET OPTIMIZATION")
            self.logger.info("="*60)
            
            # Record start time
            self.state['optimization_start'] = datetime.now().isoformat()
            self.save_state()
            
            # Collect source files
            source_files = self.collect_source_files()
            if not source_files:
                return False
            
            # Run optimization
            success = self.optimize_files(source_files)
            
            if success:
                # Verify results
                if self.verify_optimization_results():
                    self.state['optimization_complete'] = True
                    self.save_state()
                    
                    self.logger.info("="*60)
                    self.logger.info("OPTIMIZATION COMPLETED SUCCESSFULLY")
                    self.logger.info("="*60)
                    return True
                else:
                    self.logger.error("Verification failed")
                    return False
            else:
                self.logger.error("Optimization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Optimization process failed: {e}")
            return False
        finally:
            # Cleanup temp directory
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Robust Parquet optimizer with corruption prevention"
    )
    parser.add_argument('--source', required=True, help='Source directory with parquet files')
    parser.add_argument('--target', required=True, help='Target directory for optimized files')
    parser.add_argument('--max-size', type=int, default=10, help='Maximum file size in GB')
    parser.add_argument('--compression', default='snappy', choices=['snappy', 'gzip', 'lz4'],
                       help='Compression algorithm')
    parser.add_argument('--verify-checksum', action='store_true', help='Enable checksum verification')
    parser.add_argument('--keep-backup', action='store_true', help='Keep backup of original files')
    
    args = parser.parse_args()
    
    # Create configuration
    config = OptimizationConfig(
        max_file_size_gb=args.max_size,
        compression=args.compression,
        verify_checksum=args.verify_checksum,
        keep_backup=args.keep_backup
    )
    
    # Create optimizer
    optimizer = RobustParquetOptimizer(args.source, args.target, config)
    
    # Run optimization
    success = optimizer.run_optimization()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()