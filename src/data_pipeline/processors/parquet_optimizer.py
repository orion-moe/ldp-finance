#!/usr/bin/env python3
"""
Enhanced Parquet optimizer with robust cleanup and verification features
"""

import os
import glob
import shutil
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from validators.missing_dates_validator import MissingDatesValidator

class EnhancedParquetOptimizer:
    def __init__(self, source_dir: str, target_dir: str, max_size_gb: int = 10, compression: str = "snappy"):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.max_size_bytes = max_size_gb * 1024**3
        self.compression = compression
        self._auto_confirm = False
        self.symbol = None  # Will be auto-detected

        # Create target directory
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _detect_symbol_from_files(self) -> str:
        """Auto-detect symbol from first parquet file in source directory"""
        import re

        # Get first parquet file
        parquet_files = list(self.source_dir.glob("*.parquet"))
        if not parquet_files:
            self.logger.warning("No parquet files found, defaulting to BTCUSDT")
            return "BTCUSDT"

        # Extract symbol from filename (pattern: SYMBOL-trades-YYYY-MM-DD.parquet or SYMBOL-Trades-YYYY-MM.parquet)
        first_file = parquet_files[0].name
        match = re.match(r"([A-Z]+)-[Tt]rades-", first_file)
        if match:
            symbol = match.group(1)
            self.logger.info(f"Auto-detected symbol: {symbol}")
            return symbol

        self.logger.warning(f"Could not detect symbol from {first_file}, defaulting to BTCUSDT")
        return "BTCUSDT"
    
    def get_parquet_info(self, file_path: Path) -> dict:
        """Get information about a Parquet file"""
        pf = pq.ParquetFile(file_path)
        return {
            'path': file_path,
            'size': file_path.stat().st_size,
            'rows': pf.metadata.num_rows,
            'row_groups': pf.num_row_groups
        }
    
    def verify_file_integrity(self, file_path: Path) -> bool:
        """Verify that a Parquet file can be read correctly"""
        try:
            pf = pq.ParquetFile(file_path)
            expected_rows = pf.metadata.num_rows
            
            # Test read a sample of the file
            test_table = pq.read_table(file_path, columns=['trade_id'], use_threads=False)
            actual_rows = len(test_table)
            
            if actual_rows != expected_rows:
                self.logger.error(f"Row count mismatch in {file_path.name}: expected {expected_rows}, got {actual_rows}")
                return False
            
            return True
        except Exception as e:
            self.logger.error(f"Error verifying {file_path.name}: {e}")
            return False
    
    def check_missing_dates(self) -> bool:
        """Check for missing dates before optimization"""
        self.logger.info("\n🔍 Checking for missing dates in the dataset...")

        # Auto-detect symbol if not already set
        if not self.symbol:
            self.symbol = self._detect_symbol_from_files()

        # Initialize validator
        validator = MissingDatesValidator(str(self.source_dir), self.symbol)
        report = validator.generate_report(check_daily_gaps=False)
        
        # Print summary
        summary = report['summary']
        self.logger.info(f"📊 Data completeness: {summary['completeness_percentage']:.1f}%")
        
        if report['missing_months']:
            self.logger.warning(f"⚠️  Found {len(report['missing_months'])} missing months:")
            for missing in report['missing_months'][:10]:  # Show first 10
                self.logger.warning(f"   - {missing['date_str']}")
            if len(report['missing_months']) > 10:
                self.logger.warning(f"   ... and {len(report['missing_months']) - 10} more")
            
            # Ask user if they want to continue
            if not self._auto_confirm:
                print("\n" + "="*60)
                print("⚠️  MISSING DATA DETECTED")
                print("="*60)
                print(f"There are {len(report['missing_months'])} missing months in the dataset.")
                print("This might affect your analysis or model training.")
                response = input("\nDo you want to continue with optimization anyway? (yes/no): ").strip().lower()
                if response != 'yes':
                    self.logger.info("Optimization cancelled by user.")
                    return False
            else:
                self.logger.warning("Auto-confirm enabled. Proceeding despite missing data.")
        else:
            self.logger.info("✅ No missing months detected!")
        
        return True
    
    def optimize_parquet_files(self):
        """Main optimization process with enhanced verification"""
        self.logger.info("🚀 Starting Enhanced Parquet optimization process")
        self.logger.info(f"Source: {self.source_dir}")
        self.logger.info(f"Target: {self.target_dir}")
        self.logger.info(f"Max file size: {self.max_size_bytes / (1024**3):.1f} GB")
        
        # Check for missing dates first
        if not self.check_missing_dates():
            return
        
        # Get all parquet files
        source_files = sorted(glob.glob(str(self.source_dir / "*.parquet")))
        
        if not source_files:
            self.logger.error("No Parquet files found in source directory!")
            return
        
        self.logger.info(f"Found {len(source_files)} source files")
        
        # Verify all source files before processing
        self.logger.info("🔍 Pre-processing verification...")
        valid_files = []
        for file_path in source_files:
            if self.verify_file_integrity(Path(file_path)):
                valid_files.append(file_path)
                self.logger.info(f"✅ Valid: {Path(file_path).name}")
            else:
                self.logger.error(f"❌ Invalid: {Path(file_path).name} - skipping")
        
        if not valid_files:
            self.logger.error("No valid source files found!")
            return
        
        # Collect file information
        file_infos = []
        total_size = 0
        total_rows = 0
        
        for file_path in valid_files:
            info = self.get_parquet_info(Path(file_path))
            file_infos.append(info)
            total_size += info['size']
            total_rows += info['rows']
        
        self.logger.info(f"Total data: {total_size / (1024**3):.2f} GB, {total_rows:,} rows")
        
        # Process files
        current_writer = None
        current_output_path = None
        current_size = 0
        output_file_count = 1
        processed_files = []
        created_files = []
        
        # Detect schema from first valid file
        standard_schema = None
        first_file_table = pq.read_table(valid_files[0])
        standard_schema = first_file_table.schema
        self.logger.info(f"Detected schema from {Path(valid_files[0]).name}: {standard_schema}")
        
        try:
            for i, file_info in enumerate(file_infos):
                self.logger.info(f"Processing {i+1}/{len(file_infos)}: {file_info['path'].name} "
                               f"({file_info['size'] / (1024**2):.1f} MB, {file_info['rows']:,} rows)")
                
                # Read the entire parquet file
                table = pq.read_table(file_info['path'])
                
                # Normalize schema to match standard schema
                if not table.schema.equals(standard_schema):
                    try:
                        # Cast to standard schema
                        table = table.cast(standard_schema)
                        self.logger.debug(f"  Normalized schema for {file_info['path'].name}")
                    except Exception as e:
                        self.logger.warning(f"  Schema normalization failed for {file_info['path'].name}: {e}")
                        self.logger.warning(f"    Source schema: {table.schema}")
                        self.logger.warning(f"    Target schema: {standard_schema}")
                        # Continue with original table
                
                # Standardize schema (legacy handling for isBestMatch)
                if 'isBestMatch' not in table.column_names:
                    # Add missing column as boolean
                    null_column = pa.array([None] * len(table), type=pa.bool_())
                    table = table.append_column('isBestMatch', null_column)
                else:
                    # Check if isBestMatch needs type conversion
                    isBestMatch_col = table.column('isBestMatch')
                    if isBestMatch_col.type != pa.bool_():
                        # Convert to boolean
                        self.logger.info(f"  Converting isBestMatch from {isBestMatch_col.type} to bool")
                        # If numeric, convert non-zero to True
                        if pa.types.is_floating(isBestMatch_col.type) or pa.types.is_integer(isBestMatch_col.type):
                            bool_array = pa.compute.not_equal(isBestMatch_col, pa.scalar(0))
                        else:
                            # For other types, convert to bool directly
                            bool_array = pa.compute.cast(isBestMatch_col, pa.bool_())
                        
                        # Replace the column
                        table = table.drop(['isBestMatch'])
                        table = table.append_column('isBestMatch', bool_array)
                
                # Ensure columns are in the correct order
                table = table.select(['trade_id', 'price', 'qty', 'quoteQty', 'time', 'isBuyerMaker', 'isBestMatch'])
                
                file_size = file_info['size']
                
                # Check if we need a new output file
                # Note: We need to check the actual size on disk, not the input file size
                # because compression and schema changes affect the final size
                if current_writer is not None and current_output_path.exists():
                    # Get actual current size from disk
                    actual_current_size = current_output_path.stat().st_size
                else:
                    actual_current_size = current_size
                
                # Estimate size after adding this table (conservative estimate)
                # The input file size is not a good predictor due to recompression
                estimated_addition = file_size * 0.9  # Assume similar compression ratio
                
                if current_writer is None or (actual_current_size + estimated_addition > self.max_size_bytes * 0.95):
                    # Close current writer if exists
                    if current_writer:
                        current_writer.close()
                        actual_size = current_output_path.stat().st_size
                        self.logger.info(f"✅ Completed: {current_output_path.name} "
                                       f"({actual_size / (1024**3):.2f} GB)")
                        created_files.append(current_output_path)
                    
                    # Create new output file
                    current_output_path = self.target_dir / f"{self.symbol}-Trades-Optimized-{output_file_count:03d}.parquet"
                    current_writer = pq.ParquetWriter(
                        current_output_path,
                        standard_schema,
                        compression=self.compression,
                        use_dictionary=True,
                        use_deprecated_int96_timestamps=False
                    )
                    current_size = 0
                    output_file_count += 1
                    self.logger.info(f"📝 Creating new file: {current_output_path.name}")
                
                # Write table to current file
                current_writer.write_table(table)
                # Track approximate size (writers buffer data, so disk size isn't accurate until close)
                # Use a more accurate estimate based on the compression ratio observed
                current_size += file_size * 0.85  # Typical compression ratio for this data
                processed_files.append(file_info['path'])
            
            # Close final writer
            if current_writer:
                current_writer.close()
                self.logger.info(f"✅ Completed: {current_output_path.name} "
                               f"({current_output_path.stat().st_size / (1024**3):.2f} GB)")
                created_files.append(current_output_path)
            
            # Summary
            self.logger.info("\n" + "="*60)
            self.logger.info("📊 OPTIMIZATION COMPLETE")
            self.logger.info("="*60)
            self.logger.info(f"Source files processed: {len(processed_files)}")
            self.logger.info(f"Optimized files created: {len(created_files)}")
            
            # Calculate space difference
            optimized_size = sum(f.stat().st_size for f in created_files)
            self.logger.info(f"Original size: {total_size / (1024**3):.2f} GB")
            self.logger.info(f"Optimized size: {optimized_size / (1024**3):.2f} GB")
            self.logger.info(f"Space difference: {(optimized_size - total_size) / (1024**3):.2f} GB")
            
            # Comprehensive verification
            if self._verify_optimization(created_files, total_rows):
                self._perform_cleanup(processed_files, total_size)
            else:
                self.logger.error("❌ Verification failed - cleanup aborted")
        
        except Exception as e:
            self.logger.error(f"❌ Error during optimization: {e}")
            if current_writer:
                current_writer.close()
            raise
    
    def _verify_optimization(self, created_files: list, expected_total_rows: int) -> bool:
        """Comprehensive verification of optimized files"""
        self.logger.info("\n🔍 Comprehensive verification...")
        
        try:
            total_optimized_rows = 0
            
            for file_path in created_files:
                if not self.verify_file_integrity(file_path):
                    return False
                
                pf = pq.ParquetFile(file_path)
                file_rows = pf.metadata.num_rows
                total_optimized_rows += file_rows
                self.logger.info(f"✅ Verified: {file_path.name} ({file_rows:,} rows)")
            
            if total_optimized_rows == expected_total_rows:
                self.logger.info(f"✅ Total row count verified: {total_optimized_rows:,} rows")
                return True
            else:
                self.logger.error(f"❌ Row count mismatch! Expected: {expected_total_rows:,}, Got: {total_optimized_rows:,}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Verification error: {e}")
            return False
    
    def _perform_cleanup(self, processed_files: list, total_size: int):
        """Safe cleanup of original files"""
        
        # Final safety check
        optimized_files = list(self.target_dir.glob("*.parquet"))
        if len(optimized_files) == 0:
            self.logger.error("❌ No optimized files found! Aborting cleanup.")
            return
        
        # Ask for confirmation
        print("\n" + "="*60)
        print("⚠️  READY TO DELETE ORIGINAL FILES")
        print("="*60)
        print(f"This will delete {len(processed_files)} original Parquet files")
        print(f"Total size to delete: {total_size / (1024**3):.2f} GB")
        print(f"Optimized files location: {self.target_dir}")
        print(f"Number of optimized files: {len(optimized_files)}")
        
        if self._auto_confirm:
            response = 'yes'
            print("\n🤖 Auto-confirm enabled. Proceeding with deletion...")
        else:
            response = input("\nProceed with deletion? (yes/no): ").strip().lower()
        
        if response == 'yes':
            self.logger.info("\n🗑️  Deleting original files...")
            deleted_count = 0
            deleted_size = 0
            failed_deletions = []
            
            for file_path in processed_files:
                try:
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_count += 1
                    deleted_size += file_size
                    self.logger.info(f"✅ Deleted: {file_path.name}")
                except Exception as e:
                    self.logger.error(f"❌ Error deleting {file_path.name}: {e}")
                    failed_deletions.append(file_path.name)
            
            # Final summary
            self.logger.info(f"\n✅ Cleanup summary:")
            self.logger.info(f"   - Deleted: {deleted_count} files ({deleted_size / (1024**3):.2f} GB)")
            self.logger.info(f"   - Failed: {len(failed_deletions)} files")
            
            if failed_deletions:
                self.logger.warning(f"⚠️  Failed to delete: {', '.join(failed_deletions)}")
            
            # Check if source directory is clean
            remaining_files = list(self.source_dir.glob("*.parquet"))
            if not remaining_files:
                self.logger.info(f"🗂️  Source directory is now clean: {self.source_dir}")
            else:
                self.logger.warning(f"⚠️  {len(remaining_files)} files remain in source directory")
            
            self.logger.info("✅ Optimization and cleanup complete!")
        else:
            self.logger.info("\n⚠️  Deletion cancelled. Original files preserved.")
            self.logger.info("💡 To delete manually later:")
            self.logger.info(f"   rm {self.source_dir}/*.parquet")
            self.logger.info("💡 Or run again with --auto-confirm flag")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced Parquet optimizer with robust cleanup')
    parser.add_argument('--source', type=str, 
                       default='dataset-raw-monthly-compressed/futures-um',
                       help='Source directory with Parquet files')
    parser.add_argument('--target', type=str,
                       default='dataset-raw-monthly-compressed-optimized/futures-um',
                       help='Target directory for optimized files')
    parser.add_argument('--max-size', type=int, default=10,
                       help='Maximum file size in GB (default: 10)')
    parser.add_argument('--compression', type=str, default='snappy',
                       choices=['snappy', 'zstd', 'lz4', 'brotli', 'gzip', 'none'],
                       help='Compression algorithm (default: snappy)')
    parser.add_argument('--auto-confirm', action='store_true',
                       help='Automatically confirm deletion of original files')

    args = parser.parse_args()

    optimizer = EnhancedParquetOptimizer(args.source, args.target, args.max_size, args.compression)
    
    if args.auto_confirm:
        optimizer._auto_confirm = True
    
    optimizer.optimize_parquet_files()

if __name__ == "__main__":
    main()