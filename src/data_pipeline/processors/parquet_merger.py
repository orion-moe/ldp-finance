"""
Parquet Merger Module
Merges daily parquet files into existing optimized parquet files
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
import shutil
from datetime import datetime


class ParquetMerger:
    """Handles merging of daily parquet files into optimized files"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol
        
    def find_last_optimized_file(self, optimized_dir: Path) -> Optional[Path]:
        """Find the last optimized parquet file in the directory"""
        pattern = f"{self.symbol}-Trades-Optimized-*.parquet"
        optimized_files = sorted(optimized_dir.glob(pattern))
        
        if not optimized_files:
            logger.warning(f"No optimized files found in {optimized_dir}")
            return None
            
        # Return the last file (highest number)
        return optimized_files[-1]
    
    def get_last_timestamp(self, file_path: Path) -> Optional[int]:
        """Get the last timestamp from a parquet file"""
        try:
            # Read only the last row to get the timestamp
            df = pd.read_parquet(file_path, columns=['time'])
            if df.empty:
                return None
            return int(df['time'].max())
        except Exception as e:
            logger.error(f"Error reading last timestamp from {file_path}: {e}")
            return None
    
    def merge_daily_files(self, 
                         optimized_dir: Path,
                         daily_dir: Path,
                         daily_files: List[Path],
                         max_file_size_gb: float = 10.0,
                         delete_after_merge: bool = True) -> Tuple[int, int]:
        """
        Merge daily parquet files into the last optimized file or create new ones
        
        Args:
            optimized_dir: Directory containing optimized parquet files
            daily_dir: Directory containing daily parquet files
            daily_files: List of daily parquet files to merge
            max_file_size_gb: Maximum file size in GB (default 10.0)
            delete_after_merge: Delete daily files after successful merge (default True)
            
        Returns:
            Tuple of (files_merged, rows_added)
        """
        if not daily_files:
            logger.info("No daily files to merge")
            return 0, 0
            
        # Find the last optimized file
        last_optimized = self.find_last_optimized_file(optimized_dir)
        
        # Sort daily files by date to ensure chronological order
        daily_files_sorted = sorted(daily_files)
        
        logger.info(f"Processing {len(daily_files_sorted)} daily files...")
        
        # Track overall progress
        total_files_merged = 0
        total_rows_added = 0
        successfully_processed_files = []
        
        # Get the last optimized file and its info
        current_optimized = self.find_last_optimized_file(optimized_dir)
        current_data = None
        current_size_gb = 0.0
        last_timestamp = None
        
        if current_optimized and current_optimized.exists():
            current_size_gb = current_optimized.stat().st_size / (1024**3)
            # Read current data to get last timestamp
            current_data = pd.read_parquet(current_optimized)
            last_timestamp = current_data['time'].max()
            logger.info(f"Current optimized file: {current_optimized.name} ({current_size_gb:.2f} GB, {len(current_data)} rows)")
            logger.info(f"Last timestamp in optimized data: {last_timestamp}")
        
        # Process daily files one by one
        for i, daily_file in enumerate(daily_files_sorted):
            try:
                logger.info(f"Processing file {i+1}/{len(daily_files_sorted)}: {daily_file.name}")
                
                # Read the daily file
                daily_df = pd.read_parquet(daily_file)
                logger.debug(f"Read {len(daily_df)} rows from {daily_file.name}")
                
                # Filter out data that already exists (if we have a last timestamp)
                if last_timestamp is not None:
                    daily_df_filtered = daily_df[daily_df['time'] > last_timestamp]
                    logger.debug(f"Filtered to {len(daily_df_filtered)} new rows after timestamp {last_timestamp}")
                else:
                    daily_df_filtered = daily_df
                
                if len(daily_df_filtered) == 0:
                    logger.info(f"No new data in {daily_file.name}, skipping")
                    successfully_processed_files.append(daily_file)
                    continue
                
                # Check if adding this data would exceed the size limit
                # Note: memory_usage gives in-memory size, not compressed parquet size
                # Parquet compression ratio is typically 3-5x, so we use a conservative 3x compression factor
                estimated_memory_size = daily_df_filtered.memory_usage(deep=True).sum() / (1024**3)  # GB
                estimated_new_size = estimated_memory_size / 3.0  # Conservative estimate of compressed size
                
                # Get actual current file size from disk if exists
                if current_optimized and current_optimized.exists():
                    current_size_gb = current_optimized.stat().st_size / (1024**3)
                
                if current_data is not None and (current_size_gb + estimated_new_size) > max_file_size_gb * 0.95:  # 5% buffer
                    # Need to create a new optimized file
                    logger.info(f"Current file would exceed {max_file_size_gb}GB limit, creating new optimized file")
                    
                    # Write current data to current file (if modified)
                    if total_rows_added > 0 and current_optimized is not None and current_optimized.exists():
                        # Sort by time
                        current_data = current_data.sort_values('time').reset_index(drop=True)
                        
                        try:
                            # Write data
                            table = pa.Table.from_pandas(current_data)
                            pq.write_table(table, current_optimized, compression='snappy')
                            logger.success(f"Saved {current_optimized.name} with {len(current_data)} rows")
                        except Exception as e:
                            logger.error(f"Error writing data: {e}")
                            raise
                    
                    # Create new optimized file
                    # Always find the highest numbered file to determine next number
                    existing_files = sorted(optimized_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))
                    if existing_files:
                        # Get the highest number from all files
                        last_file = existing_files[-1]
                        last_filename = last_file.stem
                        last_number_str = last_filename.split('-')[-1]
                        last_number = int(last_number_str)
                        file_number = last_number + 1
                    else:
                        file_number = 1
                    
                    current_optimized = optimized_dir / f"{self.symbol}-Trades-Optimized-{file_number:03d}.parquet"
                    current_data = daily_df_filtered
                    current_size_gb = estimated_new_size
                    total_files_merged += 1
                else:
                    # Append to current data
                    if current_data is None:
                        # First file, no existing optimized data
                        current_data = daily_df_filtered
                        current_size_gb = estimated_new_size
                        if current_optimized is None:
                            current_optimized = optimized_dir / f"{self.symbol}-Trades-Optimized-001.parquet"
                            total_files_merged = 1
                    else:
                        # Append to existing data
                        current_data = pd.concat([current_data, daily_df_filtered], ignore_index=True)
                        # Update size estimate based on new total memory usage
                        total_memory = current_data.memory_usage(deep=True).sum() / (1024**3)
                        current_size_gb = total_memory / 3.0  # Conservative compression estimate
                
                # Update tracking
                total_rows_added += len(daily_df_filtered)
                last_timestamp = daily_df_filtered['time'].max()
                successfully_processed_files.append(daily_file)
                
                logger.info(f"Added {len(daily_df_filtered)} rows from {daily_file.name}")
                
                # Delete the daily file if requested
                if delete_after_merge:
                    try:
                        daily_file.unlink()
                        logger.debug(f"Deleted processed file: {daily_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {daily_file.name}: {e}")
                
            except Exception as e:
                logger.error(f"Error processing {daily_file.name}: {e}")
                # Don't delete files that had errors
                continue
        
        # Write any remaining data
        if current_data is not None and len(current_data) > 0:
            try:
                # Sort by time
                current_data = current_data.sort_values('time').reset_index(drop=True)
                
                # Write data
                table = pa.Table.from_pandas(current_data)
                pq.write_table(table, current_optimized, compression='snappy')
                
                if current_optimized.exists():
                    logger.success(f"Final save: {current_optimized.name} with {len(current_data)} rows")
                else:
                    logger.success(f"Created: {current_optimized.name} with {len(current_data)} rows")
                    if total_files_merged == 0:
                        total_files_merged = 1
            except Exception as e:
                logger.error(f"Error writing final optimized file: {e}")
                raise
        
        logger.info(f"Merge completed: {total_rows_added} rows added, {len(successfully_processed_files)} files processed")
        return total_files_merged, total_rows_added
    
    def _create_new_optimized_file(self, optimized_dir: Path, data: pd.DataFrame, file_number: Optional[int] = None) -> int:
        """Create a new optimized parquet file"""
        if file_number is None:
            # Find the next available number
            existing_files = sorted(optimized_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))
            if existing_files:
                # Extract number from filename like BTCUSDT-Trades-Optimized-012.parquet
                last_filename = existing_files[-1].stem  # e.g., 'BTCUSDT-Trades-Optimized-012'
                last_number_str = last_filename.split('-')[-1]  # '012'
                last_number = int(last_number_str)
                file_number = last_number + 1
                logger.info(f"Found {len(existing_files)} existing optimized files, last number: {last_number}")
            else:
                file_number = 1
                logger.info("No existing optimized files found, starting with 001")
                
        output_path = optimized_dir / f"{self.symbol}-Trades-Optimized-{file_number:03d}.parquet"
        
        # Write data
        logger.info(f"Writing {len(data)} rows to {output_path}")
        table = pa.Table.from_pandas(data)
        pq.write_table(table, output_path, compression='snappy')
        logger.success(f"Created new optimized file: {output_path.name} with {len(data)} rows")
        
        return 1
    
    def cleanup_merged_daily_files(self, daily_files: List[Path], dry_run: bool = False) -> int:
        """Remove daily files that have been successfully merged"""
        cleaned = 0
        for file in daily_files:
            if dry_run:
                logger.info(f"Would delete: {file}")
            else:
                try:
                    file.unlink()
                    logger.debug(f"Deleted {file}")
                    cleaned += 1
                except Exception as e:
                    logger.error(f"Error deleting {file}: {e}")
        
        if not dry_run:
            logger.info(f"Cleaned up {cleaned} daily files")
        
        return cleaned