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
                         max_file_size_gb: float = 10.0) -> Tuple[int, int]:
        """
        Merge daily parquet files into the last optimized file or create new ones
        
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
        
        # Read all daily data
        logger.info(f"Reading {len(daily_files_sorted)} daily files...")
        daily_dfs = []
        for file in daily_files_sorted:
            try:
                df = pd.read_parquet(file)
                daily_dfs.append(df)
                logger.debug(f"Read {len(df)} rows from {file.name}")
            except Exception as e:
                logger.error(f"Error reading {file}: {e}")
                continue
        
        if not daily_dfs:
            logger.error("Failed to read any daily files")
            return 0, 0
            
        # Combine all daily data
        new_data = pd.concat(daily_dfs, ignore_index=True)
        logger.info(f"Combined {len(new_data)} rows from daily files")
        
        # Sort by time to ensure chronological order
        new_data = new_data.sort_values('time').reset_index(drop=True)
        
        files_merged = 0
        rows_added = len(new_data)
        
        if last_optimized and last_optimized.exists():
            # Check if we should append to the last file or create a new one
            last_file_size = last_optimized.stat().st_size / (1024**3)  # GB
            
            if last_file_size < max_file_size_gb * 0.9:  # Leave 10% buffer
                # Append to existing file
                logger.info(f"Appending to {last_optimized.name} (current size: {last_file_size:.2f} GB)")
                
                # Read existing data
                existing_data = pd.read_parquet(last_optimized)
                logger.info(f"Existing file has {len(existing_data)} rows")
                
                # Get last timestamp from existing data
                last_timestamp = existing_data['time'].max()
                
                # Filter new data to only include rows after the last timestamp
                new_data_filtered = new_data[new_data['time'] > last_timestamp]
                logger.info(f"Filtered to {len(new_data_filtered)} new rows after timestamp {last_timestamp}")
                
                if len(new_data_filtered) > 0:
                    # Combine data
                    combined_data = pd.concat([existing_data, new_data_filtered], ignore_index=True)
                    
                    # Sort by time
                    combined_data = combined_data.sort_values('time').reset_index(drop=True)
                    
                    # Create backup
                    backup_path = last_optimized.with_suffix('.parquet.backup')
                    shutil.copy2(last_optimized, backup_path)
                    logger.info(f"Created backup at {backup_path}")
                    
                    try:
                        # Write combined data
                        table = pa.Table.from_pandas(combined_data)
                        pq.write_table(table, last_optimized, compression='snappy')
                        logger.success(f"Successfully updated {last_optimized.name} with {len(new_data_filtered)} new rows")
                        
                        # Remove backup on success
                        backup_path.unlink()
                        files_merged = 1
                        rows_added = len(new_data_filtered)
                    except Exception as e:
                        # Restore backup on failure
                        logger.error(f"Error writing merged data: {e}")
                        shutil.move(backup_path, last_optimized)
                        raise
                else:
                    logger.info("No new data to add (all timestamps already exist)")
                    rows_added = 0
            else:
                # Create new optimized file
                logger.info(f"Last file is too large ({last_file_size:.2f} GB), creating new file")
                files_merged = self._create_new_optimized_file(optimized_dir, new_data)
        else:
            # No existing optimized files, create the first one
            logger.info("No existing optimized files, creating first file")
            files_merged = self._create_new_optimized_file(optimized_dir, new_data, file_number=1)
            
        return files_merged, rows_added
    
    def _create_new_optimized_file(self, optimized_dir: Path, data: pd.DataFrame, file_number: Optional[int] = None) -> int:
        """Create a new optimized parquet file"""
        if file_number is None:
            # Find the next available number
            existing_files = sorted(optimized_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))
            if existing_files:
                last_number = int(existing_files[-1].stem.split('-')[-1])
                file_number = last_number + 1
            else:
                file_number = 1
                
        output_path = optimized_dir / f"{self.symbol}-Trades-Optimized-{file_number:03d}.parquet"
        
        # Write data
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