"""
Parquet Merger Module (Memory-Optimized Version)
Merges daily parquet files into existing optimized parquet files with memory management
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Optional, Tuple
from loguru import logger
import shutil
from datetime import datetime
import gc  # Garbage collection


class ParquetMerger:
    """Handles merging of daily parquet files into optimized files with memory management"""

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
        """Get the last timestamp from a parquet file without loading entire file"""
        try:
            # Use PyArrow to read only metadata and last row batch
            parquet_file = pq.ParquetFile(file_path)

            # Get the last row group
            if parquet_file.num_row_groups == 0:
                return None

            # Read only the time column from the last row group
            last_row_group = parquet_file.read_row_group(
                parquet_file.num_row_groups - 1,
                columns=['time']
            )

            df = last_row_group.to_pandas()
            if df.empty:
                return None

            # Convert Timestamp to milliseconds (int)
            max_timestamp = df['time'].max()
            if hasattr(max_timestamp, 'value'):
                # If it's a pandas Timestamp, convert from nanoseconds to milliseconds
                max_time = int(max_timestamp.value // 10**6)
            else:
                # If it's already a number, just convert to int
                max_time = int(max_timestamp)

            # Clean up
            del df
            del last_row_group
            gc.collect()

            return max_time

        except Exception as e:
            logger.error(f"Error reading last timestamp from {file_path}: {e}")
            return None

    def get_file_row_count(self, file_path: Path) -> int:
        """Get row count without loading file into memory"""
        try:
            parquet_file = pq.ParquetFile(file_path)
            return parquet_file.metadata.num_rows
        except Exception as e:
            logger.error(f"Error getting row count from {file_path}: {e}")
            return 0

    def append_to_parquet_file(self, existing_file: Path, new_data: pd.DataFrame, output_file: Optional[Path] = None) -> Path:
        """Append new data to existing parquet file in a memory-efficient way"""
        if output_file is None:
            output_file = existing_file

        # Create a temporary file for writing
        temp_file = existing_file.parent / f"{existing_file.stem}_temp.parquet"

        try:
            # Open the existing file for reading
            existing_pf = pq.ParquetFile(existing_file)
            schema = existing_pf.schema_arrow

            # Create writer with the same schema
            writer = pq.ParquetWriter(temp_file, schema, compression='snappy')

            # Copy existing data in chunks (row groups)
            for i in range(existing_pf.num_row_groups):
                row_group = existing_pf.read_row_group(i)
                writer.write_table(row_group)
                del row_group  # Free memory immediately

                # Periodic garbage collection
                if i % 10 == 0:
                    gc.collect()

            # Write new data
            new_table = pa.Table.from_pandas(new_data, schema=schema)
            writer.write_table(new_table)

            # Clean up
            writer.close()
            del new_table
            gc.collect()

            # Replace original file with temp file
            if output_file == existing_file:
                existing_file.unlink()
            temp_file.rename(output_file)

            return output_file

        except Exception as e:
            # Clean up temp file on error
            if temp_file.exists():
                temp_file.unlink()
            raise e

    def merge_daily_files(self,
                         optimized_dir: Path,
                         daily_dir: Path,
                         daily_files: List[Path],
                         max_file_size_gb: float = 10.0,
                         delete_after_merge: bool = True) -> Tuple[int, int]:
        """
        Memory-optimized merge of daily parquet files into optimized files

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

        # Sort daily files by date
        daily_files_sorted = sorted(daily_files)
        logger.info(f"Processing {len(daily_files_sorted)} daily files...")

        # Track progress
        total_files_merged = 0
        total_rows_added = 0
        successfully_processed_files = []

        # Find current optimized file
        current_optimized = self.find_last_optimized_file(optimized_dir)

        # Get last timestamp without loading entire file
        last_timestamp = None
        current_row_count = 0
        current_size_gb = 0.0

        if current_optimized and current_optimized.exists():
            last_timestamp = self.get_last_timestamp(current_optimized)
            current_row_count = self.get_file_row_count(current_optimized)
            current_size_gb = current_optimized.stat().st_size / (1024**3)

            logger.info(f"Current optimized file: {current_optimized.name}")
            logger.info(f"  Size: {current_size_gb:.2f} GB")
            logger.info(f"  Rows: {current_row_count:,}")
            logger.info(f"  Last timestamp: {last_timestamp}")

        # Accumulate data for batch writing
        batch_data = []
        batch_size_mb = 0
        batch_size_threshold_mb = 500  # Write in 500MB batches

        for i, daily_file in enumerate(daily_files_sorted):
            try:
                logger.info(f"Processing file {i+1}/{len(daily_files_sorted)}: {daily_file.name}")

                # Read the daily file
                daily_df = pd.read_parquet(daily_file)
                logger.debug(f"Read {len(daily_df)} rows from {daily_file.name}")

                # Filter out existing data
                if last_timestamp is not None:
                    # Convert last_timestamp to pandas Timestamp if it's an int (milliseconds)
                    if isinstance(last_timestamp, int):
                        last_timestamp_dt = pd.Timestamp(last_timestamp, unit='ms')
                    else:
                        last_timestamp_dt = last_timestamp

                    daily_df_filtered = daily_df[daily_df['time'] > last_timestamp_dt]
                    logger.debug(f"Filtered to {len(daily_df_filtered)} new rows")
                else:
                    daily_df_filtered = daily_df

                # Clean up original dataframe
                del daily_df
                gc.collect()

                if len(daily_df_filtered) == 0:
                    logger.info(f"No new data in {daily_file.name}, skipping")
                    successfully_processed_files.append(daily_file)
                    continue

                # Add to batch
                batch_data.append(daily_df_filtered)
                batch_size_mb += daily_df_filtered.memory_usage(deep=True).sum() / (1024**2)

                # Update tracking
                total_rows_added += len(daily_df_filtered)
                last_timestamp = daily_df_filtered['time'].max()
                successfully_processed_files.append(daily_file)

                # Check if we should write the batch
                estimated_total_size = current_size_gb + (batch_size_mb / 1024)

                if batch_size_mb >= batch_size_threshold_mb or estimated_total_size >= max_file_size_gb * 0.95:
                    # Write batch to file
                    if batch_data:
                        logger.info(f"Writing batch of {len(batch_data)} dataframes ({batch_size_mb:.1f} MB)")

                        # Combine batch data
                        combined_df = pd.concat(batch_data, ignore_index=True)
                        combined_df = combined_df.sort_values('time').reset_index(drop=True)

                        # Clear batch data to free memory
                        batch_data.clear()
                        gc.collect()

                        if estimated_total_size >= max_file_size_gb * 0.95:
                            # Need new file
                            logger.info(f"Creating new optimized file (current would exceed {max_file_size_gb} GB)")

                            # Get next file number
                            existing_files = sorted(optimized_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))
                            if existing_files:
                                last_number = int(existing_files[-1].stem.split('-')[-1])
                                file_number = last_number + 1
                            else:
                                file_number = 1

                            # Create new file
                            current_optimized = optimized_dir / f"{self.symbol}-Trades-Optimized-{file_number:03d}.parquet"

                            # Write as new file
                            table = pa.Table.from_pandas(combined_df)
                            pq.write_table(table, current_optimized, compression='snappy')

                            logger.success(f"Created new file: {current_optimized.name}")
                            total_files_merged += 1

                            # Update current file info
                            current_size_gb = current_optimized.stat().st_size / (1024**3)
                            current_row_count = len(combined_df)

                        else:
                            # Append to existing file
                            if current_optimized and current_optimized.exists():
                                self.append_to_parquet_file(current_optimized, combined_df)
                                logger.success(f"Appended {len(combined_df)} rows to {current_optimized.name}")
                            else:
                                # Create first optimized file
                                current_optimized = optimized_dir / f"{self.symbol}-Trades-Optimized-001.parquet"
                                table = pa.Table.from_pandas(combined_df)
                                pq.write_table(table, current_optimized, compression='snappy')
                                logger.success(f"Created first optimized file: {current_optimized.name}")
                                total_files_merged = 1

                            # Update current file info
                            current_size_gb = current_optimized.stat().st_size / (1024**3)
                            current_row_count += len(combined_df)

                        # Clean up
                        del combined_df
                        gc.collect()

                        # Reset batch size
                        batch_size_mb = 0

                # Delete processed file if requested
                if delete_after_merge:
                    try:
                        daily_file.unlink()
                        logger.debug(f"Deleted: {daily_file.name}")
                    except Exception as e:
                        logger.warning(f"Could not delete {daily_file.name}: {e}")

            except Exception as e:
                logger.error(f"Error processing {daily_file.name}: {e}")
                continue

        # Write any remaining batch data
        if batch_data:
            logger.info(f"Writing final batch of {len(batch_data)} dataframes")

            # Combine remaining data
            combined_df = pd.concat(batch_data, ignore_index=True)
            combined_df = combined_df.sort_values('time').reset_index(drop=True)

            if current_optimized and current_optimized.exists():
                self.append_to_parquet_file(current_optimized, combined_df)
                logger.success(f"Final append: {len(combined_df)} rows to {current_optimized.name}")
            else:
                # Create first file
                current_optimized = optimized_dir / f"{self.symbol}-Trades-Optimized-001.parquet"
                table = pa.Table.from_pandas(combined_df)
                pq.write_table(table, current_optimized, compression='snappy')
                logger.success(f"Created: {current_optimized.name} with {len(combined_df)} rows")
                total_files_merged = 1

            # Clean up
            del combined_df
            batch_data.clear()
            gc.collect()

        logger.info(f"Merge completed: {total_rows_added:,} rows added")
        logger.info(f"Files processed: {len(successfully_processed_files)}")

        # Final garbage collection
        gc.collect()

        return total_files_merged, total_rows_added