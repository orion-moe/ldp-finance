#!/usr/bin/env python3
"""
Cleanup script to remove daily parquet files that have already been merged into optimized files.
This script verifies that the data exists in optimized files before deletion.
"""

import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_date_from_filename(filename: str) -> datetime:
    """Extract date from filename like BTCUSDT-Trades-2024-08-05.parquet"""
    try:
        date_str = filename.split('-Trades-')[1].replace('.parquet', '')
        return datetime.strptime(date_str, '%Y-%m-%d')
    except:
        return None

def verify_data_in_optimized(daily_file: Path, optimized_dir: Path) -> bool:
    """Verify if daily file data exists in optimized files"""
    try:
        # Read the daily file to get its date range
        daily_df = pq.read_table(daily_file, columns=['time']).to_pandas()
        if daily_df.empty:
            logger.warning(f"Empty file: {daily_file.name}")
            return False
        
        daily_min_time = daily_df['time'].min()
        daily_max_time = daily_df['time'].max()
        daily_count = len(daily_df)
        
        logger.info(f"Daily file {daily_file.name}:")
        logger.info(f"  - Records: {daily_count:,}")
        logger.info(f"  - Time range: {pd.to_datetime(daily_min_time, unit='ms')} to {pd.to_datetime(daily_max_time, unit='ms')}")
        
        # Check each optimized file
        optimized_files = sorted(optimized_dir.glob("*-Trades-Optimized-*.parquet"))
        
        for opt_file in optimized_files:
            try:
                # Read only the time column
                opt_df = pq.read_table(opt_file, columns=['time']).to_pandas()
                
                # Check if the daily data time range overlaps with this optimized file
                opt_min = opt_df['time'].min()
                opt_max = opt_df['time'].max()
                
                if daily_min_time >= opt_min and daily_max_time <= opt_max:
                    # Count records in the daily time range
                    records_in_range = len(opt_df[(opt_df['time'] >= daily_min_time) & 
                                                  (opt_df['time'] <= daily_max_time)])
                    
                    if records_in_range > 0:
                        logger.info(f"  ✓ Found {records_in_range:,} records in {opt_file.name}")
                        return True
                        
            except Exception as e:
                logger.error(f"Error reading {opt_file.name}: {e}")
                continue
        
        logger.warning(f"  ✗ Data not found in any optimized file")
        return False
        
    except Exception as e:
        logger.error(f"Error verifying {daily_file.name}: {e}")
        return False

def main():
    # Define paths
    base_dir = Path("datasets")
    
    print("\n🔍 Daily Parquet File Cleanup Tool")
    print("=" * 50)
    
    # Get market type
    print("\nSelect market type:")
    print("1. Spot")
    print("2. Futures (USD-M)")
    print("3. Futures (COIN-M)")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        market_type = 'spot'
    elif choice == '2':
        market_type = 'futures-um'
    elif choice == '3':
        market_type = 'futures-cm'
    else:
        print("❌ Invalid choice")
        return
    
    daily_dir = base_dir / "dataset-raw-daily-compressed" / market_type
    optimized_dir = base_dir / "dataset-raw-daily-compressed-optimized" / market_type
    
    if not daily_dir.exists():
        print(f"❌ Daily directory not found: {daily_dir}")
        return
        
    if not optimized_dir.exists():
        print(f"❌ Optimized directory not found: {optimized_dir}")
        return
    
    # Find all daily parquet files
    daily_files = sorted(daily_dir.glob("*-Trades-*.parquet"))
    
    if not daily_files:
        print(f"ℹ️  No daily parquet files found in {daily_dir}")
        return
    
    print(f"\n📊 Found {len(daily_files)} daily parquet files")
    print(f"📁 Optimized directory: {optimized_dir}")
    
    # Analyze files
    verified_files = []
    unverified_files = []
    
    print("\n🔍 Verifying files...")
    for daily_file in daily_files:
        if verify_data_in_optimized(daily_file, optimized_dir):
            verified_files.append(daily_file)
        else:
            unverified_files.append(daily_file)
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  - Verified (safe to delete): {len(verified_files)} files")
    print(f"  - Unverified (keep): {len(unverified_files)} files")
    
    if unverified_files:
        print(f"\n⚠️  Unverified files (will NOT be deleted):")
        for f in unverified_files:
            print(f"  - {f.name}")
    
    if not verified_files:
        print("\nℹ️  No files to delete")
        return
    
    # Calculate space to be freed
    total_size = sum(f.stat().st_size for f in verified_files)
    total_size_mb = total_size / (1024 * 1024)
    
    print(f"\n💾 Space to be freed: {total_size_mb:.1f} MB")
    
    # Confirm deletion
    print(f"\n⚠️  Ready to delete {len(verified_files)} verified files")
    confirm = input("Proceed with deletion? (yes/no): ").strip().lower()
    
    if confirm != 'yes':
        print("❌ Deletion cancelled")
        return
    
    # Delete files
    deleted_count = 0
    for daily_file in verified_files:
        try:
            daily_file.unlink()
            logger.info(f"Deleted: {daily_file.name}")
            deleted_count += 1
        except Exception as e:
            logger.error(f"Failed to delete {daily_file.name}: {e}")
    
    print(f"\n✅ Deleted {deleted_count} files")
    print(f"💾 Freed {total_size_mb:.1f} MB of disk space")

if __name__ == "__main__":
    main()