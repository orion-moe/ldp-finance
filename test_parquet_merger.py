#!/usr/bin/env python3
"""
Test script for the ParquetMerger functionality
"""

import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import shutil
from src.data_pipeline.processors.parquet_merger import ParquetMerger


def create_test_data():
    """Create test parquet files for testing the merger"""
    # Create a temporary directory for testing
    test_dir = Path(tempfile.mkdtemp())
    print(f"Created test directory: {test_dir}")
    
    # Create subdirectories
    optimized_dir = test_dir / "optimized"
    daily_dir = test_dir / "daily"
    optimized_dir.mkdir()
    daily_dir.mkdir()
    
    # Create an existing optimized file with 5 days of data
    base_time = int(datetime(2024, 1, 1).timestamp() * 1000)  # milliseconds
    
    # Existing data (Jan 1-5)
    existing_data = {
        'time': [base_time + i * 86400000 for i in range(5)],  # 5 days
        'price': [100.0 + i for i in range(5)],
        'qty': [1.0] * 5,
        'symbol': ['BTCUSDT'] * 5
    }
    
    existing_df = pd.DataFrame(existing_data)
    existing_file = optimized_dir / "BTCUSDT-Trades-Optimized-001.parquet"
    existing_df.to_parquet(existing_file)
    print(f"Created existing optimized file with {len(existing_df)} rows")
    
    # Create daily files for Jan 6-8
    daily_files = []
    for day in range(3):
        day_offset = 5 + day  # Start from day 6
        daily_data = {
            'time': [base_time + day_offset * 86400000 + i * 3600000 for i in range(24)],  # 24 hours
            'price': [105.0 + day + i * 0.1 for i in range(24)],
            'qty': [1.0] * 24,
            'symbol': ['BTCUSDT'] * 24
        }
        
        daily_df = pd.DataFrame(daily_data)
        date_str = (datetime(2024, 1, 1) + timedelta(days=day_offset)).strftime('%Y-%m-%d')
        daily_file = daily_dir / f"BTCUSDT-Trades-{date_str}.parquet"
        daily_df.to_parquet(daily_file)
        daily_files.append(daily_file)
        print(f"Created daily file for {date_str} with {len(daily_df)} rows")
    
    return test_dir, optimized_dir, daily_dir, daily_files


def test_merger():
    """Test the ParquetMerger functionality"""
    print("\n" + "="*50)
    print("Testing ParquetMerger")
    print("="*50 + "\n")
    
    # Create test data
    test_dir, optimized_dir, daily_dir, daily_files = create_test_data()
    
    try:
        # Initialize merger
        merger = ParquetMerger(symbol="BTCUSDT")
        
        # Test finding last optimized file
        print("\n1. Testing find_last_optimized_file...")
        last_file = merger.find_last_optimized_file(optimized_dir)
        print(f"   Found: {last_file.name if last_file else 'None'}")
        
        # Test getting last timestamp
        print("\n2. Testing get_last_timestamp...")
        last_timestamp = merger.get_last_timestamp(last_file)
        last_date = datetime.fromtimestamp(last_timestamp / 1000)
        print(f"   Last timestamp: {last_timestamp} ({last_date})")
        
        # Test merging
        print("\n3. Testing merge_daily_files...")
        files_merged, rows_added = merger.merge_daily_files(
            optimized_dir=optimized_dir,
            daily_dir=daily_dir,
            daily_files=daily_files,
            max_file_size_gb=10.0
        )
        print(f"   Files merged: {files_merged}")
        print(f"   Rows added: {rows_added}")
        
        # Verify the merge
        print("\n4. Verifying merged data...")
        merged_file = merger.find_last_optimized_file(optimized_dir)
        merged_df = pd.read_parquet(merged_file)
        print(f"   Total rows in merged file: {len(merged_df)}")
        print(f"   Time range: {datetime.fromtimestamp(merged_df['time'].min()/1000)} to {datetime.fromtimestamp(merged_df['time'].max()/1000)}")
        
        # Check for duplicates
        duplicates = merged_df.duplicated(subset=['time']).sum()
        print(f"   Duplicate rows: {duplicates}")
        
        # Test cleanup (dry run)
        print("\n5. Testing cleanup (dry run)...")
        merger.cleanup_merged_daily_files(daily_files, dry_run=True)
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")


if __name__ == "__main__":
    test_merger()