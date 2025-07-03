#!/usr/bin/env python3
"""
Quick integrity test script for spot and futures compressed-optimized parquet files.
Provides fast validation with minimal resource usage.
"""

import sys
from pathlib import Path
import pyarrow.parquet as pq
from typing import Tuple, List
from concurrent.futures import ThreadPoolExecutor
import time


def quick_validate_file(file_path: Path) -> Tuple[bool, str]:
    """Quick validation of a single parquet file."""
    try:
        # Check file exists and has reasonable size
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"
        
        size = file_path.stat().st_size
        if size < 1000:  # Less than 1KB
            return False, f"File too small ({size} bytes): {file_path}"
        
        # Try to read metadata
        pf = pq.ParquetFile(file_path)
        metadata = pf.metadata
        
        # Basic checks
        if metadata.num_rows == 0:
            return False, f"No rows in file: {file_path}"
        
        if metadata.num_columns < 4:  # time, price, qty, is_buyer_maker
            return False, f"Missing columns ({metadata.num_columns} found): {file_path}"
        
        # Quick schema check
        schema = pf.schema
        required_columns = {'time', 'price', 'qty', 'is_buyer_maker'}
        actual_columns = {field.name for field in schema}
        
        if not required_columns.issubset(actual_columns):
            missing = required_columns - actual_columns
            return False, f"Missing columns {missing}: {file_path}"
        
        return True, f"OK: {file_path.name} ({metadata.num_rows:,} rows)"
        
    except Exception as e:
        return False, f"Error reading {file_path}: {str(e)}"


def find_data_directories(base_path: Path) -> Tuple[List[Path], List[Path]]:
    """Find spot and futures compressed-optimized directories."""
    spot_dirs = []
    futures_dirs = []
    
    # Look for compressed-optimized directories
    for pattern in ['*-compressed-optimized', '*-monthly-compressed-optimized', '*-daily-compressed-optimized']:
        for dir_path in base_path.glob(pattern):
            if dir_path.is_dir():
                # Classify as spot or futures
                dir_str = str(dir_path).lower()
                if 'futures' in dir_str or '/um/' in dir_str:
                    futures_dirs.append(dir_path)
                else:
                    spot_dirs.append(dir_path)
    
    return spot_dirs, futures_dirs


def test_directory(directory: Path, max_workers: int = 4) -> Tuple[int, int, List[str]]:
    """Test all parquet files in a directory."""
    parquet_files = list(directory.glob('*.parquet'))
    passed = 0
    failed = 0
    errors = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(quick_validate_file, parquet_files)
        
        for success, message in results:
            if success:
                passed += 1
            else:
                failed += 1
                errors.append(message)
    
    return passed, failed, errors


def main():
    base_path = Path('.')
    
    print("=" * 60)
    print("QUICK PARQUET INTEGRITY TEST")
    print("=" * 60)
    
    # Find directories
    spot_dirs, futures_dirs = find_data_directories(base_path)
    
    print(f"\nFound {len(spot_dirs)} spot directories and {len(futures_dirs)} futures directories")
    
    total_passed = 0
    total_failed = 0
    all_errors = []
    
    # Test spot data
    if spot_dirs:
        print("\n--- TESTING SPOT DATA ---")
        for directory in spot_dirs:
            print(f"\nTesting: {directory}")
            start_time = time.time()
            passed, failed, errors = test_directory(directory)
            elapsed = time.time() - start_time
            
            total_passed += passed
            total_failed += failed
            all_errors.extend(errors)
            
            print(f"  ✓ Passed: {passed}")
            print(f"  ✗ Failed: {failed}")
            print(f"  Time: {elapsed:.1f}s")
            
            if errors and len(errors) <= 5:
                print("  Errors:")
                for error in errors:
                    print(f"    - {error}")
            elif errors:
                print(f"  First 5 errors (out of {len(errors)}):")
                for error in errors[:5]:
                    print(f"    - {error}")
    
    # Test futures data
    if futures_dirs:
        print("\n--- TESTING FUTURES DATA ---")
        for directory in futures_dirs:
            print(f"\nTesting: {directory}")
            start_time = time.time()
            passed, failed, errors = test_directory(directory)
            elapsed = time.time() - start_time
            
            total_passed += passed
            total_failed += failed
            all_errors.extend(errors)
            
            print(f"  ✓ Passed: {passed}")
            print(f"  ✗ Failed: {failed}")
            print(f"  Time: {elapsed:.1f}s")
            
            if errors and len(errors) <= 5:
                print("  Errors:")
                for error in errors:
                    print(f"    - {error}")
            elif errors:
                print(f"  First 5 errors (out of {len(errors)}):")
                for error in errors[:5]:
                    print(f"    - {error}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files tested: {total_passed + total_failed}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    
    if total_passed + total_failed > 0:
        pass_rate = (total_passed / (total_passed + total_failed)) * 100
        print(f"Pass rate: {pass_rate:.1f}%")
    
    if total_failed > 0:
        print(f"\n⚠️  {total_failed} files failed validation!")
        sys.exit(1)
    else:
        print("\n✅ All files passed validation!")
        sys.exit(0)


if __name__ == '__main__':
    main()