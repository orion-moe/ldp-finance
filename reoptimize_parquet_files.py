#!/usr/bin/env python3
"""
Re-optimize existing parquet files to properly reach 10GB target size.
This fixes the issue where files were created smaller than intended due to incorrect size calculations.
"""

import sys
from pathlib import Path
import pyarrow.parquet as pq
import pandas as pd
import pyarrow as pa
from datetime import datetime
import shutil
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_file_info(file_path: Path) -> dict:
    """Get file information including size and row count"""
    try:
        size_gb = file_path.stat().st_size / (1024**3)
        # Get row count without loading full file
        parquet_file = pq.ParquetFile(file_path)
        row_count = parquet_file.metadata.num_rows
        return {
            'path': file_path,
            'size_gb': size_gb,
            'rows': row_count
        }
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return None

def reoptimize_directory(data_dir: Path, target_size_gb: float = 10.0, dry_run: bool = False):
    """Re-optimize parquet files in a directory to reach target size"""
    
    # Find all optimized parquet files
    pattern = "*-Trades-Optimized-*.parquet"
    files = sorted(data_dir.glob(pattern))
    
    if not files:
        logger.info(f"No optimized parquet files found in {data_dir}")
        return
    
    logger.info(f"Found {len(files)} optimized parquet files")
    
    # Get info for all files
    file_infos = []
    total_size = 0
    total_rows = 0
    
    for f in files:
        info = get_file_info(f)
        if info:
            file_infos.append(info)
            total_size += info['size_gb']
            total_rows += info['rows']
            logger.info(f"  {f.name}: {info['size_gb']:.2f} GB, {info['rows']:,} rows")
    
    logger.info(f"\nTotal: {total_size:.2f} GB, {total_rows:,} rows")
    
    # Check if re-optimization is needed
    avg_size = total_size / len(file_infos) if file_infos else 0
    if avg_size >= target_size_gb * 0.9:
        logger.info("Files are already close to target size. No re-optimization needed.")
        return
    
    # Calculate new file distribution
    estimated_files_needed = int(total_size / (target_size_gb * 0.95)) + 1  # 95% of target
    logger.info(f"\nRe-optimization plan:")
    logger.info(f"  Current files: {len(file_infos)}")
    logger.info(f"  Target files: {estimated_files_needed}")
    logger.info(f"  Target size per file: ~{target_size_gb * 0.95:.1f} GB")
    
    if dry_run:
        logger.info("\nDRY RUN - No changes will be made")
        return
    
    # Confirm with user
    confirm = input(f"\nProceed with re-optimization? (yes/no): ").strip().lower()
    if confirm != 'yes':
        logger.info("Re-optimization cancelled")
        return
    
    # Create backup directory
    backup_dir = data_dir.parent / f"{data_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(exist_ok=True)
    logger.info(f"\nBacking up files to: {backup_dir}")
    
    # Backup all files first
    for info in file_infos:
        src = info['path']
        dst = backup_dir / src.name
        logger.info(f"  Backing up {src.name}...")
        shutil.copy2(src, dst)
    
    # Get symbol from first file name
    first_file = file_infos[0]['path'].name
    symbol = first_file.split('-Trades-')[0]
    
    logger.info(f"\nRe-optimizing files for {symbol}...")
    
    # Process files in order and create new optimized files
    current_data = []
    current_size_gb = 0
    output_file_num = 1
    files_created = 0
    
    for i, info in enumerate(file_infos):
        logger.info(f"Processing {info['path'].name}...")
        
        # Read the file
        df = pd.read_parquet(info['path'])
        
        # Estimate compressed size (conservative 3x compression)
        memory_size_gb = df.memory_usage(deep=True).sum() / (1024**3)
        compressed_size_gb = memory_size_gb / 3.0
        
        # Check if adding this would exceed target
        if current_data and (current_size_gb + compressed_size_gb) > target_size_gb * 0.95:
            # Write current batch
            output_path = data_dir / f"{symbol}-Trades-Optimized-{output_file_num:03d}.parquet.new"
            
            logger.info(f"  Writing {output_path.name} with {sum(len(d) for d in current_data):,} rows...")
            
            # Combine and sort data
            combined_df = pd.concat(current_data, ignore_index=True)
            combined_df = combined_df.sort_values('time').reset_index(drop=True)
            
            # Write to new file
            table = pa.Table.from_pandas(combined_df)
            pq.write_table(table, output_path, compression='snappy')
            
            # Check actual size
            actual_size_gb = output_path.stat().st_size / (1024**3)
            logger.info(f"  Created: {actual_size_gb:.2f} GB")
            
            files_created += 1
            output_file_num += 1
            current_data = []
            current_size_gb = 0
        
        # Add to current batch
        current_data.append(df)
        current_size_gb += compressed_size_gb
    
    # Write any remaining data
    if current_data:
        output_path = data_dir / f"{symbol}-Trades-Optimized-{output_file_num:03d}.parquet.new"
        
        logger.info(f"  Writing final file {output_path.name} with {sum(len(d) for d in current_data):,} rows...")
        
        combined_df = pd.concat(current_data, ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        table = pa.Table.from_pandas(combined_df)
        pq.write_table(table, output_path, compression='snappy')
        
        actual_size_gb = output_path.stat().st_size / (1024**3)
        logger.info(f"  Created: {actual_size_gb:.2f} GB")
        files_created += 1
    
    # Replace old files with new ones
    logger.info(f"\nReplacing old files with new optimized files...")
    
    # First, delete old files
    for info in file_infos:
        info['path'].unlink()
        logger.info(f"  Deleted {info['path'].name}")
    
    # Then rename new files
    for new_file in sorted(data_dir.glob("*.parquet.new")):
        final_name = new_file.with_suffix('')  # Remove .new
        new_file.rename(final_name)
        logger.info(f"  Renamed {new_file.name} to {final_name.name}")
    
    logger.info(f"\n✅ Re-optimization completed!")
    logger.info(f"  Files created: {files_created}")
    logger.info(f"  Backup location: {backup_dir}")
    
    # Show new file sizes
    logger.info(f"\nNew file sizes:")
    new_files = sorted(data_dir.glob(pattern))
    for f in new_files:
        size_gb = f.stat().st_size / (1024**3)
        logger.info(f"  {f.name}: {size_gb:.2f} GB")

def main():
    print("\n🔧 Parquet File Re-Optimizer")
    print("=" * 50)
    print("This tool re-optimizes existing parquet files to reach the 10GB target size")
    print("It will create a backup before making any changes")
    
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
    
    # Determine directory
    base_dir = Path("datasets")
    data_dir = base_dir / "dataset-raw-daily-compressed-optimized" / market_type
    
    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return
    
    print(f"\n📁 Directory: {data_dir}")
    
    # Ask for target size
    target_input = input("\nTarget file size in GB (default: 10): ").strip()
    target_size = float(target_input) if target_input else 10.0
    
    # Ask for dry run
    dry_run = input("\nDry run? (yes/no, default: no): ").strip().lower() == 'yes'
    
    # Run re-optimization
    reoptimize_directory(data_dir, target_size, dry_run)

if __name__ == "__main__":
    main()