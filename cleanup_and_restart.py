#!/usr/bin/env python3
"""
Clean up corrupted data and reset for fresh download with proper naming
"""
import os
import shutil
from pathlib import Path
import json

def cleanup_corrupted_data():
    """Remove corrupted parquet files and reset progress tracking"""
    
    print("🧹 Bitcoin Data Cleanup Script")
    print("="*50)
    
    # Define paths
    compressed_dir = Path("datasets/dataset-raw-monthly-compressed/spot")
    progress_file = Path("datasets/download_progress_BTCUSDT_spot_monthly.json")
    
    # 1. Remove corrupted parquet files
    if compressed_dir.exists():
        parquet_files = list(compressed_dir.glob("BTCUSDT-Trades-*.parquet"))
        
        if parquet_files:
            print(f"\n📦 Found {len(parquet_files)} parquet files to remove:")
            for f in parquet_files[:5]:
                print(f"   - {f.name}")
            if len(parquet_files) > 5:
                print(f"   ... and {len(parquet_files) - 5} more")
            
            confirm = input("\n⚠️  Remove all corrupted parquet files? (yes/no): ").strip().lower()
            
            if confirm == 'yes':
                for f in parquet_files:
                    f.unlink()
                print(f"✅ Removed {len(parquet_files)} corrupted parquet files")
            else:
                print("❌ Cleanup cancelled")
                return
    
    # 2. Reset progress tracking
    if progress_file.exists():
        print(f"\n📊 Current progress tracking:")
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        print(f"   - Downloaded: {len(progress.get('downloaded', []))} dates")
        print(f"   - Processed: {len(progress.get('processed', []))} dates")
        
        # Keep downloaded list but clear processed
        new_progress = {
            'downloaded': progress.get('downloaded', []),
            'processed': [],  # Clear this since parquet files are corrupted
            'failed': progress.get('failed', []),
            'processing_failed': [],
            'last_update': None
        }
        
        confirm = input("\n⚠️  Reset progress tracking (keep download list)? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            with open(progress_file, 'w') as f:
                json.dump(new_progress, f, indent=2)
            print("✅ Reset progress tracking")
        else:
            print("❌ Progress tracking not modified")
            return
    
    print("\n✅ Cleanup complete!")
    print("\n📌 Next steps:")
    print("1. Run the download again to reprocess CSV files with proper naming:")
    print("   python main.py download")
    print("2. The downloader will skip already downloaded files and process them with correct dates")
    print("3. Once complete, run optimization:")
    print("   python main.py optimize --source datasets/dataset-raw-monthly-compressed/spot \\")
    print("                           --target data/optimized/spot \\")
    print("                           --max-size 10")

if __name__ == "__main__":
    cleanup_corrupted_data()