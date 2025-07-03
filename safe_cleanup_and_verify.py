#!/usr/bin/env python3
"""
Safely clean up corrupted data with verification before deleting source files
"""
import os
import shutil
from pathlib import Path
import json
import pandas as pd
import pyarrow.parquet as pq
import zipfile
from datetime import datetime

def verify_zip_contents():
    """Verify ZIP files and their extraction status"""
    raw_dir = Path("datasets/dataset-raw-monthly/spot")
    
    print("🔍 Verifying ZIP files and extraction status...")
    print("="*50)
    
    zip_files = sorted(raw_dir.glob("*.zip"))
    csv_files = sorted(raw_dir.glob("*.csv"))
    
    zip_status = []
    
    for zip_file in zip_files:
        # Extract expected CSV name
        expected_csv = zip_file.with_suffix('.csv')
        
        # Check if CSV exists
        csv_exists = expected_csv.exists()
        
        # Verify ZIP integrity
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                # Test ZIP file integrity
                test_result = zf.testzip()
                zip_ok = test_result is None
                
                # Get ZIP contents
                namelist = zf.namelist()
                csv_in_zip = [f for f in namelist if f.endswith('.csv')]
        except Exception as e:
            zip_ok = False
            csv_in_zip = []
        
        status = {
            'zip_file': zip_file.name,
            'zip_ok': zip_ok,
            'csv_exists': csv_exists,
            'csv_in_zip': len(csv_in_zip) > 0,
            'can_delete_zip': zip_ok and csv_exists
        }
        zip_status.append(status)
    
    # Summary
    print(f"\n📦 Found {len(zip_files)} ZIP files")
    print(f"📄 Found {len(csv_files)} CSV files")
    
    can_delete = [s for s in zip_status if s['can_delete_zip']]
    cannot_delete = [s for s in zip_status if not s['can_delete_zip']]
    
    print(f"\n✅ {len(can_delete)} ZIP files can be safely deleted (CSV extracted)")
    print(f"⚠️  {len(cannot_delete)} ZIP files should be kept (CSV missing or ZIP corrupted)")
    
    if cannot_delete:
        print("\n❌ ZIP files that need attention:")
        for status in cannot_delete[:5]:
            print(f"   - {status['zip_file']}: ", end="")
            if not status['zip_ok']:
                print("ZIP corrupted")
            elif not status['csv_exists']:
                print("CSV not extracted")
        if len(cannot_delete) > 5:
            print(f"   ... and {len(cannot_delete) - 5} more")
    
    return zip_status

def verify_parquet_files():
    """Check parquet files for corruption and timestamp issues"""
    compressed_dir = Path("datasets/dataset-raw-monthly-compressed/spot")
    
    print("\n🔍 Analyzing existing parquet files...")
    print("="*50)
    
    parquet_files = sorted(compressed_dir.glob("BTCUSDT-Trades-*.parquet"))
    
    if not parquet_files:
        print("No parquet files found")
        return []
    
    print(f"Found {len(parquet_files)} parquet files")
    
    file_analysis = []
    corrupted_count = 0
    
    for pf in parquet_files[:10]:  # Check first 10 files
        try:
            # Read sample data
            df_sample = pd.read_parquet(pf, columns=['time'], engine='pyarrow')
            
            # Convert timestamp
            df_sample['datetime'] = pd.to_datetime(df_sample['time'], unit='ms')
            
            # Get date range
            date_min = df_sample['datetime'].min()
            date_max = df_sample['datetime'].max()
            
            # Check if corrupted (1970 dates)
            is_corrupted = date_min.year == 1970
            
            if is_corrupted:
                corrupted_count += 1
            
            analysis = {
                'file': pf.name,
                'rows': len(df_sample),
                'date_range': f"{date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}",
                'corrupted': is_corrupted
            }
            file_analysis.append(analysis)
            
            print(f"  {pf.name}: {analysis['date_range']} {'❌ CORRUPTED' if is_corrupted else '✅'}")
            
        except Exception as e:
            print(f"  {pf.name}: ❌ Error reading file: {e}")
            corrupted_count += 1
    
    if len(parquet_files) > 10:
        print(f"  ... and {len(parquet_files) - 10} more files")
    
    print(f"\n📊 Summary: {corrupted_count}/{min(10, len(parquet_files))} files checked are corrupted")
    
    return file_analysis

def create_safe_cleanup_plan():
    """Create a safe cleanup plan with verification"""
    
    print("\n🛡️ Safe Cleanup Plan")
    print("="*50)
    
    # 1. Check ZIP files
    zip_status = verify_zip_contents()
    
    # 2. Check parquet files  
    parquet_analysis = verify_parquet_files()
    
    # 3. Load progress tracking
    progress_file = Path("datasets/download_progress_BTCUSDT_spot_monthly.json")
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        print(f"\n📊 Progress tracking shows:")
        print(f"   - Downloaded: {len(progress.get('downloaded', []))} dates")
        print(f"   - Processed: {len(progress.get('processed', []))} dates")
    
    # 4. Recommendations
    print("\n💡 Recommended Actions:")
    print("="*50)
    
    print("\n1. DO NOT DELETE ZIP FILES YET")
    print("   - Keep all ZIP files until data is properly processed")
    
    print("\n2. Remove corrupted parquet files:")
    print("   python -c \"from pathlib import Path; [f.unlink() for f in Path('datasets/dataset-raw-monthly-compressed/spot').glob('*.parquet')]\"")
    
    print("\n3. Reset progress tracking (keep download history):")
    print("   - This will mark files as needing reprocessing")
    
    print("\n4. Re-run download to process with correct naming:")
    print("   python main.py download")
    print("   - Will skip downloading (ZIPs already exist)")
    print("   - Will extract and process with date-based names")
    
    print("\n5. After successful processing, verify new parquet files")
    
    print("\n6. ONLY THEN delete ZIP files if desired")
    
    # Create action script
    print("\n📝 Creating safe_cleanup_actions.sh script...")
    
    with open("safe_cleanup_actions.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Safe cleanup script - run each step manually\n\n")
        
        f.write("# Step 1: Backup progress file\n")
        f.write("cp datasets/download_progress_BTCUSDT_spot_monthly.json datasets/download_progress_BTCUSDT_spot_monthly.json.backup\n\n")
        
        f.write("# Step 2: Remove corrupted parquet files\n")
        f.write("echo 'Removing corrupted parquet files...'\n")
        f.write("rm -f datasets/dataset-raw-monthly-compressed/spot/BTCUSDT-Trades-*.parquet\n\n")
        
        f.write("# Step 3: Reset progress tracking\n")
        f.write("echo 'Resetting progress tracking...'\n")
        f.write("""python -c "
import json
with open('datasets/download_progress_BTCUSDT_spot_monthly.json', 'r') as f:
    progress = json.load(f)
progress['processed'] = []
progress['processing_failed'] = []
with open('datasets/download_progress_BTCUSDT_spot_monthly.json', 'w') as f:
    json.dump(progress, f, indent=2)
print('Progress tracking reset')
"\n\n""")
        
        f.write("# Step 4: Show next steps\n")
        f.write("echo 'Next: Run python main.py download to reprocess with correct naming'\n")
        f.write("echo 'DO NOT delete ZIP files until processing is verified!'\n")
    
    os.chmod("safe_cleanup_actions.sh", 0o755)
    print("✅ Created safe_cleanup_actions.sh")

if __name__ == "__main__":
    create_safe_cleanup_plan()