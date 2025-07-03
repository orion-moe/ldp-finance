#!/bin/bash
# Safe cleanup script - run each step manually

# Step 1: Backup progress file
cp datasets/download_progress_BTCUSDT_spot_monthly.json datasets/download_progress_BTCUSDT_spot_monthly.json.backup

# Step 2: Remove corrupted parquet files
echo 'Removing corrupted parquet files...'
rm -f datasets/dataset-raw-monthly-compressed/spot/BTCUSDT-Trades-*.parquet

# Step 3: Reset progress tracking
echo 'Resetting progress tracking...'
python -c "
import json
with open('datasets/download_progress_BTCUSDT_spot_monthly.json', 'r') as f:
    progress = json.load(f)
progress['processed'] = []
progress['processing_failed'] = []
with open('datasets/download_progress_BTCUSDT_spot_monthly.json', 'w') as f:
    json.dump(progress, f, indent=2)
print('Progress tracking reset')
"

# Step 4: Show next steps
echo 'Next: Run python main.py download to reprocess with correct naming'
echo 'DO NOT delete ZIP files until processing is verified!'
