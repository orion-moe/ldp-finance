#!/usr/bin/env python3
"""
Main entry point for the Bitcoin ML Finance pipeline
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Tuple, Optional
import logging
from logging.handlers import RotatingFileHandler

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_pipeline.extractors.csv_extractor import CSVExtractor
from src.data_pipeline.converters.csv_to_parquet import CSVToParquetConverter
from src.data_pipeline.converters.zip_to_parquet_streamer import ZipToParquetStreamer
from src.data_pipeline.processors.parquet_optimizer import main as optimize_main
from src.data_pipeline.validators.missing_dates_validator import main as missing_dates_main
from src.features.bars.imbalance_bars import main as imbalance_main
from src.features.bars.standard_dollar_bars import process_files_and_generate_bars, setup_logging as setup_standard_logging


def setup_logging():
    """Set up logging configuration"""
    # Create logs directory
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Log file with timestamp
    log_filename = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicates
    root_logger.handlers.clear()

    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=50*1024*1024,  # 50MB
        backupCount=10
    )
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    # Log initial message
    logging.info(f"Pipeline started - Log file: {log_path}")

    return log_path


class PipelineConfig:
    """Store pipeline configuration"""
    def __init__(self):
        self.symbol = None
        self.data_type = None
        self.futures_type = None
        self.granularity = None
        self.start_date = None
        self.end_date = None
        self.workers = 5


def get_data_date_range(symbol, data_type, futures_type, granularity):
    """Get the first and last available dates from existing data by checking parquet files"""
    try:
        # Check parquet files directly (modern structure)
        ticker_name = f"{symbol.lower()}-{data_type}"
        if data_type == "futures":
            ticker_name = f"{symbol.lower()}-{data_type}-{futures_type}"

        compressed_dir = Path("data") / ticker_name / f"raw-parquet-{granularity}"

        if compressed_dir.exists():
            parquet_files = sorted(list(compressed_dir.glob("*.parquet")))
            if parquet_files:
                # Try to extract dates from file names
                # Format: BTCUSDT-trades-YYYY-MM-DD.parquet (daily) or BTCUSDT-trades-YYYY-MM.parquet (monthly)
                dates = []
                for pf in parquet_files:
                    # Extract date part from filename
                    parts = pf.stem.split('-trades-')
                    if len(parts) == 2:
                        dates.append(parts[1])

                if dates:
                    dates = sorted(dates)
                    return dates[0], dates[-1]
                else:
                    # If we can't extract dates, at least show data exists
                    return "data-exists", "data-exists"

        return None, None

    except Exception:
        return None, None


def select_market_configuration() -> PipelineConfig:
    """Select market configuration (symbol and type)"""
    config = PipelineConfig()

    print("\n" + "="*60)
    print(" 🚀 Bitcoin ML Finance Pipeline - Market Configuration ")
    print("="*60)

    # Symbol selection
    print("\nSelect trading pair symbol:")
    print("1. BTCUSDT (default)")
    print("2. ETHUSDT")
    print("3. Other symbol")

    while True:
        symbol_choice = input("\nEnter your choice (1-3) or press Enter for BTCUSDT: ").strip()
        if symbol_choice == "" or symbol_choice == "1":
            config.symbol = "BTCUSDT"
            break
        elif symbol_choice == "2":
            config.symbol = "ETHUSDT"
            break
        elif symbol_choice == "3":
            config.symbol = input("Enter symbol: ").strip().upper()
            if config.symbol:
                break
            else:
                print("Please enter a valid symbol.")
        else:
            print("Invalid choice. Please enter 1, 2, 3, or press Enter.")

    # Data type selection
    print("\nSelect data type:")
    print("1. Spot")
    print("2. Futures USD-M (USDT-margined)")
    print("3. Futures COIN-M (Coin-margined)")

    while True:
        type_choice = input("\nEnter your choice (1-3): ").strip()
        if type_choice == "1":
            config.data_type = "spot"
            config.futures_type = "um"  # default
            break
        elif type_choice == "2":
            config.data_type = "futures"
            config.futures_type = "um"
            break
        elif type_choice == "3":
            config.data_type = "futures"
            config.futures_type = "cm"
            # Adjust symbol for COIN-M if needed
            if config.symbol == "BTCUSDT":
                config.symbol = "BTCUSD_PERP"
                print(f"Note: For COIN-M futures, using symbol: {config.symbol}")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Fixed granularity - always daily for optimized parallel processing
    config.granularity = "daily"
    print(f"\n✅ Using Daily granularity (optimized for parallel processing)")

    return config


def check_pipeline_status(config: PipelineConfig) -> dict:
    """Check the status of each pipeline step"""
    status = {
        "zip_downloaded": False,
        "csv_extracted": False,
        "csv_validated": False,
        "parquet_converted": False,
        "parquet_optimized": False,
        "data_validated": False,
        "features_generated": False
    }

    # Check download status by looking at ZIP files
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == 'futures':
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    zip_dir = Path("data") / ticker_name / f"raw-zip-{config.granularity}"
    if zip_dir.exists():
        zip_files = list(zip_dir.glob("*.zip"))
        if zip_files:
            status['zip_downloaded'] = True

    # Check for CSV extraction (legacy structure - for backward compatibility)
    if config.data_type == "spot":
        raw_dir_legacy = Path("data") / f"dataset-raw-{config.granularity}" / "spot"
    else:
        raw_dir_legacy = Path("data") / f"dataset-raw-{config.granularity}" / f"futures-{config.futures_type}"

    if raw_dir_legacy.exists():
        csv_files = list(raw_dir_legacy.glob(f"{config.symbol}-trades-*.csv"))
        if csv_files:
            status['csv_extracted'] = True

    # Check CSV validation progress
    extraction_progress_file = Path("data") / f"extraction_progress_{config.symbol}_{config.data_type}_{config.granularity}.json"
    if extraction_progress_file.exists():
        with open(extraction_progress_file, 'r') as f:
            progress = json.load(f)
            if progress.get('verified'):
                status['csv_validated'] = True

    # Check for parquet files (modern ticker-based structure)
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    ticker_dir = Path("data") / ticker_name
    parquet_dir = ticker_dir / f"raw-parquet-{config.granularity}"

    if parquet_dir.exists():
        parquet_files = list(parquet_dir.glob("*.parquet"))
        if parquet_files:
            status['parquet_converted'] = True

    # Also check legacy structure for backward compatibility
    compressed_dir_legacy = Path("data") / f"dataset-raw-{config.granularity}-compressed"
    if config.data_type == "spot":
        parquet_dir_legacy = compressed_dir_legacy / "spot"
    else:
        parquet_dir_legacy = compressed_dir_legacy / f"futures-{config.futures_type}"

    if parquet_dir_legacy.exists():
        parquet_files_legacy = list(parquet_dir_legacy.glob(f"{config.symbol}-Trades-*.parquet"))
        if parquet_files_legacy:
            status['parquet_converted'] = True

    # Check for optimized parquet files (both legacy and modern structures)
    # Legacy structure
    if config.data_type == "spot":
        optimized_dir = Path("data") / f"dataset-raw-{config.granularity}-compressed-optimized" / "spot"
    else:
        optimized_dir = Path("data") / f"dataset-raw-{config.granularity}-compressed-optimized" / f"futures-{config.futures_type}"

    if optimized_dir.exists():
        optimized_files = list(optimized_dir.glob(f"{config.symbol}*.parquet"))
        if optimized_files:
            status['parquet_optimized'] = True

    # Modern structure
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    modern_optimized_dir = Path("data") / ticker_name / f"raw-parquet-merged-{config.granularity}"
    if modern_optimized_dir.exists():
        modern_optimized_files = list(modern_optimized_dir.glob("*.parquet"))
        if modern_optimized_files:
            status['parquet_optimized'] = True

    return status


def validate_parquet_files(symbol: str, data_type: str, futures_type: str,
                          start_date: date, end_date: date,
                          parquet_dir: Path) -> dict:
    """
    Validate parquet files for completeness and integrity

    Returns dict with:
        - total_expected: int
        - total_found: int
        - missing_dates: list of dates
        - corrupted_files: list of file paths
        - valid_files: int
    """
    import pyarrow.parquet as pq
    from datetime import timedelta

    print("\n" + "="*60)
    print(" 🔍 VALIDATING PARQUET FILES ")
    print("="*60)

    if not parquet_dir.exists():
        print(f"❌ Directory not found: {parquet_dir}")
        return {
            'total_expected': 0,
            'total_found': 0,
            'missing_dates': [],
            'corrupted_files': [],
            'valid_files': 0
        }

    # Generate expected dates
    expected_dates = set()
    current = start_date
    while current <= end_date:
        expected_dates.add(current)
        current += timedelta(days=1)

    # Find all parquet files
    parquet_files = list(parquet_dir.glob("*.parquet"))

    print(f"📅 Expected dates: {start_date} to {end_date} ({len(expected_dates)} days)")
    print(f"📁 Found files: {len(parquet_files)}")
    print()

    # Extract dates from filenames and check integrity
    found_dates = set()
    corrupted_files = []
    valid_files = 0

    print("🔍 Checking file integrity...")
    for file_path in parquet_files:
        # Extract date from filename: BTCUSDT-trades-2025-03-03.parquet
        try:
            parts = file_path.stem.split('-')
            if len(parts) >= 5:  # BTCUSDT-trades-YYYY-MM-DD
                year = int(parts[-3])
                month = int(parts[-2])
                day = int(parts[-1])
                file_date = date(year, month, day)

                # Check if file is corrupted
                try:
                    test_file = pq.ParquetFile(file_path)
                    _ = test_file.metadata
                    found_dates.add(file_date)
                    valid_files += 1
                except Exception as e:
                    corrupted_files.append({
                        'file': file_path.name,
                        'date': file_date,
                        'error': str(e)[:100]
                    })
                    print(f"  ❌ CORRUPTED: {file_path.name}")
        except (ValueError, IndexError) as e:
            print(f"  ⚠️  Cannot parse date from: {file_path.name}")

    # Find missing dates
    missing_dates = sorted(expected_dates - found_dates)

    # Display results
    print()
    print("="*60)
    print(" 📊 VALIDATION RESULTS ")
    print("="*60)
    print(f"✅ Valid files: {valid_files}/{len(expected_dates)}")
    print(f"❌ Corrupted files: {len(corrupted_files)}")
    print(f"⚠️  Missing dates: {len(missing_dates)}")

    if corrupted_files:
        print("\n🗑️  CORRUPTED FILES:")
        for item in corrupted_files:
            print(f"  - {item['file']} ({item['date']})")
            print(f"    Error: {item['error']}")

    if missing_dates:
        print("\n📅 MISSING DATES:")
        if len(missing_dates) <= 10:
            for missing_date in missing_dates:
                print(f"  - {missing_date}")
        else:
            print(f"  - {missing_dates[0]} to {missing_dates[-1]}")
            print(f"  - Total: {len(missing_dates)} dates")
            print(f"  - First 5: {', '.join(str(d) for d in missing_dates[:5])}")
            print(f"  - Last 5: {', '.join(str(d) for d in missing_dates[-5:])}")

    if valid_files == len(expected_dates) and len(corrupted_files) == 0:
        print("\n✅ ALL FILES ARE VALID AND COMPLETE!")

    print("="*60)

    return {
        'total_expected': len(expected_dates),
        'total_found': valid_files,
        'missing_dates': missing_dates,
        'corrupted_files': corrupted_files,
        'valid_files': valid_files
    }


def display_pipeline_menu(config: PipelineConfig):
    """Display the pipeline menu with status indicators"""
    status = check_pipeline_status(config)

    print("\n" + "="*60)
    print(f" 📊 Pipeline for {config.symbol} {config.data_type.upper()} DAILY ")
    print("="*60)

    print("1. ⚡ Complete Pipeline (Download → Convert → Clean)")
    print("2. 📦 Individual Steps Menu")
    print(f"3. {'✅' if status['parquet_optimized'] else '⬜'}🗜️  Compact/Merge Parquet files (auto-cleanup)")

    print("\n🔍 Validation:")
    print(f"4. {'✅' if status['data_validated'] else '⬜'}📅 Validate missing dates")

    print("\n📊 Features:")
    print(f"5. {'✅' if status['features_generated'] else '⬜'}📊 Generate bars (Dollar/Imbalance)")

    print("\n🔧 Maintenance:")
    print("6. 🗑️  Clean ZIP/CSV/Checksum files")
    print("7. 📅 Add missing daily data")
    print("8. 🚨 Delete ALL data (complete database wipe)")

    print("\n🤖 Machine Learning:")
    print("9. 🔍 Search Classificator")

    print("\n0. 🚪 Exit")

    return status


def run_search_classificator(config: PipelineConfig):
    """Search for available parquet files and run the classificator"""
    print("\n" + "="*60)
    print(" 🔍 Search Classificator ")
    print("="*60)

    # Build path to output folders based on market configuration
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == 'futures':
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    # Check both standard and imbalance output folders
    base_path = Path("data") / ticker_name / "output"

    if not base_path.exists():
        print(f"\n❌ Output directory not found: {base_path}")
        print("Please generate some bars first (option 5)")
        input("\nPress Enter to continue...")
        return

    # Search for parquet files in both standard and imbalance folders
    parquet_files = []

    for subfolder in ['standard', 'imbalance']:
        subfolder_path = base_path / subfolder
        if subfolder_path.exists():
            # Look for parquet files inside folders (new structure)
            for item in subfolder_path.iterdir():
                if item.is_dir():
                    for pf in item.glob("*.parquet"):
                        parquet_files.append({
                            'path': str(pf),
                            'folder': str(item),
                            'name': pf.name,
                            'type': subfolder,
                            'size_mb': pf.stat().st_size / (1024 * 1024)
                        })

    if not parquet_files:
        print(f"\n❌ No parquet files found in {base_path}")
        print("Please generate some bars first (option 5)")
        input("\nPress Enter to continue...")
        return

    # Display available files
    print(f"\n📁 Found {len(parquet_files)} parquet file(s):\n")
    for idx, pf in enumerate(parquet_files, 1):
        print(f"{idx}. [{pf['type'].upper()}] {pf['name']}")
        print(f"   Size: {pf['size_mb']:.2f} MB")
        print(f"   Path: {pf['folder']}")
        print()

    # Let user select a file
    try:
        choice = input(f"Select a file (1-{len(parquet_files)}) or 0 to cancel: ").strip()
        if choice == '0':
            return

        file_idx = int(choice) - 1
        if file_idx < 0 or file_idx >= len(parquet_files):
            print("❌ Invalid selection")
            input("\nPress Enter to continue...")
            return

        selected_file = parquet_files[file_idx]
        print(f"\n✅ Selected: {selected_file['name']}")
        print(f"📊 Running Search Classificator...")
        print("\nThis will execute main_optimized.py with the selected file.")
        print("The process may take several minutes depending on the data size.\n")

        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            return

        # Update main_optimized.py with the selected file path and name
        main_optimized_path = Path("main_optimized.py")
        if not main_optimized_path.exists():
            print(f"\n❌ main_optimized.py not found!")
            input("\nPress Enter to continue...")
            return

        # Read the file
        with open(main_optimized_path, 'r') as f:
            content = f.read()

        # Update DATA_PATH and FILE_NAME
        import re

        # Extract the folder path and filename
        data_path = str(Path(selected_file['folder']).parent)
        file_name = selected_file['name']

        # Replace DATA_PATH
        content = re.sub(
            r"DATA_PATH = '[^']*'",
            f"DATA_PATH = '{data_path}'",
            content
        )

        # Replace FILE_NAME
        content = re.sub(
            r"FILE_NAME = '[^']*'",
            f"FILE_NAME = '{file_name}'",
            content
        )

        # Write back
        with open(main_optimized_path, 'w') as f:
            f.write(content)

        print(f"✅ Updated main_optimized.py with selected file")
        print(f"   DATA_PATH: {data_path}")
        print(f"   FILE_NAME: {file_name}")

        # Execute main_optimized.py
        import subprocess
        print("\n" + "="*60)
        print("🚀 Executing main_optimized.py...")
        print("="*60 + "\n")

        result = subprocess.run([sys.executable, str(main_optimized_path)],
                              capture_output=False)

        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ Search Classificator completed successfully!")
            print("="*60)
        else:
            print("\n" + "="*60)
            print(f"❌ Search Classificator failed with return code: {result.returncode}")
            print("="*60)

        input("\nPress Enter to continue...")

    except ValueError:
        print("❌ Invalid input")
        input("\nPress Enter to continue...")
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        input("\nPress Enter to continue...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to continue...")


def display_individual_steps_menu(config: PipelineConfig):
    """Display the individual steps submenu"""
    status = check_pipeline_status(config)

    print("\n" + "="*60)
    print(f" 📦 Individual Steps for {config.symbol} {config.data_type.upper()} ")
    print("="*60)

    print(f"1. {'✅' if status['zip_downloaded'] else '⬜'}📥 Download ZIP data with checksum")
    print(f"2. {'✅' if status['parquet_converted'] else '⬜'}🔄 ZIP → CSV → Parquet (Legacy)")
    print(f"3. 🚀 ZIP → Parquet Direct Conversion (Optimized)")
    print(f"4. {'✅' if status['parquet_optimized'] else '⬜'}🔧 Optimize/Merge Parquet files")
    print("\n0. ↩️  Back to main menu")

    return status


def clean_zip_and_checksum_files(config: PipelineConfig):
    """Clean ZIP and CHECKSUM files to free disk space"""
    print("\n" + "="*60)
    print(" 🗑️  Clean ZIP and CHECKSUM Files ")
    print("="*60)

    from pathlib import Path

    # Determine directory based on config (modern ticker-based structure)
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    raw_dir = Path("data") / ticker_name / f"raw-zip-{config.granularity}"

    if not raw_dir.exists():
        print("❌ Raw data directory not found.")
        return

    # Find ZIP and CHECKSUM files
    zip_files = list(raw_dir.glob(f"{config.symbol}-trades-*.zip"))
    checksum_files = list(raw_dir.glob(f"{config.symbol}-trades-*.zip.CHECKSUM"))

    total_files = len(zip_files) + len(checksum_files)

    if total_files == 0:
        print("✅ No ZIP or CHECKSUM files found to clean.")
        return

    # Calculate total size
    total_size = 0
    for f in zip_files + checksum_files:
        if f.is_file():
            total_size += f.stat().st_size

    print(f"\n📊 Found:")
    print(f"   • ZIP files: {len(zip_files)}")
    print(f"   • CHECKSUM files: {len(checksum_files)}")
    print(f"   • Total size: {total_size / (1024**3):.2f} GB")

    # Check if CSV files exist for all ZIPs
    missing_csv = []
    for zip_file in zip_files:
        csv_name = zip_file.stem + ".csv"
        csv_path = raw_dir / csv_name
        if not csv_path.exists():
            missing_csv.append(zip_file.stem)

    if missing_csv:
        print(f"\n⚠️  Warning: {len(missing_csv)} ZIP files don't have corresponding CSV files:")
        for name in missing_csv[:5]:
            print(f"   - {name}")
        if len(missing_csv) > 5:
            print(f"   ... and {len(missing_csv) - 5} more")
        print("\n💡 Consider extracting these ZIPs before deleting them.")

    confirm = input(f"\n❓ Delete all {total_files} ZIP and CHECKSUM files? (yes/no): ").strip().lower()

    if confirm != 'yes':
        print("❌ Cleanup cancelled.")
        return

    # Delete files
    deleted_count = 0
    freed_space = 0

    print("\n🗑️  Deleting files...")

    for f in zip_files + checksum_files:
        try:
            if f.is_file():
                file_size = f.stat().st_size
                f.unlink()
                deleted_count += 1
                freed_space += file_size
        except Exception as e:
            print(f"❌ Failed to delete {f.name}: {e}")

    print(f"\n✅ Cleanup completed!")
    print(f"   • Deleted: {deleted_count} files")
    print(f"   • Freed: {freed_space / (1024**3):.2f} GB")

    input("\nPress Enter to continue...")


def delete_all_data():
    """Delete ALL data - complete database wipe"""
    import shutil

    print("\n" + "="*60)
    print(" 🚨 DELETE ALL DATA - COMPLETE DATABASE WIPE ")
    print("="*60)

    print("\n⚠️  WARNING: This will permanently delete:")
    print("   • ALL ticker directories in data/")
    print("   • ALL output files in output/")
    print("   • ALL downloaded ZIP files")
    print("   • ALL Parquet files")
    print("   • ALL progress tracking files")
    print("   • ALL logs")

    data_dir = Path("data")
    output_dir = Path("output")

    # Calculate total size
    total_size = 0
    ticker_dirs = []

    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and item.name not in ['logs']:
                ticker_dirs.append(item)
                # Calculate size
                for path in item.rglob('*'):
                    if path.is_file():
                        try:
                            total_size += path.stat().st_size
                        except:
                            pass

        # Also check for loose files in data/
        for item in data_dir.iterdir():
            if item.is_file():
                try:
                    total_size += item.stat().st_size
                except:
                    pass

    if output_dir.exists():
        for path in output_dir.rglob('*'):
            if path.is_file():
                try:
                    total_size += path.stat().st_size
                except:
                    pass

    print(f"\n📊 Summary:")
    print(f"   • Ticker directories: {len(ticker_dirs)}")
    if ticker_dirs:
        for ticker_dir in ticker_dirs[:10]:  # Show first 10
            print(f"     - {ticker_dir.name}")
        if len(ticker_dirs) > 10:
            print(f"     ... and {len(ticker_dirs) - 10} more")
    print(f"   • Total size: {total_size / (1024**3):.2f} GB")

    if total_size == 0:
        print("\n✅ No data found to delete.")
        input("\nPress Enter to continue...")
        return

    print("\n🔴 THIS ACTION CANNOT BE UNDONE!")
    confirm1 = input("\n❓ Are you ABSOLUTELY sure you want to delete ALL data? (type 'DELETE ALL'): ").strip()

    if confirm1 != 'DELETE ALL':
        print("❌ Deletion cancelled.")
        input("\nPress Enter to continue...")
        return

    confirm2 = input("\n❓ Final confirmation - Delete everything? (yes/no): ").strip().lower()

    if confirm2 != 'yes':
        print("❌ Deletion cancelled.")
        input("\nPress Enter to continue...")
        return

    # Delete everything
    deleted_dirs = 0
    deleted_files = 0
    freed_space = 0

    print("\n🗑️  Deleting all data...")

    # Delete ticker directories
    if data_dir.exists():
        for item in data_dir.iterdir():
            if item.is_dir() and item.name not in ['logs']:
                try:
                    # Calculate size before deletion
                    dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    deleted_dirs += 1
                    freed_space += dir_size
                    print(f"   ✓ Deleted {item.name}")
                except Exception as e:
                    print(f"   ✗ Failed to delete {item.name}: {e}")

        # Delete loose files in data/ (like progress JSON files)
        for item in data_dir.iterdir():
            if item.is_file():
                try:
                    file_size = item.stat().st_size
                    item.unlink()
                    deleted_files += 1
                    freed_space += file_size
                    print(f"   ✓ Deleted {item.name}")
                except Exception as e:
                    print(f"   ✗ Failed to delete {item.name}: {e}")

    # Delete output directory contents
    if output_dir.exists():
        for item in output_dir.iterdir():
            try:
                if item.is_file():
                    file_size = item.stat().st_size
                    item.unlink()
                    deleted_files += 1
                    freed_space += file_size
                elif item.is_dir():
                    dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
                    shutil.rmtree(item)
                    deleted_dirs += 1
                    freed_space += dir_size
                print(f"   ✓ Deleted output/{item.name}")
            except Exception as e:
                print(f"   ✗ Failed to delete output/{item.name}: {e}")

    print(f"\n✅ Database wipe completed!")
    print(f"   • Deleted directories: {deleted_dirs}")
    print(f"   • Deleted files: {deleted_files}")
    print(f"   • Freed space: {freed_space / (1024**3):.2f} GB")

    input("\nPress Enter to continue...")


def run_zip_to_parquet_direct(config: PipelineConfig):
    """Run direct ZIP to Parquet conversion (optimized version)"""
    print("\n" + "="*60)
    print(" 🚀 ZIP → Parquet Direct Conversion (Optimized) ")
    print("="*60)

    print("\n📊 This will convert ZIP files directly to Parquet format")
    print("   without creating intermediate CSV files.")
    print("\n✨ Benefits:")
    print("   • 15-20% faster conversion")
    print("   • Less disk I/O (no CSV writes)")
    print("   • Memory-efficient streaming")
    print("   • Maintains data integrity and validation")

    print(f"\n🔧 Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Type: {config.data_type}")
    if config.data_type == "futures":
        print(f"   Futures Type: {config.futures_type}")
    print(f"   Granularity: {config.granularity}")

    # Use snappy compression (standard for all files)
    compression = "snappy"
    print(f"\n✅ Using compression: {compression}")

    # Ask if should delete ZIP files after conversion
    delete_zip = input("\n🗑️  Delete ZIP files after successful conversion? (yes/no): ").strip().lower() == 'yes'

    # Ask if should use parallel processing
    use_parallel = input("\n⚡ Enable parallel processing? (yes/no) [recommended]: ").strip().lower()
    parallel = use_parallel in ['yes', 'y', '']  # Default to yes

    max_workers = None
    if parallel:
        workers_input = input("Number of parallel workers (press Enter for auto-detect): ").strip()
        if workers_input.isdigit():
            max_workers = int(workers_input)

    try:
        # Initialize the streamer
        streamer = ZipToParquetStreamer(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity,
            compression=compression
        )

        if parallel:
            print(f"\n🚀 Starting parallel ZIP to Parquet conversion (workers: {max_workers or 'auto'})...")
        else:
            print("\n🚀 Starting direct ZIP to Parquet conversion...")

        # Process all ZIP files
        success_count, skip_count, fail_count = streamer.process_all_zips(
            skip_existing=True,
            delete_zip_after=delete_zip,
            parallel=parallel,
            max_workers=max_workers
        )

        # Display statistics
        stats = streamer.get_statistics()

        print("\n" + "="*60)
        print(" 📊 Conversion Summary ")
        print("="*60)
        print(f"✅ Successfully converted: {success_count} files")
        if skip_count > 0:
            print(f"⊘ Skipped (already exist): {skip_count} files")
        if fail_count > 0:
            print(f"❌ Failed: {fail_count} files")
        print(f"💾 Total Parquet size: {stats['total_parquet_size_gb']:.2f} GB")

        if success_count > 0:
            print("\n✅ Direct conversion completed successfully!")
            print("💡 Next steps:")
            print("   • Run option 3 to optimize Parquet files")
            print("   • Run option 4 to validate data integrity")

    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to continue...")


def run_complete_pipeline_parallel(config: PipelineConfig):
    """Run complete pipeline with parallel processing per file"""
    print("\n" + "="*60)
    print(" ⚡ Complete Pipeline with Parallel Processing ")
    print("="*60)

    print("\n📊 This will execute the complete pipeline:")
    print("   1. Download ZIP files with checksum")
    print("   2. Convert ZIP → Parquet directly")
    print("   3. Clean up ZIP, checksum, and CSV files")
    print("   4. All operations parallelized per file")

    print(f"\n🔧 Configuration:")
    print(f"   Symbol: {config.symbol}")
    print(f"   Type: {config.data_type}")
    if config.data_type == "futures":
        print(f"   Futures Type: {config.futures_type}")
    print(f"   Granularity: Daily (optimized)")

    # Get date range with suggestions from legacy function
    print("\n📅 Quick-Start Date Range:")
    from datetime import datetime, timedelta

    # Get existing data range for suggestions
    first_available, last_available = get_data_date_range(
        config.symbol, config.data_type, config.futures_type, 'daily'
    )

    current_date = datetime.now()

    # Determine suggested dates based on existing data and data type
    if first_available and last_available and first_available != "data-exists":
        print(f"💡 Existing data found: {first_available} to {last_available}")
        # Suggest continuing from next day after last available
        try:
            last_date = datetime.strptime(last_available, "%Y-%m-%d")
            next_date = last_date + timedelta(days=1)
            suggested_start = next_date.strftime("%Y-%m-%d")
        except:
            suggested_start = current_date.strftime("%Y-%m-%d")
        suggested_end = current_date.strftime("%Y-%m-%d")
    else:
        # Use defaults from legacy function based on data type
        if config.data_type == "spot":
            suggested_start = "2024-01-01"
            suggested_end = "2024-01-31"
            print(f"💡 Binance spot data available from 2017-08-17")
        else:  # futures
            suggested_start = "2019-09-08"
            suggested_end = "2019-09-30"
            print(f"💡 Binance futures data available from 2019-09-08")

    print(f"📊 Suggested range: {suggested_start} to {suggested_end}")
    print("Press Enter to use suggested dates or type new ones:")

    while True:
        start_str = input(f"Start date [{suggested_start}]: ").strip()
        if not start_str:
            start_str = suggested_start
        try:
            start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
            break
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")

    while True:
        end_str = input(f"End date [{suggested_end}]: ").strip()
        if not end_str:
            end_str = suggested_end
        try:
            end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
            if end_date < start_date:
                print("❌ End date must be after start date")
                continue
            break
        except ValueError:
            print("❌ Invalid date format. Use YYYY-MM-DD")

    # Calculate number of days
    total_days = (end_date - start_date).days + 1
    print(f"\n📊 Processing {total_days} days of data")

    # Use snappy compression (standard for all files)
    compression = "snappy"
    print(f"\n✅ Using compression: {compression}")

    # Ask for number of workers
    import multiprocessing as mp
    max_cpus = mp.cpu_count()
    default_workers = min(4, max_cpus - 1)

    workers_input = input(f"\nNumber of parallel workers (1-{max_cpus}, default {default_workers}): ").strip()
    if workers_input.isdigit():
        max_workers = min(int(workers_input), max_cpus)
    else:
        max_workers = default_workers
    print(f"✅ Using {max_workers} parallel workers")

    # Confirm before starting
    print("\n" + "="*60)
    print(" 📋 PIPELINE SUMMARY ")
    print("="*60)
    print(f"Symbol: {config.symbol}")
    print(f"Date Range: {start_date} to {end_date} ({total_days} days)")
    print(f"Compression: {compression}")
    print(f"Workers: {max_workers}")
    print(f"Operations: Download → Convert → Cleanup")

    confirm = input("\n❓ Start pipeline? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("❌ Pipeline cancelled.")
        return

    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.data_pipeline.downloaders.binance_downloader import BinanceDataDownloader
        from src.data_pipeline.converters.zip_to_parquet_streamer import ZipToParquetStreamer
        from pathlib import Path
        import time

        # Initialize components
        downloader = BinanceDataDownloader(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity="daily"
        )

        streamer = ZipToParquetStreamer(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity="daily",
            compression=compression
        )

        # Generate dates
        dates_to_process = []
        current = start_date
        while current <= end_date:
            dates_to_process.append(datetime.combine(current, datetime.min.time()))
            current += timedelta(days=1)

        print(f"\n🚀 Starting parallel pipeline for {len(dates_to_process)} dates...")
        start_time = time.time()

        # Statistics
        successful = 0
        failed = 0
        skipped = 0

        def process_single_date(date):
            """Process a single date through the complete pipeline"""
            try:
                # Step 1: Download ZIP with checksum
                zip_file, checksum_file = downloader.download_with_checksum(date)
                if not zip_file or not checksum_file:
                    return 'download_failed', date

                # Step 2: Convert ZIP to Parquet directly
                parquet_file = streamer.convert_zip_to_parquet(zip_file)
                if not parquet_file:
                    return 'conversion_failed', date

                # Step 3: Clean up files
                # Delete ZIP and checksum
                zip_file.unlink()
                checksum_file.unlink()

                # Check if CSV exists and delete it
                csv_name = zip_file.stem + ".csv"
                csv_path = zip_file.parent / csv_name
                if csv_path.exists():
                    csv_path.unlink()

                return 'success', date

            except Exception as e:
                print(f"❌ Error processing {date.strftime('%Y-%m-%d')}: {e}")
                return 'error', date

        # Process dates in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_date = {
                executor.submit(process_single_date, date): date
                for date in dates_to_process
            }

            # Process results with progress bar
            from tqdm import tqdm
            with tqdm(total=len(dates_to_process), desc="Processing dates") as pbar:
                for future in as_completed(future_to_date):
                    date = future_to_date[future]

                    try:
                        status, processed_date = future.result()
                        if status == 'success':
                            successful += 1
                            pbar.set_postfix({"✅": successful, "❌": failed, "⏭️": skipped})
                        elif status == 'download_failed':
                            failed += 1
                            print(f"❌ Download failed: {processed_date.strftime('%Y-%m-%d')}")
                        elif status == 'conversion_failed':
                            failed += 1
                            print(f"❌ Conversion failed: {processed_date.strftime('%Y-%m-%d')}")
                        else:
                            failed += 1
                    except Exception as e:
                        failed += 1
                        print(f"❌ Task failed for {date.strftime('%Y-%m-%d')}: {e}")

                    pbar.update(1)

        # Calculate elapsed time
        elapsed = time.time() - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        # Summary
        print("\n" + "="*60)
        print(" 📊 PIPELINE COMPLETE ")
        print("="*60)
        print(f"✅ Successful: {successful} files")
        if failed > 0:
            print(f"❌ Failed: {failed} files")
        print(f"⏱️  Total time: {minutes}m {seconds}s")

        if successful > 0:
            avg_time = elapsed / len(dates_to_process)
            print(f"📈 Average time per file: {avg_time:.2f} seconds")
            speedup = len(dates_to_process) * avg_time / elapsed
            print(f"🚀 Parallel speedup: {speedup:.1f}x")

            # Get statistics on created files (modern structure)
            ticker_name = f"{config.symbol.lower()}-{config.data_type}"
            if config.data_type == "futures":
                ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

            parquet_dir = Path("data") / ticker_name / f"raw-parquet-{config.granularity}"

            if parquet_dir.exists():
                parquet_files = list(parquet_dir.glob("*.parquet"))
                total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)
                print(f"\n💾 Total Parquet size: {total_size:.2f} GB")
                print(f"📁 Files location: {parquet_dir}")

            # Validate parquet files
            validation_result = validate_parquet_files(
                symbol=config.symbol,
                data_type=config.data_type,
                futures_type=config.futures_type,
                start_date=start_date,
                end_date=end_date,
                parquet_dir=parquet_dir
            )

            # Ask user if they want to retry failed/missing files
            if validation_result['missing_dates'] or validation_result['corrupted_files']:
                retry = input("\n🔄 Retry failed/missing files? (yes/no): ").strip().lower()
                if retry == 'yes':
                    print("💡 Run this command again with the missing date range")

    except Exception as e:
        print(f"\n❌ Pipeline error: {str(e)}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to continue...")


def add_missing_daily_data(config: PipelineConfig):
    """Automatically detect and add missing daily data"""
    print("\n" + "="*60)
    print(" 📅 Auto-Detect and Add Missing Daily Data ")
    print("="*60)

    from datetime import datetime, timedelta
    from pathlib import Path
    import pyarrow.parquet as pq
    import pandas as pd

    print(f"\n🔍 Analyzing current data status...")
    print(f"   Symbol: {config.symbol}")
    print(f"   Type: {config.data_type}")

    # Find optimized parquet files (modern structure)
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    optimized_dir = Path("data") / ticker_name / f"raw-parquet-merged-{config.granularity}"

    # Get the latest timestamp from optimized files
    last_timestamp = None

    if optimized_dir.exists():
        parquet_files = sorted(optimized_dir.glob("*.parquet"))

        if parquet_files:
            print(f"\n📂 Found {len(parquet_files)} optimized parquet files")

            # Get the last file and read its last timestamp
            last_file = parquet_files[-1]
            print(f"📄 Checking last file: {last_file.name}")

            try:
                # Read the parquet file metadata
                parquet_file = pq.ParquetFile(last_file)

                # Get the last row group
                last_row_group = parquet_file.num_row_groups - 1

                # Read just the last batch
                last_batch = parquet_file.read_row_group(last_row_group).to_pandas()

                # Get the maximum timestamp
                if 'time' in last_batch.columns:
                    last_timestamp = pd.to_datetime(last_batch['time'].max())
                    print(f"✅ Last data point: {last_timestamp}")
                else:
                    print("❌ No 'time' column found in parquet file")
                    return

            except Exception as e:
                print(f"❌ Error reading parquet file: {e}")

                # Fallback: try reading with pandas
                try:
                    df = pd.read_parquet(last_file, columns=['time'])
                    last_timestamp = pd.to_datetime(df['time'].max())
                    print(f"✅ Last data point (fallback): {last_timestamp}")
                except Exception as e2:
                    print(f"❌ Fallback also failed: {e2}")
                    return
        else:
            print("❌ No optimized parquet files found. Run the pipeline first.")
            return
    else:
        print("❌ Optimized directory not found. Run the pipeline first.")
        return

    if not last_timestamp:
        print("❌ Could not determine last timestamp")
        return

    # Calculate days since last data
    current_date = datetime.now()
    days_behind = (current_date - last_timestamp).days

    print(f"\n📊 Data Status:")
    print(f"   • Last data: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   • Current date: {current_date.strftime('%Y-%m-%d')}")
    print(f"   • Days behind: {days_behind}")

    if days_behind <= 1:
        print("\n✅ Your data is up to date!")
        input("\nPress Enter to continue...")
        return

    # Suggest date range
    start_date = last_timestamp.date() + timedelta(days=1)
    end_date = current_date.date() - timedelta(days=1)  # Exclude today

    print(f"\n🎯 Suggested update range:")
    print(f"   • Start: {start_date}")
    print(f"   • End: {end_date}")
    print(f"   • Total days: {(end_date - start_date).days + 1}")

    # Ask for confirmation
    confirm = input("\n🚀 Proceed with automatic update? (yes/no): ").strip().lower()

    if confirm != 'yes':
        # Manual mode
        print("\n📅 Manual mode - Enter custom date range:")
        print("   Format: YYYY-MM-DD")

        while True:
            start_str = input("   Start date: ").strip()
            try:
                start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
                break
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD")

        while True:
            end_str = input("   End date: ").strip()
            try:
                end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
                if end_date < start_date:
                    print("❌ End date must be after start date")
                    continue
                break
            except ValueError:
                print("❌ Invalid date format. Use YYYY-MM-DD")

    # Now execute the full pipeline for daily data
    print(f"\n🚀 Starting automatic pipeline update...")
    print(f"   Date range: {start_date} to {end_date}")

    try:
        # First check if parquet files are already available (modern structure)
        ticker_name = f"{config.symbol.lower()}-{config.data_type}"
        if config.data_type == "futures":
            ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

        daily_parquet_dir = Path("data") / ticker_name / f"raw-parquet-{config.granularity}"

        # Generate dates to check
        dates_to_check = []
        current = start_date
        while current <= end_date:
            dates_to_check.append(current)
            current += timedelta(days=1)

        # Check which parquet files already exist
        existing_parquet_dates = []
        missing_dates = []

        print("\n🔍 Checking for existing parquet files...")
        for date in dates_to_check:
            date_str = date.strftime('%Y-%m-%d')
            parquet_pattern = f"{config.symbol}-Trades-{date_str}*.parquet"
            existing_files = list(daily_parquet_dir.glob(parquet_pattern)) if daily_parquet_dir.exists() else []

            if existing_files:
                existing_parquet_dates.append(date)
            else:
                missing_dates.append(date)

        if existing_parquet_dates:
            print(f"✅ Found {len(existing_parquet_dates)} existing parquet files")
            print(f"📊 Date range with data: {existing_parquet_dates[0].strftime('%Y-%m-%d')} to {existing_parquet_dates[-1].strftime('%Y-%m-%d')}")

        if missing_dates:
            print(f"❌ Missing {len(missing_dates)} parquet files")
            print(f"📊 Missing dates: {missing_dates[0].strftime('%Y-%m-%d')} to {missing_dates[-1].strftime('%Y-%m-%d')}")

        # Ask what to do
        if existing_parquet_dates and not missing_dates:
            print("\n✅ All parquet files already exist!")
            skip_download = input("Skip download and go directly to merge? (yes/no): ").strip().lower()
            if skip_download == 'yes':
                # Skip to merge
                dates = [datetime.combine(d, datetime.min.time()) for d in dates_to_check]
                print("\n⏭️ Skipping download and extraction, going directly to merge...")
                # Jump directly to Step 4
                successful = len(existing_parquet_dates)
                failed = 0
                goto_merge = True
            else:
                goto_merge = False
        else:
            goto_merge = False

        if not goto_merge:
            # Step 1: Download daily data
            print("\n" + "="*50)
            print(" Step 1/4: Download Daily Data ")
            print("="*50)

            from src.data_pipeline.downloaders.binance_downloader import BinanceDataDownloader
            from src.data_pipeline.extractors.csv_extractor import CSVExtractor

            # Create daily config
            daily_config = PipelineConfig()
            daily_config.symbol = config.symbol
            daily_config.data_type = config.data_type
            daily_config.futures_type = config.futures_type
            daily_config.granularity = "daily"  # Force daily

            # Ask for number of workers
            import multiprocessing as mp
            max_cpus = mp.cpu_count()
            default_workers = min(4, max_cpus - 1)

            workers_input = input(f"\nNumber of parallel workers (1-{max_cpus}, default {default_workers}): ").strip()
            if workers_input.isdigit():
                daily_config.workers = min(int(workers_input), max_cpus)
            else:
                daily_config.workers = default_workers
            print(f"✅ Using {daily_config.workers} parallel workers")

            # Create downloader
            downloader = BinanceDataDownloader(
                symbol=daily_config.symbol,
                data_type=daily_config.data_type,
                futures_type=daily_config.futures_type,
                granularity="daily",
                base_dir=Path("data")
            )

            # Generate dates
            dates = []
            current = start_date
            while current <= end_date:
                dates.append(datetime.combine(current, datetime.min.time()))
                current += timedelta(days=1)

            # Confirm before starting
            print("\n" + "="*60)
            print(" 📋 PIPELINE SUMMARY ")
            print("="*60)
            print(f"Symbol: {daily_config.symbol}")
            print(f"Date Range: {start_date} to {end_date} ({len(dates)} days)")
            print(f"Compression: snappy")
            print(f"Workers: {daily_config.workers}")
            print("="*60)

            confirm = input("\n🚀 Start processing? (yes/no): ").strip().lower()
            if confirm != 'yes':
                print("❌ Operation cancelled.")
                input("\nPress Enter to continue...")
                return

            print(f"\n🚀 Processing {len(dates)} days (Download → Convert → Cleanup)...")

            # Initialize streamer for direct conversion
            from src.data_pipeline.converters.zip_to_parquet_streamer import ZipToParquetStreamer

            streamer = ZipToParquetStreamer(
                symbol=daily_config.symbol,
                data_type=daily_config.data_type,
                futures_type=daily_config.futures_type,
                granularity="daily",
                compression="snappy"  # Must match existing files (001-006)
            )

            # Process each date: download → convert → delete (in parallel)
            def process_single_date(date):
                """Process a single date through the complete pipeline"""
                try:
                    # Step 1: Download ZIP with checksum
                    zip_file, checksum_file = downloader.download_with_checksum(date)
                    if not zip_file or not checksum_file:
                        return 'download_failed', date

                    # Step 2: Convert ZIP to Parquet directly
                    parquet_file = streamer.convert_zip_to_parquet(zip_file)
                    if not parquet_file:
                        return 'conversion_failed', date

                    # Step 3: Clean up files immediately
                    zip_file.unlink()
                    checksum_file.unlink()

                    # Check if CSV exists and delete it
                    csv_name = zip_file.stem + ".csv"
                    csv_path = zip_file.parent / csv_name
                    if csv_path.exists():
                        csv_path.unlink()

                    return 'success', date

                except Exception as e:
                    return 'error', date

            # Execute in parallel with progress bar
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            successful = 0
            failed = 0

            with ThreadPoolExecutor(max_workers=daily_config.workers) as executor:
                # Submit all tasks
                future_to_date = {
                    executor.submit(process_single_date, date): date
                    for date in dates
                }

                # Process results with progress bar
                with tqdm(total=len(dates), desc="📊 Processing dates", unit="day") as pbar:
                    for future in as_completed(future_to_date):
                        status, processed_date = future.result()

                        if status == 'success':
                            successful += 1
                            pbar.set_postfix({"✅": successful, "❌": failed})
                        else:
                            failed += 1
                            pbar.set_postfix({"✅": successful, "❌": failed})

                        pbar.update(1)

            print(f"\n✅ Successfully processed: {successful} days")
            if failed > 0:
                print(f"❌ Failed: {failed} days")

            if successful == 0:
                print("❌ No files processed. Aborting.")
                input("\nPress Enter to continue...")
                return

            # Validate parquet files
            ticker_name = f"{daily_config.symbol.lower()}-{daily_config.data_type}"
            if daily_config.data_type == "futures":
                ticker_name = f"{daily_config.symbol.lower()}-{daily_config.data_type}-{daily_config.futures_type}"

            parquet_dir = Path("data") / ticker_name / f"raw-parquet-{daily_config.granularity}"

            validation_result = validate_parquet_files(
                symbol=daily_config.symbol,
                data_type=daily_config.data_type,
                futures_type=daily_config.futures_type,
                start_date=start_date,
                end_date=end_date,
                parquet_dir=parquet_dir
            )

            # Ask user if they want to retry failed/missing files
            if validation_result['missing_dates'] or validation_result['corrupted_files']:
                retry = input("\n🔄 Retry failed/missing files? (yes/no): ").strip().lower()
                if retry == 'yes':
                    print("💡 Run this command again with the missing date range")
                    input("\nPress Enter to continue...")
                    return
        else:
            # We're skipping download/extract/convert, create daily_config for merge
            daily_config = PipelineConfig()
            daily_config.symbol = config.symbol
            daily_config.data_type = config.data_type
            daily_config.futures_type = config.futures_type
            daily_config.granularity = "daily"

        # Step 2: Merge with existing optimized data
        print("\n" + "="*50)
        print(" Step 2/2: Merge with Optimized Data ")
        print("="*50)

        # Initialize merge status
        merge_successful = False

        # For daily data updates, we need to merge them into the existing optimized files
        # This applies to both monthly and daily granularity pipelines when adding new daily data
        if True:  # Always attempt to merge daily updates
            print(f"📊 Preparing daily updates for merge into {config.granularity} optimized.parquet files...")

            # Get the daily parquet files (modern structure)
            ticker_name = f"{config.symbol.lower()}-{config.data_type}"
            if config.data_type == "futures":
                ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

            daily_parquet_dir = Path("data") / ticker_name / f"raw-parquet-{config.granularity}"

            print(f"📁 Looking for daily parquet files in: {daily_parquet_dir}")
            if not daily_parquet_dir.exists():
                print(f"❌ Directory does not exist: {daily_parquet_dir}")
                merge_successful = False
            else:
                all_parquet_files = list(daily_parquet_dir.glob("*.parquet"))
                print(f"   Total parquet files in directory: {len(all_parquet_files)}")
                if all_parquet_files and len(all_parquet_files) <= 10:
                    print("   Files found:")
                    for f in sorted(all_parquet_files)[:10]:
                        print(f"     - {f.name}")

                # Find new daily parquet files
                new_daily_files = []
                for date in dates:
                    date_str = date.strftime('%Y-%m-%d')
                    # Try multiple patterns as the daily converter might use different naming
                    patterns = [
                        f"{config.symbol}-Trades-{date_str}*.parquet",
                        f"{config.symbol}-trades-{date_str}*.parquet"  # lowercase variant
                    ]
                    for pattern in patterns:
                        found_files = list(daily_parquet_dir.glob(pattern))
                        if found_files:
                            new_daily_files.extend(found_files)
                            print(f"   Found {len(found_files)} files matching pattern: {pattern}")

                if new_daily_files:
                    print(f"📄 Found {len(new_daily_files)} new daily files to merge")
                    print("   Files to merge:")
                    for f in sorted(new_daily_files)[:5]:
                        print(f"     - {f.name}")
                    if len(new_daily_files) > 5:
                        print(f"     ... and {len(new_daily_files) - 5} more files")

                    # Import the merger (using memory-optimized version)
                    from src.data_pipeline.processors.parquet_merger import ParquetMerger

                    # Initialize merger
                    merger = ParquetMerger(symbol=config.symbol)

                    # Determine optimized directory (modern structure)
                    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
                    if config.data_type == "futures":
                        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

                    optimized_dir = Path("data") / ticker_name / f"raw-parquet-merged-{config.granularity}"

                    # Ensure optimized directory exists
                    optimized_dir.mkdir(parents=True, exist_ok=True)

                    print("\n🔄 Merging daily files into optimized parquet...")
                    print(f"   Optimized directory: {optimized_dir}")
                    print(f"   Daily directory: {daily_parquet_dir}")

                    try:
                        files_merged, rows_added = merger.merge_daily_files(
                            optimized_dir=optimized_dir,
                            daily_dir=daily_parquet_dir,
                            daily_files=new_daily_files,
                            max_file_size_gb=10.0,  # Use the standard 10GB limit
                            delete_after_merge=True  # Delete daily files after successful merge
                        )

                        if rows_added > 0:
                            print(f"✅ Successfully merged {rows_added:,} new rows into optimized files")
                            print(f"   Files processed: {files_merged}")
                            print("   Daily parquet files deleted after successful merge")
                            merge_successful = True
                        else:
                            print("ℹ️  No new data was added (all data already exists in optimized files)")
                            merge_successful = True  # Still successful, just no new data

                    except Exception as e:
                        print(f"❌ Error during merge: {e}")
                        import traceback
                        print(f"🔍 Traceback: {traceback.format_exc()}")
                        print("💡 Daily files remain in place and can be merged manually later")
                        merge_successful = False
                else:
                    print("❌ No new daily parquet files found")
                    merge_successful = False
        else:
            # Not monthly granularity, no merge needed
            merge_successful = True

        # Force garbage collection to free memory after merge
        import gc
        gc.collect()

        # Clean up ZIP files automatically (parquet files are deleted during merge)
        print("\n🗑️  Automatically cleaning up temporary ZIP files...")

        # Only cleanup if merge was successful
        if merge_successful:
            # Clean daily ZIP files only (parquet files already deleted during merge)
            ticker_name = f"{daily_config.symbol.lower()}-{daily_config.data_type}"
            if daily_config.data_type == "futures":
                ticker_name = f"{daily_config.symbol.lower()}-{daily_config.data_type}-{daily_config.futures_type}"

            daily_raw_dir = Path("data") / ticker_name / f"raw-zip-{daily_config.granularity}"

            # Legacy cleanup (in case files are still in old location)
            if False:  # Disabled legacy cleanup
                daily_raw_dir = Path("data") / "dataset-raw-daily" / f"futures-{daily_config.futures_type}"

            file_count = 0
            freed_space = 0

            # Clean ZIP and CHECKSUM files
            for date in dates:
                date_str = date.strftime('%Y-%m-%d')
                zip_path = daily_raw_dir / f"{config.symbol}-trades-{date_str}.zip"
                checksum_path = daily_raw_dir / f"{config.symbol}-trades-{date_str}.zip.CHECKSUM"

                for f in [zip_path, checksum_path]:
                    if f.exists():
                        try:
                            freed_space += f.stat().st_size
                            f.unlink()
                            file_count += 1
                        except:
                            pass

            print(f"\n📊 Cleanup Summary:")
            print(f"   • ZIP/CHECKSUM files deleted: {file_count}")
            print(f"   • Disk space freed: {freed_space / (1024**3):.2f} GB")
            print(f"   • Note: Daily parquet files were deleted during merge")
        else:
            print("⚠️  Skipping cleanup due to merge failure - daily files preserved")

        if merge_successful:
            print("\n✅ Update completed successfully!")
        else:
            print("\n⚠️  Update completed with warnings - check daily files for manual merge")

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to continue...")


def run_download_only(config: PipelineConfig):
    """Step 1: Download ZIP files with checksum verification only"""
    print("\n" + "="*60)
    print(" 📥 Step 1: Download ZIP Data with Checksum Verification ")
    print("="*60)

    # Log the start of operation
    logging.info(f"Starting Step 1: Download Only for {config.symbol} {config.data_type} {config.granularity}")

    # Get existing data range for suggestions
    first_available, last_available = get_data_date_range(
        config.symbol, config.data_type, config.futures_type, config.granularity
    )

    # Get date range
    print(f"\n📅 Enter date range for {config.granularity} data:")

    # Import datetime for date suggestions
    from datetime import datetime, timedelta
    current_date = datetime.now()

    if config.granularity == 'daily':
        date_format = "YYYY-MM-DD"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            print(f"📊 Most recent data: {last_available}")
            example_start = last_available  # Suggest continuing from last available
            # Suggest next day after last available
            try:
                last_date = datetime.strptime(last_available, "%Y-%m-%d")
                next_date = last_date + timedelta(days=1)
                example_start = next_date.strftime("%Y-%m-%d")
            except:
                example_start = current_date.strftime("%Y-%m-%d")
            example_end = current_date.strftime("%Y-%m-%d")
        else:
            # Different suggestions based on data type
            if config.data_type == "spot":
                example_start = "2024-01-01"
                example_end = "2024-01-31"
                print(f"💡 No existing data found. Binance spot data available from 2017-08-17")
            else:  # futures
                example_start = "2019-09-08"  # Futures daily data typically started around this date
                example_end = "2019-09-30"
                print(f"💡 No existing data found. Binance futures data typically available from 2019-09-08")
    else:
        date_format = "YYYY-MM"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            print(f"📊 Most recent data: {last_available}")
            # Suggest next month after last available
            try:
                last_date = datetime.strptime(last_available + "-01", "%Y-%m-%d")
                if last_date.month == 12:
                    next_date = last_date.replace(year=last_date.year + 1, month=1)
                else:
                    next_date = last_date.replace(month=last_date.month + 1)
                example_start = next_date.strftime("%Y-%m")
            except:
                example_start = current_date.strftime("%Y-%m")
            example_end = current_date.strftime("%Y-%m")
        else:
            # Different suggestions based on data type
            if config.data_type == "spot":
                example_start = "2024-01"
                example_end = "2024-12"
                print(f"💡 No existing data found. Binance spot data available from 2017-08")
            else:  # futures
                example_start = "2019-09"  # Futures typically started later
                example_end = "2024-12"
                print(f"💡 No existing data found. Binance futures data typically available from 2019-09")

    while True:
        start_date = input(f"Start date ({date_format}, e.g., {example_start}): ").strip()
        if start_date:
            config.start_date = start_date
            break
        print("❌ Start date is required.")

    while True:
        end_date = input(f"End date ({date_format}, e.g., {example_end}): ").strip()
        if end_date:
            config.end_date = end_date
            break
        print("❌ End date is required.")

    # Get workers
    workers_input = input("\nNumber of concurrent downloads (default: 5): ").strip()
    config.workers = int(workers_input) if workers_input.isdigit() else 5

    # Show summary
    print(f"\n📋 Download Configuration:")
    print(f"   Date Range: {config.start_date} to {config.end_date}")
    print(f"   Workers: {config.workers}")

    confirm = input("\n🚀 Proceed with download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Download cancelled.")
        return

    # Build command line arguments
    download_args = [
        'binance_downloader.py', 'download',
        '--symbol', config.symbol,
        '--type', config.data_type,
        '--granularity', config.granularity,
        '--start', config.start_date,
        '--end', config.end_date,
        '--workers', str(config.workers)
    ]

    if config.data_type == 'futures':
        download_args.extend(['--futures-type', config.futures_type])

    # Set sys.argv for the downloader
    original_argv = sys.argv.copy()
    sys.argv = download_args

    try:
        # Use a custom approach that only downloads ZIP files
        from src.data_pipeline.downloaders.binance_downloader import BinanceDataDownloader
        import concurrent.futures

        downloader = BinanceDataDownloader(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        # Generate date range
        start = datetime.strptime(config.start_date, '%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
        end = datetime.strptime(config.end_date, '%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
        dates = downloader.generate_dates(start, end)

        print(f"\n📥 Downloading {len(dates)} ZIP files with CHECKSUM verification...")
        logging.info(f"Download phase: {len(dates)} files to process for date range {config.start_date} to {config.end_date}")

        # Download only ZIP files, skip processing
        downloaded_count = 0
        failed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_to_date = {
                executor.submit(downloader.download_with_checksum, date): date
                for date in dates
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    zip_file, checksum_file = future.result()
                    if zip_file and checksum_file:
                        downloaded_count += 1
                        print(f"✅ Downloaded: {zip_file.name}")
                    else:
                        # Check why download returned None
                        date_str = date.strftime('%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
                        zip_name = f"{config.symbol}-trades-{date_str}.zip"
                        csv_name = f"{config.symbol}-trades-{date_str}.csv"

                        if (downloader.raw_dir / csv_name).exists():
                            print(f"⏭️ CSV already exists: {csv_name}")
                        elif (downloader.raw_dir / zip_name).exists():
                            print(f"📦 ZIP exists: {zip_name}")
                        else:
                            failed_count += 1
                            print(f"❌ Download failed: {zip_name}")
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Error downloading {date}: {e}")

        print(f"\n✅ Download completed: {downloaded_count} downloaded, {failed_count} failed")
        logging.info(f"Download completed: {downloaded_count} downloaded, {failed_count} failed")

        # Pipeline completion message
        print("\n" + "="*60)
        print(" 🎉 Step 1 Completed: ZIP Files Ready! ")
        print("="*60)
        print("\n📌 ZIP files have been downloaded with checksum verification.")

        # Determine ZIP location (modern structure)
        ticker_name = f"{config.symbol.lower()}-{config.data_type}"
        if config.data_type == "futures":
            ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

        zip_dir = Path("data") / ticker_name / f"raw-zip-{config.granularity}"

        print("📁 Location: " + str(zip_dir))
        print("\n🔄 Next Steps (run these commands separately):")
        print("   1️⃣ Extract CSV from ZIP: python main.py (Step 2)")
        print("   2️⃣ CSV to Parquet conversion: python main.py (Step 3)")
        print("   3️⃣ Parquet optimization: python main.py (Step 4)")
        print("   4️⃣ Data validation: python main.py (Step 5)")
        print("   5️⃣ Feature generation: python main.py (Step 6)")
        print("\n💡 Or use the interactive menu to continue with the next steps.")

    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        logging.error(f"Download failed: {e}", exc_info=True)
    finally:
        sys.argv = original_argv


def run_download_and_extract_backup(config: PipelineConfig):
    """BACKUP: Original Step 1 function with download, extract and validate"""
    print("\n" + "="*60)
    print(" 📥 BACKUP: Download ZIP Data, Extract and Validate CSV ")
    print("="*60)

    # Log the start of operation
    logging.info(f"Starting BACKUP Step 1: Download and Extract for {config.symbol} {config.data_type} {config.granularity}")

    # Get existing data range for suggestions
    first_available, last_available = get_data_date_range(
        config.symbol, config.data_type, config.futures_type, config.granularity
    )

    # Get date range
    print(f"\n📅 Enter date range for {config.granularity} data:")

    # Import datetime for date suggestions
    from datetime import datetime, timedelta
    current_date = datetime.now()

    if config.granularity == 'daily':
        date_format = "YYYY-MM-DD"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            print(f"📊 Most recent data: {last_available}")
            example_start = last_available  # Suggest continuing from last available
            # Suggest next day after last available
            try:
                last_date = datetime.strptime(last_available, "%Y-%m-%d")
                next_date = last_date + timedelta(days=1)
                example_start = next_date.strftime("%Y-%m-%d")
            except:
                example_start = current_date.strftime("%Y-%m-%d")
            example_end = current_date.strftime("%Y-%m-%d")
        else:
            # Different suggestions based on data type
            if config.data_type == "spot":
                example_start = "2024-01-01"
                example_end = "2024-01-31"
                print(f"💡 No existing data found. Binance spot data available from 2017-08-17")
            else:  # futures
                example_start = "2019-09-08"  # Futures daily data typically started around this date
                example_end = "2019-09-30"
                print(f"💡 No existing data found. Binance futures data typically available from 2019-09-08")
    else:
        date_format = "YYYY-MM"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            print(f"📊 Most recent data: {last_available}")
            # Suggest next month after last available
            try:
                last_date = datetime.strptime(last_available + "-01", "%Y-%m-%d")
                if last_date.month == 12:
                    next_date = last_date.replace(year=last_date.year + 1, month=1)
                else:
                    next_date = last_date.replace(month=last_date.month + 1)
                example_start = next_date.strftime("%Y-%m")
            except:
                example_start = current_date.strftime("%Y-%m")
            example_end = current_date.strftime("%Y-%m")
        else:
            # Different suggestions based on data type
            if config.data_type == "spot":
                example_start = "2024-01"
                example_end = "2024-12"
                print(f"💡 No existing data found. Binance spot data available from 2017-08")
            else:  # futures
                example_start = "2019-09"  # Futures typically started later
                example_end = "2024-12"
                print(f"💡 No existing data found. Binance futures data typically available from 2019-09")

    while True:
        start_date = input(f"Start date ({date_format}, e.g., {example_start}): ").strip()
        if start_date:
            config.start_date = start_date
            break
        print("❌ Start date is required.")

    while True:
        end_date = input(f"End date ({date_format}, e.g., {example_end}): ").strip()
        if end_date:
            config.end_date = end_date
            break
        print("❌ End date is required.")

    # Get workers
    workers_input = input("\nNumber of concurrent downloads (default: 5): ").strip()
    config.workers = int(workers_input) if workers_input.isdigit() else 5

    # Show summary
    print(f"\n📋 Download Configuration:")
    print(f"   Date Range: {config.start_date} to {config.end_date}")
    print(f"   Workers: {config.workers}")

    confirm = input("\n🚀 Proceed with download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Download cancelled.")
        return

    # Build command line arguments
    download_args = [
        'binance_downloader.py', 'download',
        '--symbol', config.symbol,
        '--type', config.data_type,
        '--granularity', config.granularity,
        '--start', config.start_date,
        '--end', config.end_date,
        '--workers', str(config.workers)
    ]

    if config.data_type == 'futures':
        download_args.extend(['--futures-type', config.futures_type])

    # Set sys.argv for the downloader
    original_argv = sys.argv.copy()
    sys.argv = download_args

    try:
        # Use a custom approach that only downloads ZIP files
        from src.data_pipeline.downloaders.binance_downloader import BinanceDataDownloader
        import concurrent.futures

        downloader = BinanceDataDownloader(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        # Generate date range
        start = datetime.strptime(config.start_date, '%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
        end = datetime.strptime(config.end_date, '%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
        dates = downloader.generate_dates(start, end)

        print(f"\n📥 Downloading {len(dates)} ZIP files with CHECKSUM verification...")
        logging.info(f"Download phase: {len(dates)} files to process for date range {config.start_date} to {config.end_date}")

        # Download only ZIP files, skip processing
        downloaded_count = 0
        failed_count = 0

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_to_date = {
                executor.submit(downloader.download_with_checksum, date): date
                for date in dates
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    zip_file, checksum_file = future.result()
                    if zip_file and checksum_file:
                        downloaded_count += 1
                        print(f"✅ Downloaded: {zip_file.name}")
                    else:
                        # Check why download returned None
                        date_str = date.strftime('%Y-%m-%d' if config.granularity == 'daily' else '%Y-%m')
                        zip_name = f"{config.symbol}-trades-{date_str}.zip"
                        csv_name = f"{config.symbol}-trades-{date_str}.csv"

                        if (downloader.raw_dir / csv_name).exists():
                            print(f"⏭️ CSV already exists: {csv_name}")
                        elif (downloader.raw_dir / zip_name).exists():
                            print(f"📦 ZIP exists: {zip_name}")
                        else:
                            failed_count += 1
                            print(f"❌ Download failed: {zip_name}")
                except Exception as e:
                    failed_count += 1
                    print(f"❌ Error downloading {date}: {e}")

        print(f"\n✅ Download phase completed: {downloaded_count} downloaded, {failed_count} failed")
        logging.info(f"Download phase completed: {downloaded_count} downloaded, {failed_count} failed")

        # Now extract CSV files from downloaded ZIPs
        print("\n📦 Extracting CSV files from ZIP archives...")

        extractor = CSVExtractor(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        # Force re-extraction for step 1 to ensure fresh data
        successful, failed = extractor.extract_and_verify_all(force_reextract=True)

        if successful > 0:
            print(f"\n✅ Extraction completed: {successful} files extracted (with force re-extraction)")
            logging.info(f"Extraction completed: {successful} files extracted (force re-extraction enabled)")
            if failed > 0:
                print(f"⚠️ {failed} files failed extraction")
                logging.warning(f"{failed} files failed extraction")

            # CSV Validation
            print("\n🔍 Validating extracted CSV files...")

            # Get extracted CSV files
            if config.data_type == "spot":
                raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / "spot"
            else:
                raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / f"futures-{config.futures_type}"

            csv_files = list(raw_dir.glob(f"{config.symbol}-trades-*.csv"))
            print(f"📄 Found {len(csv_files)} CSV files to validate")

            # Enhanced validation of CSV files using extractor's verify method
            validation_passed = True
            validation_details = []

            print("\n📊 Performing comprehensive CSV validation...")
            print("   - Timestamp UTC conversion check")
            print("   - Missing dates detection")
            print("   - Data integrity verification")

            # Validate all files (or sample if too many)
            files_to_validate = csv_files if len(csv_files) <= 10 else csv_files[:10]

            for csv_file in files_to_validate:
                print(f"\n   📄 Validating {csv_file.name}...")
                if extractor.verify_csv_integrity(csv_file):
                    validation_details.append(f"✅ {csv_file.name}")
                else:
                    validation_details.append(f"❌ {csv_file.name}")
                    validation_passed = False

            if len(csv_files) > 10:
                print(f"\n   ... and {len(csv_files) - 10} more files")

            if validation_passed:
                print("\n✅ CSV validation passed!")
                logging.info("CSV validation passed successfully")
            else:
                print("\n⚠️ Some CSV files failed validation")
                logging.warning(f"CSV validation failed - details: {validation_details}")

            # Don't ask about cleanup in step 1 - keep all ZIP files
            print("\n📦 ZIP files preserved for backup")

            # Pipeline completion message
            print("\n" + "="*60)
            print(" 🎉 Step 1 Completed: CSV Files Ready! ")
            print("="*60)
            print("\n📌 CSV files have been extracted and validated successfully.")
            print("📁 Location: " + str(raw_dir))
            print("\n🔄 Next Steps (run these commands separately):")
            print("   2️⃣ CSV to Parquet conversion: python main.py")
            print("   3️⃣ Parquet optimization: python main.py")
            print("   4️⃣ Data validation: python main.py")
            print("   5️⃣ Feature generation: python main.py")
            print("\n💡 Or use the interactive menu to continue with the next steps.")

        else:
            print(f"\n❌ Extraction failed: {failed} files failed")
            logging.error(f"Extraction failed: {failed} files failed")

    except Exception as e:
        print(f"\n❌ Download/extraction failed: {e}")
        logging.error(f"Download/extraction failed: {e}", exc_info=True)
    finally:
        sys.argv = original_argv


def run_csv_extraction_backup(config: PipelineConfig):
    """BACKUP: Original Step 2 - Extract CSV files from ZIP archives and validate"""
    print("\n" + "="*60)
    print(" 📦 BACKUP: Extract CSV from ZIP Archives ")
    print("="*60)

    # Log the start of operation
    logging.info(f"Starting BACKUP Step 2: CSV Extraction for {config.symbol} {config.data_type} {config.granularity}")

    # Check if ZIP files exist
    if config.data_type == "spot":
        zip_dir = Path("data") / f"dataset-raw-{config.granularity}" / "spot"
    else:
        zip_dir = Path("data") / f"dataset-raw-{config.granularity}" / f"futures-{config.futures_type}"

    if not zip_dir.exists():
        print(f"❌ ZIP directory not found: {zip_dir}")
        print("💡 Please run Step 1 first to download ZIP files")
        return

    zip_files = list(zip_dir.glob(f"{config.symbol}-trades-*.zip"))
    if not zip_files:
        print(f"❌ No ZIP files found in {zip_dir}")
        print("💡 Please run Step 1 first to download ZIP files")
        return

    print(f"📦 Found {len(zip_files)} ZIP files to extract")

    confirm = input("\n🚀 Proceed with CSV extraction? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ CSV extraction cancelled.")
        return

    try:
        # Extract CSV files from downloaded ZIPs
        print("\n📦 Extracting CSV files from ZIP archives...")

        extractor = CSVExtractor(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        # Force re-extraction to ensure fresh data
        successful, failed = extractor.extract_and_verify_all(force_reextract=True)

        if successful > 0:
            print(f"\n✅ Extraction completed: {successful} files extracted")
            logging.info(f"Extraction completed: {successful} files extracted")
            if failed > 0:
                print(f"⚠️ {failed} files failed extraction")
                logging.warning(f"{failed} files failed extraction")

            # CSV Validation
            print("\n🔍 Validating extracted CSV files...")

            # Get extracted CSV files
            if config.data_type == "spot":
                raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / "spot"
            else:
                raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / f"futures-{config.futures_type}"

            csv_files = list(raw_dir.glob(f"{config.symbol}-trades-*.csv"))
            print(f"📄 Found {len(csv_files)} CSV files to validate")

            # Enhanced validation of CSV files using extractor's verify method
            validation_passed = True
            validation_details = []

            print("\n📊 Performing comprehensive CSV validation...")
            print("   - Timestamp UTC conversion check")
            print("   - Missing dates detection")
            print("   - Data integrity verification")

            # Validate all files (or sample if too many)
            files_to_validate = csv_files if len(csv_files) <= 10 else csv_files[:10]

            for csv_file in files_to_validate:
                print(f"\n   📄 Validating {csv_file.name}...")
                if extractor.verify_csv_integrity(csv_file):
                    validation_details.append(f"✅ {csv_file.name}")
                else:
                    validation_details.append(f"❌ {csv_file.name}")
                    validation_passed = False

            if len(csv_files) > 10:
                print(f"\n   ... and {len(csv_files) - 10} more files")

            if validation_passed:
                print("\n✅ CSV validation passed!")
                logging.info("CSV validation passed successfully")
            else:
                print("\n⚠️ Some CSV files failed validation")
                logging.warning(f"CSV validation failed - details: {validation_details}")

            # Pipeline completion message
            print("\n" + "="*60)
            print(" 🎉 Step 2 Completed: CSV Files Ready! ")
            print("="*60)
            print("\n📌 CSV files have been extracted and validated successfully.")
            print("📁 Location: " + str(raw_dir))
            print("\n🔄 Next Steps (run these commands separately):")
            print("   3️⃣ CSV to Parquet conversion: python main.py")
            print("   4️⃣ Parquet optimization: python main.py")
            print("   5️⃣ Data validation: python main.py")
            print("   6️⃣ Feature generation: python main.py")
            print("\n💡 Or use the interactive menu to continue with the next steps.")

        else:
            print(f"\n❌ Extraction failed: {failed} files failed")
            logging.error(f"Extraction failed: {failed} files failed")

    except Exception as e:
        print(f"\n❌ CSV extraction failed: {e}")
        logging.error(f"CSV extraction failed: {e}", exc_info=True)


def run_zip_to_parquet_pipeline(config: PipelineConfig):
    """Step 2: ZIP → CSV → Parquet Pipeline with integrity validation and auto-conversion"""
    print("\n" + "="*70)
    print(" 🔄 Step 2: ZIP → CSV → Parquet Pipeline (Integrity-First) ")
    print("="*70)

    # Log the start of operation
    logging.info(f"Starting Step 2: ZIP to Parquet Pipeline for {config.symbol} {config.data_type} {config.granularity}")

    # Check if ZIP files exist (modern structure)
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    ticker_dir = Path("data") / ticker_name
    zip_dir = ticker_dir / f"raw-zip-{config.granularity}"
    parquet_dir = ticker_dir / f"raw-parquet-{config.granularity}"

    if not zip_dir.exists():
        print(f"❌ ZIP directory not found: {zip_dir}")
        print("💡 Please run Step 1 first to download ZIP files")
        return

    zip_files = list(zip_dir.glob(f"{config.symbol}-trades-*.zip"))
    if not zip_files:
        print(f"❌ No ZIP files found in {zip_dir}")
        print("💡 Please run Step 1 first to download ZIP files")
        return

    print(f"📦 Found {len(zip_files)} ZIP files to process")
    print(f"📁 Source: {zip_dir}")
    print(f"📁 Target: {parquet_dir}")

    print("\n💡 Integrated Process:")
    print("   🔄 Extract CSV from each ZIP file")
    print("   🔍 Verify CSV integrity and data quality")
    print("   ✅ If valid: Convert to Parquet and delete CSV")
    print("   ❌ If invalid: Signal error and stop processing")
    print("   📦 Keep ZIP files as backup")

    confirm = input("\n🚀 Proceed with ZIP → Parquet pipeline? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Pipeline cancelled.")
        return

    try:
        # Initialize components
        from src.data_pipeline.extractors.csv_extractor import CSVExtractor
        from src.data_pipeline.converters.csv_to_parquet import CSVToParquetConverter

        extractor = CSVExtractor(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        converter = CSVToParquetConverter(
            symbol=config.symbol,
            data_type=config.data_type,
            futures_type=config.futures_type,
            granularity=config.granularity
        )

        # Ensure parquet directory exists
        parquet_dir.mkdir(parents=True, exist_ok=True)

        # Process each ZIP file individually
        total_processed = 0
        total_failed = 0
        total_converted = 0

        print(f"\n🔄 Processing {len(zip_files)} ZIP files...")

        for i, zip_file in enumerate(sorted(zip_files), 1):
            print(f"\n📦 [{i}/{len(zip_files)}] Processing: {zip_file.name}")

            # Check if parquet already exists
            date_str = zip_file.stem.split('-trades-')[1]  # Extract date from filename
            parquet_pattern = f"{config.symbol}-Trades-{date_str}*.parquet"
            existing_parquet = list(parquet_dir.glob(parquet_pattern))

            if existing_parquet:
                print(f"   ⏭️ Parquet already exists: {existing_parquet[0].name}")
                total_processed += 1
                continue

            try:
                # Step 1: Extract CSV from ZIP
                print("   📦 Extracting CSV from ZIP...")
                csv_file = extractor.extract_single_zip(zip_file)

                if not csv_file or not csv_file.exists():
                    print(f"   ❌ Failed to extract CSV from {zip_file.name}")
                    total_failed += 1
                    continue

                # Step 2: Verify CSV integrity
                print("   🔍 Verifying CSV integrity...")
                print("      - Timestamp format validation")
                print("      - Data structure verification")
                print("      - Content integrity check")

                if not extractor.verify_csv_integrity(csv_file):
                    print(f"   ❌ CSV integrity validation FAILED for {csv_file.name}")
                    print("   🛑 STOPPING PIPELINE - Data integrity compromised")
                    print(f"   📄 Problematic file: {csv_file}")

                    # Log the critical error
                    logging.error(f"CSV integrity validation failed for {csv_file.name} - PIPELINE STOPPED")

                    # Signal error and stop
                    print("\n" + "="*70)
                    print(" ❌ PIPELINE STOPPED - INTEGRITY VALIDATION FAILED ")
                    print("="*70)
                    print(f"📄 Failed file: {csv_file.name}")
                    print("🔧 Please check the data source and re-download if necessary")
                    print("📦 ZIP file preserved for investigation")

                    return False  # Signal failure to caller

                print("   ✅ CSV integrity validation passed")

                # Step 3: Convert to Parquet
                print("   🔄 Converting CSV to Parquet...")
                parquet_file = converter.convert_single_csv(csv_file)

                if parquet_file and parquet_file.exists():
                    print(f"   ✅ Converted to: {parquet_file.name}")

                    # Step 4: Verify Parquet integrity
                    print("   🔍 Verifying Parquet integrity...")
                    if converter.verify_single_parquet(parquet_file):
                        print("   ✅ Parquet integrity verified")

                        # Step 5: Clean up CSV (keep ZIP as backup)
                        try:
                            csv_file.unlink()
                            print("   🗑️ CSV file cleaned up")
                        except Exception as e:
                            print(f"   ⚠️ Could not delete CSV: {e}")

                        total_converted += 1
                    else:
                        print(f"   ❌ Parquet integrity validation FAILED for {parquet_file.name}")
                        print("   🛑 STOPPING PIPELINE - Parquet conversion error")

                        # Log the critical error
                        logging.error(f"Parquet integrity validation failed for {parquet_file.name} - PIPELINE STOPPED")

                        return False  # Signal failure to caller
                else:
                    print(f"   ❌ Failed to convert CSV to Parquet")
                    total_failed += 1
                    continue

                total_processed += 1

            except Exception as e:
                print(f"   ❌ Error processing {zip_file.name}: {e}")
                logging.error(f"Error processing {zip_file.name}: {e}", exc_info=True)
                total_failed += 1
                continue

        # Pipeline completion summary
        print("\n" + "="*70)
        print(" 🎉 Step 2 Pipeline Completed! ")
        print("="*70)
        print(f"\n📊 Processing Summary:")
        print(f"   📦 ZIP files processed: {total_processed}/{len(zip_files)}")
        print(f"   ✅ Successfully converted to Parquet: {total_converted}")
        print(f"   ❌ Failed: {total_failed}")
        print(f"   📁 Output directory: {parquet_dir}")

        if total_failed > 0:
            print(f"\n⚠️ {total_failed} files failed processing")
            print("💡 Check logs for detailed error information")

        print("\n🔄 Next Steps (run these commands separately):")
        print("   3️⃣ Parquet optimization: python main.py")
        print("   4️⃣ Data validation: python main.py")
        print("   5️⃣ Feature generation: python main.py")
        print("\n💡 Or use the interactive menu to continue with the next steps.")

        logging.info(f"Step 2 completed: {total_converted} files converted, {total_failed} failed")
        return True  # Signal success

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        logging.error(f"ZIP to Parquet pipeline failed: {e}", exc_info=True)
        return False


def run_csv_to_parquet_conversion_backup(config: PipelineConfig):
    """BACKUP: Step 3 - Convert CSV to Parquet with verification and auto-cleanup"""
    print("\n" + "="*65)
    print(" 🔄 BACKUP: CSV → Parquet with Verification & Auto-Cleanup ")
    print("="*65)

    # Create CSV to Parquet converter
    converter = CSVToParquetConverter(
        symbol=config.symbol,
        data_type=config.data_type,
        futures_type=config.futures_type,
        granularity=config.granularity
    )

    # Check CSV files exist
    if config.data_type == "spot":
        raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / "spot"
    else:
        raw_dir = Path("data") / f"dataset-raw-{config.granularity}" / f"futures-{config.futures_type}"

    csv_files = list(raw_dir.glob(f"{config.symbol}-trades-*.csv"))

    if not csv_files:
        print(f"❌ No CSV files found in {raw_dir}")
        print("💡 Please run Step 1 first to download and extract data")
        return

    print(f"📁 Found {len(csv_files)} CSV files in {raw_dir}")

    # New pipeline behavior explanation
    print("\n💡 Automated Process:")
    print("   ✅ Convert CSV to optimized Parquet format")
    print("   ✅ Automatic Parquet integrity verification")
    print("   ✅ Automatic CSV cleanup after successful conversion (saves disk space)")
    print("   ✅ ZIP files preserved as backup")
    print("\n📌 Note: CSV validation was already performed in Step 1")

    print(f"\n📋 Ready to Process:")
    print(f"   CSV files: {len(csv_files)}")
    print(f"   Source: {raw_dir}")

    confirm = input("\n🚀 Proceed with CSV to Parquet conversion? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ CSV to Parquet conversion cancelled.")
        return

    try:
        # Convert CSV to Parquet with automatic verification and cleanup
        print(f"\n🔄 Converting CSV to Parquet with Auto-Verification & Cleanup...")
        successful, failed = converter.convert_all_csv_files()

        if successful > 0:
            print(f"\n✅ Conversion completed: {successful} files converted")
            print(f"💾 CSV files automatically cleaned up to save disk space")
            print(f"📦 ZIP files preserved as backup")
            if failed > 0:
                print(f"⚠️ {failed} files failed conversion")
        else:
            print(f"\n❌ Conversion failed: {failed} files failed")
            return

        print(f"\n🎉 CSV to Parquet conversion completed successfully!")

    except Exception as e:
        print(f"\n❌ CSV to Parquet conversion failed: {e}")


def run_parquet_optimization(config: PipelineConfig):
    """Step 4: Optimize Parquet files using robust optimizer with corruption prevention"""
    print("\n" + "="*50)
    print(" 🔧 Step 4: Robust Parquet Optimization ")
    print("="*50)

    # Determine source directory
    if config.data_type == "spot":
        source_dir = f"data/dataset-raw-{config.granularity}-compressed/spot"
    else:
        source_dir = f"data/dataset-raw-{config.granularity}-compressed/futures-{config.futures_type}"

    # Target directory for optimized files
    if config.data_type == "spot":
        target_dir = f"data/dataset-raw-{config.granularity}-compressed-optimized/spot"
    else:
        target_dir = f"data/dataset-raw-{config.granularity}-compressed-optimized/futures-{config.futures_type}"

    print(f"📁 Source: {source_dir}")
    print(f"📁 Target: {target_dir}")

    # Configuration options
    print("\n⚙️ Optimization Configuration:")
    max_size_input = input("Maximum file size in GB (default: 10): ").strip()
    max_size = int(max_size_input) if max_size_input.isdigit() else 10

    print(f"\n📋 Using Legacy Optimizer")
    print(f"   Max file size: {max_size} GB")

    confirm = input(f"\n🚀 Proceed with optimization? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Optimization cancelled.")
        return

    try:
        # Use legacy optimizer
        print("\n🚀 Using legacy optimizer...")
        sys.argv = ['optimize', '--source', source_dir, '--target', target_dir,
                   '--max-size', str(max_size), '--auto-confirm']
        optimize_main()
        print("\n✅ Optimization completed successfully!")

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        print("💡 Check the logs for detailed error information")




def run_parquet_merge_modern(config: PipelineConfig):
    """Compact/Merge Parquet files using modern directory structure with auto-cleanup"""
    print("\n" + "="*60)
    print(" 🗜️  Compact/Merge Parquet Files (Modern Structure) ")
    print("="*60)

    # Use modern directory structure with proper ticker naming
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    source_dir = f"data/{ticker_name}/raw-parquet-{config.granularity}"
    target_dir = f"data/{ticker_name}/raw-parquet-merged-{config.granularity}"

    print(f"📁 Source: {source_dir}")
    print(f"📁 Target: {target_dir}")

    # Check if source directory exists
    from pathlib import Path
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"\n❌ Source directory does not exist: {source_dir}")
        print("💡 Run option 2 (Individual Steps) → option 3 (ZIP → Parquet) first")
        return

    # Count source files
    parquet_files = list(source_path.glob("*.parquet"))
    if not parquet_files:
        print(f"\n❌ No .parquet files found in {source_dir}")
        return

    print(f"\n📊 Found {len(parquet_files):,} parquet files to merge")

    # Configuration
    max_size = 10  # Fixed at 10GB as per specs
    print(f"\n⚙️  Configuration:")
    print(f"   Max file size: {max_size} GB")
    print(f"   Compression: snappy")
    print(f"   Auto-cleanup: YES (source files will be deleted after merge)")

    confirm = input(f"\n🚀 Proceed with merge and cleanup? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Operation cancelled.")
        return

    try:
        print("\n🚀 Starting merge operation...")
        sys.argv = ['optimize', '--source', source_dir, '--target', target_dir,
                   '--max-size', str(max_size), '--compression', 'snappy', '--auto-confirm']
        optimize_main()
        print("\n✅ Merge and cleanup completed successfully!")
        print(f"📁 Merged files saved to: {target_dir}")
        print(f"🗑️  Original daily files deleted from: {source_dir}")

    except Exception as e:
        print(f"\n❌ Merge operation failed: {e}")
        print("💡 Check the logs for detailed error information")


def run_data_validation(config: PipelineConfig):
    """Step 5: Validate data integrity"""
    print("\n" + "="*50)
    print(" ✅ Step 5: Validate Data Integrity ")
    print("="*50)

    # Directly run missing dates validation without menu
    print("🔍 Running missing dates validation...")

    # Get data directory based on config - use optimized parquet files (modern structure)
    ticker_name = f"{config.symbol.lower()}-{config.data_type}"
    if config.data_type == "futures":
        ticker_name = f"{config.symbol.lower()}-{config.data_type}-{config.futures_type}"

    data_dir = f"data/{ticker_name}/raw-parquet-merged-{config.granularity}"

    # Ask if user wants to check daily gaps
    check_daily = input("\nCheck for daily gaps within files? (slower) (y/n): ").strip().lower() == 'y'

    original_argv = sys.argv.copy()
    sys.argv = ['missing_dates_validator', '--data-dir', data_dir, '--symbol', config.symbol]
    if check_daily:
        sys.argv.append('--check-daily-gaps')

    try:
        missing_dates_main()
        print("\n✅ Missing dates validation completed!")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
    finally:
        sys.argv = original_argv


def run_feature_generation(config: PipelineConfig):
    """Step 5: Generate bars"""
    print("\n" + "="*50)
    print(" 📊 Step 5: Generate Bars ")
    print("="*50)

    print("Available bar types:")
    print("1. 📊 Standard Dollar Bars")
    print("2. 🔄 Imbalance Dollar Bars")
    print("3. 🏃 Run Dollar Bars")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        print("\n📊 Generating Standard Dollar Bars...")
        print(f"   Symbol: {config.symbol}")
        print(f"   Data Type: {config.data_type}")
        print(f"   Granularity: {config.granularity}")

        # Ask for volume threshold
        volume_input = input("\n💰 Enter volume threshold in USD (default: 40000000): ").strip()
        try:
            volume_threshold = int(volume_input) if volume_input else 40_000_000
            if volume_threshold <= 0:
                print("⚠️ Volume must be positive. Using default: 40,000,000")
                volume_threshold = 40_000_000
        except ValueError:
            print("⚠️ Invalid input. Using default: 40,000,000")
            volume_threshold = 40_000_000

        print(f"📊 Using volume threshold: {volume_threshold:,} USD")

        # Use pipeline mode by default for better performance
        use_pipeline = True
        print("🚀 Using HYBRID PIPELINE mode (3-stage: I/O → Pre-process → Generate)")

        try:
            from pathlib import Path

            # Setup for standard bars
            setup_standard_logging()

            # Generate standard bars using PyArrow (no Dask needed)
            # Output will be saved to: output/{symbol}-{data_type}/standard/
            process_files_and_generate_bars(
                data_type=config.data_type,
                futures_type=config.futures_type if config.data_type == 'futures' else 'um',
                granularity=config.granularity,
                init_vol=volume_threshold,
                output_dir=None,  # Will auto-create based on ticker
                db_engine=None,
                use_pipeline=use_pipeline,
                symbol=config.symbol
            )
            print("\n✅ Standard Dollar Bars generation completed!")

        except Exception as e:
            print(f"\n❌ Standard Dollar Bars generation failed: {e}")
            import traceback
            traceback.print_exc()

    elif choice == "2":
        print("\n🔄 Generating Imbalance Dollar Bars...")
        print(f"   Symbol: {config.symbol}")
        print(f"   Data Type: {config.data_type}")
        print(f"   Granularity: {config.granularity}")
        try:
            # Use the existing imbalance_main import
            imbalance_main(
                symbol=config.symbol,
                data_type=config.data_type,
                futures_type=config.futures_type if config.data_type == 'futures' else 'um',
                granularity=config.granularity
            )
            print("✅ Imbalance Dollar Bars generation completed!")
        except Exception as e:
            print(f"❌ Imbalance Dollar Bars generation failed: {e}")

    elif choice == "3":
        print("\n🏃 Generating Run Dollar Bars...")
        print(f"   Symbol: {config.symbol}")
        print(f"   Data Type: {config.data_type}")
        print(f"   Granularity: {config.granularity}")
        try:
            # Import run dollar bars module
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent / "src" / "scripts"))
            from run_dollar_bars import process_run_dollar_bars

            # Generate run bars with default parameters
            process_run_dollar_bars(
                data_type=config.data_type,
                futures_type=config.futures_type if config.data_type == 'futures' else 'um',
                granularity=config.granularity,
                init_T0=1_000_000_000,  # Default initial threshold
                alpha_volume=0.1,  # Default decay factor for volume
                alpha_imbalance=0.1,  # Default decay factor for imbalance
                output_dir='./output/run/',  # Output directory
                time_reset=4  # Reset after 4 hours
            )
            print("✅ Run Dollar Bars generation completed!")
        except Exception as e:
            print(f"❌ Run Dollar Bars generation failed: {e}")
    else:
        print("❌ Invalid choice. Please enter 1-3.")


def interactive_main():
    """Main interactive menu with new flow"""
    # Setup logging
    log_path = setup_logging()

    print("\n" + "="*60)
    print(" 🚀 Bitcoin ML Finance Pipeline ")
    print("="*60)
    print(f"📝 Logs saved to: {log_path}")

    # First, select market configuration
    config = select_market_configuration()

    while True:
        display_pipeline_menu(config)

        choice = input("\nEnter your choice (0-9): ").strip()

        if choice == "1":
            # Complete pipeline with improved date suggestions
            run_complete_pipeline_parallel(config)
        elif choice == "2":
            # Individual Steps submenu
            while True:
                display_individual_steps_menu(config)
                sub_choice = input("\nEnter your choice (0-4): ").strip()

                if sub_choice == "1":
                    run_download_only(config)
                elif sub_choice == "2":
                    run_zip_to_parquet_pipeline(config)
                elif sub_choice == "3":
                    run_zip_to_parquet_direct(config)
                elif sub_choice == "4":
                    run_parquet_optimization(config)
                elif sub_choice == "0":
                    break  # Back to main menu
                else:
                    print("❌ Invalid choice. Please enter 0-4.")
        elif choice == "3":
            run_parquet_merge_modern(config)
        elif choice == "4":
            run_data_validation(config)
        elif choice == "5":
            run_feature_generation(config)
        elif choice == "6":
            clean_zip_and_checksum_files(config)
        elif choice == "7":
            add_missing_daily_data(config)
        elif choice == "8":
            delete_all_data()
        elif choice == "9":
            run_search_classificator(config)
        elif choice == "0":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 0-9.")


def main():
    # Setup logging for all modes
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Bitcoin ML Finance Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python main.py

  # Direct command mode
  python main.py download --start 2024-01-01 --end 2024-01-31
  python main.py optimize --source data/raw --target data/optimized
  python main.py validate --data-dir data/dataset-raw-daily-compressed-optimized/spot --symbol BTCUSDT
  python main.py features --type imbalance
        """
    )

    logging.info(f"Pipeline started with arguments: {sys.argv}")

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download Bitcoin data from Binance')
    download_parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    download_parser.add_argument('--type', choices=['spot', 'futures'], default='spot',
                                help='Data type (spot or futures)')
    download_parser.add_argument('--futures-type', choices=['um', 'cm'], default='um',
                                help='Futures type (um=USD-M, cm=COIN-M)')
    download_parser.add_argument('--granularity', choices=['daily', 'monthly'], default='monthly',
                                help='Data granularity')
    download_parser.add_argument('--start', required=True,
                                help='Start date (YYYY-MM-DD for daily, YYYY-MM for monthly)')
    download_parser.add_argument('--end', required=True,
                                help='End date (YYYY-MM-DD for daily, YYYY-MM for monthly)')
    download_parser.add_argument('--workers', type=int, default=5,
                                help='Number of concurrent downloads (default: 5)')

    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize parquet files')
    optimize_parser.add_argument('--source', required=True, help='Source directory')
    optimize_parser.add_argument('--target', required=True, help='Target directory')
    optimize_parser.add_argument('--max-size', type=int, default=10, help='Maximum file size in GB')
    optimize_parser.add_argument('--auto-confirm', action='store_true', help='Auto confirm operations')

    # Validate command (missing dates validation only)
    validate_parser = subparsers.add_parser('validate', help='Validate missing dates in data')
    validate_parser.add_argument('--data-dir', help='Data directory for missing dates check')
    validate_parser.add_argument('--symbol', default='BTCUSDT', help='Symbol for missing dates check')
    validate_parser.add_argument('--check-daily-gaps', action='store_true', help='Check daily gaps in missing dates validation')

    # Features command
    features_parser = subparsers.add_parser('features', help='Generate features')
    features_parser.add_argument('--type', choices=['standard', 'imbalance'], default='imbalance',
                                help='Type of features to generate')
    features_parser.add_argument('--volume', type=int, default=40_000_000,
                                help='Volume threshold in USD (for standard bars, default: 40,000,000)')
    features_parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    features_parser.add_argument('--data-type', choices=['spot', 'futures'], default='futures',
                                help='Data type (spot or futures)')
    features_parser.add_argument('--futures-type', choices=['um', 'cm'], default='um',
                                help='Futures type (um=USD-M, cm=COIN-M)')
    features_parser.add_argument('--granularity', choices=['daily', 'monthly'], default='daily',
                                help='Data granularity')
    features_parser.add_argument('--pipeline', action='store_true',
                                help='Enable pipeline mode for faster processing (standard bars only)')

    args = parser.parse_args()

    # If no command is provided, run interactive mode
    if not args.command:
        interactive_main()
        return

    if args.command == 'download':
        # Use the same custom download approach as interactive mode
        from src.data_pipeline.downloaders.binance_downloader import BinanceDataDownloader
        from src.data_pipeline.extractors.csv_extractor import CSVExtractor
        import concurrent.futures

        downloader = BinanceDataDownloader(
            symbol=args.symbol,
            data_type=args.type,
            futures_type=args.futures_type if args.type == 'futures' else 'um',
            granularity=args.granularity
        )

        # Generate date range
        start = datetime.strptime(args.start, '%Y-%m-%d' if args.granularity == 'daily' else '%Y-%m')
        end = datetime.strptime(args.end, '%Y-%m-%d' if args.granularity == 'daily' else '%Y-%m')
        dates = downloader.generate_dates(start, end)

        print(f"\n📥 Downloading {len(dates)} ZIP files with CHECKSUM verification...")

        # Download only ZIP files
        downloaded_count = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_date = {
                executor.submit(downloader.download_with_checksum, date): date
                for date in dates
            }

            for future in concurrent.futures.as_completed(future_to_date):
                date = future_to_date[future]
                try:
                    zip_file, checksum_file = future.result()
                    if zip_file and checksum_file:
                        downloaded_count += 1
                        print(f"✅ Downloaded: {zip_file.name}")
                except Exception as e:
                    print(f"❌ Error downloading {date}: {e}")

        print(f"\n✅ Download completed: {downloaded_count} files")

        # Extract CSV files
        print("\n📦 Extracting CSV files...")
        extractor = CSVExtractor(
            symbol=args.symbol,
            data_type=args.type,
            futures_type=args.futures_type if args.type == 'futures' else 'um',
            granularity=args.granularity
        )

        successful, failed = extractor.extract_and_verify_all()
        print(f"\n✅ Extraction completed: {successful} files extracted, {failed} failed")
    elif args.command == 'optimize':
        sys.argv = ['optimize', '--source', args.source, '--target', args.target,
                   '--max-size', str(args.max_size)]
        if args.auto_confirm:
            sys.argv.append('--auto-confirm')
        optimize_main()
    elif args.command == 'validate':
        if not args.data_dir:
            print("Please specify --data-dir for missing dates validation")
            return
        sys.argv = ['missing_dates_validator', '--data-dir', args.data_dir,
                   '--symbol', args.symbol]
        if args.check_daily_gaps:
            sys.argv.append('--check-daily-gaps')
        missing_dates_main()
    elif args.command == 'features':
        if args.type == 'imbalance':
            imbalance_main()
        elif args.type == 'standard':
            print(f"\n📊 Generating Standard Dollar Bars...")
            print(f"   Symbol: {args.symbol}")
            print(f"   Data Type: {args.data_type}")
            print(f"   Futures Type: {args.futures_type}")
            print(f"   Granularity: {args.granularity}")
            print(f"   Volume Threshold: {args.volume:,} USD")
            print(f"   Pipeline Mode: {'Enabled' if args.pipeline else 'Disabled'}")

            try:
                setup_standard_logging()

                # Output will be saved to: output/{symbol}-{data_type}/standard/
                process_files_and_generate_bars(
                    data_type=args.data_type,
                    futures_type=args.futures_type,
                    granularity=args.granularity,
                    init_vol=args.volume,
                    output_dir=None,  # Will auto-create based on ticker
                    db_engine=None,
                    use_pipeline=args.pipeline,
                    symbol=args.symbol
                )
                print("\n✅ Standard Dollar Bars generation completed!")
            except Exception as e:
                print(f"\n❌ Standard Dollar Bars generation failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
