#!/usr/bin/env python3
"""
Main entry point for the Bitcoin ML Finance pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_pipeline.downloaders.binance_downloader import main as download_main
from data_pipeline.processors.parquet_optimizer import main as optimize_main
from data_pipeline.validators.quick_validator import main as quick_validate_main
from data_pipeline.validators.advanced_validator import main as advanced_validate_main
from features.imbalance_bars import main as imbalance_main


def get_data_date_range(symbol, data_type, futures_type, granularity):
    """Get the first and last available dates from existing data using progress file"""
    try:
        # Check progress file for processed dates (more reliable than parsing corrupted parquet timestamps)
        progress_file = Path("datasets") / f"download_progress_{symbol}_{data_type}_{granularity}.json"

        if progress_file.exists():
            import json
            with open(progress_file, 'r') as f:
                progress = json.load(f)

            downloaded = progress.get('downloaded', [])
            if downloaded:
                # Sort dates and get first/last
                sorted_dates = sorted(downloaded)
                return sorted_dates[0], sorted_dates[-1]

        # Fallback: check if parquet files exist (even if timestamps are corrupted)
        if data_type == "spot":
            compressed_dir = Path("datasets") / f"dataset-raw-{granularity}-compressed" / "spot"
        else:  # futures
            compressed_dir = Path("datasets") / f"dataset-raw-{granularity}-compressed" / f"futures-{futures_type}"

        if compressed_dir.exists():
            parquet_files = list(compressed_dir.glob(f"{symbol}-Trades-*.parquet"))
            if parquet_files:
                # If we have parquet files but no valid progress info,
                # let user know data exists but we can't determine range
                return "data-exists", "data-exists"

        return None, None

    except Exception:
        return None, None


def get_download_options():
    """Interactive prompts for download options"""
    print("\n" + "="*50)
    print(" Bitcoin ML Finance - Download Configuration ")
    print("="*50)

    # Symbol selection
    print("\nSelect trading pair symbol:")
    print("1. BTCUSDT (default)")
    print("2. ETHUSDT")
    print("3. Other symbol")

    while True:
        symbol_choice = input("\nEnter your choice (1-3) or press Enter for BTCUSDT: ").strip()
        if symbol_choice == "" or symbol_choice == "1":
            symbol = "BTCUSDT"
            break
        elif symbol_choice == "2":
            symbol = "ETHUSDT"
            break
        elif symbol_choice == "3":
            symbol = input("Enter symbol: ").strip().upper()
            if symbol:
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
            data_type = "spot"
            futures_type = "um"  # default
            break
        elif type_choice == "2":
            data_type = "futures"
            futures_type = "um"
            break
        elif type_choice == "3":
            data_type = "futures"
            futures_type = "cm"
            # Adjust symbol for COIN-M if needed
            if symbol == "BTCUSDT":
                symbol = "BTCUSD_PERP"
                print(f"Note: For COIN-M futures, using symbol: {symbol}")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    # Granularity selection
    print("\nSelect data granularity:")
    print("1. Daily")
    print("2. Monthly")

    while True:
        gran_choice = input("\nEnter your choice (1-2): ").strip()
        if gran_choice == "1":
            granularity = "daily"
            break
        elif gran_choice == "2":
            granularity = "monthly"
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    return symbol, data_type, futures_type, granularity


def interactive_main():
    """Main interactive menu"""
    print("\n" + "="*60)
    print(" 🚀 Bitcoin ML Finance Pipeline ")
    print("="*60)

    while True:
        print("\nSelect an action:")
        print("1. 📥 Download data from Binance")
        print("2. 🔧 Optimize parquet files")
        print("3. ✅ Validate data integrity")
        print("4. 📊 Generate features")
        print("5. 🚪 Exit")

        choice = input("\nEnter your choice (1-5): ").strip()

        if choice == "1":
            run_interactive_download()
        elif choice == "2":
            run_interactive_optimize()
        elif choice == "3":
            run_interactive_validate()
        elif choice == "4":
            run_interactive_features()
        elif choice == "5":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, 4, or 5.")


def run_interactive_download():
    """Interactive download process"""
    print("\n" + "="*50)
    print(" 📥 Data Download Configuration ")
    print("="*50)

    # Get symbol, type, futures-type, granularity
    symbol, data_type, futures_type, granularity = get_download_options()

    # Get existing data range for suggestions
    first_available, last_available = get_data_date_range(symbol, data_type, futures_type, granularity)

    # Get date range
    print(f"\n📅 Enter date range for {granularity} data:")

    if granularity == 'daily':
        date_format = "YYYY-MM-DD"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            example_start = first_available
            example_end = last_available
        elif first_available == "data-exists":
            print(f"💡 Existing {symbol} {data_type} data found (dates not determinable)")
            # Use recent dates as examples
            example_start = "2024-01-01"
            example_end = "2024-01-31"
        else:
            # Show data availability info from Binance
            if data_type == "spot":
                print("📊 Binance spot data available from 2017-08-17")
                example_start = "2024-01-01"
                example_end = "2024-01-31"
            elif data_type == "futures" and futures_type == "um":
                print("📊 Binance futures USD-M data available from 2019-09-08")
                example_start = "2024-01-01"
                example_end = "2024-01-31"
            else:  # futures cm
                print("📊 Binance futures COIN-M data available from 2020-02-10")
                example_start = "2024-01-01"
                example_end = "2024-01-31"
    else:
        date_format = "YYYY-MM"
        if first_available and last_available and first_available != "data-exists":
            print(f"💡 Available data range: {first_available} to {last_available}")
            example_start = first_available
            example_end = last_available
        elif first_available == "data-exists":
            print(f"💡 Existing {symbol} {data_type} data found (dates not determinable)")
            # Use recent dates as examples
            example_start = "2024-01"
            example_end = "2024-12"
        else:
            # Show data availability info from Binance
            if data_type == "spot":
                print("📊 Binance spot data available from 2017-08")
                example_start = "2024-01"
                example_end = "2024-12"
            elif data_type == "futures" and futures_type == "um":
                print("📊 Binance futures USD-M data available from 2019-09")
                example_start = "2024-01"
                example_end = "2024-12"
            else:  # futures cm
                print("📊 Binance futures COIN-M data available from 2020-02")
                example_start = "2024-01"
                example_end = "2024-12"

    while True:
        start_date = input(f"Start date ({date_format}, e.g., {example_start}): ").strip()
        if start_date:
            break
        print("❌ Start date is required.")

    while True:
        end_date = input(f"End date ({date_format}, e.g., {example_end}): ").strip()
        if end_date:
            break
        print("❌ End date is required.")

    # Get workers
    workers_input = input("\nNumber of concurrent downloads (default: 5): ").strip()
    workers = int(workers_input) if workers_input.isdigit() else 5

    # Show summary and confirm
    print(f"\n" + "="*50)
    print(" 📋 Download Configuration Summary ")
    print("="*50)
    print(f"Symbol: {symbol}")
    print(f"Type: {data_type.upper()}")
    if data_type == 'futures':
        print(f"Futures Type: {futures_type.upper()}")
    print(f"Granularity: {granularity.upper()}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Workers: {workers}")
    print("="*50)

    confirm = input("\n🚀 Proceed with download? (y/n): ").strip().lower()
    if confirm != 'y':
        print("❌ Download cancelled.")
        return

    # Build command line arguments for binance_downloader
    download_args = [
        'binance_downloader.py', 'download',
        '--symbol', symbol,
        '--type', data_type,
        '--granularity', granularity,
        '--start', start_date,
        '--end', end_date,
        '--workers', str(workers)
    ]

    # Add futures type if needed
    if data_type == 'futures':
        download_args.extend(['--futures-type', futures_type])

    # Set sys.argv for the downloader
    import sys
    original_argv = sys.argv.copy()
    sys.argv = download_args

    try:
        download_main()
        print("\n✅ Download completed!")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
    finally:
        sys.argv = original_argv


def run_interactive_optimize():
    """Interactive optimize process"""
    print("\n🔧 Parquet Optimization")
    print("This feature optimizes parquet files by combining them into larger files.")
    print("❌ Not implemented in interactive mode yet.")
    print("💡 Use: python main.py optimize --source <source> --target <target>")


def run_interactive_validate():
    """Interactive validate process"""
    print("\n✅ Data Validation")
    print("Choose validation type:")
    print("1. Quick validation")
    print("2. Advanced validation")

    choice = input("\nEnter your choice (1-2): ").strip()

    if choice == "1":
        print("🔍 Running quick validation...")
        try:
            quick_validate_main()
            print("✅ Quick validation completed!")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
    elif choice == "2":
        print("🔍 Running advanced validation...")
        output_dir = input("Output directory (default: reports): ").strip() or "reports"
        base_path = input("Base path (default: .): ").strip() or "."

        import sys
        original_argv = sys.argv.copy()
        sys.argv = ['validate', '--base-path', base_path, '--output-dir', output_dir]

        try:
            advanced_validate_main()
            print("✅ Advanced validation completed!")
        except Exception as e:
            print(f"❌ Validation failed: {e}")
        finally:
            sys.argv = original_argv
    else:
        print("❌ Invalid choice.")


def run_interactive_features():
    """Interactive features process"""
    print("\n📊 Feature Generation")
    print("Available features:")
    print("1. Imbalance bars")

    choice = input("\nEnter your choice (1): ").strip()

    if choice == "1":
        print("📊 Generating imbalance bars...")
        try:
            imbalance_main()
            print("✅ Feature generation completed!")
        except Exception as e:
            print(f"❌ Feature generation failed: {e}")
    else:
        print("❌ Invalid choice.")


def main():
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
  python main.py validate --quick
  python main.py features --type imbalance
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download command
    download_parser = subparsers.add_parser('download', help='Download Bitcoin data from Binance')
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

    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data integrity')
    validate_parser.add_argument('--quick', action='store_true', help='Run quick validation')
    validate_parser.add_argument('--advanced', action='store_true', help='Run advanced validation')
    validate_parser.add_argument('--output-dir', default='reports', help='Output directory for reports')
    validate_parser.add_argument('--base-path', default='.', help='Base path for data')

    # Features command
    features_parser = subparsers.add_parser('features', help='Generate features')
    features_parser.add_argument('--type', choices=['imbalance'], default='imbalance',
                                help='Type of features to generate')

    args = parser.parse_args()

    # If no command is provided, run interactive mode
    if not args.command:
        interactive_main()
        return

    if args.command == 'download':
        # Get interactive options
        symbol, data_type, futures_type, granularity = get_download_options()

        # Show summary and confirm
        print(f"\n" + "="*50)
        print(" Download Configuration Summary ")
        print("="*50)
        print(f"Symbol: {symbol}")
        print(f"Type: {data_type.upper()}")
        if data_type == 'futures':
            print(f"Futures Type: {futures_type.upper()}")
        print(f"Granularity: {granularity.upper()}")
        print(f"Start Date: {args.start}")
        print(f"End Date: {args.end}")
        print(f"Workers: {args.workers}")
        print("="*50)

        confirm = input("\nProceed with download? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Download cancelled.")
            return

        # Build command line arguments for binance_downloader
        download_args = [
            'binance_downloader.py', 'download',
            '--symbol', symbol,
            '--type', data_type,
            '--granularity', granularity,
            '--start', args.start,
            '--end', args.end,
            '--workers', str(args.workers)
        ]

        # Add futures type if needed
        if data_type == 'futures':
            download_args.extend(['--futures-type', futures_type])

        # Set sys.argv for the downloader
        sys.argv = download_args
        download_main()
    elif args.command == 'optimize':
        sys.argv = ['optimize', '--source', args.source, '--target', args.target,
                   '--max-size', str(args.max_size)]
        if args.auto_confirm:
            sys.argv.append('--auto-confirm')
        optimize_main()
    elif args.command == 'validate':
        if args.quick:
            quick_validate_main()
        elif args.advanced:
            sys.argv = ['validate', '--base-path', args.base_path,
                       '--output-dir', args.output_dir]
            advanced_validate_main()
        else:
            print("Please specify --quick or --advanced for validation")
    elif args.command == 'features':
        if args.type == 'imbalance':
            imbalance_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()