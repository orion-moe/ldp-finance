# Data Directory Structure

This directory contains all trading data organized by ticker and market type.

## Organization

Each ticker gets its own directory following this pattern:
```
{symbol}-{market_type}/
```

Examples:
- `btcusdt-spot/` - BTCUSDT spot market
- `btcusdt-futures-um/` - BTCUSDT USD-M futures
- `ethusdt-spot/` - ETHUSDT spot market

## Subdirectory Structure

Inside each ticker directory:

```
btcusdt-spot/
├── raw-daily/                  # Downloaded ZIP/CSV files (daily granularity)
├── raw-monthly/                # Downloaded ZIP/CSV files (monthly granularity)
├── optimized-daily/            # Processed Parquet files (daily)
├── optimized-monthly/          # Processed Parquet files (monthly)
├── optimized-daily-merged/     # Large optimized Parquet files (~10GB each)
├── optimized-monthly-merged/   # Large optimized Parquet files (~10GB each)
├── logs/                       # Download and processing logs
├── download_progress_daily.json    # Progress tracking for daily downloads
├── download_progress_monthly.json  # Progress tracking for monthly downloads
└── failed_downloads.txt        # List of failed download attempts
```

## Git Protection

All files in this directory are automatically ignored by git via `.gitignore`:
- ✅ No raw data will ever be committed
- ✅ No progress files will be committed
- ✅ Only the directory structure is tracked

## Usage

When you run:
```bash
python main.py download --symbol BTCUSDT --type spot --granularity daily
```

Data will be stored in:
```
data/btcusdt-spot/raw-daily/           # Downloaded files
data/btcusdt-spot/optimized-daily/     # Processed parquet files
data/btcusdt-spot/logs/                # Download logs
```

## Benefits of This Structure

1. **Isolated**: Each ticker's data is completely separated
2. **Scalable**: Easy to add new tickers without affecting existing ones
3. **Clean**: Simple to understand and navigate
4. **Safe**: Git will never track your data files
5. **Organized**: All related files for a ticker are together
