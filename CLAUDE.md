# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin trading data pipeline for machine learning and quantitative finance. The pipeline downloads historical tick-level trade data from Binance, processes it into optimized Parquet format, and generates specialized financial features called "information-driven bars" (dollar bars, imbalance bars) as described in Advances in Financial Machine Learning by Marcos López de Prado.

The codebase handles massive datasets (multi-GB to TB scale) with memory-efficient processing using PyArrow, Numba JIT compilation, and parallel processing.

## Running the Pipeline

### Interactive Mode (Recommended)
```bash
python main.py
```
Launches an interactive menu system with options for:
1. Download data from Binance
2. Convert to Parquet
3. Optimize Parquet files (merge small files)
4. Validate data integrity
5. Generate features (dollar bars)
6. Run full pipeline end-to-end
7. Cleanup utilities
8. Delete all data

### Direct Command Mode
```bash
# Download data (requires --start and --end dates)
python main.py download --start 2024-01-01 --end 2024-01-31 --type spot --granularity daily

# Optimize parquet files (merge and compress)
python main.py optimize --source data/btcusdt-spot/raw-parquet-daily --target data/btcusdt-spot/raw-parquet-merged-daily

# Validate data integrity
python main.py validate --data-dir data/btcusdt-spot/raw-parquet-merged-daily --symbol BTCUSDT

# Generate features (dollar bars)
python main.py features --type standard --volume 40000000
python main.py features --type imbalance
```

### Installation
```bash
pip install -r requirements.txt
```

## Architecture

### Data Flow Pipeline

```
1. Download (Binance API)
   └─> data/{ticker}/raw-zip-{granularity}/*.zip

2. Extract & Convert
   └─> data/{ticker}/raw-parquet-{granularity}/*.parquet
       (one file per day/month)

3. Optimize & Merge
   └─> data/{ticker}/raw-parquet-merged-{granularity}/*.parquet
       (merged into ~10GB chunks)

4. Generate Features
   └─> src/output/standard/ or src/output/imbalance/
       (information-driven bars)
```

### Directory Structure

```
data/
├── {symbol}-{type}/              # e.g., btcusdt-spot, btcusdt-futures-um
│   ├── raw-zip-{granularity}/    # Downloaded ZIP files
│   ├── raw-parquet-{granularity}/# Extracted daily/monthly parquet
│   ├── raw-parquet-merged-{granularity}/  # Optimized merged files
│   └── logs/                     # Per-ticker logs
├── logs/                         # Pipeline execution logs
└── download_progress_*.json      # Download progress tracking

src/
├── data_pipeline/
│   ├── downloaders/              # BinanceDataDownloader
│   ├── converters/               # ZIP→Parquet conversion
│   ├── processors/               # Parquet optimization/merging
│   ├── validators/               # Data integrity checks
│   ├── extractors/               # CSV extraction (legacy)
│   └── utils/                    # Parallel processing utilities
└── features/
    └── bars/                     # Dollar bars generators
        ├── standard_dollar_bars.py       # Standard bars
        ├── imbalance_bars.py            # Imbalance bars
        └── imbalance_dollar_bars.py     # Imbalance dollar bars

main.py                           # Main entry point with CLI and interactive mode
```

### Key Components

**BinanceDataDownloader** (src/data_pipeline/downloaders/binance_downloader.py)
- Downloads historical tick data from Binance public data API
- Supports spot and futures (USD-M, COIN-M) markets
- Daily or monthly granularity
- Parallel downloads with progress tracking
- Checksum verification
- Resume capability via JSON progress files

**ZipToParquetStreamer** (src/data_pipeline/converters/zip_to_parquet_streamer.py)
- Streams ZIP files to Parquet without full extraction to disk
- Memory-efficient for large files
- Handles CSV format variations (with/without headers)

**EnhancedParquetOptimizer** (src/data_pipeline/processors/parquet_optimizer.py)
- Merges small daily Parquet files into larger chunks (~10GB)
- Reduces file count from thousands to dozens
- Improves query performance
- Validates merged data integrity

**Standard Dollar Bars** (src/features/bars/standard_dollar_bars.py)
- Generates fixed-volume dollar bars (default: $40M USD)
- Two processing modes:
  - Sequential: Simple, stable, ~1-2GB memory
  - Pipeline: 1.5-2x faster, ~3-4GB memory (parallel I/O, preprocessing, bar generation)
- Uses Numba JIT for performance-critical loops
- PyArrow chunked reading for memory efficiency
- Outputs to database or Parquet files

**Imbalance Bars** (src/features/bars/imbalance_bars.py)
- Generates imbalance-based dollar bars
- Uses Dask for distributed processing
- More sophisticated feature engineering based on order flow imbalance

### Memory Management

The codebase is designed to handle datasets too large to fit in RAM:

1. **Chunked Processing**: Files are read in chunks (standard_dollar_bars.py uses PyArrow chunked reading)
2. **Streaming Conversions**: ZIP→Parquet conversion streams data without full extraction
3. **Progress Tracking**: JSON files track progress so processes can resume after crashes
4. **Pipeline Architecture**: Standard bars use a 3-stage pipeline (I/O → Preprocess → Generate) to overlap operations

### Data Structures

**Raw Trade Data Schema**:
```
trade_id: int64
price: float64
qty: float64
quoteQty: float64
time: int64 (milliseconds since epoch)
isBuyerMaker: bool
isBestMatch: bool (may be absent in some files)
```

**Dollar Bars Output**:
```
time_open: datetime
time_close: datetime
price_open: float
price_high: float
price_low: float
price_close: float
volume: float
dollar_volume: float
num_trades: int
side: int (1=buy, -1=sell for imbalance bars)
```

## Development Notes

### Progress Tracking

The pipeline tracks progress by checking existing files directly:
- Downloads are verified by checking for ZIP files in `data/{ticker}/raw-zip-{granularity}/`
- Conversions are verified by checking for Parquet files in `data/{ticker}/raw-parquet-{granularity}/`

The system automatically skips files that have already been downloaded or converted, making it efficient to resume interrupted operations.

### Data Directory Naming

The pipeline supports two directory structures:
- **Modern** (preferred): `data/btcusdt-spot/`, `data/btcusdt-futures-um/`
- **Legacy** (backward compatible): `data/dataset-raw-daily/spot/`, `data/dataset-raw-monthly/futures-um/`

New code should use the modern ticker-based structure.

### Logging

All operations are logged to:
- Console: INFO level with timestamps
- File: `data/logs/pipeline_YYYYMMDD_HHMMSS.log` (rotating, max 50MB, 10 backups)

### Performance Considerations

1. **Download workers**: Default 5 concurrent downloads. Increase with `--workers` for faster downloads on good connections.
2. **Parquet chunk size**: Optimized files are ~10GB each. Adjust with `--max-size` in optimize command.
3. **Bar generation mode**: Standard bars support `use_pipeline=True` for 1.5-2x speedup at cost of more memory.
4. **Numba**: First run is slower due to JIT compilation, subsequent runs are fast.

### Binance Data API Structure

Data is downloaded from: `https://data.binance.vision/data/{spot|futures}/{um|cm}/{granularity}/trades/{SYMBOL}/{SYMBOL}-trades-{date}.zip`

Examples:
- Spot daily: `https://data.binance.vision/data/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-01-01.zip`
- Futures monthly: `https://data.binance.vision/data/futures/um/monthly/trades/BTCUSDT/BTCUSDT-trades-2024-01.zip`

### Testing Data Integrity

Always validate data after downloading/converting:
```bash
python main.py validate --data-dir data/btcusdt-spot/raw-parquet-merged-daily --symbol BTCUSDT --check-daily-gaps
```

This checks:
- File completeness (no missing dates)
- File integrity (readable Parquet files)
- Data quality (timestamp ordering, no corrupted records)
