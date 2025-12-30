# LDP Finance

Bitcoin data pipeline for machine learning based on "Advances in Financial Machine Learning" (Marcos López de Prado).

## What It Does

Downloads tick data from Binance, converts to optimized Parquet, and generates **information-driven bars** (dollar bars, imbalance bars) for ML models.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Interactive Mode
```bash
python main.py
```

### Direct Commands

| Command | Description |
|---------|-------------|
| `python main.py download` | Download data from Binance |
| `python main.py optimize` | Optimize/merge Parquet files |
| `python main.py validate` | Validate data integrity |
| `python main.py features` | Generate dollar bars or imbalance bars |

## Main Functions

### 1. Data Download

```bash
python main.py download --start 2024-01-01 --end 2024-01-31 --type spot --granularity daily
```

**Parameters:**
- `--start` / `--end`: Period (required)
- `--type`: `spot` or `futures`
- `--futures-type`: `um` or `cm` (for futures)
- `--granularity`: `daily` or `monthly`
- `--workers`: Number of parallel downloads (default: 5)

### 2. Parquet Optimization

```bash
python main.py optimize --source data/btcusdt-spot/raw-parquet-daily --target data/btcusdt-spot/raw-parquet-merged-daily
```

Merges daily files into ~10GB chunks for better read performance.

**Parameters:**
- `--source`: Directory with daily Parquet files
- `--target`: Output directory
- `--max-size`: Maximum file size in GB (default: 10)

### 3. Data Validation

```bash
python main.py validate --data-dir data/btcusdt-spot/raw-parquet-merged-daily --symbol BTCUSDT
```

Checks:
- Complete files (no missing dates)
- Parquet file integrity
- Timestamp ordering

**Parameters:**
- `--data-dir`: Data directory
- `--symbol`: Symbol (e.g., BTCUSDT)
- `--check-daily-gaps`: Check for daily gaps

### 4. Feature Generation (Bars)

#### Standard Dollar Bars
```bash
python main.py features --type standard --volume 40000000
```

Generates bars when $40M USD is traded.

#### Imbalance Bars
```bash
python main.py features --type imbalance
```

Generates bars based on order flow imbalance.

**Parameters:**
- `--type`: `standard` or `imbalance`
- `--volume`: Threshold in USD (for standard bars)

### 5. ML Pipeline

```bash
python src/search_rf_classifier.py
```

Complete pipeline that executes:
1. Load imbalance bars
2. Train/val/test split (60/20/20)
3. Fractional differentiation
4. AR modeling
5. CUSUM event detection
6. Triple-barrier labeling
7. Feature engineering (microstructure + entropy)
8. GridSearchCV with Random Forest
9. Generate reports and visualizations

## Directory Structure

```
data/
  btcusdt-spot/
    raw-zip-daily/           # Downloaded ZIPs
    raw-parquet-daily/       # Daily Parquet
    raw-parquet-merged-daily/# Optimized Parquet

src/
  data_pipeline/
    downloaders/             # BinanceDataDownloader
    converters/              # ZipToParquetStreamer
    processors/              # EnhancedParquetOptimizer
    validators/              # DataValidator
  features/
    bars/
      standard_dollar_bars.py  # Standard dollar bars
      imbalance_bars.py        # Imbalance bars
  ml_pipeline/
    steps/                   # ML pipeline steps
    feature_engineering/     # Microstructure features
    models/                  # Random Forest, AR
```

## Main Classes

| Class | File | Function |
|-------|------|----------|
| `BinanceDataDownloader` | `downloaders/binance_downloader.py` | Parallel download from Binance |
| `ZipToParquetStreamer` | `converters/zip_to_parquet_streamer.py` | Streaming ZIP to Parquet conversion |
| `EnhancedParquetOptimizer` | `processors/parquet_optimizer.py` | Parquet merge and compression |
| `StandardDollarBarsGenerator` | `bars/standard_dollar_bars.py` | Dollar bars generation (Numba JIT) |
| `ImbalanceBarsGenerator` | `bars/imbalance_bars.py` | Imbalance bars generation |

## Data Schema

**Trade Data (input):**
```
trade_id, price, qty, quoteQty, time, isBuyerMaker, isBestMatch
```

**Dollar Bars (output):**
```
time_open, time_close, price_open, price_high, price_low, price_close, volume, dollar_volume, num_trades
```

## Requirements

- Python 3.8+
- 16GB RAM (8GB minimum)
- Disk: ~500GB/year of tick data

## Reference

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Ch. 2: Information-Driven Bars
- Ch. 3: Triple-Barrier Labeling
- Ch. 4: Sample Weights
- Ch. 5: Fractional Differentiation

## License

MIT
