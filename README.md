# LDP Finance

Bitcoin data pipeline for ML based on "Advances in Financial Machine Learning" (Lopez de Prado).

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py  # Interactive menu
```

Direct commands: `download`, `optimize`, `validate`, `features` (use `--help` for options).

## Structure

```
src/
├── data_pipeline/
│   ├── downloaders/    # Binance data download
│   ├── converters/     # ZIP -> Parquet
│   ├── processors/     # Parquet optimization/merge
│   └── validators/     # Data validation
├── features/
│   └── bars/           # Dollar bars and imbalance bars
└── ml_pipeline/
    ├── core/           # Config, data loading, visualizations
    ├── steps/          # Pipeline steps (frac diff, AR, CUSUM, triple barrier)
    ├── feature_engineering/  # Microstructure features
    └── models/         # Random Forest, AR

data/
└── {symbol}-{type}/    # E.g.: btcusdt-spot/
    ├── raw-zip-*/      # Downloaded ZIPs
    ├── raw-parquet-*/  # Daily Parquet
    └── raw-parquet-merged-*/  # Optimized Parquet
```

## Main Functions

| Module | Class/Function | Description |
|--------|----------------|-------------|
| `downloaders/binance_downloader.py` | `BinanceDataDownloader` | Parallel tick data download |
| `converters/zip_to_parquet_streamer.py` | `ZipToParquetStreamer` | Streaming ZIP->Parquet conversion |
| `processors/parquet_optimizer.py` | `EnhancedParquetOptimizer` | File merge and compression |
| `bars/standard_dollar_bars.py` | `process_files_and_generate_bars` | Dollar bars generation (Numba) |
| `bars/imbalance_bars.py` | `ImbalanceBarsGenerator` | Order flow based bars |
| `search_rf_classifier.py` | `main` | Complete ML pipeline |

## Reference

Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
