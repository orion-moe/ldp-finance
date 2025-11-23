# LDP Finance - Information-Driven Bars for Bitcoin Trading

A production-ready Python pipeline for downloading, processing, and generating **information-driven bars** from Bitcoin tick-level trade data. This implementation is based on the methodologies described in **"Advances in Financial Machine Learning"** by **Marcos López de Prado**.

## Overview

Traditional time-based sampling (e.g., 1-minute bars) doesn't capture the irregular arrival of information in financial markets. This project implements López de Prado's **information-driven bars** which sample data based on meaningful market activity rather than arbitrary time intervals, resulting in better-behaved statistical properties for machine learning applications.

### Implemented Bar Types

Based on López de Prado's framework:

- **Standard Dollar Bars**: Sample data when a fixed dollar volume threshold is reached (e.g., every $40M traded)
- **Imbalance Bars**: Sample based on order flow imbalance, capturing shifts in buying vs. selling pressure
- **Imbalance Dollar Bars**: Combine dollar volume with order flow imbalance for adaptive sampling

These bars exhibit superior properties for ML/statistical analysis:
- More normally distributed returns
- Reduced serial correlation
- Better stationarity characteristics
- Information arrives at more uniform intervals

## Features

- **Scalable Data Pipeline**: Downloads and processes multi-TB Bitcoin tick data from Binance
- **Memory-Efficient Processing**: Handles datasets larger than RAM using chunked processing and streaming
- **High-Performance Computation**: Numba JIT compilation for performance-critical bar generation
- **Parallel Processing**: Concurrent downloads and multi-threaded data processing
- **Resume Capability**: Progress tracking allows pipeline to resume after interruptions
- **Data Validation**: Comprehensive integrity checks for downloaded and processed data
- **Interactive & CLI Modes**: User-friendly interactive menu and scriptable command-line interface

## Installation

### Requirements

- Python 3.8+
- 16GB+ RAM recommended (8GB minimum for small datasets)
- Sufficient disk space for data (Bitcoin tick data: ~500GB/year)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ldp-finance.git
cd ldp-finance

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=12.0.0
numba>=0.57.0
requests>=2.31.0
tqdm>=4.65.0
loguru>=0.7.0
```

## Quick Start

### Interactive Mode (Recommended)

```bash
python main.py
```

This launches an interactive menu with guided workflows for:
1. Downloading data from Binance
2. Converting to optimized Parquet format
3. Validating data integrity
4. Generating information-driven bars
5. Running complete end-to-end pipelines

### Command Line Interface

```bash
# Download Bitcoin spot data for January 2024
python main.py download \
  --symbol BTCUSDT \
  --type spot \
  --granularity daily \
  --start 2024-01-01 \
  --end 2024-01-31

# Convert and optimize to Parquet format
python main.py optimize \
  --source data/btcusdt-spot/raw-parquet-daily \
  --target data/btcusdt-spot/raw-parquet-merged-daily

# Generate Standard Dollar Bars (López de Prado Chapter 2)
python main.py features \
  --type standard \
  --volume 40000000

# Generate Imbalance Bars (López de Prado Chapter 2)
python main.py features \
  --type imbalance
```

## Usage Examples

### Downloading Historical Data

```bash
# Spot market data (daily granularity)
python main.py download --start 2023-01-01 --end 2023-12-31 --type spot --granularity daily

# Futures market data (monthly granularity for faster bulk downloads)
python main.py download --start 2023-01 --end 2023-12 --type futures --futures-type um --granularity monthly
```

### Generating Information-Driven Bars

```python
# Standard Dollar Bars with custom threshold
python main.py features --type standard --volume 50000000  # $50M threshold

# Imbalance Bars (adaptive sampling)
python main.py features --type imbalance
```

### Validating Data Quality

```bash
python main.py validate \
  --data-dir data/btcusdt-spot/raw-parquet-merged-daily \
  --symbol BTCUSDT \
  --check-daily-gaps
```

## Pipeline Architecture

```
┌──────────────────┐
│ Binance API      │
│ (Tick Data)      │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Download         │ ZIP files with trade data
│ (Parallel)       │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Stream Convert   │ ZIP → Parquet (memory-efficient)
│ (PyArrow)        │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Optimize & Merge │ Daily files → ~10GB chunks
│ (Compression)    │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Validate         │ Integrity & completeness checks
│                  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────┐
│ Generate Information Bars    │
│ • Standard Dollar Bars       │
│ • Imbalance Bars             │
│ • Imbalance Dollar Bars      │
└──────────────────────────────┘
```

## Data Directory Structure

```
data/
├── btcusdt-spot/
│   ├── raw-zip-daily/              # Downloaded ZIP files
│   ├── raw-parquet-daily/          # Extracted Parquet files
│   ├── raw-parquet-merged-daily/   # Optimized merged files
│   └── logs/                       # Processing logs
├── btcusdt-futures-um/
│   └── ...
└── logs/                           # Pipeline execution logs

src/output/
├── standard/                       # Standard dollar bars output
└── imbalance/                      # Imbalance bars output
```

## Theoretical Background

### Information-Driven Bars (López de Prado)

Traditional time-based sampling has several drawbacks:
- Information doesn't arrive at constant time intervals
- Markets are inactive during certain periods (low volume)
- Time bars oversample during quiet periods and undersample during active periods

**López de Prado's Solution**: Sample based on market activity metrics:

1. **Dollar Bars**: Sample every time a fixed dollar amount is exchanged
   - Better statistical properties than time/tick bars
   - Volume-normalized sampling

2. **Imbalance Bars**: Sample when order flow imbalance exceeds expected levels
   - Captures shifts in buying/selling pressure
   - More sensitive to market microstructure changes

3. **Run Bars**: Sample on sequences of same-side trades
   - Identifies persistent buying/selling activity

### Why Information-Driven Bars?

From *Advances in Financial Machine Learning*, Chapter 2:

> "Time bars are a chronological sequence that ignores the fact that exchanges are closed and inactive at night and on weekends... A more sensible approach is to sample bars as a subordinated process of trading activity."

**Benefits for Machine Learning**:
- Returns closer to normal distribution (better for ML algorithms)
- Reduced autocorrelation (satisfies IID assumptions)
- Better stationarity properties
- Information arrives at more uniform intervals
- Improved signal-to-noise ratio

## Performance Considerations

### Memory Usage

- **Sequential Mode**: ~1-2GB per processing chunk
- **Pipeline Mode**: ~3-4GB (parallel processing stages)
- Much more efficient than naive approaches (tested: avoids 30-50GB Dask overhead)

### Processing Speed

- **Downloads**: ~5-10 concurrent connections (configurable via `--workers`)
- **Conversion**: Streaming ZIP→Parquet avoids disk I/O bottlenecks
- **Bar Generation**: Numba JIT provides near-C performance for core loops
- **Pipeline Mode**: 1.5-2x speedup via parallel I/O and preprocessing

### Optimization Tips

```bash
# Faster downloads (if bandwidth allows)
python main.py download --workers 10 --start 2024-01-01 --end 2024-01-31

# Larger optimized chunks for better query performance
python main.py optimize --max-size 15 --source ... --target ...

# Pipeline mode for faster bar generation (uses more memory)
# Edit src/features/bars/standard_dollar_bars.py: use_pipeline=True
```

## Data Source

All data is sourced from [Binance Public Data](https://data.binance.vision/):
- Spot market tick data
- Futures market tick data (USD-M and COIN-M)
- Available from 2017-08-17 onwards (varies by symbol)
- Free and publicly accessible

## References

### Primary Reference

**López de Prado, Marcos.** *Advances in Financial Machine Learning.*
Wiley, 2018.
- **Chapter 2**: Financial Data Structures (Information-Driven Bars)
- **Chapter 3**: Labeling (Triple-Barrier Method, Meta-Labeling)
- **Chapter 5**: Fractionally Differentiated Features

### Related Papers

- López de Prado, M. (2019). "Beyond Econometrics: A Roadmap Towards Financial Machine Learning." *SSRN Electronic Journal*.
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.

## Project Status

**Active Development** - This project is actively maintained and regularly updated.

### Implemented Features ✅

- ✅ Binance data downloader (spot & futures)
- ✅ Parquet conversion and optimization
- ✅ Standard dollar bars generation
- ✅ Imbalance bars generation
- ✅ Data validation and integrity checks
- ✅ Interactive CLI interface
- ✅ Progress tracking and resume capability

### Planned Features 🚧

- 🚧 Run bars implementation
- 🚧 Volume bars
- 🚧 Triple-barrier labeling (Chapter 3)
- 🚧 Meta-labeling framework
- 🚧 Fractionally differentiated features (Chapter 5)
- 🚧 Additional exchanges support (Coinbase, Kraken)

## Contributing

Contributions are welcome! This project aims to provide production-ready implementations of López de Prado's methodologies.

Areas for contribution:
- Additional bar types (run bars, volume imbalance bars)
- Labeling methods (triple-barrier, trend-scanning)
- Feature engineering (fractional differentiation, microstructure features)
- Additional data sources
- Performance optimizations
- Documentation improvements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Marcos López de Prado** for his groundbreaking work in *Advances in Financial Machine Learning*
- **Binance** for providing free public market data
- The Python scientific computing community (NumPy, Pandas, PyArrow, Numba)

## Citation

If you use this project in your research, please cite:

```bibtex
@book{lopez2018advances,
  title={Advances in Financial Machine Learning},
  author={L{\'o}pez de Prado, Marcos},
  year={2018},
  publisher={John Wiley \& Sons}
}
```

## Contact & Support

- **Issues**: Please report bugs and feature requests via [GitHub Issues](https://github.com/yourusername/ldp-finance/issues)
- **Documentation**: See [CLAUDE.md](CLAUDE.md) for detailed developer documentation

---

**Disclaimer**: This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk. Past performance does not guarantee future results.
