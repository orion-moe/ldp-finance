# LDP Finance - Machine Learning Pipeline for Bitcoin Trading

A production-ready Python pipeline implementing **Marcos López de Prado's** methodologies from **"Advances in Financial Machine Learning"**. The pipeline covers the complete workflow: from downloading tick-level trade data to generating information-driven bars, applying fractional differentiation, triple-barrier labeling, and training machine learning models with proper cross-validation.

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

### Data Pipeline
- **Scalable Data Pipeline**: Downloads and processes multi-TB Bitcoin tick data from Binance
- **Memory-Efficient Processing**: Handles datasets larger than RAM using chunked processing and streaming
- **High-Performance Computation**: Numba JIT compilation for performance-critical operations
- **Parallel Processing**: Concurrent downloads and multi-threaded data processing
- **Resume Capability**: Progress tracking allows pipeline to resume after interruptions
- **Data Validation**: Comprehensive integrity checks for downloaded and processed data

### Information-Driven Bars (Chapter 2)
- **Standard Dollar Bars**: Fixed dollar volume threshold sampling
- **Imbalance Bars**: Order flow imbalance-based sampling
- **Imbalance Dollar Bars**: Adaptive sampling combining volume and imbalance

### Machine Learning Pipeline (Chapters 3-5)
- **Fractional Differentiation**: Two-stage optimization for stationarity while preserving memory
- **Triple-Barrier Labeling**: Meta-labeling with profit target, stop loss, and time barriers
- **CUSUM Event Detection**: Volatility-scaled event identification
- **AR Modeling**: Automatic order selection with multicollinearity treatment (OLS, Ridge, Lasso, ElasticNet)
- **Sample Weighting**: Uniqueness, magnitude, and time-decay weights (Numba optimized)

### Feature Engineering
- **Microstructure Features**: Corwin-Schultz spread, VPIN, OIR, Kyle's Lambda
- **Entropy Features**: Shannon entropy with multiple windows and bin configurations
- **Feature Importance Analysis**: Automated feature categorization and ranking

### Model Training
- **Random Forest with GridSearchCV**: Hyperparameter optimization with stratified K-fold CV
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **End-to-End Pipeline**: From raw data to trained model with comprehensive reporting

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

## Machine Learning Pipeline

### End-to-End Training Pipeline

Run the complete ML pipeline from imbalance bars to trained Random Forest model:

```bash
python src/search_rf_classifier.py
```

This executes the full López de Prado methodology:

1. **Load Data**: Read imbalance bars from Parquet
2. **Train/Val/Test Split**: Time-series aware splitting (60/20/20)
3. **Fractional Differentiation**: Find optimal d for stationarity
4. **AR Modeling**: Fit autoregressive model with multicollinearity treatment
5. **Event Detection**: CUSUM filter for significant price movements
6. **Triple-Barrier Labeling**: Generate meta-labels
7. **Feature Engineering**: Microstructure + entropy features
8. **Model Training**: Random Forest with GridSearchCV
9. **Feature Analysis**: Importance ranking and categorization
10. **Visualization**: Generate reports and plots

### Pipeline Configuration

Configure the pipeline via `src/ml_pipeline/core/config.py`:

```python
@dataclass
class PipelineConfig:
    # Data paths
    data_path: str = "src/output/imbalance"

    # Train/Val/Test split
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2

    # Fractional differentiation
    frac_diff_threshold: float = 1e-5

    # Event detection (CUSUM)
    tau1: float = 2.0  # CUSUM threshold multiplier
    tau2: float = 1.0  # Target scaling factor
    volatility_span: int = 200

    # Triple barrier
    pt_sl: List[float] = [1.0, 1.0]  # [profit_target, stop_loss]

    # Random Forest hyperparameters
    rf_param_grid: dict = {
        'n_estimators': [50, 100, 150],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [20, 50, 100],
        'min_samples_leaf': [10, 20, 40],
        'max_features': ['sqrt', 'log2', 0.3]
    }
```

### Feature Engineering Details

#### Microstructure Features

| Feature | Description | Reference |
|---------|-------------|-----------|
| **Corwin-Schultz Spread** | Bid-ask spread from high/low prices | López de Prado Ch. 19 |
| **VPIN** | Volume-Synchronized Probability of Informed Trading | Easley et al. (2012) |
| **OIR** | Order Imbalance Ratio | Market microstructure |
| **Kyle's Lambda** | Price impact coefficient | Kyle (1985) |

#### Entropy Features

Shannon entropy calculated over multiple configurations:
- **Windows**: [10, 20, 50, 100] bars
- **Bins**: [5, 10, 20] discretization levels

### Output Structure

```
src/ml_pipeline/results/rf_search_YYYYMMDD_HHMMSS/
├── config.json                 # Pipeline configuration
├── gridsearch_results.csv      # All CV results
├── model.joblib                # Trained model
├── feature_importance.csv      # Feature rankings
├── confusion_matrix.png        # Validation results
├── roc_curve.png              # ROC-AUC plot
└── report.pdf                 # Complete PDF report
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

src/
├── data_pipeline/                  # Data downloading and processing
│   ├── downloaders/                # Binance API downloader
│   ├── converters/                 # ZIP to Parquet conversion
│   ├── processors/                 # Parquet optimization
│   └── validators/                 # Data integrity checks
├── features/
│   └── bars/                       # Information-driven bars generators
├── ml_pipeline/                    # Machine learning pipeline
│   ├── core/                       # Config, data loading, utils
│   ├── steps/                      # Pipeline steps (3-10)
│   ├── feature_engineering/        # Microstructure & entropy features
│   └── models/                     # RF trainer, AR models, weights
└── output/
    ├── standard/                   # Standard dollar bars
    └── imbalance/                  # Imbalance bars
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

### Fractional Differentiation (Chapter 5)

Standard differentiation makes series stationary but destroys memory. López de Prado's fractional differentiation finds the minimum differentiation order `d` that achieves stationarity while preserving predictive information.

```
d = 0.0  →  Original series (non-stationary, full memory)
d = 1.0  →  First difference (stationary, no memory)
d = 0.4  →  Fractional (stationary, partial memory preserved)
```

Our implementation uses a **two-stage optimization**:
1. **Coarse search**: d ∈ [0.1, 1.0] with step 0.1
2. **Fine search**: Around optimal d with step 0.02

### Triple-Barrier Labeling (Chapter 3)

Instead of fixed-horizon returns, the triple-barrier method labels observations based on which barrier is touched first:

```
                    ┌─── Profit Target (Upper Barrier) → Label: +1
                    │
Price ──────────────┼─── Entry Point
                    │
                    └─── Stop Loss (Lower Barrier) → Label: -1

        ─────────────────────────────────────
        │         Time Barrier              │ → Label: 0
        └───────────────────────────────────┘
```

**Labels**:
- `+1`: Profit target hit first (profitable trade)
- `-1`: Stop loss hit first (losing trade)
- `0`: Time barrier hit (no conclusion)

### CUSUM Event Detection (Chapter 2)

The CUSUM filter identifies significant directional moves in the series, creating events when cumulative returns exceed a volatility-scaled threshold:

```python
# Events are triggered when:
|cumsum(returns)| > tau * volatility
```

This ensures we focus on meaningful price movements rather than noise.

### Sample Weighting (Chapter 4)

Proper sample weighting addresses:
- **Overlapping labels**: Events that share outcome periods
- **Uniqueness**: Weight by how unique each sample's information is
- **Time decay**: More recent samples weighted higher

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
- **Chapter 2**: Financial Data Structures (Information-Driven Bars, CUSUM Filter)
- **Chapter 3**: Labeling (Triple-Barrier Method, Meta-Labeling)
- **Chapter 4**: Sample Weights (Uniqueness, Time Decay)
- **Chapter 5**: Fractionally Differentiated Features
- **Chapter 19**: Microstructural Features

### Related Papers

- López de Prado, M. (2019). "Beyond Econometrics: A Roadmap Towards Financial Machine Learning." *SSRN Electronic Journal*.
- Bailey, D. H., & López de Prado, M. (2012). "The Sharpe Ratio Efficient Frontier." *Journal of Risk*.
- Easley, D., López de Prado, M., & O'Hara, M. (2012). "Flow Toxicity and Liquidity in a High-Frequency World." *Review of Financial Studies* (VPIN).
- Kyle, A. S. (1985). "Continuous Auctions and Insider Trading." *Econometrica* (Kyle's Lambda).
- Corwin, S. A., & Schultz, P. (2012). "A Simple Way to Estimate Bid-Ask Spreads from Daily High and Low Prices." *Journal of Finance*.

## Project Status

**Active Development** - This project is actively maintained and regularly updated.

### Implemented Features ✅

#### Data Pipeline
- ✅ Binance data downloader (spot & futures)
- ✅ Parquet conversion and optimization
- ✅ Data validation and integrity checks
- ✅ Interactive CLI interface
- ✅ Progress tracking and resume capability

#### Information-Driven Bars (Chapter 2)
- ✅ Standard dollar bars generation
- ✅ Imbalance bars generation
- ✅ Imbalance dollar bars

#### Machine Learning Pipeline (Chapters 3-5)
- ✅ **Fractional Differentiation** - Two-stage optimization with ADF testing
- ✅ **Triple-Barrier Labeling** - Meta-labeling with configurable barriers
- ✅ **CUSUM Event Detection** - Volatility-scaled event identification
- ✅ **AR Modeling** - Order selection with multicollinearity treatment
- ✅ **Sample Weighting** - Uniqueness, magnitude, time-decay (Numba optimized)

#### Feature Engineering
- ✅ **Microstructure Features** - Corwin-Schultz, VPIN, OIR, Kyle's Lambda
- ✅ **Entropy Features** - Shannon entropy with multiple configurations
- ✅ **Feature Importance Analysis** - Automated categorization and ranking

#### Model Training
- ✅ **Random Forest with GridSearchCV** - Hyperparameter optimization
- ✅ **Stratified K-Fold CV** - Proper cross-validation
- ✅ **End-to-End Pipeline** - From raw data to trained model
- ✅ **Visualization & Reporting** - Confusion matrix, ROC curves, PDF reports

### Planned Features 🚧

- 🚧 Run bars implementation
- 🚧 Volume bars
- 🚧 Additional exchanges support (Coinbase, Kraken)
- 🚧 Deep learning models (LSTM, Transformer)
- 🚧 Walk-forward validation
- 🚧 Backtesting framework integration

## Contributing

Contributions are welcome! This project aims to provide production-ready implementations of López de Prado's methodologies.

Areas for contribution:
- Additional bar types (run bars, volume bars)
- Deep learning models (LSTM, Transformer architectures)
- Walk-forward validation framework
- Backtesting integration
- Additional data sources (Coinbase, Kraken)
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
