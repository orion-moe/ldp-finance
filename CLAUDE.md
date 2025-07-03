# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Main Entry Point
```bash
# Interactive mode (recommended for new users)
python main.py

# Command line interface
python main.py <command> [options]
```

### Data Pipeline Commands
```bash
# Download Bitcoin trading data from Binance
python main.py download --start 2024-01-01 --end 2024-01-31

# Optimize parquet files (combine small files into larger ones)
python main.py optimize --source data/raw/dataset-raw-monthly-compressed/spot \
                       --target data/optimized/spot \
                       --max-size 10

# Validate data integrity
python main.py validate --quick
python main.py validate --advanced --output-dir reports/

# Generate features (imbalance bars)
python main.py features --type imbalance
```

### Direct Module Access
```bash
# Direct access to downloaders
python src/data_pipeline/downloaders/binance_downloader.py

# Run specific validators
python src/data_pipeline/validators/quick_validator.py
python src/data_pipeline/validators/advanced_validator.py

# Generate imbalance bars directly
python src/features/imbalance_bars.py
```

### Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# Core dependencies: pandas, numpy, pyarrow, dask, numba, loguru
```

## Architecture Overview

### Core Pipeline Structure
The system follows a modular data pipeline architecture:

1. **Data Acquisition**: Downloads Bitcoin trading data from Binance API
2. **Data Processing**: Optimizes and validates parquet files
3. **Feature Engineering**: Generates advanced trading features (imbalance bars)
4. **Data Storage**: Organized in raw/processed/optimized directories

### Key Components

#### Data Pipeline (`src/data_pipeline/`)
- **Downloaders**: Binance API integration for spot/futures data
- **Processors**: Parquet file optimization and compression
- **Validators**: Data integrity and quality checks

#### Features (`src/features/`)
- **Imbalance Bars**: Advanced trading bar generation using dollar imbalance
- Uses Dask for distributed computing and Numba for performance optimization

#### Main Entry Point (`main.py`)
- Interactive CLI interface with menus
- Command-line argument parsing
- Orchestrates all pipeline components

### Data Organization
```
data/
├── raw/                    # Downloaded raw data
├── processed/              # Processed data
└── optimized/             # Optimized parquet files

datasets/
├── dataset-raw-monthly/    # Monthly raw data archives
├── dataset-raw-monthly-compressed/  # Compressed monthly data
└── logs/                  # Download logs and progress tracking
```

### Configuration Management
- Progress tracking via JSON files (`download_progress_*.json`)
- Logging configuration in individual modules
- Interactive prompts for user configuration

## Development Guidelines

### Data Processing
- Uses Dask for distributed computing on large datasets
- Numba JIT compilation for performance-critical operations
- Parquet format for efficient storage and querying
- Column-oriented processing with type optimization

### Error Handling
- Comprehensive validation at each pipeline stage
- Progress tracking and resume capabilities
- Detailed logging with timestamps and error traces

### Performance Optimization
- Concurrent downloads with configurable worker threads
- Memory-efficient processing with Dask
- File size optimization (target 10GB chunks)
- Column type optimization (float32 for price data)

## Testing and Validation

### Data Validation
- Quick validation: Basic file integrity checks
- Advanced validation: Comprehensive data quality reports
- Checksum verification for downloaded files
- Parquet file structure validation

### Pipeline Testing
- Each component has standalone execution capability
- Interactive mode for testing configurations
- Progress tracking prevents duplicate work
- Detailed logging for debugging

## Common Data Formats

### Trading Data Schema
```python
COLUMN_NAMES = [
    'trade_id',     # Unique trade identifier
    'price',        # Trade price (float32)
    'qty',          # Trade quantity (float32)
    'quoteQty',     # Quote quantity (float32)
    'time',         # Trade timestamp
    'isBuyerMaker', # Market taker side
    'isBestMatch'   # Best price match
]
```

### Feature Engineering
- Imbalance bars based on dollar volume imbalance
- Price change direction calculation
- Side assignment for market microstructure analysis