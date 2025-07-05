# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a Bitcoin ML Finance pipeline project that downloads, processes, and analyzes cryptocurrency trading data from Binance. The pipeline is designed for machine learning applications in quantitative finance, with a focus on data quality, integrity, and optimization.

## Common Development Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Main Pipeline Commands

**Interactive Mode (Recommended for most tasks):**
```bash
python main.py
```

**Direct Commands:**
```bash
# Download data
python main.py download --start YYYY-MM --end YYYY-MM --symbol BTCUSDT --type spot --granularity monthly --workers 5

# Optimize parquet files
python main.py optimize --source datasets/raw --target data/optimized

# Validate data
python main.py validate --quick  # Quick validation
python main.py validate --advanced  # Generate detailed reports
python main.py validate --integrity  # Full integrity check

# Generate features
python main.py features --type imbalance
```

**Note**: Step 2 (CSV to Parquet conversion) is streamlined in the interactive mode and automatically handles verification and cleanup.

### Individual Module Execution
```bash
# Run specific pipeline components
python -m src.data_pipeline.downloaders.binance_downloader
python -m src.data_pipeline.extractors.csv_extractor
python -m src.data_pipeline.converters.csv_to_parquet
python -m src.data_pipeline.processors.robust_parquet_optimizer
python -m src.data_pipeline.validators.data_integrity_validator
```

### Testing & Validation
```bash
# Test robust optimization
python test_robust_optimization.py

# Cleanup corrupted data
python cleanup_and_restart.py
python safe_cleanup_and_verify.py
```

## High-Level Architecture

### Data Pipeline Flow
The pipeline follows a sequential process designed for data integrity and optimization:

```
1. Download (ZIP) → 2. Extract (CSV) → 3. Convert (Parquet) → 4. Optimize → 5. Validate → 6. Features
```

### Key Architectural Components

1. **Data Pipeline (`src/data_pipeline/`)**
   - **Downloaders**: Parallel download from Binance with SHA256 checksum verification
   - **Extractors**: Safe ZIP extraction with CSV integrity checking
   - **Converters**: CSV to Parquet conversion with automatic verification and CSV cleanup
     - Type optimization (float32 for prices)
     - Automatic Parquet integrity verification
     - Automatic CSV cleanup after successful conversion (saves disk space)
     - ZIP files preserved as backup
   - **Processors**: 
     - `parquet_optimizer.py`: Combines small files into ~10GB chunks
     - `robust_parquet_optimizer.py`: Enhanced version with corruption prevention
   - **Validators**: Multi-level validation from quick checks to comprehensive integrity analysis

2. **Feature Engineering (`src/features/`)**
   - `imbalance_bars.py`: Generates imbalance dollar bars for ML
   - Uses Dask for distributed processing
   - Numba JIT compilation for performance

3. **Data Organization**
   - Raw data: `datasets/dataset-raw-monthly/`
   - Compressed: `datasets/dataset-raw-monthly-compressed/`
   - Optimized: `datasets/dataset-raw-monthly-compressed-optimized/`
   - Progress tracking: JSON files for resumable operations
   - Logs: `datasets/logs/` with rotating file handlers

### Critical Design Patterns

1. **Resumable Operations**: All major operations track progress in JSON files
2. **Data Integrity**: Multiple validation layers with checksums at each step
3. **Error Recovery**: Fail-safe mechanisms with automatic rollback
4. **Performance**: Parallel processing, Dask distribution, Numba optimization
5. **Memory Efficiency**: Streaming processing for large files

### Important Considerations

- **Sequential Execution**: Always run pipeline steps in order
- **Data Validation**: The pipeline includes robust validation at each step to prevent data corruption
- **Progress Files**: Don't delete `*_progress_*.json` files - they enable resume functionality
- **Disk Space Management**: 
  - Each month of data requires ~5-10GB in CSV format
  - Step 2 automatically cleans CSV files after successful Parquet conversion
  - ZIP files are preserved as backup (can be cleaned manually later if needed)
  - Parquet files are ~50-70% smaller than CSV files
- **Logging**: Check `datasets/logs/` for detailed operation logs

### Working with the Pipeline

When modifying the pipeline:
1. Maintain the sequential flow - each step depends on the previous one
2. Preserve progress tracking functionality for resumable operations
3. Include validation checks when adding new data transformations
4. Use the existing logging framework (loguru) for consistency
5. Follow the established data type conventions (float32 for prices, optimized dtypes)

### Data Schema

The processed Parquet files contain:
- `time`: int64 timestamp (various formats supported)
- `price`: float32 trade price
- `qty`: float32 trade quantity
- Additional columns preserved from source (symbol, id, etc.)

**Timestamp Formats:**
- **16 digits**: Microseconds (2025+ data)
- **13 digits**: Milliseconds (most common format)
- **10 digits**: Seconds (older format)

Files are organized by date (YYYY-MM format) and maintain chronological order within each file.

### Common Issues & Solutions

**Duplicate Files in ZIP Archives:**
Some ZIP files may contain duplicate CSV files in subdirectories (e.g., `fsx-data/collector_data/...`). The extractor automatically detects and handles duplicates by using the root-level file.

**Timestamp Format Changes:**
Binance changed their timestamp format in 2025 from milliseconds to microseconds. The pipeline automatically detects and handles all supported formats.

**Incomplete Data from Binance:**
Some months may have incomplete data from Binance's servers. For example:
- **2023-03**: Only contains ~11 days of data (Feb 28 to Mar 11) instead of the full month
- **2017-08**: Missing the first 16 days of data

The pipeline will detect and warn about these incomplete months during extraction. This is a limitation of Binance's historical data availability, not a bug in the pipeline.

**PyArrow Compatibility:**
The verification functions are compatible with different PyArrow versions. The `nrows` parameter issue in `ParquetFile.read()` has been resolved by using pandas sampling instead.