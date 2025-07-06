# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Bitcoin ML Finance pipeline for downloading, processing, and analyzing cryptocurrency trading data from Binance. The pipeline transforms raw trading data into ML-ready features through a series of optimized processing steps.

## Common Development Commands

### Installation and Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

#### Interactive Mode (Recommended)
```bash
python main.py
```

#### Command Line Mode
```bash
# Download data
python main.py download --symbol BTCUSDT --type spot --granularity daily --start 2024-01-01 --end 2024-01-31 --workers 5

# Run individual pipeline components
python src/data_pipeline/downloaders/binance_downloader.py
python src/data_pipeline/extractors/csv_extractor.py
python src/data_pipeline/converters/csv_to_parquet.py
python src/data_pipeline/processors/parquet_optimizer.py
python src/data_pipeline/validators/data_integrity_validator.py
```

### Testing
```bash
# Note: No test suite currently implemented
# Future tests can be added in a tests/ directory
```

### Linting and Type Checking
```bash
# Note: Project does not currently have lint/typecheck setup
# When adding, use:
# ruff check .
# mypy src/
```

## High-Level Architecture

### Data Pipeline Flow
```
1. Download (ZIP + CHECKSUM) → 2. Extract (CSV) → 3. Convert (Parquet) → 4. Optimize (10GB chunks) → 5. Validate → 6. Features
```

### Key Architectural Patterns

#### 1. **Modular Pipeline Design**
Each stage is independent with its own module:
- `downloaders/`: Binance data fetching with integrity checks
- `extractors/`: ZIP to CSV extraction with validation
- `converters/`: CSV to Parquet with type optimization
- `processors/`: File optimization and merging
- `validators/`: Multi-level data validation
- `features/`: ML feature engineering

#### 2. **Progress Tracking System**
- JSON files track progress: `{operation}_progress_{symbol}_{type}_{granularity}.json`
- Enables resume capability for interrupted operations
- Located in `datasets/` directory

#### 3. **Data Integrity Framework**
- SHA256/CHECKSUM verification during download
- CSV validation after extraction
- Parquet verification after conversion
- Comprehensive validation with quality scoring

#### 4. **Performance Optimizations**
- Parallel downloads with ThreadPoolExecutor
- Type optimization (float32 for prices, bool for flags)
- Numba JIT compilation for compute-intensive operations
- Dask for distributed processing of large datasets
- Parquet with Snappy compression

#### 5. **Error Recovery**
- Temporary file staging for safe operations
- Automatic rollback on failures
- Comprehensive logging for debugging
- Resume from last successful state

### Directory Structure Understanding

```
datasets/
├── dataset-raw-{granularity}/          # ZIP and CSV files
│   └── {spot|futures-um|futures-cm}/
├── dataset-raw-{granularity}-compressed/  # Individual Parquet files
│   └── {spot|futures-um|futures-cm}/
└── dataset-raw-{granularity}-compressed-optimized/  # Merged 10GB Parquet files
    └── {spot|futures-um|futures-cm}/
```

### Important Considerations

1. **Data Volume**: Each month of data can be 5-10GB. Plan disk space accordingly.

2. **Binance Data Quirks**:
   - Timestamp format changed from seconds to milliseconds (handled automatically)
   - Some daily files may have incomplete data
   - Weekends excluded for non-crypto pairs

3. **Memory Management**:
   - Use chunk processing for large files
   - Configure Dask workers based on available RAM
   - Monitor memory during optimization phase

4. **Progress Files**:
   - Don't delete progress JSON files during operations
   - These enable resume functionality
   - Clean up only after full pipeline completion

5. **Validation Levels**:
   - Quick: Basic file integrity
   - Advanced: Statistical analysis with reports
   - Missing Dates: Temporal continuity check
   - Comprehensive: Full data quality assessment (recommended)

### Common Issues and Solutions

1. **Download Failures**
   - Check internet connection
   - Reduce workers if rate-limited
   - Verify date range availability

2. **Memory Errors**
   - Reduce chunk_size in processors
   - Use fewer Dask workers
   - Process in smaller batches

3. **Corrupt Parquet Files**
   - Use RobustParquetOptimizer for safer processing
   - Validate after each conversion
   - Keep backups of critical data

4. **Missing Data**
   - Run missing dates validator
   - Use add_missing_daily_data feature
   - Check Binance data availability

### Development Tips

1. Always use absolute paths in file operations
2. Check for existing data before downloading
3. Validate data after each transformation
4. Use the interactive menu for guided operations
5. Monitor logs in `datasets/logs/` for debugging
6. Test with small date ranges first
7. Use progress tracking for long operations
8. Keep original ZIP files as backup until pipeline completes

### Performance Tuning

- **Downloads**: Adjust worker count (5-10 typically optimal)
- **Conversion**: Process monthly files individually
- **Optimization**: Set appropriate chunk size (10GB default)
- **Validation**: Use multi-threading (4-8 workers)
- **Features**: Configure Dask based on system resources