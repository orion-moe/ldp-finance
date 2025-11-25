# py_modules_main_optimized - Refactored Pipeline

Modular, clean, and maintainable implementation of the financial ML pipeline following López de Prado's best practices.

## 📦 Package Structure

```
py_modules_main_optimized/
├── __init__.py                          # Package initialization with exports
├── config.py                            # ⭐ Centralized configuration
├── utils.py                             # ⭐ Logging, memory, helpers
├── data_loader.py                       # ⭐ STEP 1: Data loading
├── data_splitter.py                     # ⭐ STEP 2: Train/val/test split
├── model_trainer.py                     # ⭐ STEP 9: RF training + GridSearch
├── visualizations.py                    # ⭐ STEP 11: Plots generation
│
├── entropy_features.py                  # ✅ Entropy calculations
├── microstructure_features.py           # ✅ Market microstructure
├── unified_microstructure_features.py   # ✅ Unified microstructure API
├── sample_weights_numba.py              # ✅ Sample weighting (Numba optimized)
├── improved_ar_model.py                 # ✅ Autoregressive modeling
└── ...                                  # Other existing modules
```

⭐ = New refactored modules
✅ = Existing modules

## 🚀 Quick Start

### Basic Usage

```python
from py_modules_main_optimized import (
    PipelineConfig,
    setup_logging,
    load_and_prepare_data,
    split_timeseries_data,
    RandomForestTrainer,
    generate_all_plots
)

# 1. Setup
config = PipelineConfig()
log_file = setup_logging()

# 2. Load data
df = load_and_prepare_data(config.data_path, config.file_name)

# 3. Split data
train_df, val_df, test_df = split_timeseries_data(
    df,
    train_ratio=0.7,
    val_ratio=0.15
)

# 4. Train model (after feature engineering)
trainer = RandomForestTrainer(
    param_grid=config.get_rf_param_grid(reduced=True),
    cv_n_splits=5,
    cv_scoring='f1'
)

model = trainer.train(X_train, y_train, sample_weight=weights)
metrics = trainer.evaluate(X_train, y_train, dataset_name="Training")

# 5. Generate visualizations
feature_importance = trainer.get_feature_importance(X_train.columns)

plots = generate_all_plots(
    y_true=y_train,
    y_pred=model.predict(X_train),
    y_pred_proba=model.predict_proba(X_train)[:, 1],
    feature_importance_df=feature_importance,
    output_path=output_dir
)
```

## 📋 Module Reference

### config.py - Configuration Management

```python
from py_modules_main_optimized import PipelineConfig

# Use default config
config = PipelineConfig()

# Or customize
config = PipelineConfig(
    data_path='data/btcusdt-futures-um/output/standard',
    file_name='20251123-003308-standard-futures-volume200000000.parquet',
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    frac_diff_d=0.4,
    cusum_threshold=0.02,
    barrier_num_days=5
)

# Get RF parameter grid
param_grid = config.get_rf_param_grid(reduced=True)  # Fast
param_grid = config.get_rf_param_grid(reduced=False)  # Full search
```

### utils.py - Utilities

```python
from py_modules_main_optimized import (
    setup_logging,
    log_memory,
    create_output_directory,
    print_step_header
)

# Setup logging
log_file = setup_logging(log_dir='data/logs', log_level='INFO')

# Monitor memory
log_memory("after feature engineering")

# Create output directory
output_path, folder_name, metadata = create_output_directory(
    base_path='data/btcusdt-futures-um/output/standard',
    file_name='20251123-003308-standard-futures-volume200000000.parquet'
)

# Print formatted headers
print_step_header(1, "DATA LOADING")
```

### data_loader.py - Data Loading

```python
from py_modules_main_optimized import load_and_prepare_data

# Load and prepare data in one call
df = load_and_prepare_data(
    data_path='data/btcusdt-futures-um/output/standard',
    file_name='20251123-003308-standard-futures-volume200000000.parquet'
)

# Or use individual functions
from py_modules_main_optimized.data_loader import (
    find_data_file,
    load_parquet_data,
    aggregate_data
)

filepath = find_data_file(data_path, file_name)
raw_df = load_parquet_data(filepath)
prepared_df = aggregate_data(raw_df)
```

### data_splitter.py - Data Splitting

```python
from py_modules_main_optimized import split_timeseries_data

train_df, val_df, test_df = split_timeseries_data(
    df,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### model_trainer.py - Model Training

```python
from py_modules_main_optimized import RandomForestTrainer

# Initialize trainer
trainer = RandomForestTrainer(
    param_grid={
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    },
    cv_n_splits=5,
    cv_scoring='f1',
    random_state=42
)

# Train with optional sample weights
model = trainer.train(X_train, y_train, sample_weight=weights)

# Evaluate
train_metrics = trainer.evaluate(X_train, y_train, "Training")
val_metrics = trainer.evaluate(X_val, y_val, "Validation")

# Get feature importance
feature_importance = trainer.get_feature_importance(X_train.columns)

# Get CV results
cv_results = trainer.get_cv_results()
```

### visualizations.py - Plot Generation

```python
from py_modules_main_optimized import generate_all_plots

# Generate all plots at once
plots = generate_all_plots(
    y_true=y_train,
    y_pred=y_pred_train,
    y_pred_proba=y_pred_proba_train,
    feature_importance_df=feature_importance,
    output_path='output/20251123-003308-standard-futures-volume200000000',
    dpi=300,
    n_features=20
)

# Returns dict with paths:
# {
#     'confusion_matrix': 'path/to/confusion_matrix.png',
#     'roc_curve': 'path/to/roc_curve.png',
#     'best_features': 'path/to/top_20_best_features.png',
#     'worst_features': 'path/to/top_20_worst_features.png'
# }

# Or generate individual plots
from py_modules_main_optimized.visualizations import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_feature_importance
)
```

## 🎯 Benefits of Refactored Code

### Before (main_optimized.py - 1401 lines)
```python
# Everything in one massive file
# Hard to test
# Hard to reuse
# Hard to maintain
```

### After (Modular Structure)
```python
# ✅ Clean separation of concerns
# ✅ Each module < 300 lines
# ✅ Easily testable
# ✅ Reusable across projects
# ✅ Easy to maintain and extend
# ✅ Type hints and docstrings
# ✅ Centralized configuration
```

## 📊 Migration Guide

Old code in `main_optimized.py` can be gradually migrated:

| Old Code | New Module |
|----------|------------|
| Lines 66-132 (logging setup) | `utils.setup_logging()` |
| Lines 284-348 (data loading) | `data_loader.load_and_prepare_data()` |
| Lines 353-378 (data split) | `data_splitter.split_timeseries_data()` |
| Lines 915-1050 (model training) | `model_trainer.RandomForestTrainer` |
| Lines 1086-1180 (visualizations) | `visualizations.generate_all_plots()` |

## 🔧 TODO: Remaining Modules

The following modules still need to be created to complete the refactoring:

- [ ] `frac_diff.py` - Fractional differentiation (STEP 3)
- [ ] `ar_model.py` - Consolidate AR modeling (STEP 4)
- [ ] `event_detection.py` - CUSUM event detection (STEP 5)
- [ ] `labeling.py` - Triple Barrier Method (STEP 6)
- [ ] `feature_pipeline.py` - Orchestrate feature engineering (STEP 7)

## 📝 Testing

Each module can be tested independently:

```python
# Test data loading
from py_modules_main_optimized import load_and_prepare_data
df = load_and_prepare_data('data/btcusdt-futures-um/output/standard', 'file.parquet')
assert len(df) > 0
assert 'close' in df.columns

# Test model training
from py_modules_main_optimized import RandomForestTrainer
trainer = RandomForestTrainer()
model = trainer.train(X_train, y_train)
assert model is not None
```

## 📚 References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Project CLAUDE.md for pipeline architecture details
