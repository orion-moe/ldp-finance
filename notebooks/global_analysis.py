# Complete Dollar Bars Analysis - Lopez de Prado Methodology

# Essential imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10
sns.set_palette('husl')


def analyze_imbalance_bars(dataset_path):
    """Dollar Imbalance Bars Analysis"""

    df = pd.read_parquet(dataset_path)

    print("="*80)
    print(f"DOLLAR IMBALANCE BARS ANALYSIS - LOPEZ DE PRADO METHODOLOGY")
    print(f"SAMPLE - {dataset_path}")
    print("="*80)

    # Convert time columns to datetime if needed
    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ns')
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ns')

    # Basic information
    print(f"\nDataset loaded: {len(df):,} bars")

    if 'close_time' in df.columns:
        print(f"Period: {df['close_time'].min()} to {df['close_time'].max()}")
        duration = (df['close_time'].max() - df['close_time'].min()).total_seconds() / 86400
        print(f"Duration: {duration:.1f} days")
        if duration > 0:
            print(f"Average frequency: {len(df) / duration:.1f} bars/day")
        else:
            duration_hours = (df['close_time'].max() - df['close_time'].min()).total_seconds() / 3600
            print(f"Duration: {duration_hours:.1f} hours")
            print(f"Average frequency: {len(df) / duration_hours:.1f} bars/hour")
    else:
        print(f"Indices: {df.index[0]} to {df.index[-1]}")

    # Data structure
    print("\nData structure:")
    print(df.info())

    print("\nFirst 5 bars:")
    print(df.head())

    df['total_volume_sell_usd'] = df['total_volume_buy_usd'] - df['total_volume_usd']
    df['thres'] = df['ewma_T'] * df['ewma_imbalance']

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print("\n1. PRICES:")
    price_stats = df[['open', 'high', 'low', 'close']].describe()
    print(price_stats)

    print("\n2. VOLUMES:")
    volume_stats = df[['total_volume_usd', 'total_volume_buy_usd', 'total_volume_sell_usd', 'total_volume']].describe()
    print(volume_stats)

    print("\n3. RUN METRICS:")
    imbalance_stats = df[['ewma_T', 'ewma_imbalance', 'thres']].describe()
    print(imbalance_stats)

    time_between_bars = df['end_time'].diff().dt.total_seconds() / 60
    time_between_bars = time_between_bars.dropna()

    print("\n4. Time Between Bars Statistics:")
    print(f"   - Mean: {time_between_bars.mean():.2f} minutes")
    print(f"   - Median: {time_between_bars.median():.2f} minutes")
    print(f"   - Std dev: {time_between_bars.std():.2f} minutes")
    print(f"   - Min: {time_between_bars.min():.2f} minutes")
    print(f"   - Max: {time_between_bars.max():.2f} minutes")

    # Visualization
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Temporal Analysis of Imbalance Run Bars', fontsize=16)

    df_plot = df

    ax1 = axes[0]
    ax1.scatter(df_plot['end_time'], np.log(df_plot['close']),
            color='darkblue', s=1, alpha=0.6)
    ax1.set_ylabel('Log(Price) ($)')
    ax1.set_title('Log-Close Price Series')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(df_plot['end_time'], np.log(df_plot['thres']),
            color='darkblue', s=1, alpha=0.6)
    ax2.set_ylabel('Volume USD')
    ax2.set_title('Threshold Series')
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.scatter(df_plot['end_time'], np.log(df_plot['ewma_T']),
            color='darkblue', s=1, alpha=0.6)
    ax3.set_ylabel('EWMA Ticks')
    ax3.set_title('EWMA Ticks')
    ax3.grid(True, alpha=0.3)

    ax4 = axes[3]
    ax4.scatter(df_plot['end_time'], np.log(df_plot['ewma_imbalance']),
            color='darkblue', s=1, alpha=0.6)
    ax4.set_ylabel('Imbalance / Ticks')
    ax4.set_title('EWMA Imbalance')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_standard_bars(dataset_path):
    """Dollar Standard Bars Analysis"""

    df = pd.read_parquet(dataset_path)

    print("="*80)
    print(f"DOLLAR STANDARD BARS ANALYSIS - LOPEZ DE PRADO METHODOLOGY")
    print(f"SAMPLE - {dataset_path}")
    print("="*80)

    if 'close_time' in df.columns:
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ns')
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ns')

    print(f"\nDataset loaded: {len(df):,} bars")

    if 'close_time' in df.columns:
        print(f"Period: {df['close_time'].min()} to {df['close_time'].max()}")
        duration = (df['close_time'].max() - df['close_time'].min()).total_seconds() / 86400
        print(f"Duration: {duration:.1f} days")
        if duration > 0:
            print(f"Average frequency: {len(df) / duration:.1f} bars/day")
        else:
            duration_hours = (df['close_time'].max() - df['close_time'].min()).total_seconds() / 3600
            print(f"Duration: {duration_hours:.1f} hours")
            print(f"Average frequency: {len(df) / duration_hours:.1f} bars/hour")
    else:
        print(f"Indices: {df.index[0]} to {df.index[-1]}")

    print("\nData structure:")
    print(df.info())

    print("\nFirst 5 bars:")
    print(df.head())

    df['total_volume_sell_usd'] = df['total_volume_buy_usd'] - df['total_volume_usd']

    print("\nDescriptive Statistics:")
    print("\n1. PRICES:")
    price_stats = df[['open', 'high', 'low', 'close']].describe()
    print(price_stats)

    print("\n2. VOLUMES:")
    volume_stats = df[['total_volume_usd', 'total_volume_buy_usd', 'total_volume_sell_usd', 'total_volume']].describe()
    print(volume_stats)

    time_between_bars = df['end_time'].diff().dt.total_seconds() / 60
    time_between_bars = time_between_bars.dropna()

    print("\n3. Time Between Bars Statistics:")
    print(f"   - Mean: {time_between_bars.mean():.2f} minutes")
    print(f"   - Median: {time_between_bars.median():.2f} minutes")
    print(f"   - Std dev: {time_between_bars.std():.2f} minutes")
    print(f"   - Min: {time_between_bars.min():.2f} minutes")
    print(f"   - Max: {time_between_bars.max():.2f} minutes")

    # Visualization
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle('Temporal Analysis of Standard Dollar Bars', fontsize=16)

    df_plot = df

    ax1 = axes[0]
    ax1.scatter(df_plot['end_time'], np.log(df_plot['close']),
            color='darkblue', s=1, alpha=0.6)
    ax1.set_ylabel('Log(Price) ($)')
    ax1.set_title('Log-Close Price Series')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(df_plot['end_time'], df_plot['total_volume_usd'],
            color='darkblue', s=1, alpha=0.6)
    ax2.set_ylabel('Volume USD')
    ax2.set_title('Volume USD per Bar')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Specific file for analysis
    dataset_path = "/Users/felipe/Desktop/hub/trading/ldp-finance/data/btcusdt-spot/output/standard/20251122-223805-standard-spot-volume40000000/20251122-223805-standard-spot-volume40000000.parquet"

    # Use analyze_standard_bars for standard bars
    # Use analyze_imbalance_bars for imbalance bars
    analyze_standard_bars(dataset_path)
