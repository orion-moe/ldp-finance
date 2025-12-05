"""
Compression Benchmark - Test different compression algorithms for Parquet files
Helps find the optimal compression for your specific data
"""

import sys
import time
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm
import json
from datetime import datetime
import psutil
import tempfile
import shutil


class CompressionBenchmark:
    """
    Benchmark different compression algorithms for Parquet files

    Tests:
    - Compression ratio
    - Write speed
    - Read speed
    - Memory usage
    """

    # Available compression algorithms
    COMPRESSION_ALGORITHMS = [
        'snappy',      # Default - balanced speed and compression
        'gzip',        # Good compression, slower
        'brotli',      # Best compression, slowest
        'lz4',         # Fastest, less compression
        'zstd',        # Good balance, modern algorithm
        'none'         # No compression (baseline)
    ]

    # Compression levels for algorithms that support it
    COMPRESSION_LEVELS = {
        'gzip': [1, 6, 9],      # Min, default, max
        'brotli': [1, 6, 11],   # Min, default, max
        'zstd': [1, 3, 22],     # Min, default, max
    }

    def __init__(self, test_file: Optional[Path] = None, output_dir: Optional[Path] = None):
        """
        Initialize benchmark

        Args:
            test_file: Parquet file to use for testing (None to use sample data)
            output_dir: Directory for benchmark outputs
        """
        self.test_file = test_file
        self.output_dir = output_dir or Path("data/benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self.results = {}

        # Setup logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging"""
        log_file = self.output_dir / f"compression_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger.remove()
        logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        logger.add(log_file, rotation="100 MB")

        self.logger = logger

    def create_sample_data(self, num_rows: int = 5_000_000) -> pa.Table:
        """
        Create sample trading data for testing

        Args:
            num_rows: Number of rows to generate

        Returns:
            PyArrow table with sample data
        """
        self.logger.info(f"Creating sample data with {num_rows:,} rows...")

        # Generate realistic trading data
        np.random.seed(42)

        data = {
            'trade_id': np.arange(num_rows, dtype=np.int64),
            'price': np.random.uniform(30000, 40000, num_rows).astype(np.float32),
            'qty': np.random.exponential(0.1, num_rows).astype(np.float32),
            'quoteQty': np.random.exponential(1000, num_rows).astype(np.float32),
            'time': pd.date_range(start='2024-01-01', periods=num_rows, freq='100ms'),
            'isBuyerMaker': np.random.choice([True, False], num_rows),
            'isBestMatch': np.random.choice([True, False], num_rows, p=[0.9, 0.1])
        }

        # Create PyArrow table
        table = pa.table(data)

        self.logger.info(f"Sample data created: {len(table)} rows, {table.nbytes / (1024**2):.2f} MB in memory")
        return table

    def load_test_data(self) -> pa.Table:
        """Load test data from file or create sample"""
        if self.test_file and self.test_file.exists():
            self.logger.info(f"Loading test data from {self.test_file}")
            return pq.read_table(self.test_file)
        else:
            return self.create_sample_data()

    def benchmark_compression(self,
                            table: pa.Table,
                            compression: str,
                            compression_level: Optional[int] = None,
                            iterations: int = 3) -> Dict:
        """
        Benchmark a single compression algorithm

        Args:
            table: Data to compress
            compression: Compression algorithm
            compression_level: Compression level (if supported)
            iterations: Number of test iterations

        Returns:
            Dictionary with benchmark results
        """
        results = {
            'compression': compression,
            'compression_level': compression_level,
            'iterations': iterations,
            'file_sizes_mb': [],
            'write_times_sec': [],
            'read_times_sec': [],
            'memory_usage_mb': []
        }

        # Create temporary file for testing
        temp_file = self.output_dir / f"temp_benchmark_{compression}.parquet"

        try:
            for i in range(iterations):
                # Measure memory before
                process = psutil.Process()
                mem_before = process.memory_info().rss / (1024**2)

                # Write benchmark
                write_start = time.time()
                pq.write_table(
                    table,
                    temp_file,
                    compression=compression,
                    compression_level=compression_level,
                    use_dictionary=True,
                    row_group_size=100_000
                )
                write_time = time.time() - write_start
                results['write_times_sec'].append(write_time)

                # Measure file size
                file_size_mb = temp_file.stat().st_size / (1024**2)
                results['file_sizes_mb'].append(file_size_mb)

                # Read benchmark
                read_start = time.time()
                read_table = pq.read_table(temp_file)
                read_time = time.time() - read_start
                results['read_times_sec'].append(read_time)

                # Measure memory after
                mem_after = process.memory_info().rss / (1024**2)
                memory_used = mem_after - mem_before
                results['memory_usage_mb'].append(memory_used)

                # Verify data integrity
                if len(read_table) != len(table):
                    raise ValueError(f"Data corruption: expected {len(table)} rows, got {len(read_table)}")

                # Clean up for next iteration
                temp_file.unlink()

        except Exception as e:
            self.logger.error(f"Benchmark failed for {compression}: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

        # Calculate averages
        results['avg_file_size_mb'] = np.mean(results['file_sizes_mb'])
        results['avg_write_time_sec'] = np.mean(results['write_times_sec'])
        results['avg_read_time_sec'] = np.mean(results['read_times_sec'])
        results['avg_memory_usage_mb'] = np.mean(results['memory_usage_mb'])

        # Calculate compression ratio (compared to uncompressed size in memory)
        uncompressed_size_mb = table.nbytes / (1024**2)
        results['compression_ratio'] = uncompressed_size_mb / results['avg_file_size_mb']

        # Calculate throughput
        results['write_throughput_mb_s'] = uncompressed_size_mb / results['avg_write_time_sec']
        results['read_throughput_mb_s'] = uncompressed_size_mb / results['avg_read_time_sec']

        return results

    def run_benchmarks(self,
                      algorithms: Optional[List[str]] = None,
                      test_levels: bool = False,
                      iterations: int = 3) -> Dict:
        """
        Run benchmarks for multiple compression algorithms

        Args:
            algorithms: List of algorithms to test (None for all)
            test_levels: Test different compression levels
            iterations: Number of iterations per test

        Returns:
            Dictionary with all benchmark results
        """
        if algorithms is None:
            algorithms = self.COMPRESSION_ALGORITHMS

        # Load test data
        table = self.load_test_data()

        self.logger.info("=" * 60)
        self.logger.info(" Starting Compression Benchmarks ")
        self.logger.info("=" * 60)
        self.logger.info(f"Test data: {len(table):,} rows, {table.nbytes / (1024**2):.2f} MB in memory")
        self.logger.info(f"Algorithms to test: {', '.join(algorithms)}")
        self.logger.info(f"Iterations per test: {iterations}")

        all_results = {}

        # Test each algorithm
        for algo in tqdm(algorithms, desc="Testing algorithms"):
            self.logger.info(f"\nBenchmarking {algo}...")

            if test_levels and algo in self.COMPRESSION_LEVELS:
                # Test different compression levels
                for level in self.COMPRESSION_LEVELS[algo]:
                    key = f"{algo}_level_{level}"
                    self.logger.info(f"  Testing {algo} with level {level}...")

                    try:
                        results = self.benchmark_compression(
                            table, algo, compression_level=level, iterations=iterations
                        )
                        all_results[key] = results
                    except Exception as e:
                        self.logger.error(f"  Failed: {e}")
                        all_results[key] = {'error': str(e)}
            else:
                # Test with default level
                try:
                    results = self.benchmark_compression(
                        table, algo, compression_level=None, iterations=iterations
                    )
                    all_results[algo] = results
                except Exception as e:
                    self.logger.error(f"  Failed: {e}")
                    all_results[algo] = {'error': str(e)}

        self.results = all_results
        return all_results

    def print_results(self, results: Optional[Dict] = None):
        """Print formatted benchmark results"""
        if results is None:
            results = self.results

        if not results:
            self.logger.warning("No results to display")
            return

        print("\n" + "=" * 80)
        print(" 📊 COMPRESSION BENCHMARK RESULTS ")
        print("=" * 80)

        # Prepare data for table
        valid_results = {k: v for k, v in results.items() if 'error' not in v}

        if not valid_results:
            print("No successful benchmarks")
            return

        # Sort by file size (compression effectiveness)
        sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['avg_file_size_mb'])

        # Print header
        print(f"\n{'Algorithm':<20} {'Size (MB)':<12} {'Ratio':<8} {'Write (s)':<10} {'Read (s)':<10} {'Write MB/s':<12} {'Read MB/s':<12}")
        print("-" * 100)

        # Print results
        for name, res in sorted_results:
            print(f"{name:<20} "
                  f"{res['avg_file_size_mb']:<12.2f} "
                  f"{res['compression_ratio']:<8.2f} "
                  f"{res['avg_write_time_sec']:<10.2f} "
                  f"{res['avg_read_time_sec']:<10.2f} "
                  f"{res['write_throughput_mb_s']:<12.1f} "
                  f"{res['read_throughput_mb_s']:<12.1f}")

        # Find best options
        print("\n" + "=" * 80)
        print(" 🏆 RECOMMENDATIONS ")
        print("=" * 80)

        # Best compression
        best_compression = min(valid_results.items(), key=lambda x: x[1]['avg_file_size_mb'])
        print(f"\n📦 Best Compression: {best_compression[0]}")
        print(f"   • File size: {best_compression[1]['avg_file_size_mb']:.2f} MB")
        print(f"   • Compression ratio: {best_compression[1]['compression_ratio']:.2f}x")

        # Fastest write
        fastest_write = min(valid_results.items(), key=lambda x: x[1]['avg_write_time_sec'])
        print(f"\n⚡ Fastest Write: {fastest_write[0]}")
        print(f"   • Write time: {fastest_write[1]['avg_write_time_sec']:.2f} seconds")
        print(f"   • Throughput: {fastest_write[1]['write_throughput_mb_s']:.1f} MB/s")

        # Fastest read
        fastest_read = min(valid_results.items(), key=lambda x: x[1]['avg_read_time_sec'])
        print(f"\n📖 Fastest Read: {fastest_read[0]}")
        print(f"   • Read time: {fastest_read[1]['avg_read_time_sec']:.2f} seconds")
        print(f"   • Throughput: {fastest_read[1]['read_throughput_mb_s']:.1f} MB/s")

        # Best balance (scoring based on normalized metrics)
        def balance_score(res):
            # Normalize metrics (lower is better)
            size_score = res['avg_file_size_mb'] / best_compression[1]['avg_file_size_mb']
            write_score = res['avg_write_time_sec'] / fastest_write[1]['avg_write_time_sec']
            read_score = res['avg_read_time_sec'] / fastest_read[1]['avg_read_time_sec']
            # Average of normalized scores
            return (size_score + write_score + read_score) / 3

        best_balance = min(valid_results.items(), key=lambda x: balance_score(x[1]))
        print(f"\n⚖️  Best Balance: {best_balance[0]}")
        print(f"   • File size: {best_balance[1]['avg_file_size_mb']:.2f} MB")
        print(f"   • Write time: {best_balance[1]['avg_write_time_sec']:.2f} seconds")
        print(f"   • Read time: {best_balance[1]['avg_read_time_sec']:.2f} seconds")

    def save_results(self, results: Optional[Dict] = None, filename: Optional[str] = None):
        """Save benchmark results to JSON file"""
        if results is None:
            results = self.results

        if filename is None:
            filename = f"compression_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_path}")
        return output_path

    def compare_with_production_data(self, parquet_file: Path) -> Dict:
        """
        Run benchmarks using actual production data

        Args:
            parquet_file: Path to production Parquet file

        Returns:
            Benchmark results
        """
        self.logger.info(f"Loading production data from {parquet_file}")

        # Load production data
        table = pq.read_table(parquet_file)
        self.logger.info(f"Loaded {len(table):,} rows, {table.nbytes / (1024**2):.2f} MB in memory")

        # Get current compression
        metadata = pq.read_metadata(parquet_file)
        current_compression = metadata.row_group(0).column(0).compression
        current_size_mb = parquet_file.stat().st_size / (1024**2)

        self.logger.info(f"Current file:")
        self.logger.info(f"  • Compression: {current_compression}")
        self.logger.info(f"  • Size: {current_size_mb:.2f} MB")

        # Run benchmarks
        results = {}
        for algo in self.COMPRESSION_ALGORITHMS:
            self.logger.info(f"Testing {algo}...")
            try:
                result = self.benchmark_compression(table, algo, iterations=1)
                results[algo] = result

                # Calculate improvement
                size_diff = current_size_mb - result['avg_file_size_mb']
                size_diff_pct = (size_diff / current_size_mb) * 100

                if size_diff > 0:
                    self.logger.info(f"  • {algo} would save {size_diff:.2f} MB ({size_diff_pct:.1f}%)")
                else:
                    self.logger.info(f"  • {algo} would increase size by {abs(size_diff):.2f} MB ({abs(size_diff_pct):.1f}%)")

            except Exception as e:
                self.logger.error(f"  Failed: {e}")

        return results


def main():
    """Example usage and CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark Parquet compression algorithms")
    parser.add_argument("--file", type=Path, help="Parquet file to use for testing")
    parser.add_argument("--algorithms", nargs="+", help="Algorithms to test")
    parser.add_argument("--test-levels", action="store_true", help="Test different compression levels")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per test")
    parser.add_argument("--rows", type=int, default=5_000_000, help="Number of rows for sample data")
    parser.add_argument("--output-dir", type=Path, help="Output directory for results")

    args = parser.parse_args()

    # Create benchmark
    benchmark = CompressionBenchmark(
        test_file=args.file,
        output_dir=args.output_dir
    )

    # Run benchmarks
    results = benchmark.run_benchmarks(
        algorithms=args.algorithms,
        test_levels=args.test_levels,
        iterations=args.iterations
    )

    # Print results
    benchmark.print_results(results)

    # Save results
    benchmark.save_results(results)


if __name__ == "__main__":
    main()