#!/usr/bin/env python3
"""
Advanced integrity checker with comparison capabilities between original and optimized files.
Includes deep data validation and cross-reference checking.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class FileStats:
    """Statistics for a parquet file."""
    path: str
    rows: int
    file_size_mb: float
    time_min: datetime
    time_max: datetime
    price_min: float
    price_max: float
    price_mean: float
    volume_total: float
    compression_ratio: float
    row_groups: int
    avg_rows_per_group: float


@dataclass
class ValidationReport:
    """Detailed validation report for a file or dataset."""
    name: str
    passed: bool
    warnings: List[str]
    errors: List[str]
    stats: Optional[FileStats]
    checks: Dict[str, bool]


class AdvancedIntegrityChecker:
    """Advanced integrity checker with cross-validation capabilities."""
    
    def __init__(self, base_path: str = '.'):
        self.base_path = Path(base_path)
        self.validation_rules = {
            'min_rows': 100,  # Minimum rows per file
            'max_price_change': 0.5,  # Maximum 50% price change in single file
            'max_time_gap_seconds': 3600,  # Maximum 1 hour gap between trades
            'min_compression_ratio': 0.1,  # Minimum compression effectiveness
        }
        
    def get_file_stats(self, file_path: Path) -> Optional[FileStats]:
        """Extract comprehensive statistics from a parquet file."""
        try:
            # Get file info
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            
            # Read parquet metadata
            parquet_file = pq.ParquetFile(file_path)
            metadata = parquet_file.metadata
            
            # Read data for detailed stats
            df = parquet_file.read().to_pandas()
            
            if len(df) == 0:
                return None
            
            # Calculate statistics
            stats = FileStats(
                path=str(file_path),
                rows=len(df),
                file_size_mb=file_size_mb,
                time_min=df['time'].min(),
                time_max=df['time'].max(),
                price_min=df['price'].min(),
                price_max=df['price'].max(),
                price_mean=df['price'].mean(),
                volume_total=df['qty'].sum(),
                compression_ratio=file_size_mb / (len(df) * df.memory_usage(deep=True).sum() / (1024 * 1024)),
                row_groups=metadata.num_row_groups,
                avg_rows_per_group=len(df) / metadata.num_row_groups if metadata.num_row_groups > 0 else 0
            )
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting stats for {file_path}: {str(e)}")
            return None
    
    def validate_file_deep(self, file_path: Path) -> ValidationReport:
        """Perform deep validation of a single file."""
        report = ValidationReport(
            name=str(file_path),
            passed=True,
            warnings=[],
            errors=[],
            stats=None,
            checks={}
        )
        
        try:
            # Get file statistics
            stats = self.get_file_stats(file_path)
            if not stats:
                report.errors.append("Could not extract file statistics")
                report.passed = False
                return report
            
            report.stats = stats
            
            # Read data for validation
            df = pd.read_parquet(file_path)
            
            # Check 1: Minimum rows
            report.checks['min_rows'] = len(df) >= self.validation_rules['min_rows']
            if not report.checks['min_rows']:
                report.warnings.append(f"File has only {len(df)} rows, less than minimum {self.validation_rules['min_rows']}")
            
            # Check 2: Time continuity
            time_gaps = df['time'].diff().dt.total_seconds().dropna()
            max_gap = time_gaps.max() if len(time_gaps) > 0 else 0
            report.checks['time_continuity'] = max_gap <= self.validation_rules['max_time_gap_seconds']
            if not report.checks['time_continuity']:
                report.errors.append(f"Found time gap of {max_gap:.0f} seconds, exceeding maximum {self.validation_rules['max_time_gap_seconds']}")
                report.passed = False
            
            # Check 3: Price stability
            price_change = (stats.price_max - stats.price_min) / stats.price_min if stats.price_min > 0 else 0
            report.checks['price_stability'] = price_change <= self.validation_rules['max_price_change']
            if not report.checks['price_stability']:
                report.warnings.append(f"Price changed by {price_change*100:.1f}% in single file")
            
            # Check 4: Data integrity
            report.checks['no_nulls'] = df.isnull().sum().sum() == 0
            if not report.checks['no_nulls']:
                null_counts = df.isnull().sum()
                report.errors.append(f"Found null values: {null_counts[null_counts > 0].to_dict()}")
                report.passed = False
            
            # Check 5: Positive values
            report.checks['positive_values'] = (df['price'] > 0).all() and (df['qty'] > 0).all()
            if not report.checks['positive_values']:
                report.errors.append("Found non-positive price or quantity values")
                report.passed = False
            
            # Check 6: Time ordering
            report.checks['time_ordered'] = df['time'].is_monotonic_increasing
            if not report.checks['time_ordered']:
                report.errors.append("Time values are not in ascending order")
                report.passed = False
            
            # Check 7: Compression effectiveness
            report.checks['compression_effective'] = stats.compression_ratio >= self.validation_rules['min_compression_ratio']
            if not report.checks['compression_effective']:
                report.warnings.append(f"Poor compression ratio: {stats.compression_ratio:.3f}")
            
            # Check 8: Boolean values
            if 'is_buyer_maker' in df.columns:
                unique_values = df['is_buyer_maker'].unique()
                report.checks['valid_boolean'] = set(unique_values).issubset({True, False, 0, 1})
                if not report.checks['valid_boolean']:
                    report.errors.append(f"Invalid boolean values in is_buyer_maker: {unique_values}")
                    report.passed = False
            
        except Exception as e:
            report.errors.append(f"Validation failed: {str(e)}")
            report.passed = False
            
        return report
    
    def compare_datasets(self, original_dir: Path, optimized_dir: Path) -> Dict:
        """Compare original and optimized datasets for consistency."""
        comparison = {
            'original_dir': str(original_dir),
            'optimized_dir': str(optimized_dir),
            'matching_files': 0,
            'missing_in_optimized': [],
            'extra_in_optimized': [],
            'row_count_matches': 0,
            'row_count_mismatches': [],
            'data_consistency': []
        }
        
        # Get file lists
        original_files = {f.name: f for f in original_dir.glob('*.parquet')}
        optimized_files = {f.name: f for f in optimized_dir.glob('*.parquet')}
        
        # Check file coverage
        comparison['missing_in_optimized'] = list(set(original_files.keys()) - set(optimized_files.keys()))
        comparison['extra_in_optimized'] = list(set(optimized_files.keys()) - set(original_files.keys()))
        
        # Compare matching files
        matching_files = set(original_files.keys()) & set(optimized_files.keys())
        comparison['matching_files'] = len(matching_files)
        
        for filename in matching_files:
            try:
                # Get stats for both files
                orig_stats = self.get_file_stats(original_files[filename])
                opt_stats = self.get_file_stats(optimized_files[filename])
                
                if orig_stats and opt_stats:
                    # Check row counts
                    if orig_stats.rows == opt_stats.rows:
                        comparison['row_count_matches'] += 1
                    else:
                        comparison['row_count_mismatches'].append({
                            'file': filename,
                            'original_rows': orig_stats.rows,
                            'optimized_rows': opt_stats.rows,
                            'difference': opt_stats.rows - orig_stats.rows
                        })
                    
                    # Check data consistency (sample-based for performance)
                    orig_df = pd.read_parquet(original_files[filename], columns=['time', 'price'])
                    opt_df = pd.read_parquet(optimized_files[filename], columns=['time', 'price'])
                    
                    # Sample comparison
                    sample_size = min(1000, len(orig_df))
                    sample_indices = np.random.choice(len(orig_df), sample_size, replace=False)
                    
                    orig_sample = orig_df.iloc[sample_indices].sort_values('time')
                    opt_sample = opt_df.iloc[sample_indices].sort_values('time')
                    
                    if not orig_sample.equals(opt_sample):
                        comparison['data_consistency'].append({
                            'file': filename,
                            'issue': 'Sample data mismatch'
                        })
                        
            except Exception as e:
                logger.error(f"Error comparing {filename}: {str(e)}")
                
        return comparison
    
    def generate_report(self, output_dir: str = '.'):
        """Generate comprehensive integrity report with visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Collect all compressed-optimized directories
        directories = list(self.base_path.glob('*-compressed-optimized'))
        
        all_reports = []
        all_stats = []
        
        logger.info(f"Analyzing {len(directories)} directories...")
        
        for directory in directories:
            logger.info(f"Processing {directory}...")
            
            # Validate each file in directory
            for file_path in directory.glob('*.parquet'):
                report = self.validate_file_deep(file_path)
                all_reports.append(report)
                if report.stats:
                    all_stats.append(report.stats)
        
        # Generate summary statistics
        summary = {
            'total_files': len(all_reports),
            'passed': sum(1 for r in all_reports if r.passed),
            'failed': sum(1 for r in all_reports if not r.passed),
            'total_rows': sum(s.rows for s in all_stats),
            'total_size_gb': sum(s.file_size_mb for s in all_stats) / 1024,
            'avg_compression_ratio': np.mean([s.compression_ratio for s in all_stats]) if all_stats else 0
        }
        
        # Save detailed report
        report_data = {
            'summary': summary,
            'validation_reports': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'warnings': r.warnings,
                    'errors': r.errors,
                    'checks': r.checks,
                    'stats': asdict(r.stats) if r.stats else None
                }
                for r in all_reports
            ]
        }
        
        with open(output_path / 'advanced_integrity_report.json', 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate visualizations if we have data
        if all_stats:
            self._generate_visualizations(all_stats, output_path)
        
        # Print summary
        print("\n" + "="*60)
        print("ADVANCED INTEGRITY CHECK SUMMARY")
        print("="*60)
        print(f"Total Files Analyzed: {summary['total_files']}")
        if summary['total_files'] > 0:
            pass_percentage = summary['passed']/summary['total_files']*100
            print(f"Files Passed: {summary['passed']} ({pass_percentage:.1f}%)")
        else:
            print(f"Files Passed: {summary['passed']} (N/A - no files found)")
        print(f"Files Failed: {summary['failed']}")
        print(f"Total Rows: {summary['total_rows']:,}")
        print(f"Total Size: {summary['total_size_gb']:.2f} GB")
        print(f"Average Compression Ratio: {summary['avg_compression_ratio']:.3f}")
        
        return summary
    
    def _generate_visualizations(self, stats: List[FileStats], output_path: Path):
        """Generate visualization plots for the integrity report."""
        # Convert stats to DataFrame for easier plotting
        df_stats = pd.DataFrame([asdict(s) for s in stats])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Parquet File Integrity Analysis', fontsize=16)
        
        # Plot 1: File sizes distribution
        axes[0, 0].hist(df_stats['file_size_mb'], bins=30, edgecolor='black')
        axes[0, 0].set_xlabel('File Size (MB)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('File Size Distribution')
        
        # Plot 2: Row count distribution
        axes[0, 1].hist(df_stats['rows'], bins=30, edgecolor='black')
        axes[0, 1].set_xlabel('Number of Rows')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Row Count Distribution')
        axes[0, 1].ticklabel_format(style='plain', axis='x')
        
        # Plot 3: Compression ratio
        axes[1, 0].scatter(df_stats['file_size_mb'], df_stats['compression_ratio'])
        axes[1, 0].set_xlabel('File Size (MB)')
        axes[1, 0].set_ylabel('Compression Ratio')
        axes[1, 0].set_title('Compression Effectiveness')
        
        # Plot 4: Price range over time
        axes[1, 1].plot(pd.to_datetime(df_stats['time_min']), df_stats['price_mean'], 'b-', label='Mean Price')
        axes[1, 1].fill_between(
            pd.to_datetime(df_stats['time_min']),
            df_stats['price_min'],
            df_stats['price_max'],
            alpha=0.3,
            label='Price Range'
        )
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Price (USD)')
        axes[1, 1].set_title('Price Evolution')
        axes[1, 1].legend()
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'integrity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path / 'integrity_analysis.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced parquet integrity checker')
    parser.add_argument('--base-path', type=str, default='.', 
                        help='Base path to search for data')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for reports')
    parser.add_argument('--compare', nargs=2, metavar=('ORIGINAL', 'OPTIMIZED'),
                        help='Compare two directories')
    
    args = parser.parse_args()
    
    checker = AdvancedIntegrityChecker(args.base_path)
    
    if args.compare:
        # Run comparison
        original_dir = Path(args.compare[0])
        optimized_dir = Path(args.compare[1])
        
        if not original_dir.exists() or not optimized_dir.exists():
            logger.error("One or both comparison directories do not exist")
            sys.exit(1)
            
        comparison = checker.compare_datasets(original_dir, optimized_dir)
        
        # Save comparison report
        with open('comparison_report.json', 'w') as f:
            json.dump(comparison, f, indent=2)
            
        print("\nComparison Results:")
        print(f"Matching files: {comparison['matching_files']}")
        print(f"Missing in optimized: {len(comparison['missing_in_optimized'])}")
        print(f"Row count matches: {comparison['row_count_matches']}")
        print(f"Row count mismatches: {len(comparison['row_count_mismatches'])}")
        
    else:
        # Run full integrity check
        summary = checker.generate_report(args.output_dir)
        
        # Exit with error code if failures found
        if summary['failed'] > 0:
            sys.exit(1)


if __name__ == '__main__':
    main()