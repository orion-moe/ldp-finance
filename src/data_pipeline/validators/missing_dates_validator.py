#!/usr/bin/env python3
"""
Missing dates validator for Bitcoin trading data
Identifies gaps in monthly data files
"""

import os
import re
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import logging
from typing import List, Dict, Tuple, Optional
import pandas as pd
import pyarrow.parquet as pq


class MissingDatesValidator:
    def __init__(self, data_dir: str, symbol: str = "BTCUSDT"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        # Support both monthly and daily file patterns (case-insensitive)
        self.pattern_monthly = f"{symbol}-[Tt]rades-(\\d{{4}})-(\\d{{2}})\\.parquet"
        self.pattern_daily = f"{symbol}-[Tt]rades-(\\d{{4}})-(\\d{{2}})-(\\d{{2}})\\.parquet"
        self.optimized_pattern = f"{symbol}-[Tt]rades-[Oo]ptimized-\\d{{3}}\\.parquet"

        # Get logger (logging should be configured by main.py)
        self.logger = logging.getLogger(__name__)
    
    def extract_date_from_filename(self, filename: str) -> Optional[Tuple[int, int]]:
        """Extract year and month from filename (supports both monthly and daily files)"""
        # Try daily pattern first (YYYY-MM-DD)
        match = re.match(self.pattern_daily, filename)
        if match:
            year, month, day = match.groups()
            return int(year), int(month)

        # Try monthly pattern (YYYY-MM)
        match = re.match(self.pattern_monthly, filename)
        if match:
            year, month = match.groups()
            return int(year), int(month)

        return None
    
    def _get_months_from_optimized_files(self, optimized_files: List[Path]) -> List[Tuple[int, int]]:
        """Extract year-month tuples from optimized files by reading the actual data"""
        existing_months = set()
        
        for file_path in optimized_files:
            try:
                # Read only a small sample to get the date range
                parquet_file = pq.ParquetFile(file_path)
                
                # Read first and last row groups to get min/max dates
                first_batch = parquet_file.read_row_group(0, columns=['time'])
                last_batch = parquet_file.read_row_group(parquet_file.num_row_groups - 1, columns=['time'])
                
                # Convert to pandas and get min/max dates
                first_df = first_batch.to_pandas()
                last_df = last_batch.to_pandas()
                
                min_date = pd.to_datetime(first_df['time'].min())
                max_date = pd.to_datetime(last_df['time'].max())
                
                # Check if timestamps are corrupted (Bitcoin trading started in 2017)
                if min_date.year < 2017:
                    self.logger.warning(f"Corrupted timestamps detected in {file_path.name} (min date: {min_date})")
                    continue
                
                # Add all months between min and max dates
                current_date = min_date.to_period('M')
                end_date = max_date.to_period('M')
                
                while current_date <= end_date:
                    existing_months.add((current_date.year, current_date.month))
                    current_date += 1
                    
            except Exception as e:
                self.logger.error(f"Error reading {file_path.name}: {e}")
        
        return sorted(list(existing_months))
    
    def get_existing_months(self) -> List[Tuple[int, int]]:
        """Get list of existing year-month tuples from files"""
        existing_months = []
        
        if not self.data_dir.exists():
            self.logger.error(f"Directory not found: {self.data_dir}")
            return existing_months
        
        # Check if we have optimized files (case-insensitive search)
        optimized_files = [f for f in self.data_dir.glob("*.parquet")
                          if re.match(self.optimized_pattern, f.name, re.IGNORECASE)]
        if optimized_files:
            self.logger.info(f"Found {len(optimized_files)} optimized files")
            optimized_months = self._get_months_from_optimized_files(optimized_files)
            
            # If optimized files have valid timestamps, use them
            if optimized_months:
                return optimized_months
            else:
                self.logger.error("ALL optimized files have corrupted timestamps (showing 1970 dates)")
                self.logger.error("This indicates an issue with the optimization process")
                self.logger.error("Please re-run the optimization step to fix timestamp corruption")
                # Return empty list to indicate no valid data found
                return []
        
        # Regular monthly/daily files (case-insensitive search)
        for file_path in self.data_dir.glob("*.parquet"):
            date_tuple = self.extract_date_from_filename(file_path.name)
            if date_tuple:
                existing_months.append(date_tuple)

        return sorted(list(set(existing_months)))  # Remove duplicates and sort
    
    def find_missing_months(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> List[Dict[str, any]]:
        """
        Find missing months in the dataset
        
        Args:
            start_date: Start date in YYYY-MM format (if None, uses first file date)
            end_date: End date in YYYY-MM format (if None, uses current date)
        
        Returns:
            List of dictionaries with missing month information
        """
        existing_months = self.get_existing_months()
        
        if not existing_months:
            self.logger.warning("No existing data files found")
            return []
        
        # Determine date range
        if start_date:
            start_year, start_month = map(int, start_date.split('-'))
        else:
            start_year, start_month = existing_months[0]
        
        if end_date:
            end_year, end_month = map(int, end_date.split('-'))
        else:
            # Use current date
            now = datetime.now()
            end_year, end_month = now.year, now.month
        
        # Generate all expected months
        current_date = datetime(start_year, start_month, 1)
        end_datetime = datetime(end_year, end_month, 1)
        
        expected_months = []
        while current_date <= end_datetime:
            expected_months.append((current_date.year, current_date.month))
            current_date += relativedelta(months=1)
        
        # Convert existing months to set for faster lookup
        existing_set = set(existing_months)
        
        # Find missing months
        missing_months = []
        for year, month in expected_months:
            if (year, month) not in existing_set:
                missing_months.append({
                    'year': year,
                    'month': month,
                    'date_str': f"{year:04d}-{month:02d}",
                    'expected_file': f"{self.symbol}-Trades-{year:04d}-{month:02d}.parquet"
                })
        
        return missing_months
    
    def check_daily_gaps_in_files(self) -> Dict[str, List[str]]:
        """
        Check for missing days within each monthly file
        Reads the actual data to identify gaps in trading days
        """
        daily_gaps = {}
        
        # Check for optimized files first
        optimized_files = list(self.data_dir.glob(f"{self.symbol}-Trades-Optimized-*.parquet"))
        if optimized_files:
            file_pattern = optimized_files
        else:
            file_pattern = sorted(self.data_dir.glob(f"{self.symbol}-Trades-*.parquet"))
        
        for file_path in file_pattern:
            self.logger.info(f"Checking daily gaps in {file_path.name}")
            
            try:
                # Read the parquet file and get unique dates
                df = pd.read_parquet(file_path, columns=['time'])
                df['date'] = pd.to_datetime(df['time']).dt.date
                unique_dates = sorted(df['date'].unique())
                
                if len(unique_dates) == 0:
                    continue
                
                # Check for gaps
                missing_days = []
                for i in range(len(unique_dates) - 1):
                    current_date = unique_dates[i]
                    next_date = unique_dates[i + 1]
                    
                    # If there's more than 1 day gap
                    days_diff = (next_date - current_date).days
                    if days_diff > 1:
                        # Add all missing days
                        for j in range(1, days_diff):
                            missing_date = current_date + timedelta(days=j)
                            missing_days.append(missing_date.strftime('%Y-%m-%d'))
                
                if missing_days:
                    daily_gaps[file_path.name] = missing_days
                    
            except Exception as e:
                self.logger.error(f"Error reading {file_path.name}: {e}")
        
        return daily_gaps
    
    def generate_report(self, check_daily_gaps: bool = False) -> Dict[str, any]:
        """
        Generate comprehensive missing dates report
        
        Args:
            check_daily_gaps: Whether to check for missing days within files (slower)
        
        Returns:
            Dictionary with complete analysis
        """
        self.logger.info("🔍 Starting missing dates validation...")
        
        existing_months = self.get_existing_months()
        missing_months = self.find_missing_months()
        
        report = {
            'summary': {
                'data_directory': str(self.data_dir),
                'symbol': self.symbol,
                'total_files': len(existing_months),
                'missing_months': len(missing_months),
                'date_range': None,
                'completeness_percentage': 0
            },
            'existing_months': [],
            'missing_months': missing_months,
            'daily_gaps': {}
        }
        
        if existing_months:
            first_date = f"{existing_months[0][0]:04d}-{existing_months[0][1]:02d}"
            last_date = f"{existing_months[-1][0]:04d}-{existing_months[-1][1]:02d}"
            report['summary']['date_range'] = f"{first_date} to {last_date}"
            
            # Calculate completeness
            total_expected = len(self.find_missing_months()) + len(existing_months)
            if total_expected > 0:
                report['summary']['completeness_percentage'] = (
                    len(existing_months) / total_expected * 100
                )
            
            # Add existing months to report
            report['existing_months'] = [
                f"{year:04d}-{month:02d}" for year, month in existing_months
            ]
        
        # Check for daily gaps if requested
        if check_daily_gaps and existing_months:
            self.logger.info("Checking for daily gaps within files...")
            report['daily_gaps'] = self.check_daily_gaps_in_files()
        
        return report
    
    def print_report(self, report: Dict[str, any]):
        """Print formatted report"""
        print("\n" + "="*60)
        print("📊 MISSING DATES VALIDATION REPORT")
        print("="*60)
        
        summary = report['summary']
        print(f"\n📁 Data Directory: {summary['data_directory']}")
        print(f"💱 Symbol: {summary['symbol']}")
        print(f"📈 Total Files: {summary['total_files']}")
        print(f"📅 Date Range: {summary['date_range'] or 'N/A'}")
        print(f"✅ Completeness: {summary['completeness_percentage']:.1f}%")
        
        # Missing months
        if report['missing_months']:
            print(f"\n❌ Missing Months ({len(report['missing_months'])} total):")
            for missing in report['missing_months']:
                print(f"   - {missing['date_str']} ({missing['expected_file']})")
        else:
            print("\n✅ No missing months detected!")
        
        # Daily gaps
        if report['daily_gaps']:
            print(f"\n⚠️  Files with Daily Gaps:")
            for filename, missing_days in report['daily_gaps'].items():
                print(f"\n   {filename}:")
                print(f"   Missing {len(missing_days)} days:")
                # Show first 5 missing days
                for day in missing_days[:5]:
                    print(f"     - {day}")
                if len(missing_days) > 5:
                    print(f"     ... and {len(missing_days) - 5} more")
        elif 'daily_gaps' in report and len(report['daily_gaps']) == 0:
            print("\n✅ No daily gaps detected in existing files!")
        
        print("\n" + "="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate missing dates in Bitcoin trading data')
    parser.add_argument('--data-dir', type=str, 
                       default='data/dataset-raw-monthly-compressed/spot',
                       help='Directory containing parquet files')
    parser.add_argument('--symbol', type=str, default='BTCUSDT',
                       help='Trading pair symbol')
    parser.add_argument('--start-date', type=str,
                       help='Start date in YYYY-MM format')
    parser.add_argument('--end-date', type=str,
                       help='End date in YYYY-MM format')
    parser.add_argument('--check-daily-gaps', action='store_true',
                       help='Check for missing days within files (slower)')
    parser.add_argument('--output-json', type=str,
                       help='Output report to JSON file')
    
    args = parser.parse_args()
    
    validator = MissingDatesValidator(args.data_dir, args.symbol)
    report = validator.generate_report(check_daily_gaps=args.check_daily_gaps)
    
    validator.print_report(report)
    
    if args.output_json:
        import json
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.output_json}")


if __name__ == "__main__":
    main()