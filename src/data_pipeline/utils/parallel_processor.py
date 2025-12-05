"""
Parallel Processor - Utilities for parallel processing in the data pipeline
Provides thread-safe, memory-efficient parallel execution with progress tracking
"""

import os
import sys
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Callable, Any, Optional, Dict, Tuple, Union
from pathlib import Path
import logging
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import json
from functools import partial
import traceback


class ParallelProcessor:
    """
    Parallel processing manager for data pipeline operations

    Features:
    - Auto-detects optimal worker count based on CPU/memory
    - Progress tracking with tqdm
    - Error resilience with detailed logging
    - Memory monitoring to prevent OOM
    - Supports both thread and process pools
    """

    def __init__(self,
                 max_workers: Optional[int] = None,
                 use_threads: bool = False,
                 memory_limit_gb: float = None,
                 progress_bar: bool = True,
                 task_name: str = "Processing"):
        """
        Initialize parallel processor

        Args:
            max_workers: Maximum number of workers (None for auto-detect)
            use_threads: Use ThreadPoolExecutor instead of ProcessPoolExecutor
            memory_limit_gb: Maximum memory usage in GB (None for 80% of available)
            progress_bar: Show progress bar
            task_name: Name for progress bar
        """
        self.use_threads = use_threads
        self.progress_bar = progress_bar
        self.task_name = task_name

        # Auto-detect optimal workers
        if max_workers is None:
            cpu_count = mp.cpu_count()
            # Leave 1-2 CPUs free for system
            self.max_workers = max(1, min(cpu_count - 1, 8))
        else:
            self.max_workers = max_workers

        # Memory monitoring
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_limit_gb is None:
            # Use 80% of available memory
            self.memory_limit_gb = available_memory_gb * 0.8
        else:
            self.memory_limit_gb = min(memory_limit_gb, available_memory_gb * 0.9)

        # Setup logging
        self.setup_logging()

        self.logger.info(f"Initialized ParallelProcessor:")
        self.logger.info(f"  Workers: {self.max_workers}")
        self.logger.info(f"  Type: {'Threads' if use_threads else 'Processes'}")
        self.logger.info(f"  Memory limit: {self.memory_limit_gb:.2f} GB")

    def setup_logging(self):
        """Configure logging"""
        log_dir = Path("data/logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f"parallel_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger.remove()
        logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
        logger.add(log_file, rotation="100 MB", retention="7 days")

        self.logger = logger

    def check_memory(self) -> bool:
        """Check if memory usage is within limits"""
        current_memory_gb = psutil.virtual_memory().used / (1024**3)
        available_memory_gb = psutil.virtual_memory().available / (1024**3)

        if available_memory_gb < 1.0:  # Less than 1GB available
            self.logger.warning(f"Low memory: {available_memory_gb:.2f} GB available")
            return False
        return True

    def process_batch(self,
                     items: List[Any],
                     process_func: Callable,
                     batch_size: Optional[int] = None,
                     **kwargs) -> Tuple[List[Any], List[Any]]:
        """
        Process items in parallel batches

        Args:
            items: List of items to process
            process_func: Function to apply to each item
            batch_size: Items per batch (None for auto)
            **kwargs: Additional arguments for process_func

        Returns:
            Tuple of (successful_results, failed_items)
        """
        if not items:
            return [], []

        # Auto-determine batch size
        if batch_size is None:
            batch_size = max(1, len(items) // (self.max_workers * 4))

        # Create batches
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]

        self.logger.info(f"Processing {len(items)} items in {len(batches)} batches")

        successful_results = []
        failed_items = []

        # Choose executor type
        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        with ExecutorClass(max_workers=self.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self._process_batch_worker, batch, process_func, **kwargs): batch
                for batch in batches
            }

            # Process results with progress bar
            if self.progress_bar:
                pbar = tqdm(total=len(items), desc=self.task_name, unit="items")

            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]

                try:
                    results, failures = future.result()
                    successful_results.extend(results)
                    failed_items.extend(failures)

                    if self.progress_bar:
                        pbar.update(len(batch))

                    # Check memory after each batch
                    if not self.check_memory():
                        self.logger.warning("Memory limit reached, reducing workers")
                        self.max_workers = max(1, self.max_workers // 2)

                except Exception as e:
                    self.logger.error(f"Batch processing failed: {e}")
                    failed_items.extend(batch)
                    if self.progress_bar:
                        pbar.update(len(batch))

            if self.progress_bar:
                pbar.close()

        self.logger.info(f"Completed: {len(successful_results)} successful, {len(failed_items)} failed")
        return successful_results, failed_items

    def _process_batch_worker(self, batch: List[Any], process_func: Callable, **kwargs) -> Tuple[List[Any], List[Any]]:
        """Worker function to process a batch"""
        results = []
        failures = []

        for item in batch:
            try:
                result = process_func(item, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.debug(f"Item processing failed: {e}")
                failures.append(item)

        return results, failures

    def map_parallel(self,
                    func: Callable,
                    items: List[Any],
                    chunksize: Optional[int] = None,
                    ordered: bool = False) -> List[Any]:
        """
        Parallel map function similar to multiprocessing.Pool.map

        Args:
            func: Function to apply
            items: Items to process
            chunksize: Chunk size for processing
            ordered: Preserve original order

        Returns:
            List of results
        """
        if not items:
            return []

        if chunksize is None:
            chunksize = max(1, len(items) // (self.max_workers * 4))

        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor

        results = []
        failed_count = 0

        with ExecutorClass(max_workers=self.max_workers) as executor:
            if ordered:
                # Use map to preserve order
                if self.progress_bar:
                    with tqdm(total=len(items), desc=self.task_name) as pbar:
                        for result in executor.map(func, items, chunksize=chunksize):
                            results.append(result)
                            pbar.update(1)
                else:
                    results = list(executor.map(func, items, chunksize=chunksize))
            else:
                # Use submit for potentially better performance
                futures = [executor.submit(func, item) for item in items]

                if self.progress_bar:
                    pbar = tqdm(total=len(items), desc=self.task_name)

                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        self.logger.error(f"Task failed: {e}")
                        failed_count += 1
                        results.append(None)

                    if self.progress_bar:
                        pbar.update(1)

                if self.progress_bar:
                    pbar.close()

        if failed_count > 0:
            self.logger.warning(f"{failed_count} tasks failed")

        return results


class ParallelFileProcessor(ParallelProcessor):
    """
    Specialized parallel processor for file operations
    """

    def __init__(self, **kwargs):
        """Initialize with thread pool (better for I/O operations)"""
        super().__init__(use_threads=True, **kwargs)

    def process_files(self,
                     file_paths: List[Path],
                     process_func: Callable,
                     output_dir: Optional[Path] = None,
                     skip_existing: bool = True,
                     **kwargs) -> Dict[str, List[Path]]:
        """
        Process multiple files in parallel

        Args:
            file_paths: List of file paths to process
            process_func: Function to process each file
            output_dir: Output directory for processed files
            skip_existing: Skip files that already exist in output
            **kwargs: Additional arguments for process_func

        Returns:
            Dictionary with 'successful', 'skipped', and 'failed' file lists
        """
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            'successful': [],
            'skipped': [],
            'failed': []
        }

        # Filter files to process
        files_to_process = []
        for file_path in file_paths:
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                results['failed'].append(file_path)
                continue

            if skip_existing and output_dir:
                output_path = output_dir / file_path.name
                if output_path.exists():
                    results['skipped'].append(file_path)
                    continue

            files_to_process.append(file_path)

        if not files_to_process:
            self.logger.info("No files to process")
            return results

        # Process files in parallel
        self.logger.info(f"Processing {len(files_to_process)} files...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(process_func, file_path, output_dir=output_dir, **kwargs): file_path
                for file_path in files_to_process
            }

            if self.progress_bar:
                pbar = tqdm(total=len(files_to_process), desc="Processing files", unit="files")

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]

                try:
                    result = future.result()
                    if result:
                        results['successful'].append(file_path)
                    else:
                        results['failed'].append(file_path)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
                    results['failed'].append(file_path)

                if self.progress_bar:
                    pbar.update(1)

            if self.progress_bar:
                pbar.close()

        # Summary
        self.logger.info(f"Processing complete:")
        self.logger.info(f"  Successful: {len(results['successful'])}")
        self.logger.info(f"  Skipped: {len(results['skipped'])}")
        self.logger.info(f"  Failed: {len(results['failed'])}")

        return results


def parallel_download(urls: List[str],
                     download_func: Callable,
                     max_workers: int = 5,
                     **kwargs) -> Tuple[List[Any], List[str]]:
    """
    Convenience function for parallel downloads

    Args:
        urls: List of URLs to download
        download_func: Function to download each URL
        max_workers: Number of concurrent downloads
        **kwargs: Additional arguments for download_func

    Returns:
        Tuple of (successful_downloads, failed_urls)
    """
    processor = ParallelProcessor(
        max_workers=max_workers,
        use_threads=True,  # Better for I/O bound tasks
        task_name="Downloading"
    )

    return processor.process_batch(urls, download_func, **kwargs)


def parallel_convert(files: List[Path],
                    convert_func: Callable,
                    output_dir: Path,
                    max_workers: Optional[int] = None,
                    use_processes: bool = True,
                    **kwargs) -> Dict[str, List[Path]]:
    """
    Convenience function for parallel file conversion

    Args:
        files: List of files to convert
        convert_func: Conversion function
        output_dir: Output directory
        max_workers: Number of workers (None for auto)
        use_processes: Use processes instead of threads
        **kwargs: Additional arguments for convert_func

    Returns:
        Dictionary with conversion results
    """
    processor = ParallelFileProcessor(
        max_workers=max_workers,
        use_threads=not use_processes,
        task_name="Converting"
    )

    return processor.process_files(
        files,
        convert_func,
        output_dir=output_dir,
        **kwargs
    )


# Example usage functions
def example_process_function(item: Any) -> Any:
    """Example processing function"""
    import time
    import random
    time.sleep(random.uniform(0.1, 0.5))
    return f"Processed: {item}"


def example_file_converter(file_path: Path, output_dir: Path = None, **kwargs) -> bool:
    """Example file conversion function"""
    try:
        # Simulate file processing
        import time
        time.sleep(0.2)

        if output_dir:
            output_path = output_dir / f"converted_{file_path.name}"
            output_path.touch()

        return True
    except Exception:
        return False


if __name__ == "__main__":
    # Example: Process a list of items in parallel
    print("Example 1: Processing items in parallel")
    processor = ParallelProcessor(max_workers=4, task_name="Example Processing")

    items = list(range(20))
    results, failures = processor.process_batch(items, example_process_function)
    print(f"Results: {len(results)}, Failures: {len(failures)}")

    # Example: Process files in parallel
    print("\nExample 2: Processing files in parallel")
    file_processor = ParallelFileProcessor(max_workers=4)

    # Create dummy files for testing
    test_dir = Path("data/test_parallel")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_files = []
    for i in range(10):
        file_path = test_dir / f"test_{i}.txt"
        file_path.touch()
        test_files.append(file_path)

    output_dir = test_dir / "output"
    results = file_processor.process_files(
        test_files,
        example_file_converter,
        output_dir=output_dir
    )

    print(f"File processing results: {results}")

    # Cleanup
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)