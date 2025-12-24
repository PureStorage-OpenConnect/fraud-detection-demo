#!/usr/bin/env python3
"""
Pod 1: High-Performance Data Gather Service
============================================
Stress-testing tool for Pure Storage FlashBlade that generates massive amounts
of synthetic credit card transaction data using parallel workers.

Features:
- 128 parallel worker threads writing simultaneously
- Schema-based generation from Kaggle creditcard.csv template
- Continuous append mode for sustained I/O pressure
- Real-time throughput monitoring (MB/s, Records/s)
- Configurable runtime duration (default: 5 minutes)
"""

import os
import sys
import time
import logging
import signal
import threading
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional
import queue

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerStats:
    """Statistics for a single worker thread"""
    worker_id: int
    records_written: int = 0
    bytes_written: int = 0
    chunks_written: int = 0


@dataclass
class GlobalStats:
    """Aggregated statistics across all workers"""
    total_records: int = 0
    total_bytes: int = 0
    start_time: float = 0.0
    
    def records_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_records / elapsed if elapsed > 0 else 0
    
    def mb_per_second(self) -> float:
        elapsed = time.time() - self.start_time
        return (self.total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
    
    def gb_written(self) -> float:
        return self.total_bytes / (1024 * 1024 * 1024)


class SchemaTemplate:
    """Loads and analyzes schema from template CSV file"""
    
    def __init__(self, template_path: Path):
        self.template_path = template_path
        self.columns: list = []
        self.dtypes: dict = {}
        self.stats: dict = {}  # min, max, mean, std for numeric columns
        self._load_template()
    
    def _load_template(self):
        """Load schema from creditcard.csv template"""
        logger.info(f"Loading schema template from: {self.template_path}")
        
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template file not found: {self.template_path}")
        
        # Read sample to get schema (first 10k rows for stats)
        df_sample = pd.read_csv(self.template_path, nrows=10000)
        
        self.columns = list(df_sample.columns)
        self.dtypes = df_sample.dtypes.to_dict()
        
        # Calculate statistics for each numeric column
        for col in self.columns:
            if np.issubdtype(df_sample[col].dtype, np.number):
                self.stats[col] = {
                    'min': float(df_sample[col].min()),
                    'max': float(df_sample[col].max()),
                    'mean': float(df_sample[col].mean()),
                    'std': float(df_sample[col].std())
                }
        
        logger.info(f"Schema loaded: {len(self.columns)} columns")
        logger.info(f"Columns: {', '.join(self.columns[:5])}... (and {len(self.columns)-5} more)")


class SyntheticDataGenerator:
    """Generates synthetic data matching the template schema"""
    
    def __init__(self, schema: SchemaTemplate, seed: Optional[int] = None):
        self.schema = schema
        self.rng = np.random.default_rng(seed)
    
    def generate_chunk(self, num_rows: int) -> pd.DataFrame:
        """Generate a chunk of synthetic data matching the schema"""
        data = {}
        
        for col in self.schema.columns:
            if col in self.schema.stats:
                stats = self.schema.stats[col]
                
                if col == 'Class':
                    # Fraud label: ~0.17% fraud rate (matching original dataset)
                    data[col] = self.rng.choice([0, 1], size=num_rows, p=[0.9983, 0.0017])
                elif col == 'Time':
                    # Time in seconds (0 to ~172800 for 2 days)
                    data[col] = self.rng.uniform(0, 172800, size=num_rows)
                elif col == 'Amount':
                    # Log-normal distribution for transaction amounts
                    data[col] = np.abs(self.rng.lognormal(mean=3.0, sigma=2.0, size=num_rows))
                    data[col] = np.clip(data[col], 0, 25000)  # Cap at reasonable max
                else:
                    # V1-V28 features: normal distribution based on observed stats
                    data[col] = self.rng.normal(
                        loc=stats['mean'],
                        scale=max(stats['std'], 0.01),  # Prevent zero std
                        size=num_rows
                    )
            else:
                # Non-numeric columns (shouldn't exist in creditcard.csv but handle anyway)
                data[col] = ['synthetic'] * num_rows
        
        return pd.DataFrame(data)


class DataWriter:
    """Handles file I/O with buffering for maximum throughput"""
    
    def __init__(self, output_path: Path, worker_id: int, buffer_size: int = 10000):
        self.output_path = output_path
        self.worker_id = worker_id
        self.buffer_size = buffer_size
        self.file_path = output_path / f"thread_{worker_id:03d}_data.csv"
        self.header_written = False
    
    def write_chunk(self, df: pd.DataFrame) -> int:
        """Write a chunk of data, returns bytes written"""
        mode = 'a' if self.header_written else 'w'
        header = not self.header_written
        
        # Convert to CSV string for size calculation
        csv_data = df.to_csv(index=False, header=header)
        bytes_written = len(csv_data.encode('utf-8'))
        
        # Write to file
        with open(self.file_path, mode) as f:
            f.write(csv_data)
        
        self.header_written = True
        return bytes_written


class WorkerThread:
    """Individual worker that generates and writes data"""
    
    def __init__(
        self,
        worker_id: int,
        schema: SchemaTemplate,
        output_path: Path,
        stats_queue: queue.Queue,
        stop_event: threading.Event,
        chunk_size: int = 10000
    ):
        self.worker_id = worker_id
        self.generator = SyntheticDataGenerator(schema, seed=worker_id * 12345)
        self.writer = DataWriter(output_path, worker_id)
        self.stats_queue = stats_queue
        self.stop_event = stop_event
        self.chunk_size = chunk_size
        self.stats = WorkerStats(worker_id=worker_id)
    
    def run(self):
        """Main worker loop - generate and write until stopped"""
        logger.debug(f"Worker {self.worker_id} started")
        
        try:
            while not self.stop_event.is_set():
                # Generate chunk
                chunk = self.generator.generate_chunk(self.chunk_size)
                
                # Write chunk
                bytes_written = self.writer.write_chunk(chunk)
                
                # Update stats
                self.stats.records_written += len(chunk)
                self.stats.bytes_written += bytes_written
                self.stats.chunks_written += 1
                
                # Report stats periodically (every 10 chunks)
                if self.stats.chunks_written % 10 == 0:
                    self.stats_queue.put(self.stats)
                    self.stats = WorkerStats(worker_id=self.worker_id)
                    
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
        finally:
            # Final stats report
            if self.stats.records_written > 0:
                self.stats_queue.put(self.stats)
            logger.debug(f"Worker {self.worker_id} stopped")


class StatsAggregator:
    """Collects and reports statistics from all workers"""
    
    def __init__(self, stats_queue: queue.Queue, report_interval: float = 5.0):
        self.stats_queue = stats_queue
        self.report_interval = report_interval
        self.global_stats = GlobalStats(start_time=time.time())
        self.stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_report_time = time.time()
        self._last_records = 0
        self._last_bytes = 0
    
    def collect_stats(self):
        """Collect stats from queue (non-blocking)"""
        while True:
            try:
                worker_stats = self.stats_queue.get_nowait()
                with self._lock:
                    self.global_stats.total_records += worker_stats.records_written
                    self.global_stats.total_bytes += worker_stats.bytes_written
            except queue.Empty:
                break
    
    def should_report(self) -> bool:
        return time.time() - self._last_report_time >= self.report_interval
    
    def report(self):
        """Print current throughput statistics"""
        self.collect_stats()
        
        with self._lock:
            elapsed = time.time() - self._last_report_time
            
            # Calculate interval rates
            interval_records = self.global_stats.total_records - self._last_records
            interval_bytes = self.global_stats.total_bytes - self._last_bytes
            
            interval_rps = interval_records / elapsed if elapsed > 0 else 0
            interval_mbps = (interval_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else 0
            
            # Update tracking
            self._last_records = self.global_stats.total_records
            self._last_bytes = self.global_stats.total_bytes
            self._last_report_time = time.time()
            
            total_elapsed = time.time() - self.global_stats.start_time
            
            logger.info(
                f"[{total_elapsed:6.1f}s] "
                f"Records: {self.global_stats.total_records:,} | "
                f"Size: {self.global_stats.gb_written():.2f} GB | "
                f"Throughput: {interval_mbps:.1f} MB/s | "
                f"Rate: {interval_rps:,.0f} rec/s"
            )
    
    def final_report(self):
        """Print final summary statistics"""
        self.collect_stats()
        
        with self._lock:
            total_elapsed = time.time() - self.global_stats.start_time
            
            logger.info("=" * 70)
            logger.info("FINAL RESULTS - FlashBlade Stress Test Complete")
            logger.info("=" * 70)
            logger.info(f"Duration:        {total_elapsed:.1f} seconds")
            logger.info(f"Total Records:   {self.global_stats.total_records:,}")
            logger.info(f"Total Data:      {self.global_stats.gb_written():.2f} GB")
            logger.info(f"Avg Throughput:  {self.global_stats.mb_per_second():.1f} MB/s")
            logger.info(f"Avg Rate:        {self.global_stats.records_per_second():,.0f} records/s")
            logger.info("=" * 70)


class FlashBladeStressTester:
    """Main orchestrator for the FlashBlade stress test"""
    
    def __init__(
        self,
        template_path: Path,
        output_path: Path,
        num_workers: int = 128,
        duration_seconds: int = 300,
        chunk_size: int = 10000
    ):
        self.template_path = template_path
        self.output_path = output_path
        self.num_workers = num_workers
        self.duration_seconds = duration_seconds
        self.chunk_size = chunk_size
        
        self.stop_event = threading.Event()
        self.stats_queue = queue.Queue()
        self.schema: Optional[SchemaTemplate] = None
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.stop_event.set()
    
    def setup(self):
        """Initialize directories and load schema"""
        logger.info("=" * 70)
        logger.info("Pod 1: FlashBlade High-Performance Stress Test")
        logger.info("=" * 70)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_path}")
        
        # Clear any existing files
        existing_files = list(self.output_path.glob("thread_*_data.csv"))
        if existing_files:
            logger.info(f"Clearing {len(existing_files)} existing output files...")
            for f in existing_files:
                f.unlink()
        
        # Load schema template
        self.schema = SchemaTemplate(self.template_path)
        
        logger.info(f"Workers:         {self.num_workers}")
        logger.info(f"Chunk size:      {self.chunk_size:,} rows")
        logger.info(f"Duration:        {self.duration_seconds} seconds")
        logger.info("=" * 70)
    
    def run(self):
        """Execute the stress test"""
        self.setup()
        
        stats_aggregator = StatsAggregator(self.stats_queue)
        workers = []
        
        logger.info(f"Starting {self.num_workers} worker threads...")
        
        # Start workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all workers
            futures = []
            for worker_id in range(self.num_workers):
                worker = WorkerThread(
                    worker_id=worker_id,
                    schema=self.schema,
                    output_path=self.output_path,
                    stats_queue=self.stats_queue,
                    stop_event=self.stop_event,
                    chunk_size=self.chunk_size
                )
                workers.append(worker)
                futures.append(executor.submit(worker.run))
            
            logger.info(f"All {self.num_workers} workers started. Running for {self.duration_seconds}s...")
            logger.info("-" * 70)
            
            # Monitor and report until duration expires
            start_time = time.time()
            while not self.stop_event.is_set():
                elapsed = time.time() - start_time
                
                if elapsed >= self.duration_seconds:
                    logger.info("\nDuration limit reached, stopping workers...")
                    self.stop_event.set()
                    break
                
                if stats_aggregator.should_report():
                    stats_aggregator.report()
                
                time.sleep(0.5)  # Small sleep to avoid busy-waiting
            
            # Wait for all workers to finish
            logger.info("Waiting for workers to complete...")
            for future in futures:
                future.result(timeout=10)
        
        # Final report
        stats_aggregator.final_report()
        
        # List output files
        output_files = list(self.output_path.glob("thread_*_data.csv"))
        total_size = sum(f.stat().st_size for f in output_files)
        logger.info(f"\nOutput files: {len(output_files)} files in {self.output_path}")
        logger.info(f"Total on disk: {total_size / (1024**3):.2f} GB")


def main():
    """Main entry point"""
    # Configuration from environment variables
    template_dir = os.getenv('TEMPLATE_DIR', '/mnt/datasets/kaggle/creditcardfraud')
    template_file = os.getenv('TEMPLATE_FILE', 'creditcard.csv')
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data')
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration_seconds = int(os.getenv('DURATION_SECONDS', '300'))  # 5 minutes default
    chunk_size = int(os.getenv('CHUNK_SIZE', '10000'))
    
    template_path = Path(template_dir) / template_file
    output_path = Path(output_dir)
    
    logger.info("Configuration:")
    logger.info(f"  Template: {template_path}")
    logger.info(f"  Output:   {output_path}")
    logger.info(f"  Workers:  {num_workers}")
    logger.info(f"  Duration: {duration_seconds}s")
    logger.info(f"  Chunk:    {chunk_size} rows")
    
    # Create and run stress tester
    tester = FlashBladeStressTester(
        template_path=template_path,
        output_path=output_path,
        num_workers=num_workers,
        duration_seconds=duration_seconds,
        chunk_size=chunk_size
    )
    
    tester.run()


if __name__ == "__main__":
    main()
