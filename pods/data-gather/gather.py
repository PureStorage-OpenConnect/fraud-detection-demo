#!/usr/bin/env python3
"""
Pod 1: High-Performance Data Gather Service
============================================
Stress-testing tool for Pure Storage FlashBlade that generates massive amounts
of synthetic credit card transaction data using parallel workers.

Features:
- 128 parallel worker PROCESSES (not threads - avoids Python GIL)
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
import multiprocessing as mp
from multiprocessing import Process, Value
from ctypes import c_longlong
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_schema(template_path: Path) -> Dict:
    """Load schema from creditcard.csv template and return stats dict"""
    logger.info(f"Loading schema template from: {template_path}")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    # Read sample to get schema (first 10k rows for stats)
    df_sample = pd.read_csv(template_path, nrows=10000)
    
    columns = list(df_sample.columns)
    stats = {}
    
    # Calculate statistics for each numeric column
    for col in columns:
        if np.issubdtype(df_sample[col].dtype, np.number):
            stats[col] = {
                'min': float(df_sample[col].min()),
                'max': float(df_sample[col].max()),
                'mean': float(df_sample[col].mean()),
                'std': float(df_sample[col].std())
            }
    
    logger.info(f"Schema loaded: {len(columns)} columns")
    logger.info(f"Columns: {', '.join(columns[:5])}... (and {len(columns)-5} more)")
    
    return {'columns': columns, 'stats': stats}


def generate_chunk(columns: List[str], stats: Dict, num_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate a chunk of synthetic data matching the schema"""
    data = {}
    
    for col in columns:
        if col in stats:
            col_stats = stats[col]
            
            if col == 'Class':
                # Fraud label: ~0.17% fraud rate (matching original dataset)
                data[col] = rng.choice([0, 1], size=num_rows, p=[0.9983, 0.0017])
            elif col == 'Time':
                # Time in seconds (0 to ~172800 for 2 days)
                data[col] = rng.uniform(0, 172800, size=num_rows)
            elif col == 'Amount':
                # Log-normal distribution for transaction amounts
                data[col] = np.abs(rng.lognormal(mean=3.0, sigma=2.0, size=num_rows))
                data[col] = np.clip(data[col], 0, 25000)
            else:
                # V1-V28 features: normal distribution based on observed stats
                data[col] = rng.normal(
                    loc=col_stats['mean'],
                    scale=max(col_stats['std'], 0.01),
                    size=num_rows
                )
        else:
            data[col] = ['synthetic'] * num_rows
    
    return pd.DataFrame(data)


def worker_process(
    worker_id: int,
    schema: Dict,
    output_path: str,
    stop_flag: Value,
    records_counter: Value,
    bytes_counter: Value,
    chunk_size: int
):
    """Worker process that generates and writes data"""
    # Each process gets its own RNG with unique seed
    rng = np.random.default_rng(seed=worker_id * 12345 + int(time.time() * 1000) % 100000)
    
    columns = schema['columns']
    stats = schema['stats']
    
    output_dir = Path(output_path)
    file_path = output_dir / f"thread_{worker_id:03d}_data.csv"
    header_written = False
    
    local_records = 0
    local_bytes = 0
    
    try:
        while stop_flag.value == 0:
            # Generate chunk
            chunk = generate_chunk(columns, stats, chunk_size, rng)
            
            # Write chunk
            mode = 'a' if header_written else 'w'
            header = not header_written
            
            csv_data = chunk.to_csv(index=False, header=header)
            bytes_written = len(csv_data.encode('utf-8'))
            
            with open(file_path, mode) as f:
                f.write(csv_data)
            
            header_written = True
            local_records += len(chunk)
            local_bytes += bytes_written
            
            # Update shared counters periodically (every 5 chunks to reduce lock contention)
            if local_records >= chunk_size * 5:
                with records_counter.get_lock():
                    records_counter.value += local_records
                with bytes_counter.get_lock():
                    bytes_counter.value += local_bytes
                local_records = 0
                local_bytes = 0
                
    except Exception as e:
        print(f"Worker {worker_id} error: {e}", file=sys.stderr)
    finally:
        # Final update
        if local_records > 0:
            with records_counter.get_lock():
                records_counter.value += local_records
            with bytes_counter.get_lock():
                bytes_counter.value += local_bytes


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
        
        # Use Value with int (0=running, 1=stop) instead of Event for better cross-process compatibility
        self.stop_flag = mp.Value('i', 0)
        self.records_counter = mp.Value(c_longlong, 0)
        self.bytes_counter = mp.Value(c_longlong, 0)
        self.schema: Optional[Dict] = None
        self.processes: List[Process] = []
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully"""
        logger.info(f"\nReceived signal {signum}, initiating graceful shutdown...")
        self.stop_flag.value = 1
    
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
        self.schema = load_schema(self.template_path)
        
        logger.info(f"Workers:         {self.num_workers}")
        logger.info(f"Chunk size:      {self.chunk_size:,} rows")
        logger.info(f"Duration:        {self.duration_seconds} seconds")
        logger.info("=" * 70)
    
    def run(self):
        """Execute the stress test"""
        self.setup()
        
        logger.info(f"Starting {self.num_workers} worker processes...")
        
        # Start worker processes
        for worker_id in range(self.num_workers):
            p = Process(
                target=worker_process,
                args=(
                    worker_id,
                    self.schema,
                    str(self.output_path),  # Pass as string for pickling
                    self.stop_flag,
                    self.records_counter,
                    self.bytes_counter,
                    self.chunk_size
                )
            )
            p.start()
            self.processes.append(p)
        
        logger.info(f"All {self.num_workers} workers started. Running for {self.duration_seconds}s...")
        logger.info("-" * 70)
        
        # Monitor and report
        start_time = time.time()
        last_report_time = start_time
        last_records = 0
        last_bytes = 0
        report_interval = 5.0
        
        try:
            while self.stop_flag.value == 0:
                elapsed = time.time() - start_time
                
                if elapsed >= self.duration_seconds:
                    logger.info("\nDuration limit reached, stopping workers...")
                    self.stop_flag.value = 1
                    break
                
                # Report every 5 seconds
                if time.time() - last_report_time >= report_interval:
                    current_records = self.records_counter.value
                    current_bytes = self.bytes_counter.value
                    
                    interval_elapsed = time.time() - last_report_time
                    interval_records = current_records - last_records
                    interval_bytes = current_bytes - last_bytes
                    
                    interval_rps = interval_records / interval_elapsed if interval_elapsed > 0 else 0
                    interval_mbps = (interval_bytes / (1024 * 1024)) / interval_elapsed if interval_elapsed > 0 else 0
                    
                    gb_written = current_bytes / (1024 * 1024 * 1024)
                    
                    logger.info(
                        f"[{elapsed:6.1f}s] "
                        f"Records: {current_records:,} | "
                        f"Size: {gb_written:.2f} GB | "
                        f"Throughput: {interval_mbps:.1f} MB/s | "
                        f"Rate: {interval_rps:,.0f} rec/s"
                    )
                    
                    last_report_time = time.time()
                    last_records = current_records
                    last_bytes = current_bytes
                
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            logger.info("\nKeyboard interrupt received...")
            self.stop_flag.value = 1
        
        # Wait for workers to finish
        logger.info("Waiting for workers to complete...")
        for p in self.processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2)
        
        # Final report
        self._final_report(start_time)
    
    def _final_report(self, start_time: float):
        """Print final summary statistics"""
        total_elapsed = time.time() - start_time
        total_records = self.records_counter.value
        total_bytes = self.bytes_counter.value
        
        gb_written = total_bytes / (1024 * 1024 * 1024)
        avg_mbps = (total_bytes / (1024 * 1024)) / total_elapsed if total_elapsed > 0 else 0
        avg_rps = total_records / total_elapsed if total_elapsed > 0 else 0
        
        logger.info("=" * 70)
        logger.info("FINAL RESULTS - FlashBlade Stress Test Complete")
        logger.info("=" * 70)
        logger.info(f"Duration:        {total_elapsed:.1f} seconds")
        logger.info(f"Total Records:   {total_records:,}")
        logger.info(f"Total Data:      {gb_written:.2f} GB")
        logger.info(f"Avg Throughput:  {avg_mbps:.1f} MB/s")
        logger.info(f"Avg Rate:        {avg_rps:,.0f} records/s")
        logger.info("=" * 70)
        
        # List output files
        output_files = list(self.output_path.glob("thread_*_data.csv"))
        if output_files:
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
    duration_seconds = int(os.getenv('DURATION_SECONDS', '300'))
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