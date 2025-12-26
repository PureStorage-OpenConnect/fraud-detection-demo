#!/usr/bin/env python3
"""
Pod 1: Data Gather Service
==========================
High-performance synthetic transaction data generator for the Financial Fraud
Detection demo. Generates realistic credit card transaction data at scale
using Pure Storage FlashBlade for high-throughput parallel writes.

This service demonstrates:
- Pure Storage FlashBlade parallel I/O capabilities
- Scalable data generation for ML training pipelines
- Schema-based synthetic data matching Kaggle creditcard.csv format

Features:
- Parallel worker processes (avoids Python GIL)
- Schema-based generation from Kaggle creditcard.csv template
- Timestamped output directories for run tracking
- Real-time throughput monitoring
- Configurable runtime duration (default: 5 minutes)
- Multiple output formats: Parquet, CSV, or raw binary
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd
import numpy as np

# Global stop flag for signal handling
STOP_FLAG = False


def log(msg: str):
    """Simple timestamped logging to stdout"""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} - {msg}", flush=True)


def load_schema(template_path: Path) -> Dict:
    """Load schema from creditcard.csv template"""
    log(f"Loading schema template from: {template_path}")
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template file not found: {template_path}")
    
    df_sample = pd.read_csv(template_path, nrows=10000)
    columns = list(df_sample.columns)
    stats = {}
    
    for col in columns:
        if np.issubdtype(df_sample[col].dtype, np.number):
            stats[col] = {
                'mean': float(df_sample[col].mean()),
                'std': float(df_sample[col].std())
            }
    
    log(f"Schema loaded: {len(columns)} columns")
    return {'columns': columns, 'stats': stats}


def get_dir_stats(output_path: Path, file_pattern: str) -> tuple:
    """Get total size and file count from directory"""
    files = list(output_path.glob(file_pattern))
    if not files:
        return 0, 0
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes, len(files)


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM"""
    global STOP_FLAG
    log(f"Received signal {signum}, stopping...")
    STOP_FLAG = True


def run_data_generation(
    template_path: Path,
    output_base: Path,
    num_workers: int,
    duration_seconds: int,
    chunk_size: int,
    output_format: str = 'parquet'
):
    """Main data generation orchestrator"""
    global STOP_FLAG
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: Financial Fraud Data Generator")
    log("=" * 70)
    log(f"Output directory: {output_path}")
    
    # Load schema
    schema = load_schema(template_path)
    
    log(f"Workers:    {num_workers}")
    log(f"Chunk size: {chunk_size:,} rows")
    log(f"Duration:   {duration_seconds} seconds")
    log(f"Format:     {output_format}")
    log("=" * 70)
    
    # Save schema for workers
    import json
    schema_file = output_path / "_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    log(f"Starting {num_workers} worker processes...")
    
    # Worker script optimized for throughput
    worker_script = f'''
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

def generate_data_fast(columns, stats, num_rows, rng):
    """Optimized data generation using pre-allocated arrays"""
    data = {{}}
    for col in columns:
        if col in stats:
            if col == 'Class':
                data[col] = rng.integers(0, 2, size=num_rows, dtype=np.int8)
            elif col == 'Time':
                data[col] = rng.uniform(0, 172800, size=num_rows).astype(np.float32)
            elif col == 'Amount':
                data[col] = np.clip(np.abs(rng.lognormal(3.0, 2.0, num_rows)), 0, 25000).astype(np.float32)
            else:
                data[col] = rng.normal(stats[col]['mean'], max(stats[col]['std'], 0.01), num_rows).astype(np.float32)
    return pd.DataFrame(data)

worker_id = int(sys.argv[1])
output_dir = sys.argv[2]
chunk_size = int(sys.argv[3])
duration = int(sys.argv[4])
schema_file = sys.argv[5]
output_format = sys.argv[6]

with open(schema_file) as f:
    schema = json.load(f)

rng = np.random.default_rng(seed=worker_id * 12345 + int(time.time() * 1000) % 100000)
columns = schema['columns']
stats = schema['stats']

start_time = time.time()
file_counter = 0

if output_format == 'parquet':
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    while (time.time() - start_time) < duration:
        chunk = generate_data_fast(columns, stats, chunk_size, rng)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.parquet"
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, file_path, compression=None)  # No compression for speed
        file_counter += 1

elif output_format == 'binary':
    # Raw binary numpy arrays - maximum speed
    while (time.time() - start_time) < duration:
        # Generate raw float32 array (31 columns x chunk_size rows)
        data = rng.standard_normal((chunk_size, 31)).astype(np.float32)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.bin"
        data.tofile(file_path)
        file_counter += 1

else:  # csv
    while (time.time() - start_time) < duration:
        chunk = generate_data_fast(columns, stats, chunk_size, rng)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.csv"
        chunk.to_csv(file_path, index=False)
        file_counter += 1
'''
    
    # Launch all workers
    processes = []
    for worker_id in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', worker_script, 
             str(worker_id), str(output_path), str(chunk_size), 
             str(duration_seconds), str(schema_file), output_format],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        processes.append(p)
    
    log(f"All {num_workers} workers launched. Monitoring for {duration_seconds}s...")
    log("-" * 70)
    
    # Determine file pattern based on format
    if output_format == 'parquet':
        file_pattern = "worker_*.parquet"
        bytes_per_row = 130  # Approximate for parquet
    elif output_format == 'binary':
        file_pattern = "worker_*.bin"
        bytes_per_row = 31 * 4  # 31 float32 columns
    else:
        file_pattern = "worker_*.csv"
        bytes_per_row = 200
    
    # Monitor progress
    start_time = time.time()
    last_bytes = 0
    last_time = start_time
    report_interval = 5.0
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        
        if elapsed >= duration_seconds + 10:
            break
        
        running = sum(1 for p in processes if p.poll() is None)
        if running == 0:
            break
        
        if time.time() - last_time >= report_interval:
            current_bytes, file_count = get_dir_stats(output_path, file_pattern)
            interval_time = time.time() - last_time
            interval_bytes = current_bytes - last_bytes
            
            mbps = (interval_bytes / (1024 * 1024)) / interval_time if interval_time > 0 else 0
            gbps = mbps / 1024
            gb_total = current_bytes / (1024 * 1024 * 1024)
            
            est_records = current_bytes // bytes_per_row
            rps = (interval_bytes // bytes_per_row) / interval_time if interval_time > 0 else 0
            
            # Color code based on throughput
            speed_str = f"{mbps:6.1f} MB/s"
            if mbps >= 1000:
                speed_str = f"{gbps:5.2f} GB/s"
            
            log(f"[{elapsed:5.0f}s] Files: {file_count:5d} | "
                f"Size: {gb_total:6.2f} GB | "
                f"Speed: {speed_str} | "
                f"~{rps/1e6:.2f}M rec/s | "
                f"Workers: {running}")
            
            last_bytes = current_bytes
            last_time = time.time()
        
        time.sleep(1)
    
    # Terminate workers
    log("Stopping workers...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    # Check for worker errors
    errors = []
    for i, p in enumerate(processes):
        if p.returncode and p.returncode != 0:
            stderr = p.stderr.read().decode() if p.stderr else ""
            if stderr:
                errors.append(f"Worker {i}: {stderr[:200]}")
    
    if errors:
        log(f"WARNING: {len(errors)} workers had errors")
        for err in errors[:3]:
            log(f"  {err}")
    
    # Final report
    total_elapsed = time.time() - start_time
    final_bytes, final_files = get_dir_stats(output_path, file_pattern)
    final_gb = final_bytes / (1024 * 1024 * 1024)
    avg_mbps = (final_bytes / (1024 * 1024)) / total_elapsed if total_elapsed > 0 else 0
    est_records = final_bytes // bytes_per_row
    
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"Output:      {output_path}")
    log(f"Format:      {output_format}")
    log(f"Duration:    {total_elapsed:.1f} seconds")
    log(f"Files:       {final_files:,}")
    log(f"Total Size:  {final_gb:.2f} GB ({final_bytes:,} bytes)")
    if avg_mbps >= 1000:
        log(f"Throughput:  {avg_mbps/1024:.2f} GB/s average")
    else:
        log(f"Throughput:  {avg_mbps:.1f} MB/s average")
    log(f"Est Records: ~{est_records:,}")
    log("=" * 70)
    
    schema_file.unlink(missing_ok=True)
    return output_path


def main():
    """Main entry point"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    template_dir = os.getenv('TEMPLATE_DIR', '/mnt/datasets/kaggle/creditcardfraud')
    template_file = os.getenv('TEMPLATE_FILE', 'creditcard.csv')
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data')
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration_seconds = int(os.getenv('DURATION_SECONDS', '300'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '2000000'))  # 2M rows = ~30 files/worker over 5min
    output_format = os.getenv('OUTPUT_FORMAT', 'parquet')  # parquet, csv, or binary
    
    template_path = Path(template_dir) / template_file
    output_base = Path(output_dir)
    
    log("Configuration:")
    log(f"  Template: {template_path}")
    log(f"  Output:   {output_base}")
    log(f"  Workers:  {num_workers}")
    log(f"  Duration: {duration_seconds}s")
    log(f"  Chunk:    {chunk_size} rows")
    log(f"  Format:   {output_format}")
    
    run_data_generation(
        template_path=template_path,
        output_base=output_base,
        num_workers=num_workers,
        duration_seconds=duration_seconds,
        chunk_size=chunk_size,
        output_format=output_format
    )


if __name__ == "__main__":
    main()