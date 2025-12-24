#!/usr/bin/env python3
"""
Pod 1: High-Performance Data Gather Service
============================================
Stress-testing tool for Pure Storage FlashBlade that generates massive amounts
of synthetic credit card transaction data using parallel workers.

Features:
- Parallel worker processes (avoids Python GIL)
- Schema-based generation from Kaggle creditcard.csv template
- Timestamped output directories for run tracking
- Real-time throughput monitoring via filesystem stats
- Configurable runtime duration (default: 5 minutes)
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


def generate_chunk(columns: List[str], stats: Dict, num_rows: int, rng) -> pd.DataFrame:
    """Generate synthetic data matching the schema"""
    data = {}
    
    for col in columns:
        if col in stats:
            if col == 'Class':
                data[col] = rng.choice([0, 1], size=num_rows, p=[0.9983, 0.0017])
            elif col == 'Time':
                data[col] = rng.uniform(0, 172800, size=num_rows)
            elif col == 'Amount':
                data[col] = np.clip(np.abs(rng.lognormal(3.0, 2.0, num_rows)), 0, 25000)
            else:
                data[col] = rng.normal(stats[col]['mean'], max(stats[col]['std'], 0.01), num_rows)
        else:
            data[col] = ['synthetic'] * num_rows
    
    return pd.DataFrame(data)


def worker_main(worker_id: int, schema: Dict, output_dir: str, chunk_size: int, duration: int):
    """Standalone worker process - runs until duration expires"""
    rng = np.random.default_rng(seed=worker_id * 12345 + int(time.time() * 1000) % 100000)
    columns = schema['columns']
    stats = schema['stats']
    
    file_path = Path(output_dir) / f"worker_{worker_id:03d}.csv"
    start_time = time.time()
    header_written = False
    
    while (time.time() - start_time) < duration:
        chunk = generate_chunk(columns, stats, chunk_size, rng)
        mode = 'a' if header_written else 'w'
        chunk.to_csv(file_path, mode=mode, header=not header_written, index=False)
        header_written = True
    
    return worker_id


def get_dir_stats(output_path: Path) -> tuple:
    """Get total size and file count from directory"""
    files = list(output_path.glob("worker_*.csv"))
    if not files:
        return 0, 0
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes, len(files)


def signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM"""
    global STOP_FLAG
    log(f"Received signal {signum}, stopping...")
    STOP_FLAG = True


def run_stress_test(
    template_path: Path,
    output_base: Path,
    num_workers: int,
    duration_seconds: int,
    chunk_size: int
):
    """Main stress test orchestrator"""
    global STOP_FLAG
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: FlashBlade High-Performance Stress Test")
    log("=" * 70)
    log(f"Output directory: {output_path}")
    
    # Load schema
    schema = load_schema(template_path)
    
    log(f"Workers:    {num_workers}")
    log(f"Chunk size: {chunk_size:,} rows")
    log(f"Duration:   {duration_seconds} seconds")
    log("=" * 70)
    
    # Save schema for workers (they'll import it)
    import json
    schema_file = output_path / "_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    # Start worker processes using subprocess for clean isolation
    log(f"Starting {num_workers} worker processes...")
    
    worker_script = f'''
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

def generate_chunk(columns, stats, num_rows, rng):
    data = {{}}
    for col in columns:
        if col in stats:
            if col == 'Class':
                data[col] = rng.choice([0, 1], size=num_rows, p=[0.9983, 0.0017])
            elif col == 'Time':
                data[col] = rng.uniform(0, 172800, size=num_rows)
            elif col == 'Amount':
                data[col] = np.clip(np.abs(rng.lognormal(3.0, 2.0, num_rows)), 0, 25000)
            else:
                data[col] = rng.normal(stats[col]['mean'], max(stats[col]['std'], 0.01), num_rows)
        else:
            data[col] = ['synthetic'] * num_rows
    return pd.DataFrame(data)

worker_id = int(sys.argv[1])
output_dir = sys.argv[2]
chunk_size = int(sys.argv[3])
duration = int(sys.argv[4])
schema_file = sys.argv[5]

with open(schema_file) as f:
    schema = json.load(f)

rng = np.random.default_rng(seed=worker_id * 12345 + int(time.time() * 1000) % 100000)
columns = schema['columns']
stats = schema['stats']

file_path = Path(output_dir) / f"worker_{{worker_id:03d}}.csv"
start_time = time.time()
header_written = False

while (time.time() - start_time) < duration:
    chunk = generate_chunk(columns, stats, chunk_size, rng)
    mode = 'a' if header_written else 'w'
    chunk.to_csv(file_path, mode=mode, header=not header_written, index=False)
    header_written = True
'''
    
    # Launch all workers
    processes = []
    for worker_id in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', worker_script, 
             str(worker_id), str(output_path), str(chunk_size), 
             str(duration_seconds), str(schema_file)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(p)
    
    log(f"All {num_workers} workers launched. Monitoring for {duration_seconds}s...")
    log("-" * 70)
    
    # Monitor progress using filesystem stats
    start_time = time.time()
    last_bytes = 0
    last_time = start_time
    report_interval = 5.0
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        
        # Check if duration exceeded
        if elapsed >= duration_seconds + 10:  # Extra 10s grace period
            break
        
        # Check if all workers finished
        running = sum(1 for p in processes if p.poll() is None)
        if running == 0:
            break
        
        # Report every 5 seconds
        if time.time() - last_time >= report_interval:
            current_bytes, file_count = get_dir_stats(output_path)
            interval_time = time.time() - last_time
            interval_bytes = current_bytes - last_bytes
            
            mbps = (interval_bytes / (1024 * 1024)) / interval_time if interval_time > 0 else 0
            gb_total = current_bytes / (1024 * 1024 * 1024)
            
            # Estimate records (approx 200 bytes per row for this schema)
            est_records = current_bytes // 200
            rps = (interval_bytes // 200) / interval_time if interval_time > 0 else 0
            
            log(f"[{elapsed:5.0f}s] Files: {file_count:3d} | "
                f"Size: {gb_total:6.2f} GB | "
                f"Speed: {mbps:6.1f} MB/s | "
                f"~{rps:,.0f} rec/s | "
                f"Workers: {running}")
            
            last_bytes = current_bytes
            last_time = time.time()
        
        time.sleep(1)
    
    # Terminate any remaining workers
    log("Stopping workers...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    # Wait for cleanup
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    # Final report
    total_elapsed = time.time() - start_time
    final_bytes, final_files = get_dir_stats(output_path)
    final_gb = final_bytes / (1024 * 1024 * 1024)
    avg_mbps = (final_bytes / (1024 * 1024)) / total_elapsed if total_elapsed > 0 else 0
    est_records = final_bytes // 200
    
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"Output:      {output_path}")
    log(f"Duration:    {total_elapsed:.1f} seconds")
    log(f"Files:       {final_files}")
    log(f"Total Size:  {final_gb:.2f} GB ({final_bytes:,} bytes)")
    log(f"Throughput:  {avg_mbps:.1f} MB/s average")
    log(f"Est Records: ~{est_records:,}")
    log("=" * 70)
    
    # Cleanup schema file
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
    chunk_size = int(os.getenv('CHUNK_SIZE', '10000'))
    
    template_path = Path(template_dir) / template_file
    output_base = Path(output_dir)
    
    log("Configuration:")
    log(f"  Template: {template_path}")
    log(f"  Output:   {output_base}")
    log(f"  Workers:  {num_workers}")
    log(f"  Duration: {duration_seconds}s")
    log(f"  Chunk:    {chunk_size} rows")
    
    run_stress_test(
        template_path=template_path,
        output_base=output_base,
        num_workers=num_workers,
        duration_seconds=duration_seconds,
        chunk_size=chunk_size
    )


if __name__ == "__main__":
    main()