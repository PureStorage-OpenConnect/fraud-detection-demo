#!/usr/bin/env python3
"""
Pod 1: Data Gather Service - High-performance synthetic transaction data generator
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

import pandas as pd
import numpy as np

# Configure logging ONCE with explicit single handler
logger = logging.getLogger('gather')
logger.setLevel(logging.INFO)
logger.handlers.clear()  # Remove any existing handlers
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
logger.addHandler(handler)
logger.propagate = False  # Prevent propagation to root logger

STOP_FLAG = False


def log(msg: str):
    logger.info(msg)


def load_schema(template_path: Path) -> Dict:
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
    files = list(output_path.glob(file_pattern))
    if not files:
        return 0, 0
    total_bytes = sum(f.stat().st_size for f in files)
    return total_bytes, len(files)


def signal_handler(signum, frame):
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
    global STOP_FLAG
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: Financial Fraud Data Generator")
    log("=" * 70)
    log(f"Output directory: {output_path}")
    
    schema = load_schema(template_path)
    
    log(f"Workers:    {num_workers}")
    log(f"Chunk size: {chunk_size:,} rows")
    log(f"Duration:   {duration_seconds} seconds")
    log(f"Format:     {output_format}")
    log("=" * 70)
    
    import json
    schema_file = output_path / "_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    log(f"Starting {num_workers} worker processes...")
    
    worker_script = '''
import sys, json, time
import numpy as np
import pandas as pd
from pathlib import Path

def gen(columns, stats, n, rng):
    data = {}
    for col in columns:
        if col in stats:
            if col == 'Class':
                data[col] = rng.integers(0, 2, size=n, dtype=np.int8)
            elif col == 'Time':
                data[col] = rng.uniform(0, 172800, size=n).astype(np.float32)
            elif col == 'Amount':
                data[col] = np.clip(np.abs(rng.lognormal(3.0, 2.0, n)), 0, 25000).astype(np.float32)
            else:
                data[col] = rng.normal(stats[col]['mean'], max(stats[col]['std'], 0.01), n).astype(np.float32)
    return pd.DataFrame(data)

wid, odir, chunk, dur, sf, fmt = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
with open(sf) as f: schema = json.load(f)
rng = np.random.default_rng(seed=wid * 12345 + int(time.time() * 1000) % 100000)
cols, stats, t0, fc = schema['columns'], schema['stats'], time.time(), 0

if fmt == 'parquet':
    import pyarrow as pa, pyarrow.parquet as pq
    while (time.time() - t0) < dur:
        pq.write_table(pa.Table.from_pandas(gen(cols, stats, chunk, rng), preserve_index=False), Path(odir)/f"worker_{wid:03d}_{fc:05d}.parquet", compression=None)
        fc += 1
elif fmt == 'binary':
    while (time.time() - t0) < dur:
        rng.standard_normal((chunk, 31)).astype(np.float32).tofile(Path(odir)/f"worker_{wid:03d}_{fc:05d}.bin")
        fc += 1
else:
    while (time.time() - t0) < dur:
        gen(cols, stats, chunk, rng).to_csv(Path(odir)/f"worker_{wid:03d}_{fc:05d}.csv", index=False)
        fc += 1
'''
    
    processes = []
    for worker_id in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', worker_script, 
             str(worker_id), str(output_path), str(chunk_size), 
             str(duration_seconds), str(schema_file), output_format],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        processes.append(p)
    
    log(f"All {num_workers} workers launched. Monitoring for {duration_seconds}s...")
    log("-" * 70)
    
    file_pattern = {"parquet": "worker_*.parquet", "binary": "worker_*.bin"}.get(output_format, "worker_*.csv")
    bytes_per_row = {"parquet": 130, "binary": 124}.get(output_format, 200)
    
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
            rps = (interval_bytes // bytes_per_row) / interval_time if interval_time > 0 else 0
            
            speed_str = f"{gbps:5.2f} GB/s" if mbps >= 1000 else f"{mbps:6.1f} MB/s"
            
            log(f"[{elapsed:5.0f}s] Files: {file_count:5d} | Size: {gb_total:6.2f} GB | Speed: {speed_str} | ~{rps/1e6:.2f}M rec/s | Workers: {running}")
            
            last_bytes = current_bytes
            last_time = time.time()
        
        time.sleep(1)
    
    log("Stopping workers...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    failed_workers = 0
    for p in processes:
        try:
            p.wait(timeout=5)
            if p.returncode and p.returncode != 0:
                failed_workers += 1
        except subprocess.TimeoutExpired:
            p.kill()
            failed_workers += 1
    
    if failed_workers > 0:
        log(f"WARNING: {failed_workers} workers exited with errors")
    
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
    log(f"Throughput:  {avg_mbps/1024:.2f} GB/s average" if avg_mbps >= 1000 else f"Throughput:  {avg_mbps:.1f} MB/s average")
    log(f"Est Records: ~{est_records:,}")
    log("=" * 70)
    
    schema_file.unlink(missing_ok=True)
    return output_path


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    template_dir = os.getenv('TEMPLATE_DIR', '/mnt/datasets/kaggle/creditcardfraud')
    template_file = os.getenv('TEMPLATE_FILE', 'creditcard.csv')
    output_dir = os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data')
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration_seconds = int(os.getenv('DURATION_SECONDS', '300'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '2000000'))
    output_format = os.getenv('OUTPUT_FORMAT', 'parquet')
    
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