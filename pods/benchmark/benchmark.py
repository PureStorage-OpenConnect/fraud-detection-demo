#!/usr/bin/env python3
"""
Pod 6: Combined Storage & Inference Benchmark
Simple, sequential tests:
1. FlashArray model loading stress test
2. Triton inference throughput test
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass, field
import statistics

import numpy as np
import pandas as pd
import xgboost as xgb

# Triton client
try:
    import tritonclient.grpc as grpcclient
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)


def is_valid_parquet(filepath: Path) -> bool:
    """Check if file has valid parquet magic bytes."""
    try:
        if filepath.stat().st_size < 100:
            return False
        with open(filepath, 'rb') as f:
            # Check header
            if f.read(4) != b'PAR1':
                return False
            # Check footer
            f.seek(-4, 2)
            if f.read(4) != b'PAR1':
                return False
        return True
    except:
        return False


@dataclass
class Metrics:
    """Simple metrics tracker."""
    count: int = 0
    total_time_ms: float = 0.0
    total_bytes: int = 0
    times: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add(self, time_ms: float, bytes_or_records: int = 0):
        with self._lock:
            self.count += 1
            self.total_time_ms += time_ms
            self.total_bytes += bytes_or_records
            self.times.append(time_ms)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def rate(self) -> float:
        return self.count / self.elapsed if self.elapsed > 0 else 0
    
    @property
    def throughput_mb(self) -> float:
        return (self.total_bytes / (1024**2)) / self.elapsed if self.elapsed > 0 else 0
    
    @property
    def avg_ms(self) -> float:
        return self.total_time_ms / self.count if self.count > 0 else 0
    
    @property
    def p95_ms(self) -> float:
        if len(self.times) < 2:
            return self.avg_ms
        return sorted(self.times)[int(len(self.times) * 0.95)]


# =============================================================================
# PART 1: FlashArray Stress Test
# =============================================================================
def run_flasharray_test(model_dir: str, duration: int, num_workers: int, num_copies: int):
    """Stress test FlashArray with model loading - bypasses page cache using dd."""
    log.info("")
    log.info("=" * 70)
    log.info("PART 1: FLASHARRAY MODEL LOADING STRESS TEST")
    log.info("=" * 70)
    
    model_path = Path(model_dir)
    
    # Find model
    model_file = None
    model_size = 0
    for model_dir_name in ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]:
        for fname in ["xgboost.json", "model.json"]:
            candidate = model_path / model_dir_name / "1" / fname
            if candidate.exists():
                model_file = candidate
                model_size = candidate.stat().st_size
                log.info(f"Model: {candidate}")
                log.info(f"Size:  {model_size / 1024:.1f} KB")
                break
        if model_file:
            break
    
    if not model_file:
        log.error("No model found!")
        return
    
    # Create copies to defeat cache
    temp_dir = model_path / "_stress_test_copies"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(exist_ok=True)
    
    log.info(f"Creating {num_copies} model copies...")
    model_content = model_file.read_bytes()
    
    copies = []
    for i in range(num_copies):
        copy_path = temp_dir / f"model_{i:04d}.json"
        # Add unique content to defeat dedup
        unique = f'\n{{"_copy_id": {i}, "_timestamp": {time.time_ns()}, "_random": "{os.urandom(32).hex()}"}}'
        copy_path.write_bytes(model_content + unique.encode())
        copies.append(copy_path)
    
    actual_size = copies[0].stat().st_size
    log.info(f"Created {len(copies)} copies ({len(copies) * actual_size / (1024**2):.1f} MB)")
    
    # Test if dd with direct works
    import subprocess
    test_result = subprocess.run(
        ['dd', f'if={copies[0]}', 'of=/dev/null', 'bs=1M', 'iflag=direct', 'count=1'],
        capture_output=True, text=True
    )
    use_direct = test_result.returncode == 0
    if use_direct:
        log.info("Using dd iflag=direct to bypass page cache")
    else:
        log.info(f"Direct I/O failed: {test_result.stderr.strip()}")
        log.info("Falling back to standard reads with large working set")
        # Create more copies to exceed cache
        if len(copies) < 500:
            log.info(f"Creating additional copies to exceed cache...")
            for i in range(len(copies), 500):
                copy_path = temp_dir / f"model_{i:04d}.json"
                unique = f'\n{{"_copy_id": {i}, "_timestamp": {time.time_ns()}, "_random": "{os.urandom(32).hex()}"}}'
                copy_path.write_bytes(model_content + unique.encode())
                copies.append(copy_path)
            log.info(f"Now have {len(copies)} copies ({len(copies) * actual_size / (1024**2):.1f} MB)")
    
    try:
        log.info(f"Running {duration}s stress test with {num_workers} workers...")
        log.info("")
        
        metrics = Metrics()
        stop_event = threading.Event()
        
        def worker(worker_id):
            idx = worker_id * 7  # Spread starting points
            
            while not stop_event.is_set():
                filepath = copies[idx % len(copies)]
                idx += 1
                try:
                    start = time.time()
                    
                    if use_direct:
                        # Use dd with direct flag - bypasses page cache
                        result = subprocess.run(
                            ['dd', f'if={filepath}', 'of=/dev/null', 'bs=1M', 'iflag=direct'],
                            capture_output=True, text=True
                        )
                        if result.returncode != 0:
                            continue
                        bytes_read = actual_size
                    else:
                        # Standard read
                        with open(filepath, 'rb') as f:
                            data = f.read()
                        bytes_read = len(data)
                    
                    # Also load into XGBoost to validate
                    model = xgb.Booster()
                    model.load_model(str(filepath))
                    
                    elapsed_ms = (time.time() - start) * 1000
                    metrics.add(elapsed_ms, bytes_read)
                    del model
                except Exception as e:
                    log.warning(f"Worker {worker_id} error: {e}")
        
        # Start workers
        workers = []
        for i in range(num_workers):
            t = threading.Thread(target=worker, args=(i,))
            t.daemon = True
            t.start()
            workers.append(t)
        
        # Progress reporting
        log.info(f"{'Time':<8} {'Loads':>10} {'Loads/s':>12} {'Latency':>12} {'MB/s':>10}")
        log.info(f"{'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        
        start_time = time.time()
        while (time.time() - start_time) < duration:
            time.sleep(5)
            log.info(f"{metrics.elapsed:>6.1f}s {metrics.count:>10,} "
                    f"{metrics.rate:>10.1f}/s {metrics.avg_ms:>10.1f}ms "
                    f"{metrics.throughput_mb:>8.1f}")
        
        stop_event.set()
        for t in workers:
            t.join(timeout=2)
        
        log.info(f"{'-'*8} {'-'*10} {'-'*12} {'-'*12} {'-'*10}")
        log.info(f"{'TOTAL':<8} {metrics.count:>10,} {metrics.rate:>10.1f}/s "
                f"{metrics.avg_ms:>10.1f}ms {metrics.throughput_mb:>8.1f}")
        
        log.info("")
        log.info("FlashArray Results:")
        log.info(f"  ✓ {metrics.count:,} model loads in {duration}s")
        log.info(f"  ✓ {metrics.throughput_mb:.1f} MB/s sustained throughput")
        log.info(f"  ✓ {metrics.avg_ms:.1f}ms avg latency, {metrics.p95_ms:.1f}ms P95")
        if use_direct:
            log.info(f"  ✓ Direct I/O bypassed page cache")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        log.info("Cleaned up model copies")


# =============================================================================
# PART 2: Triton Inference Test
# =============================================================================
def run_triton_test(model_dir: str, data_dir: str, triton_url: str, 
                    duration: int, num_workers: int, batch_size: int):
    """Stress test Triton inference."""
    log.info("")
    log.info("=" * 70)
    log.info("PART 2: TRITON INFERENCE STRESS TEST")
    log.info("=" * 70)
    
    if not GRPC_AVAILABLE:
        log.error("tritonclient[grpc] not installed - skipping Triton test")
        return
    
    # Find model name and features
    model_path = Path(model_dir)
    model_name = None
    feature_names = []
    
    for model_dir_name in ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]:
        candidate = model_path / model_dir_name
        if candidate.exists():
            model_name = model_dir_name
            feature_file = candidate / "feature_names.json"
            if feature_file.exists():
                with open(feature_file) as f:
                    feature_names = json.load(f)
            break
    
    if not model_name:
        log.error("No model found!")
        return
    
    log.info(f"Model:      {model_name}")
    log.info(f"Triton:     {triton_url}")
    log.info(f"Workers:    {num_workers}")
    log.info(f"Batch size: {batch_size}")
    
    # Check Triton connectivity
    try:
        client = grpcclient.InferenceServerClient(url=triton_url)
        if not client.is_server_live():
            log.error(f"Triton not live at {triton_url}")
            return
        if not client.is_model_ready(model_name):
            log.error(f"Model {model_name} not ready")
            return
        log.info("Triton server ready!")
    except Exception as e:
        log.error(f"Cannot connect to Triton: {e}")
        return
    
    # Load test data
    log.info("Loading test data...")
    data_path = Path(data_dir)
    run_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    
    if not run_dirs:
        log.error("No data found!")
        return
    
    # Get valid parquet files only
    all_files = list(run_dirs[-1].glob("worker_*.parquet"))
    parquet_files = [f for f in all_files if is_valid_parquet(f)]
    log.info(f"Found {len(parquet_files)} valid parquet files (of {len(all_files)} total)")
    
    if not parquet_files:
        log.error("No valid parquet files found!")
        return
    
    # Load up to 5 files
    dfs = []
    for f in parquet_files[:5]:
        try:
            df = pd.read_parquet(f)
            dfs.append(df)
        except Exception as e:
            log.warning(f"Error reading {f.name}: {e}")
            continue
    
    if not dfs:
        log.error("Could not load any parquet files!")
        return
    
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Loaded {len(df):,} records")
    
    # Feature engineering (same as training)
    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'])
        df['amt_scaled'] = (df['amt'] - df['amt'].mean()) / (df['amt'].std() + 0.001)
    
    if 'unix_time' in df.columns:
        df['hour_of_day'] = (df['unix_time'] / 3600) % 24
        df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype('int8')
    
    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        df['distance_km'] = np.sqrt(
            ((df['merch_lat'] - df['lat']) * 111.0)**2 +
            ((df['merch_long'] - df['long']) * 85.0)**2
        )
    
    for col in ['category_encoded', 'state_encoded', 'gender_encoded', 'city_pop_log', 'zip_region']:
        if col not in df.columns:
            df[col] = 0
    
    # Prepare feature matrix
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0
    
    features = df[feature_names].fillna(0).values.astype(np.float32)
    log.info(f"Features: {features.shape}")
    
    # Create batches
    num_batches = len(features) // batch_size
    batches = [features[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    log.info(f"Created {len(batches)} batches of {batch_size} records")
    
    # Run inference test
    log.info(f"Running {duration}s inference test...")
    log.info("")
    
    metrics = Metrics()
    fraud_count = 0
    fraud_lock = threading.Lock()
    stop_event = threading.Event()
    
    def worker(worker_id):
        nonlocal fraud_count
        try:
            client = grpcclient.InferenceServerClient(url=triton_url)
        except:
            return
        
        batch_idx = worker_id
        while not stop_event.is_set():
            batch = batches[batch_idx % len(batches)]
            batch_idx += 1
            
            try:
                inputs = [grpcclient.InferInput("input__0", batch.shape, "FP32")]
                inputs[0].set_data_from_numpy(batch)
                outputs = [grpcclient.InferRequestedOutput("output__0")]
                
                start = time.time()
                result = client.infer(model_name, inputs, outputs=outputs)
                elapsed_ms = (time.time() - start) * 1000
                
                metrics.add(elapsed_ms, len(batch))
                
                predictions = result.as_numpy("output__0")
                with fraud_lock:
                    fraud_count += int((predictions > 0.5).sum())
                    
            except Exception as e:
                if not stop_event.is_set():
                    log.warning(f"Inference error: {e}")
                time.sleep(0.1)
    
    # Start workers
    workers = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        t.daemon = True
        t.start()
        workers.append(t)
    
    # Progress reporting
    log.info(f"{'Time':<8} {'Records':>12} {'Throughput':>15} {'Latency':>12} {'Fraud':>10}")
    log.info(f"{'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*10}")
    
    start_time = time.time()
    while (time.time() - start_time) < duration:
        time.sleep(5)
        records = metrics.total_bytes
        throughput = records / metrics.elapsed if metrics.elapsed > 0 else 0
        log.info(f"{metrics.elapsed:>6.1f}s {records:>12,} {throughput:>13,.0f}/s "
                f"{metrics.avg_ms:>10.1f}ms {fraud_count:>10,}")
    
    stop_event.set()
    for t in workers:
        t.join(timeout=2)
    
    total_records = metrics.total_bytes
    throughput = total_records / metrics.elapsed
    
    log.info(f"{'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*10}")
    log.info(f"{'TOTAL':<8} {total_records:>12,} {throughput:>13,.0f}/s "
            f"{metrics.avg_ms:>10.1f}ms {fraud_count:>10,}")
    
    log.info("")
    log.info("Triton Results:")
    log.info(f"  ✓ {total_records:,} records processed in {duration}s")
    log.info(f"  ✓ {throughput:,.0f} records/sec throughput")
    log.info(f"  ✓ {metrics.avg_ms:.1f}ms avg batch latency, {metrics.p95_ms:.1f}ms P95")
    log.info(f"  ✓ {fraud_count:,} fraud transactions detected")


# =============================================================================
# MAIN
# =============================================================================
def main():
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    data_dir = os.getenv('DATA_DIR', '/data/input')
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    
    # FlashArray settings
    fa_workers = int(os.getenv('FA_WORKERS', '64'))
    fa_copies = int(os.getenv('FA_COPIES', '100'))
    
    # Triton settings
    triton_url = os.getenv('TRITON_URL', 'inference:8001')
    triton_workers = int(os.getenv('TRITON_WORKERS', '8'))
    triton_batch = int(os.getenv('TRITON_BATCH_SIZE', '1000'))
    
    # Test selection
    run_fa = os.getenv('RUN_FA', 'true').lower() == 'true'
    run_triton = os.getenv('RUN_TRITON', 'true').lower() == 'true'
    
    log.info("=" * 70)
    log.info("Pod 6: Combined Storage & Inference Benchmark")
    log.info("=" * 70)
    log.info(f"Duration:     {duration}s per test")
    log.info(f"FlashArray:   {run_fa} (workers={fa_workers}, copies={fa_copies})")
    log.info(f"Triton:       {run_triton} (workers={triton_workers}, batch={triton_batch})")
    log.info("=" * 70)
    
    if run_fa:
        run_flasharray_test(model_dir, duration, fa_workers, fa_copies)
    
    if run_triton:
        run_triton_test(model_dir, data_dir, triton_url, duration, triton_workers, triton_batch)
    
    log.info("")
    log.info("=" * 70)
    log.info("BENCHMARK COMPLETE")
    log.info("=" * 70)
    log.info("Check Grafana for:")
    if run_fa:
        log.info("  - FlashArray: IOPS, Bandwidth, Latency")
    if run_triton:
        log.info("  - Triton: Inference Rate, Compute Latency")
    log.info("=" * 70)


if __name__ == "__main__":
    main()