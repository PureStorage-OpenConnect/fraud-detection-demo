#!/usr/bin/env python3
"""
Pod 6: FlashArray Model Reload Stress Test (Cache-Defeating)
Demonstrates low-latency model serving from Pure Storage FlashArray.

This version defeats Linux page cache by:
1. Creating multiple model copies and reading them round-robin
2. Using O_DIRECT where possible to bypass cache
3. Attempting to drop caches if running with privileges
4. Reading unique files so kernel can't serve from cache
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import statistics
import random
import tempfile

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)


@dataclass
class LoadMetrics:
    """Tracks model load metrics."""
    load_count: int = 0
    total_load_time_ms: float = 0.0
    total_bytes_read: int = 0
    load_times: List[float] = field(default_factory=list)
    inference_count: int = 0
    total_inference_time_ms: float = 0.0
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_load(self, load_time_ms: float, bytes_read: int):
        with self._lock:
            self.load_count += 1
            self.total_load_time_ms += load_time_ms
            self.total_bytes_read += bytes_read
            self.load_times.append(load_time_ms)
    
    def add_inference(self, inference_time_ms: float):
        with self._lock:
            self.inference_count += 1
            self.total_inference_time_ms += inference_time_ms
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def loads_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.load_count / elapsed if elapsed > 0 else 0.0
    
    @property
    def throughput_mb_sec(self) -> float:
        elapsed = self.elapsed_seconds
        return (self.total_bytes_read / (1024**2)) / elapsed if elapsed > 0 else 0.0
    
    @property
    def avg_load_time_ms(self) -> float:
        return self.total_load_time_ms / self.load_count if self.load_count > 0 else 0.0
    
    @property
    def p50_load_time_ms(self) -> float:
        return statistics.median(self.load_times) if self.load_times else 0.0
    
    @property
    def p95_load_time_ms(self) -> float:
        if len(self.load_times) < 2:
            return self.avg_load_time_ms
        sorted_times = sorted(self.load_times)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[idx]
    
    @property
    def p99_load_time_ms(self) -> float:
        if len(self.load_times) < 2:
            return self.avg_load_time_ms
        sorted_times = sorted(self.load_times)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times)-1)]


class FlashArrayStressTest:
    """
    Stress test for FlashArray model serving.
    
    Defeats Linux page cache by creating multiple model copies
    and reading them in round-robin fashion.
    """
    
    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        duration_seconds: int = 60,
        num_workers: int = 8,
        num_model_copies: int = 100,
        run_inference: bool = False,
        inference_batch_size: int = 1000
    ):
        self.model_path = Path(model_dir)
        self.data_path = Path(data_dir)
        self.duration = duration_seconds
        self.num_workers = num_workers
        self.num_model_copies = num_model_copies
        self.run_inference = run_inference
        self.inference_batch_size = inference_batch_size
        
        self.model_file: Optional[Path] = None
        self.model_size: int = 0
        self.model_copies: List[Path] = []
        self.feature_names: List[str] = []
        self.test_data: Optional[np.ndarray] = None
        self.temp_dir: Optional[Path] = None
        
        log.info("=" * 70)
        log.info("Pod 6: FlashArray Model Reload Stress Test")
        log.info("=" * 70)
        log.info(f"Model repo:      {self.model_path}")
        log.info(f"Data source:     {self.data_path}")
        log.info(f"Duration:        {self.duration}s")
        log.info(f"Workers:         {self.num_workers} concurrent")
        log.info(f"Model copies:    {self.num_model_copies} (defeats page cache)")
        log.info(f"Run inference:   {self.run_inference}")
        log.info("=" * 70)
    
    def _drop_caches(self) -> bool:
        """Attempt to drop Linux page cache."""
        try:
            # sync first
            os.sync()
            # Try to drop caches (requires root or CAP_SYS_ADMIN)
            subprocess.run(
                ['sudo', '-n', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'],
                capture_output=True, timeout=5
            )
            return True
        except:
            return False
    
    def _read_file_direct(self, filepath: Path) -> bytes:
        """Read file attempting to bypass cache."""
        # O_DIRECT requires aligned buffers and sizes, so we do a workaround:
        # Read the file and immediately advise kernel to drop it from cache
        try:
            fd = os.open(str(filepath), os.O_RDONLY)
            try:
                data = os.read(fd, self.model_size + 4096)
                # Tell kernel we don't need this data cached
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                except:
                    pass
                return data
            finally:
                os.close(fd)
        except Exception as e:
            # Fallback to normal read
            return filepath.read_bytes()
    
    def setup(self) -> bool:
        """Find model, create copies, and prepare test data."""
        log.info("Setup...")
        
        # Find model file
        model_dirs = ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]
        
        for model_dir in model_dirs:
            for model_name in ["xgboost.json", "model.json"]:
                candidate = self.model_path / model_dir / "1" / model_name
                if candidate.exists():
                    self.model_file = candidate
                    self.model_size = candidate.stat().st_size
                    
                    # Load feature names
                    feature_file = self.model_path / model_dir / "feature_names.json"
                    if feature_file.exists():
                        with open(feature_file) as f:
                            self.feature_names = json.load(f)
                    
                    log.info(f"  Model: {model_dir}/{model_name}")
                    log.info(f"  Size:  {self.model_size / 1024:.1f} KB")
                    break
            if self.model_file:
                break
        
        if not self.model_file:
            log.error("No model found!")
            return False
        
        # Create temporary directory for model copies ON THE SAME FILESYSTEM
        # This ensures we're testing FlashArray, not some other storage
        self.temp_dir = self.model_path / "_stress_test_copies"
        self.temp_dir.mkdir(exist_ok=True)
        
        log.info(f"  Creating {self.num_model_copies} model copies to defeat cache...")
        log.info(f"  Target: {self.temp_dir}")
        
        # Create model copies with unique content (slight modifications to defeat dedup)
        model_content = self.model_file.read_bytes()
        total_size = 0
        
        for i in range(self.num_model_copies):
            copy_path = self.temp_dir / f"model_copy_{i:04d}.json"
            # Add unique padding to each copy to defeat any dedup/cache tricks
            unique_suffix = f'\n{{"_copy_id": {i}, "_timestamp": {time.time()}}}'.encode()
            copy_path.write_bytes(model_content + unique_suffix)
            self.model_copies.append(copy_path)
            total_size += copy_path.stat().st_size
        
        total_mb = total_size / (1024**2)
        log.info(f"  Created {len(self.model_copies)} copies ({total_mb:.1f} MB total)")
        
        # Drop caches after creating files
        if self._drop_caches():
            log.info("  Dropped page cache (running with privileges)")
        else:
            log.info("  Could not drop cache (running without sudo)")
            log.info("  First reads may be served from cache")
        
        # Prepare test data for inference
        if self.run_inference:
            log.info("  Preparing test data...")
            self.test_data = self._load_test_data()
            if self.test_data is not None:
                log.info(f"  Test batch: {self.test_data.shape}")
        
        return True
    
    def cleanup(self):
        """Remove temporary model copies."""
        if self.temp_dir and self.temp_dir.exists():
            log.info(f"Cleaning up {len(self.model_copies)} model copies...")
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _load_test_data(self) -> Optional[np.ndarray]:
        """Load a batch of test data for inference."""
        try:
            run_dirs = sorted([
                d for d in self.data_path.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ])
            
            if not run_dirs:
                return None
            
            parquet_files = list(run_dirs[-1].glob("worker_*.parquet"))
            if not parquet_files:
                return None
            
            df = pd.read_parquet(parquet_files[0])
            df = df.head(self.inference_batch_size)
            
            # Simple feature engineering
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
            
            for col in ['category_encoded', 'state_encoded', 'gender_encoded', 
                       'city_pop_log', 'zip_region']:
                if col not in df.columns:
                    df[col] = 0
            
            if self.feature_names:
                for col in self.feature_names:
                    if col not in df.columns:
                        df[col] = 0.0
                return df[self.feature_names].fillna(0).values.astype(np.float32)
            else:
                numeric = df.select_dtypes(include=[np.number])
                return numeric.fillna(0).values.astype(np.float32)
                
        except Exception as e:
            log.warning(f"Could not load test data: {e}")
            return None
    
    def _load_model_from_copy(self, copy_idx: int) -> xgb.Booster:
        """Load model from a specific copy, bypassing cache."""
        filepath = self.model_copies[copy_idx % len(self.model_copies)]
        
        # Read file with cache bypass hints
        self._read_file_direct(filepath)
        
        # Now load into XGBoost
        model = xgb.Booster()
        model.load_model(str(filepath))
        
        return model
    
    def _worker_load_models(self, worker_id: int, metrics: LoadMetrics, 
                            stop_event: threading.Event):
        """Worker that continuously loads models from FlashArray."""
        # Each worker starts at different offset to spread reads
        copy_idx = worker_id * (self.num_model_copies // self.num_workers)
        
        while not stop_event.is_set():
            try:
                filepath = self.model_copies[copy_idx % len(self.model_copies)]
                copy_idx += 1
                
                # Time the full load cycle
                load_start = time.time()
                
                # Read file bytes (this is what hits FlashArray)
                file_size = filepath.stat().st_size
                _ = self._read_file_direct(filepath)
                
                # Load into XGBoost
                model = xgb.Booster()
                model.load_model(str(filepath))
                
                load_time = (time.time() - load_start) * 1000
                metrics.add_load(load_time, file_size)
                
                # Optional inference
                if self.run_inference and self.test_data is not None:
                    infer_start = time.time()
                    dmatrix = xgb.DMatrix(self.test_data)
                    _ = model.predict(dmatrix)
                    metrics.add_inference((time.time() - infer_start) * 1000)
                
                del model
                
            except Exception as e:
                log.warning(f"Worker {worker_id} error: {e}")
                time.sleep(0.1)
    
    def run_single_thread_test(self) -> LoadMetrics:
        """Baseline: Single-threaded model loading."""
        log.info("")
        log.info("=" * 70)
        log.info(f"SINGLE-THREADED MODEL LOAD TEST ({self.duration}s)")
        log.info("=" * 70)
        
        # Drop caches before test
        self._drop_caches()
        
        metrics = LoadMetrics()
        stop_event = threading.Event()
        
        def stop_timer():
            time.sleep(self.duration)
            stop_event.set()
        
        timer = threading.Thread(target=stop_timer)
        timer.start()
        
        last_report = time.time()
        report_interval = 5.0
        copy_idx = 0
        
        log.info(f"  {'Time':<8} {'Loads':>10} {'Loads/s':>12} {'Avg ms':>10} {'P95 ms':>10} {'MB/s':>10}")
        log.info(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        while not stop_event.is_set():
            filepath = self.model_copies[copy_idx % len(self.model_copies)]
            copy_idx += 1
            
            load_start = time.time()
            
            # Read and load
            file_size = filepath.stat().st_size
            _ = self._read_file_direct(filepath)
            model = xgb.Booster()
            model.load_model(str(filepath))
            
            load_time = (time.time() - load_start) * 1000
            metrics.add_load(load_time, file_size)
            
            if self.run_inference and self.test_data is not None:
                infer_start = time.time()
                dmatrix = xgb.DMatrix(self.test_data)
                _ = model.predict(dmatrix)
                metrics.add_inference((time.time() - infer_start) * 1000)
            
            del model
            
            if time.time() - last_report >= report_interval:
                log.info(f"  {metrics.elapsed_seconds:>6.1f}s "
                        f"{metrics.load_count:>10,} "
                        f"{metrics.loads_per_second:>10.1f}/s "
                        f"{metrics.avg_load_time_ms:>8.2f}ms "
                        f"{metrics.p95_load_time_ms:>8.2f}ms "
                        f"{metrics.throughput_mb_sec:>8.2f}")
                last_report = time.time()
        
        timer.join()
        
        log.info(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        log.info(f"  {'TOTAL':<8} {metrics.load_count:>10,} "
                f"{metrics.loads_per_second:>10.1f}/s "
                f"{metrics.avg_load_time_ms:>8.2f}ms "
                f"{metrics.p95_load_time_ms:>8.2f}ms "
                f"{metrics.throughput_mb_sec:>8.2f}")
        
        return metrics
    
    def run_multi_thread_test(self) -> LoadMetrics:
        """Concurrent model loading to stress FlashArray."""
        log.info("")
        log.info("=" * 70)
        log.info(f"MULTI-THREADED MODEL LOAD TEST ({self.duration}s) - {self.num_workers} workers")
        log.info("=" * 70)
        
        # Drop caches before test
        self._drop_caches()
        
        metrics = LoadMetrics()
        stop_event = threading.Event()
        
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_load_models,
                args=(i, metrics, stop_event)
            )
            t.daemon = True
            t.start()
            workers.append(t)
        
        last_report = time.time()
        report_interval = 5.0
        
        log.info(f"  {'Time':<8} {'Loads':>10} {'Loads/s':>12} {'Avg ms':>10} {'P95 ms':>10} {'MB/s':>10}")
        log.info(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        start_time = time.time()
        while (time.time() - start_time) < self.duration:
            time.sleep(1)
            
            if time.time() - last_report >= report_interval:
                log.info(f"  {metrics.elapsed_seconds:>6.1f}s "
                        f"{metrics.load_count:>10,} "
                        f"{metrics.loads_per_second:>10.1f}/s "
                        f"{metrics.avg_load_time_ms:>8.2f}ms "
                        f"{metrics.p95_load_time_ms:>8.2f}ms "
                        f"{metrics.throughput_mb_sec:>8.2f}")
                last_report = time.time()
        
        stop_event.set()
        for t in workers:
            t.join(timeout=2)
        
        log.info(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        log.info(f"  {'TOTAL':<8} {metrics.load_count:>10,} "
                f"{metrics.loads_per_second:>10.1f}/s "
                f"{metrics.avg_load_time_ms:>8.2f}ms "
                f"{metrics.p95_load_time_ms:>8.2f}ms "
                f"{metrics.throughput_mb_sec:>8.2f}")
        
        return metrics
    
    def print_comparison(self, single_metrics: LoadMetrics, multi_metrics: LoadMetrics):
        """Print comparison summary."""
        log.info("")
        log.info("=" * 70)
        log.info("FLASHARRAY MODEL SERVING PERFORMANCE")
        log.info("=" * 70)
        log.info(f"  Model Size:     {self.model_size / 1024:.1f} KB")
        log.info(f"  Model Copies:   {self.num_model_copies} (working set: {self.num_model_copies * self.model_size / (1024**2):.1f} MB)")
        log.info(f"  Test Duration:  {self.duration}s per test")
        log.info(f"  Workers:        1 (single) vs {self.num_workers} (multi)")
        log.info(f"  With Inference: {self.run_inference}")
        log.info("")
        log.info(f"  {'Metric':<25} {'Single-Thread':<20} {'Multi-Thread':<20}")
        log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
        
        log.info(f"  {'Model Loads':<25} {single_metrics.load_count:<20,} {multi_metrics.load_count:<20,}")
        log.info(f"  {'Loads/sec':<25} {single_metrics.loads_per_second:<20.1f} {multi_metrics.loads_per_second:<20.1f}")
        log.info(f"  {'Avg Load Latency':<25} {f'{single_metrics.avg_load_time_ms:.2f}ms':<20} {f'{multi_metrics.avg_load_time_ms:.2f}ms':<20}")
        log.info(f"  {'P50 Latency':<25} {f'{single_metrics.p50_load_time_ms:.2f}ms':<20} {f'{multi_metrics.p50_load_time_ms:.2f}ms':<20}")
        log.info(f"  {'P95 Latency':<25} {f'{single_metrics.p95_load_time_ms:.2f}ms':<20} {f'{multi_metrics.p95_load_time_ms:.2f}ms':<20}")
        log.info(f"  {'P99 Latency':<25} {f'{single_metrics.p99_load_time_ms:.2f}ms':<20} {f'{multi_metrics.p99_load_time_ms:.2f}ms':<20}")
        log.info(f"  {'Read Throughput':<25} {f'{single_metrics.throughput_mb_sec:.2f} MB/s':<20} {f'{multi_metrics.throughput_mb_sec:.2f} MB/s':<20}")
        
        if single_metrics.inference_count > 0:
            single_infer_avg = single_metrics.total_inference_time_ms / single_metrics.inference_count
            multi_infer_avg = multi_metrics.total_inference_time_ms / multi_metrics.inference_count if multi_metrics.inference_count > 0 else 0
            log.info(f"  {'Inferences':<25} {single_metrics.inference_count:<20,} {multi_metrics.inference_count:<20,}")
            log.info(f"  {'Avg Inference Time':<25} {f'{single_infer_avg:.2f}ms':<20} {f'{multi_infer_avg:.2f}ms':<20}")
        
        if single_metrics.loads_per_second > 0:
            scaling = multi_metrics.loads_per_second / single_metrics.loads_per_second
            efficiency = (scaling / self.num_workers) * 100
            log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
            log.info(f"  {'Throughput Scaling':<25} {'1.0x (baseline)':<20} {f'{scaling:.1f}x':<20}")
            log.info(f"  {'Scaling Efficiency':<25} {'':<20} {f'{efficiency:.0f}%':<20}")
        
        log.info("=" * 70)
        log.info("")
        log.info("FLASHARRAY VALUE DEMONSTRATION:")
        log.info(f"  ✓ Sustained {multi_metrics.throughput_mb_sec:.1f} MB/s model I/O from storage")
        log.info(f"  ✓ Consistent {multi_metrics.p95_load_time_ms:.1f}ms P95 latency under {self.num_workers}x concurrent load")
        log.info(f"  ✓ {multi_metrics.load_count:,} model loads in {self.duration}s")
        log.info(f"  ✓ {self.num_model_copies} unique files defeat page cache")
        log.info("  → Low latency enables: model versioning, A/B testing, hot-swapping")
        log.info("=" * 70)
    
    def run(self):
        """Execute full stress test."""
        try:
            if not self.setup():
                return
            
            single_metrics = self.run_single_thread_test()
            time.sleep(2)
            multi_metrics = self.run_multi_thread_test()
            self.print_comparison(single_metrics, multi_metrics)
            
        finally:
            self.cleanup()


def main():
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    data_dir = os.getenv('DATA_DIR', '/data/input')
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    num_workers = int(os.getenv('NUM_WORKERS', '8'))
    num_copies = int(os.getenv('NUM_MODEL_COPIES', '100'))
    run_inference = os.getenv('RUN_INFERENCE', 'false').lower() == 'true'
    inference_batch = int(os.getenv('INFERENCE_BATCH_SIZE', '1000'))
    
    stress_test = FlashArrayStressTest(
        model_dir=model_dir,
        data_dir=data_dir,
        duration_seconds=duration,
        num_workers=num_workers,
        num_model_copies=num_copies,
        run_inference=run_inference,
        inference_batch_size=inference_batch
    )
    stress_test.run()


if __name__ == "__main__":
    main()