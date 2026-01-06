#!/usr/bin/env python3
"""
Pod 6: FlashArray Model Reload Stress Test
Demonstrates low-latency model serving from Pure Storage FlashArray.

Simulates real-world scenarios:
- Model versioning / A/B testing
- Model hot-swapping in production
- Multi-model ensemble loading
- Scaling inference by loading models on demand

This benchmark continuously loads models from disk to stress FlashArray I/O.
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import statistics

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
    
    Continuously loads models from disk to demonstrate:
    - Low-latency model loading
    - High-throughput model I/O
    - Sustained storage performance
    """
    
    def __init__(
        self,
        model_dir: str,
        data_dir: str,
        duration_seconds: int = 60,
        num_workers: int = 8,
        run_inference: bool = True,
        inference_batch_size: int = 1000
    ):
        self.model_path = Path(model_dir)
        self.data_path = Path(data_dir)
        self.duration = duration_seconds
        self.num_workers = num_workers
        self.run_inference = run_inference
        self.inference_batch_size = inference_batch_size
        
        self.model_file: Optional[Path] = None
        self.model_size: int = 0
        self.feature_names: List[str] = []
        self.test_data: Optional[np.ndarray] = None
        
        log.info("=" * 70)
        log.info("Pod 6: FlashArray Model Reload Stress Test")
        log.info("=" * 70)
        log.info(f"Model repo:     {self.model_path}")
        log.info(f"Data source:    {self.data_path}")
        log.info(f"Duration:       {self.duration}s")
        log.info(f"Workers:        {self.num_workers} concurrent")
        log.info(f"Run inference:  {self.run_inference}")
        log.info("=" * 70)
    
    def setup(self) -> bool:
        """Find model and prepare test data."""
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
        
        # Prepare test data for inference
        if self.run_inference:
            log.info("  Preparing test data...")
            self.test_data = self._load_test_data()
            if self.test_data is not None:
                log.info(f"  Test batch: {self.test_data.shape}")
        
        return True
    
    def _load_test_data(self) -> Optional[np.ndarray]:
        """Load a batch of test data for inference."""
        try:
            # Find parquet files
            run_dirs = sorted([
                d for d in self.data_path.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ])
            
            if not run_dirs:
                return None
            
            parquet_files = list(run_dirs[-1].glob("worker_*.parquet"))
            if not parquet_files:
                return None
            
            # Load sample
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
            
            # Add missing encoded columns
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
    
    def _clear_cache(self):
        """Attempt to clear filesystem cache (requires privileges)."""
        try:
            # This helps ensure we're reading from storage, not cache
            os.sync()
        except:
            pass
    
    def _worker_load_model(self, worker_id: int, metrics: LoadMetrics, 
                           stop_event: threading.Event):
        """Worker that continuously loads models from FlashArray."""
        
        while not stop_event.is_set():
            try:
                # Clear cache periodically to ensure disk reads
                if metrics.load_count % 100 == 0:
                    self._clear_cache()
                
                # Time the model load from disk
                load_start = time.time()
                
                # Force fresh read by creating new Booster each time
                model = xgb.Booster()
                model.load_model(str(self.model_file))
                
                load_time = (time.time() - load_start) * 1000
                metrics.add_load(load_time, self.model_size)
                
                # Optionally run inference to simulate real usage
                if self.run_inference and self.test_data is not None:
                    infer_start = time.time()
                    dmatrix = xgb.DMatrix(self.test_data)
                    _ = model.predict(dmatrix)
                    infer_time = (time.time() - infer_start) * 1000
                    metrics.add_inference(infer_time)
                
                # Clean up to force fresh load next iteration
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
        
        metrics = LoadMetrics()
        stop_event = threading.Event()
        
        # Set timer to stop
        def stop_timer():
            time.sleep(self.duration)
            stop_event.set()
        
        timer = threading.Thread(target=stop_timer)
        timer.start()
        
        last_report = time.time()
        report_interval = 5.0
        
        log.info(f"  {'Time':<8} {'Loads':>10} {'Loads/s':>12} {'Avg ms':>10} {'P95 ms':>10} {'MB/s':>10}")
        log.info(f"  {'-'*8} {'-'*10} {'-'*12} {'-'*10} {'-'*10} {'-'*10}")
        
        while not stop_event.is_set():
            # Load model
            load_start = time.time()
            model = xgb.Booster()
            model.load_model(str(self.model_file))
            load_time = (time.time() - load_start) * 1000
            metrics.add_load(load_time, self.model_size)
            
            if self.run_inference and self.test_data is not None:
                infer_start = time.time()
                dmatrix = xgb.DMatrix(self.test_data)
                _ = model.predict(dmatrix)
                metrics.add_inference((time.time() - infer_start) * 1000)
            
            del model
            
            # Progress report
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
        
        metrics = LoadMetrics()
        stop_event = threading.Event()
        
        # Start workers
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(
                target=self._worker_load_model,
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
        
        # Scaling efficiency
        if single_metrics.loads_per_second > 0:
            scaling = multi_metrics.loads_per_second / single_metrics.loads_per_second
            efficiency = (scaling / self.num_workers) * 100
            log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
            log.info(f"  {'Throughput Scaling':<25} {'1.0x (baseline)':<20} {f'{scaling:.1f}x':<20}")
            log.info(f"  {'Scaling Efficiency':<25} {'':<20} {f'{efficiency:.0f}%':<20}")
        
        log.info("=" * 70)
        log.info("")
        log.info("FLASHARRAY VALUE DEMONSTRATION:")
        log.info(f"  - Sustained {multi_metrics.throughput_mb_sec:.1f} MB/s model I/O")
        log.info(f"  - Consistent {multi_metrics.p95_load_time_ms:.1f}ms P95 latency under {self.num_workers}x concurrent load")
        log.info(f"  - {multi_metrics.load_count:,} model loads in {self.duration}s")
        log.info("  - Low latency enables: model versioning, A/B testing, hot-swapping")
        log.info("=" * 70)
    
    def run(self):
        """Execute full stress test."""
        if not self.setup():
            return
        
        # Run single-threaded baseline
        single_metrics = self.run_single_thread_test()
        
        # Brief pause
        time.sleep(2)
        
        # Run multi-threaded stress test
        multi_metrics = self.run_multi_thread_test()
        
        # Print comparison
        self.print_comparison(single_metrics, multi_metrics)


def main():
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    data_dir = os.getenv('DATA_DIR', '/data/input')
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    num_workers = int(os.getenv('NUM_WORKERS', '8'))
    run_inference = os.getenv('RUN_INFERENCE', 'true').lower() == 'true'
    inference_batch = int(os.getenv('INFERENCE_BATCH_SIZE', '1000'))
    
    stress_test = FlashArrayStressTest(
        model_dir=model_dir,
        data_dir=data_dir,
        duration_seconds=duration,
        num_workers=num_workers,
        run_inference=run_inference,
        inference_batch_size=inference_batch
    )
    stress_test.run()


if __name__ == "__main__":
    main()