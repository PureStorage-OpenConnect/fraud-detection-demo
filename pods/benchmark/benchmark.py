#!/usr/bin/env python3
"""
Pod 6: Sustained Throughput Benchmark
Measures sustained inference throughput over 60 seconds for CPU vs GPU.

GPU benchmark uses gRPC with concurrent workers to demonstrate true GPU throughput.
"""

import os
import sys
import time
import json
import logging
import threading
import concurrent.futures
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Iterator
from dataclasses import dataclass, field
from queue import Queue, Empty
import random

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

# Try to import tritonclient for gRPC (faster than HTTP)
try:
    import tritonclient.grpc as grpcclient
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# Feature engineering constants (must match Pod 2/3)
STRING_COLUMNS_TO_DROP = [
    'merchant', 'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num',
    'trans_date_trans_time', 'category', 'state', 'gender'
]

CATEGORIES = [
    'gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
    'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
    'kids_pets', 'travel', 'health_fitness', 'personal_care'
]

US_STATES = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
    'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
    'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY'
]


@dataclass
class BenchmarkMetrics:
    """Tracks benchmark metrics with thread safety."""
    records_processed: int = 0
    batches_processed: int = 0
    bytes_read: int = 0
    inference_time_ms: float = 0.0
    fraud_detected: int = 0
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add(self, records: int, batches: int, bytes_read: int, 
            inference_ms: float, fraud: int):
        with self._lock:
            self.records_processed += records
            self.batches_processed += batches
            self.bytes_read += bytes_read
            self.inference_time_ms += inference_ms
            self.fraud_detected += fraud
    
    @property
    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time
    
    @property
    def throughput_records_sec(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed > 0:
            return self.records_processed / elapsed
        return 0.0
    
    @property
    def throughput_mb_sec(self) -> float:
        elapsed = self.elapsed_seconds
        if elapsed > 0:
            return (self.bytes_read / (1024**2)) / elapsed
        return 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        if self.batches_processed > 0:
            return self.inference_time_ms / self.batches_processed
        return 0.0


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering (matches Pod 2 CPU path)."""
    df = df.copy()
    
    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'].values)
        mean, std = df['amt'].mean(), df['amt'].std()
        df['amt_scaled'] = (df['amt'] - mean) / std if std > 0.001 else 0.0
    
    if 'unix_time' in df.columns:
        hours = (df['unix_time'] / 3600) % 24
        df['hour_of_day'] = hours
        df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night'] = ((hours >= 22) | (hours <= 6)).astype('int8')
    
    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        dlat = (df['merch_lat'] - df['lat']) * 111.0
        dlon = (df['merch_long'] - df['long']) * 85.0
        df['distance_km'] = np.sqrt(dlat**2 + dlon**2)
    
    if 'category' in df.columns:
        cat_map = {c: i for i, c in enumerate(CATEGORIES)}
        df['category_encoded'] = df['category'].map(cat_map).fillna(-1).astype('int8')
    
    if 'state' in df.columns:
        state_map = {s: i for i, s in enumerate(US_STATES)}
        df['state_encoded'] = df['state'].map(state_map).fillna(-1).astype('int8')
    
    if 'gender' in df.columns:
        df['gender_encoded'] = (df['gender'] == 'M').astype('int8')
    
    if 'city_pop' in df.columns:
        df['city_pop_log'] = np.log1p(df['city_pop'].values)
    
    if 'zip' in df.columns:
        df['zip_region'] = (df['zip'] / 10000).astype('int8')
    
    cols_to_drop = [c for c in STRING_COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


class DataLoader:
    """Loads batches from FlashBlade parquet files."""
    
    def __init__(self, data_dir: Path, batch_size: int = 10000):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.files: List[Path] = []
        self._discover_files()
    
    def _discover_files(self):
        """Find all parquet files from most recent run."""
        run_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ])
        
        if not run_dirs:
            raise FileNotFoundError(f"No run directories in {self.data_dir}")
        
        run_dir = run_dirs[-1]
        self.files = sorted(run_dir.glob("worker_*.parquet"))
        
        if not self.files:
            raise FileNotFoundError(f"No parquet files in {run_dir}")
        
        log.info(f"  Data source: {run_dir.name}")
        log.info(f"  Files available: {len(self.files)}")
    
    def iterate_batches(self, duration_seconds: float) -> Iterator[tuple]:
        """Yield (dataframe, bytes_read) batches for specified duration."""
        start = time.time()
        file_idx = 0
        
        while (time.time() - start) < duration_seconds:
            filepath = self.files[file_idx % len(self.files)]
            file_idx += 1
            
            try:
                file_size = filepath.stat().st_size
                df = pd.read_parquet(filepath)
                
                for i in range(0, len(df), self.batch_size):
                    if (time.time() - start) >= duration_seconds:
                        return
                    
                    batch = df.iloc[i:i+self.batch_size].copy()
                    batch_bytes = int(file_size * len(batch) / len(df))
                    yield batch, batch_bytes
                    
            except Exception as e:
                log.warning(f"Error reading {filepath.name}: {e}")
                continue


class SustainedBenchmark:
    """Runs sustained throughput benchmark for CPU and GPU inference."""
    
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        triton_url: str,
        duration_seconds: int = 60,
        batch_size: int = 10000,
        num_workers: int = 8
    ):
        self.data_path = Path(data_dir)
        self.model_path = Path(model_dir)
        self.triton_http_url = triton_url.rstrip('/')
        # Extract host for gRPC (default port 8001)
        self.triton_grpc_url = triton_url.replace('http://', '').replace(':8000', ':8001')
        self.duration = duration_seconds
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.triton_model_name: Optional[str] = None
        
        log.info("=" * 70)
        log.info("Pod 6: Sustained Throughput Benchmark")
        log.info("=" * 70)
        log.info(f"Data source:    {self.data_path}")
        log.info(f"Model repo:     {self.model_path}")
        log.info(f"Triton HTTP:    {self.triton_http_url}")
        log.info(f"Triton gRPC:    {self.triton_grpc_url}")
        log.info(f"Duration:       {self.duration}s per model")
        log.info(f"Batch size:     {self.batch_size:,} records")
        log.info(f"GPU workers:    {self.num_workers} concurrent")
        log.info(f"gRPC available: {GRPC_AVAILABLE}")
        log.info("=" * 70)
    
    def load_model(self) -> bool:
        """Load XGBoost model for CPU inference."""
        log.info("Loading XGBoost model...")
        
        model_dirs = ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]
        model_file = None
        feature_file = None
        
        for model_dir in model_dirs:
            for model_name in ["xgboost.json", "model.json"]:
                candidate = self.model_path / model_dir / "1" / model_name
                if candidate.exists():
                    model_file = candidate
                    feature_file = self.model_path / model_dir / "feature_names.json"
                    log.info(f"  Found: {model_dir}/{model_name}")
                    break
            if model_file:
                break
        
        if model_file is None:
            log.error("No XGBoost model found")
            return False
        
        self.model = xgb.Booster()
        self.model.load_model(str(model_file))
        
        if feature_file and feature_file.exists():
            with open(feature_file) as f:
                self.feature_names = json.load(f)
            log.info(f"  Features: {len(self.feature_names)}")
        
        return True
    
    def check_triton(self) -> bool:
        """Check Triton and find model name."""
        log.info("Checking Triton server...")
        
        try:
            resp = requests.get(f"{self.triton_http_url}/v2/health/ready", timeout=5)
            if resp.status_code != 200:
                log.warning("  Triton not ready")
                return False
        except Exception as e:
            log.warning(f"  Cannot connect: {e}")
            return False
        
        for name in ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]:
            try:
                resp = requests.get(f"{self.triton_http_url}/v2/models/{name}", timeout=5)
                if resp.status_code == 200:
                    self.triton_model_name = name
                    log.info(f"  Triton model: {name}")
                    return True
            except:
                continue
        
        log.warning("  No fraud model found on Triton")
        return False
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Apply feature engineering and return feature matrix."""
        df = engineer_features(df)
        
        if self.feature_names:
            for col in self.feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            features = df[self.feature_names].fillna(0).values.astype(np.float32)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            exclude = ['is_fraud', 'cc_num', 'transaction_id']
            cols = [c for c in numeric_cols if c not in exclude]
            features = df[cols].fillna(0).values.astype(np.float32)
        
        return features
    
    def run_cpu_benchmark(self) -> BenchmarkMetrics:
        """Run sustained CPU inference benchmark."""
        log.info("")
        log.info("=" * 70)
        log.info(f"CPU INFERENCE BENCHMARK ({self.duration}s)")
        log.info("=" * 70)
        
        metrics = BenchmarkMetrics()
        data_loader = DataLoader(self.data_path, self.batch_size)
        
        last_report = time.time()
        report_interval = 5.0
        
        log.info(f"  {'Time':<8} {'Records':>12} {'Throughput':>15} {'Latency':>12} {'Read MB/s':>12}")
        log.info(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        
        for batch_df, batch_bytes in data_loader.iterate_batches(self.duration):
            features = self.prepare_features(batch_df)
            
            infer_start = time.time()
            dmatrix = xgb.DMatrix(features)
            predictions = self.model.predict(dmatrix)
            infer_time = (time.time() - infer_start) * 1000
            
            metrics.add(
                records=len(features),
                batches=1,
                bytes_read=batch_bytes,
                inference_ms=infer_time,
                fraud=int((predictions > 0.5).sum())
            )
            
            if time.time() - last_report >= report_interval:
                log.info(f"  {metrics.elapsed_seconds:>6.1f}s "
                        f"{metrics.records_processed:>12,} "
                        f"{metrics.throughput_records_sec:>12,.0f}/s "
                        f"{metrics.avg_latency_ms:>10.2f}ms "
                        f"{metrics.throughput_mb_sec:>10.1f}")
                last_report = time.time()
        
        log.info(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        log.info(f"  {'TOTAL':<8} {metrics.records_processed:>12,} "
                f"{metrics.throughput_records_sec:>12,.0f}/s "
                f"{metrics.avg_latency_ms:>10.2f}ms "
                f"{metrics.throughput_mb_sec:>10.1f}")
        
        return metrics
    
    def _gpu_worker_grpc(self, worker_id: int, batch_queue: Queue, 
                         metrics: BenchmarkMetrics, stop_event: threading.Event):
        """Worker thread for GPU inference via gRPC."""
        try:
            client = grpcclient.InferenceServerClient(url=self.triton_grpc_url)
            
            while not stop_event.is_set():
                try:
                    features, batch_bytes = batch_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                infer_start = time.time()
                
                # Create input tensor
                inputs = [grpcclient.InferInput("input__0", features.shape, "FP32")]
                inputs[0].set_data_from_numpy(features)
                
                # Run inference
                outputs = [grpcclient.InferRequestedOutput("output__0")]
                result = client.infer(self.triton_model_name, inputs, outputs=outputs)
                predictions = result.as_numpy("output__0")
                
                infer_time = (time.time() - infer_start) * 1000
                
                metrics.add(
                    records=len(features),
                    batches=1,
                    bytes_read=batch_bytes,
                    inference_ms=infer_time,
                    fraud=int((predictions > 0.5).sum())
                )
                
        except Exception as e:
            log.error(f"Worker {worker_id} error: {e}")
    
    def _gpu_worker_http(self, worker_id: int, batch_queue: Queue,
                         metrics: BenchmarkMetrics, stop_event: threading.Event):
        """Worker thread for GPU inference via HTTP."""
        session = requests.Session()
        url = f"{self.triton_http_url}/v2/models/{self.triton_model_name}/infer"
        
        while not stop_event.is_set():
            try:
                features, batch_bytes = batch_queue.get(timeout=0.1)
            except Empty:
                continue
            
            infer_start = time.time()
            
            try:
                payload = {
                    "inputs": [{
                        "name": "input__0",
                        "shape": list(features.shape),
                        "datatype": "FP32",
                        "data": features.flatten().tolist()
                    }]
                }
                resp = session.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                result = resp.json()
                predictions = np.array(result["outputs"][0]["data"])
                
                infer_time = (time.time() - infer_start) * 1000
                
                metrics.add(
                    records=len(features),
                    batches=1,
                    bytes_read=batch_bytes,
                    inference_ms=infer_time,
                    fraud=int((predictions > 0.5).sum())
                )
                
            except Exception as e:
                log.warning(f"Worker {worker_id} inference error: {e}")
    
    def run_gpu_benchmark(self) -> Optional[BenchmarkMetrics]:
        """Run sustained GPU inference benchmark with concurrent workers."""
        if not self.triton_model_name:
            log.warning("Skipping GPU benchmark - Triton not available")
            return None
        
        log.info("")
        log.info("=" * 70)
        protocol = "gRPC" if GRPC_AVAILABLE else "HTTP"
        log.info(f"GPU INFERENCE BENCHMARK ({self.duration}s) - {protocol} x{self.num_workers} workers")
        log.info("=" * 70)
        
        metrics = BenchmarkMetrics()
        batch_queue = Queue(maxsize=self.num_workers * 2)
        stop_event = threading.Event()
        
        # Choose worker function based on available protocol
        worker_fn = self._gpu_worker_grpc if GRPC_AVAILABLE else self._gpu_worker_http
        
        # Start worker threads
        workers = []
        for i in range(self.num_workers):
            t = threading.Thread(target=worker_fn, args=(i, batch_queue, metrics, stop_event))
            t.daemon = True
            t.start()
            workers.append(t)
        
        # Data loading in main thread
        data_loader = DataLoader(self.data_path, self.batch_size)
        
        last_report = time.time()
        report_interval = 5.0
        
        log.info(f"  {'Time':<8} {'Records':>12} {'Throughput':>15} {'Latency':>12} {'Read MB/s':>12}")
        log.info(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        
        for batch_df, batch_bytes in data_loader.iterate_batches(self.duration):
            features = self.prepare_features(batch_df)
            
            # Put in queue (blocks if full)
            batch_queue.put((features, batch_bytes))
            
            if time.time() - last_report >= report_interval:
                log.info(f"  {metrics.elapsed_seconds:>6.1f}s "
                        f"{metrics.records_processed:>12,} "
                        f"{metrics.throughput_records_sec:>12,.0f}/s "
                        f"{metrics.avg_latency_ms:>10.2f}ms "
                        f"{metrics.throughput_mb_sec:>10.1f}")
                last_report = time.time()
        
        # Signal workers to stop and wait for queue to drain
        time.sleep(2)  # Allow queue to drain
        stop_event.set()
        
        for t in workers:
            t.join(timeout=5)
        
        log.info(f"  {'-'*8} {'-'*12} {'-'*15} {'-'*12} {'-'*12}")
        log.info(f"  {'TOTAL':<8} {metrics.records_processed:>12,} "
                f"{metrics.throughput_records_sec:>12,.0f}/s "
                f"{metrics.avg_latency_ms:>10.2f}ms "
                f"{metrics.throughput_mb_sec:>10.1f}")
        
        return metrics
    
    def print_comparison(self, cpu_metrics: BenchmarkMetrics, gpu_metrics: Optional[BenchmarkMetrics]):
        """Print final comparison summary."""
        log.info("")
        log.info("=" * 70)
        log.info("SUSTAINED THROUGHPUT COMPARISON")
        log.info("=" * 70)
        log.info(f"  Test Duration:  {self.duration}s per model")
        log.info(f"  Batch Size:     {self.batch_size:,} records")
        log.info(f"  GPU Workers:    {self.num_workers} concurrent")
        log.info(f"  GPU Protocol:   {'gRPC' if GRPC_AVAILABLE else 'HTTP'}")
        log.info("")
        log.info(f"  {'Metric':<25} {'CPU (XGBoost)':<20} {'GPU (Triton)':<20}")
        log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
        
        cpu_tput = f"{cpu_metrics.throughput_records_sec:,.0f}/s"
        cpu_lat = f"{cpu_metrics.avg_latency_ms:.2f}ms"
        cpu_read = f"{cpu_metrics.throughput_mb_sec:.1f} MB/s"
        cpu_records = f"{cpu_metrics.records_processed:,}"
        cpu_fraud = f"{cpu_metrics.fraud_detected:,}"
        
        if gpu_metrics:
            gpu_tput = f"{gpu_metrics.throughput_records_sec:,.0f}/s"
            gpu_lat = f"{gpu_metrics.avg_latency_ms:.2f}ms"
            gpu_read = f"{gpu_metrics.throughput_mb_sec:.1f} MB/s"
            gpu_records = f"{gpu_metrics.records_processed:,}"
            gpu_fraud = f"{gpu_metrics.fraud_detected:,}"
        else:
            gpu_tput = gpu_lat = gpu_read = gpu_records = gpu_fraud = "N/A"
        
        log.info(f"  {'Records Processed':<25} {cpu_records:<20} {gpu_records:<20}")
        log.info(f"  {'Throughput':<25} {cpu_tput:<20} {gpu_tput:<20}")
        log.info(f"  {'Avg Batch Latency':<25} {cpu_lat:<20} {gpu_lat:<20}")
        log.info(f"  {'Data Read Rate':<25} {cpu_read:<20} {gpu_read:<20}")
        log.info(f"  {'Fraud Detected':<25} {cpu_fraud:<20} {gpu_fraud:<20}")
        
        if gpu_metrics and gpu_metrics.throughput_records_sec > 0:
            ratio = gpu_metrics.throughput_records_sec / cpu_metrics.throughput_records_sec
            if ratio > 1:
                log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
                log.info(f"  {'Result':<25} {'':<20} {'GPU is ' + f'{ratio:.1f}x faster':<20}")
            else:
                ratio = 1 / ratio
                log.info(f"  {'-'*25} {'-'*20} {'-'*20}")
                log.info(f"  {'Result':<25} {'CPU is ' + f'{ratio:.1f}x faster':<20} {'':<20}")
        
        log.info("=" * 70)
        log.info("")
        log.info("INTERPRETATION:")
        if GRPC_AVAILABLE:
            log.info("  - Using gRPC protocol for efficient GPU communication")
        else:
            log.info("  - Using HTTP (install tritonclient[grpc] for better performance)")
        log.info(f"  - {self.num_workers} concurrent workers maximize GPU utilization")
        log.info("  - Data read rate shows FlashBlade sustained I/O")
        log.info("  - GPU excels with concurrent requests (real-world pattern)")
        log.info("=" * 70)
    
    def run(self):
        """Execute full benchmark."""
        if not self.load_model():
            log.error("Failed to load model")
            return
        
        triton_ok = self.check_triton()
        
        cpu_metrics = self.run_cpu_benchmark()
        
        gpu_metrics = None
        if triton_ok:
            gpu_metrics = self.run_gpu_benchmark()
        
        self.print_comparison(cpu_metrics, gpu_metrics)


def main():
    data_dir = os.getenv('DATA_DIR', '/data/input')
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    triton_url = os.getenv('TRITON_URL', 'http://inference:8000')
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    batch_size = int(os.getenv('BATCH_SIZE', '10000'))
    num_workers = int(os.getenv('NUM_WORKERS', '8'))
    
    benchmark = SustainedBenchmark(
        data_dir=data_dir,
        model_dir=model_dir,
        triton_url=triton_url,
        duration_seconds=duration,
        batch_size=batch_size,
        num_workers=num_workers
    )
    benchmark.run()


if __name__ == "__main__":
    main()