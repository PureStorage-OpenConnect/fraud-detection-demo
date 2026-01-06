#!/usr/bin/env python3
"""
Pod 6: FlashArray + Triton Benchmark
Aggressive I/O stress test designed to light up FlashArray metrics.

Strategy for defeating cache:
1. Create large unique files with random data (defeats dedup)
2. Use O_DIRECT via dd subprocess (bypasses page cache)  
3. Massive parallelism (many concurrent readers)
4. Sequential + random read patterns
5. Continuous file creation to defeat any caching
"""

import os
import sys
import time
import json
import shutil
import logging
import threading
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from dataclasses import dataclass, field

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
log = logging.getLogger(__name__)


@dataclass
class IOMetrics:
    """Thread-safe metrics tracker."""
    ops: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    latencies_ms: List[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def add_read(self, bytes_count: int, latency_ms: float):
        with self._lock:
            self.ops += 1
            self.bytes_read += bytes_count
            if len(self.latencies_ms) < 10000:  # Cap for memory
                self.latencies_ms.append(latency_ms)
    
    def add_write(self, bytes_count: int, latency_ms: float):
        with self._lock:
            self.ops += 1
            self.bytes_written += bytes_count
            if len(self.latencies_ms) < 10000:
                self.latencies_ms.append(latency_ms)
    
    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    @property
    def iops(self) -> float:
        return self.ops / max(self.elapsed, 0.001)
    
    @property  
    def read_mbps(self) -> float:
        return (self.bytes_read / (1024**2)) / max(self.elapsed, 0.001)
    
    @property
    def write_mbps(self) -> float:
        return (self.bytes_written / (1024**2)) / max(self.elapsed, 0.001)
    
    @property
    def avg_latency_ms(self) -> float:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else 0
    
    @property
    def p95_latency_ms(self) -> float:
        if len(self.latencies_ms) < 2:
            return self.avg_latency_ms
        sorted_lat = sorted(self.latencies_ms)
        return sorted_lat[int(len(sorted_lat) * 0.95)]


def check_direct_io_support(test_dir: Path) -> bool:
    """Check if O_DIRECT is supported on this filesystem."""
    test_file = test_dir / "_direct_io_test"
    try:
        # Create small test file
        test_file.write_bytes(os.urandom(4096))
        # Try dd with direct
        result = subprocess.run(
            ['dd', f'if={test_file}', 'of=/dev/null', 'bs=4k', 'iflag=direct', 'count=1'],
            capture_output=True, text=True, timeout=5
        )
        test_file.unlink()
        return result.returncode == 0
    except Exception as e:
        log.warning(f"Direct I/O test failed: {e}")
        if test_file.exists():
            test_file.unlink()
        return False


def drop_caches():
    """Try to drop page cache (requires root)."""
    try:
        subprocess.run(['sync'], check=True, timeout=10)
        with open('/proc/sys/vm/drop_caches', 'w') as f:
            f.write('3')
        log.info("Dropped page cache")
        return True
    except:
        return False


class FlashArrayBenchmark:
    """Aggressive FlashArray I/O stress test."""
    
    def __init__(self, base_dir: str, duration: int, num_workers: int, 
                 file_size_mb: int = 50, num_files: int = 200):
        self.base_dir = Path(base_dir)
        self.duration = duration
        self.num_workers = num_workers
        self.file_size_mb = file_size_mb
        self.file_size = file_size_mb * 1024 * 1024
        self.num_files = num_files
        
        self.test_dir = self.base_dir / f"_fa_stress_{int(time.time())}"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = IOMetrics()
        self.stop_event = threading.Event()
        self.files: List[Path] = []
        self.use_direct = False
        
    def setup(self):
        """Create test files with random data."""
        log.info("")
        log.info("=" * 70)
        log.info("FLASHARRAY I/O STRESS TEST")
        log.info("=" * 70)
        log.info(f"Directory:   {self.test_dir}")
        log.info(f"Workers:     {self.num_workers}")
        log.info(f"Duration:    {self.duration}s")
        log.info(f"File size:   {self.file_size_mb} MB")
        log.info(f"Num files:   {self.num_files}")
        log.info(f"Total data:  {self.num_files * self.file_size_mb / 1024:.1f} GB")
        log.info("")
        
        # Check direct I/O support
        self.use_direct = check_direct_io_support(self.test_dir)
        if self.use_direct:
            log.info("✓ O_DIRECT supported - will bypass page cache")
        else:
            log.info("✗ O_DIRECT not supported - using large working set to defeat cache")
        
        # Try to drop caches
        if drop_caches():
            log.info("✓ Dropped page cache")
        
        # Create files with unique random data
        log.info(f"Creating {self.num_files} test files...")
        
        # Use multiple threads to speed up file creation
        def create_file(idx):
            filepath = self.test_dir / f"test_{idx:04d}.bin"
            # Generate unique random data (defeats dedup)
            # Mix of random + unique identifier
            data = os.urandom(self.file_size)
            filepath.write_bytes(data)
            return filepath
        
        with ThreadPoolExecutor(max_workers=min(32, self.num_workers)) as executor:
            futures = [executor.submit(create_file, i) for i in range(self.num_files)]
            for i, future in enumerate(as_completed(futures)):
                self.files.append(future.result())
                if (i + 1) % 50 == 0:
                    log.info(f"  Created {i + 1}/{self.num_files} files...")
        
        total_size_gb = len(self.files) * self.file_size / (1024**3)
        log.info(f"✓ Created {len(self.files)} files ({total_size_gb:.1f} GB)")
        
        # Drop caches again after file creation
        drop_caches()
        
    def _read_file_direct(self, filepath: Path) -> tuple:
        """Read file using dd with O_DIRECT."""
        start = time.time()
        result = subprocess.run(
            ['dd', f'if={filepath}', 'of=/dev/null', 
             f'bs={min(self.file_size, 4*1024*1024)}',  # 4MB blocks
             'iflag=direct'],
            capture_output=True, text=True
        )
        elapsed_ms = (time.time() - start) * 1000
        return (self.file_size, elapsed_ms) if result.returncode == 0 else (0, elapsed_ms)
    
    def _read_file_standard(self, filepath: Path) -> tuple:
        """Read file using standard Python I/O."""
        start = time.time()
        with open(filepath, 'rb') as f:
            data = f.read()
        elapsed_ms = (time.time() - start) * 1000
        return (len(data), elapsed_ms)
    
    def _worker_read(self, worker_id: int):
        """Worker that continuously reads files."""
        # Start at different points to spread load
        file_idx = worker_id * 7
        
        while not self.stop_event.is_set():
            filepath = self.files[file_idx % len(self.files)]
            file_idx += 1
            
            try:
                if self.use_direct:
                    bytes_read, latency = self._read_file_direct(filepath)
                else:
                    bytes_read, latency = self._read_file_standard(filepath)
                
                if bytes_read > 0:
                    self.metrics.add_read(bytes_read, latency)
            except Exception as e:
                if not self.stop_event.is_set():
                    pass  # Ignore errors during shutdown
    
    def _worker_mixed(self, worker_id: int):
        """Worker that does mixed read/write operations."""
        file_idx = worker_id * 7
        write_file = self.test_dir / f"write_{worker_id:03d}.bin"
        write_data = os.urandom(self.file_size)  # Pre-generate for speed
        
        while not self.stop_event.is_set():
            # 80% reads, 20% writes
            if np.random.random() < 0.8:
                filepath = self.files[file_idx % len(self.files)]
                file_idx += 1
                try:
                    if self.use_direct:
                        bytes_read, latency = self._read_file_direct(filepath)
                    else:
                        bytes_read, latency = self._read_file_standard(filepath)
                    if bytes_read > 0:
                        self.metrics.add_read(bytes_read, latency)
                except:
                    pass
            else:
                try:
                    start = time.time()
                    write_file.write_bytes(write_data)
                    os.sync()  # Force to disk
                    elapsed_ms = (time.time() - start) * 1000
                    self.metrics.add_write(len(write_data), elapsed_ms)
                except:
                    pass
    
    def run(self, mixed_workload: bool = False):
        """Run the stress test."""
        log.info("")
        log.info(f"Starting stress test with {self.num_workers} workers...")
        log.info(f"Workload: {'Mixed (80% read, 20% write)' if mixed_workload else 'Read-only'}")
        log.info("")
        
        # Reset metrics
        self.metrics = IOMetrics()
        
        # Start workers
        worker_fn = self._worker_mixed if mixed_workload else self._worker_read
        threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=worker_fn, args=(i,))
            t.daemon = True
            t.start()
            threads.append(t)
        
        # Progress reporting
        log.info(f"{'Time':<8} {'IOPS':>10} {'Read MB/s':>12} {'Write MB/s':>12} {'Avg Lat':>10} {'P95 Lat':>10}")
        log.info("-" * 70)
        
        start_time = time.time()
        last_ops = 0
        last_time = start_time
        
        while (time.time() - start_time) < self.duration:
            time.sleep(5)
            
            # Calculate interval stats
            current_ops = self.metrics.ops
            current_time = time.time()
            interval = current_time - last_time
            interval_iops = (current_ops - last_ops) / interval
            
            log.info(f"{self.metrics.elapsed:>6.1f}s "
                    f"{interval_iops:>10.0f} "
                    f"{self.metrics.read_mbps:>12.1f} "
                    f"{self.metrics.write_mbps:>12.1f} "
                    f"{self.metrics.avg_latency_ms:>9.1f}ms "
                    f"{self.metrics.p95_latency_ms:>9.1f}ms")
            
            last_ops = current_ops
            last_time = current_time
        
        # Stop workers
        self.stop_event.set()
        for t in threads:
            t.join(timeout=2)
        
        log.info("-" * 70)
        log.info("")
        log.info("RESULTS:")
        log.info(f"  Total I/O operations: {self.metrics.ops:,}")
        log.info(f"  Sustained IOPS:       {self.metrics.iops:,.0f}")
        log.info(f"  Read throughput:      {self.metrics.read_mbps:,.1f} MB/s")
        log.info(f"  Write throughput:     {self.metrics.write_mbps:,.1f} MB/s")
        log.info(f"  Average latency:      {self.metrics.avg_latency_ms:.2f} ms")
        log.info(f"  P95 latency:          {self.metrics.p95_latency_ms:.2f} ms")
        log.info(f"  Data read:            {self.metrics.bytes_read / (1024**3):.2f} GB")
        log.info(f"  Data written:         {self.metrics.bytes_written / (1024**3):.2f} GB")
        
    def cleanup(self):
        """Remove test files."""
        log.info("")
        log.info("Cleaning up test files...")
        shutil.rmtree(self.test_dir, ignore_errors=True)
        log.info("Done")


# =============================================================================
# Triton Inference Test (simplified)
# =============================================================================
def run_triton_test(model_dir: str, data_dir: str, triton_url: str,
                    duration: int, num_workers: int, batch_size: int):
    """Stress test Triton inference."""
    try:
        import tritonclient.grpc as grpcclient
    except ImportError:
        log.error("tritonclient[grpc] not installed - skipping Triton test")
        return
    
    import pandas as pd
    
    log.info("")
    log.info("=" * 70)
    log.info("TRITON INFERENCE STRESS TEST")
    log.info("=" * 70)
    
    # Find model
    model_path = Path(model_dir)
    model_name = None
    feature_names = []
    
    for name in ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]:
        if (model_path / name).exists():
            model_name = name
            feat_file = model_path / name / "feature_names.json"
            if feat_file.exists():
                with open(feat_file) as f:
                    feature_names = json.load(f)
            break
    
    if not model_name:
        log.error("No model found!")
        return
    
    log.info(f"Model:      {model_name}")
    log.info(f"Features:   {len(feature_names)}")
    log.info(f"Triton:     {triton_url}")
    log.info(f"Workers:    {num_workers}")
    log.info(f"Batch:      {batch_size}")
    
    # Check Triton
    try:
        client = grpcclient.InferenceServerClient(url=triton_url)
        if not client.is_server_live():
            log.error("Triton not live")
            return
        if not client.is_model_ready(model_name):
            log.error(f"Model {model_name} not ready")
            return
    except Exception as e:
        log.error(f"Cannot connect to Triton: {e}")
        return
    
    # Generate synthetic data if no real data
    log.info("Generating test data...")
    num_features = len(feature_names) if feature_names else 21
    test_data = np.random.randn(batch_size * 100, num_features).astype(np.float32)
    batches = [test_data[i:i+batch_size] for i in range(0, len(test_data), batch_size)]
    log.info(f"Created {len(batches)} batches")
    
    # Run test
    metrics = IOMetrics()
    stop_event = threading.Event()
    
    def worker(worker_id):
        try:
            cli = grpcclient.InferenceServerClient(url=triton_url)
        except:
            return
        
        batch_idx = worker_id
        while not stop_event.is_set():
            batch = batches[batch_idx % len(batches)]
            batch_idx += 1
            
            try:
                inputs = [grpcclient.InferInput("input__0", batch.shape, "FP32")]
                inputs[0].set_data_from_numpy(batch)
                
                start = time.time()
                result = cli.infer(model_name, inputs)
                elapsed_ms = (time.time() - start) * 1000
                
                metrics.add_read(len(batch), elapsed_ms)
            except Exception as e:
                if not stop_event.is_set():
                    pass
                time.sleep(0.01)
    
    # Start workers
    threads = []
    for i in range(num_workers):
        t = threading.Thread(target=worker, args=(i,))
        t.daemon = True
        t.start()
        threads.append(t)
    
    log.info("")
    log.info(f"{'Time':<8} {'Records':>12} {'Records/s':>12} {'Avg Lat':>10} {'P95 Lat':>10}")
    log.info("-" * 60)
    
    start_time = time.time()
    while (time.time() - start_time) < duration:
        time.sleep(5)
        rps = metrics.bytes_read / metrics.elapsed if metrics.elapsed > 0 else 0
        log.info(f"{metrics.elapsed:>6.1f}s "
                f"{metrics.bytes_read:>12,} "
                f"{rps:>12,.0f} "
                f"{metrics.avg_latency_ms:>9.1f}ms "
                f"{metrics.p95_latency_ms:>9.1f}ms")
    
    stop_event.set()
    for t in threads:
        t.join(timeout=2)
    
    log.info("-" * 60)
    log.info("")
    log.info("RESULTS:")
    log.info(f"  Total inferences:  {metrics.ops:,}")
    log.info(f"  Total records:     {metrics.bytes_read:,}")
    log.info(f"  Throughput:        {metrics.bytes_read / metrics.elapsed:,.0f} records/sec")
    log.info(f"  Average latency:   {metrics.avg_latency_ms:.2f} ms")
    log.info(f"  P95 latency:       {metrics.p95_latency_ms:.2f} ms")


# =============================================================================
# MAIN
# =============================================================================
def main():
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    data_dir = os.getenv('DATA_DIR', '/data/input')
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    
    # FlashArray settings - AGGRESSIVE DEFAULTS
    fa_workers = int(os.getenv('FA_WORKERS', '64'))
    fa_file_size_mb = int(os.getenv('FA_FILE_SIZE_MB', '50'))
    fa_num_files = int(os.getenv('FA_NUM_FILES', '200'))
    fa_mixed = os.getenv('FA_MIXED', 'false').lower() == 'true'
    
    # Triton settings
    triton_url = os.getenv('TRITON_URL', 'inference:8001')
    triton_workers = int(os.getenv('TRITON_WORKERS', '8'))
    triton_batch = int(os.getenv('TRITON_BATCH_SIZE', '1000'))
    
    # Test selection
    run_fa = os.getenv('RUN_FA', 'true').lower() == 'true'
    run_triton = os.getenv('RUN_TRITON', 'true').lower() == 'true'
    
    log.info("=" * 70)
    log.info("Pod 6: Storage & Inference Benchmark")
    log.info("=" * 70)
    log.info(f"FlashArray: {run_fa} (workers={fa_workers}, files={fa_num_files}x{fa_file_size_mb}MB)")
    log.info(f"Triton:     {run_triton} (workers={triton_workers}, batch={triton_batch})")
    log.info(f"Duration:   {duration}s per test")
    
    if run_fa:
        bench = FlashArrayBenchmark(
            base_dir=model_dir,
            duration=duration,
            num_workers=fa_workers,
            file_size_mb=fa_file_size_mb,
            num_files=fa_num_files
        )
        try:
            bench.setup()
            bench.run(mixed_workload=fa_mixed)
        finally:
            bench.cleanup()
    
    if run_triton:
        run_triton_test(model_dir, data_dir, triton_url, duration, triton_workers, triton_batch)
    
    log.info("")
    log.info("=" * 70)
    log.info("BENCHMARK COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()