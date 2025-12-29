#!/usr/bin/env python3
"""
Pod 2: Data Prepare Service (Multi-GPU RAPIDS)
===============================================
GPU-accelerated feature engineering using RAPIDS cuDF with Dask
for multi-GPU parallelism.
"""

# CRITICAL: Set multiprocessing start method BEFORE any other imports
# This prevents fork-related duplicate output issues
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

# Silence all distributed/dask logging before imports
import os
import logging

# Suppress Dask distributed logging completely
logging.getLogger('distributed').setLevel(logging.CRITICAL)
logging.getLogger('distributed.worker').setLevel(logging.CRITICAL)
logging.getLogger('distributed.scheduler').setLevel(logging.CRITICAL)
logging.getLogger('distributed.nanny').setLevel(logging.CRITICAL)
logging.getLogger('distributed.comm').setLevel(logging.CRITICAL)
logging.getLogger('bokeh').setLevel(logging.CRITICAL)
logging.getLogger('tornado').setLevel(logging.CRITICAL)
logging.getLogger('asyncio').setLevel(logging.CRITICAL)

# Environment variables to suppress library logging (set before imports)
os.environ['DASK_DISTRIBUTED__LOGGING__DISTRIBUTED'] = 'critical'
os.environ['DASK_DISTRIBUTED__WORKER__DAEMON'] = 'False'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'
os.environ['CUDF_LOGGING_LEVEL'] = 'CRITICAL'

import sys
import time
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Set
from dataclasses import dataclass

import cudf
import cupy as cp
import numpy as np
import pyarrow.parquet as pq

# Dask imports (after logging config)
import dask
dask.config.set({'distributed.logging.distributed': 'critical'})
dask.config.set({'distributed.worker.daemon': False})

import dask_cudf
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

STOP_FLAG = False
DASK_CLIENT = None
DASK_CLUSTER = None


@dataclass
class PrepConfig:
    input_dir: str
    output_dir: str
    poll_interval: int = 5
    batch_mode: bool = False
    max_files_per_run: int = 50
    latest_only: bool = True
    file_stable_seconds: int = 10
    use_multi_gpu: bool = True


def signal_handler(signum, frame):
    global STOP_FLAG
    log(f"Received shutdown signal")
    STOP_FLAG = True


def log(msg: str):
    """Single logging function - stdout only, no duplicates."""
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    sys.stdout.write(f"{ts} - {msg}\n")
    sys.stdout.flush()


class DirectoryWatcher:
    """Watch for new run directories from Pod 1."""
    
    def __init__(self, watch_dir: Path, state_dir: Path):
        self.watch_dir = watch_dir
        self.state_file = state_dir / ".prep_state.json"
        self.processed_dirs: Set[str] = self._load_state()
    
    def _load_state(self) -> Set[str]:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return set(json.load(f).get('processed', []))
            except:
                pass
        return set()
    
    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({'processed': list(self.processed_dirs)}, f)
        except Exception as e:
            log(f"WARNING: Could not save state: {e}")
    
    def get_new_runs(self, latest_only: bool = False, min_age_seconds: int = 10) -> List[Path]:
        """Get unprocessed run directories that are stable."""
        if not self.watch_dir.exists():
            return []
        
        now = time.time()
        new_runs = []
        
        for entry in self.watch_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("run_"):
                if entry.name not in self.processed_dirs:
                    try:
                        mtime = entry.stat().st_mtime
                        age = now - mtime
                        if age < min_age_seconds:
                            continue
                    except:
                        continue
                    
                    has_files = (list(entry.glob("worker_*.parquet"))[:1] or 
                                 list(entry.glob("worker_*.csv"))[:1] or
                                 list(entry.glob("worker_*.bin"))[:1])
                    if has_files:
                        new_runs.append(entry)
        
        new_runs = sorted(new_runs, key=lambda x: x.name)
        
        if latest_only and new_runs:
            return [new_runs[-1]]
        
        return new_runs
    
    def mark_processed(self, run_dir: Path):
        self.processed_dirs.add(run_dir.name)
        self._save_state()


class FeatureEngineer:
    """GPU-accelerated feature engineering using RAPIDS cuDF."""
    
    PCA_COLS = [f"V{i}" for i in range(1, 29)]
    
    def process(self, df: cudf.DataFrame, run_name: str) -> cudf.DataFrame:
        """Apply feature engineering on GPU."""
        start = time.time()
        num_records = len(df)
        original_cols = len(df.columns)
        log(f"Feature engineering on {num_records:,} records ({original_cols} columns)...")
        
        df['transaction_id'] = cp.arange(len(df), dtype=cp.int64)
        df['prep_timestamp'] = float(time.time())
        
        for col in self.PCA_COLS:
            if col in df.columns:
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                if std_val > 0.001:
                    df[col] = (df[col] - mean_val) / std_val
        
        if 'Amount' in df.columns:
            df['amount_log'] = cp.log1p(df['Amount'].values)
            amount_mean = float(df['Amount'].mean())
            amount_std = float(df['Amount'].std())
            if amount_std > 0.001:
                df['amount_scaled'] = (df['Amount'] - amount_mean) / amount_std
        
        if 'Time' in df.columns:
            df['hour_of_day'] = (df['Time'] / 3600) % 24
            df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype('int8')
        
        if 'V1' in df.columns and 'V2' in df.columns:
            df['V1_V2_interaction'] = df['V1'] * df['V2']
        
        if 'Amount' in df.columns and 'V1' in df.columns:
            df['amount_V1_interaction'] = df['amount_scaled'] * df['V1']
        
        for col in ['V1', 'V14', 'V17']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        elapsed = time.time() - start
        new_cols = len(df.columns) - original_cols
        throughput = num_records / elapsed if elapsed > 0 else 0
        log(f"  Added {new_cols} features in {elapsed:.1f}s ({throughput/1e6:.1f}M rec/s)")
        return df


class DataPrepService:
    def __init__(self, config: PrepConfig):
        self.config = config
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.watcher = DirectoryWatcher(self.input_path, self.output_path)
        self.engineer = FeatureEngineer()
        self.dask_client = None
        self.dask_cluster = None
        self.gpu_count = 1
        self.multi_gpu_enabled = False
        
        # Check GPU availability
        try:
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
            gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            gpu_mem = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] / (1024**3)
            log(f"GPU: {self.gpu_count}x {gpu_name} ({gpu_mem:.0f}GB each)")
        except Exception as e:
            log(f"WARNING: GPU check failed: {e}")
        
        # Initialize Dask for multi-GPU if available
        if config.use_multi_gpu and self.gpu_count > 1:
            self._init_dask_cluster()
        
        log("=" * 60)
        log("Pod 2: Data Prep Service (RAPIDS cuDF)")
        log("=" * 60)
        log(f"Input:       {self.input_path}")
        log(f"Output:      {self.output_path}")
        log(f"Mode:        {'batch' if config.batch_mode else 'continuous'}")
        log(f"Max files:   {config.max_files_per_run}")
        log(f"Latest only: {config.latest_only}")
        log(f"File stable: {config.file_stable_seconds}s")
        if self.multi_gpu_enabled:
            log(f"Multi-GPU:   ENABLED ({self.gpu_count} GPUs via Dask)")
        else:
            log(f"Multi-GPU:   disabled (using single GPU)")
        log("=" * 60)
    
    def _init_dask_cluster(self):
        """Initialize Dask LocalCUDACluster for multi-GPU processing."""
        global DASK_CLIENT, DASK_CLUSTER
        try:
            log(f"Initializing Dask cluster with {self.gpu_count} GPUs...")
            
            # Create cluster with silenced logging
            self.dask_cluster = LocalCUDACluster(
                n_workers=self.gpu_count,
                threads_per_worker=1,
                memory_limit='60GB',
                device_memory_limit='40GB',
                rmm_managed_memory=True,
                silence_logs=logging.CRITICAL,  # Silence worker logs
            )
            
            self.dask_client = Client(self.dask_cluster, set_as_default=False)
            DASK_CLIENT = self.dask_client
            DASK_CLUSTER = self.dask_cluster
            
            self.dask_client.wait_for_workers(self.gpu_count, timeout=30)
            
            self.multi_gpu_enabled = True
            log(f"  Dask cluster ready: {self.dask_client.dashboard_link}")
            
        except Exception as e:
            log(f"WARNING: Dask cluster init failed: {e}")
            log("  Falling back to single-GPU mode")
            self.multi_gpu_enabled = False
            self._cleanup_dask()
    
    def _cleanup_dask(self):
        """Clean up Dask resources."""
        global DASK_CLIENT, DASK_CLUSTER
        try:
            if self.dask_client:
                self.dask_client.close()
            if self.dask_cluster:
                self.dask_cluster.close()
        except:
            pass
        self.dask_client = None
        self.dask_cluster = None
        DASK_CLIENT = None
        DASK_CLUSTER = None
    
    def get_stable_files(self, run_dir: Path, pattern: str, min_age: int = 5) -> List[Path]:
        """Get files that haven't been modified recently."""
        now = time.time()
        stable_files = []
        
        for f in run_dir.glob(pattern):
            try:
                age = now - f.stat().st_mtime
                if age >= min_age:
                    stable_files.append(f)
            except:
                pass
        
        return sorted(stable_files)
    
    def read_run_data(self, run_dir: Path) -> Optional[cudf.DataFrame]:
        """Read data from a run directory."""
        max_files = self.config.max_files_per_run
        
        parquet_files = self.get_stable_files(run_dir, "worker_*.parquet")
        if parquet_files:
            return self._load_sampled_files(parquet_files, 'parquet', max_files)
        
        csv_files = self.get_stable_files(run_dir, "worker_*.csv")
        if csv_files:
            return self._load_sampled_files(csv_files, 'csv', max_files)
        
        bin_files = self.get_stable_files(run_dir, "worker_*.bin")
        if bin_files:
            return self._load_sampled_files(bin_files, 'binary', max_files)
        
        log(f"No stable data files found in {run_dir}")
        return None
    
    def _load_sampled_files(self, files: List[Path], file_type: str, max_files: int) -> cudf.DataFrame:
        """Load files with sampling if too many."""
        total_files = len(files)
        
        if total_files > max_files:
            step = total_files // max_files
            files = files[::step][:max_files]
            log(f"Sampling {len(files)} of {total_files:,} {file_type} files (every {step}th file)")
        else:
            log(f"Loading all {total_files} {file_type} files")
        
        if file_type == 'binary':
            return self._load_binary_files(files)
        
        if self.multi_gpu_enabled and file_type == 'parquet':
            return self._load_files_multi_gpu(files)
        else:
            return self._load_files_chunked(files, file_type)
    
    def _load_files_multi_gpu(self, files: List[Path]) -> Optional[cudf.DataFrame]:
        """Load parquet files using Dask-cuDF across multiple GPUs."""
        start = time.time()
        total_files = len(files)
        
        log(f"  Multi-GPU loading {total_files} files across {self.gpu_count} GPUs...")
        
        try:
            file_paths = [str(f) for f in files]
            
            ddf = dask_cudf.read_parquet(
                file_paths,
                split_row_groups=True,
            )
            
            num_partitions = ddf.npartitions
            log(f"  Created {num_partitions} partitions across {self.gpu_count} GPUs")
            
            ddf = ddf.persist()
            wait(ddf)
            
            log(f"  Merging partitions...")
            df = ddf.compute()
            
            elapsed = time.time() - start
            throughput = len(df) / elapsed if elapsed > 0 else 0
            log(f"  Complete: {len(df):,} records in {elapsed:.1f}s ({throughput/1e6:.1f}M rec/s) [multi-GPU]")
            
            return df
            
        except Exception as e:
            log(f"  WARNING: Multi-GPU load failed: {e}")
            log(f"  Shutting down Dask cluster to free GPU memory...")
            self._cleanup_dask()
            self.multi_gpu_enabled = False
            cp.get_default_memory_pool().free_all_blocks()
            log(f"  Falling back to single-GPU chunked loading...")
            return self._load_files_chunked(files, 'parquet')
    
    def _load_files_chunked(self, files: List[Path], file_type: str) -> Optional[cudf.DataFrame]:
        """Load and process files in chunks to avoid OOM."""
        start = time.time()
        batch_size = 5
        total_files = len(files)
        total_batches = (total_files + batch_size - 1) // batch_size
        running_records = 0
        output_parts = []
        
        log(f"  Loading {total_files} files in {total_batches} batches...")
        
        for i in range(0, total_files, batch_size):
            batch_files = files[i:i + batch_size]
            batch_num = i // batch_size + 1
            file_start = i + 1
            file_end = min(i + batch_size, total_files)
            
            try:
                if file_type == 'parquet':
                    dfs = [cudf.read_parquet(str(f)) for f in batch_files]
                else:
                    dfs = [cudf.read_csv(str(f)) for f in batch_files]
                
                batch_df = cudf.concat(dfs, ignore_index=True)
                batch_records = len(batch_df)
                running_records += batch_records
                
                del dfs
                output_parts.append(batch_df)
                
                log(f"  Batch {batch_num}/{total_batches}: files {file_start}-{file_end} -> {batch_records:,} records (total: {running_records:,})")
                
                cp.get_default_memory_pool().free_all_blocks()
                
            except Exception as e:
                log(f"  WARNING: Batch {batch_num} failed: {e}")
                cp.get_default_memory_pool().free_all_blocks()
                continue
        
        if not output_parts:
            return None
        
        if len(output_parts) > 1:
            log(f"  Merging {len(output_parts)} batches into single DataFrame...")
            while len(output_parts) > 1:
                new_parts = []
                for j in range(0, len(output_parts), 2):
                    if j + 1 < len(output_parts):
                        merged = cudf.concat([output_parts[j], output_parts[j+1]], ignore_index=True)
                        new_parts.append(merged)
                    else:
                        new_parts.append(output_parts[j])
                del output_parts
                cp.get_default_memory_pool().free_all_blocks()
                output_parts = new_parts
        
        df = output_parts[0]
        del output_parts
        cp.get_default_memory_pool().free_all_blocks()
        
        elapsed = time.time() - start
        throughput = len(df) / elapsed if elapsed > 0 else 0
        log(f"  Complete: {len(df):,} records in {elapsed:.1f}s ({throughput/1e6:.1f}M rec/s)")
        
        return df
    
    def _load_binary_files(self, files: List[Path]) -> cudf.DataFrame:
        """Load binary numpy files."""
        start = time.time()
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        
        log(f"  Loading {len(files)} binary files...")
        arrays = []
        running_records = 0
        for idx, f in enumerate(files, 1):
            try:
                data = np.fromfile(str(f), dtype=np.float32).reshape(-1, 31)
                running_records += len(data)
                arrays.append(data)
                if idx % 10 == 0 or idx == len(files):
                    log(f"  File {idx}/{len(files)}: {running_records:,} records loaded")
            except Exception as e:
                log(f"  WARNING: Failed to load {f.name}: {e}")
        
        if not arrays:
            return None
        
        combined = np.vstack(arrays)
        df = cudf.DataFrame(combined, columns=columns)
        
        elapsed = time.time() - start
        throughput = len(df) / elapsed if elapsed > 0 else 0
        log(f"  Complete: {len(df):,} records in {elapsed:.1f}s ({throughput/1e6:.1f}M rec/s)")
        return df
    
    def write_output(self, df: cudf.DataFrame, run_name: str):
        """Write processed data to parquet via CPU."""
        output_file = self.output_path / f"features_{run_name}.parquet"
        record_count = len(df)
        log(f"Writing {record_count:,} records to {output_file.name}...")
        
        transfer_start = time.time()
        log(f"  GPU -> CPU transfer...")
        pdf = df.to_pandas()
        transfer_time = time.time() - transfer_start
        log(f"  Transfer complete: {transfer_time:.1f}s")
        
        del df
        cp.get_default_memory_pool().free_all_blocks()
        
        write_start = time.time()
        log(f"  Writing parquet (snappy compression)...")
        pdf.to_parquet(str(output_file), compression='snappy', index=False)
        write_time = time.time() - write_start
        del pdf
        
        size_mb = output_file.stat().st_size / (1024*1024)
        throughput = size_mb / write_time if write_time > 0 else 0
        log(f"  Write complete: {size_mb:.1f} MB in {write_time:.1f}s ({throughput:.0f} MB/s)")
        
        pf = pq.ParquetFile(output_file)
        columns = pf.schema.names
        
        exclude = ['transaction_id', 'prep_timestamp', 'Class']
        feature_cols = [c for c in columns if c not in exclude]
        
        metadata = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "columns": columns,
            "feature_columns": feature_cols,
            "target_column": "Class",
            "record_count": pf.metadata.num_rows
        }
        
        meta_file = self.output_path / f"metadata_{run_name}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_run(self, run_dir: Path) -> bool:
        """Process a single run directory."""
        run_name = run_dir.name
        run_start = time.time()
        log("=" * 60)
        log(f"Processing: {run_name}")
        
        try:
            df = self.read_run_data(run_dir)
            if df is None or len(df) == 0:
                log(f"No data loaded from {run_name}")
                return False
            
            df = self.engineer.process(df, run_name)
            self.write_output(df, run_name)
            self.watcher.mark_processed(run_dir)
            
            cp.get_default_memory_pool().free_all_blocks()
            
            total_time = time.time() - run_start
            log(f"SUCCESS: {run_name} (total: {total_time:.1f}s)")
            log("=" * 60)
            return True
            
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            cp.get_default_memory_pool().free_all_blocks()
            return False
    
    def run(self):
        """Main run loop."""
        global STOP_FLAG
        
        if self.config.batch_mode:
            runs = self.watcher.get_new_runs(
                latest_only=self.config.latest_only,
                min_age_seconds=self.config.file_stable_seconds
            )
            log(f"Batch mode: {len(runs)} run(s) to process")
            for run_dir in runs:
                self.process_run(run_dir)
            log("Batch complete")
        else:
            log("Watching for new runs...")
            last_status_time = time.time()
            status_interval = 30
            runs_processed = 0
            
            while not STOP_FLAG:
                runs = self.watcher.get_new_runs(
                    latest_only=self.config.latest_only,
                    min_age_seconds=self.config.file_stable_seconds
                )
                
                if runs:
                    for run_dir in runs:
                        if STOP_FLAG:
                            break
                        if self.process_run(run_dir):
                            runs_processed += 1
                    last_status_time = time.time()
                else:
                    if time.time() - last_status_time >= status_interval:
                        processed_str = f" ({runs_processed} processed)" if runs_processed > 0 else ""
                        log(f"Waiting for new data in {self.input_path.name}...{processed_str}")
                        last_status_time = time.time()
                
                if not STOP_FLAG:
                    time.sleep(self.config.poll_interval)
            
            log("Stopped")


def main():
    log(f"Starting (PID: {os.getpid()})")
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = PrepConfig(
        input_dir=os.getenv('INPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'),
        output_dir=os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/prep-output'),
        poll_interval=int(os.getenv('POLL_INTERVAL', '5')),
        batch_mode=os.getenv('BATCH_MODE', 'false').lower() == 'true',
        max_files_per_run=int(os.getenv('MAX_FILES_PER_RUN', '50')),
        latest_only=os.getenv('LATEST_ONLY', 'true').lower() == 'true',
        file_stable_seconds=int(os.getenv('FILE_STABLE_SECONDS', '10')),
        use_multi_gpu=os.getenv('USE_MULTI_GPU', 'true').lower() == 'true'
    )
    
    service = DataPrepService(config)
    try:
        service.run()
    finally:
        service._cleanup_dask()
        log("Cleanup complete")


if __name__ == "__main__":
    main()