#!/usr/bin/env python3
"""
Pod 2: Feature Engineering
GPU-accelerated data preparation using RAPIDS cuDF/Dask.
Supports multi-GPU processing for large datasets.

DEMO MODE: Runs CPU processing first, then GPU, to demonstrate acceleration.
"""

import os
import sys
import time
import json
import signal
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple
from dataclasses import dataclass

# Suppress Dask logging
for name in ['distributed', 'distributed.worker', 'distributed.scheduler', 
             'distributed.nanny', 'bokeh', 'tornado', 'asyncio']:
    import logging
    logging.getLogger(name).setLevel(logging.CRITICAL)

os.environ['DASK_DISTRIBUTED__LOGGING__DISTRIBUTED'] = 'critical'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'

# CPU imports
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# GPU imports
import cudf
import cupy as cp

import dask
dask.config.set({
    'distributed.logging.distributed': 'critical',
    'distributed.scheduler.work-stealing': False,
})
import dask_cudf
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

STOP_FLAG = False
MAIN_PID = os.getpid()


def log(msg):
    if os.getpid() == MAIN_PID:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def signal_handler(signum, frame):
    global STOP_FLAG
    log("Shutdown signal received")
    STOP_FLAG = True


def free_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


@dataclass
class Config:
    input_dir: str
    output_dir: str
    batch_mode: bool = True
    max_files: int = 100
    latest_only: bool = True
    use_multi_gpu: bool = True


# Columns to drop (strings not needed for ML)
STRING_COLUMNS_TO_DROP = [
    'merchant', 'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num',
    'trans_date_trans_time', 'category', 'state', 'gender'
]


# =============================================================================
# CPU Feature Engineering (Pandas/NumPy)
# =============================================================================
def engineer_features_cpu(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features using CPU (pandas/numpy)."""
    # Amount features
    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'].values)
        mean, std = df['amt'].mean(), df['amt'].std()
        if std > 0.001:
            df['amt_scaled'] = (df['amt'] - mean) / std
    
    # Time features
    if 'unix_time' in df.columns:
        hours = (df['unix_time'] / 3600) % 24
        df['hour_of_day'] = hours
        df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night'] = ((hours >= 22) | (hours <= 6)).astype('int8')
    
    # Distance between customer and merchant
    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        dlat = (df['merch_lat'] - df['lat']) * 111.0
        dlon = (df['merch_long'] - df['long']) * 85.0
        df['distance_km'] = np.sqrt(dlat**2 + dlon**2)
    
    # Categorical encoding
    if 'category' in df.columns:
        cats = ['gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
                'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
                'kids_pets', 'travel', 'health_fitness', 'personal_care']
        cat_map = {c: i for i, c in enumerate(cats)}
        df['category_encoded'] = df['category'].map(cat_map).fillna(-1).astype('int8')
    
    if 'state' in df.columns:
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                  'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
                  'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
                  'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
                  'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
        state_map = {s: i for i, s in enumerate(states)}
        df['state_encoded'] = df['state'].map(state_map).fillna(-1).astype('int8')
    
    if 'gender' in df.columns:
        df['gender_encoded'] = (df['gender'] == 'M').astype('int8')
    
    # Population features
    if 'city_pop' in df.columns:
        df['city_pop_log'] = np.log1p(df['city_pop'].values)
    
    if 'zip' in df.columns:
        df['zip_region'] = (df['zip'] / 10000).astype('int8')
    
    # Drop string columns
    cols_to_drop = [c for c in STRING_COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


# =============================================================================
# GPU Feature Engineering (RAPIDS cuDF)
# =============================================================================
def engineer_features_gpu(df):
    """Add engineered features using GPU (cuDF/cupy)."""
    # Amount features
    if 'amt' in df.columns:
        df['amt_log'] = cp.log1p(df['amt'].values)
        mean, std = float(df['amt'].mean()), float(df['amt'].std())
        if std > 0.001:
            df['amt_scaled'] = (df['amt'] - mean) / std
    
    # Time features
    if 'unix_time' in df.columns:
        hours = (df['unix_time'] / 3600) % 24
        df['hour_of_day'] = hours
        df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype('int8')
        df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
        df['is_night'] = ((hours >= 22) | (hours <= 6)).astype('int8')
    
    # Distance between customer and merchant
    if all(c in df.columns for c in ['lat', 'long', 'merch_lat', 'merch_long']):
        dlat = (df['merch_lat'] - df['lat']) * 111.0
        dlon = (df['merch_long'] - df['long']) * 85.0
        df['distance_km'] = cp.sqrt(dlat.values**2 + dlon.values**2)
    
    # Categorical encoding
    if 'category' in df.columns:
        cats = ['gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
                'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
                'kids_pets', 'travel', 'health_fitness', 'personal_care']
        cat_map = {c: i for i, c in enumerate(cats)}
        df['category_encoded'] = df['category'].map(cat_map).fillna(-1).astype('int8')
    
    if 'state' in df.columns:
        states = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
                  'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
                  'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
                  'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
                  'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
        state_map = {s: i for i, s in enumerate(states)}
        df['state_encoded'] = df['state'].map(state_map).fillna(-1).astype('int8')
    
    if 'gender' in df.columns:
        df['gender_encoded'] = (df['gender'] == 'M').astype('int8')
    
    # Population features
    if 'city_pop' in df.columns:
        df['city_pop_log'] = cp.log1p(df['city_pop'].values)
    
    if 'zip' in df.columns:
        df['zip_region'] = (df['zip'] / 10000).astype('int8')
    
    # Drop string columns
    cols_to_drop = [c for c in STRING_COLUMNS_TO_DROP if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    return df


class DataPrepService:
    def __init__(self, config: Config):
        self.config = config
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.state_file = self.output_path / ".prep_state.json"
        self.processed: Set[str] = self._load_state()
        
        self.dask_client = None
        self.dask_cluster = None
        self.multi_gpu = False
        
        # GPU info
        try:
            self.gpu_count = cp.cuda.runtime.getDeviceCount()
            self.gpu_names = []
            for i in range(self.gpu_count):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props['name'].decode()[:12]
                self.gpu_names.append(f"GPU{i}:{name}")
        except:
            self.gpu_count = 1
            self.gpu_names = ["GPU0"]
        
        log("=" * 70)
        log("Pod 2: Feature Engineering - CPU vs GPU Comparison")
        log("=" * 70)
        log(f"Input:  {self.input_path}")
        log(f"Output: {self.output_path}")
        log(f"GPUs:   {', '.join(self.gpu_names)}")
        log(f"Max files per run: {config.max_files}")
        log("=" * 70)
    
    def _load_state(self) -> Set[str]:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    return set(json.load(f).get('processed', []))
            except:
                pass
        return set()
    
    def _save_state(self):
        with open(self.state_file, 'w') as f:
            json.dump({'processed': list(self.processed)}, f)
    
    def _init_dask(self):
        """Initialize Dask cluster for multi-GPU processing."""
        try:
            log(f"Initializing Dask with {self.gpu_count} GPUs...")
            self.dask_cluster = LocalCUDACluster(
                n_workers=self.gpu_count,
                threads_per_worker=1,
                memory_limit='60GB',
                device_memory_limit='40GB',
                rmm_managed_memory=True,
                silence_logs=50
            )
            self.dask_client = Client(self.dask_cluster, set_as_default=True)
            self.dask_client.wait_for_workers(self.gpu_count, timeout=30)
            self.multi_gpu = True
            log(f"  Dask ready: {self.dask_client.dashboard_link}")
        except Exception as e:
            log(f"  Dask init failed: {e}")
            self._cleanup_dask()
    
    def _cleanup_dask(self):
        try:
            if self.dask_client:
                self.dask_client.close()
            if self.dask_cluster:
                self.dask_cluster.close()
        except:
            pass
        self.dask_client = self.dask_cluster = None
        self.multi_gpu = False
    
    def get_pending_runs(self) -> List[Path]:
        """Find unprocessed data runs."""
        if not self.input_path.exists():
            return []
        
        runs = []
        for entry in self.input_path.iterdir():
            if entry.is_dir() and entry.name.startswith("run_"):
                if entry.name not in self.processed:
                    if list(entry.glob("worker_*.parquet"))[:1]:
                        runs.append(entry)
        
        runs = sorted(runs, key=lambda x: x.name)
        return [runs[-1]] if self.config.latest_only and runs else runs
    
    def _validate_parquet(self, filepath: Path) -> bool:
        """Check if parquet file is valid."""
        try:
            size = filepath.stat().st_size
            if size < 100:
                return False
            with open(filepath, 'rb') as f:
                magic_start = f.read(4)
                if magic_start != b'PAR1':
                    return False
                f.seek(-4, 2)
                magic_end = f.read(4)
                if magic_end != b'PAR1':
                    return False
            return True
        except:
            return False
    
    def _get_valid_files(self, run_dir: Path) -> List[Path]:
        """Get list of valid parquet files."""
        all_files = sorted(run_dir.glob("worker_*.parquet"))
        files = [f for f in all_files if self._validate_parquet(f)]
        
        # Sample if too many
        if len(files) > self.config.max_files:
            step = len(files) // self.config.max_files
            files = files[::step][:self.config.max_files]
        
        return files
    
    # =========================================================================
    # CPU Processing Path
    # =========================================================================
    def _process_cpu(self, files: List[Path]) -> Tuple[pd.DataFrame, float, float]:
        """Process data using CPU (pandas). Returns (df, load_time, eng_time)."""
        log("")
        log("=" * 70)
        log("PHASE 1: CPU Processing (Pandas/NumPy)")
        log("=" * 70)
        
        # Load data with pandas
        load_start = time.time()
        parts = []
        for f in files:
            try:
                parts.append(pd.read_parquet(str(f)))
            except Exception as e:
                log(f"  Error reading {f.name}: {e}")
                continue
        
        if not parts:
            return None, 0, 0
        
        df = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        
        load_time = time.time() - load_start
        log(f"  Loaded {len(df):,} records in {load_time:.2f}s [CPU]")
        
        # Feature engineering
        eng_start = time.time()
        df = engineer_features_cpu(df)
        eng_time = time.time() - eng_start
        log(f"  Features engineered in {eng_time:.2f}s [CPU]")
        
        total_time = load_time + eng_time
        log(f"  CPU TOTAL: {total_time:.2f}s")
        
        return df, load_time, eng_time
    
    # =========================================================================
    # GPU Processing Path
    # =========================================================================
    def _process_gpu(self, files: List[Path]) -> Tuple[cudf.DataFrame, float, float]:
        """Process data using GPU (cuDF). Returns (df, load_time, eng_time)."""
        log("")
        log("=" * 70)
        log("PHASE 2: GPU Processing (RAPIDS cuDF)")
        log("=" * 70)
        
        # Initialize Dask if multi-GPU
        if self.config.use_multi_gpu and self.gpu_count > 1 and not self.multi_gpu:
            self._init_dask()
        
        load_start = time.time()
        
        if self.multi_gpu:
            # Multi-GPU path with Dask
            ddf = dask_cudf.read_parquet([str(f) for f in files], split_row_groups=True)
            ddf = ddf.repartition(npartitions=self.gpu_count * 4).persist()
            wait(ddf)
            total_rows = len(ddf)
            load_time = time.time() - load_start
            log(f"  Loaded {total_rows:,} records in {load_time:.2f}s [multi-GPU]")
            
            # Feature engineering
            eng_start = time.time()
            
            # Build meta
            meta = ddf._meta.copy()
            string_cols_in_meta = [c for c in STRING_COLUMNS_TO_DROP if c in meta.columns]
            if string_cols_in_meta:
                meta = meta.drop(columns=string_cols_in_meta)
            for col in ['amt_log', 'amt_scaled', 'hour_of_day', 'day_of_week', 
                       'is_weekend', 'is_night', 'distance_km', 'category_encoded',
                       'state_encoded', 'gender_encoded', 'city_pop_log', 'zip_region']:
                meta[col] = np.float32(0) if 'encoded' not in col else np.int8(0)
            
            ddf = ddf.map_partitions(engineer_features_gpu, meta=meta).persist()
            wait(ddf)
            eng_time = time.time() - eng_start
            log(f"  Features engineered in {eng_time:.2f}s [multi-GPU]")
            
            df = ddf.compute()
            del ddf
            free_gpu_memory()
        else:
            # Single GPU path
            batch_size = 25
            parts = []
            
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i+batch_size]
                batch_parts = []
                for f in batch_files:
                    try:
                        batch_parts.append(cudf.read_parquet(str(f)))
                    except Exception as e:
                        log(f"  Error reading {f.name}: {e}")
                        continue
                
                if batch_parts:
                    batch_df = cudf.concat(batch_parts, ignore_index=True)
                    parts.append(batch_df)
                    del batch_parts
                    free_gpu_memory()
            
            if not parts:
                return None, 0, 0
            
            df = cudf.concat(parts, ignore_index=True)
            del parts
            free_gpu_memory()
            
            load_time = time.time() - load_start
            log(f"  Loaded {len(df):,} records in {load_time:.2f}s [GPU]")
            
            # Feature engineering
            eng_start = time.time()
            df = engineer_features_gpu(df)
            eng_time = time.time() - eng_start
            log(f"  Features engineered in {eng_time:.2f}s [GPU]")
        
        total_time = load_time + eng_time
        log(f"  GPU TOTAL: {total_time:.2f}s")
        
        return df, load_time, eng_time
    
    def process_run(self, run_dir: Path) -> bool:
        """Process a single data run - CPU first, then GPU."""
        run_name = run_dir.name
        
        log("")
        log("#" * 70)
        log(f"# Processing: {run_name}")
        log("#" * 70)
        
        # Get valid files
        files = self._get_valid_files(run_dir)
        if not files:
            log("  No valid parquet files found")
            return False
        
        log(f"Processing {len(files)} parquet files")
        
        # =================================================================
        # PHASE 1: CPU Processing
        # =================================================================
        cpu_df, cpu_load_time, cpu_eng_time = self._process_cpu(files)
        cpu_total = cpu_load_time + cpu_eng_time
        cpu_records = len(cpu_df) if cpu_df is not None else 0
        
        # Clean up CPU dataframe (we won't save it, just timing)
        del cpu_df
        gc.collect()
        
        # =================================================================
        # PHASE 2: GPU Processing
        # =================================================================
        gpu_df, gpu_load_time, gpu_eng_time = self._process_gpu(files)
        gpu_total = gpu_load_time + gpu_eng_time
        
        if gpu_df is None:
            log("GPU processing failed")
            return False
        
        # =================================================================
        # Results Comparison
        # =================================================================
        log("")
        log("=" * 70)
        log("PERFORMANCE COMPARISON")
        log("=" * 70)
        log(f"  Records processed: {cpu_records:,}")
        log("")
        log(f"  {'Stage':<20} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10}")
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        
        load_speedup = cpu_load_time / gpu_load_time if gpu_load_time > 0 else 0
        eng_speedup = cpu_eng_time / gpu_eng_time if gpu_eng_time > 0 else 0
        total_speedup = cpu_total / gpu_total if gpu_total > 0 else 0
        
        log(f"  {'Data Loading':<20} {cpu_load_time:<12.2f} {gpu_load_time:<12.2f} {load_speedup:<10.1f}x")
        log(f"  {'Feature Engineering':<20} {cpu_eng_time:<12.2f} {gpu_eng_time:<12.2f} {eng_speedup:<10.1f}x")
        log(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        log(f"  {'TOTAL':<20} {cpu_total:<12.2f} {gpu_total:<12.2f} {total_speedup:<10.1f}x")
        log("")
        log(f"  🚀 GPU is {total_speedup:.1f}x faster than CPU!")
        log("=" * 70)
        
        # =================================================================
        # Save GPU results (the actual output)
        # =================================================================
        log("")
        log("Saving GPU-processed features...")
        
        # Add transaction IDs
        gpu_df['transaction_id'] = cp.arange(len(gpu_df), dtype=cp.int64)
        
        # Write output
        output_file = self.output_path / f"features_{run_name}.parquet"
        record_count = len(gpu_df)
        
        write_start = time.time()
        pdf = gpu_df.to_pandas()
        del gpu_df
        free_gpu_memory()
        
        pdf.to_parquet(str(output_file), compression='snappy', index=False)
        size_mb = output_file.stat().st_size / (1024**2)
        log(f"  Wrote {size_mb:.0f} MB in {time.time()-write_start:.1f}s")
        del pdf
        gc.collect()
        
        # Save metadata with timing info
        meta = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "record_count": record_count,
            "file_size_mb": size_mb,
            "performance": {
                "cpu_load_seconds": cpu_load_time,
                "cpu_engineering_seconds": cpu_eng_time,
                "cpu_total_seconds": cpu_total,
                "gpu_load_seconds": gpu_load_time,
                "gpu_engineering_seconds": gpu_eng_time,
                "gpu_total_seconds": gpu_total,
                "speedup_load": load_speedup,
                "speedup_engineering": eng_speedup,
                "speedup_total": total_speedup,
            }
        }
        with open(self.output_path / f"metadata_{run_name}.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        # Mark complete
        self.processed.add(run_name)
        self._save_state()
        
        log(f"SUCCESS: {run_name}")
        return True
    
    def run(self):
        """Main execution loop."""
        runs = self.get_pending_runs()
        
        if not runs:
            log("No pending runs found")
            return
        
        for run_dir in runs:
            if STOP_FLAG:
                break
            self.process_run(run_dir)
            free_gpu_memory()
        
        log("")
        log("=" * 70)
        log("All processing complete")
        log("=" * 70)


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = Config(
        input_dir=os.getenv('INPUT_DIR', '/data/input'),
        output_dir=os.getenv('OUTPUT_DIR', '/data/output'),
        batch_mode=os.getenv('BATCH_MODE', 'true').lower() == 'true',
        max_files=int(os.getenv('MAX_FILES_PER_RUN', '100')),
        latest_only=os.getenv('LATEST_ONLY', 'true').lower() == 'true',
        use_multi_gpu=os.getenv('USE_MULTI_GPU', 'true').lower() == 'true',
    )
    
    service = DataPrepService(config)
    try:
        service.run()
    finally:
        service._cleanup_dask()
        free_gpu_memory()


if __name__ == "__main__":
    main()