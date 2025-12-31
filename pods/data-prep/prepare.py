#!/usr/bin/env python3
"""
Pod 2: Data Prepare Service (Multi-GPU RAPIDS)
Feature engineering for credit card transaction fraud detection
"""

import os
import sys
import time
import json
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Set
from dataclasses import dataclass

# Suppress library logging before imports
for name in ['distributed', 'distributed.worker', 'distributed.scheduler', 'distributed.nanny', 'bokeh', 'tornado', 'asyncio']:
    import logging
    logging.getLogger(name).setLevel(logging.CRITICAL)

os.environ['DASK_DISTRIBUTED__LOGGING__DISTRIBUTED'] = 'critical'
os.environ['RAPIDS_NO_INITIALIZE'] = '1'

import cudf
import cupy as cp
import numpy as np
import pyarrow.parquet as pq

import dask
dask.config.set({
    'distributed.logging.distributed': 'critical',
    'distributed.scheduler.work-stealing': False,
    'distributed.scheduler.bandwidth': 100000000,
})
import dask_cudf
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster

STOP_FLAG = False
MAIN_PID = os.getpid()

# Category encoding map
CATEGORY_MAP = {
    'gas_transport': 0, 'grocery_pos': 1, 'misc_pos': 2, 'misc_net': 3,
    'shopping_net': 4, 'shopping_pos': 5, 'grocery_net': 6, 'entertainment': 7,
    'food_dining': 8, 'home': 9, 'kids_pets': 10, 'travel': 11,
    'health_fitness': 12, 'personal_care': 13
}

# State encoding (top states by population)
STATE_MAP = {
    'CA': 0, 'TX': 1, 'FL': 2, 'NY': 3, 'PA': 4, 'IL': 5, 'OH': 6, 'GA': 7,
    'NC': 8, 'MI': 9, 'NJ': 10, 'VA': 11, 'WA': 12, 'AZ': 13, 'MA': 14,
    'TN': 15, 'IN': 16, 'MO': 17, 'MD': 18, 'WI': 19, 'CO': 20, 'MN': 21,
    'SC': 22, 'AL': 23, 'LA': 24, 'KY': 25, 'OR': 26, 'OK': 27, 'CT': 28,
    'UT': 29, 'IA': 30, 'NV': 31, 'AR': 32, 'MS': 33, 'KS': 34, 'NM': 35,
    'NE': 36, 'ID': 37, 'WV': 38, 'HI': 39, 'NH': 40, 'ME': 41, 'MT': 42,
    'RI': 43, 'DE': 44, 'SD': 45, 'ND': 46, 'AK': 47, 'VT': 48, 'WY': 49
}


def log(msg):
    if os.getpid() == MAIN_PID:
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def signal_handler(signum, frame):
    global STOP_FLAG
    log("Received shutdown signal")
    STOP_FLAG = True


@dataclass
class Config:
    input_dir: str
    output_dir: str
    poll_interval: int = 5
    batch_mode: bool = False
    max_files: int = 50
    latest_only: bool = True
    file_stable_seconds: int = 10
    use_multi_gpu: bool = True


def engineer_features_partition(df):
    """Feature engineering on each GPU partition."""
    import cupy as cp
    
    # === Amount features ===
    if 'amt' in df.columns:
        df['amt_log'] = cp.log1p(df['amt'].values)
        amt_mean = float(df['amt'].mean())
        amt_std = float(df['amt'].std())
        if amt_std > 0.001:
            df['amt_scaled'] = (df['amt'] - amt_mean) / amt_std
        else:
            df['amt_scaled'] = 0.0
    
    # === Time features (from unix_time) ===
    if 'unix_time' in df.columns:
        # Hour of day (0-23)
        df['hour_of_day'] = ((df['unix_time'] % 86400) // 3600).astype('int8')
        # Day of week (0=Monday, 6=Sunday) - approximate
        df['day_of_week'] = ((df['unix_time'] // 86400) % 7).astype('int8')
        # Is weekend (Saturday=5, Sunday=6)
        df['is_weekend'] = ((df['day_of_week'] >= 5)).astype('int8')
        # Is night (10pm - 6am)
        df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] <= 6)).astype('int8')
    
    # === Geographic features ===
    if all(col in df.columns for col in ['lat', 'long', 'merch_lat', 'merch_long']):
        # Distance between customer and merchant (Haversine approximation in km)
        lat1 = cp.radians(df['lat'].values)
        lat2 = cp.radians(df['merch_lat'].values)
        dlat = lat2 - lat1
        dlon = cp.radians(df['merch_long'].values - df['long'].values)
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(cp.clip(a, 0, 1)))
        df['distance_km'] = (6371 * c).astype('float32')  # Earth radius in km
    
    # === Categorical encoding ===
    if 'category' in df.columns:
        # Map category to numeric
        df['category_encoded'] = df['category'].map(CATEGORY_MAP).fillna(-1).astype('int8')
    
    if 'state' in df.columns:
        df['state_encoded'] = df['state'].map(STATE_MAP).fillna(-1).astype('int8')
    
    if 'gender' in df.columns:
        df['gender_encoded'] = (df['gender'] == 'M').astype('int8')
    
    # === City population features ===
    if 'city_pop' in df.columns:
        df['city_pop_log'] = cp.log1p(df['city_pop'].values).astype('float32')
    
    # === ZIP code features ===
    if 'zip' in df.columns:
        # Extract region from ZIP (first digit)
        df['zip_region'] = (df['zip'] // 10000).astype('int8')
    
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
                self.gpu_names.append(f"GPU{i}:{props['name'].decode()[:10]}")
            log(f"GPUs: {', '.join(self.gpu_names)}")
        except:
            self.gpu_count = 1
            self.gpu_names = ["GPU0"]
        
        if config.use_multi_gpu and self.gpu_count > 1:
            self.multi_gpu_available = True
            log(f"  Multi-GPU available (will init Dask on first job)")
        else:
            self.multi_gpu_available = False
        
        log("=" * 60)
        log("Pod 2: Data Prep Service (RAPIDS)")
        log("=" * 60)
        log(f"Input:  {self.input_path}")
        log(f"Output: {self.output_path}")
        log(f"Mode:   {'multi-GPU available' if self.multi_gpu_available else 'single GPU'}")
        log("=" * 60)
    
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
                json.dump({'processed': list(self.processed)}, f)
        except:
            pass
    
    def _init_dask(self):
        try:
            log(f"Initializing Dask with {self.gpu_count} GPUs...")
            self.dask_cluster = LocalCUDACluster(
                n_workers=self.gpu_count, threads_per_worker=1,
                memory_limit='60GB', device_memory_limit='40GB',
                rmm_managed_memory=True, silence_logs=50)
            self.dask_client = Client(self.dask_cluster, set_as_default=True)
            self.dask_client.wait_for_workers(self.gpu_count, timeout=30)
            self.multi_gpu = True
            log(f"  Dask ready: {self.dask_client.dashboard_link}")
        except Exception as e:
            log(f"  Dask failed: {e}, using single GPU")
            self._cleanup_dask()
    
    def _cleanup_dask(self):
        try:
            if self.dask_client: self.dask_client.close()
            if self.dask_cluster: self.dask_cluster.close()
        except:
            pass
        self.dask_client = self.dask_cluster = None
        self.multi_gpu = False
    
    def get_new_runs(self) -> List[Path]:
        if not self.input_path.exists():
            return []
        now = time.time()
        runs = []
        for entry in self.input_path.iterdir():
            if entry.is_dir() and entry.name.startswith("run_") and entry.name not in self.processed:
                try:
                    if now - entry.stat().st_mtime < self.config.file_stable_seconds:
                        continue
                except:
                    continue
                if list(entry.glob("worker_*.parquet"))[:1] or list(entry.glob("worker_*.csv"))[:1]:
                    runs.append(entry)
        runs = sorted(runs, key=lambda x: x.name)
        return [runs[-1]] if self.config.latest_only and runs else runs
    
    def process_multi_gpu(self, run_dir: Path, run_name: str) -> bool:
        """Process using Dask across multiple GPUs."""
        now = time.time()
        files = sorted([f for f in run_dir.glob("worker_*.parquet") if now - f.stat().st_mtime >= 5])
        
        if not files:
            log("No stable parquet files found")
            return False
        
        if len(files) > self.config.max_files:
            step = len(files) // self.config.max_files
            files = files[::step][:self.config.max_files]
            log(f"Sampling {len(files)} files (every {step}th)")
        else:
            log(f"Loading {len(files)} files")
        
        file_paths = [str(f) for f in files]
        
        try:
            start = time.time()
            log(f"  [{', '.join(self.gpu_names)}] Loading data...")
            
            ddf = dask_cudf.read_parquet(file_paths, split_row_groups=True)
            
            n_partitions = self.gpu_count * 4
            ddf = ddf.repartition(npartitions=n_partitions)
            ddf = ddf.persist()
            wait(ddf)
            
            total_rows = len(ddf)
            load_time = time.time() - start
            log(f"  [{', '.join(self.gpu_names)}] Loaded {total_rows:,} records in {load_time:.1f}s")
            
            # Feature engineering
            start = time.time()
            log(f"  [{', '.join(self.gpu_names)}] Feature engineering...")
            
            ddf['prep_timestamp'] = float(time.time())
            
            # Define output meta with new features
            meta = ddf._meta.copy()
            new_cols = ['amt_log', 'amt_scaled', 'hour_of_day', 'day_of_week', 
                       'is_weekend', 'is_night', 'distance_km', 'category_encoded',
                       'state_encoded', 'gender_encoded', 'city_pop_log', 'zip_region']
            for col in new_cols:
                if col in ['hour_of_day', 'day_of_week', 'is_weekend', 'is_night', 
                          'category_encoded', 'state_encoded', 'gender_encoded', 'zip_region']:
                    meta[col] = np.int8(0)
                else:
                    meta[col] = np.float32(0)
            
            ddf = ddf.map_partitions(engineer_features_partition, meta=meta)
            ddf = ddf.persist()
            wait(ddf)
            
            eng_time = time.time() - start
            log(f"  [{', '.join(self.gpu_names)}] Features added in {eng_time:.1f}s")
            
            # Merge partitions
            start = time.time()
            log(f"  Merging partitions...")
            df = ddf.compute()
            
            # Add transaction IDs if not present
            if 'transaction_id' not in df.columns:
                df['transaction_id'] = cp.arange(len(df), dtype=cp.int64)
            
            merge_time = time.time() - start
            log(f"  Merged {len(df):,} records in {merge_time:.1f}s")
            
            self._write_output(df, run_name)
            
            del df, ddf
            cp.get_default_memory_pool().free_all_blocks()
            
            return True
            
        except Exception as e:
            log(f"  Multi-GPU failed: {e}")
            log(f"  Falling back to single GPU...")
            cp.get_default_memory_pool().free_all_blocks()
            return self.process_single_gpu(run_dir, run_name)
    
    def process_single_gpu(self, run_dir: Path, run_name: str) -> bool:
        """Fallback single-GPU processing."""
        now = time.time()
        files = sorted([f for f in run_dir.glob("worker_*.parquet") if now - f.stat().st_mtime >= 5])
        
        if not files:
            files = sorted([f for f in run_dir.glob("worker_*.csv") if now - f.stat().st_mtime >= 5])
            file_type = 'csv'
        else:
            file_type = 'parquet'
        
        if not files:
            return False
        
        if len(files) > self.config.max_files:
            step = len(files) // self.config.max_files
            files = files[::step][:self.config.max_files]
            log(f"Sampling {len(files)} files (every {step}th)")
        else:
            log(f"Loading {len(files)} {file_type} files")
        
        gpu_id = cp.cuda.runtime.getDevice()
        gpu_name = self.gpu_names[gpu_id] if gpu_id < len(self.gpu_names) else f"GPU{gpu_id}"
        
        start = time.time()
        parts = []
        batch_size = 5
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            try:
                if file_type == 'parquet':
                    dfs = [cudf.read_parquet(str(f)) for f in batch]
                else:
                    dfs = [cudf.read_csv(str(f)) for f in batch]
                parts.append(cudf.concat(dfs, ignore_index=True))
                del dfs
                log(f"  [{gpu_name}] Batch {i//batch_size+1}: {len(parts[-1]):,} records")
                cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                log(f"  [{gpu_name}] Batch failed: {e}")
        
        if not parts:
            return False
        
        # Merge
        while len(parts) > 1:
            new_parts = []
            for j in range(0, len(parts), 2):
                if j+1 < len(parts):
                    new_parts.append(cudf.concat([parts[j], parts[j+1]], ignore_index=True))
                else:
                    new_parts.append(parts[j])
            del parts
            cp.get_default_memory_pool().free_all_blocks()
            parts = new_parts
        
        df = parts[0]
        del parts
        log(f"  [{gpu_name}] Loaded {len(df):,} records in {time.time()-start:.1f}s")
        
        # Feature engineering
        start = time.time()
        if 'transaction_id' not in df.columns:
            df['transaction_id'] = cp.arange(len(df), dtype=cp.int64)
        df['prep_timestamp'] = float(time.time())
        df = engineer_features_partition(df)
        log(f"  [{gpu_name}] Features added in {time.time()-start:.1f}s")
        
        self._write_output(df, run_name)
        
        del df
        cp.get_default_memory_pool().free_all_blocks()
        return True
    
    def _write_output(self, df: cudf.DataFrame, run_name: str):
        output_file = self.output_path / f"features_{run_name}.parquet"
        log(f"Writing {len(df):,} records...")
        
        start = time.time()
        pdf = df.to_pandas()
        log(f"  GPU->CPU: {time.time()-start:.1f}s")
        del df
        cp.get_default_memory_pool().free_all_blocks()
        
        start = time.time()
        pdf.to_parquet(str(output_file), compression='snappy', index=False)
        size_mb = output_file.stat().st_size / (1024**2)
        log(f"  Write: {size_mb:.0f} MB in {time.time()-start:.1f}s")
        del pdf
        
        # Metadata
        pf = pq.ParquetFile(output_file)
        
        # Define feature columns (exclude IDs, timestamps, raw categoricals)
        exclude = ['transaction_id', 'prep_timestamp', 'is_fraud', 'trans_num', 
                   'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
                   'street', 'city', 'job', 'dob', 'category', 'state', 'gender']
        feature_columns = [c for c in pf.schema.names if c not in exclude]
        
        meta = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "columns": pf.schema.names,
            "record_count": pf.metadata.num_rows,
            "feature_columns": feature_columns,
            "target_column": "is_fraud"
        }
        with open(self.output_path / f"metadata_{run_name}.json", 'w') as f:
            json.dump(meta, f, indent=2)
    
    def process_run(self, run_dir: Path) -> bool:
        run_name = run_dir.name
        start = time.time()
        log("=" * 60)
        log(f"Processing: {run_name}")
        
        try:
            if self.multi_gpu_available and not self.multi_gpu:
                self._init_dask()
            
            if self.multi_gpu:
                success = self.process_multi_gpu(run_dir, run_name)
            else:
                success = self.process_single_gpu(run_dir, run_name)
            
            if success:
                self.processed.add(run_name)
                self._save_state()
                log(f"SUCCESS: {run_name} ({time.time()-start:.1f}s)")
                log("=" * 60)
            return success
            
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            cp.get_default_memory_pool().free_all_blocks()
            return False
    
    def run(self):
        global STOP_FLAG
        
        if self.config.batch_mode:
            for run_dir in self.get_new_runs():
                self.process_run(run_dir)
            log("Batch complete")
        else:
            log("Watching for new runs...")
            last_status = time.time()
            processed_count = 0
            idle_poll = 10
            
            while not STOP_FLAG:
                runs = self.get_new_runs()
                if runs:
                    for run_dir in runs:
                        if STOP_FLAG: break
                        if self.process_run(run_dir):
                            processed_count += 1
                    last_status = time.time()
                elif time.time() - last_status >= 60:
                    log(f"Waiting for data... ({processed_count} processed)")
                    last_status = time.time()
                
                if not STOP_FLAG:
                    time.sleep(idle_poll)
            
            log("Stopped")


def main():
    log(f"Starting (PID: {MAIN_PID})")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = Config(
        input_dir=os.getenv('INPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'),
        output_dir=os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/prep-output'),
        poll_interval=int(os.getenv('POLL_INTERVAL', '5')),
        batch_mode=os.getenv('BATCH_MODE', 'false').lower() == 'true',
        max_files=int(os.getenv('MAX_FILES_PER_RUN', '50')),
        latest_only=os.getenv('LATEST_ONLY', 'true').lower() == 'true',
        file_stable_seconds=int(os.getenv('FILE_STABLE_SECONDS', '10')),
        use_multi_gpu=os.getenv('USE_MULTI_GPU', 'true').lower() == 'true')
    
    service = DataPrepService(config)
    try:
        service.run()
    finally:
        service._cleanup_dask()
        log("Cleanup complete")


if __name__ == "__main__":
    main()