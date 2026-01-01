#!/usr/bin/env python3
"""
Pod 2: Feature Engineering
GPU-accelerated data preparation using RAPIDS cuDF/Dask.
Supports multi-GPU processing for large datasets.
"""

import os
import sys
import time
import json
import signal
import gc
from pathlib import Path
from datetime import datetime
from typing import List, Set
from dataclasses import dataclass

# Suppress Dask logging
for name in ['distributed', 'distributed.worker', 'distributed.scheduler', 
             'distributed.nanny', 'bokeh', 'tornado', 'asyncio']:
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


# Columns to drop (strings not needed for ML, cause cuDF size limit issues)
# Note: category, state, gender are encoded first, then dropped
STRING_COLUMNS_TO_DROP = [
    'merchant', 'first', 'last', 'street', 'city', 'job', 'dob', 'trans_num',
    'trans_date_trans_time', 'category', 'state', 'gender'
]


def engineer_features(df):
    """Add engineered features to dataframe."""
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
    
    # Drop string columns at the end to avoid cuDF 2GB limit during concat
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
        
        log("=" * 60)
        log("Pod 2: Feature Engineering (RAPIDS)")
        log("=" * 60)
        log(f"Input:  {self.input_path}")
        log(f"Output: {self.output_path}")
        log(f"GPUs:   {', '.join(self.gpu_names)}")
        log(f"Max files per run: {config.max_files}")
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
        """Check if parquet file is valid (has proper footer)."""
        try:
            # Quick validation: check file size and magic bytes
            size = filepath.stat().st_size
            if size < 100:
                log(f"  Skipping {filepath.name}: too small ({size} bytes)")
                return False
            with open(filepath, 'rb') as f:
                # Check parquet magic bytes at start
                magic_start = f.read(4)
                if magic_start != b'PAR1':
                    log(f"  Skipping {filepath.name}: invalid header")
                    return False
                # Check parquet magic bytes at end of file
                f.seek(-4, 2)
                magic_end = f.read(4)
                if magic_end != b'PAR1':
                    log(f"  Skipping {filepath.name}: invalid footer (corrupted)")
                    return False
            return True
        except Exception as e:
            log(f"  Skipping {filepath.name}: {e}")
            return False
    
    def process_run(self, run_dir: Path) -> bool:
        """Process a single data run."""
        run_name = run_dir.name
        start = time.time()
        
        log("-" * 60)
        log(f"Processing: {run_name}")
        
        # Find parquet files
        all_files = sorted(run_dir.glob("worker_*.parquet"))
        if not all_files:
            log("  No parquet files found")
            return False
        
        log(f"  Found {len(all_files)} parquet files, validating...")
        
        # Validate files (skip corrupted)
        files = []
        corrupted_count = 0
        for f in all_files:
            if self._validate_parquet(f):
                files.append(f)
            else:
                corrupted_count += 1
        
        if corrupted_count > 0:
            log(f"  Skipped {corrupted_count} corrupted/invalid files")
        
        if not files:
            log("  No valid parquet files to process")
            return False
        
        log(f"  {len(files)} valid files")
        
        # Sample if too many files (memory management)
        if len(files) > self.config.max_files:
            step = len(files) // self.config.max_files
            files = files[::step][:self.config.max_files]
            log(f"  Sampling {len(files)} files to fit in memory")
        
        try:
            # Initialize Dask if needed
            if self.config.use_multi_gpu and self.gpu_count > 1 and not self.multi_gpu:
                self._init_dask()
            
            # Load data
            load_start = time.time()
            if self.multi_gpu:
                ddf = dask_cudf.read_parquet([str(f) for f in files], split_row_groups=True)
                ddf = ddf.repartition(npartitions=self.gpu_count * 4).persist()
                wait(ddf)
                total_rows = len(ddf)
                log(f"  Loaded {total_rows:,} records in {time.time()-load_start:.1f}s [multi-GPU]")
                
                # Feature engineering
                eng_start = time.time()
                
                # Build meta that reflects: original cols - dropped strings + new features
                meta = ddf._meta.copy()
                # Drop string columns from meta
                string_cols_in_meta = [c for c in STRING_COLUMNS_TO_DROP if c in meta.columns]
                if string_cols_in_meta:
                    meta = meta.drop(columns=string_cols_in_meta)
                # Add new feature columns
                for col in ['amt_log', 'amt_scaled', 'hour_of_day', 'day_of_week', 
                           'is_weekend', 'is_night', 'distance_km', 'category_encoded',
                           'state_encoded', 'gender_encoded', 'city_pop_log', 'zip_region']:
                    meta[col] = np.float32(0) if 'encoded' not in col else np.int8(0)
                
                ddf = ddf.map_partitions(engineer_features, meta=meta).persist()
                wait(ddf)
                log(f"  Features added in {time.time()-eng_start:.1f}s")
                
                # Collect to single DataFrame (strings already dropped in engineer_features)
                df = ddf.compute()
                del ddf
                free_gpu_memory()
            else:
                # Single GPU loading - process in batches to manage memory
                batch_size = 25  # Process 25 files at a time
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
                    log("  No data loaded successfully")
                    return False
                
                df = cudf.concat(parts, ignore_index=True)
                del parts
                free_gpu_memory()
                
                log(f"  Loaded {len(df):,} records in {time.time()-load_start:.1f}s")
                
                # Feature engineering
                eng_start = time.time()
                df = engineer_features(df)
                log(f"  Features added in {time.time()-eng_start:.1f}s")
            
            # Add IDs
            df['transaction_id'] = cp.arange(len(df), dtype=cp.int64)
            
            # Write output
            output_file = self.output_path / f"features_{run_name}.parquet"
            log(f"  Writing {len(df):,} records...")
            
            write_start = time.time()
            
            # Convert to pandas in chunks to manage memory
            record_count = len(df)
            pdf = df.to_pandas()
            del df
            free_gpu_memory()
            
            pdf.to_parquet(str(output_file), compression='snappy', index=False)
            size_mb = output_file.stat().st_size / (1024**2)
            log(f"  Wrote {size_mb:.0f} MB in {time.time()-write_start:.1f}s")
            del pdf
            gc.collect()
            
            # Save metadata
            meta = {
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                "record_count": record_count,
                "file_size_mb": size_mb,
            }
            with open(self.output_path / f"metadata_{run_name}.json", 'w') as f:
                json.dump(meta, f, indent=2)
            
            # Mark complete
            self.processed.add(run_name)
            self._save_state()
            
            log(f"SUCCESS: {run_name} ({time.time()-start:.1f}s total)")
            return True
            
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            free_gpu_memory()
            return False
    
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
            # Free memory between runs
            free_gpu_memory()
        
        log("=" * 60)
        log("Processing complete")
        log("=" * 60)


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