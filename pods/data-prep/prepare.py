#!/usr/bin/env python3
"""
Pod 2: Data Prepare Service (GPU - RAPIDS cuDF)
================================================
GPU-accelerated feature engineering using RAPIDS cuDF.
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set
from dataclasses import dataclass

import cudf
import cupy as cp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

STOP_FLAG = False


@dataclass
class PrepConfig:
    input_dir: str
    output_dir: str
    poll_interval: int = 5
    batch_mode: bool = False


def signal_handler(signum, frame):
    global STOP_FLAG
    logger.info(f"Received signal {signum}, shutting down...")
    STOP_FLAG = True


def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"{ts} - {msg}", flush=True)


class DirectoryWatcher:
    def __init__(self, watch_dir: Path):
        self.watch_dir = watch_dir
        self.state_file = watch_dir / ".prep_state.json"
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
            logger.warning(f"Could not save state: {e}")
    
    def get_new_runs(self) -> List[Path]:
        if not self.watch_dir.exists():
            return []
        
        new_runs = []
        for entry in self.watch_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("run_"):
                if entry.name not in self.processed_dirs:
                    files = list(entry.glob("worker_*.parquet")) or list(entry.glob("worker_*.csv"))
                    if files:
                        new_runs.append(entry)
        return sorted(new_runs)
    
    def mark_processed(self, run_dir: Path):
        self.processed_dirs.add(run_dir.name)
        self._save_state()


class FeatureEngineer:
    """GPU-accelerated feature engineering using RAPIDS cuDF."""
    
    PCA_COLS = [f"V{i}" for i in range(1, 29)]
    
    def process(self, df: cudf.DataFrame, run_name: str) -> cudf.DataFrame:
        start = time.time()
        log(f"Processing {len(df):,} records from {run_name} on GPU")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Standard scaling on V1-V28 (GPU-accelerated)
        for col in self.PCA_COLS:
            if col in df.columns:
                mean = float(df[col].mean())
                std = max(float(df[col].std()), 1e-8)
                df[f"{col}_scaled"] = (df[col] - mean) / std
        
        # Amount features
        if 'Amount' in df.columns:
            df['Amount_log'] = cp.log1p(df['Amount'].values)
            mean = float(df['Amount'].mean())
            std = max(float(df['Amount'].std()), 1e-8)
            df['Amount_scaled'] = (df['Amount'] - mean) / std
        
        # Time features
        if 'Time' in df.columns:
            df['hour_of_day'] = ((df['Time'] / 3600) % 24).astype('int32')
            df['is_night'] = ((df['hour_of_day'] >= 22) | (df['hour_of_day'] < 6)).astype('int8')
            
            # Sort and rolling features
            df = df.sort_values('Time').reset_index(drop=True)
            df['amount_rolling_mean'] = df['Amount'].rolling(100, min_periods=1).mean()
            df['amount_rolling_std'] = df['Amount'].rolling(100, min_periods=1).std().fillna(0)
            df['amount_deviation'] = (df['Amount'] - df['amount_rolling_mean']) / (df['amount_rolling_std'] + 1e-8)
        
        # Interaction features
        if 'V1' in df.columns and 'V2' in df.columns:
            df['V1_V2_interaction'] = df['V1'] * df['V2']
        if 'V1' in df.columns and 'V3' in df.columns:
            df['V1_V3_interaction'] = df['V1'] * df['V3']
        
        # Squared terms
        for col in ['V1', 'V14', 'V17']:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
        
        # Metadata
        df['source_run'] = run_name
        df['prep_timestamp'] = datetime.now().isoformat()
        df['transaction_id'] = cp.arange(len(df))
        
        elapsed = time.time() - start
        log(f"GPU feature engineering complete: {elapsed:.1f}s ({len(df)/elapsed:,.0f} rec/s)")
        return df


class DataPrepService:
    def __init__(self, config: PrepConfig):
        self.config = config
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.watcher = DirectoryWatcher(self.input_path)
        self.engineer = FeatureEngineer()
        
        log("=" * 60)
        log("Pod 2: Data Prep Service (RAPIDS cuDF - GPU)")
        log("=" * 60)
        log(f"Input:  {self.input_path}")
        log(f"Output: {self.output_path}")
        log(f"Mode:   {'batch' if config.batch_mode else 'continuous'}")
        log("=" * 60)
    
    def read_run_data(self, run_dir: Path) -> Optional[cudf.DataFrame]:
        parquet_files = sorted(run_dir.glob("worker_*.parquet"))
        if parquet_files:
            log(f"Reading {len(parquet_files)} Parquet files to GPU...")
            dfs = [cudf.read_parquet(str(f)) for f in parquet_files]
            return cudf.concat(dfs, ignore_index=True)
        
        csv_files = sorted(run_dir.glob("worker_*.csv"))
        if csv_files:
            log(f"Reading {len(csv_files)} CSV files to GPU...")
            dfs = [cudf.read_csv(str(f)) for f in csv_files]
            return cudf.concat(dfs, ignore_index=True)
        
        return None
    
    def write_output(self, df: cudf.DataFrame, run_name: str):
        output_file = self.output_path / f"features_{run_name}.parquet"
        log(f"Writing: {output_file}")
        df.to_parquet(str(output_file), compression='snappy')
        
        size_mb = output_file.stat().st_size / (1024*1024)
        log(f"Output: {size_mb:.1f} MB, {len(df):,} records")
        
        # Metadata for Pod 3
        exclude = ['transaction_id', 'source_run', 'prep_timestamp', 'Class']
        feature_cols = [c for c in df.columns if c not in exclude]
        
        metadata = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "columns": list(df.columns),
            "feature_columns": feature_cols,
            "target_column": "Class",
            "record_count": len(df)
        }
        
        meta_file = self.output_path / f"metadata_{run_name}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def process_run(self, run_dir: Path) -> bool:
        run_name = run_dir.name
        log("=" * 60)
        log(f"Processing: {run_name}")
        
        try:
            df = self.read_run_data(run_dir)
            if df is None:
                log(f"No data found in {run_dir}")
                return False
            
            df = self.engineer.process(df, run_name)
            self.write_output(df, run_name)
            self.watcher.mark_processed(run_dir)
            
            log(f"Complete: {run_name}")
            return True
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return False
    
    def run(self):
        global STOP_FLAG
        
        if self.config.batch_mode:
            for run_dir in self.watcher.get_new_runs():
                self.process_run(run_dir)
            log("Batch complete")
        else:
            log("Continuous mode - watching for new runs...")
            while not STOP_FLAG:
                for run_dir in self.watcher.get_new_runs():
                    if STOP_FLAG:
                        break
                    self.process_run(run_dir)
                
                if not STOP_FLAG:
                    time.sleep(self.config.poll_interval)
            log("Stopped")


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    config = PrepConfig(
        input_dir=os.getenv('INPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'),
        output_dir=os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/prep_output'),
        poll_interval=int(os.getenv('POLL_INTERVAL', '5')),
        batch_mode=os.getenv('BATCH_MODE', 'false').lower() == 'true'
    )
    
    DataPrepService(config).run()


if __name__ == "__main__":
    main()