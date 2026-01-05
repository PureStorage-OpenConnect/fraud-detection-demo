#!/usr/bin/env python3
"""
Pod 6: Inference Benchmark
Compares CPU (XGBoost direct) vs GPU (Triton) inference performance.
Loads raw data from Pod 1, applies feature engineering, runs inference.
"""

import os
import sys
import time
import json
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering (matches Pod 2 CPU path)."""
    df = df.copy()
    
    # Amount features
    if 'amt' in df.columns:
        df['amt_log'] = np.log1p(df['amt'].values)
        mean, std = df['amt'].mean(), df['amt'].std()
        if std > 0.001:
            df['amt_scaled'] = (df['amt'] - mean) / std
        else:
            df['amt_scaled'] = 0.0
    
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
        cat_map = {c: i for i, c in enumerate(CATEGORIES)}
        df['category_encoded'] = df['category'].map(cat_map).fillna(-1).astype('int8')
    
    if 'state' in df.columns:
        state_map = {s: i for i, s in enumerate(US_STATES)}
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


class InferenceBenchmark:
    """Benchmark CPU vs GPU inference performance."""
    
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        triton_url: str,
        sample_size: int = 10000
    ):
        self.data_path = Path(data_dir)
        self.model_path = Path(model_dir)
        self.triton_url = triton_url.rstrip('/')
        self.sample_size = sample_size
        
        self.model: Optional[xgb.Booster] = None
        self.feature_names: List[str] = []
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[np.ndarray] = None
        
        log.info("=" * 70)
        log.info("Pod 6: Inference Benchmark - CPU vs GPU Comparison")
        log.info("=" * 70)
        log.info(f"Data source: {self.data_path}")
        log.info(f"Model repo:  {self.model_path}")
        log.info(f"Triton URL:  {self.triton_url}")
        log.info(f"Sample size: {self.sample_size:,}")
        log.info("=" * 70)
    
    def load_model(self) -> bool:
        """Load XGBoost model for CPU inference."""
        log.info("Loading XGBoost model for CPU inference...")
        
        # Check for model in multiple possible locations
        model_dirs = [
            "fraud_xgboost_gpu",  # GPU-trained model (preferred)
            "fraud_xgboost_cpu",  # CPU-trained model
            "fraud_xgboost",      # Default name
        ]
        
        model_file = None
        feature_file = None
        
        for model_dir in model_dirs:
            candidate = self.model_path / model_dir / "1" / "xgboost.json"
            if candidate.exists():
                model_file = candidate
                feature_file = self.model_path / model_dir / "feature_names.json"
                log.info(f"  Found model: {model_dir}")
                break
        
        if model_file is None:
            # Try alternate model filename
            for model_dir in model_dirs:
                candidate = self.model_path / model_dir / "1" / "model.json"
                if candidate.exists():
                    model_file = candidate
                    feature_file = self.model_path / model_dir / "feature_names.json"
                    log.info(f"  Found model: {model_dir}")
                    break
        
        if not model_file.exists():
            log.error(f"Model not found: {model_file}")
            return False
        
        self.model = xgb.Booster()
        self.model.load_model(str(model_file))
        log.info(f"  Loaded model: {model_file.name}")
        
        if feature_file.exists():
            with open(feature_file) as f:
                self.feature_names = json.load(f)
            log.info(f"  Features: {len(self.feature_names)} columns")
        else:
            log.warning("  Feature names file not found, will use default order")
        
        return True
    
    def check_triton(self) -> bool:
        """Check if Triton server is ready."""
        log.info("Checking Triton server availability...")
        
        try:
            resp = requests.get(f"{self.triton_url}/v2/health/ready", timeout=5)
            if resp.status_code == 200:
                log.info("  Triton server is ready")
                return True
            else:
                log.error(f"  Triton not ready: HTTP {resp.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            log.error(f"  Cannot connect to Triton: {e}")
            return False
    
    def load_sample_data(self) -> bool:
        """Load random sample from Pod 1 output."""
        log.info(f"Loading {self.sample_size:,} random records...")
        
        # Find most recent run directory
        if not self.data_path.exists():
            log.error(f"Data path not found: {self.data_path}")
            return False
        
        run_dirs = sorted([
            d for d in self.data_path.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ])
        
        if not run_dirs:
            log.error("No run directories found")
            return False
        
        run_dir = run_dirs[-1]
        log.info(f"  Using run: {run_dir.name}")
        
        # Get parquet files
        parquet_files = sorted(run_dir.glob("worker_*.parquet"))
        if not parquet_files:
            log.error("No parquet files found in run directory")
            return False
        
        log.info(f"  Found {len(parquet_files)} parquet files")
        
        # Sample from multiple files to get diversity
        samples_per_file = max(1, self.sample_size // min(len(parquet_files), 10))
        files_to_read = min(len(parquet_files), 10)
        
        parts = []
        for f in parquet_files[:files_to_read]:
            try:
                df = pd.read_parquet(f)
                if len(df) > samples_per_file:
                    df = df.sample(n=samples_per_file, random_state=42)
                parts.append(df)
            except Exception as e:
                log.warning(f"  Error reading {f.name}: {e}")
                continue
        
        if not parts:
            log.error("Failed to load any data")
            return False
        
        self.data = pd.concat(parts, ignore_index=True)
        
        # Ensure exact sample size
        if len(self.data) > self.sample_size:
            self.data = self.data.sample(n=self.sample_size, random_state=42)
        
        log.info(f"  Loaded {len(self.data):,} records")
        
        # Store ground truth
        self.ground_truth = self.data['is_fraud'].values.copy() if 'is_fraud' in self.data.columns else None
        
        return True
    
    def prepare_features(self) -> Tuple[float, float]:
        """Apply feature engineering. Returns (eng_time, total_records)."""
        log.info("Applying feature engineering...")
        
        start = time.time()
        self.data = engineer_features(self.data)
        eng_time = time.time() - start
        
        # Get features in correct order
        if self.feature_names:
            available = [c for c in self.feature_names if c in self.data.columns]
            missing = [c for c in self.feature_names if c not in self.data.columns]
            if missing:
                log.warning(f"  Missing features (using zeros): {missing}")
                for col in missing:
                    self.data[col] = 0.0
            self.features = self.data[self.feature_names].fillna(0).values.astype(np.float32)
        else:
            # Use all numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            exclude = ['is_fraud', 'cc_num', 'transaction_id']
            cols = [c for c in numeric_cols if c not in exclude]
            self.features = self.data[cols].fillna(0).values.astype(np.float32)
        
        log.info(f"  Engineered {len(self.data):,} records in {eng_time:.3f}s")
        log.info(f"  Feature matrix shape: {self.features.shape}")
        
        return eng_time
    
    def run_cpu_inference(self, iterations: int = 5) -> Tuple[float, np.ndarray]:
        """Run inference on CPU using XGBoost directly."""
        log.info("")
        log.info("=" * 70)
        log.info("CPU INFERENCE (XGBoost Direct)")
        log.info("=" * 70)
        
        dmatrix = xgb.DMatrix(self.features)
        
        # Warmup
        log.info("  Warmup run...")
        _ = self.model.predict(dmatrix)
        
        # Timed runs
        times = []
        predictions = None
        
        log.info(f"  Running {iterations} timed iterations...")
        for i in range(iterations):
            start = time.time()
            predictions = self.model.predict(dmatrix)
            elapsed = time.time() - start
            times.append(elapsed)
            log.info(f"    Iteration {i+1}: {elapsed*1000:.2f}ms")
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(self.features) / avg_time
        
        log.info(f"  Average: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        log.info(f"  Throughput: {throughput:,.0f} records/sec")
        
        return avg_time, predictions
    
    def run_gpu_inference(self, iterations: int = 5, batch_size: int = 1000) -> Tuple[float, np.ndarray]:
        """Run inference on GPU via Triton HTTP API."""
        log.info("")
        log.info("=" * 70)
        log.info("GPU INFERENCE (Triton Server)")
        log.info("=" * 70)
        
        # Try multiple model names
        model_names = ["fraud_xgboost_gpu", "fraud_xgboost_cpu", "fraud_xgboost"]
        model_name = None
        
        for name in model_names:
            try:
                resp = requests.get(f"{self.triton_url}/v2/models/{name}", timeout=5)
                if resp.status_code == 200:
                    model_name = name
                    log.info(f"  Using Triton model: {model_name}")
                    break
            except:
                continue
        
        if model_name is None:
            log.error("  No fraud model found on Triton server")
            log.error(f"  Tried: {model_names}")
            return None, None
        
        url = f"{self.triton_url}/v2/models/{model_name}/infer"
        
        def infer_batch(batch: np.ndarray) -> np.ndarray:
            """Send batch to Triton and get predictions."""
            payload = {
                "inputs": [{
                    "name": "input__0",
                    "shape": list(batch.shape),
                    "datatype": "FP32",
                    "data": batch.flatten().tolist()
                }]
            }
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            result = resp.json()
            return np.array(result["outputs"][0]["data"])
        
        # Warmup
        log.info("  Warmup run...")
        _ = infer_batch(self.features[:batch_size])
        
        # Timed runs
        times = []
        all_predictions = []
        
        log.info(f"  Running {iterations} timed iterations (batch_size={batch_size})...")
        
        for i in range(iterations):
            start = time.time()
            preds = []
            
            # Process in batches
            for j in range(0, len(self.features), batch_size):
                batch = self.features[j:j+batch_size]
                batch_preds = infer_batch(batch)
                preds.extend(batch_preds)
            
            elapsed = time.time() - start
            times.append(elapsed)
            log.info(f"    Iteration {i+1}: {elapsed*1000:.2f}ms")
            
            if i == iterations - 1:
                all_predictions = np.array(preds)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        throughput = len(self.features) / avg_time
        
        log.info(f"  Average: {avg_time*1000:.2f}ms (±{std_time*1000:.2f}ms)")
        log.info(f"  Throughput: {throughput:,.0f} records/sec")
        
        return avg_time, all_predictions
    
    def compare_predictions(self, cpu_preds: np.ndarray, gpu_preds: np.ndarray):
        """Compare CPU vs GPU predictions for consistency."""
        log.info("")
        log.info("-" * 70)
        log.info("PREDICTION CONSISTENCY CHECK")
        log.info("-" * 70)
        
        # Ensure same length
        min_len = min(len(cpu_preds), len(gpu_preds))
        cpu_preds = cpu_preds[:min_len]
        gpu_preds = gpu_preds[:min_len]
        
        # Compare
        diff = np.abs(cpu_preds - gpu_preds)
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        log.info(f"  Max difference:  {max_diff:.6f}")
        log.info(f"  Mean difference: {mean_diff:.6f}")
        
        # Binary predictions
        cpu_binary = (cpu_preds > 0.5).astype(int)
        gpu_binary = (gpu_preds > 0.5).astype(int)
        agreement = (cpu_binary == gpu_binary).mean() * 100
        
        log.info(f"  Binary agreement: {agreement:.2f}%")
        
        if max_diff < 0.0001:
            log.info("  ✓ Predictions match within tolerance")
        else:
            log.warning("  ⚠ Predictions differ - check model versions")
        
        # Fraud detection stats
        fraud_detected_cpu = cpu_binary.sum()
        fraud_detected_gpu = gpu_binary.sum()
        log.info(f"  Fraud detected (CPU): {fraud_detected_cpu} ({fraud_detected_cpu/len(cpu_binary)*100:.2f}%)")
        log.info(f"  Fraud detected (GPU): {fraud_detected_gpu} ({fraud_detected_gpu/len(gpu_binary)*100:.2f}%)")
        
        if self.ground_truth is not None:
            actual_fraud = self.ground_truth[:min_len].sum()
            log.info(f"  Actual fraud:        {actual_fraud} ({actual_fraud/min_len*100:.2f}%)")
    
    def run(self):
        """Execute full benchmark."""
        # Load model
        if not self.load_model():
            log.error("Failed to load model")
            return
        
        # Check Triton
        triton_available = self.check_triton()
        
        # Load data
        if not self.load_sample_data():
            log.error("Failed to load sample data")
            return
        
        # Feature engineering
        eng_time = self.prepare_features()
        
        # CPU inference
        cpu_time, cpu_preds = self.run_cpu_inference()
        
        # GPU inference (if Triton available)
        gpu_time = None
        gpu_preds = None
        if triton_available:
            try:
                gpu_time, gpu_preds = self.run_gpu_inference()
            except Exception as e:
                log.error(f"GPU inference failed: {e}")
                triton_available = False
        
        # Results comparison
        log.info("")
        log.info("=" * 70)
        log.info("PERFORMANCE COMPARISON")
        log.info("=" * 70)
        log.info(f"  Records:             {len(self.features):,}")
        log.info(f"  Features:            {self.features.shape[1]}")
        log.info(f"  Feature Engineering: {eng_time*1000:.2f}ms")
        log.info("")
        log.info(f"  {'Method':<20} {'Time (ms)':<15} {'Throughput':<20} {'Speedup':<10}")
        log.info(f"  {'-'*20} {'-'*15} {'-'*20} {'-'*10}")
        
        cpu_throughput = len(self.features) / cpu_time
        log.info(f"  {'CPU (XGBoost)':<20} {cpu_time*1000:<15.2f} {cpu_throughput:>15,.0f}/s {'1.0x':<10}")
        
        if gpu_time:
            gpu_throughput = len(self.features) / gpu_time
            speedup = cpu_time / gpu_time
            log.info(f"  {'GPU (Triton)':<20} {gpu_time*1000:<15.2f} {gpu_throughput:>15,.0f}/s {speedup:<10.1f}x")
            
            # Prediction consistency
            if gpu_preds is not None:
                self.compare_predictions(cpu_preds, gpu_preds)
        else:
            log.info(f"  {'GPU (Triton)':<20} {'N/A':<15} {'Server unavailable':<20}")
        
        log.info("=" * 70)
        log.info("")
        
        # Summary
        log.info("SUMMARY")
        log.info("-" * 70)
        if gpu_time and gpu_time < cpu_time:
            log.info(f"  GPU inference is {cpu_time/gpu_time:.1f}x faster than CPU")
        elif gpu_time:
            log.info(f"  CPU inference is {gpu_time/cpu_time:.1f}x faster than GPU (batch overhead)")
        log.info(f"  Total benchmark time: {eng_time + cpu_time + (gpu_time or 0):.2f}s")
        log.info("=" * 70)


def main():
    # Configuration from environment
    data_dir = os.getenv('DATA_DIR', '/data/input')
    model_dir = os.getenv('MODEL_DIR', '/data/models')
    triton_url = os.getenv('TRITON_URL', 'http://inference:8000')
    sample_size = int(os.getenv('SAMPLE_SIZE', '10000'))
    
    benchmark = InferenceBenchmark(
        data_dir=data_dir,
        model_dir=model_dir,
        triton_url=triton_url,
        sample_size=sample_size
    )
    benchmark.run()


if __name__ == "__main__":
    main()