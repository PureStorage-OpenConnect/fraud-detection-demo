#!/usr/bin/env python3
"""
CPU Worker for Fraud Detection Demo v2

Executes 4 pipeline stages using CPU-only processing:
1. Ingest - Read transactions from parquet (pandas)
2. Data Prep - Feature engineering (pandas/numpy)
3. Model Train - XGBoost training (CPU)
4. Inference - Batch scoring via Triton (CPU model)

Communicates with dashboard via HTTP for orchestration.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import numpy as np
import pandas as pd
import xgboost as xgb
import requests
from flask import Flask, jsonify, request

# Configuration
DASHBOARD_URL = os.getenv('DASHBOARD_URL', 'http://dashboard:5000')
TRITON_URL = os.getenv('TRITON_URL', 'http://triton:8000')
DATA_DIR = Path(os.getenv('DATA_DIR', '/data/cpu'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/models'))
WORKER_PORT = int(os.getenv('WORKER_PORT', '5001'))
WORKER_TYPE = os.getenv('WORKER_TYPE', 'cpu')

# Feature engineering constants
CATEGORIES = [
    'gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
    'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
    'kids_pets', 'travel', 'health_fitness', 'personal_care'
]
CATEGORY_MAP = {cat: i for i, cat in enumerate(CATEGORIES)}

US_STATES = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
    'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
    'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY'
]
STATE_MAP = {state: i for i, state in enumerate(US_STATES)}

FEATURE_COLUMNS = [
    'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
    'merch_zipcode', 'zip', 'amt_log', 'amt_scaled', 'hour_of_day', 'day_of_week',
    'is_weekend', 'is_night', 'distance_km', 'category_encoded', 'state_encoded',
    'gender_encoded', 'city_pop_log', 'zip_region'
]

# Flask app for receiving commands
app = Flask(__name__)

# Global state
current_stage: Optional[str] = None
stage_metrics: Dict[str, Any] = {}
stage_data: Dict[str, Any] = {}  # Store intermediate data between stages
is_running = False


def log(msg: str):
    """Print timestamped log message."""
    print(f"{datetime.now():%H:%M:%S} [CPU] {msg}", flush=True)


def report_metrics(stage: str, metrics: Dict[str, Any]):
    """Send metrics update to dashboard."""
    try:
        requests.post(
            f"{DASHBOARD_URL}/api/metrics",
            json={
                'worker': WORKER_TYPE,
                'stage': stage,
                'metrics': metrics,
                'timestamp': time.time()
            },
            timeout=1
        )
    except Exception as e:
        log(f"Failed to report metrics: {e}")


def report_complete(stage: str, final_metrics: Dict[str, Any]):
    """Signal stage completion to dashboard."""
    try:
        requests.post(
            f"{DASHBOARD_URL}/api/complete",
            json={
                'worker': WORKER_TYPE,
                'stage': stage,
                'metrics': final_metrics,
                'timestamp': time.time()
            },
            timeout=5
        )
    except Exception as e:
        log(f"Failed to report completion: {e}")


class MetricsTracker:
    """Track and report metrics during stage execution."""

    def __init__(self, stage: str, total_rows: int = 0):
        self.stage = stage
        self.total_rows = total_rows
        self.rows_processed = 0
        self.bytes_processed = 0
        self.start_time = time.time()
        self.last_report = 0
        self.report_interval = 0.1  # Report every 100ms

    def update(self, rows: int = 0, bytes_read: int = 0):
        """Update metrics and report if interval elapsed."""
        self.rows_processed += rows
        self.bytes_processed += bytes_read

        now = time.time()
        if now - self.last_report >= self.report_interval:
            self.report()
            self.last_report = now

    def report(self):
        """Send current metrics to dashboard."""
        elapsed = time.time() - self.start_time
        throughput_mbps = self.bytes_processed / elapsed / (1024**2) if elapsed > 0 else 0

        report_metrics(self.stage, {
            'rows_processed': self.rows_processed,
            'total_rows': self.total_rows,
            'bytes_processed': self.bytes_processed,
            'throughput_mbps': round(throughput_mbps, 1),
            'elapsed_seconds': round(elapsed, 2)
        })

    def finalize(self) -> Dict[str, Any]:
        """Return final metrics."""
        elapsed = time.time() - self.start_time
        throughput_mbps = self.bytes_processed / elapsed / (1024**2) if elapsed > 0 else 0

        return {
            'rows_processed': self.rows_processed,
            'total_rows': self.total_rows,
            'bytes_processed': self.bytes_processed,
            'throughput_mbps': round(throughput_mbps, 1),
            'elapsed_seconds': round(elapsed, 3)
        }


# =============================================================================
# STAGE 1: INGEST
# =============================================================================
def stage_ingest() -> Dict[str, Any]:
    """Read transaction data from parquet file using pandas."""
    log("Stage 1: INGEST - Reading data with pandas...")

    input_file = DATA_DIR / 'transactions.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    file_size = input_file.stat().st_size

    # Get row count first for progress tracking
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows

    tracker = MetricsTracker('ingest', total_rows)

    # Read in chunks for progress reporting
    chunk_size = 100_000
    chunks = []

    for chunk in pd.read_parquet(input_file, engine='pyarrow').pipe(
        lambda df: [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    ):
        chunks.append(chunk)
        tracker.update(rows=len(chunk), bytes_read=int(file_size * len(chunk) / total_rows))

    df = pd.concat(chunks, ignore_index=True)

    # Store for next stage
    stage_data['raw_df'] = df

    metrics = tracker.finalize()
    log(f"  Loaded {metrics['rows_processed']:,} rows in {metrics['elapsed_seconds']:.2f}s "
        f"({metrics['throughput_mbps']:.1f} MB/s)")

    return metrics


# =============================================================================
# STAGE 2: DATA PREP
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature engineering transformations."""
    # Amount features
    df['amt_log'] = np.log1p(df['amt'].values)
    amt_mean = df['amt'].mean()
    amt_std = df['amt'].std()
    df['amt_scaled'] = (df['amt'] - amt_mean) / amt_std

    # Time features
    hours = (df['unix_time'] / 3600) % 24
    df['hour_of_day'] = hours.astype(np.float32)
    df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype(np.int8)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(np.int8)
    df['is_night'] = ((hours >= 22) | (hours <= 6)).astype(np.int8)

    # Distance: customer to merchant (simplified Euclidean)
    dlat = df['merch_lat'] - df['lat']
    dlon = df['merch_long'] - df['long']
    df['distance_km'] = np.sqrt(dlat**2 + dlon**2) * 111  # Rough km conversion

    # Categorical encoding
    df['category_encoded'] = df['category'].map(CATEGORY_MAP).fillna(-1).astype(np.int8)
    df['state_encoded'] = df['state'].map(STATE_MAP).fillna(-1).astype(np.int8)
    df['gender_encoded'] = (df['gender'] == 'M').astype(np.int8)

    # Population features
    df['city_pop_log'] = np.log1p(df['city_pop'].values)
    df['zip_region'] = (df['zip'] / 10000).astype(np.int8)

    return df


def stage_data_prep() -> Dict[str, Any]:
    """Feature engineering using pandas/numpy."""
    log("Stage 2: DATA PREP - Feature engineering with pandas...")

    df = stage_data.get('raw_df')
    if df is None:
        raise ValueError("No data from ingest stage")

    total_rows = len(df)
    tracker = MetricsTracker('data_prep', total_rows)

    # Process in chunks for progress reporting
    chunk_size = 100_000
    processed_chunks = []

    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size].copy()
        chunk = engineer_features(chunk)
        processed_chunks.append(chunk)

        # Estimate bytes (rough approximation)
        bytes_est = chunk_size * 200  # ~200 bytes per row
        tracker.update(rows=len(chunk), bytes_read=bytes_est)

    df = pd.concat(processed_chunks, ignore_index=True)

    # Keep only needed columns
    keep_cols = FEATURE_COLUMNS + ['is_fraud']
    df = df[[c for c in keep_cols if c in df.columns]]

    # Store for next stage
    stage_data['features_df'] = df
    del stage_data['raw_df']  # Free memory

    metrics = tracker.finalize()
    log(f"  Processed {metrics['rows_processed']:,} rows in {metrics['elapsed_seconds']:.2f}s")

    return metrics


# =============================================================================
# STAGE 3: MODEL TRAIN
# =============================================================================
class TrainingProgressCallback(xgb.callback.TrainingCallback):
    """Custom XGBoost callback to report training progress to dashboard."""

    def __init__(self, tracker: MetricsTracker, total_rounds: int):
        self.tracker = tracker
        self.total_rounds = total_rounds
        self.start_time = time.time()
        self.last_report_time = 0
        self.report_interval = 1.0  # Report every 1 second

    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """Called after each training iteration."""
        now = time.time()

        # Report progress periodically (every second)
        if now - self.last_report_time >= self.report_interval:
            elapsed = now - self.start_time
            progress_pct = (epoch + 1) / self.total_rounds * 100

            # Get eval metrics if available
            train_auc = None
            eval_auc = None
            if 'train' in evals_log and 'auc' in evals_log['train']:
                train_auc = evals_log['train']['auc'][-1]
            if 'eval' in evals_log and 'auc' in evals_log['eval']:
                eval_auc = evals_log['eval']['auc'][-1]

            # Estimate time remaining
            if epoch > 0:
                time_per_round = elapsed / (epoch + 1)
                remaining_rounds = self.total_rounds - epoch - 1
                eta_seconds = time_per_round * remaining_rounds
            else:
                eta_seconds = 0

            auc_str = f"{eval_auc:.4f}" if eval_auc else "N/A"
            log(f"  Training: round {epoch + 1}/{self.total_rounds} ({progress_pct:.0f}%) - "
                f"AUC: {auc_str} - ETA: {eta_seconds:.0f}s")

            # Report metrics to dashboard
            report_metrics('model_train', {
                'rows_processed': self.tracker.rows_processed,
                'total_rows': self.tracker.total_rows,
                'bytes_processed': self.tracker.bytes_processed,
                'throughput_mbps': 0,
                'elapsed_seconds': round(elapsed, 2),
                'training_round': epoch + 1,
                'total_rounds': self.total_rounds,
                'training_progress_pct': round(progress_pct, 1),
                'train_auc': round(train_auc, 4) if train_auc else None,
                'eval_auc': round(eval_auc, 4) if eval_auc else None,
                'eta_seconds': round(eta_seconds, 1)
            })

            self.last_report_time = now

        return False  # Return False to continue training


def stage_model_train() -> Dict[str, Any]:
    """Train XGBoost model using CPU."""
    log("Stage 3: MODEL TRAIN - Training XGBoost (CPU)...")

    df = stage_data.get('features_df')
    if df is None:
        raise ValueError("No data from data_prep stage")

    total_rows = len(df)
    tracker = MetricsTracker('model_train', total_rows)

    # Prepare features and labels
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    y = df['is_fraud'].values.astype(np.int32)

    # Handle any NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    tracker.update(rows=total_rows, bytes_read=X.nbytes)

    # Calculate class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # XGBoost parameters (CPU)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'nthread': -1,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Train with early stopping and progress callback
    num_rounds = 100
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    progress_callback = TrainingProgressCallback(tracker, num_rounds)

    log(f"  Starting XGBoost training with {num_rounds} rounds...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_rounds,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False,
        callbacks=[progress_callback]
    )

    # Save model for Triton
    model_path = MODEL_DIR / 'fraud_xgboost_cpu'
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / '1').mkdir(exist_ok=True)

    model.save_model(str(model_path / '1' / 'xgboost.json'))

    # Save config for Triton
    config = f'''name: "fraud_xgboost_cpu"
backend: "fil"
max_batch_size: 8192
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {len(feature_cols)} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [{{ kind: KIND_CPU, count: 2 }}]
'''
    with open(model_path / 'config.pbtxt', 'w') as f:
        f.write(config)

    # Save feature names
    with open(model_path / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f)

    # Store model reference
    stage_data['model'] = model
    stage_data['feature_cols'] = feature_cols

    metrics = tracker.finalize()
    log(f"  Trained model in {metrics['elapsed_seconds']:.2f}s")
    log(f"  Model saved to {model_path}")

    return metrics


# =============================================================================
# STAGE 4: INFERENCE
# =============================================================================
def stage_inference() -> Dict[str, Any]:
    """Batch inference using Triton (CPU model)."""
    log("Stage 4: INFERENCE - Batch scoring via Triton (CPU model)...")

    df = stage_data.get('features_df')
    feature_cols = stage_data.get('feature_cols')

    if df is None or feature_cols is None:
        raise ValueError("No data/model from previous stages")

    total_rows = len(df)
    tracker = MetricsTracker('inference', total_rows)

    # Prepare features
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Batch inference
    batch_size = 8192
    predictions = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]

        # Try Triton first, fall back to local model
        try:
            response = requests.post(
                f"{TRITON_URL}/v2/models/fraud_xgboost_cpu/infer",
                json={
                    'inputs': [{
                        'name': 'input__0',
                        'shape': list(batch.shape),
                        'datatype': 'FP32',
                        'data': batch.flatten().tolist()
                    }]
                },
                timeout=30
            )
            if response.status_code == 200:
                result = response.json()
                preds = np.array(result['outputs'][0]['data'])
            else:
                raise Exception(f"Triton error: {response.status_code}")
        except Exception as e:
            # Fall back to local model
            model = stage_data.get('model')
            if model:
                dmat = xgb.DMatrix(batch, feature_names=feature_cols)
                preds = model.predict(dmat)
            else:
                preds = np.zeros(len(batch))

        predictions.extend(preds)
        tracker.update(rows=len(batch), bytes_read=batch.nbytes)

    predictions = np.array(predictions)

    # Calculate some stats
    fraud_count = (predictions > 0.5).sum()

    metrics = tracker.finalize()
    metrics['fraud_detected'] = int(fraud_count)
    metrics['fraud_rate'] = round(fraud_count / total_rows * 100, 2)

    log(f"  Scored {metrics['rows_processed']:,} transactions in {metrics['elapsed_seconds']:.2f}s")
    log(f"  Detected {fraud_count:,} potential fraud cases ({metrics['fraud_rate']:.2f}%)")

    return metrics


# =============================================================================
# FLASK ENDPOINTS
# =============================================================================
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'worker': WORKER_TYPE})


@app.route('/api/start', methods=['POST'])
def start_stage():
    """Start a pipeline stage."""
    global current_stage, is_running

    if is_running:
        return jsonify({'error': 'Stage already running'}), 400

    data = request.json
    stage = data.get('stage')

    stages = {
        'ingest': stage_ingest,
        'data_prep': stage_data_prep,
        'model_train': stage_model_train,
        'inference': stage_inference
    }

    if stage not in stages:
        return jsonify({'error': f'Unknown stage: {stage}'}), 400

    def run_stage():
        global current_stage, is_running
        is_running = True
        current_stage = stage

        try:
            log(f"Starting stage: {stage}")
            metrics = stages[stage]()
            report_complete(stage, metrics)
            log(f"Completed stage: {stage}")
        except Exception as e:
            log(f"Error in stage {stage}: {e}")
            report_complete(stage, {'error': str(e)})
        finally:
            is_running = False
            current_stage = None

    thread = threading.Thread(target=run_stage)
    thread.start()

    return jsonify({'status': 'started', 'stage': stage})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current worker status."""
    return jsonify({
        'worker': WORKER_TYPE,
        'is_running': is_running,
        'current_stage': current_stage
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset worker state."""
    global stage_data, is_running, current_stage

    if is_running:
        return jsonify({'error': 'Cannot reset while running'}), 400

    stage_data = {}
    current_stage = None

    return jsonify({'status': 'reset'})


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("Fraud Detection Demo v2 - CPU Worker")
    log("=" * 60)
    log(f"Data directory: {DATA_DIR}")
    log(f"Model directory: {MODEL_DIR}")
    log(f"Dashboard URL: {DASHBOARD_URL}")
    log(f"Triton URL: {TRITON_URL}")
    log(f"Worker port: {WORKER_PORT}")
    log("-" * 60)

    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Start Flask server
    app.run(host='0.0.0.0', port=WORKER_PORT, threaded=True)


if __name__ == '__main__':
    main()
