#!/usr/bin/env python3
"""
GPU Worker for Fraud Detection Demo v2

Executes 4 pipeline stages using GPU-accelerated processing:
1. Ingest - Read transactions from parquet (cuDF)
2. Data Prep - Feature engineering (cuDF/cupy)
3. Model Train - XGBoost training (GPU)
4. Inference - Batch scoring via Triton (GPU model)

Communicates with dashboard via HTTP for orchestration.
"""

import os
import sys
import time
import json
import threading
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import cudf
import cupy as cp
import xgboost as xgb
import requests
from flask import Flask, jsonify, request

# Configuration
DASHBOARD_URL = os.getenv('DASHBOARD_URL', 'http://dashboard:5000')
TRITON_URL = os.getenv('TRITON_URL', 'http://triton:8000')
DATA_DIR = Path(os.getenv('DATA_DIR', '/data/gpu'))
MODEL_DIR = Path(os.getenv('MODEL_DIR', '/models'))
WORKER_PORT = int(os.getenv('WORKER_PORT', '5002'))
WORKER_TYPE = 'gpu'

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
stage_data: Dict[str, Any] = {}
is_running = False


def log(msg: str):
    """Print timestamped log message."""
    print(f"{datetime.now():%H:%M:%S} [GPU] {msg}", flush=True)


def free_gpu_memory():
    """Free GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


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
        pass  # Don't block on metrics reporting


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
        self.report_interval = 0.1

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
    """Read transaction data from parquet file using cuDF (GPU-accelerated)."""
    log("Stage 1: INGEST - Reading data with cuDF (GPU)...")

    input_file = DATA_DIR / 'transactions.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f"Data file not found: {input_file}")

    file_size = input_file.stat().st_size

    # Get row count first
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(str(input_file))
    total_rows = parquet_file.metadata.num_rows

    tracker = MetricsTracker('ingest', total_rows)

    # cuDF reads entire file at once (GPU-accelerated)
    start_read = time.time()
    df = cudf.read_parquet(str(input_file))
    read_time = time.time() - start_read

    # Report metrics
    tracker.update(rows=len(df), bytes_read=file_size)

    # Store for next stage
    stage_data['raw_df'] = df

    metrics = tracker.finalize()
    log(f"  Loaded {metrics['rows_processed']:,} rows in {metrics['elapsed_seconds']:.2f}s "
        f"({metrics['throughput_mbps']:.1f} MB/s)")

    return metrics


# =============================================================================
# STAGE 2: DATA PREP
# =============================================================================
def engineer_features_gpu(df: cudf.DataFrame) -> cudf.DataFrame:
    """Apply feature engineering transformations using cuDF."""
    # Amount features
    df['amt_log'] = cp.log1p(df['amt'].values)
    amt_mean = float(df['amt'].mean())
    amt_std = float(df['amt'].std())
    df['amt_scaled'] = (df['amt'] - amt_mean) / amt_std

    # Time features
    hours = (df['unix_time'] / 3600) % 24
    df['hour_of_day'] = hours.astype('float32')
    df['day_of_week'] = ((df['unix_time'] / 86400) % 7).astype('int8')
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int8')
    df['is_night'] = ((hours >= 22) | (hours <= 6)).astype('int8')

    # Distance: customer to merchant
    dlat = df['merch_lat'] - df['lat']
    dlon = df['merch_long'] - df['long']
    df['distance_km'] = cp.sqrt(dlat.values**2 + dlon.values**2) * 111

    # Categorical encoding
    df['category_encoded'] = df['category'].map(CATEGORY_MAP).fillna(-1).astype('int8')
    df['state_encoded'] = df['state'].map(STATE_MAP).fillna(-1).astype('int8')
    df['gender_encoded'] = (df['gender'] == 'M').astype('int8')

    # Population features
    df['city_pop_log'] = cp.log1p(df['city_pop'].values)
    df['zip_region'] = (df['zip'] / 10000).astype('int8')

    return df


def stage_data_prep() -> Dict[str, Any]:
    """Feature engineering using cuDF (GPU-accelerated)."""
    log("Stage 2: DATA PREP - Feature engineering with cuDF (GPU)...")

    df = stage_data.get('raw_df')
    if df is None:
        raise ValueError("No data from ingest stage")

    total_rows = len(df)
    tracker = MetricsTracker('data_prep', total_rows)

    # GPU processes entire DataFrame at once
    start_process = time.time()
    df = engineer_features_gpu(df)
    process_time = time.time() - start_process

    # Estimate bytes processed
    bytes_est = total_rows * 200
    tracker.update(rows=total_rows, bytes_read=bytes_est)

    # Keep only needed columns
    keep_cols = [c for c in FEATURE_COLUMNS + ['is_fraud'] if c in df.columns]
    df = df[keep_cols]

    # Write features to storage (FlashBlade I/O)
    output_file = DATA_DIR / 'features.parquet'
    log(f"  Writing features to {output_file}...")

    write_start = time.time()
    df.to_parquet(str(output_file), engine='pyarrow', compression='snappy')
    written_bytes = output_file.stat().st_size
    write_elapsed = time.time() - write_start
    write_throughput = written_bytes / write_elapsed / (1024**2) if write_elapsed > 0 else 0

    log(f"  Wrote {written_bytes / (1024**2):.1f} MB in {write_elapsed:.2f}s ({write_throughput:.1f} MB/s)")

    # Update tracker with write bytes
    tracker.bytes_processed += written_bytes

    # Clear memory - next stage will read from disk
    del stage_data['raw_df']
    free_gpu_memory()

    metrics = tracker.finalize()
    metrics['write_throughput_mbps'] = round(write_throughput, 1)
    metrics['output_file_size_mb'] = round(written_bytes / (1024**2), 1)
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
                'eta_seconds': round(eta_seconds, 1),
                'is_training': True
            })

            self.last_report_time = now

        return False  # Return False to continue training


def stage_model_train() -> Dict[str, Any]:
    """Read features from storage, then train XGBoost model using GPU."""
    log("Stage 3: MODEL TRAIN - Reading features and training XGBoost (GPU)...")

    # Read features from storage (FlashBlade I/O)
    input_file = DATA_DIR / 'features.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f"Features file not found: {input_file}")

    file_size = input_file.stat().st_size
    log(f"  Reading features from {input_file} ({file_size / (1024**2):.1f} MB)...")

    # Get row count first
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows

    tracker = MetricsTracker('model_train', total_rows)

    # Read the features file into cuDF
    read_start = time.time()
    df = cudf.read_parquet(str(input_file))
    read_elapsed = time.time() - read_start
    read_throughput = file_size / read_elapsed / (1024**2) if read_elapsed > 0 else 0

    log(f"  Read {total_rows:,} rows in {read_elapsed:.2f}s ({read_throughput:.1f} MB/s)")

    tracker.update(rows=total_rows, bytes_read=file_size)

    # Prepare features and labels
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Convert to numpy via cupy for XGBoost
    X = df[feature_cols].to_cupy().get().astype(np.float32)
    y = df['is_fraud'].to_cupy().get().astype(np.int32)

    # Store for inference stage
    stage_data['features_df'] = df
    stage_data['feature_cols'] = feature_cols

    # Handle NaN
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # XGBoost parameters (GPU)
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss'],
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'tree_method': 'hist',
        'device': 'cuda:0',
    }

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_cols)

    # Train with progress callback
    num_rounds = 20  # Reduced for demo (was 100)
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
    model_path = MODEL_DIR / 'fraud_xgboost_gpu'
    model_path.mkdir(parents=True, exist_ok=True)
    (model_path / '1').mkdir(exist_ok=True)

    model.save_model(str(model_path / '1' / 'xgboost.json'))

    # Triton config for GPU model
    config = f'''name: "fraud_xgboost_gpu"
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
instance_group [{{ kind: KIND_GPU, count: 1 }}]
'''
    with open(model_path / 'config.pbtxt', 'w') as f:
        f.write(config)

    with open(model_path / 'feature_names.json', 'w') as f:
        json.dump(feature_cols, f)

    # Store model reference
    stage_data['model'] = model
    stage_data['feature_cols'] = feature_cols

    free_gpu_memory()

    metrics = tracker.finalize()
    log(f"  Trained model in {metrics['elapsed_seconds']:.2f}s")
    log(f"  Model saved to {model_path}")

    return metrics


# =============================================================================
# STAGE 4: INFERENCE
# =============================================================================
def stage_inference() -> Dict[str, Any]:
    """Read features from storage, score, and write results."""
    log("Stage 4: INFERENCE - Reading features, scoring, and writing results...")

    # Read features from storage (FlashBlade I/O)
    input_file = DATA_DIR / 'features.parquet'
    if not input_file.exists():
        raise FileNotFoundError(f"Features file not found: {input_file}")

    file_size = input_file.stat().st_size
    log(f"  Reading features from {input_file} ({file_size / (1024**2):.1f} MB)...")

    # Get row count first
    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(input_file)
    total_rows = parquet_file.metadata.num_rows

    # Track inference progress (just row count, not 2x)
    tracker = MetricsTracker('inference', total_rows)

    # Read the features file
    read_start = time.time()
    df = cudf.read_parquet(str(input_file))
    read_elapsed = time.time() - read_start
    read_throughput = file_size / read_elapsed / (1024**2) if read_elapsed > 0 else 0

    log(f"  Read {total_rows:,} rows in {read_elapsed:.2f}s ({read_throughput:.1f} MB/s)")

    # Note: Don't update tracker here - let batch loop show progressive counting

    # Get feature columns and model from previous stage
    feature_cols = stage_data.get('feature_cols')
    if feature_cols is None:
        feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]

    # Prepare features (convert from GPU to CPU for inference)
    X = df[feature_cols].to_cupy().get().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Batch inference
    batch_size = 8192
    predictions = []

    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]

        try:
            response = requests.post(
                f"{TRITON_URL}/v2/models/fraud_xgboost_gpu/infer",
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
        # Update tracker with batch progress for visual feedback
        tracker.update(rows=len(batch), bytes_read=batch.nbytes)

    predictions = np.array(predictions)

    # Calculate some stats
    fraud_count = (predictions > 0.5).sum()

    # Finalize scoring metrics
    score_metrics = tracker.finalize()
    score_elapsed = score_metrics['elapsed_seconds']
    score_throughput = score_metrics['throughput_mbps']

    log(f"  Scored {score_metrics['rows_processed']:,} transactions in {score_elapsed:.2f}s "
        f"({score_throughput:.1f} MB/s)")
    log(f"  Detected {fraud_count:,} potential fraud cases ({fraud_count / total_rows * 100:.2f}%)")

    # Add predictions to dataframe
    df['fraud_score'] = predictions
    df['is_fraud_predicted'] = (predictions > 0.5).astype('int8')

    # Phase 2: Write scored results back to disk
    output_file = DATA_DIR / 'scored_transactions.parquet'
    log(f"  Writing scored results to {output_file}...")

    write_start = time.time()
    df.to_parquet(str(output_file), engine='pyarrow', compression='snappy')
    written_bytes = output_file.stat().st_size
    write_elapsed = time.time() - write_start
    write_throughput = written_bytes / write_elapsed / (1024**2) if write_elapsed > 0 else 0

    log(f"  Wrote {written_bytes / (1024**2):.1f} MB in {write_elapsed:.2f}s ({write_throughput:.1f} MB/s)")

    # Report final metrics with both score and write throughput
    total_elapsed = score_elapsed + write_elapsed
    total_bytes = score_metrics['bytes_processed'] + written_bytes

    metrics = {
        'rows_processed': total_rows,
        'total_rows': total_rows,
        'bytes_processed': total_bytes,
        'throughput_mbps': round(total_bytes / total_elapsed / (1024**2), 1) if total_elapsed > 0 else 0,
        'elapsed_seconds': round(total_elapsed, 3),
        'fraud_detected': int(fraud_count),
        'fraud_rate': round(fraud_count / total_rows * 100, 2),
        'score_throughput_mbps': round(score_throughput, 1),
        'write_throughput_mbps': round(write_throughput, 1),
        'output_file_size_mb': round(written_bytes / (1024**2), 1)
    }

    free_gpu_memory()

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
            import traceback
            traceback.print_exc()
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
    free_gpu_memory()

    return jsonify({'status': 'reset'})


# =============================================================================
# MAIN
# =============================================================================
def main():
    log("=" * 60)
    log("Fraud Detection Demo v2 - GPU Worker")
    log("=" * 60)
    log(f"Data directory: {DATA_DIR}")
    log(f"Model directory: {MODEL_DIR}")
    log(f"Dashboard URL: {DASHBOARD_URL}")
    log(f"Triton URL: {TRITON_URL}")
    log(f"Worker port: {WORKER_PORT}")

    # Check GPU availability
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
        log(f"GPUs available: {gpu_count}")
        for i in range(gpu_count):
            props = cp.cuda.runtime.getDeviceProperties(i)
            log(f"  GPU {i}: {props['name'].decode()}")
    except Exception as e:
        log(f"GPU check failed: {e}")

    log("-" * 60)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    app.run(host='0.0.0.0', port=WORKER_PORT, threaded=True)


if __name__ == '__main__':
    main()
