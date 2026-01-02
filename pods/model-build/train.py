#!/usr/bin/env python3
"""
Pod 3: Model Training - CPU vs GPU Comparison
Trains XGBoost models using both CPU and GPU to demonstrate acceleration.
Outputs two Triton-compatible models: fraud_xgboost_cpu and fraud_xgboost_gpu.
"""

import os
import sys
import gc
import json
import time
import logging
from pathlib import Path
from datetime import datetime

# CPU imports
import pandas as pd
import numpy as np

# GPU imports
import cudf
import cupy as cp

import xgboost as xgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
log = logging.getLogger(__name__)

# Features for model training
FEATURE_COLUMNS = [
    'amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
    'merch_zipcode', 'zip', 'amt_log', 'amt_scaled', 'hour_of_day', 'day_of_week',
    'is_weekend', 'is_night', 'distance_km', 'category_encoded', 'state_encoded',
    'gender_encoded', 'city_pop_log', 'zip_region'
]

# Columns to exclude from features
EXCLUDE_COLUMNS = [
    'transaction_id', 'trans_date_trans_time', 'cc_num', 'merchant', 'category',
    'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'dob',
    'trans_num', 'is_fraud'
]


def free_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


class ModelTrainer:
    """XGBoost model trainer with CPU vs GPU comparison and Triton output."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Timing results
        self.cpu_times = {}
        self.gpu_times = {}
        
        log.info("=" * 70)
        log.info("Pod 3: Model Training - CPU vs GPU Comparison")
        log.info("=" * 70)
        log.info(f"Input:  {self.input_path}")
        log.info(f"Output: {self.output_path}")
    
    def load_features(self, features_file: str = None) -> Path:
        """Find the features file to load."""
        if features_file:
            filepath = self.input_path / features_file
        else:
            # Find most recent features file
            files = sorted(self.input_path.glob("features_*.parquet"))
            if not files:
                raise FileNotFoundError("No feature files found")
            filepath = files[-1]
        
        log.info(f"Features file: {filepath.name}")
        return filepath
    
    # =========================================================================
    # CPU Training Path
    # =========================================================================
    def train_cpu(self, filepath: Path):
        """Train XGBoost using CPU (pandas + XGBoost CPU)."""
        log.info("")
        log.info("=" * 70)
        log.info("PHASE 1: CPU Training (Pandas + XGBoost CPU)")
        log.info("=" * 70)
        
        # Load data
        load_start = time.time()
        df = pd.read_parquet(filepath)
        load_time = time.time() - load_start
        log.info(f"  Loaded {len(df):,} records in {load_time:.2f}s [CPU]")
        self.cpu_times['load'] = load_time
        
        # Prepare data
        prep_start = time.time()
        X_train, y_train, X_test, y_test, feature_names = self._prepare_data_cpu(df)
        prep_time = time.time() - prep_start
        log.info(f"  Prepared data in {prep_time:.2f}s [CPU]")
        self.cpu_times['prep'] = prep_time
        
        del df
        gc.collect()
        
        # Train model
        train_start = time.time()
        model = self._train_xgboost_cpu(X_train, y_train, X_test, y_test)
        train_time = time.time() - train_start
        log.info(f"  Training completed in {train_time:.2f}s [CPU]")
        self.cpu_times['train'] = train_time
        
        # Evaluate
        eval_start = time.time()
        metrics = self._evaluate_cpu(model, X_test, y_test)
        eval_time = time.time() - eval_start
        self.cpu_times['eval'] = eval_time
        
        total = load_time + prep_time + train_time + eval_time
        self.cpu_times['total'] = total
        log.info(f"  CPU TOTAL: {total:.2f}s")
        
        return model, feature_names, metrics
    
    def _prepare_data_cpu(self, df: pd.DataFrame):
        """Prepare train/test split using CPU."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        
        if missing:
            log.warning(f"  Missing features: {missing}")
        log.info(f"  Using {len(available)} features")
        
        df = df.fillna(0)
        
        # Subsample if too large
        max_records = 10_000_000
        if len(df) > max_records:
            log.info(f"  Subsampling {max_records:,} from {len(df):,} records")
            fraud_df = df[df['is_fraud'] == 1]
            normal_df = df[df['is_fraud'] == 0]
            
            fraud_ratio = len(fraud_df) / len(df)
            n_fraud = int(max_records * fraud_ratio)
            n_normal = max_records - n_fraud
            
            fraud_sample = fraud_df.sample(n=min(n_fraud, len(fraud_df)), random_state=42)
            normal_sample = normal_df.sample(n=min(n_normal, len(normal_df)), random_state=42)
            
            df = pd.concat([fraud_sample, normal_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[available].values
        y_train = train_df['is_fraud'].values
        X_test = test_df[available].values
        y_test = test_df['is_fraud'].values
        
        log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        log.info(f"  Fraud rate: {y_train.mean()*100:.2f}%")
        
        return X_train, y_train, X_test, y_test, available
    
    def _train_xgboost_cpu(self, X_train, y_train, X_test, y_test):
        """Train XGBoost on CPU."""
        fraud_count = y_train.sum()
        normal_count = len(y_train) - fraud_count
        scale_pos_weight = normal_count / max(fraud_count, 1)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'tree_method': 'hist',  # CPU histogram method
            'nthread': -1,  # Use all CPU cores
        }
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=25
        )
        
        return model
    
    def _evaluate_cpu(self, model, X_test, y_test):
        """Evaluate CPU model."""
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = (y_pred_binary == y_test).mean()
        
        fraud_mask = y_test == 1
        tp = ((y_pred_binary == 1) & fraud_mask).sum()
        fp = ((y_pred_binary == 1) & ~fraud_mask).sum()
        fn = ((y_pred_binary == 0) & fraud_mask).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        log.info(f"  [CPU] Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    # =========================================================================
    # GPU Training Path
    # =========================================================================
    def train_gpu(self, filepath: Path):
        """Train XGBoost using GPU (cuDF + XGBoost GPU)."""
        log.info("")
        log.info("=" * 70)
        log.info("PHASE 2: GPU Training (cuDF + XGBoost GPU)")
        log.info("=" * 70)
        
        # Load data
        load_start = time.time()
        df = cudf.read_parquet(filepath)
        load_time = time.time() - load_start
        log.info(f"  Loaded {len(df):,} records in {load_time:.2f}s [GPU]")
        self.gpu_times['load'] = load_time
        
        # Prepare data
        prep_start = time.time()
        X_train, y_train, X_test, y_test, feature_names = self._prepare_data_gpu(df)
        prep_time = time.time() - prep_start
        log.info(f"  Prepared data in {prep_time:.2f}s [GPU]")
        self.gpu_times['prep'] = prep_time
        
        del df
        free_gpu_memory()
        
        # Train model
        train_start = time.time()
        model = self._train_xgboost_gpu(X_train, y_train, X_test, y_test)
        train_time = time.time() - train_start
        log.info(f"  Training completed in {train_time:.2f}s [GPU]")
        self.gpu_times['train'] = train_time
        
        # Evaluate
        eval_start = time.time()
        metrics = self._evaluate_gpu(model, X_test, y_test)
        eval_time = time.time() - eval_start
        self.gpu_times['eval'] = eval_time
        
        total = load_time + prep_time + train_time + eval_time
        self.gpu_times['total'] = total
        log.info(f"  GPU TOTAL: {total:.2f}s")
        
        return model, feature_names, metrics
    
    def _prepare_data_gpu(self, df: cudf.DataFrame):
        """Prepare train/test split using GPU."""
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        
        if missing:
            log.warning(f"  Missing features: {missing}")
        log.info(f"  Using {len(available)} features")
        
        df = df.fillna(0)
        
        # Subsample if too large
        max_records = 10_000_000
        if len(df) > max_records:
            log.info(f"  Subsampling {max_records:,} from {len(df):,} records")
            fraud_df = df[df['is_fraud'] == 1]
            normal_df = df[df['is_fraud'] == 0]
            
            fraud_ratio = len(fraud_df) / len(df)
            n_fraud = int(max_records * fraud_ratio)
            n_normal = max_records - n_fraud
            
            fraud_sample = fraud_df.sample(n=min(n_fraud, len(fraud_df)), random_state=42)
            normal_sample = normal_df.sample(n=min(n_normal, len(normal_df)), random_state=42)
            
            df = cudf.concat([fraud_sample, normal_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=42)
        
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[available].to_cupy()
        y_train = train_df['is_fraud'].to_cupy()
        X_test = test_df[available].to_cupy()
        y_test = test_df['is_fraud'].to_cupy()
        
        log.info(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        log.info(f"  Fraud rate: {float(y_train.mean())*100:.2f}%")
        
        return X_train, y_train, X_test, y_test, available
    
    def _train_xgboost_gpu(self, X_train, y_train, X_test, y_test):
        """Train XGBoost on GPU."""
        fraud_count = float(y_train.sum())
        normal_count = len(y_train) - fraud_count
        scale_pos_weight = normal_count / max(fraud_count, 1)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos_weight,
            'device': 'cuda:0',
            'tree_method': 'hist',
        }
        
        model = xgb.train(
            params, dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=25
        )
        
        return model
    
    def _evaluate_gpu(self, model, X_test, y_test):
        """Evaluate GPU model."""
        dtest = xgb.DMatrix(X_test)
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = float((y_pred_binary == cp.asnumpy(y_test)).mean())
        
        y_test_np = cp.asnumpy(y_test)
        fraud_mask = y_test_np == 1
        tp = ((y_pred_binary == 1) & fraud_mask).sum()
        fp = ((y_pred_binary == 1) & ~fraud_mask).sum()
        fn = ((y_pred_binary == 0) & fraud_mask).sum()
        
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        
        log.info(f"  [GPU] Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
    
    # =========================================================================
    # Model Saving
    # =========================================================================
    def _export_feature_importance(self, model, feature_names, model_dir):
        """Export feature importance analysis."""
        importance_types = ['weight', 'gain', 'cover']
        importance_data = {}
        
        for imp_type in importance_types:
            try:
                scores = model.get_score(importance_type=imp_type)
                named_scores = {}
                for feat_key, score in scores.items():
                    idx = int(feat_key.replace('f', ''))
                    if idx < len(feature_names):
                        named_scores[feature_names[idx]] = score
                importance_data[imp_type] = named_scores
            except Exception as e:
                log.warning(f"Could not get {imp_type} importance: {e}")
        
        importance_file = model_dir / "feature_importance.json"
        with open(importance_file, 'w') as f:
            json.dump(importance_data, f, indent=2)
        
        # Log top features
        if 'gain' in importance_data:
            sorted_features = sorted(importance_data['gain'].items(), key=lambda x: x[1], reverse=True)
            log.info(f"  Top 5 features: {', '.join([f[0] for f in sorted_features[:5]])}")
    
    def save_model_cpu(self, model, feature_names):
        """Save CPU model for Triton inference."""
        model_dir = self.output_path / "fraud_xgboost_cpu"
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = version_dir / "xgboost.json"
        model.save_model(str(model_file))
        log.info(f"  CPU model saved: {model_file}")
        
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        self._export_feature_importance(model, feature_names, model_dir)
        
        # CPU config - uses KIND_CPU instance group
        config = f'''name: "fraud_xgboost_cpu"
backend: "fil"
max_batch_size: 32768
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {len(feature_names)} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    count: 2
    kind: KIND_CPU
  }}
]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "false" }}
  }}
]
'''
        config_file = model_dir / "config.pbtxt"
        with open(config_file, 'w') as f:
            f.write(config)
    
    def save_model_gpu(self, model, feature_names):
        """Save GPU model for Triton inference."""
        model_dir = self.output_path / "fraud_xgboost_gpu"
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = version_dir / "xgboost.json"
        model.save_model(str(model_file))
        log.info(f"  GPU model saved: {model_file}")
        
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        self._export_feature_importance(model, feature_names, model_dir)
        
        # GPU config - uses KIND_GPU instance group
        config = f'''name: "fraud_xgboost_gpu"
backend: "fil"
max_batch_size: 32768
input [
  {{
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ {len(feature_names)} ]
  }}
]
output [
  {{
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }}
]
instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [ 0 ]
  }}
]
parameters [
  {{
    key: "model_type"
    value: {{ string_value: "xgboost_json" }}
  }},
  {{
    key: "output_class"
    value: {{ string_value: "false" }}
  }}
]
'''
        config_file = model_dir / "config.pbtxt"
        with open(config_file, 'w') as f:
            f.write(config)
    
    def print_comparison(self, cpu_metrics, gpu_metrics):
        """Print performance comparison."""
        log.info("")
        log.info("=" * 70)
        log.info("PERFORMANCE COMPARISON")
        log.info("=" * 70)
        log.info("")
        log.info(f"  {'Stage':<20} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10}")
        log.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        
        stages = ['load', 'prep', 'train', 'eval']
        for stage in stages:
            cpu_t = self.cpu_times.get(stage, 0)
            gpu_t = self.gpu_times.get(stage, 0)
            speedup = cpu_t / gpu_t if gpu_t > 0 else 0
            stage_name = {
                'load': 'Data Loading',
                'prep': 'Data Preparation', 
                'train': 'Model Training',
                'eval': 'Evaluation'
            }.get(stage, stage)
            log.info(f"  {stage_name:<20} {cpu_t:<12.2f} {gpu_t:<12.2f} {speedup:<10.1f}x")
        
        log.info(f"  {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
        
        cpu_total = self.cpu_times.get('total', 0)
        gpu_total = self.gpu_times.get('total', 0)
        total_speedup = cpu_total / gpu_total if gpu_total > 0 else 0
        log.info(f"  {'TOTAL':<20} {cpu_total:<12.2f} {gpu_total:<12.2f} {total_speedup:<10.1f}x")
        
        log.info("")
        log.info("  Model Quality:")
        log.info(f"    CPU F1: {cpu_metrics['f1']:.4f} | GPU F1: {gpu_metrics['f1']:.4f}")
        log.info("=" * 70)
    
    def run(self, features_file: str = None):
        """Execute training pipeline - CPU then GPU."""
        filepath = self.load_features(features_file)
        
        # Phase 1: CPU Training
        cpu_model, cpu_features, cpu_metrics = self.train_cpu(filepath)
        gc.collect()
        
        # Phase 2: GPU Training
        gpu_model, gpu_features, gpu_metrics = self.train_gpu(filepath)
        free_gpu_memory()
        
        # Performance comparison
        self.print_comparison(cpu_metrics, gpu_metrics)
        
        # Save both models
        log.info("")
        log.info("Saving models to Triton repository...")
        self.save_model_cpu(cpu_model, cpu_features)
        self.save_model_gpu(gpu_model, gpu_features)
        
        # Save timing metadata
        meta = {
            "timestamp": datetime.now().isoformat(),
            "cpu_times": self.cpu_times,
            "gpu_times": self.gpu_times,
            "cpu_metrics": cpu_metrics,
            "gpu_metrics": gpu_metrics,
            "speedup": {
                stage: self.cpu_times.get(stage, 0) / self.gpu_times.get(stage, 1)
                for stage in ['load', 'prep', 'train', 'eval', 'total']
            }
        }
        with open(self.output_path / "training_comparison.json", 'w') as f:
            json.dump(meta, f, indent=2)
        
        log.info("")
        log.info("=" * 70)
        log.info("Model Repository Structure:")
        log.info(f"  {self.output_path}/")
        log.info(f"    fraud_xgboost_cpu/     # CPU inference (KIND_CPU)")
        log.info(f"      config.pbtxt")
        log.info(f"      1/xgboost.json")
        log.info(f"    fraud_xgboost_gpu/     # GPU inference (KIND_GPU)")
        log.info(f"      config.pbtxt")
        log.info(f"      1/xgboost.json")
        log.info("=" * 70)
        log.info("Training complete!")
        log.info("=" * 70)


def main():
    input_dir = os.getenv('PREP_OUTPUT_DIR', '/data/input')
    output_dir = os.getenv('FA_MOUNT', '/data/models')
    features_file = os.getenv('FEATURES_FILE', '') or None
    
    trainer = ModelTrainer(input_dir, output_dir)
    trainer.run(features_file)


if __name__ == "__main__":
    main()