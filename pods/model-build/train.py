#!/usr/bin/env python3
"""
Pod 3: Model Training
GPU-accelerated XGBoost for fraud classification.
Outputs Triton-compatible model repository.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

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


class ModelTrainer:
    """XGBoost model trainer with Triton output."""
    
    def __init__(self, input_dir: str, output_dir: str):
        self.input_path = Path(input_dir)
        self.output_path = Path(output_dir) / "model_repository"
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        log.info("=" * 60)
        log.info("Pod 3: Model Training (XGBoost GPU)")
        log.info("=" * 60)
        log.info(f"Input:  {self.input_path}")
        log.info(f"Output: {self.output_path}")
    
    def load_features(self, features_file: str = None) -> cudf.DataFrame:
        """Load prepared features from parquet."""
        if features_file:
            filepath = self.input_path / features_file
        else:
            # Find most recent features file
            files = sorted(self.input_path.glob("features_*.parquet"))
            if not files:
                raise FileNotFoundError("No feature files found")
            filepath = files[-1]
        
        log.info(f"Loading: {filepath.name}")
        df = cudf.read_parquet(filepath)
        log.info(f"  {len(df):,} records, {len(df.columns)} columns")
        return df
    
    def prepare_data(self, df: cudf.DataFrame):
        """Prepare train/test split."""
        # Determine available features
        available = [c for c in FEATURE_COLUMNS if c in df.columns]
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        
        if missing:
            log.warning(f"Missing features: {missing}")
        log.info(f"Using {len(available)} features")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Subsample if too large (GPU memory limit ~10M records)
        max_records = 10_000_000
        if len(df) > max_records:
            log.info(f"Subsampling {max_records:,} from {len(df):,} records")
            # Stratified sampling to preserve fraud ratio
            fraud_df = df[df['is_fraud'] == 1]
            normal_df = df[df['is_fraud'] == 0]
            
            fraud_ratio = len(fraud_df) / len(df)
            n_fraud = int(max_records * fraud_ratio)
            n_normal = max_records - n_fraud
            
            fraud_sample = fraud_df.sample(n=min(n_fraud, len(fraud_df)), random_state=42)
            normal_sample = normal_df.sample(n=min(n_normal, len(normal_df)), random_state=42)
            
            df = cudf.concat([fraud_sample, normal_sample], ignore_index=True)
            df = df.sample(frac=1, random_state=42)  # Shuffle
        
        # Train/test split (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        X_train = train_df[available].to_cupy()
        y_train = train_df['is_fraud'].to_cupy()
        X_test = test_df[available].to_cupy()
        y_test = test_df['is_fraud'].to_cupy()
        
        log.info(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
        log.info(f"Fraud rate: {float(y_train.mean())*100:.2f}%")
        
        return X_train, y_train, X_test, y_test, available
    
    def train(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier."""
        log.info("Training XGBoost...")
        
        # Handle class imbalance
        fraud_count = float(y_train.sum())
        normal_count = len(y_train) - fraud_count
        scale_pos_weight = normal_count / max(fraud_count, 1)
        log.info(f"  Scale pos weight: {scale_pos_weight:.2f}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # XGBoost parameters
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
        
        # Train
        model = xgb.train(
            params, dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        accuracy = float((y_pred_binary == cp.asnumpy(y_test)).mean())
        
        # Fraud class metrics
        fraud_mask = cp.asnumpy(y_test) == 1
        if fraud_mask.sum() > 0:
            tp = ((y_pred_binary == 1) & fraud_mask).sum()
            fp = ((y_pred_binary == 1) & ~fraud_mask).sum()
            fn = ((y_pred_binary == 0) & fraud_mask).sum()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            
            log.info(f"Accuracy:  {accuracy:.4f}")
            log.info(f"Precision: {precision:.4f}")
            log.info(f"Recall:    {recall:.4f}")
            log.info(f"F1:        {f1:.4f}")
        
        return model
    
    def save_model(self, model, feature_names):
        """Save model for Triton inference."""
        model_dir = self.output_path / "fraud_xgboost"
        version_dir = model_dir / "1"
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (FIL backend expects xgboost.json)
        model_file = version_dir / "xgboost.json"
        model.save_model(str(model_file))
        log.info(f"Model saved: {model_file}")
        
        # Save feature names
        with open(model_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        
        # Create Triton config
        config = f'''name: "fraud_xgboost"
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
        log.info(f"Config saved: {config_file}")
    
    def run(self, features_file: str = None):
        """Execute training pipeline."""
        df = self.load_features(features_file)
        X_train, y_train, X_test, y_test, feature_names = self.prepare_data(df)
        
        del df
        cp.get_default_memory_pool().free_all_blocks()
        
        model = self.train(X_train, y_train, X_test, y_test)
        self.save_model(model, feature_names)
        
        log.info("=" * 60)
        log.info("Training complete")
        log.info("=" * 60)


def main():
    input_dir = os.getenv('PREP_OUTPUT_DIR', '/data/input')
    output_dir = os.getenv('FA_MOUNT', '/data/models')
    features_file = os.getenv('FEATURES_FILE', '') or None
    
    trainer = ModelTrainer(input_dir, output_dir)
    trainer.run(features_file)


if __name__ == "__main__":
    main()