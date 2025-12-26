#!/usr/bin/env python3
"""
Pod 3: Model Build Service
Train GNN and XGBoost models for fraud detection
Based on NVIDIA Financial Fraud Detection Blueprint
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime

import cudf
import cupy as cp
import xgboost as xgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv
import boto3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GNNModel(nn.Module):
    """Graph Neural Network for fraud detection embeddings"""
    
    def __init__(self, in_channels, hidden_channels=64, out_channels=32):
        super(GNNModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        return x

class ModelBuildService:
    """Train GNN and XGBoost models"""
    
    def __init__(self, fb_mount: str, fa_mount: str, s3_bucket: str = None, prep_output_dir: str = None):
        self.fb_mount = Path(fb_mount)
        self.fa_mount = Path(fa_mount)
        # Allow override via parameter or use default
        if prep_output_dir:
            self.prep_output_path = Path(prep_output_dir)
        else:
            self.prep_output_path = self.fb_mount / "prep-output"
        self.model_repo_path = self.fa_mount / "model_repository"
        self.s3_bucket = s3_bucket
        self.s3_client = None
        
        # Create model repository
        self.model_repo_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 if bucket specified
        if self.s3_bucket:
            try:
                s3_endpoint = os.getenv('S3_ENDPOINT')
                s3_access_key = os.getenv('S3_ACCESS_KEY')
                s3_secret_key = os.getenv('S3_SECRET_KEY')
                
                if s3_endpoint and s3_access_key and s3_secret_key:
                    self.s3_client = boto3.client(
                        's3',
                        endpoint_url=s3_endpoint,
                        aws_access_key_id=s3_access_key,
                        aws_secret_access_key=s3_secret_key
                    )
                    logger.info(f"S3 client initialized for endpoint: {s3_endpoint}")
                else:
                    logger.warning("S3 credentials not configured")
            except Exception as e:
                logger.warning(f"Could not initialize S3 client: {e}")
        
        logger.info(f"Model Build Service initialized")
        logger.info(f"Input path: {self.prep_output_path}")
        logger.info(f"Model repository: {self.model_repo_path}")
    
    def load_prepared_data(self, features_file: str = None):
        """Load prepared features from Pod 2"""
        if features_file is None:
            # Find most recent features file
            parquet_files = sorted(self.prep_output_path.glob("features_*.parquet"))
            if not parquet_files:
                raise FileNotFoundError("No feature files found")
            features_file = parquet_files[-1].name
        
        filepath = self.prep_output_path / features_file
        logger.info(f"Loading features from: {filepath}")
        
        df = cudf.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        return df
    
    def prepare_training_data(self, df: cudf.DataFrame):
        """Prepare data for training"""
        logger.info("Preparing training data...")
        
        # Feature columns (exclude IDs and target)
        exclude_cols = ['transaction_id', 'timestamp', 'user_id', 'merchant_id', 
                       'currency', 'transaction_type', 'channel', 'is_fraud']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        df = df.fillna(0)
        
        # Split train/test (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Convert to numpy/cupy
        X_train = train_df[feature_cols].to_cupy()
        y_train = train_df['is_fraud'].to_cupy()
        X_test = test_df[feature_cols].to_cupy()
        y_test = test_df['is_fraud'].to_cupy()
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Fraud rate (train): {float(y_train.mean()):.4f}")
        logger.info(f"Fraud rate (test): {float(y_test.mean()):.4f}")
        
        return X_train, y_train, X_test, y_test, feature_cols
    
    def train_gnn_embeddings(self, df: cudf.DataFrame):
        """Train GNN to generate embeddings (placeholder)"""
        logger.info("Training GNN model...")
        
        # Note: Full GNN training requires PyTorch Geometric setup
        # This is a simplified placeholder
        
        num_users = df['user_id'].max() + 1
        embedding_dim = 32
        
        # Create random embeddings as placeholder
        embeddings = cp.random.randn(num_users, embedding_dim).astype(cp.float32)
        
        logger.info(f"Generated GNN embeddings: {embeddings.shape}")
        
        return embeddings
    
    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost classifier"""
        logger.info("Training XGBoost model...")
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
            'predictor': 'gpu_predictor'
        }
        
        # Train model
        evallist = [(dtrain, 'train'), (dtest, 'test')]
        num_rounds = 100
        
        model = xgb.train(
            params,
            dtrain,
            num_rounds,
            evals=evallist,
            early_stopping_rounds=10,
            verbose_eval=10
        )
        
        # Evaluate
        y_pred = model.predict(dtest)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = float((y_pred_binary == y_test).sum() / len(y_test))
        
        logger.info(f"Model accuracy: {accuracy:.4f}")
        
        return model
    
    def save_xgboost_model(self, model, feature_names):
        """Save XGBoost model for Triton"""
        model_name = "fraud_xgboost"
        model_path = self.model_repo_path / model_name / "1"
        config_path = self.model_repo_path / model_name
        
        # Create directories
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_path / "model.json"
        model.save_model(str(model_file))
        logger.info(f"Saved XGBoost model to: {model_file}")
        
        # Create Triton config
        config = {
            "name": model_name,
            "backend": "fil",
            "max_batch_size": 32768,
            "input": [
                {
                    "name": "input__0",
                    "data_type": "TYPE_FP32",
                    "dims": [len(feature_names)]
                }
            ],
            "output": [
                {
                    "name": "output__0",
                    "data_type": "TYPE_FP32",
                    "dims": [1]
                }
            ],
            "instance_group": [
                {
                    "count": 1,
                    "kind": "KIND_GPU",
                    "gpus": [0]
                }
            ]
        }
        
        config_file = config_path / "config.pbtxt"
        with open(config_file, 'w') as f:
            f.write(self._dict_to_pbtxt(config))
        
        logger.info(f"Saved Triton config to: {config_file}")
    
    def _dict_to_pbtxt(self, d, indent=0):
        """Convert dict to protobuf text format"""
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append('  ' * indent + f'{key} {{')
                lines.append(self._dict_to_pbtxt(value, indent + 1))
                lines.append('  ' * indent + '}')
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        lines.append('  ' * indent + f'{key} {{')
                        lines.append(self._dict_to_pbtxt(item, indent + 1))
                        lines.append('  ' * indent + '}')
                    else:
                        lines.append('  ' * indent + f'{key}: {self._format_value(item)}')
            else:
                lines.append('  ' * indent + f'{key}: {self._format_value(value)}')
        return '\n'.join(lines)
    
    def _format_value(self, value):
        """Format value for pbtxt"""
        if isinstance(value, str):
            return f'"{value}"'
        return str(value)
    
    def version_model_to_s3(self, model_name: str):
        """Archive model version to S3"""
        if not self.s3_client:
            logger.warning("S3 client not initialized, skipping versioning")
            return
        
        import tarfile
        
        model_path = self.model_repo_path / model_name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tar_filename = f"{model_name}_v{timestamp}.tar.gz"
        tar_path = self.model_repo_path / tar_filename
        
        # Create tarball
        with tarfile.open(tar_path, 'w:gz') as tar:
            tar.add(model_path, arcname=model_name)
        
        # Upload to S3
        s3_key = f"model_versions/{tar_filename}"
        try:
            logger.info(f"Uploading model to S3: s3://{self.s3_bucket}/{s3_key}")
            self.s3_client.upload_file(str(tar_path), self.s3_bucket, s3_key)
            logger.info("Successfully versioned model to S3")
            
            # Clean up local tarball
            tar_path.unlink()
        except Exception as e:
            logger.error(f"Failed to version model to S3: {e}")
    
    def run(self, features_file: str = None):
        """Main execution method"""
        logger.info("=" * 60)
        logger.info("Pod 3: Model Build Service - Starting")
        logger.info("=" * 60)
        
        # Load data
        df = self.load_prepared_data(features_file)
        
        # Prepare training data
        X_train, y_train, X_test, y_test, feature_names = self.prepare_training_data(df)
        
        # Train models
        gnn_embeddings = self.train_gnn_embeddings(df)
        xgb_model = self.train_xgboost(X_train, y_train, X_test, y_test)
        
        # Save models
        self.save_xgboost_model(xgb_model, feature_names)
        
        # Version to S3
        self.version_model_to_s3("fraud_xgboost")
        
        logger.info("=" * 60)
        logger.info("Pod 3: Model Build Service - Complete")
        logger.info(f"Model repository: {self.model_repo_path}")
        logger.info("=" * 60)

def main():
    """Main entry point"""
    fb_mount = os.getenv('FB_MOUNT', '/mnt/fsaai-shared/ebiser')
    fa_mount = os.getenv('FA_MOUNT', '~/ebiser/nvidia.financial.fraud.detection')
    s3_bucket = os.getenv('S3_BUCKET')
    features_file = os.getenv('FEATURES_FILE', None)
    prep_output_dir = os.getenv('PREP_OUTPUT_DIR', None)  # Optional override
    
    service = ModelBuildService(fb_mount, fa_mount, s3_bucket, prep_output_dir)
    service.run(features_file)

if __name__ == "__main__":
    main()