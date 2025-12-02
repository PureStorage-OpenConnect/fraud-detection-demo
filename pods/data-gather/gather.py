#!/usr/bin/env python3
"""
Pod 1: Data Gather Service
Generates synthetic transaction data for fraud detection training and testing.
Based on NVIDIA Financial Fraud Detection Blueprint.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import boto3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TransactionDataGenerator:
    """Generate synthetic transaction data for fraud detection"""
    
    def __init__(self, fb_mount: str, s3_bucket: str = None):
        self.fb_mount = Path(fb_mount)
        self.raw_data_path = self.fb_mount / "raw_data"
        self.s3_bucket = s3_bucket
        self.s3_client = None
        
        # Create directories
        self.raw_data_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize S3 client if bucket specified
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
    
    def generate_users(self, num_users: int = 10000) -> pd.DataFrame:
        """Generate synthetic user profiles"""
        logger.info(f"Generating {num_users} user profiles...")
        
        users = pd.DataFrame({
            'user_id': range(num_users),
            'account_age_days': np.random.randint(1, 3650, num_users),
            'credit_limit': np.random.choice([1000, 2500, 5000, 10000, 25000], num_users),
            'risk_score': np.random.uniform(0, 1, num_users),
            'is_fraudster': np.random.choice([0, 1], num_users, p=[0.98, 0.02])
        })
        
        return users
    
    def generate_merchants(self, num_merchants: int = 1000) -> pd.DataFrame:
        """Generate synthetic merchant profiles"""
        logger.info(f"Generating {num_merchants} merchant profiles...")
        
        categories = ['grocery', 'restaurant', 'retail', 'online', 'gas', 
                     'travel', 'entertainment', 'healthcare', 'utilities']
        
        merchants = pd.DataFrame({
            'merchant_id': range(num_merchants),
            'category': np.random.choice(categories, num_merchants),
            'avg_transaction_amount': np.random.uniform(10, 500, num_merchants),
            'fraud_rate': np.random.uniform(0, 0.05, num_merchants)
        })
        
        return merchants
    
    def generate_transactions(
        self, 
        users: pd.DataFrame, 
        merchants: pd.DataFrame,
        num_transactions: int = 1000000
    ) -> pd.DataFrame:
        """Generate synthetic transaction data"""
        logger.info(f"Generating {num_transactions} transactions...")
        
        # Random selection of users and merchants
        user_ids = np.random.choice(users['user_id'].values, num_transactions)
        merchant_ids = np.random.choice(merchants['merchant_id'].values, num_transactions)
        
        # Generate timestamps (last 30 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        timestamps = [
            start_date + timedelta(seconds=np.random.randint(0, 30*24*60*60))
            for _ in range(num_transactions)
        ]
        
        # Generate transaction amounts
        amounts = np.random.lognormal(mean=4.0, sigma=1.5, size=num_transactions)
        amounts = np.clip(amounts, 1, 10000)
        
        transactions = pd.DataFrame({
            'transaction_id': range(num_transactions),
            'timestamp': timestamps,
            'user_id': user_ids,
            'merchant_id': merchant_ids,
            'amount': amounts,
            'currency': 'USD',
            'transaction_type': np.random.choice(['credit', 'debit'], num_transactions, p=[0.7, 0.3]),
            'channel': np.random.choice(['online', 'in-store', 'atm'], num_transactions, p=[0.5, 0.4, 0.1])
        })
        
        # Merge with user and merchant data to determine fraud labels
        transactions = transactions.merge(users[['user_id', 'is_fraudster']], on='user_id')
        transactions = transactions.merge(merchants[['merchant_id', 'fraud_rate']], on='merchant_id')
        
        # Determine if transaction is fraudulent
        fraud_probability = (
            transactions['is_fraudster'] * 0.8 + 
            transactions['fraud_rate'] * 0.2
        )
        transactions['is_fraud'] = (np.random.random(num_transactions) < fraud_probability).astype(int)
        
        # Drop helper columns
        transactions = transactions.drop(['is_fraudster', 'fraud_rate'], axis=1)
        
        # Sort by timestamp
        transactions = transactions.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Generated {len(transactions)} transactions with {transactions['is_fraud'].sum()} fraud cases ({transactions['is_fraud'].mean()*100:.2f}%)")
        
        return transactions
    
    def save_to_flashblade(self, data: pd.DataFrame, filename: str):
        """Save data to FlashBlade storage"""
        filepath = self.raw_data_path / filename
        logger.info(f"Saving data to FlashBlade: {filepath}")
        
        data.to_csv(filepath, index=False)
        
        file_size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"Successfully saved {len(data)} records ({file_size_mb:.2f} MB)")
    
    def archive_to_s3(self, filename: str):
        """Archive data to S3 for long-term storage"""
        if not self.s3_client:
            logger.warning("S3 client not initialized, skipping archive")
            return
        
        local_path = self.raw_data_path / filename
        s3_key = f"raw_archives/{filename}"
        
        try:
            logger.info(f"Archiving to S3: s3://{self.s3_bucket}/{s3_key}")
            self.s3_client.upload_file(str(local_path), self.s3_bucket, s3_key)
            logger.info("Successfully archived to S3")
        except Exception as e:
            logger.error(f"Failed to archive to S3: {e}")
    
    def run(self, num_transactions: int = 1000000):
        """Main execution method"""
        logger.info("=" * 60)
        logger.info("Pod 1: Data Gather Service - Starting")
        logger.info("=" * 60)
        
        # Generate synthetic data
        users = self.generate_users()
        merchants = self.generate_merchants()
        transactions = self.generate_transactions(users, merchants, num_transactions)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"transactions_{timestamp}.csv"
        
        # Save to FlashBlade
        self.save_to_flashblade(transactions, filename)
        
        # Archive to S3
        self.archive_to_s3(filename)
        
        # Save metadata
        metadata = {
            'filename': filename,
            'num_transactions': len(transactions),
            'num_fraud': int(transactions['is_fraud'].sum()),
            'fraud_rate': float(transactions['is_fraud'].mean()),
            'date_range_start': transactions['timestamp'].min().isoformat(),
            'date_range_end': transactions['timestamp'].max().isoformat(),
            'generated_at': datetime.now().isoformat()
        }
        
        metadata_file = self.raw_data_path / f"metadata_{timestamp}.json"
        pd.Series(metadata).to_json(metadata_file)
        
        logger.info("=" * 60)
        logger.info("Pod 1: Data Gather Service - Complete")
        logger.info(f"Output: {filename}")
        logger.info("=" * 60)
        
        return filename

def main():
    """Main entry point"""
    # Get configuration from environment variables
    fb_mount = os.getenv('FB_MOUNT', '/mnt/fsaai-shared/ebiser')
    s3_bucket = os.getenv('S3_BUCKET')
    num_transactions = int(os.getenv('NUM_TRANSACTIONS', '1000000'))
    
    # Create generator and run
    generator = TransactionDataGenerator(fb_mount, s3_bucket)
    generator.run(num_transactions)

if __name__ == "__main__":
    main()
