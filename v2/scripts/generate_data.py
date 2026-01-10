#!/usr/bin/env python3
"""
Offline Data Generator for Fraud Detection Demo v2

Generates synthetic credit card transactions and saves identical copies
to both CPU and GPU data directories for fair comparison.

Usage:
    python generate_data.py --rows 1000000 --output-dir /data

This will create:
    /data/cpu/transactions.parquet
    /data/gpu/transactions.parquet
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Transaction categories with realistic distribution
CATEGORIES = [
    'gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
    'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
    'kids_pets', 'travel', 'health_fitness', 'personal_care'
]
CATEGORY_WEIGHTS = np.array([
    0.243, 0.226, 0.106, 0.092, 0.083, 0.080, 0.079, 0.037,
    0.019, 0.012, 0.008, 0.005, 0.005, 0.004
])

# US states weighted by population
US_STATES = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
    'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
    'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY'
]
STATE_WEIGHTS = np.array([
    0.118, 0.087, 0.065, 0.059, 0.039, 0.038, 0.035, 0.032, 0.031, 0.030,
    0.027, 0.026, 0.023, 0.022, 0.021, 0.021, 0.020, 0.018, 0.018, 0.018,
    0.017, 0.017, 0.015, 0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.010,
    0.010, 0.009, 0.009, 0.009, 0.009, 0.006, 0.006, 0.006, 0.005, 0.004,
    0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002
])

# Sample data pools (simplified - no Faker dependency for offline generation)
FIRST_NAMES = ['James', 'Mary', 'John', 'Patricia', 'Robert', 'Jennifer', 'Michael',
               'Linda', 'William', 'Elizabeth', 'David', 'Barbara', 'Richard', 'Susan',
               'Joseph', 'Jessica', 'Thomas', 'Sarah', 'Charles', 'Karen', 'Christopher',
               'Nancy', 'Daniel', 'Lisa', 'Matthew', 'Betty', 'Anthony', 'Margaret',
               'Mark', 'Sandra', 'Donald', 'Ashley', 'Steven', 'Kimberly', 'Paul', 'Emily']

LAST_NAMES = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
              'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
              'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
              'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark', 'Ramirez']

MERCHANTS = ['Amazon', 'Walmart', 'Target', 'Costco', 'Home Depot', 'Best Buy',
             'Kroger', 'Walgreens', 'CVS', 'Starbucks', 'McDonalds', 'Shell',
             'Exxon', 'Chevron', 'Apple Store', 'Netflix', 'Uber', 'Lyft',
             'DoorDash', 'Grubhub', 'Whole Foods', 'Trader Joes', 'Safeway']

CITIES = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia',
          'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Austin', 'Jacksonville',
          'Fort Worth', 'Columbus', 'Charlotte', 'San Francisco', 'Indianapolis',
          'Seattle', 'Denver', 'Boston', 'Portland', 'Miami', 'Atlanta']

JOBS = ['Software Engineer', 'Teacher', 'Nurse', 'Sales Manager', 'Accountant',
        'Marketing Manager', 'Doctor', 'Lawyer', 'Electrician', 'Mechanic',
        'Chef', 'Designer', 'Analyst', 'Consultant', 'Project Manager']


def log(msg: str):
    """Print timestamped log message."""
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def generate_transactions(n_rows: int, seed: int = 42, fraud_rate: float = 0.005) -> pa.Table:
    """
    Generate synthetic credit card transactions.

    Args:
        n_rows: Number of transactions to generate
        seed: Random seed for reproducibility
        fraud_rate: Fraction of transactions that are fraudulent (default 0.5%)

    Returns:
        PyArrow Table with transaction data
    """
    log(f"Generating {n_rows:,} transactions (seed={seed}, fraud_rate={fraud_rate*100:.1f}%)...")

    rng = np.random.default_rng(seed)

    # Normalize weights
    cat_weights = CATEGORY_WEIGHTS / CATEGORY_WEIGHTS.sum()
    state_weights = STATE_WEIGHTS / STATE_WEIGHTS.sum()

    # Generate base timestamp (2024)
    base_time = 1704067200  # 2024-01-01 00:00:00 UTC
    unix_times = rng.integers(base_time, base_time + 31536000, size=n_rows, dtype=np.int64)

    # Geographic data (US bounds)
    lats = rng.uniform(25.0, 48.0, n_rows).astype(np.float32)
    longs = rng.uniform(-125.0, -70.0, n_rows).astype(np.float32)
    merch_lats = lats + rng.normal(0, 0.5, n_rows).astype(np.float32)
    merch_longs = longs + rng.normal(0, 0.5, n_rows).astype(np.float32)

    # Transaction amounts (lognormal distribution)
    amts = np.clip(np.abs(rng.lognormal(3.5, 1.5, n_rows)), 1.0, 25000.0).astype(np.float32)

    # Build data dictionary
    data = {
        'trans_date_trans_time': (np.datetime64('1970-01-01') + unix_times.astype('timedelta64[s]')).astype(str),
        'cc_num': rng.integers(4000000000000000, 5000000000000000, size=n_rows, dtype=np.int64),
        'merchant': rng.choice(MERCHANTS, size=n_rows),
        'category': np.array(CATEGORIES)[rng.choice(len(CATEGORIES), size=n_rows, p=cat_weights)],
        'amt': amts,
        'first': rng.choice(FIRST_NAMES, size=n_rows),
        'last': rng.choice(LAST_NAMES, size=n_rows),
        'gender': np.where(rng.random(n_rows) < 0.5, 'M', 'F'),
        'street': np.array([f"{rng.integers(100, 9999)} Main St" for _ in range(n_rows)]),
        'city': rng.choice(CITIES, size=n_rows),
        'state': np.array(US_STATES)[rng.choice(len(US_STATES), size=n_rows, p=state_weights)],
        'zip': rng.integers(10000, 99999, size=n_rows, dtype=np.int32),
        'lat': lats,
        'long': longs,
        'city_pop': rng.integers(1000, 2000000, size=n_rows, dtype=np.int32),
        'job': rng.choice(JOBS, size=n_rows),
        'dob': np.array([f"{rng.integers(1940, 2005)}-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"
                        for _ in range(n_rows)]),
        'trans_num': np.array([f"{rng.integers(0, 2**63):016x}" for _ in range(n_rows)]),
        'unix_time': unix_times,
        'merch_lat': merch_lats,
        'merch_long': merch_longs,
        'is_fraud': (rng.random(n_rows) < fraud_rate).astype(np.int8),
        'merch_zipcode': rng.integers(10000, 99999, size=n_rows, dtype=np.int32),
    }

    log(f"  Created {n_rows:,} rows with {data['is_fraud'].sum():,} fraudulent transactions")

    return pa.Table.from_pydict(data)


def save_data(table: pa.Table, output_dir: Path):
    """
    Save transaction data to both CPU and GPU directories.

    Args:
        table: PyArrow Table with transaction data
        output_dir: Base output directory
    """
    cpu_dir = output_dir / 'cpu'
    gpu_dir = output_dir / 'gpu'

    # Create directories
    cpu_dir.mkdir(parents=True, exist_ok=True)
    gpu_dir.mkdir(parents=True, exist_ok=True)

    cpu_file = cpu_dir / 'transactions.parquet'
    gpu_file = gpu_dir / 'transactions.parquet'

    # Write to CPU directory
    log(f"Writing to {cpu_file}...")
    pq.write_table(table, cpu_file, compression='snappy')
    cpu_size = cpu_file.stat().st_size / (1024**2)
    log(f"  Wrote {cpu_size:.1f} MB")

    # Copy to GPU directory (identical data)
    log(f"Copying to {gpu_file}...")
    shutil.copy2(cpu_file, gpu_file)
    log(f"  Copied {cpu_size:.1f} MB")

    return cpu_file, gpu_file


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic transaction data for fraud detection demo')
    parser.add_argument('--rows', type=int, default=1_000_000,
                        help='Number of transactions to generate (default: 1,000,000)')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory (default: ./data)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--fraud-rate', type=float, default=0.005,
                        help='Fraud rate as decimal (default: 0.005 = 0.5%%)')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    log("=" * 70)
    log("Fraud Detection Demo v2 - Data Generator")
    log("=" * 70)
    log(f"Rows:       {args.rows:,}")
    log(f"Output:     {output_dir.absolute()}")
    log(f"Seed:       {args.seed}")
    log(f"Fraud Rate: {args.fraud_rate*100:.1f}%")
    log("-" * 70)

    # Generate data
    table = generate_transactions(args.rows, seed=args.seed, fraud_rate=args.fraud_rate)

    # Save to both directories
    cpu_file, gpu_file = save_data(table, output_dir)

    log("-" * 70)
    log("Data generation complete!")
    log(f"  CPU data: {cpu_file}")
    log(f"  GPU data: {gpu_file}")
    log("=" * 70)


if __name__ == '__main__':
    main()
