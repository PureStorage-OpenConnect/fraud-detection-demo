#!/usr/bin/env python3
"""
Pod 1: Data Generator
Generates synthetic credit card transactions for fraud detection demo.
Target: 2+ GB/s write throughput to Pure Storage FlashBlade.

Uses pool-based generation with NumPy for maximum performance.
"""

import os
import sys
import time
import signal
import subprocess
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np

STOP_FLAG = False

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

# String pool sizes for Faker-generated data
POOL_SIZES = {
    'first': 10_000,
    'last': 15_000,
    'street': 50_000,
    'city': 10_000,
    'merchant': 20_000,
    'job': 5_000,
    'trans_num': 100_000,
    'dob': 25_000,
}


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {msg}", flush=True)


def signal_handler(signum, frame):
    global STOP_FLAG
    log("Shutdown signal received")
    STOP_FLAG = True


def generate_pools(output_path: Path) -> Path:
    """Generate string pools using Faker (one-time startup cost)."""
    from faker import Faker
    
    log("Generating string pools...")
    fake = Faker()
    Faker.seed(42)
    
    pools = {
        'first': np.array([fake.first_name() for _ in range(POOL_SIZES['first'])]),
        'last': np.array([fake.last_name() for _ in range(POOL_SIZES['last'])]),
        'street': np.array([fake.street_address() for _ in range(POOL_SIZES['street'])]),
        'city': np.array([fake.city() for _ in range(POOL_SIZES['city'])]),
        'merchant': np.array([f"{fake.company().replace(',', '')} {fake.company_suffix()}" 
                             for _ in range(POOL_SIZES['merchant'])]),
        'job': np.array([fake.job().replace(',', ' ') for _ in range(POOL_SIZES['job'])]),
        'trans_num': np.array([fake.uuid4().replace('-', '') for _ in range(POOL_SIZES['trans_num'])]),
        'dob': np.array([fake.date_of_birth(minimum_age=18, maximum_age=85).strftime('%Y-%m-%d') 
                        for _ in range(POOL_SIZES['dob'])]),
        'categories': np.array(CATEGORIES),
        'category_weights': CATEGORY_WEIGHTS / CATEGORY_WEIGHTS.sum(),
        'states': np.array(US_STATES),
        'state_weights': STATE_WEIGHTS / STATE_WEIGHTS.sum(),
    }
    
    pools_file = output_path / "_pools.pkl"
    with open(pools_file, 'wb') as f:
        pickle.dump(pools, f)
    
    total = sum(POOL_SIZES.values())
    log(f"  Created {total:,} pooled values")
    return pools_file


# Worker script (runs in subprocess for true parallelism)
WORKER_SCRIPT = '''
import sys, pickle, time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def generate_chunk(pools, n, rng, fraud_rate, base_time):
    """Generate n transactions using vectorized operations."""
    # Timestamps throughout 2024
    unix_times = rng.integers(base_time, base_time + 31536000, size=n, dtype=np.int64)
    
    # Geographic data (US bounds)
    lats = rng.uniform(25.0, 48.0, n).astype(np.float32)
    longs = rng.uniform(-125.0, -70.0, n).astype(np.float32)
    merch_lats = lats + rng.normal(0, 0.5, n).astype(np.float32)
    merch_longs = longs + rng.normal(0, 0.5, n).astype(np.float32)
    
    # Transaction amounts (lognormal distribution)
    amts = np.clip(np.abs(rng.lognormal(3.5, 1.5, n)), 1.0, 25000.0).astype(np.float32)
    
    # Pool sampling
    idx = lambda pool: rng.integers(0, len(pools[pool]), n)
    
    return {
        'trans_date_trans_time': (np.datetime64('1970-01-01') + unix_times.astype('timedelta64[s]')).astype(str),
        'cc_num': rng.integers(4000000000000000, 5000000000000000, size=n, dtype=np.int64),
        'merchant': pools['merchant'][idx('merchant')],
        'category': pools['categories'][rng.choice(len(pools['categories']), size=n, p=pools['category_weights'])],
        'amt': amts,
        'first': pools['first'][idx('first')],
        'last': pools['last'][idx('last')],
        'gender': np.where(rng.random(n) < 0.5, 'M', 'F'),
        'street': pools['street'][idx('street')],
        'city': pools['city'][idx('city')],
        'state': pools['states'][rng.choice(len(pools['states']), size=n, p=pools['state_weights'])],
        'zip': rng.integers(10000, 99999, size=n, dtype=np.int32),
        'lat': lats,
        'long': longs,
        'city_pop': rng.integers(1000, 2000000, size=n, dtype=np.int32),
        'job': pools['job'][idx('job')],
        'dob': pools['dob'][idx('dob')],
        'trans_num': pools['trans_num'][idx('trans_num')],
        'unix_time': unix_times,
        'merch_lat': merch_lats,
        'merch_long': merch_longs,
        'is_fraud': (rng.random(n) < fraud_rate).astype(np.int8),
        'merch_zipcode': rng.integers(10000, 99999, size=n, dtype=np.int32),
    }

# Parse args
worker_id = int(sys.argv[1])
output_dir = Path(sys.argv[2])
chunk_size = int(sys.argv[3])
duration = int(sys.argv[4])
pools_file = sys.argv[5]
fraud_rate = float(sys.argv[6])

# Load pools and init RNG
with open(pools_file, 'rb') as f:
    pools = pickle.load(f)
rng = np.random.default_rng(seed=worker_id * 54321 + int(time.time() * 1000) % 100000)
base_time = 1704067200  # 2024-01-01

# Generate until duration expires
start = time.time()
file_count = 0
try:
    while (time.time() - start) < duration:
        data = generate_chunk(pools, chunk_size, rng, fraud_rate, base_time)
        table = pa.Table.from_pydict(data)
        output_file = output_dir / f"worker_{worker_id:03d}_{file_count:05d}.parquet"
        pq.write_table(table, output_file, compression=None)
        file_count += 1
except Exception as e:
    pass  # Silently exit on error
'''


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration
    output_dir = Path(os.getenv('OUTPUT_DIR', '/data/output'))
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration = int(os.getenv('DURATION_SECONDS', '60'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '1000000'))
    fraud_rate = float(os.getenv('FRAUD_RATE', '0.005'))
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_path = output_dir / f"run_{timestamp}"
    run_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: Financial Fraud Data Generator")
    log("=" * 70)
    log(f"Output:   {run_path}")
    log(f"Workers:  {num_workers}")
    log(f"Duration: {duration}s")
    log(f"Chunk:    {chunk_size:,} rows")
    log(f"Fraud:    {fraud_rate*100:.1f}%")
    log("-" * 70)
    
    # Generate string pools
    pools_file = generate_pools(run_path)
    
    # Launch worker processes
    log(f"Launching {num_workers} workers...")
    processes = []
    for i in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', WORKER_SCRIPT, 
             str(i), str(run_path), str(chunk_size), str(duration), 
             str(pools_file), str(fraud_rate)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        processes.append(p)
    
    # Monitor progress
    start_time = time.time()
    last_bytes = 0
    last_time = start_time
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        running = sum(1 for p in processes if p.poll() is None)
        
        if elapsed >= duration + 10 or running == 0:
            break
        
        if time.time() - last_time >= 5.0:
            files = list(run_path.glob("worker_*.parquet"))
            current_bytes = sum(f.stat().st_size for f in files) if files else 0
            interval = time.time() - last_time
            speed = ((current_bytes - last_bytes) / (1024**3)) / interval
            total_gb = current_bytes / (1024**3)
            
            log(f"[{elapsed:5.0f}s] Files: {len(files):5d} | "
                f"Size: {total_gb:6.2f} GB | Speed: {speed:5.2f} GB/s | Workers: {running}")
            
            last_bytes = current_bytes
            last_time = time.time()
        
        time.sleep(1)
    
    # Cleanup
    for p in processes:
        if p.poll() is None:
            p.terminate()
    for p in processes:
        try:
            p.wait(timeout=5)
        except:
            p.kill()
    
    # Final stats
    pools_file.unlink(missing_ok=True)
    files = list(run_path.glob("worker_*.parquet"))
    total_bytes = sum(f.stat().st_size for f in files) if files else 0
    total_time = time.time() - start_time
    
    log("=" * 70)
    log(f"COMPLETE: {len(files):,} files | {total_bytes/(1024**3):.2f} GB | "
        f"{(total_bytes/(1024**3))/total_time:.2f} GB/s avg")
    log("=" * 70)


if __name__ == "__main__":
    main()