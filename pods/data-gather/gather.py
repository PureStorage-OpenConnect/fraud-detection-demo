#!/usr/bin/env python3
"""
Pod 1: Data Gather Service - High-throughput credit card transaction generator
Uses pool-based generation for maximum FlashBlade write performance (target: 2-3 GB/s)
"""

import os
import sys
import time
import signal
import subprocess
import pickle
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

STOP_FLAG = False

# Category distribution from actual fraud dataset
CATEGORIES = [
    'gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
    'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
    'kids_pets', 'travel', 'health_fitness', 'personal_care'
]
CATEGORY_WEIGHTS = [
    0.243, 0.226, 0.106, 0.092, 0.083, 0.080, 0.079, 0.037,
    0.019, 0.012, 0.008, 0.005, 0.005, 0.004
]

# US states weighted by population
US_STATES = [
    'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
    'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
    'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
    'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
    'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY'
]
STATE_WEIGHTS = [
    0.118, 0.087, 0.065, 0.059, 0.039, 0.038, 0.035, 0.032, 0.031, 0.030,
    0.027, 0.026, 0.023, 0.022, 0.021, 0.021, 0.020, 0.018, 0.018, 0.018,
    0.017, 0.017, 0.015, 0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.010,
    0.010, 0.009, 0.009, 0.009, 0.009, 0.006, 0.006, 0.006, 0.005, 0.004,
    0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002
]

POOL_SIZES = {
    'first': 10_000,
    'last': 15_000,
    'street': 50_000,
    'city': 10_000,
    'merchant': 20_000,
    'job': 5_000,
}


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def signal_handler(signum, frame):
    global STOP_FLAG
    log(f"Received signal {signum}, stopping...")
    STOP_FLAG = True


def generate_pools(output_path: Path) -> Path:
    """Generate string pools using Faker - runs ONCE at startup."""
    from faker import Faker
    
    fake = Faker('en_US')
    Faker.seed(42)  # Reproducible pools
    
    log("Generating string pools (one-time startup cost)...")
    start = time.time()
    
    pools = {
        'first': [fake.first_name() for _ in range(POOL_SIZES['first'])],
        'last': [fake.last_name() for _ in range(POOL_SIZES['last'])],
        'street': [fake.street_address() for _ in range(POOL_SIZES['street'])],
        'city': [fake.city() for _ in range(POOL_SIZES['city'])],
        'merchant': [fake.company() for _ in range(POOL_SIZES['merchant'])],
        'job': [fake.job() for _ in range(POOL_SIZES['job'])],
    }
    
    pools_file = output_path / "_pools.pkl"
    with open(pools_file, 'wb') as f:
        pickle.dump(pools, f)
    
    elapsed = time.time() - start
    total = sum(len(v) for v in pools.values())
    log(f"Pools ready: {total:,} values in {elapsed:.1f}s")
    
    return pools_file


# Worker script - runs as subprocess for true parallelism (no GIL)
WORKER_SCRIPT = '''
import sys
import time
import pickle
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

# Categories and states passed via pickle
CATEGORIES = ['gas_transport', 'grocery_pos', 'misc_pos', 'misc_net', 'shopping_net',
              'shopping_pos', 'grocery_net', 'entertainment', 'food_dining', 'home',
              'kids_pets', 'travel', 'health_fitness', 'personal_care']
CATEGORY_WEIGHTS = [0.243, 0.226, 0.106, 0.092, 0.083, 0.080, 0.079, 0.037,
                    0.019, 0.012, 0.008, 0.005, 0.005, 0.004]
US_STATES = ['CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
             'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
             'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
             'IA', 'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'ID', 'WV', 'HI',
             'NH', 'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY']
STATE_WEIGHTS = [0.118, 0.087, 0.065, 0.059, 0.039, 0.038, 0.035, 0.032, 0.031, 0.030,
                 0.027, 0.026, 0.023, 0.022, 0.021, 0.021, 0.020, 0.018, 0.018, 0.018,
                 0.017, 0.017, 0.015, 0.015, 0.014, 0.013, 0.013, 0.012, 0.011, 0.010,
                 0.010, 0.009, 0.009, 0.009, 0.009, 0.006, 0.006, 0.006, 0.005, 0.004,
                 0.004, 0.004, 0.003, 0.003, 0.003, 0.003, 0.002, 0.002, 0.002, 0.002]

def generate_chunk(n, pools, rng, categories, cat_weights, states, state_weights, fraud_rate):
    """Generate n rows using vectorized NumPy operations."""
    
    # String columns: index into pools (vectorized O(n))
    first = pools['first'][rng.integers(0, len(pools['first']), n)]
    last = pools['last'][rng.integers(0, len(pools['last']), n)]
    merchant = pools['merchant'][rng.integers(0, len(pools['merchant']), n)]
    street = pools['street'][rng.integers(0, len(pools['street']), n)]
    city = pools['city'][rng.integers(0, len(pools['city']), n)]
    job = pools['job'][rng.integers(0, len(pools['job']), n)]
    
    # Categorical: weighted random choice (vectorized)
    category = categories[rng.choice(len(categories), n, p=cat_weights)]
    state = states[rng.choice(len(states), n, p=state_weights)]
    gender = np.where(rng.random(n) < 0.55, 'F', 'M')
    
    # Numeric columns: direct vectorized generation
    cc_num = rng.integers(4_000_000_000_000_000, 6_999_999_999_999_999, n, dtype=np.int64)
    amt = np.clip(np.abs(rng.lognormal(4.0, 1.5, n)), 1.0, 2000.0).astype(np.float32)
    zip_code = rng.integers(10000, 99999, n, dtype=np.int32)
    lat = rng.uniform(24.5, 49.0, n).astype(np.float32)
    long = rng.uniform(-124.5, -66.5, n).astype(np.float32)
    city_pop = rng.integers(100, 500000, n, dtype=np.int32)
    unix_time = rng.integers(1704067200, 1735689600, n, dtype=np.int64)  # 2024
    merch_lat = (lat + rng.uniform(-0.5, 0.5, n)).astype(np.float32)
    merch_long = (long + rng.uniform(-0.5, 0.5, n)).astype(np.float32)
    is_fraud = (rng.random(n) < fraud_rate).astype(np.int8)
    merch_zipcode = rng.integers(10000, 99999, n).astype(np.float32)
    
    # Derived columns
    trans_date_trans_time = pd.to_datetime(unix_time, unit='s')
    
    # DOB: random dates between 1940-2000
    dob_timestamps = rng.integers(-946771200, 978307200, n)  # 1940-2000 as unix
    dob = pd.to_datetime(dob_timestamps, unit='s').strftime('%Y-%m-%d')
    
    # Transaction IDs: hex strings (use 32-bit chunks to avoid int64 overflow)
    trans_num = np.array([
        f'{rng.integers(0, 2**32, dtype=np.uint32):08x}{rng.integers(0, 2**32, dtype=np.uint32):08x}{rng.integers(0, 2**32, dtype=np.uint32):08x}{rng.integers(0, 2**32, dtype=np.uint32):08x}'
        for _ in range(n)
    ])
    
    return {
        'trans_date_trans_time': trans_date_trans_time,
        'cc_num': cc_num,
        'merchant': merchant,
        'category': category,
        'amt': amt,
        'first': first,
        'last': last,
        'gender': gender,
        'street': street,
        'city': city,
        'state': state,
        'zip': zip_code,
        'lat': lat,
        'long': long,
        'city_pop': city_pop,
        'job': job,
        'dob': dob,
        'trans_num': trans_num,
        'unix_time': unix_time,
        'merch_lat': merch_lat,
        'merch_long': merch_long,
        'is_fraud': is_fraud,
        'merch_zipcode': merch_zipcode,
    }


def worker_main():
    worker_id = int(sys.argv[1])
    output_dir = sys.argv[2]
    pools_file = sys.argv[3]
    chunk_size = int(sys.argv[4])
    duration = int(sys.argv[5])
    fraud_rate = float(sys.argv[6])
    
    # Load pre-generated pools
    with open(pools_file, 'rb') as f:
        pools = pickle.load(f)
    
    # Convert to numpy arrays for O(1) indexing
    for key in pools:
        pools[key] = np.array(pools[key], dtype=object)
    
    # Unique RNG per worker (seeded by worker_id + time for uniqueness)
    rng = np.random.default_rng(seed=worker_id * 54321 + int(time.time() * 1000) % 100000)
    
    # Pre-convert to numpy arrays and normalize weights
    categories = np.array(CATEGORIES)
    cat_weights = np.array(CATEGORY_WEIGHTS)
    cat_weights = cat_weights / cat_weights.sum()  # Normalize to sum to 1.0
    states = np.array(US_STATES)
    state_weights = np.array(STATE_WEIGHTS)
    state_weights = state_weights / state_weights.sum()  # Normalize to sum to 1.0
    
    file_count = 0
    start_time = time.time()
    
    # Main generation loop
    while (duration == 0) or (time.time() - start_time) < duration:
        data = generate_chunk(chunk_size, pools, rng, categories, cat_weights, 
                             states, state_weights, fraud_rate)
        
        # Write using PyArrow directly (faster than pandas)
        table = pa.Table.from_pydict(data)
        filepath = f"{output_dir}/worker_{worker_id:03d}_{file_count:05d}.parquet"
        pq.write_table(table, filepath, compression=None)  # None = max speed
        
        file_count += 1


if __name__ == "__main__":
    worker_main()
'''


def main():
    global STOP_FLAG
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration from environment
    output_base = Path(os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'))
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration = int(os.getenv('DURATION_SECONDS', '300'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '500000'))
    fraud_rate = float(os.getenv('FRAUD_RATE', '0.005'))
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: Credit Card Transaction Generator")
    log("=" * 70)
    log(f"Output: {output_path}")
    log(f"Workers: {num_workers} | Duration: {duration}s | Chunk: {chunk_size:,} rows")
    log(f"Schema: 23 columns | Fraud rate: {fraud_rate*100:.1f}%")
    log("=" * 70)
    
    # Generate pools (one-time startup cost)
    pools_file = generate_pools(output_path)
    
    # Save schema metadata
    schema = {
        'columns': [
            'trans_date_trans_time', 'cc_num', 'merchant', 'category', 'amt',
            'first', 'last', 'gender', 'street', 'city', 'state', 'zip',
            'lat', 'long', 'city_pop', 'job', 'dob', 'trans_num', 'unix_time',
            'merch_lat', 'merch_long', 'is_fraud', 'merch_zipcode'
        ],
        'fraud_rate': fraud_rate,
        'chunk_size': chunk_size,
        'num_workers': num_workers,
        'timestamp': timestamp
    }
    with open(output_path / "_schema.json", 'w') as f:
        json.dump(schema, f, indent=2)
    
    # Launch worker subprocesses
    log(f"Launching {num_workers} workers...")
    processes = []
    for i in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', WORKER_SCRIPT, 
             str(i), str(output_path), str(pools_file), 
             str(chunk_size), str(duration), str(fraud_rate)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL
        )
        processes.append(p)
    
    # Monitor throughput
    start_time = time.time()
    last_bytes, last_time = 0, start_time
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        running = sum(1 for p in processes if p.poll() is None)
        
        if elapsed >= duration + 30 or running == 0:
            break
        
        if time.time() - last_time >= 5.0:
            files = list(output_path.glob("worker_*.parquet"))
            current_bytes = sum(f.stat().st_size for f in files) if files else 0
            interval = time.time() - last_time
            
            bytes_per_sec = (current_bytes - last_bytes) / interval if interval > 0 else 0
            gb = current_bytes / (1024**3)
            
            if bytes_per_sec >= 1024**3:
                speed = f"{bytes_per_sec / (1024**3):5.2f} GB/s"
            else:
                speed = f"{bytes_per_sec / (1024**2):6.1f} MB/s"
            
            log(f"[{elapsed:5.0f}s] Files: {len(files):5d} | Size: {gb:6.2f} GB | Speed: {speed} | Workers: {running}")
            
            last_bytes, last_time = current_bytes, time.time()
        
        time.sleep(1)
    
    # Graceful shutdown
    log("Stopping workers...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    # Final statistics
    files = list(output_path.glob("worker_*.parquet"))
    final_bytes = sum(f.stat().st_size for f in files) if files else 0
    total_time = time.time() - start_time
    
    log("=" * 70)
    log(f"COMPLETE: {len(files):,} files | {final_bytes/(1024**3):.2f} GB | {(final_bytes/(1024**2))/total_time:.0f} MB/s avg")
    log(f"Output: {output_path}")
    log("=" * 70)


if __name__ == "__main__":
    main()