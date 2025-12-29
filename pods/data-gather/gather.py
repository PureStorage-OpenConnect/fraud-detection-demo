#!/usr/bin/env python3
"""Pod 1: Data Gather Service - CPU-based parallel data generator"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

STOP_FLAG = False


def log(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def signal_handler(signum, frame):
    global STOP_FLAG
    log(f"Received signal {signum}, stopping...")
    STOP_FLAG = True


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    template_path = Path(os.getenv('TEMPLATE_DIR', '/mnt/datasets/kaggle/creditcardfraud')) / os.getenv('TEMPLATE_FILE', 'creditcard.csv')
    output_base = Path(os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'))
    num_workers = int(os.getenv('NUM_WORKERS', '128'))
    duration = int(os.getenv('DURATION_SECONDS', '300'))
    chunk_size = int(os.getenv('CHUNK_SIZE', '2000000'))
    output_format = os.getenv('OUTPUT_FORMAT', 'parquet')
    
    # Load schema
    df_sample = pd.read_csv(template_path, nrows=10000)
    columns = list(df_sample.columns)
    stats = {col: {'mean': float(df_sample[col].mean()), 'std': float(df_sample[col].std())}
             for col in columns if np.issubdtype(df_sample[col].dtype, np.number)}
    
    # Create output dir
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save schema
    import json
    schema_file = output_path / "_schema.json"
    with open(schema_file, 'w') as f:
        json.dump({'columns': columns, 'stats': stats}, f)
    
    log("=" * 70)
    log("Pod 1: Financial Fraud Data Generator")
    log("=" * 70)
    log(f"Output: {output_path}")
    log(f"Workers: {num_workers} | Duration: {duration}s | Chunk: {chunk_size:,}")
    log("=" * 70)
    
    # Worker script
    worker_script = '''
import sys,json,time,numpy as np,pandas as pd
from pathlib import Path
def gen(cols,stats,n,rng):
    d={}
    for c in cols:
        if c in stats:
            if c=='Class':d[c]=rng.integers(0,2,size=n,dtype=np.int8)
            elif c=='Time':d[c]=rng.uniform(0,172800,size=n).astype(np.float32)
            elif c=='Amount':d[c]=np.clip(np.abs(rng.lognormal(3,2,n)),0,25000).astype(np.float32)
            else:d[c]=rng.normal(stats[c]['mean'],max(stats[c]['std'],0.01),n).astype(np.float32)
    return pd.DataFrame(d)
wid,odir,chunk,dur,sf,fmt=int(sys.argv[1]),sys.argv[2],int(sys.argv[3]),int(sys.argv[4]),sys.argv[5],sys.argv[6]
with open(sf) as f:schema=json.load(f)
rng=np.random.default_rng(seed=wid*12345+int(time.time()*1000)%100000)
cols,stats,t0,fc=schema['columns'],schema['stats'],time.time(),0
if fmt=='parquet':
    import pyarrow as pa,pyarrow.parquet as pq
    while(time.time()-t0)<dur:pq.write_table(pa.Table.from_pandas(gen(cols,stats,chunk,rng),preserve_index=False),Path(odir)/f"worker_{wid:03d}_{fc:05d}.parquet",compression=None);fc+=1
elif fmt=='binary':
    while(time.time()-t0)<dur:rng.standard_normal((chunk,31)).astype(np.float32).tofile(Path(odir)/f"worker_{wid:03d}_{fc:05d}.bin");fc+=1
else:
    while(time.time()-t0)<dur:gen(cols,stats,chunk,rng).to_csv(Path(odir)/f"worker_{wid:03d}_{fc:05d}.csv",index=False);fc+=1
'''
    
    # Launch workers
    processes = [subprocess.Popen([sys.executable,'-c',worker_script,str(i),str(output_path),str(chunk_size),str(duration),str(schema_file),output_format],
                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,stdin=subprocess.DEVNULL) for i in range(num_workers)]
    
    log(f"Launched {num_workers} workers...")
    
    file_pattern = {"parquet":"worker_*.parquet","binary":"worker_*.bin"}.get(output_format,"worker_*.csv")
    bytes_per_row = {"parquet":130,"binary":124}.get(output_format,200)
    
    start_time = time.time()
    last_bytes, last_time = 0, start_time
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        running = sum(1 for p in processes if p.poll() is None)
        if elapsed >= duration + 10 or running == 0:
            break
        
        if time.time() - last_time >= 5.0:
            files = list(output_path.glob(file_pattern))
            current_bytes = sum(f.stat().st_size for f in files) if files else 0
            interval = time.time() - last_time
            mbps = ((current_bytes - last_bytes) / (1024**2)) / interval if interval > 0 else 0
            gb = current_bytes / (1024**3)
            speed = f"{mbps/1024:5.2f} GB/s" if mbps >= 1000 else f"{mbps:6.1f} MB/s"
            log(f"[{elapsed:5.0f}s] Files: {len(files):5d} | Size: {gb:6.2f} GB | Speed: {speed} | Workers: {running}")
            last_bytes, last_time = current_bytes, time.time()
        time.sleep(1)
    
    # Cleanup
    for p in processes:
        if p.poll() is None: p.terminate()
    for p in processes:
        try: p.wait(timeout=5)
        except: p.kill()
    
    files = list(output_path.glob(file_pattern))
    final_bytes = sum(f.stat().st_size for f in files) if files else 0
    total_time = time.time() - start_time
    
    log("=" * 70)
    log(f"COMPLETE: {len(files):,} files | {final_bytes/(1024**3):.2f} GB | {(final_bytes/(1024**2))/total_time:.0f} MB/s avg")
    log("=" * 70)
    
    schema_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()