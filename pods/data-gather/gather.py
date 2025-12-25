#!/usr/bin/env python3
"""
Pod 1: High-Performance Data Gather Service (UPDATED for Spark Compatibility)
==============================================================================
Changes from original:
1. Added _SUCCESS marker file on completion (Spark convention)
2. Added _manifest.json with file listing and schema
3. Improved schema metadata for Spark schema inference
4. Added partition hints for optimal Spark parallelism
"""

# =============================================================================
# ADD THESE IMPORTS (if not present)
# =============================================================================
import json
from pathlib import Path

# =============================================================================
# ADD THIS FUNCTION after the existing run_stress_test function
# =============================================================================

def write_spark_metadata(output_path: Path, schema: dict, file_pattern: str, 
                         total_files: int, total_bytes: int, output_format: str):
    """
    Write Spark-compatible metadata files for efficient directory processing.
    
    Creates:
    - _SUCCESS: Marker file indicating complete write (Spark convention)
    - _manifest.json: File listing with schema for Spark readers
    - _spark_schema.json: Explicit schema for Spark schema inference
    """
    
    # 1. Write _SUCCESS marker (empty file, Spark convention)
    success_file = output_path / "_SUCCESS"
    success_file.touch()
    
    # 2. Write manifest with file listing
    files = sorted([f.name for f in output_path.glob(file_pattern)])
    manifest = {
        "format": output_format,
        "files": files,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "schema": schema,
        "partition_hint": min(total_files, 128),  # Suggested parallelism
        "completed_at": datetime.now().isoformat()
    }
    
    manifest_file = output_path / "_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # 3. Write Spark-compatible schema
    spark_schema = {
        "type": "struct",
        "fields": []
    }
    
    # Build Spark schema from columns
    for col in schema.get('columns', []):
        if col == 'Class':
            field_type = "integer"
        elif col == 'Time':
            field_type = "float"
        else:
            field_type = "float"
        
        spark_schema["fields"].append({
            "name": col,
            "type": field_type,
            "nullable": True,
            "metadata": {}
        })
    
    spark_schema_file = output_path / "_spark_schema.json"
    with open(spark_schema_file, 'w') as f:
        json.dump(spark_schema, f, indent=2)
    
    log(f"Wrote Spark metadata files to {output_path}")


# =============================================================================
# MODIFY run_stress_test() - Add this block BEFORE the final return statement
# =============================================================================

def run_stress_test_updated_ending():
    """
    ADD THIS BLOCK at the end of run_stress_test(), 
    just before 'schema_file.unlink(missing_ok=True)'
    """
    
    # Write Spark-compatible metadata
    write_spark_metadata(
        output_path=output_path,
        schema=schema,
        file_pattern=file_pattern,
        total_files=final_files,
        total_bytes=final_bytes,
        output_format=output_format
    )
    
    # Keep schema file for Spark (don't delete it)
    # Comment out or remove: schema_file.unlink(missing_ok=True)


# =============================================================================
# FULL UPDATED run_stress_test FUNCTION (for reference)
# Replace the existing function with this version
# =============================================================================

def run_stress_test(
    template_path: Path,
    output_base: Path,
    num_workers: int,
    duration_seconds: int,
    chunk_size: int,
    output_format: str = 'parquet'
):
    """Main stress test orchestrator - UPDATED with Spark metadata"""
    global STOP_FLAG
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_base / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    log("=" * 70)
    log("Pod 1: FlashBlade High-Performance Stress Test")
    log("=" * 70)
    log(f"Output directory: {output_path}")
    
    # Load schema
    schema = load_schema(template_path)
    
    log(f"Workers:    {num_workers}")
    log(f"Chunk size: {chunk_size:,} rows")
    log(f"Duration:   {duration_seconds} seconds")
    log(f"Format:     {output_format}")
    log("=" * 70)
    
    # Save schema for workers (keep for Spark too)
    schema_file = output_path / "_schema.json"
    with open(schema_file, 'w') as f:
        json.dump(schema, f)
    
    log(f"Starting {num_workers} worker processes...")
    
    # [... worker_script and process launching code remains unchanged ...]
    
    # Worker script optimized for throughput
    worker_script = f'''
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

def generate_data_fast(columns, stats, num_rows, rng):
    """Optimized data generation using pre-allocated arrays"""
    data = {{}}
    for col in columns:
        if col in stats:
            if col == 'Class':
                data[col] = rng.integers(0, 2, size=num_rows, dtype=np.int8)
            elif col == 'Time':
                data[col] = rng.uniform(0, 172800, size=num_rows).astype(np.float32)
            elif col == 'Amount':
                data[col] = np.clip(np.abs(rng.lognormal(3.0, 2.0, num_rows)), 0, 25000).astype(np.float32)
            else:
                data[col] = rng.normal(stats[col]['mean'], max(stats[col]['std'], 0.01), num_rows).astype(np.float32)
    return pd.DataFrame(data)

worker_id = int(sys.argv[1])
output_dir = sys.argv[2]
chunk_size = int(sys.argv[3])
duration = int(sys.argv[4])
schema_file = sys.argv[5]
output_format = sys.argv[6]

with open(schema_file) as f:
    schema = json.load(f)

rng = np.random.default_rng(seed=worker_id * 12345 + int(time.time() * 1000) % 100000)
columns = schema['columns']
stats = schema['stats']

start_time = time.time()
file_counter = 0

if output_format == 'parquet':
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    while (time.time() - start_time) < duration:
        chunk = generate_data_fast(columns, stats, chunk_size, rng)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.parquet"
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        pq.write_table(table, file_path, compression=None)  # No compression for speed
        file_counter += 1

elif output_format == 'binary':
    # Raw binary numpy arrays - maximum speed
    while (time.time() - start_time) < duration:
        # Generate raw float32 array (31 columns x chunk_size rows)
        data = rng.standard_normal((chunk_size, 31)).astype(np.float32)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.bin"
        data.tofile(file_path)
        file_counter += 1

else:  # csv
    while (time.time() - start_time) < duration:
        chunk = generate_data_fast(columns, stats, chunk_size, rng)
        file_path = Path(output_dir) / f"worker_{{worker_id:03d}}_{{file_counter:05d}}.csv"
        chunk.to_csv(file_path, index=False)
        file_counter += 1
'''
    
    # Launch all workers
    processes = []
    for worker_id in range(num_workers):
        p = subprocess.Popen(
            [sys.executable, '-c', worker_script, 
             str(worker_id), str(output_path), str(chunk_size), 
             str(duration_seconds), str(schema_file), output_format],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        processes.append(p)
    
    log(f"All {num_workers} workers launched. Monitoring for {duration_seconds}s...")
    log("-" * 70)
    
    # Determine file pattern based on format
    if output_format == 'parquet':
        file_pattern = "worker_*.parquet"
        bytes_per_row = 130
    elif output_format == 'binary':
        file_pattern = "worker_*.bin"
        bytes_per_row = 31 * 4
    else:
        file_pattern = "worker_*.csv"
        bytes_per_row = 200
    
    # Monitor progress
    start_time = time.time()
    last_bytes = 0
    last_time = start_time
    report_interval = 5.0
    
    while not STOP_FLAG:
        elapsed = time.time() - start_time
        
        if elapsed >= duration_seconds + 10:
            break
        
        running = sum(1 for p in processes if p.poll() is None)
        if running == 0:
            break
        
        if time.time() - last_time >= report_interval:
            current_bytes, file_count = get_dir_stats(output_path, file_pattern)
            interval_time = time.time() - last_time
            interval_bytes = current_bytes - last_bytes
            
            mbps = (interval_bytes / (1024 * 1024)) / interval_time if interval_time > 0 else 0
            gbps = mbps / 1024
            gb_total = current_bytes / (1024 * 1024 * 1024)
            
            est_records = current_bytes // bytes_per_row
            rps = (interval_bytes // bytes_per_row) / interval_time if interval_time > 0 else 0
            
            speed_str = f"{mbps:6.1f} MB/s"
            if mbps >= 1000:
                speed_str = f"{gbps:5.2f} GB/s"
            
            log(f"[{elapsed:5.0f}s] Files: {file_count:5d} | "
                f"Size: {gb_total:6.2f} GB | "
                f"Speed: {speed_str} | "
                f"~{rps/1e6:.2f}M rec/s | "
                f"Workers: {running}")
            
            last_bytes = current_bytes
            last_time = time.time()
        
        time.sleep(1)
    
    # Terminate workers
    log("Stopping workers...")
    for p in processes:
        if p.poll() is None:
            p.terminate()
    
    for p in processes:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    
    # Check for worker errors
    errors = []
    for i, p in enumerate(processes):
        if p.returncode and p.returncode != 0:
            stderr = p.stderr.read().decode() if p.stderr else ""
            if stderr:
                errors.append(f"Worker {i}: {stderr[:200]}")
    
    if errors:
        log(f"WARNING: {len(errors)} workers had errors")
        for err in errors[:3]:
            log(f"  {err}")
    
    # Final report
    total_elapsed = time.time() - start_time
    final_bytes, final_files = get_dir_stats(output_path, file_pattern)
    final_gb = final_bytes / (1024 * 1024 * 1024)
    avg_mbps = (final_bytes / (1024 * 1024)) / total_elapsed if total_elapsed > 0 else 0
    est_records = final_bytes // bytes_per_row
    
    # =========================================================================
    # NEW: Write Spark-compatible metadata
    # =========================================================================
    write_spark_metadata(
        output_path=output_path,
        schema=schema,
        file_pattern=file_pattern,
        total_files=final_files,
        total_bytes=final_bytes,
        output_format=output_format
    )
    
    log("=" * 70)
    log("FINAL RESULTS")
    log("=" * 70)
    log(f"Output:      {output_path}")
    log(f"Format:      {output_format}")
    log(f"Duration:    {total_elapsed:.1f} seconds")
    log(f"Files:       {final_files:,}")
    log(f"Total Size:  {final_gb:.2f} GB ({final_bytes:,} bytes)")
    if avg_mbps >= 1000:
        log(f"Throughput:  {avg_mbps/1024:.2f} GB/s average")
    else:
        log(f"Throughput:  {avg_mbps:.1f} MB/s average")
    log(f"Est Records: ~{est_records:,}")
    log(f"Spark Ready: _SUCCESS, _manifest.json, _spark_schema.json written")
    log("=" * 70)
    
    # NOTE: Keep schema_file for Spark - don't delete it
    # schema_file.unlink(missing_ok=True)  # REMOVED
    
    return output_path