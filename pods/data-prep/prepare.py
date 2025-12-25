#!/usr/bin/env python3
"""
Pod 2: Data Prepare Service
============================
GPU-accelerated ETL and feature engineering using NVIDIA RAPIDS Accelerator for Apache Spark.
Continuously monitors for new data runs from Pod 1 (gather.py) and outputs refined feature sets.

Features:
- Spark RAPIDS plugin for GPU-accelerated processing
- Continuous directory watcher for new run batches
- Standard scaling/normalization on V1-V28 features
- Time-based windowed feature engineering
- GPUDirect Storage (GDS) optimized I/O
- Parquet output optimized for cudf consumption
"""

import os
import sys
import time
import json
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Set
from dataclasses import dataclass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, FloatType, DoubleType, 
    IntegerType, LongType, StringType, TimestampType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global stop flag for graceful shutdown
STOP_FLAG = False


@dataclass
class PrepConfig:
    """Configuration for data preparation service"""
    input_dir: str
    output_dir: str
    poll_interval: int = 5  # Reduced for faster demo response
    batch_mode: bool = False
    num_gpus: int = 2
    gpu_memory_fraction: float = 0.8
    enable_gds: bool = True
    coalesce_output: int = 8  # Number of output partitions


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global STOP_FLAG
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    STOP_FLAG = True


def create_spark_session(config: PrepConfig) -> SparkSession:
    """
    Create Spark session with NVIDIA RAPIDS Accelerator configuration.
    
    Key RAPIDS settings:
    - spark.plugins: Enable SQL plugin for GPU acceleration
    - spark.rapids.sql.enabled: Master switch for GPU SQL operations
    - spark.rapids.memory.pinnedPool.size: Pinned memory for GPU transfers
    - spark.rapids.sql.concurrentGpuTasks: Concurrent GPU operations
    """
    logger.info("Initializing Spark session with RAPIDS Accelerator...")
    
    # Calculate memory settings based on available GPUs
    executor_memory = "32g"
    driver_memory = "16g"
    pinned_pool_size = "4g"
    
    builder = SparkSession.builder \
        .appName("FraudDetection-DataPrep-RAPIDS") \
        .master(os.getenv("SPARK_MASTER", "local[*]"))
    
    # Core Spark configuration
    builder = builder \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.parquet.enableVectorizedReader", "true")
    
    # RAPIDS Accelerator configuration
    builder = builder \
        .config("spark.plugins", "com.nvidia.spark.SQLPlugin") \
        .config("spark.rapids.sql.enabled", "true") \
        .config("spark.rapids.memory.pinnedPool.size", pinned_pool_size) \
        .config("spark.rapids.sql.concurrentGpuTasks", "2") \
        .config("spark.rapids.sql.variableFloatAgg.enabled", "true") \
        .config("spark.rapids.sql.explain", "NOT_ON_GPU") \
        .config("spark.rapids.sql.incompatibleOps.enabled", "true")
    
    # GPU resource configuration
    builder = builder \
        .config("spark.executor.resource.gpu.amount", str(config.num_gpus)) \
        .config("spark.task.resource.gpu.amount", "0.5") \
        .config("spark.rapids.sql.batchSizeBytes", "512m")
    
    # GPUDirect Storage configuration (if enabled)
    if config.enable_gds:
        builder = builder \
            .config("spark.rapids.memory.gpu.direct.storage.spill.enabled", "true") \
            .config("spark.rapids.shuffle.mode", "UCX") \
            .config("spark.rapids.shuffle.transport.ucxMgr.useWakeup", "true")
    
    # Parquet optimization for GPU
    builder = builder \
        .config("spark.rapids.sql.format.parquet.read.enabled", "true") \
        .config("spark.rapids.sql.format.parquet.write.enabled", "true") \
        .config("spark.sql.files.maxPartitionBytes", "512m")
    
    spark = builder.getOrCreate()
    
    # Log configuration summary
    logger.info("=" * 60)
    logger.info("RAPIDS Spark Configuration Summary")
    logger.info("=" * 60)
    logger.info(f"  RAPIDS Plugin: {spark.conf.get('spark.plugins', 'NOT SET')}")
    logger.info(f"  GPU Tasks: {spark.conf.get('spark.rapids.sql.concurrentGpuTasks', 'NOT SET')}")
    logger.info(f"  Pinned Memory: {spark.conf.get('spark.rapids.memory.pinnedPool.size', 'NOT SET')}")
    logger.info(f"  GDS Enabled: {config.enable_gds}")
    logger.info("=" * 60)
    
    return spark


def get_schema() -> StructType:
    """
    Define schema matching gather.py output (creditcard.csv format).
    Using explicit schema improves Spark read performance.
    """
    fields = [StructField("Time", FloatType(), True)]
    
    # V1 through V28 PCA features
    for i in range(1, 29):
        fields.append(StructField(f"V{i}", FloatType(), True))
    
    fields.extend([
        StructField("Amount", FloatType(), True),
        StructField("Class", IntegerType(), True)
    ])
    
    return StructType(fields)


class DirectoryWatcher:
    """
    Watches for new run directories from gather.py.
    Tracks processed directories to avoid reprocessing.
    """
    
    def __init__(self, watch_dir: Path, state_file: Path = None):
        self.watch_dir = watch_dir
        self.state_file = state_file or (watch_dir / ".prep_state.json")
        self.processed_dirs: Set[str] = self._load_state()
    
    def _load_state(self) -> Set[str]:
        """Load previously processed directories from state file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                    return set(state.get('processed', []))
            except Exception as e:
                logger.warning(f"Could not load state file: {e}")
        return set()
    
    def _save_state(self):
        """Persist processed directories to state file"""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({'processed': list(self.processed_dirs)}, f)
        except Exception as e:
            logger.warning(f"Could not save state file: {e}")
    
    def get_new_runs(self) -> List[Path]:
        """
        Find new run directories that haven't been processed.
        Run directories match pattern: run_YYYYMMDD_HHMMSS
        """
        new_runs = []
        
        if not self.watch_dir.exists():
            logger.warning(f"Watch directory does not exist: {self.watch_dir}")
            return new_runs
        
        for entry in self.watch_dir.iterdir():
            if entry.is_dir() and entry.name.startswith("run_"):
                if entry.name not in self.processed_dirs:
                    # Verify directory has data files
                    data_files = list(entry.glob("worker_*.parquet")) or \
                                 list(entry.glob("worker_*.csv")) or \
                                 list(entry.glob("worker_*.bin"))
                    if data_files:
                        new_runs.append(entry)
        
        # Sort by directory name (timestamp) to process in order
        return sorted(new_runs, key=lambda p: p.name)
    
    def mark_processed(self, run_dir: Path):
        """Mark a run directory as processed"""
        self.processed_dirs.add(run_dir.name)
        self._save_state()


class FeatureEngineer:
    """
    GPU-accelerated feature engineering using Spark RAPIDS.
    Implements standard scaling, normalization, and time-based features.
    """
    
    # PCA feature columns
    PCA_COLS = [f"V{i}" for i in range(1, 29)]
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        # Pre-compute statistics for standard scaling (will be updated per batch)
        self._stats_cache: Dict[str, tuple] = {}
    
    def compute_statistics(self, df: DataFrame) -> Dict[str, tuple]:
        """
        Compute mean and stddev for PCA features (GPU-accelerated).
        Returns dict of {column: (mean, stddev)}
        """
        logger.info("Computing feature statistics on GPU...")
        
        # Build aggregation expressions
        agg_exprs = []
        for col in self.PCA_COLS:
            agg_exprs.extend([
                F.mean(col).alias(f"{col}_mean"),
                F.stddev(col).alias(f"{col}_std")
            ])
        agg_exprs.extend([
            F.mean("Amount").alias("Amount_mean"),
            F.stddev("Amount").alias("Amount_std")
        ])
        
        # Execute single aggregation pass on GPU
        stats_row = df.agg(*agg_exprs).collect()[0]
        
        stats = {}
        for col in self.PCA_COLS:
            mean_val = stats_row[f"{col}_mean"] or 0.0
            std_val = stats_row[f"{col}_std"] or 1.0
            stats[col] = (mean_val, max(std_val, 1e-8))  # Prevent division by zero
        
        stats["Amount"] = (
            stats_row["Amount_mean"] or 0.0,
            max(stats_row["Amount_std"] or 1.0, 1e-8)
        )
        
        return stats
    
    def apply_standard_scaling(self, df: DataFrame, stats: Dict[str, tuple]) -> DataFrame:
        """
        Apply standard scaling (z-score normalization) to features.
        GPU-accelerated through RAPIDS SQL plugin.
        """
        logger.info("Applying standard scaling on GPU...")
        
        for col in self.PCA_COLS:
            mean_val, std_val = stats[col]
            df = df.withColumn(
                f"{col}_scaled",
                (F.col(col) - F.lit(mean_val)) / F.lit(std_val)
            )
        
        # Scale Amount separately (log transform + scaling)
        amt_mean, amt_std = stats["Amount"]
        df = df.withColumn(
            "Amount_log",
            F.log1p(F.col("Amount"))
        ).withColumn(
            "Amount_scaled",
            (F.col("Amount") - F.lit(amt_mean)) / F.lit(amt_std)
        )
        
        return df
    
    def handle_missing_values(self, df: DataFrame) -> DataFrame:
        """
        Handle missing values using GPU-backed operations.
        Strategy: Fill with 0 for PCA features, median for Amount.
        """
        logger.info("Handling missing values...")
        
        # For PCA features, fill with 0 (they're already centered)
        fill_values = {col: 0.0 for col in self.PCA_COLS}
        fill_values["Amount"] = 0.0
        fill_values["Class"] = 0
        
        df = df.fillna(fill_values)
        
        # Replace infinities with 0
        for col in self.PCA_COLS + ["Amount"]:
            df = df.withColumn(
                col,
                F.when(F.col(col).isNull() | F.isnan(col) | 
                       (F.col(col) == float('inf')) | 
                       (F.col(col) == float('-inf')), 0.0)
                .otherwise(F.col(col))
            )
        
        return df
    
    def create_time_features(self, df: DataFrame) -> DataFrame:
        """
        Create time-based features using Spark Window functions.
        These leverage GPU acceleration through RAPIDS.
        """
        logger.info("Creating time-based features with Window functions...")
        
        # Convert Time (seconds) to hours for windowing
        df = df.withColumn("Time_hours", F.col("Time") / 3600.0)
        
        # Extract time-of-day features
        df = df.withColumn(
            "hour_of_day",
            F.floor(F.col("Time") / 3600.0) % 24
        ).withColumn(
            "is_night",  # Transactions between 10pm-6am
            F.when((F.col("hour_of_day") >= 22) | (F.col("hour_of_day") < 6), 1).otherwise(0)
        )
        
        # Time-based window for transaction frequency
        # Window: transactions within last 1 hour (3600 seconds)
        time_window = Window.orderBy("Time").rangeBetween(-3600, 0)
        
        df = df.withColumn(
            "tx_count_1h",
            F.count("*").over(time_window)
        ).withColumn(
            "amount_sum_1h",
            F.sum("Amount").over(time_window)
        ).withColumn(
            "amount_avg_1h",
            F.avg("Amount").over(time_window)
        ).withColumn(
            "amount_max_1h",
            F.max("Amount").over(time_window)
        )
        
        # 15-minute window for high-frequency detection
        time_window_15m = Window.orderBy("Time").rangeBetween(-900, 0)
        
        df = df.withColumn(
            "tx_count_15m",
            F.count("*").over(time_window_15m)
        ).withColumn(
            "amount_sum_15m",
            F.sum("Amount").over(time_window_15m)
        )
        
        # Velocity features (rate of change)
        df = df.withColumn(
            "tx_velocity_1h",
            F.col("tx_count_1h") / F.lit(60.0)  # transactions per minute
        ).withColumn(
            "amount_velocity_1h",
            F.col("amount_sum_1h") / F.greatest(F.col("tx_count_1h"), F.lit(1))
        )
        
        # Amount deviation from rolling average
        df = df.withColumn(
            "amount_deviation",
            F.when(F.col("amount_avg_1h") > 0,
                   (F.col("Amount") - F.col("amount_avg_1h")) / F.col("amount_avg_1h"))
            .otherwise(0.0)
        )
        
        # High-value transaction flag
        df = df.withColumn(
            "is_high_value",
            F.when(F.col("Amount") > F.col("amount_max_1h") * 0.9, 1).otherwise(0)
        )
        
        return df
    
    def create_interaction_features(self, df: DataFrame) -> DataFrame:
        """
        Create interaction features between PCA components.
        Focus on features commonly associated with fraud patterns.
        """
        logger.info("Creating interaction features...")
        
        # V1-V3 interactions (often capture card-not-present fraud)
        df = df.withColumn("V1_V2_interaction", F.col("V1") * F.col("V2"))
        df = df.withColumn("V1_V3_interaction", F.col("V1") * F.col("V3"))
        
        # V4-V6 interactions (often capture merchant anomalies)
        df = df.withColumn("V4_V6_interaction", F.col("V4") * F.col("V6"))
        
        # Amount interactions with key PCA components
        df = df.withColumn("Amount_V1_interaction", F.col("Amount_scaled") * F.col("V1"))
        df = df.withColumn("Amount_V14_interaction", F.col("Amount_scaled") * F.col("V14"))
        
        # Squared terms for non-linear patterns
        df = df.withColumn("V1_squared", F.col("V1") ** 2)
        df = df.withColumn("V14_squared", F.col("V14") ** 2)
        df = df.withColumn("V17_squared", F.col("V17") ** 2)
        
        # Absolute value features (magnitude matters for some fraud patterns)
        df = df.withColumn("V1_abs", F.abs(F.col("V1")))
        df = df.withColumn("V3_abs", F.abs(F.col("V3")))
        df = df.withColumn("V14_abs", F.abs(F.col("V14")))
        
        return df
    
    def add_metadata(self, df: DataFrame, run_name: str) -> DataFrame:
        """Add processing metadata for lineage tracking"""
        return df.withColumn(
            "source_run", F.lit(run_name)
        ).withColumn(
            "prep_timestamp", F.lit(datetime.now().isoformat())
        ).withColumn(
            "transaction_id", F.monotonically_increasing_id()
        )
    
    def process(self, df: DataFrame, run_name: str) -> DataFrame:
        """
        Full feature engineering pipeline.
        All operations are GPU-accelerated through RAPIDS.
        """
        start_time = time.time()
        initial_count = df.count()
        logger.info(f"Processing {initial_count:,} records from {run_name}")
        
        # Step 1: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 2: Compute statistics and apply scaling
        stats = self.compute_statistics(df)
        df = self.apply_standard_scaling(df, stats)
        
        # Step 3: Create time-based features
        df = self.create_time_features(df)
        
        # Step 4: Create interaction features
        df = self.create_interaction_features(df)
        
        # Step 5: Add metadata
        df = self.add_metadata(df, run_name)
        
        elapsed = time.time() - start_time
        logger.info(f"Feature engineering complete in {elapsed:.1f}s "
                   f"({initial_count / elapsed:.0f} records/sec)")
        
        return df


class DataPrepService:
    """
    Main service orchestrator for Pod 2.
    Coordinates file watching, processing, and output.
    """
    
    def __init__(self, config: PrepConfig):
        self.config = config
        self.input_path = Path(config.input_dir)
        self.output_path = Path(config.output_dir)
        
        # Ensure output directory exists
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.spark = create_spark_session(config)
        self.watcher = DirectoryWatcher(self.input_path)
        self.engineer = FeatureEngineer(self.spark)
        self.schema = get_schema()
        
        logger.info("=" * 60)
        logger.info("Pod 2: Data Prep Service - Initialized")
        logger.info("=" * 60)
        logger.info(f"Input directory:  {self.input_path}")
        logger.info(f"Output directory: {self.output_path}")
        logger.info(f"Poll interval:    {config.poll_interval}s")
        logger.info(f"Batch mode:       {config.batch_mode}")
        logger.info("=" * 60)
    
    def read_run_data(self, run_dir: Path) -> Optional[DataFrame]:
        """
        Read data from a run directory, supporting multiple formats.
        Uses schema inference optimization and GPU-accelerated Parquet reader.
        """
        logger.info(f"Reading data from: {run_dir}")
        
        # Try Parquet first (preferred for GPU)
        parquet_files = list(run_dir.glob("worker_*.parquet"))
        if parquet_files:
            logger.info(f"Found {len(parquet_files)} Parquet files")
            return self.spark.read \
                .schema(self.schema) \
                .parquet(str(run_dir / "worker_*.parquet"))
        
        # Try CSV
        csv_files = list(run_dir.glob("worker_*.csv"))
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files")
            return self.spark.read \
                .schema(self.schema) \
                .option("header", "true") \
                .csv(str(run_dir / "worker_*.csv"))
        
        # Try binary (raw numpy arrays) - requires special handling
        bin_files = list(run_dir.glob("worker_*.bin"))
        if bin_files:
            logger.warning("Binary format detected - converting via pandas")
            return self._read_binary_files(bin_files)
        
        logger.warning(f"No data files found in {run_dir}")
        return None
    
    def _read_binary_files(self, bin_files: List[Path]) -> Optional[DataFrame]:
        """
        Read raw binary numpy arrays and convert to Spark DataFrame.
        Binary files contain float32 arrays with 31 columns.
        """
        import numpy as np
        import pandas as pd
        
        all_data = []
        for bf in bin_files:
            try:
                data = np.fromfile(str(bf), dtype=np.float32)
                # Reshape to 31 columns (Time, V1-V28, Amount, Class)
                num_rows = len(data) // 31
                data = data[:num_rows * 31].reshape(-1, 31)
                all_data.append(data)
            except Exception as e:
                logger.warning(f"Error reading {bf}: {e}")
        
        if not all_data:
            return None
        
        combined = np.vstack(all_data)
        columns = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
        pdf = pd.DataFrame(combined, columns=columns)
        pdf["Class"] = pdf["Class"].astype(int)
        
        return self.spark.createDataFrame(pdf)
    
    def write_output(self, df: DataFrame, run_name: str):
        """
        Write processed data to output directory.
        Optimized for cudf consumption with efficient partitioning.
        """
        output_file = self.output_path / f"features_{run_name}.parquet"
        logger.info(f"Writing output to: {output_file}")
        
        # Coalesce to reduce file count for efficient reading
        df.coalesce(self.config.coalesce_output) \
            .write \
            .mode("overwrite") \
            .option("compression", "snappy") \
            .parquet(str(output_file))
        
        # Get output stats
        output_size = sum(f.stat().st_size for f in output_file.parent.glob(f"features_{run_name}.parquet/**/*.parquet"))
        logger.info(f"Output size: {output_size / (1024*1024):.1f} MB")
        
        # Save feature metadata for Pod 3
        self._save_feature_metadata(run_name, df)
    
    def _save_feature_metadata(self, run_name: str, df: DataFrame):
        """Save feature column metadata for training pod"""
        metadata = {
            "run_name": run_name,
            "timestamp": datetime.now().isoformat(),
            "columns": df.columns,
            "feature_columns": [c for c in df.columns if c not in 
                               ["transaction_id", "source_run", "prep_timestamp", "Class"]],
            "target_column": "Class",
            "record_count": df.count()
        }
        
        metadata_file = self.output_path / f"metadata_{run_name}.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved feature metadata: {metadata_file}")
    
    def process_run(self, run_dir: Path) -> bool:
        """Process a single run directory"""
        run_name = run_dir.name
        logger.info("=" * 60)
        logger.info(f"Processing run: {run_name}")
        logger.info("=" * 60)
        
        try:
            # Read input data
            df = self.read_run_data(run_dir)
            if df is None:
                logger.error(f"Failed to read data from {run_dir}")
                return False
            
            # Apply feature engineering
            df_processed = self.engineer.process(df, run_name)
            
            # Write output
            self.write_output(df_processed, run_name)
            
            # Mark as processed
            self.watcher.mark_processed(run_dir)
            
            logger.info(f"Successfully processed: {run_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {run_name}: {e}", exc_info=True)
            return False
    
    def run_continuous(self):
        """
        Continuous processing mode - watches for new runs and processes them.
        Main loop for production deployment.
        """
        global STOP_FLAG
        
        logger.info("Starting continuous processing mode...")
        logger.info(f"Watching: {self.input_path}")
        logger.info(f"Poll interval: {self.config.poll_interval}s")
        
        runs_processed = 0
        
        while not STOP_FLAG:
            # Check for new runs
            new_runs = self.watcher.get_new_runs()
            
            if new_runs:
                logger.info(f"Found {len(new_runs)} new run(s) to process")
                
                for run_dir in new_runs:
                    if STOP_FLAG:
                        break
                    
                    if self.process_run(run_dir):
                        runs_processed += 1
            
            # Wait before next poll
            if not STOP_FLAG:
                time.sleep(self.config.poll_interval)
        
        logger.info(f"Continuous processing stopped. Total runs processed: {runs_processed}")
    
    def run_batch(self, specific_run: str = None):
        """
        Batch processing mode - process all pending runs once.
        Useful for backfilling or testing.
        """
        logger.info("Starting batch processing mode...")
        
        if specific_run:
            # Process specific run
            run_dir = self.input_path / specific_run
            if run_dir.exists():
                self.process_run(run_dir)
            else:
                logger.error(f"Run directory not found: {run_dir}")
        else:
            # Process all pending runs
            new_runs = self.watcher.get_new_runs()
            logger.info(f"Found {len(new_runs)} run(s) to process")
            
            for run_dir in new_runs:
                self.process_run(run_dir)
        
        logger.info("Batch processing complete")
    
    def shutdown(self):
        """Clean shutdown of Spark session"""
        logger.info("Shutting down Spark session...")
        self.spark.stop()


def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load configuration from environment
    config = PrepConfig(
        input_dir=os.getenv('INPUT_DIR', '/mnt/fsaai-shared/ebiser/fraud-data'),
        output_dir=os.getenv('OUTPUT_DIR', '/mnt/fsaai-shared/ebiser/prep_output'),
        poll_interval=int(os.getenv('POLL_INTERVAL', '5')),
        batch_mode=os.getenv('BATCH_MODE', 'false').lower() == 'true',
        num_gpus=int(os.getenv('NUM_GPUS', '2')),
        gpu_memory_fraction=float(os.getenv('GPU_MEMORY_FRACTION', '0.8')),
        enable_gds=os.getenv('ENABLE_GDS', 'true').lower() == 'true',
        coalesce_output=int(os.getenv('COALESCE_OUTPUT', '8'))
    )
    
    # Print configuration
    logger.info("=" * 60)
    logger.info("Pod 2: Data Prep Service - Starting")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  Input:       {config.input_dir}")
    logger.info(f"  Output:      {config.output_dir}")
    logger.info(f"  Batch Mode:  {config.batch_mode}")
    logger.info(f"  GPUs:        {config.num_gpus}")
    logger.info(f"  GDS Enabled: {config.enable_gds}")
    
    # Initialize and run service
    service = DataPrepService(config)
    
    try:
        if config.batch_mode:
            specific_run = os.getenv('SPECIFIC_RUN')
            service.run_batch(specific_run)
        else:
            service.run_continuous()
    finally:
        service.shutdown()
    
    logger.info("=" * 60)
    logger.info("Pod 2: Data Prep Service - Stopped")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()