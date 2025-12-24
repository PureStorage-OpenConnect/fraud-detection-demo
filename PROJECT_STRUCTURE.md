# Project Structure

Complete file structure for NVIDIA Financial Fraud Detection Pipeline.

```
nvidia-fraud-detection-pipeline/
│
├── README.md                           # Main project documentation
├── PROJECT_STRUCTURE.md               # This file
├── LICENSE                            # Apache 2.0 license
├── .env                               # Environment variables (DO NOT COMMIT)
├── .env.example                       # Environment variables template
├── .gitignore                         # Git exclusions
├── docker-compose.yaml                # Container orchestration
├── Makefile                           # Build and run commands
│
├── pods/                              # Microservice containers
│   │
│   ├── data-gather/                  # Pod 1: FlashBlade Stress Test
│   │   ├── Dockerfile
│   │   ├── gather.py                 # High-performance parallel generator
│   │   └── requirements.txt          # pandas, numpy
│   │
│   ├── data-prep/                    # Pod 2: Feature Engineering (GPU)
│   │   ├── Dockerfile
│   │   ├── prep.py                   # RAPIDS-based feature engineering
│   │   └── requirements.txt
│   │
│   ├── model-build/                  # Pod 3: Model Training (GPU)
│   │   ├── Dockerfile
│   │   ├── train.py                  # GNN + XGBoost training
│   │   └── requirements.txt
│   │
│   ├── inference/                    # Pod 4: Triton Inference (GPU)
│   │   ├── Dockerfile
│   │   └── config/
│   │       └── README.md             # Triton configuration notes
│   │
│   └── notification/                 # Pod 5: Alert Service
│       ├── Dockerfile
│       ├── app.py                    # Flask webhook server
│       └── requirements.txt
│
└── scripts/
    └── build_all.sh                  # Build all containers script
```

## Storage Structure

### Template Input (Read-Only)
```
/mnt/datasets/kaggle/creditcardfraud/
└── creditcard.csv                    # Kaggle schema template (284,807 rows)
```

### FlashBlade Output - Pod 1 (High-Throughput Parallel Write)
```
/mnt/fsaai-shared/ebiser/fraud-data/
├── thread_000_data.csv               # Worker 0 output
├── thread_001_data.csv               # Worker 1 output
├── thread_002_data.csv               # Worker 2 output
├── ...
└── thread_127_data.csv               # Worker 127 output
```

### FlashBlade Features - Pod 2 (Parallel I/O)
```
/mnt/fsaai-shared/ebiser/
└── prep_output/
    ├── features_YYYYMMDD_HHMMSS.parquet
    └── graph_edges_YYYYMMDD_HHMMSS.csv
```

### FlashArray (FA) - Low Latency
```
~/ebiser/nvidia.financial.fraud.detection/
└── model_repository/
    ├── fraud_xgboost/
    │   ├── config.pbtxt
    │   └── 1/
    │       └── model.json
    └── fraud_gnn/
        ├── config.pbtxt
        └── 1/
            └── model.pt
```

### S3 - Archival & Versioning
```
s3://fraud-detection-bucket/
└── model_versions/
    ├── fraud_xgboost_vYYYYMMDD_HHMMSS.tar.gz
    └── fraud_gnn_vYYYYMMDD_HHMMSS.tar.gz
```

## File Descriptions

### Root Files
- **README.md**: Main project documentation with architecture diagrams
- **PROJECT_STRUCTURE.md**: This file - complete project layout
- **LICENSE**: Apache 2.0 license
- **.env**: Local environment configuration with actual credentials (NOT in git)
- **.env.example**: Template showing required environment variables
- **.gitignore**: Git exclusion rules to protect secrets and build artifacts
- **docker-compose.yaml**: Orchestrates all 5 pods with storage mounts
- **Makefile**: Convenience commands for build/run operations

### Pod 1: Data Gather (FlashBlade Stress Test)
- **gather.py**: High-performance parallel data generator
  - 128 worker threads writing simultaneously
  - Schema-based generation from Kaggle template
  - Continuous append mode for sustained I/O
  - Real-time throughput monitoring
- **Dockerfile**: Python 3.11 slim with pandas, numpy
- **requirements.txt**: pandas>=2.1.0, numpy>=1.26.0

### Pod 2: Data Prep
- **prep.py**: GPU-accelerated feature engineering with RAPIDS
- **Dockerfile**: RAPIDS container (cuDF, cuGraph, cuML)

### Pod 3: Model Build
- **train.py**: Trains GNN embeddings and XGBoost classifier
- **Dockerfile**: PyTorch + RAPIDS + XGBoost

### Pod 4: Inference
- **Dockerfile**: NVIDIA Triton Inference Server
- **config/**: Placeholder for custom Triton configurations

### Pod 5: Notification
- **app.py**: Flask REST API for fraud alerts
- **Dockerfile**: Python slim with Flask and Gunicorn

### Scripts
- **build_all.sh**: Automated build script for all containers

## Environment Variables

### Pod 1 Configuration
```bash
# Template source
TEMPLATE_MOUNT=/mnt/datasets/kaggle/creditcardfraud
TEMPLATE_DIR=/mnt/datasets/kaggle/creditcardfraud
TEMPLATE_FILE=creditcard.csv

# Output destination
FB_OUTPUT_MOUNT=/mnt/fsaai-shared/ebiser/fraud-data
OUTPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data

# Performance tuning
NUM_WORKERS=128          # Parallel worker threads
DURATION_SECONDS=300     # Test duration (5 minutes)
CHUNK_SIZE=10000         # Rows per write operation
```

### General Configuration
```bash
# Storage paths
FB_MOUNT=/mnt/fsaai-shared/ebiser
FA_MOUNT=~/ebiser/nvidia.financial.fraud.detection

# S3 (optional)
S3_ENDPOINT=https://fb-array.example.com
S3_ACCESS_KEY=your-access-key
S3_SECRET_KEY=your-secret-key
S3_BUCKET=fraud-detection-bucket
```

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd nvidia-fraud-detection-pipeline

# 2. Configure environment
cp .env.example .env
# Edit .env with your storage paths

# 3. Ensure Kaggle dataset is available
ls /mnt/datasets/kaggle/creditcardfraud/creditcard.csv

# 4. Build all containers
make build

# 5. Run FlashBlade stress test
docker-compose up data-gather

# 6. Monitor throughput
docker-compose logs -f data-gather

# 7. Run full pipeline
make up
```

## Development Workflow

### Running Individual Pods

```bash
# Run FlashBlade stress test only
docker-compose up data-gather

# Custom stress test configuration
NUM_WORKERS=256 DURATION_SECONDS=600 docker-compose up data-gather

# Run data preparation (requires generated data)
docker-compose up data-prep

# Run model training (requires prepared features)
docker-compose up model-build

# Start inference and notification services
docker-compose up inference notification
```

### Debugging

```bash
# View logs for specific pod
docker-compose logs -f data-gather

# Execute shell in running container
docker exec -it fraud-detection-gather bash

# Check output files
ls -la /mnt/fsaai-shared/ebiser/fraud-data/

# Monitor disk I/O during stress test
iostat -x 1
```

## Pod 1 Output Format

Each worker generates CSV files with the Kaggle creditcard.csv schema:

| Column | Type | Description |
|--------|------|-------------|
| Time | float64 | Seconds elapsed (0-172800) |
| V1-V28 | float64 | PCA-transformed features |
| Amount | float64 | Transaction amount |
| Class | int64 | Fraud label (0=normal, 1=fraud) |

Example output:
```csv
Time,V1,V2,...,V28,Amount,Class
45623.5,-1.359,-0.072,...,0.015,149.62,0
45624.1,1.191,0.266,...,-0.189,2.69,0
```

## Next Steps

1. Download Kaggle creditcard.csv to template directory
2. Run Pod 1 stress test to verify FlashBlade throughput
3. Review generated data schema compatibility
4. Adjust Pod 2 to read from `/fraud-data/` output
5. Train models and deploy to Triton
6. Set up monitoring dashboards for production
