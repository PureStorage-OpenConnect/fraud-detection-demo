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
│   ├── data-gather/                  # Pod 1: Data Generation
│   │   ├── Dockerfile
│   │   ├── gather.py                 # Transaction data generator
│   │   └── requirements.txt
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

### FlashBlade (FB) - Parallel I/O
```
/mnt/fsaai-shared/ebiser/
├── raw_data/
│   ├── transactions_YYYYMMDD_HHMMSS.csv
│   └── metadata_YYYYMMDD_HHMMSS.json
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
├── raw_archives/
│   └── transactions_YYYYMMDD_HHMMSS.csv
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
- **docker-compose.yaml**: Orchestrates all 5 pods with GPU allocation
- **Makefile**: Convenience commands for build/run operations

### Pod 1: Data Gather
- **gather.py**: Generates synthetic transaction data with fraud labels
- **Dockerfile**: Python 3.10 slim with pandas, numpy, boto3

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

## Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd nvidia-fraud-detection-pipeline

# 2. Configure environment
cp .env.example .env
# Edit .env with your storage paths and S3 credentials

# 3. Build all containers
make build

# 4. Start pipeline
make up

# 5. Monitor logs
make logs

# 6. Test services
make test
```

## Development Workflow

### Running Individual Pods

```bash
# Run data generation only
docker-compose up data-gather

# Run data preparation (requires data-gather output)
docker-compose up data-prep

# Run model training (requires data-prep output)
docker-compose up model-build

# Start inference and notification services
docker-compose up inference notification
```

### Debugging

```bash
# View logs for specific pod
docker-compose logs -f data-prep

# Execute shell in running container
docker exec -it fraud-detection-prep bash

# Check GPU allocation
docker exec -it fraud-detection-prep nvidia-smi
```

## Next Steps

1. Review and test each Python script with your data
2. Adjust hyperparameters in training scripts
3. Customize Triton inference configurations
4. Add monitoring and alerting integrations
5. Implement production security measures
6. Set up CI/CD pipeline
