# NVIDIA Financial Fraud Detection Pipeline

[![NVIDIA](https://img.shields.io/badge/NVIDIA-L40S-76B900?logo=nvidia)](https://www.nvidia.com/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![RAPIDS](https://img.shields.io/badge/RAPIDS-cuDF%20%7C%20cuGraph-blueviolet)](https://rapids.ai/)
[![FlashArray](https://img.shields.io/badge/Pure_Storage-FlashArray-FF6600)](https://www.purestorage.com/products/unified-block-file-storage.html)
[![FlashBlade](https://img.shields.io/badge/Pure_Storage-FlashBlade-FF6600)](https://www.purestorage.com/products/unstructured-data-storage/flashblade-s.html)

## Overview

A containerized fraud detection pipeline optimized for dual NVIDIA L40S GPUs. This project re-architects the NVIDIA Financial Fraud Detection AI Blueprint into 5 independent Docker containers that work together to process transactions, train models, and detect fraud in real-time.

**Original Blueprint**: [NVIDIA Financial Fraud Detection](https://github.com/NVIDIA-AI-Blueprints/Financial-Fraud-Detection)

---

## System Architecture

```mermaid
graph TB
    A[Pod 1<br/>Data Gather] -->|CSV| C[Pod 2<br/>Data Prep]
    C -->|Features| D[Pod 3<br/>Model Build]
    D -->|Models| E[Pod 4<br/>Inference]
    E -->|Alerts| F[Pod 5<br/>Notification]
    
    A -.->|Write Raw Data| FB
    A -.->|Archive| S3
    C -.->|Read/Write Features| FB
    D -.->|Read Training Data| FB
    D -.->|Write Models| FA
    D -.->|Version Models| S3
    E -.->|Load Models| FA
    FB <-.->|Integrated| S3
    
    style A fill:#1a5490,stroke:#333,stroke-width:2px,color:#fff
    style C fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style D fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style E fill:#d85e00,stroke:#333,stroke-width:2px,color:#fff
    style F fill:#1a5490,stroke:#333,stroke-width:2px,color:#fff
    style FA fill:#ffb366,stroke:#333,stroke-width:2px,color:#fff
    style FB fill:#ffb366,stroke:#333,stroke-width:2px,color:#fff
    style S3 fill:#ffb366,stroke:#333,stroke-width:2px,color:#fff
```

---

## 5-Pod Architecture

| Pod | Container | GPU | Storage | Purpose |
|-----|-----------|-----|---------|---------|
| 1 | `data-gather` | No | FB + S3 | Generate/ingest raw transaction data |
| 2 | `data-prep` | 2x L40S | FB | GPU-accelerated feature engineering (RAPIDS) |
| 3 | `model-build` | 2x L40S | FB + FA + S3 | Train GNN and XGBoost models |
| 4 | `inference` | 2x L40S | FA | Real-time fraud detection (Triton Server) |
| 5 | `notification` | No | None | Handle fraud alerts via webhook |

**Data Flow**: Raw data (FB) → Prepared features (FB) → Trained models (FA) → Real-time predictions → Alerts

**Storage Strategy**:

This architecture leverages Pure Storage's dual-protocol approach with optimized storage placement for different workload characteristics:

### File Storage (NFS Mounts)
- **FA**: `/root/ebiser/nvidia.financial.fraud.detection`
  - **Protocol**: NFS file mount
  - **Optimized for**: Low-latency random I/O (<1ms read latency)
  - **Use case**: Real-time model serving where inference requests require immediate model access
  - **Pods**: Pod 3 (writes models), Pod 4 (reads models for serving)
  
- **FB**: `/mnt/fsaai-shared/ebiser`
  - **Protocol**: NFS file mount
  - **Optimized for**: High-throughput parallel I/O (>5GB/s)
  - **Use case**: Bulk data processing where multiple GPU workers read/write large datasets simultaneously
  - **Pods**: Pod 1 (writes raw data), Pod 2 (reads/writes features), Pod 3 (reads training data)

### Object Storage (S3 Protocol)
- **FB S3 Endpoint**: `s3://fraud-detection-bucket`
  - **Protocol**: S3 API on FlashBlade
  - **Optimized for**: Archival, versioning, and cross-region access
  - **Use case**: Long-term storage of raw data archives and model versions for compliance and rollback
  - **Pods**: Pod 1 (archives raw data), Pod 3 (versions trained models)

**Mount Configuration**:
```bash
# FlashArray (FA) - Low Latency NFS Mount
mount -t nfs fa-array.example.com:/volume/fraud-models \
  ~/ebiser/nvidia.financial.fraud.detection

# FlashBlade (FB) - High Throughput NFS Mount  
mount -t nfs fb-array.example.com:/export/fraud-data \
  /mnt/fsaai-shared/ebiser

# FlashBlade S3 - Configure endpoint in .env
S3_ENDPOINT=https://fb-array.example.com
```

This separation ensures that high-throughput ETL operations (data prep, feature engineering) don't interfere with latency-sensitive inference serving, while S3 provides durable archival storage.

---

## Infrastructure and Technology Stack

- **GPUs**: 2x NVIDIA L40S (48GB each)
- **Data Processing**: RAPIDS (cuDF, cuGraph)
- **ML Training**: cuXGBoost, PyTorch
- **Inference**: NVIDIA Triton Inference Server
- **Orchestration**: Docker Compose
- **Storage**: 
  - **FA (FlashArray X70R3)**: Low-latency file storage
  - **FB (FlashBlade S200)**: Parallel I/O, file + S3 protocol
  - **S3**: Object storage for archival and versioning

---

## Quick Start

### Prerequisites

```bash
# Required Hardware
- 2x NVIDIA L40S GPUs (48GB each)
- 1024 GB RAM (512 GB per CPU)
- Pure Storage FlashArray (FA)
- Pure Storage FlashBlade (FB)

# Required Software
- Ubuntu 22.04.5 LTS
- NVIDIA Driver >= 550.x (CUDA 12.4)
- Docker >= 24.x
- Docker Compose >= 2.x
- NVIDIA Container Toolkit

# Verify GPU access
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/nvidia-fraud-detection-pipeline.git
cd nvidia-fraud-detection-pipeline

# Configure storage mount points
export FA_MOUNT=~/ebiser/nvidia.financial.fraud.detection
export FB_MOUNT=/mnt/fsaai-shared/ebiser

# Create required directories
mkdir -p $FB_MOUNT/{raw_data,prep_output}
mkdir -p $FA_MOUNT/model_repository

# Build all containers
docker-compose build

# Start the pipeline
docker-compose up
```

---

## Project Structure

```
nvidia-fraud-detection-pipeline/
├── docker-compose.yaml           # Container orchestration
├── pods/
│   ├── data-gather/
│   │   ├── Dockerfile
│   │   └── gather.py
│   ├── data-prep/
│   │   ├── Dockerfile
│   │   └── prep.py
│   ├── model-build/
│   │   ├── Dockerfile
│   │   └── train.py
│   ├── inference/
│   │   ├── Dockerfile
│   │   └── config/
│   └── notification/
│       ├── Dockerfile
│       └── app.py
└── README.md

# Storage Mounts (Pure Storage)
FA: ~/ebiser/nvidia.financial.fraud.detection/
    └── model_repository/              # Low-latency model storage
        ├── fraud_gnn/
        └── fraud_xgboost/

FB: /mnt/fsaai-shared/ebiser/
    ├── raw_data/                      # High-throughput data ingestion
    └── prep_output/                   # Parallel feature processing

S3: s3://fraud-detection-bucket/       # Archival and versioning
    ├── raw_archives/
    └── model_versions/
```

---

## Usage

See `docker-compose.yaml` for complete container orchestration configuration.

### Run Complete Pipeline

```bash
# Start all containers
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f
```

### Run Individual Pods

```bash
# Pod 1: Generate data
docker-compose up data-gather

# Pod 2: Prepare features (requires data from Pod 1)
docker-compose up data-prep

# Pod 3: Train models (requires data from Pod 2)
docker-compose up model-build

# Pods 4 & 5: Start inference and notification services
docker-compose up inference notification
```

### Test Inference Endpoint

```bash
# Check Triton server health
curl http://localhost:8002/v2/health/ready

# Check notification service
curl http://localhost:5000/health

# Send test transaction (once models are trained)
curl -X POST http://localhost:8000/v2/models/fraud_xgboost/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [1, 50],
      "datatype": "TYPE_FP32",
      "data": [0.5, 0.3, 0.8, ...]
    }]
  }'
```

---

## Docker Compose Configuration

```yaml
version: '3.8'

services:
  data-gather:
    build: ./pods/1-data-gather
    volumes:
      - ./data:/data
    
  data-prep:
    build: ./pods/2-data-prep
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    
  model-build:
    build: ./pods/3-model-build
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    
  inference:
    build: ./pods/4-inference
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    volumes:
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 2
              capabilities: [gpu]
    
  notification:
    build: ./pods/5-notification
    ports:
      - "5000:5000"
```

---

## Data Flow

### Storage Paths

**FlashArray (FA) - Low Latency:**
```bash
~/ebiser/nvidia.financial.fraud.detection/
└── model_repository/
    ├── fraud_gnn/
    │   ├── config.pbtxt
    │   └── 1/model.pt                # Pod 3 writes, Pod 4 reads
    └── fraud_xgboost/
        ├── config.pbtxt
        └── 1/model.json              # Pod 3 writes, Pod 4 reads
```

**FlashBlade (FB) - Parallel I/O:**
```bash
/mnt/fsaai-shared/ebiser/
├── raw_data/
│   └── transactions.csv              # Pod 1 writes
└── prep_output/
    ├── features.parquet              # Pod 2 writes, Pod 3 reads
    ├── graph_nodes.csv
    └── graph_edges.csv
```

**S3 - Archival & Versioning:**
```bash
s3://fraud-detection-bucket/
├── raw_archives/
│   └── transactions_2024-12-01.csv   # Pod 1 archives
└── model_versions/
    ├── fraud_gnn_v1.0.tar.gz         # Pod 3 versions
    └── fraud_xgboost_v1.0.tar.gz
```

### Pipeline Flow

1. **Pod 1** generates synthetic transactions → **FB** `/raw_data/` + **S3** archive
2. **Pod 2** reads from **FB**, processes with RAPIDS → **FB** `/prep_output/`
3. **Pod 3** reads features from **FB**, trains models → **FA** `/model_repository/` + **FB** **S3** versions
4. **Pod 4** loads models from **FA**, serves predictions via Triton
5. **Pod 5** receives alerts from Pod 4 when fraud detected

---

## Monitoring

```bash
# Check GPU usage
watch -n 1 nvidia-smi

# View container logs
docker-compose logs -f data-prep
docker-compose logs -f inference

# Check Triton metrics
curl http://localhost:8002/metrics

# Monitor resource usage
docker stats

# Check storage I/O performance
# FlashBlade throughput
iostat -x 1 /mnt/fsaai-shared/ebiser

# FlashArray latency
iostat -x 1 ~/ebiser/nvidia.financial.fraud.detection
```

---

## Troubleshooting

### GPU not detected

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi

# Verify nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Container fails to start

```bash
# Check logs
docker-compose logs <service-name>

# Rebuild container
docker-compose build --no-cache <service-name>

# Verify storage mounts
ls -la ~/ebiser/nvidia.financial.fraud.detection/
ls -la /mnt/fsaai-shared/ebiser/

# Check mount permissions
sudo chmod -R 755 ~/ebiser/nvidia.financial.fraud.detection/
sudo chmod -R 755 /mnt/fsaai-shared/ebiser/
```

### Out of GPU memory

```bash
# Check GPU memory
nvidia-smi

# Stop other GPU processes
docker-compose down

# Reduce batch size in training configs
```

---

## Performance Targets

> **Note**: These are initial performance targets and will be updated as the project matures and undergoes testing.

| Metric | Target | Storage Component |
|--------|--------|-------------------|
| Data Prep Throughput | > 1M records/sec | FB Parallel I/O |
| Model Training Time | < 30 minutes | FB read, FA write |
| Inference Latency (p99) | < 10ms | FA low-latency reads |
| GPU Utilization | > 85% | Pods 2, 3, 4 |
| Storage Write Speed (FB) | > 5 GB/s | Pod 1, Pod 2 |
| Storage Read Latency (FA) | < 1ms | Pod 4 |

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) file

---

## Contact

**Repository**: [https://github.com/yourusername/nvidia-fraud-detection-pipeline](https://github.com/yourusername/nvidia-fraud-detection-pipeline)

---

**Built for High-Performance Fraud Detection with Docker & NVIDIA L40S GPUs**