# financial-fraud-demo
Financial Fraud Detection: 5-Tier Container Architecture (Dual L40S Optimized)

This document outlines the proposed re-architecture of the NVIDIA Financial Fraud Detection blueprint into five scalable microservices, designed for deployment on a Kubernetes cluster leveraging dual NVIDIA L40S GPUs.

The workflow is strictly mapped to five decoupled, containerized services.

0. Data Gathering Service (data-gather-service)

This service is the entry point, responsible for ingesting or generating the raw transactional data.

Component

Description

Primary Goal

Simulate pulling raw transaction data from source systems/APIs and saving it to persistent storage.

Input

N/A (Internal generation or external API access).

Output

Raw transaction data (CSV/Parquet) saved to shared storage.

Key Technologies

Python, Standard I/O operations.

Scalability Model

Designed as a Kubernetes Job that runs first, or an adapter for streaming platforms.

1. Data Preparation Service (data-prep-service)

This service transforms raw data into graph structures and features, maximizing use of the L40S GPUs to test Pure Storage I/O bandwidth.

Component

Description

Primary Goal

High-speed ingestion of raw data, feature engineering, and building the graph structure.

GPU Utilization

Dual L40S (2x): Used to saturate I/O and accelerate processing via cuDF/cuGraph.

Input

Raw transaction data from data-gather-service.

Output

1. Graph Artifacts: Node features, Edge lists, Adjacency matrices (cuGraph-ready format). 2. Tabular Features: XGBoost-ready features for subsequent training.

Key Technologies

Python, RAPIDS (cuDF, cuGraph).

Scalability Model

Designed as a Kubernetes Job with high GPU limits (2x L40S).

2. Model Building Service (model-build-service)

This service consumes the prepared artifacts and executes the GNN and XGBoost training steps.

Component

Description

Primary Goal

Load prepared features, train the GNN for embeddings, and train the final XGBoost model in a distributed fashion.

GPU Utilization

Dual L40S (2x): Utilized for distributed training (GNNs via PyTorch/TF, cuXGBoost).

Input

Graph Artifacts and Tabular Features from the data-prep-service.

Output

1. GNN Model: Serialized GNN model. 2. XGBoost Model: Serialized XGBoost model. 3. Triton Model Repository: Optimized configuration files.

Key Technologies

Python, PyTorch/TensorFlow (for GNN), RAPIDS (cuML, cuXGBoost).

Scalability Model

Designed as a Kubernetes Job with high GPU resource limits (2x L40S).

3. Inference Service (inference-service)

This service hosts the trained models for real-time, low-latency scoring using Triton.

Component

Description

Primary Goal

Real-time serving of transaction fraud scores with maximum throughput.

GPU Utilization

Dual L40S (2x): Utilized by Triton for parallel execution and optimized serving (TensorRT).

Input

Real-time transaction data (via REST/gRPC API request).

Output

Fraud prediction score/label. Initiates notification callback on high-risk scores.

Key Technologies

NVIDIA Triton Inference Server, TensorRT.

Scalability Model

Designed as a Kubernetes Deployment/Service, autoscaling based on request load (HPA).

4. Notification Service (notification-service)

This service receives alerts from the Inference Service and simulates real-time stream processing.

Component

Description

Primary Goal

Provide a reliable webhook endpoint to receive high-risk transaction alerts and stream them to downstream consumers (e.g., fraud investigation team, Kafka queue).

Input

JSON payload of high-risk transaction (transaction_id, fraud_score) from inference-service.

Output

Streamed log output (mocking Kafka/Queue insertion).

Key Technologies

Python, Flask/FastAPI (Web Service).

Scalability Model

Standard Kubernetes Deployment/Service (no GPU needed).

Data Flow and Persistence

Model artifacts and prepared data must be shared between services using a shared Persistent Volume Claim (PVC) mounted across all pods at /data, which maps to your Pure Storage mounts.