FROM nvcr.io/nvidia/pytorch:23.10-py3

LABEL maintainer="your.email@example.com"
LABEL description="Pod 3: Model Build Service for NVIDIA Fraud Detection Pipeline"

# Set working directory
WORKDIR /app

# Install RAPIDS and XGBoost
RUN pip install --no-cache-dir \
    cudf-cu12==23.10.* \
    cuml-cu12==23.10.* \
    cugraph-cu12==23.10.* \
    xgboost==2.0.2 \
    torch-geometric==2.4.0 \
    boto3==1.34.34 \
    pyarrow==14.0.1

# Copy application files
COPY train.py /app/

# Set environment variables
ENV FB_MOUNT=/mnt/fsaai-shared/ebiser
ENV FA_MOUNT=/root/ebiser/nvidia.financial.fraud.detection
ENV S3_ENDPOINT=""
ENV S3_ACCESS_KEY=""
ENV S3_SECRET_KEY=""
ENV S3_BUCKET=""
ENV FEATURES_FILE=""
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1

# Run the application
CMD ["python", "train.py"]
