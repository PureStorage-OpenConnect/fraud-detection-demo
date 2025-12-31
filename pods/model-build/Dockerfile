FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.10

WORKDIR /app

# Install XGBoost with GPU support and boto3
RUN pip install --no-cache-dir \
    xgboost>=2.0.0 \
    boto3>=1.34.0

COPY train.py /app/

# Default configuration
ENV FB_MOUNT=/mnt/fsaai-shared/ebiser
ENV FA_MOUNT=/workspace/model-output
ENV PREP_OUTPUT_DIR=/mnt/fsaai-shared/ebiser/prep-output
ENV S3_ENDPOINT=""
ENV S3_ACCESS_KEY=""
ENV S3_SECRET_KEY=""
ENV S3_BUCKET=""
ENV FEATURES_FILE=""
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]