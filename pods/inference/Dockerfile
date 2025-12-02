FROM nvcr.io/nvidia/tritonserver:23.10-py3

LABEL maintainer="your.email@example.com"
LABEL description="Pod 4: Inference Service using Triton for NVIDIA Fraud Detection Pipeline"

# Set working directory
WORKDIR /workspace

# Install additional Python dependencies for custom backends
RUN pip install --no-cache-dir \
    requests==2.31.0 \
    numpy==1.26.3

# Copy configuration files
COPY config/ /workspace/config/

# Set environment variables
ENV FA_MOUNT=/root/ebiser/nvidia.financial.fraud.detection
ENV MODEL_REPOSITORY=/root/ebiser/nvidia.financial.fraud.detection/model_repository
ENV NOTIFICATION_ENDPOINT=http://notification:5000/notify/fraud
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0,1

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton Inference Server
CMD ["tritonserver", \
     "--model-repository=${MODEL_REPOSITORY}", \
     "--strict-model-config=false", \
     "--log-verbose=1"]
