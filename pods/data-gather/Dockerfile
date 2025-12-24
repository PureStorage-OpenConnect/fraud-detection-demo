FROM python:3.11-slim

LABEL maintainer="NVIDIA Fraud Detection Pipeline"
LABEL description="High-performance data generator for FlashBlade stress testing"

# Set working directory
WORKDIR /app

# Install system dependencies for better performance
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gather.py .

# Make script executable
RUN chmod +x gather.py

# Environment variables with defaults
ENV TEMPLATE_DIR=/mnt/datasets/kaggle/creditcardfraud
ENV TEMPLATE_FILE=creditcard.csv
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV NUM_WORKERS=128
ENV DURATION_SECONDS=300
ENV CHUNK_SIZE=10000

# Health check - verify Python and dependencies
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas; import numpy; print('OK')" || exit 1

# Run the data generator
CMD ["python", "-u", "gather.py"]
