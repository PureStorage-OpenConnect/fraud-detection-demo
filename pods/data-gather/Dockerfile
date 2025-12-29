FROM python:3.11-slim

LABEL maintainer="NVIDIA Fraud Detection Pipeline"
LABEL description="High-performance data generator for FlashBlade stress testing"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gather.py .
RUN chmod +x gather.py

ENV TEMPLATE_DIR=/mnt/datasets/kaggle/creditcardfraud
ENV TEMPLATE_FILE=creditcard.csv
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV NUM_WORKERS=128
ENV DURATION_SECONDS=300
ENV CHUNK_SIZE=10000
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pandas; import numpy; print('OK')" || exit 1

# Use exec form without -u flag (PYTHONUNBUFFERED=1 handles it)
CMD ["python", "gather.py"]