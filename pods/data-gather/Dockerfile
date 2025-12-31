FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gather.py .

ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV NUM_WORKERS=128
ENV DURATION_SECONDS=300
ENV CHUNK_SIZE=500000
ENV FRAUD_RATE=0.005
ENV PYTHONUNBUFFERED=1

CMD ["python", "gather.py"]