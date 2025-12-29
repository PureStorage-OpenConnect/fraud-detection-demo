FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gather.py .

ENV TEMPLATE_DIR=/mnt/datasets/kaggle/creditcardfraud
ENV TEMPLATE_FILE=creditcard.csv
ENV OUTPUT_DIR=/mnt/fsaai-shared/ebiser/fraud-data
ENV NUM_WORKERS=128
ENV DURATION_SECONDS=300
ENV CHUNK_SIZE=2000000
ENV PYTHONUNBUFFERED=1

# stderr handled in Python
CMD ["python", "gather.py"]