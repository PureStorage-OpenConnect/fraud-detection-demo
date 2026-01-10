FROM nvcr.io/nvidia/rapidsai/base:24.02-cuda12.0-py3.11

WORKDIR /app

# Install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY worker.py .

# Create directories
RUN mkdir -p /data /models

ENV PYTHONUNBUFFERED=1

EXPOSE 5002

CMD ["python", "worker.py"]
