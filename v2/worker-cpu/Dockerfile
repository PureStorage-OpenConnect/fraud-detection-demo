FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker code
COPY worker.py .

# Create directories
RUN mkdir -p /data /models

ENV PYTHONUNBUFFERED=1

EXPOSE 5001

CMD ["python", "worker.py"]
