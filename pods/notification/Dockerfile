FROM python:3.10-slim

LABEL maintainer="your.email@example.com"
LABEL description="Pod 5: Notification Service for NVIDIA Fraud Detection Pipeline"

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    requests==2.31.0

# Copy application files
COPY app.py /app/

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=False
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
