# Pod 5: Notification Service
# REST API for fraud alerts

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir flask gunicorn

COPY app.py .

ENV HOST=0.0.0.0
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]