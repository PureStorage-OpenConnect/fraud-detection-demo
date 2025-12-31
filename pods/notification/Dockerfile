FROM python:3.10-slim

LABEL description="Pod 5: Notification Service for Fraud Detection Pipeline"

WORKDIR /app

RUN pip install --no-cache-dir \
    flask==3.0.0 \
    gunicorn==21.2.0 \
    requests==2.31.0

COPY app.py /app/

ENV HOST=0.0.0.0
ENV PORT=5000
ENV DEBUG=False
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]