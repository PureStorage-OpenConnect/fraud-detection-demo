# Triton Inference API - Sample Curl Commands

Both CPU and GPU models are served by Triton on the same port. The only difference is the model name in the URL.

## Quick Reference

| Endpoint | Model | Use Case |
|----------|-------|----------|
| `/v2/models/fraud_xgboost_cpu/infer` | CPU | Lower latency for small batches |
| `/v2/models/fraud_xgboost_gpu/infer` | GPU | Higher throughput for large batches |

## Health Checks

```bash
# Server ready?
curl http://localhost:8000/v2/health/ready

# Server alive?
curl http://localhost:8000/v2/health/live

# List all models
curl http://localhost:8000/v2/models

# CPU model status
curl http://localhost:8000/v2/models/fraud_xgboost_cpu

# GPU model status
curl http://localhost:8000/v2/models/fraud_xgboost_gpu
```

## Single Transaction Inference

The input requires 21 features in this order:
```
amt, lat, long, city_pop, unix_time, merch_lat, merch_long, merch_zipcode, zip,
amt_log, amt_scaled, hour_of_day, day_of_week, is_weekend, is_night, distance_km,
category_encoded, state_encoded, gender_encoded, city_pop_log, zip_region
```

### CPU Model

```bash
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_cpu/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [1, 21],
      "datatype": "FP32",
      "data": [100.0, 35.5, -90.2, 150000, 1704100000, 35.6, -90.3, 30301, 30301, 4.6, 0.5, 14, 2, 0, 0, 12.5, 1, 3, 1, 11.9, 3]
    }]
  }'
```

### GPU Model

```bash
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [1, 21],
      "datatype": "FP32",
      "data": [100.0, 35.5, -90.2, 150000, 1704100000, 35.6, -90.3, 30301, 30301, 4.6, 0.5, 14, 2, 0, 0, 12.5, 1, 3, 1, 11.9, 3]
    }]
  }'
```

## Batch Inference (3 transactions)

```bash
# CPU Model - Batch of 3
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_cpu/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [3, 21],
      "datatype": "FP32",
      "data": [
        50.0, 35.5, -90.2, 150000, 1704100000, 35.6, -90.3, 30301, 30301, 3.9, -0.2, 14, 2, 0, 0, 12.5, 1, 3, 1, 11.9, 3,
        5000.0, 35.5, -90.2, 150000, 1704100000, 45.6, -80.3, 30301, 30301, 8.5, 3.5, 3, 6, 1, 1, 1500.0, 5, 15, 0, 11.9, 3,
        2500.0, 25.0, -120.0, 5000, 1704100000, 48.0, -70.0, 90210, 10001, 7.8, 2.8, 2, 0, 0, 1, 5000.0, 3, 0, 1, 8.5, 1
      ]
    }]
  }'

# GPU Model - Batch of 3
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{
      "name": "input__0",
      "shape": [3, 21],
      "datatype": "FP32",
      "data": [
        50.0, 35.5, -90.2, 150000, 1704100000, 35.6, -90.3, 30301, 30301, 3.9, -0.2, 14, 2, 0, 0, 12.5, 1, 3, 1, 11.9, 3,
        5000.0, 35.5, -90.2, 150000, 1704100000, 45.6, -80.3, 30301, 30301, 8.5, 3.5, 3, 6, 1, 1, 1500.0, 5, 15, 0, 11.9, 3,
        2500.0, 25.0, -120.0, 5000, 1704100000, 48.0, -70.0, 90210, 10001, 7.8, 2.8, 2, 0, 0, 1, 5000.0, 3, 0, 1, 8.5, 1
      ]
    }]
  }'
```

## Example Response

```json
{
  "model_name": "fraud_xgboost_gpu",
  "model_version": "1",
  "outputs": [{
    "name": "output__0",
    "datatype": "FP32",
    "shape": [1, 1],
    "data": [0.0234567]
  }]
}
```

**Interpreting the score:**
- `< 0.1` = Low risk (normal transaction)
- `0.1 - 0.5` = Medium risk (review recommended)  
- `> 0.5` = High risk (likely fraud)

## Test Scenarios

### Normal Transaction (low risk)
```bash
# $50 purchase, close to home, daytime, weekday
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input__0", "shape": [1, 21], "datatype": "FP32", 
       "data": [50.0, 35.5, -90.2, 150000, 1704100000, 35.6, -90.3, 30301, 30301, 3.9, -0.2, 14, 2, 0, 0, 12.5, 1, 3, 1, 11.9, 3]}]}'
```

### Suspicious Transaction (higher risk)
```bash
# $2500 purchase, far from home, 2AM, different state
curl -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
  -H "Content-Type: application/json" \
  -d '{"inputs": [{"name": "input__0", "shape": [1, 21], "datatype": "FP32",
       "data": [2500.0, 25.0, -120.0, 5000, 1704100000, 48.0, -70.0, 90210, 10001, 7.8, 2.8, 2, 0, 0, 1, 5000.0, 3, 0, 1, 8.5, 1]}]}'
```

## Latency Comparison

```bash
# Compare CPU vs GPU latency with 100 requests
echo "CPU Model:"
time for i in {1..100}; do
  curl -s -X POST http://localhost:8000/v2/models/fraud_xgboost_cpu/infer \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"name": "input__0", "shape": [1, 21], "datatype": "FP32", "data": [100.0, 35.0, -90.0, 50000, 1704067200, 35.1, -90.1, 12345, 30301, 4.6, 0.5, 12, 3, 0, 0, 10.5, 1, 10, 1, 10.8, 3]}]}' > /dev/null
done

echo "GPU Model:"
time for i in {1..100}; do
  curl -s -X POST http://localhost:8000/v2/models/fraud_xgboost_gpu/infer \
    -H "Content-Type: application/json" \
    -d '{"inputs": [{"name": "input__0", "shape": [1, 21], "datatype": "FP32", "data": [100.0, 35.0, -90.0, 50000, 1704067200, 35.1, -90.1, 12345, 30301, 4.6, 0.5, 12, 3, 0, 0, 10.5, 1, 10, 1, 10.8, 3]}]}' > /dev/null
done
```

## gRPC (Port 8001)

For production workloads, gRPC typically provides better performance. Use the Triton Python client:

```python
import tritonclient.grpc as grpcclient
import numpy as np

client = grpcclient.InferenceServerClient(url="localhost:8001")

# Prepare input
data = np.array([[100.0, 35.0, -90.0, 50000, 1704067200, 35.1, -90.1, 
                  12345, 30301, 4.6, 0.5, 12, 3, 0, 0, 10.5, 1, 10, 1, 10.8, 3]], 
                dtype=np.float32)

inputs = [grpcclient.InferInput("input__0", data.shape, "FP32")]
inputs[0].set_data_from_numpy(data)

# Inference
result = client.infer(model_name="fraud_xgboost_gpu", inputs=inputs)
score = result.as_numpy("output__0")[0][0]
print(f"Fraud score: {score:.6f}")
```

## Prometheus Metrics (Port 8002)

```bash
# Get all metrics
curl http://localhost:8002/metrics

# Key metrics to watch:
# - nv_inference_request_success: Successful inferences
# - nv_inference_request_failure: Failed inferences
# - nv_inference_compute_infer_duration_us: Inference latency
# - nv_gpu_utilization: GPU utilization (GPU model only)
```