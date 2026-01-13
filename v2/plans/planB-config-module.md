# PLAN B: Config Module (Future)

## Overview

Create a configuration module that exposes infrastructure highlights on the dashboard (GPU count/type, storage type, CPU cores).

## Architecture

```
docker-compose.yaml          v2/config/config.yaml
(env vars)                   (static config file)
      │                              │
      └──────────────┬───────────────┘
                     ▼
     dashboard/config.py
     - Loads config.yaml
     - Merges with env vars
     - Exposes /api/config endpoint
                     │
                     ▼
     Frontend (dashboard.js)
     - Fetches /api/config on load
     - Renders dynamic badges/info
```

## Config File Structure

```yaml
infrastructure:
  gpu:
    enabled: true
    count: 4
    type: "NVIDIA A100"
    memory: "40GB"
  cpu:
    cores: 32
    type: "Intel Xeon"
  storage:
    type: "FlashBlade"
    capacity: "100TB"
    throughput: "17GB/s"
  instance:
    type: "p3.8xlarge"
    provider: "AWS"
```

## Storage Options

| Option | Badge Color | Description |
|--------|-------------|-------------|
| `FlashBlade` | #FE5000 (Orange) | High-throughput unstructured data |
| `FlashBlade//S` | #FE5000 | Scale-out FlashBlade |
| `FlashArray//X` | #FE5000 | All-flash block storage |
| `AIRI` | #FE5000 | AI-Ready Infrastructure |
| `SSD` | #666666 (Gray) | Generic SSD storage |
| `Local` | #444444 | Local disk storage |

## GPU Options

| Option | Badge Color |
|--------|-------------|
| `NVIDIA A100` | #76B900 (Green) |
| `NVIDIA H100` | #76B900 |
| `NVIDIA V100` | #76B900 |
| `NVIDIA T4` | #76B900 |
| `CPU Only` | #666666 (Gray) |

## Preset Configs

- `v2/config/presets/airi-dgx.yaml` - AIRI + DGX
- `v2/config/presets/flashblade-4gpu.yaml` - FlashBlade + 4 GPU
- `v2/config/presets/flasharray-inference.yaml` - FlashArray inference
- `v2/config/presets/local-cpu-only.yaml` - Local dev
- `v2/config/presets/aws-current.yaml` - Current AWS t3.xlarge

## Files to Create/Modify

| File | Action |
|------|--------|
| `v2/config/config.yaml` | Create |
| `v2/config/presets/*.yaml` | Create |
| `v2/dashboard/config.py` | Create |
| `v2/dashboard/app.py` | Add /api/config endpoint |
| `v2/dashboard/static/js/dashboard.js` | Fetch and display config |
| `v2/dashboard/templates/index.html` | Dynamic badge containers |
| `v2/docker-compose.yaml` | Mount config volume |
