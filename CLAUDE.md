# Fraud Detection Demo - Claude Notes

## AWS Deployment Information

### Instance Details
- **AWS Account**: PureStorage-Demo (808491096064)
- **Region**: us-west-2
- **Instance IP**: 52.25.179.22
- **Dashboard URL**: http://52.25.179.22:8080
- **Instance Type**: t3.xlarge (CPU only while GPU quota pending)

### SSH Access
Use the following SSH key to connect:
```bash
ssh -i /Users/haranath/genai/fraud-pure/v2/terraform/fraud-demo-key.pem -o StrictHostKeyChecking=no ubuntu@52.25.179.22
```

### AWS Credentials
Use the `spear` profile in `~/.aws/credentials` for AWS CLI operations:
```bash
aws --profile spear <command>
```

### Deployment Commands
To deploy changes to AWS:

1. Commit and push changes to GitHub
2. SSH to instance and pull/rebuild:
```bash
ssh -i /Users/haranath/genai/fraud-pure/v2/terraform/fraud-demo-key.pem ubuntu@52.25.179.22 "cd /home/ubuntu/fraud-detection-demo && git pull && cd v2 && docker compose -f docker-compose.dual-cpu.yaml down && docker compose build --no-cache dashboard worker-cpu && docker compose -f docker-compose.dual-cpu.yaml up -d"
```

### Docker Compose Files
- `docker-compose.yaml` - Standard compose with GPU worker (requires NVIDIA runtime)
- `docker-compose.dual-cpu.yaml` - **Currently in use** - Uses two CPU workers while waiting for GPU quota approval

### GitHub Repository
- **Repo**: PureStorage-OpenConnect/fraud-detection-demo
- **Branch**: spear/CPU-GPU-Comparison-Dashboard
- **Main branch**: main

### Git Authentication on AWS Instance
Git is configured with a PAT (Personal Access Token) stored in ~/.git-credentials on the instance.

### Data Files
Data files are stored in S3 and downloaded to the instance:
- S3 Bucket: `fraud-detection-v2-demo-data`
- Local paths on instance:
  - `/home/ubuntu/fraud-detection-demo/v2/data/cpu/transactions.parquet`
  - `/home/ubuntu/fraud-detection-demo/v2/data/gpu/transactions.parquet`

To download data from S3:
```bash
aws s3 cp s3://fraud-detection-v2-demo-data/generated-data/cpu/transactions.parquet /home/ubuntu/fraud-detection-demo/v2/data/cpu/
aws s3 cp s3://fraud-detection-v2-demo-data/generated-data/gpu/transactions.parquet /home/ubuntu/fraud-detection-demo/v2/data/gpu/
```

## Current Status
- GPU quota still pending approval
- Using dual CPU workers (both worker-cpu and worker-gpu run the same CPU code)
- Dashboard accessible at http://52.25.179.22:8080
