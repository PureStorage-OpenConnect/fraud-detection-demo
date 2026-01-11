# Fraud Detection Demo - Terraform Infrastructure

This Terraform configuration deploys the Fraud Detection Demo v2 on AWS.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AWS Cloud                             │
│  ┌───────────────────────────────────────────────────────┐  │
│  │                    VPC (10.0.0.0/16)                  │  │
│  │                                                        │  │
│  │  ┌─────────────────┐    ┌─────────────────┐          │  │
│  │  │  Public Subnet  │    │  Public Subnet  │          │  │
│  │  │   10.0.1.0/24   │    │   10.0.2.0/24   │          │  │
│  │  │                 │    │                 │          │  │
│  │  │ ┌─────────────┐ │    │                 │          │  │
│  │  │ │g4dn.xlarge  │ │    │                 │          │  │
│  │  │ │ GPU (T4)    │ │    │                 │          │  │
│  │  │ │             │ │    │                 │          │  │
│  │  │ │ - Dashboard │ │    │                 │          │  │
│  │  │ │ - Workers   │ │    │                 │          │  │
│  │  │ │ - Triton    │ │    │                 │          │  │
│  │  │ └─────────────┘ │    │                 │          │  │
│  │  └────────┬────────┘    └─────────────────┘          │  │
│  │           │                                           │  │
│  │  ┌────────▼────────┐                                  │  │
│  │  │ Internet Gateway│                                  │  │
│  │  └─────────────────┘                                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

1. AWS CLI configured with spear profile
2. Terraform >= 1.0 installed
3. Access to the PureStorage-Demo AWS account (808491096064)

## Quick Start

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Deploy infrastructure
terraform apply

# Get connection info
terraform output
```

## Connecting to the Instance

After deployment:

```bash
# SSH into the instance (key is auto-generated)
ssh -i fraud-demo-key.pem ubuntu@<public_ip>

# Or use the output command
$(terraform output -raw ssh_command)
```

## Starting the Demo

Once connected to the instance:

```bash
# Check setup progress
tail -f /var/log/user-data.log

# Navigate to project
cd /home/ubuntu/fraud-detection-demo/v2

# Start containers
docker compose up --build -d

# Check status
docker compose ps
```

Then access the dashboard at: `http://<public_ip>:8080`

## Costs

| Resource | Type | Est. Cost/Hour |
|----------|------|----------------|
| GPU Instance | g4dn.xlarge | $0.526 |
| EBS (root) | 50GB gp3 | ~$0.01 |
| EBS (data) | 100GB gp3 | ~$0.02 |
| NAT Gateway | per hour | $0.045 |
| Elastic IP | while attached | $0.00 |

**Estimated total: ~$0.60/hour** (varies by region)

## Tear Down

```bash
# Destroy all resources
terraform destroy
```

## Configuration Options

Edit `terraform.tfvars` to customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `gpu_instance_type` | g4dn.xlarge | GPU instance type |
| `enable_spot_instance` | false | Use spot for savings |
| `data_volume_size` | 100 | EBS data volume in GB |
| `allowed_ssh_cidrs` | 0.0.0.0/0 | Restrict SSH access |

## Files

| File | Purpose |
|------|---------|
| `provider.tf` | AWS provider and data sources |
| `variables.tf` | Input variables |
| `vpc.tf` | VPC and networking |
| `security.tf` | Security groups and IAM |
| `ec2.tf` | GPU EC2 instance |
| `outputs.tf` | Output values |
| `terraform.tfvars` | Variable values |
