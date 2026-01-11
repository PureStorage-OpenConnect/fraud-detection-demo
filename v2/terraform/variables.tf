# Variables for Fraud Detection Demo Infrastructure

variable "project_name" {
  description = "Project name for resource naming"
  type        = string
  default     = "fraud-detection-demo"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "demo"
}

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-west-2"
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "public_subnet_cidrs" {
  description = "CIDR blocks for public subnets"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24"]
}

variable "private_subnet_cidrs" {
  description = "CIDR blocks for private subnets"
  type        = list(string)
  default     = ["10.0.10.0/24", "10.0.11.0/24"]
}

variable "gpu_instance_type" {
  description = "EC2 instance type for GPU workloads"
  type        = string
  default     = "g4dn.xlarge"  # 1x T4 GPU, 4 vCPU, 16GB RAM
}

variable "cpu_instance_type" {
  description = "EC2 instance type for CPU workloads (optional separate instance)"
  type        = string
  default     = "t3.xlarge"  # 4 vCPU, 16GB RAM
}

variable "key_name" {
  description = "SSH key pair name (will be created if not exists)"
  type        = string
  default     = "fraud-demo-key"
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # Restrict in production!
}

variable "data_volume_size" {
  description = "Size of EBS data volume in GB"
  type        = number
  default     = 100
}

variable "enable_spot_instance" {
  description = "Use spot instances for cost savings"
  type        = bool
  default     = false
}

variable "spot_max_price" {
  description = "Maximum spot price (leave empty for on-demand price cap)"
  type        = string
  default     = ""
}

# Tags
variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}

locals {
  common_tags = merge(
    {
      Project     = var.project_name
      Environment = var.environment
      ManagedBy   = "terraform"
      Client      = "PureStorage"
    },
    var.tags
  )
}
