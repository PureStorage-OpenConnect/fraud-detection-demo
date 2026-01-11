# AWS Provider Configuration

provider "aws" {
  region  = var.aws_region
  profile = "purestorage-demo"  # Uses assume_role to PureStorage-Demo account (808491096064)

  default_tags {
    tags = local.common_tags
  }
}

# Data source for availability zones
data "aws_availability_zones" "available" {
  state = "available"
}

# Data source for current account
data "aws_caller_identity" "current" {}

# Data source for latest Deep Learning AMI with GPU support
data "aws_ami" "deep_learning_gpu" {
  most_recent = true
  owners      = ["amazon"]

  filter {
    name   = "name"
    values = ["Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) *"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }
}
