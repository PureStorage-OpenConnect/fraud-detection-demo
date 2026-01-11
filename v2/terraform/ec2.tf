# EC2 GPU Instance for Fraud Detection Demo

# User data script to set up the instance
locals {
  user_data = <<-EOF
    #!/bin/bash
    set -ex

    # Log output
    exec > >(tee /var/log/user-data.log) 2>&1
    echo "Starting user data script at $(date)"

    # Update system
    apt-get update -y
    apt-get upgrade -y

    # Install Docker
    apt-get install -y apt-transport-https ca-certificates curl software-properties-common
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update -y
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

    # Install NVIDIA Container Toolkit
    distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
      tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -y
    apt-get install -y nvidia-container-toolkit

    # Configure Docker to use NVIDIA runtime
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    # Add ubuntu user to docker group
    usermod -aG docker ubuntu

    # Install git and other tools
    apt-get install -y git python3-pip jq htop nvtop

    # Create project directory
    mkdir -p /home/ubuntu/fraud-detection-demo
    chown -R ubuntu:ubuntu /home/ubuntu/fraud-detection-demo

    # Clone the repository
    cd /home/ubuntu
    git clone https://github.com/PureStorage-OpenConnect/fraud-detection-demo.git fraud-detection-demo || true
    cd fraud-detection-demo
    git fetch origin spear/CPU-GPU-Comparison-Dashboard
    git checkout spear/CPU-GPU-Comparison-Dashboard

    # Set ownership
    chown -R ubuntu:ubuntu /home/ubuntu/fraud-detection-demo

    # Create data directories
    mkdir -p /home/ubuntu/fraud-detection-demo/v2/data
    mkdir -p /home/ubuntu/fraud-detection-demo/v2/models
    chown -R ubuntu:ubuntu /home/ubuntu/fraud-detection-demo

    # Generate synthetic data (1M rows)
    cd /home/ubuntu/fraud-detection-demo/v2
    pip3 install pandas pyarrow faker numpy
    python3 scripts/generate_data.py --rows 1000000 --output-dir ./data

    # Pull Docker images (will be built later)
    echo "Instance setup complete at $(date)"
    echo "To start the demo, run: cd /home/ubuntu/fraud-detection-demo/v2 && docker compose up --build -d"
  EOF
}

# GPU Instance
resource "aws_instance" "gpu" {
  ami                    = data.aws_ami.deep_learning_gpu.id
  instance_type          = var.gpu_instance_type
  key_name               = aws_key_pair.main.key_name
  vpc_security_group_ids = [aws_security_group.gpu_instance.id]
  subnet_id              = aws_subnet.public[0].id
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  root_block_device {
    volume_size           = 100  # Deep Learning AMI requires >= 75GB
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  # Data volume for demo data
  ebs_block_device {
    device_name           = "/dev/sdf"
    volume_size           = var.data_volume_size
    volume_type           = "gp3"
    iops                  = 3000
    throughput            = 125
    delete_on_termination = true
    encrypted             = true
  }

  user_data = base64encode(local.user_data)

  # Enable detailed monitoring
  monitoring = true

  # Metadata options (IMDSv2)
  metadata_options {
    http_endpoint               = "enabled"
    http_tokens                 = "required"
    http_put_response_hop_limit = 1
  }

  tags = {
    Name = "${var.project_name}-gpu-instance"
    Type = "gpu"
  }

  lifecycle {
    ignore_changes = [ami]
  }
}

# Elastic IP for GPU instance
resource "aws_eip" "gpu" {
  instance = aws_instance.gpu.id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-gpu-eip"
  }

  depends_on = [aws_internet_gateway.main]
}

# Optional: Spot Instance Request (for cost savings)
resource "aws_spot_instance_request" "gpu_spot" {
  count = var.enable_spot_instance ? 1 : 0

  ami                    = data.aws_ami.deep_learning_gpu.id
  instance_type          = var.gpu_instance_type
  key_name               = aws_key_pair.main.key_name
  vpc_security_group_ids = [aws_security_group.gpu_instance.id]
  subnet_id              = aws_subnet.public[0].id
  iam_instance_profile   = aws_iam_instance_profile.ec2_profile.name

  spot_price           = var.spot_max_price != "" ? var.spot_max_price : null
  wait_for_fulfillment = true
  spot_type            = "one-time"

  root_block_device {
    volume_size           = 100  # Deep Learning AMI requires >= 75GB
    volume_type           = "gp3"
    delete_on_termination = true
    encrypted             = true
  }

  user_data = base64encode(local.user_data)

  tags = {
    Name = "${var.project_name}-gpu-spot"
    Type = "gpu-spot"
  }
}
