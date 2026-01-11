# Terraform Outputs

output "account_id" {
  description = "AWS Account ID"
  value       = data.aws_caller_identity.current.account_id
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "gpu_instance_id" {
  description = "GPU EC2 Instance ID"
  value       = aws_instance.gpu.id
}

output "gpu_instance_public_ip" {
  description = "Public IP of the GPU instance"
  value       = aws_eip.gpu.public_ip
}

output "gpu_instance_private_ip" {
  description = "Private IP of the GPU instance"
  value       = aws_instance.gpu.private_ip
}

output "ssh_command" {
  description = "SSH command to connect to the GPU instance"
  value       = "ssh -i ${path.module}/${var.key_name}.pem ubuntu@${aws_eip.gpu.public_ip}"
}

output "dashboard_url" {
  description = "URL to access the Fraud Detection Dashboard"
  value       = "http://${aws_eip.gpu.public_ip}:8080"
}

output "triton_http_url" {
  description = "URL for Triton HTTP endpoint"
  value       = "http://${aws_eip.gpu.public_ip}:8000"
}

output "triton_grpc_endpoint" {
  description = "Triton gRPC endpoint"
  value       = "${aws_eip.gpu.public_ip}:8001"
}

output "private_key_file" {
  description = "Path to the private key file"
  value       = "${path.module}/${var.key_name}.pem"
}

output "ami_used" {
  description = "AMI used for the GPU instance"
  value       = data.aws_ami.deep_learning_gpu.id
}

output "ami_name" {
  description = "Name of the AMI used"
  value       = data.aws_ami.deep_learning_gpu.name
}

output "s3_data_bucket" {
  description = "S3 bucket for demo data"
  value       = aws_s3_bucket.demo_data.bucket
}

output "s3_data_bucket_arn" {
  description = "S3 bucket ARN"
  value       = aws_s3_bucket.demo_data.arn
}

# Connection instructions
output "connection_instructions" {
  description = "Instructions to connect and start the demo"
  value       = <<-EOT

    ============================================================
    FRAUD DETECTION DEMO - CONNECTION INSTRUCTIONS
    ============================================================

    1. SSH into the instance:
       ssh -i ${path.module}/${var.key_name}.pem ubuntu@${aws_eip.gpu.public_ip}

    2. Wait for setup to complete (check /var/log/user-data.log):
       tail -f /var/log/user-data.log

    3. Start the demo:
       cd /home/ubuntu/fraud-detection-demo/v2
       docker compose up --build -d

    4. Access the dashboard:
       ${aws_eip.gpu.public_ip}:8080

    5. Check container status:
       docker compose ps
       docker compose logs -f

    ============================================================
  EOT
}
