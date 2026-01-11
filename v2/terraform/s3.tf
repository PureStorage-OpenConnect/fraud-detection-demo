# S3 Bucket for Demo Data Storage

# Data bucket for synthetic transactions
resource "aws_s3_bucket" "demo_data" {
  bucket = "fraud-detection-v2-demo-data"

  tags = {
    Name = "${var.project_name}-data"
  }
}

# Enable versioning
resource "aws_s3_bucket_versioning" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle rule to clean up old data
resource "aws_s3_bucket_lifecycle_configuration" "demo_data" {
  bucket = aws_s3_bucket.demo_data.id

  rule {
    id     = "cleanup-old-data"
    status = "Enabled"

    filter {
      prefix = "generated-data/"
    }

    # Delete objects older than 90 days
    expiration {
      days = 90
    }

    # Move to cheaper storage after 30 days
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
  }
}

# IAM policy to allow EC2 to access S3
resource "aws_iam_role_policy" "s3_access" {
  name = "${var.project_name}-s3-access"
  role = aws_iam_role.ec2_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.demo_data.arn,
          "${aws_s3_bucket.demo_data.arn}/*"
        ]
      }
    ]
  })
}
