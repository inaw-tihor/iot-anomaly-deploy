#!/usr/bin/env bash
set -euo pipefail
# Simple validation checks for replication and DR
. ../infra/ansible/group_vars/all.yml || true

echo "Checking S3 DR bucket contents..."
aws s3 ls s3://${s3_data_lake_bucket_dr} --region ${aws_dr_region} --profile ${aws_profile} || true

echo "Describe DynamoDB (primary)..."
aws dynamodb describe-table --table-name ${dynamodb_table} --region ${aws_primary_region} --profile ${aws_profile} || true

echo "Describe DynamoDB (DR)..."
aws dynamodb describe-table --table-name ${dynamodb_table} --region ${aws_dr_region} --profile ${aws_profile} || true

echo "List SageMaker models (DR)..."
aws sagemaker list-models --region ${aws_dr_region} --profile ${aws_profile} || true

echo "Kinesis streams (DR)..."
aws kinesis list-streams --region ${aws_dr_region} --profile ${aws_profile} || true

echo "Validation complete."
