

#!/usr/bin/env bash
set -euo pipefail

# ----- config -----
TENANT_ID="tenant0011"
AWS_PRIMARY_REGION="us-east-1"
AWS_PROFILE="default"

S3_BUCKET="iot-telemetry-${TENANT_ID}-${AWS_PRIMARY_REGION}"
S3_PREFIX_STATIC="iot/tenants/${TENANT_ID}/telemetry/"
S3_KEY_TEMPLATE="${S3_PREFIX_STATIC}\${topic()}/\${timestamp()}.json"

RULE_NAME="to_s3_telemetry_${TENANT_ID}"
ROLE_NAME="iotrule-s3-writer-${TENANT_ID}"
TOPIC_FILTER="tenants/${TENANT_ID}/devices/+/telemetry"

# ----- 1) S3 bucket (create if missing) -----
if [[ "${AWS_PRIMARY_REGION}" == "us-east-1" ]]; then
  aws s3api create-bucket \
    --bucket "${S3_BUCKET}" \
    --region "${AWS_PRIMARY_REGION}" \
    --profile "${AWS_PROFILE}" 2>/dev/null || true
else
  aws s3api create-bucket \
    --bucket "${S3_BUCKET}" \
    --region "${AWS_PRIMARY_REGION}" \
    --create-bucket-configuration LocationConstraint="${AWS_PRIMARY_REGION}" \
    --profile "${AWS_PROFILE}" 2>/dev/null || true
fi

# Block all public access
aws s3api put-public-access-block \
  --bucket "${S3_BUCKET}" \
  --public-access-block-configuration '{
    "BlockPublicAcls":true,
    "IgnorePublicAcls":true,
    "BlockPublicPolicy":true,
    "RestrictPublicBuckets":true
  }' \
  --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}"

# Default encryption = SSE-S3 (AES256)
aws s3api put-bucket-encryption \
  --bucket "${S3_BUCKET}" \
  --server-side-encryption-configuration '{
    "Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]
  }' \
  --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}"

# Lifecycle: move to STANDARD_IA after 30d + abort incomplete MPU after 7d
aws s3api put-bucket-lifecycle-configuration \
  --bucket "${S3_BUCKET}" \
  --lifecycle-configuration '{
    "Rules": [{
      "ID": "to-ia-30d",
      "Status": "Enabled",
      "Filter": {"Prefix": ""},
      "Transitions": [{"Days": 30, "StorageClass": "STANDARD_IA"}],
      "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7}
    }]
  }' \
  --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}"

# ----- 2) IAM role for the IoT rule -----
cat > /tmp/iot_s3_trust.json <<'JSON'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "iot.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
JSON

aws iam create-role \
  --role-name "${ROLE_NAME}" \
  --assume-role-policy-document file:///tmp/iot_s3_trust.json \
  --profile "${AWS_PROFILE}" 2>/dev/null || true

# Least-privilege: allow only PutObject under our static prefix
cat > /tmp/iot_s3_put_policy.json <<JSON
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["s3:PutObject"],
    "Resource": ["arn:aws:s3:::${S3_BUCKET}/${S3_PREFIX_STATIC}*"]
  }]
}
JSON

aws iam put-role-policy \
  --role-name "${ROLE_NAME}" \
  --policy-name "S3WriteTelemetry" \
  --policy-document file:///tmp/iot_s3_put_policy.json \
  --profile "${AWS_PROFILE}"

ROLE_ARN=$(aws iam get-role \
  --role-name "${ROLE_NAME}" \
  --query 'Role.Arn' --output text \
  --profile "${AWS_PROFILE}")

echo "ROLE_ARN=${ROLE_ARN}"

# ----- 3) IoT Topic Rule → S3 -----
# Build the rule payload (note the escaped ${topic()} & ${timestamp()})
cat > /tmp/iot_rule_payload.json <<JSON
{
  "sql": "SELECT * FROM '${TOPIC_FILTER}'",
  "awsIotSqlVersion": "2016-03-23",
  "ruleDisabled": false,
  "actions": [{
    "s3": {
      "roleArn": "${ROLE_ARN}",
      "bucketName": "${S3_BUCKET}",
      "key": "${S3_KEY_TEMPLATE}"
    }
  }]
}
JSON

# Create or replace the rule idempotently
if aws iot get-topic-rule --rule-name "${RULE_NAME}" \
     --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}" >/dev/null 2>&1; then
  aws iot replace-topic-rule \
    --rule-name "${RULE_NAME}" \
    --topic-rule-payload file:///tmp/iot_rule_payload.json \
    --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}"
else
  aws iot create-topic-rule \
    --rule-name "${RULE_NAME}" \
    --topic-rule-payload file:///tmp/iot_rule_payload.json \
    --region "${AWS_PRIMARY_REGION}" --profile "${AWS_PROFILE}"
fi

echo "IoT rule '${RULE_NAME}' is set to write to s3://${S3_BUCKET}/${S3_PREFIX_STATIC}..."
echo "Listening on topic filter: ${TOPIC_FILTER}"
