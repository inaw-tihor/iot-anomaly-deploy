---
id: prereqs
title: "Prerequisites & Environment Inputs"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [aws cli, ansible, docker, jq, env vars, profiles]
---

# Prerequisites & Environment Inputs

## Tools
```bash
aws --version
ansible --version
docker --version
jq --version
```

## Environment variables
```bash
export AWS_PROFILE="default"
export AWS_PRIMARY_REGION="us-east-1"
export AWS_DR_REGION="us-west-2"
export AWS_ACCOUNT_ID="<12-digit>"
export TENANT_ID="tenant001"

export KINESIS_STREAM="anomaly-stream-${TENANT_ID}"
export DDB_TABLE="anomaly-results-${TENANT_ID}"
export S3_DATALAKE_BUCKET="iot-data-lake-${TENANT_ID}-${AWS_PRIMARY_REGION}"
```

> Store these in a `.env` and `source .env` before running.
