---
id: backup
title: "Backup & Restore (All Stores)"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [s3, pitr, dynamodb, models, replication]
---

# Backup & Restore (All Stores)

## S3 data & models
```bash
aws s3 sync s3://$S3_DATALAKE_BUCKET s3://$S3_BACKUP_BUCKET/data-lake
aws s3 sync s3://$S3_MODEL_BUCKET s3://$S3_BACKUP_BUCKET/models
```

## DynamoDB PITR
```bash
aws dynamodb export-table-to-point-in-time   --table-arn arn:aws:dynamodb:${AWS_PRIMARY_REGION}:${AWS_ACCOUNT_ID}:table/${DDB_TABLE}   --s3-bucket $S3_BACKUP_BUCKET
```

## Restore
```bash
aws s3 sync s3://$S3_BACKUP_BUCKET/data-lake s3://$S3_DATALAKE_BUCKET
# DDB restore from export -> cut over
```
