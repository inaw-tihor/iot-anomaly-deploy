---
id: validation
title: "Validation — Smoke & Data Plane"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [validation, kinesis, dynamodb, s3, lambda, iot]
---

# Validation — Smoke & Data Plane

Run automated checks:
```bash
cd iot-anomaly-deploy/validate
./validate_replication.sh
```

Manual asserts:
```bash
# Kinesis
SHARD=$(aws kinesis list-shards --stream-name "$KINESIS_STREAM"   --query 'Shards[0].ShardId' --output text   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE")
ITER=$(aws kinesis get-shard-iterator --stream-name "$KINESIS_STREAM"   --shard-id "$SHARD" --shard-iterator-type LATEST   --query 'ShardIterator' --output text   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE")
aws kinesis get-records --shard-iterator "$ITER"   --region "$AWS_PRIMARY_REGION" --profile "$AWS_PROFILE"
```
