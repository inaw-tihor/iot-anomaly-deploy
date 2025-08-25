---
id: cleanup
title: "Environment Cleanup"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [delete, teardown, stream, table, bucket]
---

# Environment Cleanup

```bash
aws s3 rb s3://$S3_DATALAKE_BUCKET --force
aws dynamodb delete-table --table-name $DDB_TABLE
aws kinesis delete-stream --stream-name $KINESIS_STREAM
```
