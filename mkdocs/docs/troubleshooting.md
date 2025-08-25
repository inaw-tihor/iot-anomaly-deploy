---
id: troubleshooting
title: "Troubleshooting"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [connectivity, policy, lag, replication]
---

# Troubleshooting

| Symptom | Likely Cause | Action |
|---|---|---|
| Device cannot connect | Wrong ATS endpoint or cert/policy | Recheck endpoint, mTLS files, and policy scope. |
| Kinesis lag | Under-sharded | Increase shards, check consumer back-pressure. |
| No S3 replication | Bad CRR role or rule | Verify IAM role & replication config. |
| DR empty | Devices still on Primary | Confirm dual-publish or DR endpoint switch. |
