---
id: dr
title: "DR / Replica & Failover"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [failover, failback, dual publish, bridge, s3 crr, ddb global]
---

# DR / Replica & Failover

## Failover playbook
1. **Quiesce Primary ingest** (disable IoT Rules).  
2. **Flip publishing to DR**:  
   - IoT Things: dual-publish or endpoint switch.  
   - Greengrass: run dual-publish config to DR endpoint.  
3. **Verify DR path**: Kinesis, DynamoDB, SageMaker logs.  
4. **Failback**: re-enable Primary, reverse endpoint switch, verify replication catch-up.
