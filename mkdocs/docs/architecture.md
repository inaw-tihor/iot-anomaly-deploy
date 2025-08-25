---
id: architecture
title: "Reference Architecture"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [architecture, hot path, warm path, cold path, iot rules, streams, sitewise, sns, quicksight]
---

# Reference Architecture

**Primary Region** hosts production. **DR Region** mirrors critical paths. Devices connect via **Greengrass** _or_ simple **IoT Things** to **IoT Core**. Routing uses **IoT Rules** to **Kinesis (warm)**, **S3 (cold)**, **DynamoDB** for anomaly state, **SageMaker** for inference, **SNS** for alerts, and **QuickSight** for visualization.

## Components
| Layer | Service | Purpose |
|---|---|---|
| Edge | Greengrass **or** IoT Thing | Local processing or direct telemetry. |
| Ingestion | IoT Core + Rules | MQTT ingress and routing. |
| Stream | Kinesis Data Streams/Analytics | Real-time/warm processing. |
| Storage | S3, DynamoDB | Data lake, anomaly state. |
| ML | SageMaker endpoint | Inference at scale. |
| Alerts | SNS | Operator notifications. |
| Viz | QuickSight | Dashboards. |
| DR | S3 CRR, DynamoDB Global Tables | Cross-region parity. |
| Control (local) | Rancher, ArgoCD, Semaphore | Manual gates; CLI-first remains authoritative. |

## Primary ↔ DR
- **S3**: Cross-Region Replication (versioning on).  
- **DynamoDB**: Global Table (regions: Primary, DR).  
- **Kinesis/IoT**: Mirror resources in DR.  
- **SageMaker**: Replicate model artifacts; create DR model/endpoint.
