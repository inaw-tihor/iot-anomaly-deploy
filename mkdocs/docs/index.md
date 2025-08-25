---
id: index
title: "Industrial IoT Anomaly Detection — Operations Runbook"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [greengrass, iot core, kinesis, dynamodb, s3, crr, sagemaker, dr, tenancy, onboarding]
---

# Industrial IoT Anomaly Detection — Operations Runbook (RAG-Optimized v3)

This site documents how to **deploy, operate, validate, back up, restore, and fail over** an Industrial IoT Anomaly Detection platform across **Primary** and **DR** AWS regions, using **CLI-first** automation. Optional local control planes (Rancher, ArgoCD, Ansible Semaphore) can be used for manual approvals.

> **Reliability targets (example):** RTO ≤ 15m, RPO ≤ 5m — tune per tenant.

## Quick start
```bash
# Clone infra repo (ground truth playbooks)
git clone https://github.com/inaw-tihor/iot-anomaly-deploy
cd iot-anomaly-deploy/infra/ansible

# Primary
ansible-playbook -i inventories/hosts.ini playbooks/01-create-prim-resources.yml

# DR
ansible-playbook -i inventories/hosts.ini playbooks/05-sync-models-to-dr.yml
```

See **Reference Architecture** → for the hot/warm/cold flows. Then follow **Deployment** → **Tenant Onboarding** → **Validation**.
