---
id: deploy-dr
title: "Deployment — DR / Replica Region"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [dr, replication, global tables, s3 crr, playbooks]
---

# Deployment — DR / Replica Region

```bash
cd iot-anomaly-deploy/infra/ansible
ansible-playbook -i inventories/hosts.ini playbooks/05-sync-models-to-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/06-create-sagemaker-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/07-create-kinesis-dr.yml
ansible-playbook -i inventories/hosts.ini playbooks/08-create-iot-dr.yml
```
