---
id: deploy-primary
title: "Deployment — Primary Region"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [ansible, primary, create, resources, playbooks]
---

# Deployment — Primary Region

Execute from the cloned **infra repo**: `iot-anomaly-deploy/infra/ansible`

```bash
git clone https://github.com/inaw-tihor/iot-anomaly-deploy
cd iot-anomaly-deploy/infra/ansible

ansible-playbook -i inventories/hosts.ini playbooks/01-create-prim-resources.yml
ansible-playbook -i inventories/hosts.ini playbooks/02-create-s3-replication-role.yml
ansible-playbook -i inventories/hosts.ini playbooks/03-enable-s3-replication.yml
ansible-playbook -i inventories/hosts.ini playbooks/04-create-dynamodb-global.yml
```
