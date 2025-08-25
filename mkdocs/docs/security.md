---
id: security
title: "Security, Patching & Upgrades"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [least privilege, iot policy, cert rotation, patching]
---

# Security, Patching & Upgrades

- **IoT policy**: limit to `tenants/{{TENANT_ID}}/**` and clientId = Thing name.  
- **Greengrass role alias**: IAM scoped to only required services.  
- **Cert rotation**: Use Ansible `cert_rollout.yml`.  
- **Patching cadence**: monthly OS/Java; refresh local control plane containers monthly.  
