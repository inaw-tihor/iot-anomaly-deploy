---
id: tooling
title: "Local Tooling (Optional Controls)"
version: "3.0"
domain: "industrial-iot-anomaly"
keywords: [rancher, argocd, semaphore, docker]
---

# Local Tooling (Optional Controls)

These controls run locally in Docker Desktop and are **optional**—all deployment actions are available via CLI/Ansible.

```bash
# Rancher
docker run -d --name rancher --restart=unless-stopped   -p 80:80 -p 443:443 --privileged   -v /opt/rancher:/var/lib/rancher rancher/rancher:latest

# Semaphore (Ansible UI)
docker run -d --name semaphore -p 3000:3000   -e SEMAPHORE_DB_DIALECT=bolt   -e SEMAPHORE_ADMIN=admin   -e SEMAPHORE_ADMIN_PASSWORD=changeme   -e SEMAPHORE_ADMIN_NAME="Admin"   -e SEMAPHORE_ADMIN_EMAIL=admin@localhost   -v semaphore-data:/var/lib/semaphore   -v semaphore-config:/etc/semaphore   semaphoreui/semaphore:latest
```
