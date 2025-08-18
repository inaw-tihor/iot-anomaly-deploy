# IoT Industrial Anomaly Detection — Repo and Runbook

This repository contains Ansible playbooks, templates, and helper scripts to deploy, replicate, and validate
an Industrial IoT Anomaly Detection environment on AWS, with local CI/CD control panels (Rancher, ArgoCD, Semaphore)
running as Docker containers on Docker Desktop.

**IMPORTANT:** This repo is referenced by the operational runbook. Operators should `git clone` this repo
locally and run the CLI commands shown in the runbook.

## Repo layout

```
infra/
  ansible/
    inventories/
      hosts.ini
    group_vars/
      all.yml
    playbooks/
      01-create-prim-resources.yml
      02-create-s3-replication-role.yml
      03-enable-s3-replication.yml
      04-create-dynamodb-global.yml
      05-sync-models-to-dr.yml
      06-create-sagemaker-dr.yml
      07-create-kinesis-dr.yml
      08-create-iot-dr.yml
      greengrass_dual_publish.yml
      cert_rollout.yml
    templates/
      uploader-config.j2
    files/
      iot_policy.json
  semaphore/
    semaphore_pipeline.yaml
validate/
  validate_replication.sh
runbook/
  full_runbook.md
